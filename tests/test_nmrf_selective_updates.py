import copy
import unittest

import torch

import nmrf_runtime  # Registers the pinned vendor import path.
from nmrf.models.NMP import SwinNMP, WindowAttention
from nmrf_selective_updates import ALTERNATING_SELECTOR, SelectiveSwinNMP, install_selective_updates


class SelectiveUpdateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def fixture(self, shift=0, pre=False, keep=0.5, coding="continuous"):
        torch.manual_seed(81)
        base = SwinNMP(16, 23, 4, window_size=2, shift_size=shift,
                       normalize_before=pre).double().eval()
        base.H = base.W = 4
        with torch.no_grad():
            base.attn.relative_position_enc_table.normal_(0, .1)
        compact = SelectiveSwinNMP(copy.deepcopy(base), keep, coding, phase=1)
        x = torch.randn(32, 4, 16, dtype=torch.double, requires_grad=True)
        g = torch.randn(32, 4, 7, dtype=torch.double, requires_grad=True)
        if shift:
            mask = WindowAttention.gen_shift_window_attn_mask((4, 4), (2, 2, 4), shift, 'cpu').double()
        else:
            mask = WindowAttention.gen_window_attn_mask((2, 2, 4), 'cpu').double()[None].repeat(4, 1, 1)
        return base, compact, x, g, mask

    def test_full_budget_matches_upstream_outputs_and_gradients(self):
        for shift in (0, 1):
            for pre in (False, True):
                base, compact, x, g, mask = self.fixture(shift, pre, keep=1)
                y0, y1 = base(x, g, mask), compact(x, g, mask)
                torch.testing.assert_close(y0, y1, atol=1e-12, rtol=1e-11)
                probe = torch.randn_like(y0)
                inputs0 = (x, g, *base.parameters())
                inputs1 = (x, g, *compact.parameters())
                grads0 = torch.autograd.grad((y0 * probe).sum(), inputs0)
                grads1 = torch.autograd.grad((y1 * probe).sum(), inputs1)
                for a, b in zip(grads0, grads1):
                    torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-9)

    def test_compact_matches_independent_dense_masked_reference(self):
        for shift in (0, 1):
            for pre in (False, True):
                base, compact, x, g, mask = self.fixture(shift, pre)
                pixels = compact.selected_pixels(x.device)
                selected = torch.zeros(4, dtype=torch.bool)
                selected[pixels] = True
                token_keep = selected.repeat_interleave(4)
                dense_mask = mask.clone().masked_fill(~token_keep[None, None, :], -torch.inf)
                # Inactive queries are discarded, but keep their softmax defined.
                dense_mask.diagonal(dim1=-2, dim2=-1).fill_(0)
                dense = base(x, g, dense_mask)
                expected = compact.unpartition(torch.where(
                    selected[None, :, None, None], compact.partition(dense), compact.partition(x)))
                actual = compact(x, g, mask)
                torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-11)
                self.assertTrue(torch.equal(compact.partition(actual)[:, ~selected],
                                            compact.partition(x)[:, ~selected]))
                probe = torch.randn_like(actual)
                gr0 = torch.autograd.grad((expected * probe).sum(), (x, g, *base.parameters()))
                gr1 = torch.autograd.grad((actual * probe).sum(), (x, g, *compact.parameters()))
                for a, b in zip(gr0, gr1):
                    torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-9)

    def test_real_contraction_work_reduced_and_signed_training_finite(self):
        from torch.utils.flop_counter import FlopCounterMode
        base, compact, x, g, mask = self.fixture(shift=1)
        counts = []
        for model in (base, compact):
            with torch.no_grad(), FlopCounterMode(display=False) as counter:
                model(x, g, mask)
            counts.append(counter.get_total_flops())
        self.assertLess(counts[1], counts[0] * .6)
        for coding in ('ternary', 'tanh'):
            _, signed, x, g, mask = self.fixture(1, keep=.25, coding=coding)
            signed.train()
            seen = []
            hook = signed.mlp.register_forward_pre_hook(lambda m, args: seen.append(args[0].shape))
            result = signed(x, g, mask)
            hook.remove()
            self.assertEqual(seen, [torch.Size([8, 4, 16])])
            self.assertTrue(torch.isfinite(result).all())
            result.square().mean().backward()
            for p in signed.parameters():
                self.assertIsNotNone(p.grad)
                self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(float(signed.qkv.weight.grad.abs().sum()), 0)

    def test_installation_checkpoint_and_native_pipeline(self):
        model, _, _ = nmrf_runtime.build_native_nmrf(seed=81)
        model.eval()
        original = model.parent.inference.layers[0].nmp
        keys = set(model.state_dict())
        left = torch.randint(0, 2, (1, 50, 32, 64)).float()
        right = torch.randint(0, 2, left.shape).float()
        with torch.no_grad():
            expected = model(left, right)['disp']
        config = install_selective_updates(model, keep_fraction=1)
        self.assertEqual(config['layers'], [1, 2, 3])
        self.assertIs(model.parent.inference.layers[0].nmp, original)
        self.assertEqual(keys, set(model.state_dict()))
        with torch.no_grad():
            actual = model(left, right)['disp']
        torch.testing.assert_close(actual, expected, atol=5e-4, rtol=1e-4)
        for layer in model.parent.inference.layers[1:4]:
            layer.nmp.keep_fraction = .5
            layer.nmp.coding = 'ternary'
        with torch.no_grad():
            output = model(left, right)['disp']
        self.assertEqual(output.shape, expected.shape)
        self.assertTrue(torch.isfinite(output).all())
        with self.assertRaisesRegex(ValueError, 'already'):
            install_selective_updates(model)

    def test_alternating_global_column_coverage_preserves_compute_shapes(self):
        from torch.utils.flop_counter import FlopCounterMode
        base, _, _ = nmrf_runtime.build_native_nmrf(seed=81)
        legacy, alternate = copy.deepcopy(base).eval(), base.eval()
        legacy_config = install_selective_updates(legacy, coding='ternary')
        alt_config = install_selective_updates(alternate, coding='ternary', selector=ALTERNATING_SELECTOR)
        self.assertEqual(legacy_config['architecture'], 'selective_spatial_nmrf_v1')
        self.assertEqual(alt_config['architecture'], 'selective_spatial_nmrf_v2')
        self.assertEqual(alt_config['phases'], [1, 1, 1])
        self.assertEqual(alt_config['global_column_parities'], [0, 1, 0])
        blocks = [[model.parent.inference.layers[i].nmp for i in (1, 2, 3)]
                  for model in (legacy, alternate)]
        for (h, w), (padded_h, padded_w) in [((16, 32), (18, 36)), ((32, 50), (36, 54))]:
            masks = []
            for arm in blocks:
                masks.append([])
                for block in arm:
                    block.H, block.W = padded_h, padded_w
                    tiles = block.partition(torch.zeros(padded_h*padded_w, 4, 1, dtype=torch.bool))
                    selected = block.selected_pixels('cpu')
                    self.assertEqual(selected.numel(), 18)
                    tiles[:, selected] = True
                    chosen = block.unpartition(tiles).reshape(padded_h, padded_w, 4)
                    self.assertTrue(torch.equal(chosen, chosen[:, :, :1].expand_as(chosen)))
                    masks[-1].append(chosen[:, :, 0])
            self.assertTrue(torch.equal(masks[0][0], masks[1][0]))
            self.assertTrue(torch.equal(masks[0][2], masks[1][2]))
            self.assertTrue(torch.equal(masks[0][1], ~masks[1][1]))
            y, x = (padded_h-h)//2, (padded_w-w)//2
            histograms = [torch.bincount(torch.stack(arm).sum(0)[y:y+h, x:x+w].flatten(), minlength=4)
                          for arm in masks]
            self.assertEqual(histograms[0].tolist(), [h*w//2, 0, 0, h*w//2])
            self.assertEqual(histograms[1].tolist(), [0, h*w//2, h*w//2, 0])
        left = torch.randint(0, 2, (1, 50, 32, 64)).float()
        right = torch.randint(0, 2, left.shape).float()
        counts = []
        for model in (legacy, alternate):
            with torch.no_grad(), FlopCounterMode(display=False) as counter:
                disp = model(left, right)['disp']
            self.assertEqual(disp.shape, (1, 32, 64))
            counts.append(counter.get_total_flops())
        self.assertEqual(counts[0], counts[1])


if __name__ == '__main__':
    unittest.main()
