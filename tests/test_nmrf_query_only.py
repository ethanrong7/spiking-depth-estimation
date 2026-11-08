import copy
import math
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F
from torch.utils.flop_counter import FlopCounterMode

from benchmark_nmrf_selective_updates import load_benchmark_weights
from nmrf_runtime import build_native_nmrf
from nmrf.models.NMP import SwinNMP, WindowAttention
from nmrf_selective_updates import (ALTERNATING_SELECTOR, COMPACT_SUPPORT, FULL_KV_SUPPORT,
                                    LEGACY_SELECTOR, SelectiveSwinNMP)
from run_nmrf_finetune import FORMAT, install_architecture, restore_architecture


def parity_case(shift, coding, keep=.5):
    torch.manual_seed(81)
    candidates, channels, geometry = 4, 16, 7
    base = SwinNMP(channels, channels+geometry, 4, window_size=6, shift_size=shift,
                   normalize_before=True).eval()
    base.H = base.W = 12
    with torch.no_grad():
        base.attn.relative_position_enc_table.normal_(0, .1)
    selected = SelectiveSwinNMP(copy.deepcopy(base), keep, coding, phase=1,
                                attention_support=FULL_KV_SUPPORT)
    reference = base if coding == 'continuous' else SelectiveSwinNMP(
        copy.deepcopy(base), 1., coding, phase=1)
    batch = 2 if shift else 1

    def padded_fixture(ch):
        real = torch.randn(batch, 10, 10, candidates, ch)
        return F.pad(real, (0, 0, 0, 0, 1, 1, 1, 1)).reshape(-1, candidates, ch).requires_grad_()

    x, g = padded_fixture(channels), padded_fixture(geometry)
    window = (6, 6, candidates)
    mask = (WindowAttention.gen_shift_window_attn_mask((12, 12), window, shift, 'cpu') if shift
            else WindowAttention.gen_window_attn_mask(window, 'cpu')[None])
    actual = selected(x, g, mask)
    dense = reference(x, g, mask)
    pixels = selected.selected_pixels('cpu')
    active_dense = selected.partition(dense).index_select(1, pixels)
    expected = selected.unpartition(selected.partition(x).index_copy(1, pixels, active_dense))
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=3e-6)

    inactive = torch.ones(36, dtype=torch.bool)
    inactive[pixels] = False
    inactive_x = selected.partition(x)[:, inactive]
    assert torch.equal(selected.partition(actual)[:, inactive], inactive_x)

    probe = torch.randn_like(active_dense) / math.sqrt(active_dense.numel())
    losses = ((selected.partition(actual).index_select(1, pixels) * probe).sum(),
              (active_dense * probe).sum())
    grads = [torch.autograd.grad(loss, (x, g, *block.parameters()))
             for block, loss in zip((selected, reference), losses)]
    for value, ref in zip(*grads):
        torch.testing.assert_close(value, ref, rtol=2e-4, atol=3e-6)
    inactive_label_grad = selected.partition(grads[0][0])[:, inactive].abs()
    inactive_geometry_grad = selected.partition(grads[0][1])[:, inactive].abs()
    if keep < 1:
        assert inactive_label_grad.max() > 0 and inactive_geometry_grad.max() > 0
    return dict(max_output_error=float((actual-expected).detach().abs().max()),
                max_gradient_error=max(float((a-b).abs().max()) for a, b in zip(*grads)),
                max_inactive_label_gradient=float(inactive_label_grad.max()) if keep < 1 else None,
                max_inactive_geometry_gradient=float(inactive_geometry_grad.max()) if keep < 1 else None)


class FullKeyValueSupportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_selected_forward_backward_matches_dense_reference(self):
        for shift in (0, 3):
            for coding in ('continuous', 'ternary'):
                with self.subTest(shift=shift, coding=coding):
                    parity_case(shift, coding)
        parity_case(3, 'continuous', keep=1.)

    def test_rectangular_mask_and_contractions(self):
        torch.manual_seed(81)
        base = SwinNMP(128, 159, 4, window_size=6, shift_size=3, normalize_before=True).eval()
        base.H = base.W = 6
        x, geometry = torch.randn(36, 4, 128), torch.randn(36, 4, 31)
        mask = WindowAttention.gen_shift_window_attn_mask((6, 6), (6, 6, 4), 3, 'cpu')
        query = SelectiveSwinNMP(copy.deepcopy(base), .5, phase=1, attention_support=FULL_KV_SUPPORT)
        compact = SelectiveSwinNMP(copy.deepcopy(base), .5, phase=1)
        token_ids = (query.selected_pixels('cpu')[:, None]*4+torch.arange(4)).flatten()
        scores, ffn_inputs = [], []
        hooks = [query.attn.attn_drop.register_forward_hook(lambda _, args, __: scores.append(args[0])),
                 query.mlp.register_forward_pre_hook(lambda _, args: ffn_inputs.append(args[0].shape))]
        contractions = []
        try:
            for model in (base, compact, query):
                with torch.inference_mode(), FlopCounterMode(display=False) as counter:
                    model(x, geometry, mask)
                contractions.append(counter.get_total_flops())
            with torch.inference_mode(), patch('nmrf_selective_updates.F.linear', wraps=F.linear) as linear:
                query(x, geometry, mask)
            shapes = [(call.args[0].shape[1], call.args[1].shape[0]) for call in linear.call_args_list[:4]]
            self.assertEqual(shapes, [(18, 128), (18, 128), (36, 256), (36, 256)])
        finally:
            for hook in hooks:
                hook.remove()
        self.assertEqual(scores[0].shape, (1, 4, 72, 144))
        self.assertEqual(ffn_inputs[0], torch.Size([18, 4, 128]))
        supported = mask.index_select(-2, token_ids).isfinite()[:, None].expand_as(scores[0])
        self.assertTrue(torch.equal(scores[0] > 0, supported))
        native_block_counts = [4_676_050_944, 1_979_707_392, 2_654_539_776]
        self.assertEqual(contractions, [value // 54 for value in native_block_counts])

    def test_checkpoint_and_benchmark_preserve_support_and_selector(self):
        original, _, _ = build_native_nmrf(81)
        transplant = {'model': original.state_dict()}
        for support in (COMPACT_SUPPORT, FULL_KV_SUPPORT):
            model, _, _ = build_native_nmrf(81)
            metadata = install_architecture(model, 'ternary', selector=ALTERNATING_SELECTOR,
                                            attention_support=support)
            self.assertEqual(metadata['selective'].get('attention_support', COMPACT_SUPPORT), support)
            self.assertEqual(metadata['selective']['phases'], [1, 1, 1])
            state = dict(format=FORMAT, architecture=metadata, step=500,
                         model=model.state_dict(), run_config={'recipe': 'test'})
            receipt = dict(architecture=metadata, steps=500, identity=state['run_config'])
            fresh, _, _ = build_native_nmrf(9)
            with self.assertRaisesRegex(ValueError, 'selector'):
                load_benchmark_weights(fresh, state, receipt, 'ternary', selector=LEGACY_SELECTOR)
            wrong = COMPACT_SUPPORT if support == FULL_KV_SUPPORT else FULL_KV_SUPPORT
            with self.assertRaisesRegex(ValueError, 'attention support'):
                load_benchmark_weights(fresh, state, receipt, 'ternary', attention_support=wrong)
            self.assertEqual(load_benchmark_weights(fresh, state, receipt, 'ternary'), metadata)
            self.assertEqual(fresh.parent.inference.layers[2].nmp.attention_support, support)
            self.assertEqual(fresh.parent.inference.layers[2].nmp.phase, 1)
        fresh, _, _ = build_native_nmrf(9)
        transplanted = load_benchmark_weights(fresh, transplant, {}, 'ternary',
                                             selector=ALTERNATING_SELECTOR, attention_support=FULL_KV_SUPPORT)
        self.assertEqual(transplanted, metadata['selective'])
        corrupted = copy.deepcopy(state)
        del corrupted['architecture']['selective']['attention_support']
        fresh, _, _ = build_native_nmrf(9)
        with self.assertRaisesRegex(ValueError, 'architecture'):
            restore_architecture(fresh, corrupted)


if __name__ == '__main__':
    unittest.main()
