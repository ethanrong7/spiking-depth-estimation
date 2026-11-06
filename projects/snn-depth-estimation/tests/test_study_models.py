"""Tiny CPU correctness checks, never training/benchmark evidence."""
import copy
import unittest

import torch
from core.study_losses import disparity_loss

from core.study_models import (TemporalNeuron, build_model, epipolar_mask,
                               row_attention, local_attention, StereoMatcher)


class StudyModelsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def config(self, architecture='A', steps=3, **extra):
        return dict(architecture=architecture, spiking=True, widths=[8, 16, 24, 32],
                    time_steps=steps, max_disparity=12, input_channels=1,
                    checkpoint=False, blocks_per_scale=1, **extra)

    def test_disparity_geometry_and_row_direction(self):
        mask = epipolar_mask(4, 2, torch.device('cpu'))
        self.assertTrue(mask[3, 1])
        self.assertFalse(mask[1, 3])
        self.assertFalse(mask[3, 0])
        self.assertTrue(torch.equal(mask.T, epipolar_mask(4, 2, 'cpu', True)))
        q = torch.ones(1, 1, 1, 1, 4)
        v = torch.arange(4.).reshape_as(q)
        result = row_attention(q, q, v, torch.ones(1, 1, 1, 4, dtype=torch.bool), 2)
        self.assertTrue(torch.allclose(result[0, 0, 0, 0], torch.tensor([0., .5, 1., 2.]), atol=2e-6))

    def test_matcher_disparity_units_and_right_sign(self):
        matcher = StereoMatcher(4, descriptor_channels=4, temperature=.001)
        matcher.descriptor = torch.nn.Identity()
        # Perfect one-hot matches: left x=2 observes right x=1 => d=+stride.
        right = torch.eye(4).T.reshape(1, 4, 1, 4)
        left = torch.roll(right, 1, -1)
        mask = torch.ones(1, 1, 1, 4, dtype=torch.bool)
        dl, _, _ = matcher(left, right, mask, 4, 12)
        dr, _, _ = matcher(right, left, mask, 4, 12, reverse=True)
        self.assertAlmostEqual(dl[0, 0, 0, 2].item(), 4.)
        self.assertAlmostEqual(dr[0, 0, 0, 1].item(), 4.)

    def test_empty_coincidence_recovery_gradient_is_bounded(self):
        for operator in ('row', 'local'):
            for version, expected in [('epsilon_v1', 1e6), ('count_floor_v2', 1.)]:
                q = torch.zeros(1, 1, 1, 1, 1, requires_grad=True)
                k = torch.ones_like(q)
                valid = torch.ones(1, 1, 1, 1, dtype=torch.bool)
                if operator == 'row':
                    result = row_attention(q, k, k, valid, 1, normalization=version)
                else:
                    result = local_attention(q, k, k, valid, normalization=version)
                self.assertEqual(result.item(), 0.)
                result.sum().backward()
                self.assertAlmostEqual(q.grad.item(), expected, delta=1e-5)
        # Nonempty binary coincidence rows preserve normalized count outputs.
        torch.manual_seed(8)
        q, k, v = [torch.randint(0, 2, (1, 2, 4, 2, 5)).float() for _ in range(3)]
        valid = torch.ones(1, 1, 2, 5, dtype=torch.bool)
        for operator in ('row', 'local'):
            run = lambda version: row_attention(q, k, v, valid, 4, normalization=version) if operator == 'row' else local_attention(q, k, v, valid, normalization=version)
            self.assertTrue(torch.allclose(run('epsilon_v1'), run('count_floor_v2'), atol=3e-6, rtol=3e-6))

    def test_neuron_full_temporal_gradient_and_reset(self):
        neuron = TemporalNeuron(True)
        neuron.state_id = 0
        first = torch.tensor([.01], requires_grad=True)
        states = {}
        for step in range(50):
            updated = {}
            z = neuron(first if step == 0 else torch.zeros_like(first), states, updated, 0)
            states = updated
        z.sum().backward()
        self.assertGreater(first.grad.abs().item(), 0)
        # No detached reset: exact derivative includes the surrogate reset term.
        x = torch.tensor([.3], requires_grad=True)
        updated = {}
        neuron(x, {}, updated, 0)
        updated[(0, 0)].sum().backward()
        self.assertAlmostEqual(x.grad.item(), 1 - 1 / 1.7**2, places=6)

    def test_all_families_polarities_shapes_finite_and_fresh_reset(self):
        for architecture, spiking, channels in [('A', True, 1), ('B', True, 2), ('conv', True, 1), ('A', False, 1), ('B', False, 2)]:
            with self.subTest(architecture=architecture, spiking=spiking):
                config = self.config(architecture)
                config.update(spiking=spiking, input_channels=channels)
                model = build_model(config).eval()
                x = torch.rand(1, 3, channels, 17, 19)
                with torch.no_grad():
                    output = model(x, x.flip(-1))
                    repeated = model(x, x.flip(-1))
                self.assertEqual(output['disparity'].shape, (1, 1, 17, 19))
                self.assertTrue(torch.isfinite(output['disparity']).all())
                self.assertTrue(torch.equal(output['disparity'], repeated['disparity']))
                self.assertEqual(set(output['aux_disparities']), {4, 8, 16})
                self.assertTrue(output['diagnostics']['hybrid'])

    def test_checkpoint_same_outputs_parameter_and_early_input_gradients(self):
        torch.manual_seed(42)
        plain_config = self.config(steps=6)
        plain_config['temporal_chunk_size'] = 1
        plain = build_model(plain_config).train()
        checked = copy.deepcopy(plain)
        object.__setattr__(checked.config, 'checkpoint', True)
        object.__setattr__(checked.config, 'temporal_chunk_size', 3)
        x = torch.rand(1, 6, 1, 17, 19, requires_grad=True)
        y = torch.rand_like(x, requires_grad=True)
        xx, yy = x.detach().clone().requires_grad_(), y.detach().clone().requires_grad_()
        a = plain(x, y)['disparity']
        b = checked(xx, yy)['disparity']
        self.assertTrue(torch.allclose(a, b, atol=1e-6, rtol=1e-5))
        a.mean().backward()
        b.mean().backward()
        self.assertTrue(torch.allclose(x.grad, xx.grad, atol=1e-6, rtol=1e-5))
        self.assertGreater(x.grad[:, 0].abs().sum().item(), 0)
        self.assertGreater(y.grad[:, 0].abs().sum().item(), 0)
        for (name, p), (_, q) in zip(plain.named_parameters(), checked.named_parameters()):
            if p.grad is not None:
                self.assertTrue(torch.allclose(p.grad, q.grad, atol=1e-6, rtol=1e-5), name)

    def test_t50_complete_conv_early_gradient(self):
        config = self.config('conv', 50)
        config['checkpoint'] = True
        model = build_model(config)
        x = torch.rand(1, 50, 1, 16, 16, requires_grad=True)
        model(x, x.flip(-1))['disparity'].mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(x.grad[:, 0].abs().sum().item(), 0)

    def test_static_diagnostics_flag_preserves_output_and_full_gradients(self):
        torch.manual_seed(17)
        config = self.config(steps=3)
        config['checkpoint'] = True
        collected = build_model(config)
        config['collect_diagnostics'] = False
        disabled = build_model(config)
        disabled.load_state_dict(collected.state_dict(), strict=True)
        x = torch.rand(1, 3, 1, 17, 19, requires_grad=True)
        y = torch.rand_like(x, requires_grad=True)
        xx, yy = x.detach().clone().requires_grad_(), y.detach().clone().requires_grad_()
        a, b = collected(x, y), disabled(xx, yy)
        self.assertTrue(torch.equal(a['disparity'], b['disparity']))
        self.assertTrue(a['diagnostics']['neuron_activity'])
        self.assertEqual(b['diagnostics']['neuron_activity'], {})
        self.assertFalse(b['diagnostics']['collect_diagnostics'])
        a['disparity'].mean().backward()
        b['disparity'].mean().backward()
        self.assertTrue(torch.equal(x.grad, xx.grad))
        self.assertTrue(torch.equal(y.grad, yy.grad))
        for (name, p), (_, q) in zip(collected.named_parameters(), disabled.named_parameters()):
            if p.grad is not None:
                self.assertTrue(torch.equal(p.grad, q.grad), name)

    def test_actual_A_physical2_vs_accumulated_physical1_gradients(self):
        torch.manual_seed(29)
        cfg = self.config(steps=6)
        cfg['checkpoint'] = True
        cfg['temporal_chunk_size'] = 3
        cfg['feature_norm_eps'] = .01
        batched = build_model(cfg)
        accumulated = copy.deepcopy(batched)
        left = torch.rand(2, 6, 1, 17, 19, requires_grad=True)
        right = torch.rand_like(left, requires_grad=True)
        split_left, split_right = left.detach().clone().requires_grad_(), right.detach().clone().requires_grad_()
        target = torch.ones(2, 1, 17, 19) * 2
        valid = torch.ones_like(target, dtype=torch.bool)
        out = batched(left, right)
        loss, _ = disparity_loss(out, target, valid, 12)
        loss.backward()
        splits, split_losses = [], []
        for index in range(2):
            item = accumulated(split_left[index:index+1], split_right[index:index+1])
            value, _ = disparity_loss(item, target[index:index+1], valid[index:index+1], 12)
            (value / 2).backward()
            splits.append(item['disparity'].detach())
            split_losses.append(value.detach())
        self.assertTrue(torch.allclose(out['disparity'].detach(), torch.cat(splits), atol=2e-5, rtol=1e-4))
        self.assertTrue(torch.allclose(loss.detach(), torch.stack(split_losses).mean(), atol=2e-5, rtol=1e-4))
        gradients, differences = [], []
        for (name, p), (_, q) in zip(batched.named_parameters(), accumulated.named_parameters()):
            if p.grad is not None:
                gradients.append(p.grad.double().flatten())
                differences.append((p.grad-q.grad).double().flatten())
        relative_l2 = torch.cat(differences).norm() / torch.cat(gradients).norm().clamp_min(1e-12)
        self.assertLess(relative_l2.item(), 1e-4)
        for full, split in ((left.grad, split_left.grad), (right.grad, split_right.grad)):
            self.assertLess(((full-split).double().norm()/full.double().norm().clamp_min(1e-12)).item(), 1e-4)

    def test_feature_epsilon_scope_and_quiet_fullT50_backward(self):
        for architecture in ('A', 'B', 'conv'):
            cfg = self.config(architecture)
            cfg['feature_norm_eps'] = .01
            model = build_model(cfg)
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.GroupNorm):
                    self.assertEqual(module.eps, .01 if name.startswith(('stages.', 'blocks.')) else 1e-5)
        torch.manual_seed(9471)
        cfg = dict(architecture='A', spiking=True, widths=[32, 64, 96, 128], time_steps=50,
                   max_disparity=16, checkpoint=True, temporal_chunk_size=5, collect_diagnostics=False,
                   feature_norm_eps=1e-5)
        original = build_model(cfg)
        cfg['feature_norm_eps'] = .01
        repaired = build_model(cfg)
        repaired.load_state_dict(original.state_dict(), strict=True)
        generator = torch.Generator().manual_seed(9471)
        left = (torch.rand(1, 50, 1, 16, 16, generator=generator) < .05).float()
        right = left.roll(-2, -1).clone()
        left[:, :4] = 0
        right[:, :4] = 0
        target = torch.ones(1, 1, 16, 16) * 2
        valid = torch.ones_like(target, dtype=torch.bool)
        norms = []
        for model in (original, repaired):
            output = model(left, right)
            loss, _ = disparity_loss(output, target, valid, 16)
            loss.backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
            norms.append(torch.stack([g.double().square().sum() for g in gradients]).sum().sqrt().item())
        self.assertGreater(norms[0] / max(norms[1], 1e-12), 1e6)
        self.assertLess(norms[1], 1e10)
        self.assertGreater(repaired.decoder.residual[-1].weight.grad.abs().sum().item(), 0)
        repaired.zero_grad(set_to_none=True)
        quiet = torch.zeros_like(left)
        output = repaired(quiet, quiet)
        loss, _ = disparity_loss(output, target, valid, 16)
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in repaired.parameters() if p.grad is not None))


if __name__ == '__main__':
    unittest.main()
