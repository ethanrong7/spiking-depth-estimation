"""Synthetic wiring/temporal tests only; no datasets, EMA, GPU or benchmark."""
import copy
import unittest

import torch

from core.ssdeformer_reconstruction import (CumulativeCode, SSDEFormerReconstruction,
    SpikformerBlock, build_from_config, lif_sequence, local_disparity)


class ReconstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def config(self, variant):
        return dict(initialization_seed=1234, model=dict(variant=variant, width=16,
            heads=2, layers=2, mlp_ratio=2, neural_steps=50,
            pyramid=[8, 8, 16, 16], row_chunk=4))

    def test_shared_components_identical_without_dormant_parameter_padding(self):
        ann = build_from_config(self.config('softmax'))
        snn = build_from_config(self.config('spikformer'))
        for name, value in ann.state_dict().items():
            if name.startswith(('features.', 'match_query.', 'match_key.')):
                self.assertTrue(torch.equal(value, snn.state_dict()[name]), name)

    def test_raw_prefix_counts_all50_bins_and_exact_shape_validation(self):
        x = torch.ones(1, 50, 2, 2, requires_grad=True)
        y = CumulativeCode()(x)
        self.assertEqual(y[0, -1, 0, 0], 50)
        y.sum().backward()
        self.assertTrue((x.grad != 0).all())
        with self.assertRaises(ValueError):
            CumulativeCode()(x[:, :49])

    def test_lif_oracle_reset_isolation_and_untruncated_early_gradient(self):
        x = torch.full((50, 2, 3), .1, requires_grad=True)
        y = lif_sequence(x)
        y[-1].sum().backward()
        self.assertTrue((x.grad[0].abs() > 0).all())
        self.assertTrue(torch.equal(y, lif_sequence(x)))
        actual = lif_sequence(torch.tensor([0., 3., 0., 3.])[:, None])
        self.assertEqual(actual[:, 0].tolist(), [0., 1., 0., 1.])

    def test_spiking_projection_outputs_binary_and_uses_all50_steps(self):
        layer = SpikformerBlock(16, 2, 2).eval()
        x = torch.randn(50, 2, 4, 16)
        q = layer.q(x)
        self.assertEqual(q.shape, x.shape)
        self.assertTrue(((q == 0) | (q == 1)).all())
        value = layer(x, x.flip(1), torch.ones(2, 4, dtype=torch.bool))
        self.assertTrue(torch.isfinite(value).all())

    def test_known_correspondence_readout_sign_border_and_three_pixel_support(self):
        p = torch.zeros(1, 8, 8)
        for x in range(8):
            p[0, x, max(0, x - 2)] = 1
        result = local_disparity(p, torch.ones(1, 8, dtype=torch.bool))
        torch.testing.assert_close(result[0], torch.tensor([0., 1., 2., 2., 2., 2., 2., 2.]))

    def test_both_complete_tiny_models_shape_backward_both_views_and_final_bin(self):
        for variant in ('softmax', 'spikformer'):
            with self.subTest(variant=variant):
                model = build_from_config(self.config(variant)).train()
                left = torch.rand(1, 50, 16, 16, requires_grad=True)
                right = torch.rand(1, 50, 16, 16, requires_grad=True)
                output = model(left, right)['disparity']
                self.assertEqual(output.shape, (1, 1, 16, 16))
                self.assertTrue(torch.isfinite(output).all())
                output.square().mean().backward()
                for value in (left.grad, right.grad):
                    self.assertTrue(torch.isfinite(value).all())
                    self.assertGreater(value[:, -1].abs().sum().item(), 0)
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_eval_state_isolation_and_row_chunk_output_gradient_parity(self):
        model = build_from_config(self.config('spikformer')).eval()
        other = copy.deepcopy(model)
        other.row_chunk = 16
        left = torch.rand(1, 50, 16, 16, requires_grad=True)
        right = torch.rand_like(left, requires_grad=True)
        first = model(left, right)['disparity']
        second = other(left, right)['disparity']
        torch.testing.assert_close(first, second, rtol=1e-5, atol=1e-5)
        a = torch.autograd.grad(first.sum(), (left, right), retain_graph=True)
        b = torch.autograd.grad(second.sum(), (left, right))
        for aa, bb in zip(a, b):
            torch.testing.assert_close(aa, bb, rtol=1e-4, atol=1e-4)
        model(torch.zeros_like(left), torch.zeros_like(right))
        torch.testing.assert_close(first, model(left, right)['disparity'], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
