"""Tiny CPU mechanism checks; these are not accuracy or GPU experiments."""
import sys
import unittest
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.spiking_attention_update import LocalAttentionRefinement, RefinementLIF


class AttentionTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)

    def make_inputs(self):
        return (torch.randn(1, 16, 4, 5),
                [torch.randn(1, 16, 4, 5) for _ in range(3)],
                torch.randn(1, 8, 4, 5))

    def test_spiking_shape_gradient_and_reset(self):
        cell = LocalAttentionRefinement(16, 8)
        inputs = self.make_inputs()
        outputs = [cell(*inputs) for _ in range(3)]
        self.assertEqual(outputs[-1][0].shape, (1, 16, 4, 5))
        self.assertTrue(set(outputs[-1][0].detach().unique().tolist()) <= {0., 1.})
        self.assertFalse(torch.equal(outputs[0][1], outputs[1][1]))
        sum(x[1].square().mean() for x in outputs).backward()
        diagnostics = cell.diagnostics()
        for name in ("q", "k", "v"):
            self.assertGreater(diagnostics[name + "_grad_norm"], 0)
            self.assertGreater(diagnostics[name + "_rate"], 0)
            self.assertLess(diagnostics[name + "_rate"], 1)
        cell.reset_state()
        repeated = cell(*inputs)
        torch.testing.assert_close(repeated[1], outputs[0][1])
        self.assertEqual(cell.diagnostics()["steps"], 1)

    def test_no_spatial_mixing_and_padding(self):
        spatial = LocalAttentionRefinement(16, 8)
        center = LocalAttentionRefinement(16, 8, spatial_mixing=False)
        q = torch.ones(1, 32, 3, 3)
        k = q.clone()
        v = torch.zeros_like(q)
        v[:, :, 0, 1] = 1
        mixed, weights = spatial._attention(q, k, v)
        unmixed, _ = center._attention(q, k, v)
        self.assertGreater(mixed[:, :, 1, 1].sum().item(), 0)
        self.assertEqual(unmixed[:, :, 1, 1].sum().item(), 0)
        self.assertTrue(torch.all(weights.sum(2) < 1))
        self.assertEqual(weights[0, 0, 0, 0].item(), 0)  # padded top-left
        silent, _ = spatial._attention(torch.zeros_like(q), k, v)
        self.assertEqual(silent.count_nonzero().item(), 0)

    def test_analog_matches_parameterization(self):
        spiking = LocalAttentionRefinement(16, 8)
        analog = LocalAttentionRefinement(16, 8, mode="analog")
        analog.load_state_dict(spiking.state_dict())
        inputs = self.make_inputs()
        _, readout = analog(*inputs)
        readout.square().mean().backward()
        self.assertGreater(analog.qkv.weight.grad.norm().item(), 0)
        q = torch.randn(1, 32, 4, 5)
        _, weights = analog._attention(q, q, q)
        torch.testing.assert_close(weights.sum(2), torch.ones_like(weights.sum(2)))

    def test_gradient_crosses_refinement_steps(self):
        neuron = RefinementLIF()
        first = torch.tensor([0.7], requires_grad=True)
        neuron(first)
        _, last_membrane = neuron(torch.tensor([0.4]))
        last_membrane.sum().backward()
        self.assertTrue(torch.isfinite(first.grad).all())
        self.assertGreater(first.grad.abs().item(), 0)

    def test_opt_in_residual_normalization_preserves_qkv_and_controls_current(self):
        raw = LocalAttentionRefinement(16, 8)
        normalized = LocalAttentionRefinement(16, 8, normalize_residual_current=True)
        with torch.no_grad():
            raw.input_proj.weight.mul_(50)
            raw.input_proj.bias.fill_(20)
        normalized.load_state_dict(raw.state_dict())
        hidden, context, motion = self.make_inputs()
        raw_hidden = hidden.clone().requires_grad_()
        normalized_hidden = hidden.clone().requires_grad_()
        seen = []
        handles = [cell.qkv.register_forward_pre_hook(lambda module, inputs: seen.append(inputs[0].detach().clone()))
                   for cell in (raw, normalized)]
        _, raw_voltage = raw(raw_hidden, context, motion)
        _, normalized_voltage = normalized(normalized_hidden, context, motion)
        for handle in handles:
            handle.remove()
        torch.testing.assert_close(seen[0], seen[1], rtol=0, atol=0)
        raw_voltage.square().mean().backward()
        normalized_voltage.square().mean().backward()
        self.assertGreater(raw_voltage.detach().square().mean().sqrt().item(), 10)
        self.assertLess(normalized_voltage.detach().square().mean().sqrt().item(), 2)
        self.assertTrue(torch.isfinite(normalized_hidden.grad).all())
        self.assertGreater(normalized_hidden.grad.norm().item(), 0)
        self.assertLess(normalized_hidden.grad.norm().item(), raw_hidden.grad.norm().item())
        self.assertLess(normalized.diagnostics()["output_post_reset_above_threshold_fraction"],
                        raw.diagnostics()["output_post_reset_above_threshold_fraction"])
        self.assertLess(normalized.diagnostics()["output_voltage_excursion_proxy_fraction"],
                        raw.diagnostics()["output_voltage_excursion_proxy_fraction"])
        self.assertGreater(normalized.diagnostics()["output_surrogate_derivative_mean"],
                           raw.diagnostics()["output_surrogate_derivative_mean"])


@unittest.skipUnless(importlib.util.find_spec("spikingjelly"), "full wrapper requires spikingjelly")
class WrapperTest(unittest.TestCase):
    def test_common_initialization_full_shape_backward_and_reset(self):
        from core.spikestereonet import SpikeStereoNet
        from core.spiking_transformer_stereo import SpikingTransformerStereo
        torch.set_num_threads(1)
        torch.manual_seed(13)
        args = SimpleNamespace(hidden_dims=[128] * 3, n_snn_layers=3,
            n_downsample=2, corr_levels=4, corr_radius=4, context_norm="batch",
            shared_backbone=False, mixed_precision=False, corr_implementation="reg")
        reference = SpikeStereoNet(args)
        model = SpikingTransformerStereo(args).eval()
        model.load_common_state(reference)
        for name in ("cnet", "fnet", "context_zqr_convs", "update_block.encoder",
                     "update_block.disp_head", "update_block.mask"):
            for key, value in model.get_submodule(name).state_dict().items():
                torch.testing.assert_close(value, reference.get_submodule(name).state_dict()[key])
        left, right = (torch.rand(1, 50, 32, 64) for _ in range(2))
        predictions, spikes, voltages = model(left, right, iters=2)
        self.assertEqual(predictions[-1].shape, (1, 1, 32, 64))
        self.assertEqual(len(spikes), 2)
        self.assertEqual(len(voltages), 2)
        predictions[-1].square().mean().backward()
        diagnostics = model.update_block.diagnostics()
        self.assertGreater(diagnostics["scale_0"]["q_grad_norm"], 0)
        self.assertGreater(diagnostics["scale_0"]["k_grad_norm"], 0)
        for cell in model.update_block.cells:
            self.assertIsNone(cell.q_lif.membrane)
            self.assertIsNone(cell.output_lif.membrane)
        with torch.no_grad():
            repeated = model(left, right, iters=2)[0][-1]
        torch.testing.assert_close(predictions[-1], repeated)
        self.assertEqual(model.update_block.diagnostics()["scale_0"]["steps"], 2)
        for kwargs in ({"center_only": True}, {"disable_attention": True}):
            with model.update_block.attention_intervention(**kwargs), torch.no_grad():
                intervention = model(left, right, iters=2)[0][-1]
            self.assertFalse(torch.equal(intervention, repeated))
            self.assertTrue(all(c.spatial_mixing and c.attention_enabled
                                for c in model.update_block.cells))
        # A failed outer forward also clears sample-batch state.
        with self.assertRaises(RuntimeError):
            model(left[:, :49], right[:, :49], iters=2)
        for cell in model.update_block.cells:
            self.assertIsNone(cell.q_lif.membrane)


if __name__ == "__main__":
    unittest.main()
