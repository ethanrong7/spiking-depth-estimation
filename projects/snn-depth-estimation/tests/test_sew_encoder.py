import unittest

import torch

from core.sew_encoder import (
    FunctionalLIF,
    SpikingBasicEncoder,
    SpikingMultiBasicEncoder,
)


class SEWEncoderTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_lif_recurrence_is_hard_reset_without_detachment(self):
        lif = FunctionalLIF()
        x = torch.tensor([[[[3.0]]]], requires_grad=True)
        v = torch.tensor([[[[0.5]]]], requires_grad=True)
        spike, next_v = lif(x, v)

        expected_h = v + (x - v) / 2.0
        self.assertEqual(spike.item(), 1.0)
        self.assertEqual(next_v.item(), 0.0)
        next_v.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(v.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(torch.isfinite(v.grad).all())
        self.assertEqual(lif.surrogate_function.alpha, 2.0)
        self.assertEqual(expected_h.item(), 1.75)

    def test_checkpoint_matches_full_graph_outputs_and_gradients(self):
        reference = SpikingBasicEncoder(
            output_dim=8,
            norm_fn="instance",
            downsample=2,
            timesteps=6,
            checkpoint_chunk_size=0,
        ).train()
        checkpointed = SpikingBasicEncoder(
            output_dim=8,
            norm_fn="instance",
            downsample=2,
            timesteps=6,
            checkpoint_chunk_size=3,
            spatial_checkpoint=True,
        ).train()
        checkpointed.load_state_dict(reference.state_dict())

        input_reference = torch.randn(1, 6, 16, 16, requires_grad=True)
        input_checkpointed = input_reference.detach().clone().requires_grad_(True)
        output_reference = reference(input_reference)
        output_checkpointed = checkpointed(input_checkpointed)
        self.assertTrue(
            torch.allclose(output_reference, output_checkpointed, atol=1e-6, rtol=1e-5)
        )

        output_reference.square().mean().backward()
        output_checkpointed.square().mean().backward()
        self.assertTrue(
            torch.allclose(
                input_reference.grad,
                input_checkpointed.grad,
                atol=1e-5,
                rtol=2e-5,
            )
        )
        for (name_a, parameter_a), (name_b, parameter_b) in zip(
            reference.named_parameters(), checkpointed.named_parameters()
        ):
            self.assertEqual(name_a, name_b)
            self.assertIsNotNone(parameter_a.grad, name_a)
            self.assertIsNotNone(parameter_b.grad, name_b)
            self.assertTrue(
                torch.allclose(
                    parameter_a.grad,
                    parameter_b.grad,
                    atol=2e-4,
                    rtol=2e-5,
                ),
                name_a,
            )

        first_input_gradient = input_checkpointed.grad.clone()
        first_parameter_gradients = {
            name: parameter.grad.clone()
            for name, parameter in checkpointed.named_parameters()
        }
        checkpointed.zero_grad(set_to_none=True)
        replay_input = input_checkpointed.detach().clone().requires_grad_(True)
        replay_output = checkpointed(replay_input)
        self.assertTrue(
            torch.equal(output_checkpointed.detach(), replay_output.detach())
        )
        replay_output.square().mean().backward()
        self.assertTrue(torch.equal(first_input_gradient, replay_input.grad))
        for name, parameter in checkpointed.named_parameters():
            self.assertTrue(
                torch.equal(first_parameter_gradients[name], parameter.grad), name
            )

    def test_frozen_batchnorm_is_unchanged_by_checkpoint_replay(self):
        encoder = SpikingMultiBasicEncoder(
            output_dim=[[8, 8, 8], [8, 8, 8]],
            norm_fn="batch",
            downsample=2,
            timesteps=2,
            checkpoint_chunk_size=1,
        ).train()
        batch_norms = [
            module for module in encoder.modules() if isinstance(module, torch.nn.BatchNorm2d)
        ]
        for module in batch_norms:
            module.eval()
        before = [(module.running_mean.clone(), module.running_var.clone()) for module in batch_norms]

        outputs = encoder(torch.randn(1, 2, 16, 16))
        sum(value.mean() for scale in outputs for value in scale).backward()

        for module, (mean, variance) in zip(batch_norms, before):
            self.assertTrue(torch.equal(module.running_mean, mean))
            self.assertTrue(torch.equal(module.running_var, variance))

    def test_context_checkpoint_matches_full_graph_outputs_and_gradients(self):
        reference = SpikingMultiBasicEncoder(
            output_dim=[[8, 8, 8], [8, 8, 8]],
            norm_fn="instance",
            downsample=2,
            timesteps=6,
            checkpoint_chunk_size=0,
        ).train()
        checkpointed = SpikingMultiBasicEncoder(
            output_dim=[[8, 8, 8], [8, 8, 8]],
            norm_fn="instance",
            downsample=2,
            timesteps=6,
            checkpoint_chunk_size=3,
            spatial_checkpoint=True,
        ).train()
        checkpointed.load_state_dict(reference.state_dict())
        spike_values = []
        checkpointed_spike_values = []
        lif_count = sum(isinstance(module, FunctionalLIF) for module in reference.modules())
        handles = []
        for values, encoder in (
            (spike_values, reference),
            (checkpointed_spike_values, checkpointed),
        ):
            handles.extend(
                module.register_forward_hook(
                    lambda _module, _inputs, output, target=values: target.append(
                        output[0].detach().sum()
                    )
                )
                for module in encoder.modules()
                if isinstance(module, FunctionalLIF)
            )

        input_reference = torch.randn(1, 6, 32, 32, requires_grad=True)
        input_checkpointed = input_reference.detach().clone().requires_grad_(True)
        outputs_reference = reference(input_reference)
        outputs_checkpointed = checkpointed(input_checkpointed)
        for handle in handles:
            handle.remove()
        self.assertGreater(float(torch.stack(spike_values).sum()), 0.0)
        self.assertGreater(float(torch.stack(checkpointed_spike_values).sum()), 0.0)
        self.assertEqual(len(spike_values), lif_count * 6)
        self.assertEqual(len(checkpointed_spike_values), lif_count * 6)
        flat_reference = [value for scale in outputs_reference for value in scale]
        flat_checkpointed = [value for scale in outputs_checkpointed for value in scale]
        for value_reference, value_checkpointed in zip(
            flat_reference, flat_checkpointed
        ):
            self.assertTrue(
                torch.allclose(
                    value_reference, value_checkpointed, atol=1e-6, rtol=1e-5
                )
            )

        sum(value.square().mean() for value in flat_reference).backward()
        sum(value.square().mean() for value in flat_checkpointed).backward()
        self.assertTrue(
            torch.allclose(
                input_reference.grad,
                input_checkpointed.grad,
                atol=1e-5,
                rtol=2e-5,
            )
        )
        for (name_a, parameter_a), (name_b, parameter_b) in zip(
            reference.named_parameters(), checkpointed.named_parameters()
        ):
            self.assertEqual(name_a, name_b)
            self.assertIsNotNone(parameter_a.grad, name_a)
            self.assertIsNotNone(parameter_b.grad, name_b)
            self.assertTrue(
                torch.allclose(
                    parameter_a.grad,
                    parameter_b.grad,
                    atol=2e-4,
                    rtol=2e-5,
                ),
                name_a,
            )
        reference_parameters = dict(reference.named_parameters())
        for name in (
            "conv1.weight",
            "layer3.block1.conv1.weight",
            "layer5.block1.conv1.weight",
            "outputs08.0.projection.weight",
            "outputs32.0.weight",
        ):
            self.assertGreater(float(reference_parameters[name].grad.norm()), 0.0, name)

        first_input_gradient = input_checkpointed.grad.clone()
        first_parameter_gradients = {
            name: parameter.grad.clone()
            for name, parameter in checkpointed.named_parameters()
        }
        checkpointed.zero_grad(set_to_none=True)
        replay_input = input_checkpointed.detach().clone().requires_grad_(True)
        replay_outputs = checkpointed(replay_input)
        flat_replay = [value for scale in replay_outputs for value in scale]
        for first, replay in zip(flat_checkpointed, flat_replay):
            self.assertTrue(torch.equal(first.detach(), replay.detach()))
        sum(value.square().mean() for value in flat_replay).backward()
        self.assertTrue(torch.equal(first_input_gradient, replay_input.grad))
        for name, parameter in checkpointed.named_parameters():
            self.assertTrue(
                torch.equal(first_parameter_gradients[name], parameter.grad), name
            )

    def test_training_batchnorm_is_rejected_for_checkpoint_replay(self):
        encoder = SpikingMultiBasicEncoder(
            output_dim=[[8, 8, 8], [8, 8, 8]],
            norm_fn="batch",
            downsample=2,
            timesteps=2,
            checkpoint_chunk_size=1,
        ).train()
        with self.assertRaisesRegex(RuntimeError, "frozen BatchNorm"):
            encoder(torch.randn(1, 2, 16, 16))

    def test_no_grad_and_checkpoint_replay_use_identical_chunk_grouping(self):
        feature = SpikingBasicEncoder(
            output_dim=8,
            norm_fn="instance",
            downsample=2,
            timesteps=6,
            checkpoint_chunk_size=3,
            spatial_checkpoint=True,
        ).train()
        context = SpikingMultiBasicEncoder(
            output_dim=[[8, 8, 8], [8, 8, 8]],
            norm_fn="instance",
            downsample=2,
            timesteps=6,
            checkpoint_chunk_size=3,
            spatial_checkpoint=True,
        ).train()
        for encoder, size in ((feature, 16), (context, 32)):
            input_value = torch.randn(1, 6, size, size)
            with torch.no_grad():
                expected = encoder(input_value)
            actual = encoder(input_value.requires_grad_(True))
            expected_flat = (
                [value for scale in expected for value in scale]
                if isinstance(encoder, SpikingMultiBasicEncoder)
                else [expected]
            )
            actual_flat = (
                [value for scale in actual for value in scale]
                if isinstance(encoder, SpikingMultiBasicEncoder)
                else [actual]
            )
            for expected_value, actual_value in zip(expected_flat, actual_flat):
                self.assertTrue(torch.equal(expected_value, actual_value.detach()))

    def test_reset_is_implicit_and_repeatable(self):
        encoder = SpikingBasicEncoder(
            output_dim=8,
            norm_fn="instance",
            downsample=2,
            timesteps=3,
        ).eval()
        spikes = torch.randn(1, 3, 16, 16)
        first = encoder(spikes)
        second = encoder(spikes)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(list(encoder.named_buffers()), [])

    def test_production_encoder_requires_all_fifty_bins(self):
        encoder = SpikingBasicEncoder(
            output_dim=4,
            norm_fn="instance",
            downsample=2,
            timesteps=50,
        ).eval()
        with self.assertRaisesRegex(ValueError, "exactly 50 ordered bins"):
            encoder(torch.randn(1, 49, 8, 8))

    def test_full_fifty_step_graph_reaches_first_and_last_bins(self):
        encoder = SpikingBasicEncoder(
            output_dim=4,
            norm_fn="instance",
            downsample=2,
            timesteps=50,
            checkpoint_chunk_size=5,
            spatial_checkpoint=True,
        ).train()
        spikes = torch.randn(1, 50, 8, 8, requires_grad=True)
        encoder(spikes).square().mean().backward()
        bin_norms = spikes.grad.flatten(2).norm(dim=2)
        self.assertGreater(float(bin_norms[0, 0]), 0.0)
        self.assertGreater(float(bin_norms[0, -1]), 0.0)
        self.assertTrue(torch.isfinite(bin_norms).all())


if __name__ == "__main__":
    unittest.main()
