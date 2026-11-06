"""Tiny synthetic CPU implementation checks, not training/accuracy or GPU fit."""
from pathlib import Path
import math
import sys
import unittest
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from sensor_time_ssde import (build_model, state_step, _HardATan,
                              MatchingCompletionHead, global_disparity_expectation)


class SensorTimeChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    @staticmethod
    def inputs(batch=1):
        generator = torch.Generator().manual_seed(901)
        return tuple(torch.randint(0, 2, (batch, 50, 8, 12), generator=generator).float() for _ in range(2))

    def test_common_initialization_rng_and_fixed_graph(self):
        before = torch.get_rng_state().clone()
        hard, continuous = build_model("hard", 1234), build_model("continuous", 1234)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        for key, value in hard.state_dict().items():
            torch.testing.assert_close(value, continuous.state_dict()[key], atol=0, rtol=0)
        self.assertEqual([layer.cross for layer in hard.layers], [False, True] * 3)
        self.assertEqual(hard._padding(250, 400), (8, 8, 3, 3))
        states = hard._initial_states(torch.zeros(2, 50, 32, 32))
        self.assertEqual(len(states), 17)
        self.assertTrue(all(value.count_nonzero() == 0 for value in states))
        self.assertTrue(all(not hasattr(module, "membrane") for module in hard.modules()))
        self.assertEqual(hard.head.residual_conv2.weight.count_nonzero().item(), 0)
        self.assertEqual(hard.head.residual_conv2.bias.count_nonzero().item(), 0)

    def test_hard_binary_actual_cdf_and_matched_local_derivative_reset(self):
        margins = torch.tensor([-2., -.3, 0., .7, 2.])
        hard_input = margins.clone().requires_grad_()
        smooth_input = margins.clone().requires_grad_()
        hard = _HardATan.apply(hard_input)
        smooth = .5 + torch.atan(math.pi * smooth_input) / math.pi
        self.assertEqual(set(hard.tolist()), {0., 1.})
        self.assertEqual(hard[2].item(), 1.)
        self.assertEqual(smooth[2].item(), .5)
        self.assertTrue(torch.all((smooth > 0) & (smooth < 1)))
        hard.sum().backward(); smooth.sum().backward()
        expected = 1 / (1 + (math.pi * margins).square())
        torch.testing.assert_close(hard_input.grad, expected)
        torch.testing.assert_close(smooth_input.grad, expected)
        for mode in ("hard", "continuous"):
            current = torch.tensor([.7], requires_grad=True)
            previous = torch.tensor([.2], requires_grad=True)
            _, membrane = state_step(current, previous, mode)
            membrane.sum().backward()
            margin = .9 * .2 + .7 - 1
            derivative = 1 - 1 / (1 + (math.pi * margin) ** 2)
            torch.testing.assert_close(current.grad, torch.tensor([derivative]))
            torch.testing.assert_close(previous.grad, torch.tensor([.9 * derivative]))

    def test_recurrent_path_early_middle_final_gradient_without_mean_shortcut(self):
        # Read ONLY the final membrane of one specified state unit. There is no
        # temporal-mean/output-sum path, and every transition participates.
        for mode in ("hard", "continuous"):
            currents = torch.full((50, 1), .02, requires_grad=True)
            membrane = torch.zeros(1)
            for index in range(50):
                _, membrane = state_step(currents[index], membrane, mode)
            membrane.sum().backward()
            for index in (0, 24, 49):
                self.assertTrue(torch.isfinite(currents.grad[index]).all())
                self.assertGreater(currents.grad[index].abs().sum().item(), 0)

    def test_full50_checkpoint_output_and_parameter_input_gradients(self):
        for mode in ("hard", "continuous"):
            plain, recomputed = build_model(mode, 31), build_model(mode, 31)
            inputs = self.inputs()
            first = tuple(value.clone().requires_grad_() for value in inputs)
            second = tuple(value.clone().requires_grad_() for value in inputs)
            output_plain = plain(*first, checkpointed=False)
            output_checkpoint = recomputed(*second, checkpointed=True)
            self.assertEqual(output_plain.shape, (1, 1, 8, 12))
            torch.testing.assert_close(output_plain, output_checkpoint, atol=1e-6, rtol=1e-5)
            target = torch.full_like(output_plain, -20.)
            F.smooth_l1_loss(output_plain, target).backward()
            F.smooth_l1_loss(output_checkpoint, target).backward()
            for (name, a), (other_name, b) in zip(plain.named_parameters(), recomputed.named_parameters()):
                self.assertEqual(name, other_name)
                self.assertIsNotNone(a.grad, name)
                self.assertIsNotNone(b.grad, name)
                self.assertTrue(torch.isfinite(a.grad).all(), name)
                torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=2e-4, msg=name)
            for a, b in zip(first, second):
                torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=2e-4)
                for index in (0, 24, 49):
                    self.assertGreater(a.grad[:, index].abs().sum().item(), 0)
            for parameter in (plain.features.convs[0].weight, plain.layers[0].attention.in_proj_weight,
                              plain.head.match_right.weight, plain.head.residual_conv2.weight):
                self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_full_chunk_state_threading_without_temporal_sum_readout(self):
        # Discard all chunk output sums. Only the final sixth-layer FF membrane
        # defines this loss, so a detached state boundary cannot use the mean
        # descriptor as an alternate path to early or middle inputs.
        for mode in ("hard", "continuous"):
            model = build_model(mode, 42)
            packed = torch.randint(0, 2, (2, 50, 32, 32),
                                   generator=torch.Generator().manual_seed(123)).float().requires_grad_()
            states = model._initial_states(packed)
            for start in range(0, 50, 10):
                output = model._run_chunk(packed[:, start:start + 10], *states)
                states = output[2:]
            states[-1].square().mean().backward()
            for index in (0, 24, 49):
                self.assertTrue(torch.isfinite(packed.grad[:, index]).all())
                self.assertGreater(packed.grad[:, index].abs().sum().item(), 0)

    def test_shared_feature_weights_separate_view_states(self):
        model = build_model("hard", 5)
        generator = torch.Generator().manual_seed(6)
        plane = torch.randint(0, 2, (2, 1, 32, 32), generator=generator).float()
        states = model._initial_states(torch.zeros(2, 50, 32, 32))[:5]
        with torch.no_grad():
            first, first_states, _ = model.features(plane, states, "hard")
            changed = plane.clone(); changed[1] = 1 - changed[1]
            second, second_states, _ = model.features(changed, states, "hard")
            swapped, swapped_states, _ = model.features(plane.flip(0), states, "hard")
        torch.testing.assert_close(first[0], second[0], atol=0, rtol=0)
        torch.testing.assert_close(first.flip(0), swapped, atol=0, rtol=0)
        for a, b, c in zip(first_states, second_states, swapped_states):
            torch.testing.assert_close(a[0], b[0], atol=0, rtol=0)
            torch.testing.assert_close(a.flip(0), c, atol=0, rtol=0)

    def test_batch_reset_isolation_and_detached_scalar_diagnostics(self):
        for mode in ("hard", "continuous"):
            model = build_model(mode, 17).eval()
            left, right = self.inputs(batch=2)
            with torch.no_grad():
                batched = model(left, right)
                first, diagnostics = model(left[:1], right[:1], return_diagnostics=True)
                model(left[1:], right[1:])
                repeated = model(left[:1], right[:1])
            torch.testing.assert_close(batched[:1], first, atol=2e-5, rtol=1e-5)
            torch.testing.assert_close(first, repeated, atol=0, rtol=0)
            self.assertTrue(all(isinstance(value, (float, int)) for value in diagnostics.values()))
            prefix = "hard_firing_rate" if mode == "hard" else "continuous_activation_mean"
            rates = [value for key, value in diagnostics.items() if key.startswith(prefix)]
            self.assertEqual(len(rates), 17)
            self.assertTrue(all(0 <= value <= 1 for value in rates))
            self.assertEqual(diagnostics["sensor_steps"], 50)
            self.assertEqual(diagnostics["residual_abs_max_coarse"], 0)

    def test_global_head_far_gradient_border_zero_residual_and_out_of_view(self):
        logits = torch.zeros(1, 8, 8)
        logits[0, 7, 6] = 2
        logits.requires_grad_()
        disparity = global_disparity_expectation(logits)
        self.assertEqual(disparity[0, 0].item(), 0)
        self.assertTrue(torch.isfinite(disparity).all())
        (disparity[0, 7] - 6).square().backward()
        self.assertGreater(logits.grad[0, 7, 0].abs().item(), 0)  # far outside old local window
        head = MatchingCompletionHead()
        left, right = torch.randn(2, 8, 64), torch.randn(2, 8, 64)
        d, d0, residual = head(left, right, batch=1, rows=2)
        torch.testing.assert_close(d, d0, atol=0, rtol=0)
        self.assertEqual(residual.count_nonzero().item(), 0)
        with torch.no_grad():
            head.residual_conv2.bias.fill_(12)
        d, _, _ = head(left, right, batch=1, rows=2)
        self.assertTrue(torch.all(d > torch.arange(8)[None, None, None]))
        # All labelled out-of-view targets stay in an externally supplied mask;
        # the model has no GT/visibility argument with which to remove them.
        loss = F.smooth_l1_loss(-4 * d, torch.full_like(d, -128.))
        loss.backward()
        self.assertGreater(head.residual_conv2.bias.grad.abs().item(), 0)

    def test_eval_fields_same_forward_units_and_default_equivalence(self):
        for mode in ("hard", "continuous"):
            model = build_model(mode, 27)
            left, right = self.inputs()
            with torch.no_grad(), self.assertRaises(ValueError):
                model(left, right, return_fields=True)
            model.eval()
            with self.assertRaises(ValueError):
                model(left, right, return_fields=True)
            with torch.no_grad():
                ordinary = model(left, right)
                flow, scalars, fields = model(left, right, return_fields=True)
            torch.testing.assert_close(flow, ordinary, atol=0, rtol=0)
            self.assertEqual(set(fields), {"d0", "r", "d", "p"})
            self.assertEqual(fields["p"].shape, (1, 8, 8, 8))
            for name in ("d0", "r", "d"):
                self.assertEqual(fields[name].shape, (1, 1, 8, 8))
            self.assertTrue(all(not value.requires_grad and value.grad_fn is None for value in fields.values()))
            torch.testing.assert_close(fields["d"], fields["d0"] + fields["r"], atol=0, rtol=0)
            torch.testing.assert_close(fields["p"].sum(-1), torch.ones(1, 8, 8))
            self.assertEqual(fields["p"].triu(1).count_nonzero().item(), 0)
            index = torch.arange(8)
            expected = (fields["p"] * (index[:, None] - index[None, :])).sum(-1)
            torch.testing.assert_close(fields["d0"][:, 0], expected, atol=0, rtol=0)
            padded_flow = -4 * F.interpolate(fields["d"], size=(32, 32), mode="bilinear", align_corners=False)
            torch.testing.assert_close(flow, padded_flow[..., 12:20, 10:22], atol=0, rtol=0)
            self.assertTrue(all(isinstance(value, (int, float)) for value in scalars.values()))

    def test_native_units_sign_unbounded_residual_and_input_contract(self):
        model = build_model("hard", 9)
        left, right = self.inputs()
        with torch.no_grad():
            zero_residual = model(left, right)
            model.head.residual_conv2.bias.fill_(12.)
            positive_residual = model(left, right)
            model.head.residual_conv2.bias.fill_(-100.)
            negative_disparity = model(left, right)
        self.assertTrue(torch.all(zero_residual <= 0))
        torch.testing.assert_close(positive_residual - zero_residual, torch.full_like(zero_residual, -48.), atol=1e-5, rtol=1e-6)
        self.assertTrue(torch.all(negative_disparity > 0))  # deliberately not clamped
        for bad_left, bad_right in ((left[:, :49], right[:, :49]), (left[:, :51], right[:, :49]),
                                    (2 * left - 1, right), (left.double(), right.double()), (left, right[:, :, :, :-1])):
            with self.assertRaises(ValueError):
                model(bad_left, bad_right)
        with self.assertRaises(ValueError):
            build_model("other", 0)


if __name__ == "__main__":
    unittest.main()
