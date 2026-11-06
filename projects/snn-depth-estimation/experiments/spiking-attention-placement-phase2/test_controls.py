"""Tiny CPU-only functional checks; no training/evaluation experiment."""
import sys
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from torch import nn
from core.corr import CorrBlock1D
from core.spiking_attention_placement import CostFilter
from core.spiking_attention_phase2 import CELLS, MatchedCostFilter, install
import core.spikestereonet as native


class ControlsTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(4)

    def test_initial_state_and_parameters(self):
        states = []
        for cell in CELLS:
            torch.manual_seed(2017)
            model = MatchedCostFilter(cell)
            states.append(model.state_dict())
            x = torch.randn(1, 2, 3, 1, 7)
            self.assertTrue(torch.equal(model(x), x))
        torch.manual_seed(2017)
        archived = CostFilter().state_dict()
        for state in states:
            self.assertEqual(list(state), list(archived))
            for key in state:
                self.assertTrue(torch.equal(state[key], archived[key]), key)

    def test_archived_forward_and_gradient_parity(self):
        old, new = CostFilter(), MatchedCostFilter('SC')
        with torch.no_grad():
            old.output.weight.normal_()
            old.output.bias.fill_(.2)
        new.load_state_dict(old.state_dict(), strict=True)
        x = torch.randn(1, 2, 3, 1, 7)
        for training in (False, True):
            old.train(training)
            new.train(training)
            old.zero_grad()
            new.zero_grad()
            a, b = x.clone().requires_grad_(), x.clone().requires_grad_()
            ya, yb = old(a), new(b)
            self.assertTrue(torch.equal(ya, yb))
            ya.square().mean().backward()
            yb.square().mean().backward()
            self.assertTrue(torch.equal(a.grad, b.grad))
            for pa, pb in zip(old.parameters(), new.parameters()):
                self.assertTrue(torch.equal(pa.grad, pb.grad))

    def test_positional_invariance_and_continuous_definition(self):
        x = torch.randn(1, 2, 3, 1, 7)
        changed = x * -3 + 2
        for cell in CELLS:
            torch.manual_seed(2017)
            model = MatchedCostFilter(cell)
            a, b = model.routing_weights(x), model.routing_weights(changed)
            q, k, v = model.projected_codes(x)
            if cell in ('SP', 'CP'):
                self.assertTrue(torch.equal(a, b))
                self.assertTrue(torch.equal(a[0], a[-1]))
                self.assertFalse(torch.equal(v, model.projected_codes(changed)[2]))
            else:
                self.assertFalse(torch.equal(a, b))
            if cell in ('CC', 'CP'):
                rows = x.reshape(-1, 7, 1)
                pos = torch.linspace(-1, 1, 7).view(1, 7, 1).expand(rows.shape[0], -1, -1)
                value_z = model.codes.norm(model.input(torch.cat([rows, pos], -1)))
                route_z = model.codes.norm(model.input(torch.cat([torch.zeros_like(rows), pos], -1))) if cell == 'CP' else value_z
                for actual, layer, z in ((q, model.codes.q, route_z), (k, model.codes.k, route_z), (v, model.codes.v, value_z)):
                    self.assertTrue(torch.equal(actual, torch.tanh(layer(z))))

    def test_head_opening_and_all_projection_gradients(self):
        for cell in CELLS:
            torch.manual_seed(2017)
            model = MatchedCostFilter(cell)
            x = torch.randn(1, 2, 3, 1, 7, requires_grad=True)
            optimizer = torch.optim.SGD(model.parameters(), lr=.1)
            model(x).square().mean().backward()
            self.assertGreater(model.output.weight.grad.norm().item(), 0)
            for layer in (model.codes.q, model.codes.k, model.codes.v):
                self.assertEqual(layer.weight.grad.norm().item(), 0)
            optimizer.step()  # A single tiny synthetic gradient check opens the head.
            optimizer.zero_grad()
            model(x).square().mean().backward()
            for layer in (model.codes.q, model.codes.k, model.codes.v):
                self.assertTrue(torch.isfinite(layer.weight.grad).all())
                self.assertGreater(layer.weight.grad.norm().item(), 0, cell)
            # Shared embedding's actual-cost column stays active through V.
            self.assertGreater(model.input.weight.grad[:, 0].norm().item(), 0, cell)

    def test_boundaries_and_five_candidate_locality(self):
        for cell in CELLS:
            model = MatchedCostFilter(cell).eval()
            with torch.no_grad():
                model.output.weight.normal_()
            for width in (1, 2, 7):
                x = torch.randn(1, 1, 1, 1, width)
                weights = model.routing_weights(x)
                indices = torch.arange(width)[:, None] + torch.arange(-2, 3)[None, :]
                invalid = (indices < 0) | (indices >= width)
                self.assertTrue(torch.equal(weights[0][invalid], torch.zeros_like(weights[0][invalid])))
                torch.testing.assert_close(weights.sum(-1), torch.ones_like(weights[..., 0]))
                self.assertTrue(torch.isfinite(model(x)).all())
            x = torch.randn(1, 1, 2, 1, 7)
            changed = x.clone()
            changed[..., -1] += 50
            self.assertTrue(torch.equal(model(x)[..., :4], model(changed)[..., :4]))
            changed = x.clone()
            changed[:, :, 1] += 50
            self.assertTrue(torch.equal(model(x)[:, :, 0], model(changed)[:, :, 0]))

    def test_diagnostic_weight_variation(self):
        spec = importlib.util.spec_from_file_location('phase2_replay_test', Path(__file__).with_name('replay.py'))
        replay = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(replay)
        for cell in CELLS:
            torch.manual_seed(2017)
            model = MatchedCostFilter(cell)
            values = replay.weight_variation(model, torch.randn(1, 2, 3, 1, 7), max_rows=4)
            self.assertEqual(len(values['sampled_rows']), 4)
            self.assertEqual(values['total_rows'], 6)
            if cell in ('SP', 'CP'):
                self.assertEqual(values['cost_vs_zero_mean_abs'], 0)
                self.assertEqual(values['fixed_position_row_sd_mean'], 0)
            else:
                self.assertGreater(values['cost_vs_zero_mean_abs'], 0)

    def test_install_rng_seeds_and_native_correlation(self):
        for seed in (2017, 3128, 4239):
            states = []
            for cell in CELLS:
                model = nn.Module()
                model.common = nn.Linear(3, 2)
                common = model.common.weight.detach().clone()
                state = torch.random.get_rng_state().clone()
                with patch.object(native, 'CorrBlock1D'):
                    install(model, cell, seed=seed)
                    self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
                    self.assertTrue(torch.equal(common, model.common.weight))
                    left, right = torch.randn(1, 4, 1, 8), torch.randn(1, 4, 1, 8)
                    actual = native.CorrBlock1D(left, right, num_levels=2)
                    expected = CorrBlock1D(left, right, num_levels=2)
                    for a, b in zip(actual.corr_pyramid, expected.corr_pyramid):
                        self.assertTrue(torch.equal(a, b))
                states.append(model.placement.state_dict())
            for state in states[1:]:
                for key in state:
                    self.assertTrue(torch.equal(state[key], states[0][key]))
        with self.assertRaises(ValueError):
            MatchedCostFilter('unknown')


if __name__ == '__main__':
    unittest.main()
