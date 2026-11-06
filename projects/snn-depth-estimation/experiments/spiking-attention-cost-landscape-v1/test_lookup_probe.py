"""Tiny synthetic CPU checks only; never load checkpoints or datasets."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.corr import CorrBlock1D
from core.spiking_attention_phase2 import CELLS, install
import core.spikestereonet as native_model
from lookup_probe import observe_lookups


class LinearProfile(CorrBlock1D):
    def __init__(self):
        self.radius, self.num_levels = 1, 2
        self.corr_pyramid = [(3 * torch.arange(width).float() + 2).reshape(1, 1, 1, width)
                             for width in (16, 8)]


class ProbeTest(unittest.TestCase):
    def test_exact_return_gradient_and_known_derivative(self):
        native = SimpleNamespace(CorrBlock1D=LinearProfile)
        coords = torch.tensor([[[[6.25]], [[0.0]]]], requires_grad=True)
        expected = LinearProfile()(coords)
        expected.sum().backward()
        expected_gradient = coords.grad.clone()
        coords.grad = None
        records = []
        with observe_lookups(native, lambda c: torch.ones(1, 1, 1, dtype=torch.bool), records.append):
            actual = native.CorrBlock1D()(coords)
            actual.sum().backward()
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(coords.grad, expected_gradient))
        self.assertIs(native.CorrBlock1D, LinearProfile)
        self.assertEqual(len(records), 1)
        for level, expected_slope in enumerate((3., 1.5)):
            row = records[0]['levels'][level]
            self.assertEqual(row['slope_per_base_feature_cell']['count'], 3)
            self.assertAlmostEqual(row['slope_per_base_feature_cell']['mean'], expected_slope, places=5)
            self.assertLess(row['curvature_per_base_feature_cell_squared']['abs_mean'], 1e-4)

    def test_boundary_and_empty_support_remain_explicit(self):
        native = SimpleNamespace(CorrBlock1D=LinearProfile)
        records = []
        coords = torch.zeros(1, 2, 1, 1)
        with observe_lookups(native, lambda c: torch.ones(1, 1, 1, dtype=torch.bool), records.append):
            native.CorrBlock1D()(coords)
        for row in records[0]['levels']:
            self.assertEqual(row['lookup']['count'], 3)
            self.assertEqual(row['inbounds_lookup']['count'], 1)
            self.assertEqual(row['profile_range']['count'], 0)
            self.assertIsNone(row['profile_range']['mean'])
        records.clear()
        with observe_lookups(native, lambda c: torch.zeros(1, 1, 1, dtype=torch.bool), records.append):
            native.CorrBlock1D()(coords)
        self.assertTrue(all(r['lookup']['count'] == 0 for r in records[0]['levels']))

    def test_actual_coordinates_and_rounded_zero_step(self):
        native = SimpleNamespace(CorrBlock1D=LinearProfile)
        records = []
        coords = torch.tensor([[[[6.25]], [[0.0]]]])
        with observe_lookups(native, lambda c: torch.ones(1, 1, 1, dtype=torch.bool),
                             records.append, epsilon=1e-10):
            native.CorrBlock1D()(coords)
        self.assertTrue(torch.equal(records[0]['coordinates_base_feature_cells'], coords))
        self.assertEqual(records[0]['actual_lower_delta'].item(), 0)
        for row in records[0]['levels']:
            self.assertEqual(row['slope_per_base_feature_cell']['count'], 0)
            self.assertIsNone(row['curvature_per_base_feature_cell_squared']['mean'])

    def test_failure_restores_class(self):
        native = SimpleNamespace(CorrBlock1D=LinearProfile)
        with self.assertRaises(ValueError):
            with observe_lookups(native, lambda c: torch.ones(1, 1, 1), lambda r: None):
                native.CorrBlock1D()(torch.zeros(1, 2, 1, 1))
        self.assertIs(native.CorrBlock1D, LinearProfile)

    def test_installed_phase2_cells_unchanged(self):
        original = native_model.CorrBlock1D
        try:
            torch.manual_seed(20)
            left, right = torch.randn(1, 2, 1, 32), torch.randn(1, 2, 1, 32)
            coords = torch.zeros(1, 2, 1, 32)
            coords[:, 0] = torch.arange(32) + .25
            for cell in CELLS:
                model = install(SimpleNamespace(), cell)
                model.placement.eval()
                with torch.no_grad():
                    model.placement.output.weight.fill_(.1)
                state = {k: v.clone() for k, v in model.placement.state_dict().items()}
                with torch.no_grad():
                    expected = native_model.CorrBlock1D(left, right, 2, 1)(coords)
                    records = []
                    with observe_lookups(native_model, lambda c: torch.ones(1, 1, 32, dtype=torch.bool), records.append):
                        actual = native_model.CorrBlock1D(left, right, 2, 1)(coords)
                self.assertTrue(torch.equal(expected, actual), cell)
                self.assertEqual(len(records), 1)
                self.assertTrue(all(torch.equal(state[k], v) for k, v in model.placement.state_dict().items()))
        finally:
            native_model.CorrBlock1D = original


if __name__ == '__main__':
    unittest.main()
