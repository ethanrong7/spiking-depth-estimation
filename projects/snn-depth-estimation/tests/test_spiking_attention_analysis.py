"""Tiny-array checks for saved-prediction analysis, no model/dataset execution."""
import importlib.util
import unittest
from pathlib import Path

import numpy as np

PATH = Path(__file__).resolve().parents[1] / "experiments/spiking-transformer-stereo/scripts/analyze_results.py"
SPEC = importlib.util.spec_from_file_location("attention_analysis", PATH)
ANALYSIS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYSIS)


class AnalysisTest(unittest.TestCase):
    def test_spatial_shape_and_bias_are_distinct(self):
        truth = np.array([[0., 1., 3.], [0., 1., 3.]])
        valid = np.ones_like(truth, dtype=bool)
        shifted = ANALYSIS.image_metrics(truth + 2, truth, valid, np.zeros_like(truth))
        self.assertEqual(shifted["all_epe"], 2)
        self.assertAlmostEqual(shifted["centered_spatial_mae"], 0)
        self.assertAlmostEqual(shifted["spatial_correlation"], 1)
        self.assertEqual(shifted["all_bad1"], 100)
        self.assertEqual(shifted["all_bad3"], 0)
        constant = ANALYSIS.image_metrics(np.full_like(truth, truth.mean()), truth, valid, truth * 0)
        self.assertIsNone(constant["spatial_correlation"])
        self.assertGreater(constant["centered_spatial_mae"], 0)

    def test_invalid_boundary_cannot_create_edge(self):
        image = np.array([[1., 1000., 1.], [1., 1000., 1.]])
        valid = np.array([[True, False, True], [True, False, True]])
        self.assertEqual(ANALYSIS.gradient_magnitude(image, valid).sum(), 0)
        with self.assertRaises(ValueError):
            ANALYSIS.image_metrics(image * np.nan, image, valid, image)

    def test_scene_pairing_and_bootstrap(self):
        reference = {"a": {"all_epe": 3.}, "b": {"all_epe": 5.}}
        candidate = {"a": {"all_epe": 2.}, "b": {"all_epe": 4.}}
        result = ANALYSIS.paired_bootstrap(reference, candidate, samples=100)
        self.assertEqual(result["candidate_minus_reference_scene_epe"], -1)
        self.assertEqual(result["descriptive_scene_bootstrap_95_interval"], [-1, -1])
        with self.assertRaises(ValueError):
            ANALYSIS.paired_bootstrap(reference, {"a": {"all_epe": 2.}})

    def test_geometric_out_of_view_and_constrained_oracle_floor(self):
        truth = np.array([[0., -3., -1.], [0., -3., -1.]])
        result = ANALYSIS.image_metrics(truth, truth, np.ones_like(truth, dtype=bool), truth * 0)
        self.assertEqual(result["all_epe"], 0)
        self.assertEqual(result["out_of_view_pixels"], 2)
        self.assertEqual(result["in_view_pixels"], 4)
        self.assertAlmostEqual(result["out_of_view_valid_fraction"], 1 / 3)
        self.assertAlmostEqual(result["physical_in_view_oracle_floor_epe"], 2 / 3)
        self.assertEqual(result["inspired_padded_head_oracle_floor_epe"], 0)
        np.testing.assert_allclose(ANALYSIS.padded_head_disparity_bound(400), np.arange(400) + 6.5)
        # Compare the analytic support to the actual interpolation used by the head.
        import torch
        coarse_max = torch.arange(104, dtype=torch.float64).reshape(1, 1, 1, 104)
        padded_max = torch.nn.functional.interpolate(coarse_max, size=(1, 416),
                                                     mode="bilinear", align_corners=False) * 4
        np.testing.assert_allclose(ANALYSIS.padded_head_disparity_bound(400),
                                   padded_max[0, 0, 0, 8:408].numpy())


if __name__ == "__main__":
    unittest.main()
