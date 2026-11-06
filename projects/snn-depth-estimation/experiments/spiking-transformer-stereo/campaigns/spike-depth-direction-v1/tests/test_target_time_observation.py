"""Tiny synthetic S5 semantic checks only; no dataset payload or model imports."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import target_time_observation as s5


def artificial_stream(value, bracket):
    """Measurement fixture, not a claim about a real I&F sequence."""
    value = np.asarray(value, dtype=np.float64)
    bracket = np.asarray(bracket, dtype=bool)
    return {"count": np.full(value.shape, 5, np.int16), "a": np.where(bracket, 14, -1),
            "b": np.where(bracket, 24, -1), "B": bracket,
            "C": value.copy(), "T": np.where(bracket, value, np.nan)}


class TargetTimeChecks(unittest.TestCase):
    def test_codec_exact_bit_order_shape_and_bin24_mapping(self):
        bins = np.zeros((50, 2, 4), dtype=bool)
        bins[0, 0, 0] = bins[1, 0, 1] = bins[49, 1, 3] = True
        raw = s5.codec.pack_spikes(bins)
        self.assertEqual((raw[0], raw[1], raw[-1]), (1, 2, 128))
        np.testing.assert_array_equal(s5.codec.unpack_spikes(raw, 2, 4), bins)
        with self.assertRaises(ValueError):
            s5.codec.unpack_spikes(raw[:-1], 2, 4)
        for center in range(1, 9):
            indices = s5.codec.window_indices(center)
            self.assertEqual((len(indices), indices[0], indices[24], indices[-1]),
                             (50, 50*center-24, 50*center, 50*center+25))

    def test_rgb_scale_channels_and_float32_arithmetic(self):
        rgb = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 0, 0], [255, 255, 255]]], dtype=np.uint8)
        gray = s5.target_gray(rgb)
        self.assertEqual(gray.dtype, np.float32)
        np.testing.assert_allclose(gray, [[.299, .587, .114, 0, 1]], atol=1e-7, rtol=0)
        with self.assertRaises(ValueError):
            s5.target_gray(rgb.astype(np.float32))

    def test_fp32_threshold_overshoot_and_static_no_stack(self):
        spike, voltage = s5.codec.integrate_numpy(np.array([4.5, 4.75], np.float32), np.array([.5, .5], np.float32))
        np.testing.assert_array_equal(spike, [True, True])
        np.testing.assert_array_equal(voltage, [0., .25])
        self.assertEqual(voltage.dtype, np.float32)
        for intensity, count in ((0., 0), (.5, 5), (1., 10)):
            summary = s5.simulate_static(np.full((1, 1), intensity, np.float32), 0.)
            self.assertEqual(summary["count"].item(), count)
        summary = s5.simulate_static(np.full((1, 1), .5, np.float32), 2.5)
        self.assertEqual((summary["a"].item(), summary["b"].item()), (14, 24))
        self.assertEqual(summary["T"].item(), .5)

    def test_target_spike_tie_and_censoring_no_boundary_fallback(self):
        voltage = np.zeros((1, 4), np.float32)
        bins = []
        for k in range(50):
            intensity = np.zeros((1, 4), np.float32)
            intensity[0, 0] = 1. if k <= 24 else 0.
            event, voltage = s5.codec.integrate_numpy(voltage, intensity)
            bins.append(event)
        bins = np.stack(bins)
        bins[30, 0, 1] = True  # missing left only
        bins[20, 0, 2] = True  # missing right only
        summary = s5.summarize_spikes(bins)
        self.assertEqual((summary["a"][0, 0], summary["b"][0, 0]), (19, 24))
        np.testing.assert_array_equal(summary["B"], [[True, False, False, False]])
        self.assertTrue(np.isnan(summary["T"][0, 1:]).all())
        np.testing.assert_array_equal(summary["a"] < 0, [[False, True, False, True]])
        np.testing.assert_array_equal(summary["b"] < 0, [[False, False, True, True]])

    def test_common_mask_uses_all_three_conditions_and_same_error_population(self):
        gray = {v: np.array([[.1, .9]], dtype=np.float32) for v in s5.VIEWS}
        streams = {z: {v: artificial_stream([[.2, .2]], [[True, z != "StaticHalf"]]) for v in s5.VIEWS}
                   for z in s5.CONDITIONS}
        record = s5.measure_frame(gray, streams, np.ones((1, 2)), {"split": "dev", "source_id": "fixture"})
        self.assertEqual(record["support"]["L.common_count"]["value"], 1)
        for condition in s5.CONDITIONS:
            for estimator in ("C", "T"):
                metric = record["metrics"][f"target.L.primary.{condition}.{estimator}.mae"]
                self.assertEqual(metric["support_count"], 1)
                self.assertAlmostEqual(metric["value"], .1, places=7)
        self.assertAlmostEqual(record["metrics"]["target.L.actual_count_all.mae"]["value"], .4, places=7)

    def test_positive_weight_bilinear_integer_border_and_nan_neighbor(self):
        # Native positive disparity cannot sample the final column, but the
        # row sampler must still handle it correctly without a ceil+1 read.
        q = np.array([[0., .5, 3., np.nan]])
        mask = np.ones((1, 4), bool)
        right = np.array([[True, False, True, True]])
        geometry = s5.horizontal_support(q, mask, mask, right)
        np.testing.assert_array_equal(geometry["J"], [[True, False, True, False]])
        warped = s5.warp_selected(np.array([[.2, np.nan, .6, .8]]), geometry)
        np.testing.assert_array_equal(warped, [.2, .8])
        q[0, 0] = np.nextafter(np.float64(0), np.float64(1))
        self.assertFalse(s5.horizontal_support(q, mask, mask, right)["J"][0, 0])

    def test_known_translation_target_difference_not_plain_photo_error(self):
        gray = {"R": np.array([[.1, .2, .3, .4]], np.float32),
                "L": np.array([[.8, .2, .3, .4]], np.float32)}
        streams = {z: {v: artificial_stream(gray[v], np.ones((1, 4), bool)) for v in s5.VIEWS}
                   for z in s5.CONDITIONS}
        record = s5.measure_frame(gray, streams, np.ones((1, 4)), {"split": "dev", "source_id": "fixture"})
        self.assertEqual(record["support"]["stereo.F_count"]["value"], 3)
        for condition in s5.CONDITIONS:
            for estimator in ("C", "T"):
                prefix = f"stereo.{condition}.{estimator}"
                self.assertLessEqual(record["metrics"][prefix + ".fidelity.mae"]["value"], 1e-12)
                self.assertGreater(record["metrics"][prefix + ".plain.mae"]["value"], .09)
                self.assertEqual(record["metrics"][prefix + ".individual_L.mae"]["value"], 0)
                self.assertEqual(record["metrics"][prefix + ".individual_R.mae"]["value"], 0)

    def test_null_empty_masks_retained_and_equal_frame_scene_reduction(self):
        empty = {z: {v: artificial_stream([[.2, .2]], [[False, False]]) for v in s5.VIEWS}
                 for z in s5.CONDITIONS}
        gray = {v: np.full((1, 2), .2, np.float32) for v in s5.VIEWS}
        record = s5.measure_frame(gray, empty, np.ones((1, 2)), {"split": "dev", "source_id": "empty"})
        self.assertIsNone(record["metrics"]["target.L.primary.Actual.T.mae"]["value"])
        self.assertEqual(record["metrics"]["stereo.Actual.T.fidelity.mae"]["support_count"], 0)
        rows = []
        for scene, value, nullable in (("a", 1., None), ("a", 3., None), ("b", 8., 4.)):
            rows.append({"split": "dev", "source_id": scene, "metrics": {"error": s5.scalar_record(value, 1),
                        "nullable": s5.scalar_record(nullable, 0 if nullable is None else 1)}, "support": {}})
        scenes, splits = s5.reduce_records(rows)
        self.assertEqual(splits["dev"]["metrics"]["error"]["value"], 5.)  # mean(mean(1,3),8)
        self.assertEqual(splits["dev"]["metrics"]["nullable"]["nonempty_scenes"], 1)
        self.assertEqual(splits["dev"]["metrics"]["nullable"]["value"], 4.)

    def test_static_envelope_before_reduction_and_linear_tail(self):
        rows = []
        for static0, static_half in ((.2, 0.), (0., .2)):
            metrics = {}
            s5.add_gains(metrics, "target.L", {"Actual": (.1, 0.), "Static0": (static0, 0.),
                                              "StaticHalf": (static_half, 0.)}, 10)
            rows.append({"split": "dev", "source_id": "same", "metrics": metrics, "support": {}})
        _, splits = s5.reduce_records(rows)
        self.assertAlmostEqual(splits["dev"]["metrics"]["target.L.excess"]["value"], -.1)
        self.assertAlmostEqual(s5.describe([0., 10.], (.9,))["q90"], 9.)


if __name__ == "__main__":
    unittest.main()
