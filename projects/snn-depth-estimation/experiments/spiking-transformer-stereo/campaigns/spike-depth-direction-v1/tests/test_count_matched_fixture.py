"""Eleven named tiny software checks; never render the complete fixture/oracle."""
import importlib.util
from fractions import Fraction as F
from pathlib import Path
import unittest

import numpy as np

SPEC = importlib.util.spec_from_file_location("s7fixture", Path(__file__).resolve().parents[1] / "scripts/count_matched_fixture.py")
s7 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(s7)


def record(events):
    result = np.zeros((50, 1), dtype=np.uint8)
    result[list(events), 0] = 1
    return result


class TinyFixtureChecks(unittest.TestCase):
    def test_pinhole_sign_known_world_projection(self):
        left = s7.project_world(F(0), F(0), F(6, 5), "L")
        right = s7.project_world(F(0), F(0), F(6, 5), "R")
        self.assertEqual(left, (F(259, 10), F(31, 2)))
        self.assertEqual(right, (F(211, 10), F(31, 2)))
        self.assertEqual(left[0] - right[0], F(s7.DISPNUM, 6_000_000))

    def test_rectangle_texture_half_open_boundaries(self):
        # Independent rational locations on every half-open outer edge and a checker edge.
        cases = [(F(-1, 4), F(-7, 40), True, 0), (F(1, 4), F(0), False, 0),
                 (F(0), F(7, 40), False, 0), (F(-1, 4)-F(1, 1000), F(0), False, 0),
                 (F(-1, 5), F(-7, 40), True, 1), (F(-1, 5)-F(1, 1000), F(-7, 40), True, 0)]
        for x, y, hit, intensity in cases:
            den = 2000
            found, charge = s7.classify_local(int(x*den), int(y*den), den)
            self.assertEqual(bool(found), hit)
            self.assertEqual(int(charge), intensity)

    def test_background_occlusion_including_black_foreground(self):
        # k0 rectangle Z1.2: background left point at u15,v15 is in right FoV
        # but its continuous right ray meets the rectangle. Test by independent Fractions.
        k, u, v = 0, 15, 15
        zp, zr = F(5, 2), F(6, 5)
        world_x = F(-3, 50) + (F(u)-F(47, 2))*zp/48
        x_at_rect = F(3, 50) + (world_x-F(3, 50))*zr/zp
        self.assertTrue(F(-1, 4) <= x_at_rect < F(1, 4))
        self.assertFalse(bool(s7.target_visibility(k, s7.BG, u, v)))
        charge, depth, hit = s7.render_pixels(0, "L", 22, 15)
        self.assertTrue(bool(hit))
        self.assertEqual(int(charge), 0)
        self.assertEqual(int(depth), 6_000_000)  # black checker still foreground depth
        self.assertTrue(bool(s7.target_visibility(0, depth, 22, 15)))
        self.assertFalse(bool(s7.target_visibility(0, s7.BG, 0, 0)))  # out of FoV

    def test_target_phase_indexing_and_regeneration(self):
        self.assertEqual([s7.base_index(24, h) for h in (0, 13, 49)], [24, 37, 23])
        intensity = record(list(range(16)))
        shifted = intensity[(np.arange(50)+13) % 50]
        actual, _ = s7.encode(shifted, 4)
        unshifted, _ = s7.encode(intensity, 4)
        self.assertFalse(np.array_equal(actual, unshifted[(np.arange(50)+13) % 50]))

    def test_integer_charge_conservation_and_reset(self):
        intensity = record([0, 1, 2, 3, 4, 5, 13, 24, 49])
        phi = np.array([3], dtype=np.int16)
        spikes, residual = s7.encode(intensity, phi)
        self.assertEqual(int(residual[0]), 2)
        self.assertEqual(np.flatnonzero(spikes[:, 0]).tolist(), [1, 13])
        self.assertEqual(5*int(spikes.sum()), int(phi[0])+int(intensity.sum())-int(residual[0]))
        fresh, final = s7.encode(intensity, phi)
        np.testing.assert_array_equal(fresh, spikes)
        np.testing.assert_array_equal(final, residual)
        self.assertEqual(s7.charge_field("L", 1, 1).shape, (1, 1))

    def test_zero_record_and_boundary_charge_oracle_strictness(self):
        zero = record([])
        lo, hi = s7.full_bounds(zero, zero)
        self.assertEqual((int(lo[0]), int(hi[0])), (0, 5))
        intensity = record([0])
        lo, hi = s7.full_bounds(zero, intensity)
        self.assertEqual((int(lo[0]), int(hi[0])), (0, 4))
        at_four, _ = s7.encode(intensity, 4)
        lo, hi = s7.full_bounds(at_four, intensity)
        self.assertEqual((int(lo[0]), int(hi[0])), (4, 5))
        # Empty interval is rejected, no equality acceptance at upper boundary.
        lo, hi = s7.full_bounds(record([0]), zero)
        self.assertEqual(int(lo[0]), int(hi[0]))
        self.assertFalse(bool(lo[0] < hi[0]))

    def test_five_cell_equivalence_fractional_charges(self):
        intensity = record([0, 2, 4, 6, 8, 10, 15, 22, 24, 27, 49])
        for r in range(5):
            expected, _ = s7.encode(intensity, r)
            for frac in (F(0), F(1, 4), F(3, 4), F(999, 1000)):
                voltage, actual = F(r)+frac, []
                for i in intensity[:, 0]:
                    voltage += int(i)
                    fired = voltage >= 5
                    actual.append(int(fired))
                    voltage -= 5*int(fired)
                    self.assertTrue(0 <= voltage < 5)
                np.testing.assert_array_equal(expected[:, 0], actual)
                lo, hi, raw, summary, _ = s7.oracle_pixel_evidence(expected, intensity)
                self.assertTrue(int(lo[0]) <= F(r)+frac < int(hi[0]))
                self.assertTrue(int(raw[0]) & (1 << r))
                self.assertTrue(int(summary[0]) & (1 << r))

    def test_summary_sentinels_and_inclusive_windows(self):
        np.testing.assert_array_equal(s7.summary(record([]))[:, 0], [0, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0])
        np.testing.assert_array_equal(s7.summary(record([24]))[:, 0], [1, 24, 24, -1, -1, 24, -1, 1, 1, 1, 1])
        nums, dens = s7.baseline_components(s7.summary(record([24])))
        np.testing.assert_array_equal(dens, np.zeros((3, 1)))
        self.assertEqual(s7.representative_cells(np.array([0, 6, 16], dtype=np.uint8)).tolist(), [-1, 1, 4])
        self.assertEqual(s7.first_incompatible(np.array([[1, 0], [0, 1]], dtype=np.uint8)), [0, 1])
        self.assertIsNone(s7.first_incompatible(np.array([[1]], dtype=np.uint8)))
        events = [0, 12, 16, 20, 23, 24, 28, 32, 36, 48, 49]
        np.testing.assert_array_equal(s7.summary(record(events))[:, 0], [11, 0, 49, 20, 23, 24, 28, 4, 6, 8, 10])

    def test_candidate_containment_depth_equivalence_and_outcomes(self):
        intensity = record([0, 1, 2, 3, 4, 24, 25, 26, 27, 28])
        obs, _ = s7.encode(intensity, 2)
        lo, hi, raw, summaries, count = s7.oracle_pixel_evidence(obs, intensity)
        self.assertTrue(np.all((raw == 0) | (summaries != 0)))
        self.assertTrue(np.all(count != 0))
        mask = np.array([True, False])
        fields = np.array([[5, 9], [5, 2], [6, 9]], dtype=np.int64)
        groups, ids = s7.target_partition(fields, mask)
        self.assertEqual(groups, [[0, 1], [2]])
        correct = s7.describe_set([0, 1], ids, 0)
        wrong = s7.describe_set([0, 2], ids, 0)
        self.assertTrue(correct["all_have_correct_target_field"])
        self.assertFalse(wrong["all_have_correct_target_field"])
        episodes = [{"full": {"stereo": correct}, "summary": {"stereo": wrong}},
                    {"full": {"stereo": correct}, "summary": {"stereo": correct}}]
        self.assertEqual(s7.classify_outcome(F(1), episodes), "RAW_ADDS_TARGET_DISCRIMINATION")
        episodes[0]["summary"]["stereo"] = correct
        self.assertEqual(s7.classify_outcome(F(1), episodes), "COUNT_GAP_SUMMARIES_SUFFICE")
        episodes[0]["full"]["stereo"] = wrong
        self.assertEqual(s7.classify_outcome(F(1), episodes), "COUNT_AMBIGUOUS_RAW_UNRESOLVED")
        self.assertEqual(s7.classify_outcome(F(0), episodes), "NO_COUNT_TARGET_AMBIGUITY")
        self.assertEqual(s7.classify_outcome(None, episodes), "UNSUITABLE_SUPPORT")

    def test_multiset_exact_multiplicity_and_synchronization(self):
        a = np.array([[[0]], [[0]], [[1]]], dtype=np.uint8)
        b = np.array([[[1]], [[0]], [[0]]], dtype=np.uint8)
        c = np.array([[[1]], [[1]], [[0]]], dtype=np.uint8)
        self.assertEqual(s7.plane_multiset(a), s7.plane_multiset(b))
        self.assertNotEqual(s7.plane_multiset(a), s7.plane_multiset(c))
        # Each view multiset can agree while synchronous pair multisets differ.
        left = np.array([[[0]], [[1]]], dtype=np.uint8)
        right = left[::-1]
        self.assertNotEqual(s7.plane_multiset(np.concatenate((left, left), axis=2)),
                            s7.plane_multiset(np.concatenate((left, right), axis=2)))

    def test_two_value_exact_mae_bound(self):
        d0 = np.array([6_000_000, s7.BG])
        d1 = np.array([s7.BG, s7.BG])
        mask = np.array([True, True])
        bound = s7.mae_bound(d0, d1, mask)
        self.assertEqual(bound, F(78, 125))
        # A midpoint common prediction attains the exact paired-pixel average bound.
        a, b = F(s7.DISPNUM, int(d0[0])), F(s7.DISPNUM, int(d1[0]))
        prediction = (a+b)/2
        self.assertEqual((abs(a-prediction)+abs(b-prediction))/4, bound)
        self.assertIsNone(s7.mae_bound(d0, d1, np.array([False, False])))


if __name__ == "__main__":
    unittest.main()
