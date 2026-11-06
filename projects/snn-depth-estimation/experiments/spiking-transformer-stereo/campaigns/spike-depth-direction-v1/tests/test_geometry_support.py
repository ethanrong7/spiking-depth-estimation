"""Tiny synthetic software checks only: no dataset/model/checkpoint access."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from geometry_support import box, moments, warp, measure, overlap_report, aggregate
from audit_geometry_support import shifted, control_checks


class GeometryChecks(unittest.TestCase):
    def test_explicit_horizontal_interpolation(self):
        a = np.arange(48, dtype=np.float64).reshape(4, 12)/48
        g = np.broadcast_to(np.linspace(-.5, 1.5, 12), a.shape)
        w, m = warp(a, g, 0)
        for y in range(4):
            for x in range(12):
                q = x-g[y, x]
                good = 0 <= q <= 11
                self.assertEqual(bool(m[y, x]), good)
                if good:
                    lo, hi = int(np.floor(q)), int(np.ceil(q))
                    expected = a[y, lo] if lo == hi else (hi-q)*a[y, lo]+(q-lo)*a[y, hi]
                    self.assertAlmostEqual(w[y, x], expected, places=15)
        w, m = warp(a, np.zeros_like(a), 0)
        self.assertTrue(m[:, -1].all())  # last integer coordinate needs no extrapolated hi
        np.testing.assert_array_equal(w, a)

    def test_box_and_population_moments_direct(self):
        rng = np.random.default_rng(4)
        a, b = rng.random((13, 19)), rng.random((13, 19))
        lm, lv, wm, wv, cov = moments(a, b)
        for y, x in [(0, 0), (6, 8), (12, 18)]:
            aa = np.pad(a, 3)[y:y+7, x:x+7]
            bb = np.pad(b, 3)[y:y+7, x:x+7]
            for actual, expected in [(lm[y, x], aa.mean()), (wm[y, x], bb.mean()),
                                     (lv[y, x], aa.var()), (wv[y, x], bb.var()),
                                     (cov[y, x], (aa*bb).mean()-aa.mean()*bb.mean())]:
                self.assertAlmostEqual(actual, expected, places=14)

    def test_geometric_edges_zero_variance_and_null_rows(self):
        a = np.full((20, 50), .3)
        result = measure(a, a, np.full_like(a, 4.), np.ones(a.shape, bool))
        # x-g±8 valid -> x in [12,45]; full footprint -> x15..42,y3..16
        self.assertEqual(result['support']['common_point'], 20*34)
        self.assertEqual(result['support']['common_patch'], 14*28)
        self.assertEqual(result['support']['common_N'], 0)
        self.assertIsNone(result['costs']['normalized_disagreement']['candidates']['0']['mean'])
        empty = measure(a, a, np.full_like(a, 100.), np.ones(a.shape, bool))
        self.assertEqual(empty['support']['common_patch'], 0)
        self.assertIsNone(empty['costs']['rate_MAE']['paired_margins_vs_zero']['1']['mean'])

    def test_dense_field_and_common_N_direct(self):
        rng = np.random.default_rng(9)
        a, b = rng.random((15, 50)), rng.random((15, 50))
        g = np.tile(np.linspace(1., 5., 50), (15, 1))
        result = measure(a, b, g, np.ones(a.shape, bool))
        counts = {s['count'] for s in result['costs']['normalized_disagreement']['candidates'].values()}
        self.assertEqual(len(counts), 1)
        # Explicit native patch at y7,x25, each neighbor uses its OWN g.
        warped, _ = warp(b, g, 2)
        aa, bb = a[4:11, 22:29], warped[4:11, 22:29]
        cov = np.mean((aa-aa.mean())*(bb-bb.mean()))
        _, lv, _, wv, cc = moments(a, warped)
        self.assertAlmostEqual(cc[7, 25]/np.sqrt(lv[7, 25]*wv[7, 25]), cov/np.sqrt(aa.var()*bb.var()), places=13)

    def test_known_shift_and_affine_radiometry(self):
        a = np.random.default_rng(5).random((20, 60))
        for shift in [0, 4, 12]:
            b, rv, g = shifted(a, shift)
            clean = measure(a, b, g, np.ones(a.shape, bool), rv)
            affine = measure(a, .5*b+.25, g, np.ones(a.shape, bool), rv)
            self.assertTrue(control_checks(clean)['numerical_construction_pass'])
            self.assertTrue(control_checks(affine, True)['numerical_construction_pass'])
            self.assertGreater(affine['costs']['rate_MAE']['candidates']['0']['mean'], 0)

    def test_aggregation_empty_contributors_equal_records(self):
        r = aggregate([{'x': {'mean': 1., 'count': 100}}, {'x': {'mean': 3., 'count': 1}}, {'x': {'mean': None, 'count': 0}}])
        self.assertEqual(r['x.mean']['mean_of_record_summaries'], 2.)
        self.assertEqual(r['x.mean']['contributing_records'], 2)
        self.assertEqual(r['x.mean']['total_records'], 3)

    def test_byte_overlap_unique_ambiguous_and_invalid(self):
        # Small software fixture; actual audit always uses50*12500 bytes.
        a = b''.join(bytes([i])*8 for i in range(6))
        b = b''.join(bytes([i])*8 for i in [2, 3, 4, 5, 6, 7])
        result = overlap_report(a, b, plane_bytes=8, planes=6)
        self.assertEqual(result['suffix_prefix_matching_lengths'], [4])
        self.assertEqual(result['observed_relative_bin_displacement'], 2)
        identical = overlap_report(a, a, plane_bytes=8, planes=6)
        self.assertTrue(identical['full_window_identity'])
        self.assertIsNone(identical['observed_relative_bin_displacement'])
        repeated = overlap_report(bytes(48), bytes(48), plane_bytes=8, planes=6)
        self.assertTrue(repeated['multiple_alignments'])
        self.assertIsNone(repeated['observed_relative_bin_displacement'])
        self.assertEqual(overlap_report(a[:-1], b, plane_bytes=8, planes=6)['status'], 'INVALID_LENGTH')


if __name__ == '__main__':
    unittest.main()
