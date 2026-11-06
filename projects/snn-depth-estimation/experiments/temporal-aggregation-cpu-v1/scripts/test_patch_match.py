"""Tiny synthetic CPU contracts; distinct from the real-data experiment."""
import unittest
import numpy as np
from patch_match import chunk_rates, zncc, aggregate_scores, score_metrics


class PatchTest(unittest.TestCase):
    def test_positive_disparity_and_constant_patch_handling(self):
        rng = np.random.default_rng(4)
        left = rng.normal(size=(17, 30))
        right = np.zeros_like(left)
        right[:, :-3] = left[:, 3:]
        queries = np.array([[5, 12], [9, 21]])
        disparities = np.arange(7)
        scores, zero = zncc(left, right, queries, disparities, radius=2)
        self.assertTrue(np.array_equal(disparities[scores.argmax(1)], [3, 3]))
        self.assertTrue(np.allclose(scores[:, 3], 1))
        score, zeros = zncc(np.ones_like(left), right, queries, disparities, radius=2)
        self.assertTrue(zeros.all())
        self.assertTrue((score == 0).all())
        with self.assertRaises(ValueError):
            zncc(left, right, np.array([[5, 3]]), disparities, radius=2)

    def test_chunk_aggregate_and_joint_permutation_invariance(self):
        rng = np.random.default_rng(7)
        left, right = [rng.integers(0, 2, (10, 13, 25)) for _ in range(2)]
        l, r = chunk_rates(left, 2), chunk_rates(right, 2)
        self.assertTrue(np.allclose(l.mean(0), left.mean(0)))
        permutation = np.array([2, 4, 1, 0, 3])
        self.assertTrue(np.array_equal(r.sum(0), r[permutation].sum(0)))
        queries, disparities = np.array([[5, 12], [8, 19]]), np.arange(5)
        a, _ = aggregate_scores(l, r, queries, disparities, 2)
        b, _ = aggregate_scores(l[permutation], r[permutation], queries, disparities, 2)
        self.assertTrue(np.allclose(a, b, atol=1e-15, rtol=0))

    def test_rank_ties_and_strict_bad_thresholds(self):
        result = score_metrics(np.array([[1., 1., 0.], [0., 1., 0.]]), np.array([0., 2.]), np.arange(3))
        self.assertTrue(np.array_equal(result['_predicted'], [0, 1]))
        self.assertEqual(result['bad1'], 0.)
        self.assertEqual(result['mean_gt_rank'], 1.5)
        self.assertEqual(result['argmax_tie_fraction'], .5)


if __name__ == '__main__':
    unittest.main()
