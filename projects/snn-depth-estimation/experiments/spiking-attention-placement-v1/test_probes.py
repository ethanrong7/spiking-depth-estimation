"""Tiny synthetic CPU checks; no dataset replay or performance measurement."""
import copy
import unittest

import torch

from probes import (ITERATIONS, MANIFEST_SHA, assert_parity, compare_replays,
                    fine_geometry, frame_metrics, rank_scores, summarize)


class DiagnosticTests(unittest.TestCase):
    def test_padded_grid_signed_truth_and_real_candidates(self):
        # Native grid x in image coordinates: -2,2,6,10. Last site is padding.
        target = torch.full((1, 1, 5, 10), -4.)
        valid = torch.ones((1, 5, 10), dtype=torch.bool)
        g = fine_geometry(target, valid, [2, 4, 1, 2], (2, 4))
        self.assertEqual(g['candidate_mask'].tolist(), [False, True, True, False])
        self.assertEqual(int(g['support'].sum()), 2)
        self.assertEqual(int(g['inview'].sum()), 1)
        self.assertEqual(float(g['right_cells'][0, 1, 2]), 1.)
        scores = torch.tensor([[[[999., 1., 5., 999.]]]*4]*2).reshape(1, 2, 4, 4)
        result = rank_scores(scores, g)
        self.assertEqual(result['pixel_count'], 1)
        self.assertEqual(result['excluded_oov_count'], 1)
        self.assertEqual(result['mean_rank'], 2.)
        self.assertEqual(result['mean_argmax_error_cells'], 1.)

    def test_score_and_coordinate_ties(self):
        scores = torch.tensor([[[[1., 1., 0.]]]])
        geometry = dict(support=torch.ones((1, 1, 1), dtype=torch.bool),
                        inview=torch.ones((1, 1, 1), dtype=torch.bool),
                        right_cells=torch.tensor([[[.5]]]), candidate_mask=torch.ones(3, dtype=torch.bool))
        result = rank_scores(scores, geometry)
        self.assertEqual(result['mean_rank'], 1.)
        self.assertEqual(result['top1_fraction'], 1.)
        self.assertEqual(result['truth_score_tied_count'], 1)
        self.assertEqual(result['max_score_tied_count'], 1)
        self.assertEqual(result['nearest_coordinate_tied_count'], 1)
        self.assertEqual(result['mean_argmax_error_cells'], .5)

    def test_all_oov_is_undefined_rank(self):
        g = dict(support=torch.ones((1, 1, 1), dtype=torch.bool),
                 inview=torch.zeros((1, 1, 1), dtype=torch.bool),
                 right_cells=torch.tensor([[[-1.]]]), candidate_mask=torch.ones(2, dtype=torch.bool))
        result = rank_scores(torch.ones((1, 1, 1, 2)), g)
        self.assertIsNone(result['mean_rank'])
        self.assertEqual(result['excluded_oov_count'], 1)

    def test_full_mask_keeps_oov_and_empty_strata(self):
        target = torch.full((1, 1, 1, 4), -2.)
        valid = torch.ones((1, 1, 4), dtype=torch.bool)
        prediction = target + torch.tensor([[[[4., 2., 1., 1.]]]])
        metrics = frame_metrics(prediction, target, valid)[0]
        self.assertEqual(metrics['full']['valid_pixels'], 4)
        self.assertEqual(metrics['full']['epe'], 2.)
        self.assertEqual(metrics['oov']['valid_pixels'], 2)
        self.assertEqual(metrics['oov']['epe'], 3.)
        self.assertEqual(metrics['inview']['epe'], 1.)
        empty = frame_metrics(target, target, torch.zeros_like(valid))[0]
        self.assertIsNone(empty['full']['epe'])

    def rows(self):
        rows = []
        for scene, frame, epe, count in [('a', 0, 1., 1), ('a', 1, 3., 1), ('b', 0, 10., 9)]:
            metric = dict(valid_pixels=count, error_sum=epe*count, epe=epe)
            rows.append(dict(scene=scene, frame=frame, iterations={str(i): {k: dict(metric) for k in ('full', 'inview', 'oov')} for i in ITERATIONS}))
        return rows

    def test_official_aggregation_and_parity_fail_closed(self):
        rows = self.rows()
        summary = summarize(rows)['refinement']['32']['full']
        self.assertEqual(summary['mean_scene_epe'], 6.)
        self.assertAlmostEqual(summary['pooled_epe'], 94/11)
        saved = [dict(scene=r['scene'], frame=r['frame'], epe=r['iterations']['32']['full']['epe'], valid_pixels=r['iterations']['32']['full']['valid_pixels']) for r in rows]
        self.assertTrue(assert_parity(rows, saved)['passed'])
        wrong = copy.deepcopy(saved)
        wrong[0]['epe'] += .01
        with self.assertRaises(AssertionError):
            assert_parity(rows, wrong)
        wrong = copy.deepcopy(saved)
        wrong[0]['valid_pixels'] += 1
        with self.assertRaises(AssertionError):
            assert_parity(rows, wrong)
        with self.assertRaises(AssertionError):
            assert_parity(rows, saved+[saved[0]])

    def test_comparison_requires_identical_targets_and_grid(self):
        base = self.rows()[0]
        base.update(target_support_sha256='target', fine_cohort_sha256='grid')
        frames = [dict(copy.deepcopy(base), scene=f's{i//2}', frame=i%2) for i in range(60)]
        original = dict(arm='Original', update=7500, manifest_sha256=MANIFEST_SHA,
                        parity=dict(passed=True), per_frame=frames, summary=summarize(frames))
        candidate = copy.deepcopy(original)
        candidate['arm'] = 'A'
        comparison = compare_replays(candidate, original)
        self.assertEqual(comparison['refinement_vs_original']['32']['full']['mean_scene_epe_delta'], 0.)
        candidate['per_frame'][0]['fine_cohort_sha256'] = 'different'
        with self.assertRaises(AssertionError):
            compare_replays(candidate, original)
        candidate['arm'] = 'C'
        candidate['per_frame'][0]['target_support_sha256'] = 'different'
        with self.assertRaises(AssertionError):
            compare_replays(candidate, original)


if __name__ == '__main__':
    unittest.main()
