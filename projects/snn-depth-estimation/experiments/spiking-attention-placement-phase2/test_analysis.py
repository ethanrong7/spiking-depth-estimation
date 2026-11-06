"""CPU-only unit checks for diagnostic reductions on tiny fabricated fixtures.

These fixtures exercise statistics and are not experiment results.
"""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('summary_under_test', HERE/'summarize.py')
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def fixture():
    result = {}
    for seed in summary.SEEDS:
        for arm, offset in zip(summary.ARMS, (1., 0., .2, -.1, .1)):
            frames = []
            for scene in range(30):
                for frame in (0, 1):
                    metric = dict(epe=2+offset+scene/100, valid_pixels=2, bad1=10+offset, bad3=5+offset, bad5=1+offset)
                    ranks = dict(pixel_count=1, rank_sum=1, top1_count=1, argmax_error_cells_sum=0.)
                    frames.append(dict(scene=f'{scene:04}', frame=frame, target_support_sha256=f't{scene}/{frame}',
                                       fine_cohort_sha256=f'f{scene}/{frame}',
                                       iterations={str(i): {s: dict(metric) for s in ('full', 'inview', 'oov')} for i in (1, 2, 4, 8, 16, 32)},
                                       rank={name: dict(ranks) for name in (('native',) if arm == 'Original' else ('pre', 'post'))}))
            result[(seed, arm)] = dict(seed=seed, arm=arm, update=30000, samples_seen=240000, parity=dict(passed=True),
                checkpoint_sha256='fixture', source_identity='fixture', manifest_sha256='fixture', per_frame=frames)
    return result


class SummaryTest(unittest.TestCase):
    def test_complete_paired_summary(self):
        result = summary.summarize(fixture())
        sp = result['contrasts']['SP-SC']
        self.assertAlmostEqual(sp['mean_epe_delta'], .2)
        np.testing.assert_allclose(sp['paired_97_5_percent_interval'], [.2, .2])
        self.assertEqual(sp['decision'], 'reference_advantage')
        self.assertEqual(result['contrasts']['CC-SC']['decision'], 'replacement_advantage')
        self.assertAlmostEqual(result['contrasts']['interaction']['mean_epe_delta'], 0)
        self.assertEqual(sp['scene_losses'], 30)
        self.assertAlmostEqual(sp['bad_delta_pp']['bad3'], .2)
        self.assertEqual(len(result['rank_per_scene']['1234']['SC']), 30)

    def test_missing_seed_rejected(self):
        rows = fixture()
        del rows[(3456, 'CP')]
        with self.assertRaises(ValueError):
            summary.summarize(rows)

    def test_bad_support_and_parity_rejected(self):
        for field in ('target_support_sha256', 'fine_cohort_sha256'):
            rows = fixture()
            rows[(1234, 'SC')]['per_frame'][0][field] = 'wrong'
            with self.assertRaises(AssertionError):
                summary.summarize(rows)
        rows = fixture()
        rows[(1234, 'SC')]['parity']['passed'] = False
        with self.assertRaises(AssertionError):
            summary.summarize(rows)

    def test_bootstrap_and_decision(self):
        # Nonconstant effect ensures retained whole-scene variation is tested.
        rows = fixture()
        for seed in summary.SEEDS:
            for row in rows[(seed, 'SP')]['per_frame']:
                for strata in row['iterations'].values():
                    for metric in strata.values():
                        metric['epe'] += int(row['scene'])/100
        result = summary.summarize(rows)
        effects = .2 + np.arange(30)/100
        index = np.random.default_rng(20261104).integers(0, 30, (20000, 30))
        expected = np.percentile(effects[index].mean(1), [1.25, 98.75])
        np.testing.assert_allclose(result['contrasts']['SP-SC']['paired_97_5_percent_interval'], expected)
        self.assertEqual(summary.classify([-.03, .03], [0., 0., 0.]), 'two_sided_practical_equivalence')
        self.assertEqual(summary.classify([-.2, .04], [-.1, -.1, -.1]), 'replacement_substitutability')
        self.assertEqual(summary.classify([.1, .2], [.2, -.1, .2]), 'unresolved')


if __name__ == '__main__':
    unittest.main()
