"""Synthetic CPU integration contracts; no native data/checkpoints or GPU."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from core.corr import CorrBlock1D
from core.utils.utils import InputPadder
from core.spiking_attention_phase2 import install, CELLS
import core.spikestereonet as native_model
from lookup_evaluator import (load_module, candidate_pyramid, native_neighbors, anchor_coords,
    moments, profile_records, reduce_records, LookupObserver, capture_predictions, official_parity, admit,
    error_decomposition_tau5)

probes = load_module('synthetic_lookup_probes', HERE.parent/'spiking-attention-placement-v1/probes.py')


class Features(torch.nn.Module):
    def forward(self, left, right):
        return F.avg_pool2d(left, 4), F.avg_pool2d(right, 4)


class EvaluationTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_exact_stride4_geometry_and_common_anchor(self):
        target = torch.full((1, 1, 8, 8), -2.)
        valid = torch.ones(1, 8, 8, dtype=torch.bool)
        padding = [4, 4, 4, 4]
        geometry = probes.fine_geometry(target, valid, padding, (4, 4), stride=4)
        expected = torch.zeros(1, 4, 4, dtype=torch.bool)
        expected[:, 1:3, 1:3] = True
        self.assertTrue(torch.equal(geometry['support'], expected))
        self.assertEqual(geometry['inview'].sum().item(), 2)
        self.assertTrue(torch.equal(geometry['candidate_mask'], torch.tensor([False, True, True, False])))
        c1, c2 = torch.zeros(1, 2, 4, 4), torch.ones(1, 2, 4, 4)*7
        for offset in (-2., 0., 2.):
            a, b = anchor_coords(c1, geometry, offset), anchor_coords(c2, geometry, offset)
            self.assertTrue(torch.equal(a[:, 0], b[:, 0]))
            self.assertTrue(torch.equal(a[:, 0], geometry['right_cells'] + offset))
        self.assertTrue(torch.equal(c1, torch.zeros_like(c1)))

    def test_native_pooling_and_interpolation_integer_neighbor(self):
        pyramid = candidate_pyramid(torch.tensor([False, True, True, True, True, False]), 2)
        self.assertTrue(torch.equal(pyramid[1], torch.tensor([False, True, False])))
        candidate = torch.tensor([False, True, False])
        self.assertTrue(torch.equal(native_neighbors(torch.tensor([1., 1.25, .75, -1.]), candidate),
                                    torch.tensor([True, False, False, False])))

    def test_sufficient_moments_and_equal_scene_reducer(self):
        m = moments(torch.tensor([2., 4.]), torch.tensor([True, True]))
        self.assertEqual((m['count'], m['sum'], m['sum_sq']), (2, 6., 20.))
        rows = []
        for scene, frame, values in [('s1', 'a', [1.]), ('s1', 'b', [3., 3., 3.]), ('s2', 'c', [10.]), ('s2', 'd', [])]:
            x = torch.tensor(values)
            rows.append(dict(scene=scene, frame=frame, prefix=32, iteration=1, regime='actual_trajectory',
                anchor_offset=None, level=0, left_stratum='inview', support_type='unpadded',
                statistics={'lookup': moments(x, torch.ones_like(x, dtype=torch.bool))}))
        result = reduce_records(rows)[0]['statistics']['lookup']
        self.assertEqual(result['per_scene'], {'s1': 2., 's2': 10.})
        self.assertEqual(result['mean_scene'], 6.)
        self.assertEqual(result['defined_frames'], 3)
        empty = dict(rows[0], statistics={'lookup': moments(torch.empty(0), torch.empty(0, dtype=torch.bool))})
        self.assertIsNone(reduce_records([empty])[0]['statistics']['lookup']['mean_scene'])

    def test_linear_finite_difference_and_independent_frame_scale(self):
        class Linear(CorrBlock1D):
            def __init__(self):
                self.radius, self.num_levels = 1, 1
                row = torch.arange(16).float()
                self.corr_pyramid = [torch.stack([row, row*10]).reshape(2, 1, 1, 16)]
        block = Linear()
        coords = torch.zeros(2, 2, 1, 1)
        coords[:, 0] = 6.25
        lower, upper = coords.clone(), coords.clone()
        lower[:, 0] -= .25
        upper[:, 0] += .25
        geometry = dict(support=torch.ones(2, 1, 1, dtype=torch.bool),
                        inview=torch.ones(2, 1, 1, dtype=torch.bool), candidate_mask=torch.ones(16, dtype=torch.bool))
        records = profile_records(block, coords, block(coords), lower, block(lower), upper, block(upper), geometry,
            [dict(scene='s', frame=str(i), prefix=32, iteration=1) for i in range(2)], 'actual_trajectory', None)
        rows = [r for r in records if r['left_stratum']=='inview' and r['support_type']=='unpadded']
        self.assertAlmostEqual(rows[0]['statistics']['slope']['mean'], 1., places=5)
        self.assertAlmostEqual(rows[1]['statistics']['slope']['mean'], 10., places=5)
        self.assertAlmostEqual(rows[0]['statistics']['normalized_slope']['mean'],
                               rows[1]['statistics']['normalized_slope']['mean'], places=5)
        self.assertLess(rows[0]['statistics']['finite_difference_curvature']['rms'], 1e-4)
        self.assertEqual(rows[0]['statistics']['stencil_knot_crossing']['mean'], 0.)

    def test_real_replay_route_predictions_prefixes_and_common_anchors(self):
        native = SimpleNamespace(CorrBlock1D=CorrBlock1D)
        model = SimpleNamespace(fnet=Features())
        def forward(model, left, right, iters):
            padder = InputPadder(left.shape, divis_by=32)
            l, r = model.fnet(*padder.pad(left, right))
            block = native.CorrBlock1D(l, r, num_levels=2, radius=1)
            coords = torch.zeros(l.shape[0], 2, *l.shape[-2:])
            coords[:, 0] = torch.arange(l.shape[-1]) + .25
            for _ in range(iters):
                evidence = block(coords)
                coords = coords + .01
            return padder.unpad(evidence[:, :1].repeat_interleave(4, -1).repeat_interleave(4, -2))
        driver = SimpleNamespace(forward=forward, reset_model=lambda model: None)
        torch.manual_seed(3)
        left, right = torch.randn(2, 1, 8, 8), torch.randn(2, 1, 8, 8)
        target, valid = torch.zeros(2, 1, 8, 8), torch.ones(2, 8, 8, dtype=torch.bool)
        baseline_predictions, actual_predictions = {}, {}
        with capture_predictions(driver, baseline_predictions):
            baseline = probes.replay_batch(model, driver, left, right, target, valid, 'Original')
        labels = [dict(scene='s', frame=str(i)) for i in range(2)]
        observer = LookupObserver(probes, target, valid, InputPadder(left.shape, divis_by=32)._pad, labels)
        with observer.installed(native), capture_predictions(driver, actual_predictions, observer):
            observed = probes.replay_batch(model, driver, left, right, target, valid, 'Original')
        self.assertIs(native.CorrBlock1D, CorrBlock1D)
        for prefix in probes.ITERATIONS:
            self.assertTrue(torch.equal(baseline_predictions[prefix], actual_predictions[prefix]))
        for b, label in enumerate(labels):
            baseline[b].update(label)
            observed[b].update(label)
        official = [dict(label, **baseline[i]['iterations']['32']['full']) for i, label in enumerate(labels)]
        self.assertTrue(official_parity(probes, baseline, official)['passed'])
        self.assertTrue(official_parity(probes, observed, official)['passed'])
        self.assertTrue(probes.summarize(observed)['refinement']['32']['full']['mean_scene_epe'] >= 0)
        trajectory = [r for r in observer.records if r['regime']=='actual_trajectory']
        self.assertEqual({r['prefix'] for r in trajectory}, {32})
        self.assertEqual({r['iteration'] for r in trajectory}, set(range(1, 33)))
        anchors = [r for r in observer.records if r['regime']=='gt_anchor']
        self.assertEqual({r['anchor_offset'] for r in anchors}, {-2., 0., 2.})
        self.assertEqual({r['iteration'] for r in anchors}, {None})
        self.assertEqual(len(observer.traces), 35)

    def test_lineage_rejects_wrong_seed(self):
        checkpoint = dict(config={'seed':2345}, updates=30000, samples_seen=240000)
        with self.assertRaises(AssertionError):
            admit(checkpoint, {}, None, 'SC', 'sha')

    def test_official_bad_metric_and_support_gate(self):
        full = dict(valid_pixels=10, epe=1., bad1=2., bad3=3., bad5=4.)
        row = dict(scene='s', frame='f', iterations={'32': {'full': full}})
        reference = dict(scene='s', frame='f', **full)
        with self.assertRaises(AssertionError):
            official_parity(probes, [row], [dict(reference, bad3=3.01)])
        with self.assertRaises(AssertionError):
            official_parity(probes, [row], [dict(reference, valid_pixels=9)])

    def test_installed_factorial_replay_hooks_and_observer_stack(self):
        original = native_model.CorrBlock1D
        try:
            torch.manual_seed(5)
            left, right = torch.randn(1, 1, 8, 8), torch.randn(1, 1, 8, 8)
            target, valid = torch.zeros(1, 1, 8, 8), torch.ones(1, 8, 8, dtype=torch.bool)
            def forward(model, left, right, iters):
                padder = InputPadder(left.shape, divis_by=32)
                l, r = model.fnet(*padder.pad(left, right))
                block = native_model.CorrBlock1D(l, r, num_levels=2, radius=1)
                coords = torch.zeros(1, 2, *l.shape[-2:])
                coords[:, 0] = torch.arange(l.shape[-1]) + .25
                for _ in range(iters):
                    evidence = block(coords)
                    coords = coords + .01
                return padder.unpad(evidence[:, :1].repeat_interleave(4, -1).repeat_interleave(4, -2))
            driver = SimpleNamespace(forward=forward, reset_model=lambda model: None)
            for cell in CELLS:
                with self.subTest(cell=cell), torch.no_grad():
                    model = install(SimpleNamespace(fnet=Features()), cell)
                    model.placement.eval()
                    model.placement.output.weight.fill_(.1)
                    model.placement.output.bias.fill_(.05)
                    before = {k: v.clone() for k, v in model.placement.state_dict().items()}
                    baseline_predictions, observed_predictions = {}, {}
                    with capture_predictions(driver, baseline_predictions):
                        baseline = probes.replay_batch(model, driver, left, right, target, valid, 'A')
                    observer = LookupObserver(probes, target, valid, InputPadder(left.shape, divis_by=32)._pad,
                        [dict(scene='s', frame='f', arm=cell)])
                    installed_class = native_model.CorrBlock1D
                    with observer.installed(native_model), capture_predictions(driver, observed_predictions, observer):
                        observed = probes.replay_batch(model, driver, left, right, target, valid, 'A')
                    self.assertIs(native_model.CorrBlock1D, installed_class)
                    for prefix in probes.ITERATIONS:
                        self.assertTrue(torch.equal(baseline_predictions[prefix], observed_predictions[prefix]))
                    self.assertEqual(set(observed[0]['rank']), {'pre', 'post'})
                    self.assertEqual(baseline, observed)
                    self.assertTrue(all(torch.equal(before[k], v) for k, v in model.placement.state_dict().items()))
                    self.assertEqual(len(observer.traces), 35)
                    self.assertTrue(observer.records)
        finally:
            native_model.CorrBlock1D = original

    def test_tau5_identity_exact_mask_counts_and_empty_oov(self):
        target = torch.tensor([[[[-1., 0., -1., 1.]]], [[[0., 0., 0., 0.]]]])
        errors = torch.tensor([[[[2., 5., 8., 30.]]], [[[0., 1., 10., 1000.]]]])
        prediction = target + errors
        valid = torch.tensor([[[True, True, True, False]], [[True, True, True, True]]])
        result = error_decomposition_tau5(probes, prediction, target, valid)
        reference = probes.frame_metrics(prediction, target, valid)
        for b in range(2):
            for name in ('full', 'inview', 'oov'):
                row = result[b][name]
                self.assertEqual(row['statistics']['absolute_error']['count'], reference[b][name]['valid_pixels'])
                self.assertEqual(row['identity_sum_residual'], 0.)
        full = result[0]['full']['statistics']
        self.assertEqual(full['absolute_error']['sum'], 15.)
        self.assertEqual(full['capped_error']['sum'], 12.)
        self.assertEqual(full['excess_error']['sum'], 3.)
        self.assertEqual(full['over5_fraction']['sum'], 1.)
        self.assertEqual(result[0]['oov']['statistics']['absolute_error']['sum'], 2.)
        self.assertIsNone(result[1]['oov']['statistics']['capped_error']['mean'])


if __name__ == '__main__':
    unittest.main()
