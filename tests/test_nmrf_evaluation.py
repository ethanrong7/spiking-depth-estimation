import unittest
import tempfile
from pathlib import Path
import torch
from evaluate_nmrf import metric_counts, summarize, select_train, input_tensors, evaluate


class EvaluationTests(unittest.TestCase):
    def test_threshold_mask_and_pixel_weighting(self):
        p = torch.tensor([0., 5., 7., 100., 100.])
        t = torch.tensor([1., 2., 3., 4., float('nan')])
        counts = metric_counts(p, t, torch.tensor([1, 1, 1, 0, 1], dtype=torch.bool))
        self.assertEqual(counts, dict(pixels=3, absolute_error=8., bad1_pixels=2, bad3_pixels=1))
        other = metric_counts(torch.tensor([2.]), torch.tensor([2.]), torch.tensor([True]))
        self.assertEqual(summarize([counts, other])['epe'], 2.)
        self.assertEqual(summarize([counts, other])['bad3_percent'], 25.)
        self.assertIsNone(summarize([])['epe'])
        with self.assertRaises(FloatingPointError):
            metric_counts(torch.tensor([float('nan')]), torch.tensor([1.]), torch.tensor([True]))

    def test_scene_selection_does_not_weight_many_frame_scenes_more(self):
        rows = [dict(scene='a', frame=1), dict(scene='a', frame=2), dict(scene='b', frame=3)]
        self.assertEqual(select_train(rows), [rows[0], rows[2]])

    def test_crop_alignment_and_native_shape(self):
        left = torch.arange(50*10*18).reshape(50, 10, 18).float()
        flow = -torch.arange(180).reshape(1, 10, 18).float()
        sample = (left, left+1, flow, torch.ones(10, 18, dtype=torch.bool), 0)
        l, r, t, v = input_tensors(sample, 'cpu', [8, 16])
        torch.testing.assert_close(l[0], left[:, 1:9, 1:17])
        torch.testing.assert_close(r-l, torch.ones_like(l))
        torch.testing.assert_close(t, -flow[0, 1:9, 1:17])
        self.assertEqual(input_tensors(sample, 'cpu', None)[0].shape, (1, 50, 10, 18))

    def test_native_center_comparison_scores_identical_crop_pixels(self):
        left = torch.zeros(50, 10, 18)
        sample = (left, left, -torch.ones(1, 10, 18), torch.ones(10, 18, dtype=torch.bool), 0)
        prediction = torch.zeros(1, 10, 18)
        prediction[:, 1:9, 1:17] = 3
        def model(left, right):
            return {'disp': prediction}
        with tempfile.TemporaryDirectory() as directory:
            result = evaluate(model, [sample], [dict(scene='a', frame='1')], 'cpu', None,
                              Path(directory)/'eval', 192, center_reference=[8, 16])
        self.assertEqual(result['metrics']['training_range_center']['pixels'], 128)
        self.assertEqual(result['metrics']['training_range_center']['epe'], 2.)
        self.assertAlmostEqual(result['metrics']['training_range']['epe'], 308/180)


if __name__ == '__main__':
    unittest.main()
