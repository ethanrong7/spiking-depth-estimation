"""Tiny CPU functional tests of isolated frozen-checkpoint interventions."""
import sys
from pathlib import Path
import unittest
import torch
from torch import nn

sys.path[:0] = [str(Path(__file__).resolve().parents[2]), str(Path(__file__).resolve().parent)]
from core.spiking_attention_placement import CostFilter, DescriptorAttention
from checkpoint_ablation import intervention, official_parity, pair, continuous_codes


class AblationTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)

    def model(self, arm):
        model = nn.Module()
        model.placement = (CostFilter() if arm == 'A' else DescriptorAttention()).eval()
        with torch.no_grad():
            model.placement.output.weight.normal_(std=.1)
            model.placement.output.bias.normal_(std=.1)
        return model

    def test_baseline_off_tanh_and_restoration(self):
        for arm in ('A', 'B'):
            model = self.model(arm)
            args = (torch.randn(1, 1, 2, 1, 7),) if arm == 'A' else (torch.randn(1, 256, 1, 3), torch.randn(1, 256, 1, 3))
            before = {k: v.clone() for k, v in model.state_dict().items()}
            baseline = model.placement(*args)
            with intervention(model, arm, 'baseline'):
                actual = model.placement(*args)
            if arm == 'A':
                self.assertTrue(torch.equal(baseline, actual))
            else:
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(baseline, actual)))
            with intervention(model, arm, 'branch_off'):
                actual = model.placement(*args)
                if arm == 'A':
                    self.assertIs(actual, args[0])
                else:
                    self.assertIs(actual[0], args[0])
                    self.assertIs(actual[1], args[1])
            with intervention(model, arm, 'code_tanh'):
                x = torch.randn(1, 3, 16)
                z = model.placement.codes.norm(x)
                for output, layer in zip(model.placement.codes(x), (model.placement.codes.q, model.placement.codes.k, model.placement.codes.v)):
                    self.assertTrue(torch.equal(output, torch.tanh(layer(z))))
            with self.assertRaises(RuntimeError):
                with intervention(model, arm, 'code_tanh'):
                    raise RuntimeError('exercise restoration')
            self.assertNotIn('forward', model.placement.codes.__dict__)
            self.assertNotIn('compute', model.placement.__dict__)
            after = model.placement(*args)
            if arm == 'A':
                self.assertTrue(torch.equal(after, baseline))
            else:
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(after, baseline)))
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, before[key]))

    def test_uniform_cost_matches_valid_neighbor_mean(self):
        model = self.model('A')
        for width in (1, 2, 7):
            cost = torch.randn(1, 1, 1, 1, width)
            x = cost.reshape(1, width, 1)
            position = torch.linspace(-1, 1, width).view(1, width, 1)
            value = model.placement.codes(model.placement.input(torch.cat([x, position], -1)))[2]
            expected = torch.stack([value[:, max(0, i-2):min(width, i+3)].mean(1) for i in range(width)], 1)
            expected = (x+model.placement.output(expected)).reshape(cost.shape)
            with intervention(model, 'A', 'uniform_weights'):
                actual = model.placement(cost)
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    def test_uniform_binocular_uses_original_pair(self):
        model = self.model('B')
        left, right = torch.randn(1, 256, 1, 3), torch.randn(1, 256, 1, 3)
        l, r = left.permute(0, 2, 3, 1), right.permute(0, 2, 3, 1)
        vl = model.placement.codes(model.placement.input(l))[2]
        vr = model.placement.codes(model.placement.input(r))[2]
        expected_left = (l+model.placement.output(vr.mean(-2, keepdim=True).expand_as(vr))).permute(0, 3, 1, 2)
        expected_right = (r+model.placement.output(vl.mean(-2, keepdim=True).expand_as(vl))).permute(0, 3, 1, 2)
        with intervention(model, 'B', 'uniform_weights'):
            actual_left, actual_right = model.placement(left, right)
        torch.testing.assert_close(actual_left, expected_left)
        torch.testing.assert_close(actual_right, expected_right)

    def test_paired_metrics_and_official_parity(self):
        baseline = [dict(scene=f'{scene:04}', frame=str(frame), valid_pixels=10, target_support_sha256='fixture',
                         epe=1.+scene/100, bad1=20., bad3=10., bad5=5.) for scene in range(30) for frame in range(2)]
        candidate = [dict(row, epe=row['epe']+.2, bad3=row['bad3']+1) for row in baseline]
        result = pair(candidate, baseline)
        self.assertAlmostEqual(result['mean_scene']['epe'], .2)
        self.assertAlmostEqual(result['mean_scene']['bad3'], 1.)
        self.assertEqual(result['scene_losses'], 30)
        self.assertTrue(official_parity(baseline, dict(per_frame=baseline))['all_frame_metrics_exact'])
        with self.assertRaises(AssertionError):
            official_parity(candidate, dict(per_frame=baseline))
        wrong_support = [dict(row) for row in candidate]
        wrong_support[0]['target_support_sha256'] = 'other'
        with self.assertRaises(AssertionError):
            pair(wrong_support, baseline)
        self.assertEqual(official_parity(baseline, None)['status'], 'BASELINE_TIED_TO_CHECKPOINT')


if __name__ == '__main__':
    unittest.main()
