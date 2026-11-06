"""Tiny CPU/source identity checks for legacy baselines; no experiments."""
import os
from pathlib import Path
import unittest

import torch

from core.study_baselines import (BASE_ID, StudyBaseline, expand_polarity_stem,
                                  flatten_sensor_channels, legacy_flow_to_disparity)


ROOT = Path(__file__).resolve().parents[3]
CAPSULE = Path(os.environ.get('STUDY_BASELINE_CAPSULE', ROOT / 'storage/experiment-runs/snn-depth-estimation/spiking-transformer-stereo/source-alif-mechanism-screen-v1'))


class BaselineInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_time_major_polarity_kept_and_stem_gradients_distinct(self):
        x = torch.zeros(1, 50, 2, 1, 1)
        x[0, 0, 0] = 1
        x[0, 0, 1] = 2
        x[0, 1, 0] = 3
        flat = flatten_sensor_channels(x)
        self.assertEqual(flat[0, :3, 0, 0].tolist(), [1, 2, 3])
        original = torch.nn.Conv2d(50, 2, 1)
        rng = torch.get_rng_state().clone()
        expanded = expand_polarity_stem(original)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(torch.equal(expanded.weight[:, 0], expanded.weight[:, 1]))
        self.assertTrue(torch.allclose(expanded.weight[:, ::2] * (2 ** .5), original.weight, atol=1e-7, rtol=1e-6))
        expanded(flat).sum().backward()
        self.assertFalse(torch.equal(expanded.weight.grad[:, 0], expanded.weight.grad[:, 1]))
        with self.assertRaises(ValueError):
            flatten_sensor_channels(torch.zeros(1, 49, 2, 1, 1))

    def test_flow_coordinate_convention(self):
        left_x = torch.tensor([5.])
        right_x = torch.tensor([2.])
        self.assertEqual(legacy_flow_to_disparity(right_x - left_x).item(), 3)


@unittest.skipUnless((CAPSULE / 'source.tar.gz').is_file(), 'pinned capsule absent; set STUDY_BASELINE_CAPSULE')
class PinnedBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.original = StudyBaseline(CAPSULE, train_refinements=1, eval_refinements=1)

    def test_exact_identity_isolated_namespace_and_parameter_count(self):
        p = self.original.provenance()
        self.assertEqual(p['source_identity_sha256'], BASE_ID)
        self.assertEqual(sum(x.numel() for x in self.original.parameters()), 12148246)
        self.assertTrue(type(self.original.model).__module__.startswith('_study_baseline_'))
        self.assertTrue(all(len(v['source_sha256']) == len(v['transformed_source_sha256']) == 64 for v in p['module_sources'].values()))
        self.original.train()
        self.assertTrue(all(not x.training for x in self.original.modules() if isinstance(x, torch.nn.BatchNorm2d)))

    def test_native_wrapper_matches_original_with_sign_and_reset(self):
        from spikingjelly.activation_based import functional
        self.original.eval()
        torch.manual_seed(234)
        x, y = torch.randint(0, 2, (1, 50, 1, 64, 64)).float(), torch.randint(0, 2, (1, 50, 1, 64, 64)).float()
        with torch.no_grad():
            wrapped = self.original(x, y)
            repeated = self.original(x, y)
            functional.reset_net(self.original.model)
            _, direct, _ = self.original.model(x[:, :, 0], y[:, :, 0], iters=1, test_mode=True)
        self.assertEqual(wrapped['disparity'].shape, (1, 1, 64, 64))
        self.assertTrue(torch.isfinite(wrapped['disparity']).all())
        self.assertTrue(torch.equal(wrapped['disparity'], -direct))
        self.assertTrue(torch.equal(wrapped['disparity'], repeated['disparity']))

    def test_current_identity_and_event_extra_capacity(self):
        current = StudyBaseline(CAPSULE, architecture='original_I', input_channels=2, train_refinements=1, eval_refinements=1)
        extra_stem_parameters = 2 * 64 * 50 * 7 * 7
        self.assertEqual(current.event_added_parameters, extra_stem_parameters)
        self.assertEqual(sum(p.numel() for p in current.parameters()), 12427926 + extra_stem_parameters)
        self.assertEqual(current.model.fnet.conv1.in_channels, 100)
        self.assertEqual(current.model.cnet.conv1.in_channels, 100)
        self.assertTrue(current.provenance()['event_adaptation'])
        self.assertEqual(current.model.update_block.snn08.current_proj.in_channels, 768)
        self.assertEqual(current.model.update_block.snn32.current_proj.in_channels, 640)

    def test_pinned_I_event_physical2_forward_matches_independent_samples(self):
        model = StudyBaseline(CAPSULE, architecture='original_I', input_channels=2,
                              train_refinements=1, eval_refinements=1).eval()
        torch.manual_seed(34)
        left, right = [torch.randint(0, 3, (2, 50, 2, 64, 64)).float() for _ in range(2)]
        with torch.no_grad():
            batched = model(left, right)['disparity']
            split = torch.cat([model(left[i:i+1], right[i:i+1])['disparity'] for i in range(2)])
        self.assertTrue(torch.allclose(batched, split, atol=2e-4, rtol=2e-4))


if __name__ == '__main__':
    unittest.main()
