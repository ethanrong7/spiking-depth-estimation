"""Tiny synthetic sensor/serialization checks; no EMA, dataset or GPU run."""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

SCRIPTS = Path(__file__).resolve().parents[1]/'experiments/stereo-spiking-transformer-study/scripts'
sys.path.insert(0, str(SCRIPTS))
import prepare_train_fidelity as driver


class LinearInteriors:
    def interiors(self, first, last):
        first = torch.from_numpy(first.copy()).float()/255
        last = torch.from_numpy(last.copy()).float()/255
        for step in range(1, 50):
            yield first+(last-first)*(step/50), dict(overshoot_values=0, min_before_clamp=0., max_before_clamp=1.)


class FidelityDriverTests(unittest.TestCase):
    def frames(self):
        return [np.full((4, 4, 3), 180, dtype=np.uint8) for _ in range(10)]

    def config(self):
        return json.loads((SCRIPTS.parent/'configs/train-fidelity-native-v1.example.json').read_text())

    def test_full450_history_and_causal_saved_targets_ignore_later_endpoint(self):
        frames = self.frames()
        first, rows, scalar = driver.simulate_view(frames, LinearInteriors(), driver.SensorConfig(), 'cpu')
        changed = self.frames()
        changed[9][:] = 250
        second, later, other = driver.simulate_view(changed, LinearInteriors(), driver.SensorConfig(), 'cpu')
        self.assertEqual([r['global_index'] for r in rows], list(range(1, 451)))
        self.assertEqual(set(first), {1, 4, 8})
        for center in first:
            np.testing.assert_array_equal(first[center]['spike'], second[center]['spike'])
            np.testing.assert_array_equal(first[center]['event'], second[center]['event'])
        np.testing.assert_array_equal(scalar['gray'][:400], other['gray'][:400])
        self.assertFalse(np.array_equal(scalar['gray'][400:], other['gray'][400:]))
        self.assertTrue(driver.compare_scalar_trace(scalar, driver.SensorConfig())['exact_counts_and_charge_agree'])

    def test_scalar_reference_preserves_multiple_crossings_and_reports_mismatch(self):
        frames = self.frames()
        frames[0][:] = 0
        for frame in frames[1:]:
            frame[:] = 255
        _, _, scalar = driver.simulate_view(frames, LinearInteriors(), driver.SensorConfig(), 'cpu')
        self.assertGreater(int(scalar['event'].max()), 1)
        self.assertTrue(driver.compare_scalar_trace(scalar, driver.SensorConfig())['exact_counts_and_charge_agree'])
        scalar['spike'][0, 0] = 1-scalar['spike'][0, 0]
        scalar['event'][0, 0, 0] += 1
        result = driver.compare_scalar_trace(scalar, driver.SensorConfig())
        self.assertEqual(result['spike_mismatches'], 1)
        self.assertEqual(result['event_mismatches'], 1)

    def test_registration_native_resolution_and_sensor_parameters_are_explicit(self):
        config = self.config()
        driver.validate_config(config, require_registered=False)
        with self.assertRaisesRegex(ValueError, 'registered=true'):
            driver.validate_config(config)
        for field, value in [('output_scale', .25), ('max_seconds', 1800)]:
            changed = copy.deepcopy(config)
            changed[field] = value
            with self.assertRaises(ValueError):
                driver.validate_config(changed, require_registered=False)
        config['sensor']['T'] = 49
        with self.assertRaises(ValueError):
            driver.validate_config(config, require_registered=False)

    def test_extra_interpolated_sample_is_rejected(self):
        class Extra(LinearInteriors):
            def interiors(self, first, last):
                yield from super().interiors(first, last)
                yield torch.from_numpy(last.copy()).float()/255, dict(overshoot_values=0)
        with self.assertRaisesRegex(ValueError, 'extra temporal'):
            driver.simulate_view(self.frames(), Extra(), driver.SensorConfig(), 'cpu')

    def test_saved_target_readback_uses_persisted_counts_and_gt(self):
        targets, rows, _ = driver.simulate_view(self.frames(), LinearInteriors(), driver.SensorConfig(), 'cpu')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for modality in ('spike', 'event'):
                np.savez_compressed(root/('0007-'+modality+'.npz'), left=targets[1][modality], right=targets[1][modality])
            np.save(root/'0007-disparity.npy', np.ones((4, 4), dtype=np.float32))
            result = driver.saved_target_check(root, '0007', (4, 4), rows[:50], rows[:50])
            self.assertEqual(result['support']['valid_fraction'], 1.)
            bad = targets[1]['spike'].copy()
            bad[0, 0, 0, 0] = 1-bad[0, 0, 0, 0]
            np.savez_compressed(root/'0007-spike.npz', left=bad, right=bad)
            with self.assertRaises(AssertionError):
                driver.saved_target_check(root, '0007', (4, 4), rows[:50], rows[:50])

    def test_generation_rejects_mac_context_before_cuda_or_source_array_access(self):
        args = argparse.Namespace(source_registry=None, output=None)
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(torch.cuda, 'is_available') as cuda:
            with self.assertRaisesRegex(ValueError, 'Slurm allocation'):
                driver.generation_context(args, self.config(), {})
        cuda.assert_not_called()


if __name__ == '__main__':
    unittest.main()
