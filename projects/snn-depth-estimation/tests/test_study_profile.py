"""Config/control tests only: never execute the GPU runtime profiler locally."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest


STUDY = Path(__file__).resolve().parents[1] / 'experiments/stereo-spiking-transformer-study'
SPEC = importlib.util.spec_from_file_location('study_profile_control_test', STUDY / 'scripts/profile_study.py')
PROFILE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROFILE)


class ProfileControlTests(unittest.TestCase):
    def config(self):
        return json.loads((STUDY / 'configs/runtime-profile-v1.example.json').read_text())

    def test_complete_grid_examples_are_static_and_unregistered(self):
        config = self.config()
        PROFILE.validate_profile(config, require_registered=False)
        self.assertEqual(len(config['cases']), 20)
        with self.assertRaisesRegex(ValueError, 'unregistered'):
            PROFILE.validate_profile(config)
        config['registered'] = True
        for case in config['cases']:
            case['manifest_sha256'] = '0' * 64
        PROFILE.validate_profile(config)
        half_only = copy.deepcopy(config)
        half_only['cases'] = half_only['cases'][:10]
        PROFILE.validate_profile(half_only)
        for micro in (1, 2, 4, 8):
            half_only['training']['microbatch'] = micro
            PROFILE.validate_profile(half_only)
        for change in ('shape', 'time', 'budget', 'grid'):
            bad = copy.deepcopy(config)
            if change == 'shape':
                bad['cases'][0]['shape'] = [250, 400]
            elif change == 'time':
                bad['cases'][0]['model']['config']['time_steps'] = 49
            elif change == 'budget':
                bad['max_seconds'] = 3601
            else:
                bad['cases'].pop()
            with self.assertRaises(ValueError):
                PROFILE.validate_profile(bad)

    def test_single_sample_and_frozen_recipe_control(self):
        data = ['first', 'second']
        sample = PROFILE.OneSample(data, 1)
        self.assertEqual(len(sample), 1)
        self.assertEqual(sample[0], 'second')
        with self.assertRaises(IndexError):
            sample[1]
        config = self.config()
        recipe = PROFILE.training_config(config, config['cases'][0])
        self.assertEqual(recipe['total_updates'], 1000)
        self.assertEqual(recipe['training']['effective_batch'], 8)
        self.assertEqual(recipe['scope'], 'engineering')
        self.assertEqual(recipe['test_access'], 'sealed')

    def test_registered_prefetch_comparison_keeps_full_models_and_schedule(self):
        config = json.loads((STUDY / 'configs/prefetch-quarter-event-v1.json').read_text())
        PROFILE.validate_profile(config)
        self.assertEqual(config['updates'], 16)
        self.assertEqual(config['recipe_total_updates'], 1000)
        self.assertEqual(config['training']['microbatch'], 8)
        config['cases'][0]['training_override'] = {'prefetch': {'workers': 4}}
        with self.assertRaisesRegex(ValueError, 'modes are fixed'):
            PROFILE.validate_profile(config)

    def test_full_train_order_head_health_grid(self):
        config = json.loads((STUDY / 'configs/cross-family-head-health-quarter-v1.json').read_text())
        PROFILE.validate_profile(config)
        self.assertEqual(len(config['cases']), 10)
        self.assertEqual(config['max_seconds'], 600)
        self.assertEqual(config['scheduler_resources']['time'], '00:12:00')
        self.assertEqual(config['allocated_gpu_hour_cap'], 24)
        self.assertEqual(config['stage'], 'engineering_diagnostic')
        self.assertEqual(config['recipe_total_updates'], 1000)
        self.assertEqual({c['dataset_version'] for c in config['cases']}, {'causal-paired-quarter-pilot-v1'})
        self.assertEqual({c['manifest_sha256'] for c in config['cases']},
                         {'be80defb27c948b9766a6bb46381b56ce9dd1ee7062782f53dc6be2b518b99c5'})
        data = list(range(320))
        self.assertIs(PROFILE.profile_train_dataset(config, data), data)
        indices = PROFILE.sample_indices(9471, 0, 24, len(data))
        self.assertEqual(len(set(indices)), 24)
        self.assertEqual(indices[:8], PROFILE.sample_indices(9471, 0, 8, len(data)))
        self.assertEqual(indices[8:16], PROFILE.sample_indices(9471, 8, 8, len(data)))
        with self.assertRaisesRegex(ValueError, 'population'):
            PROFILE.profile_train_dataset(config, data[:2])
        study_case=next(c for c in config['cases'] if c['model']['family']=='study')
        baseline_case=next(c for c in config['cases'] if c['model']['family']=='baseline')
        self.assertEqual(len(PROFILE.training_config(config,study_case)['training']['health_parameter_names']),4)
        self.assertNotIn('health_parameter_names',PROFILE.training_config(config,baseline_case)['training'])
        for change in ('mode', 'grid', 'health', 'prefetch', 'targets', 'epsilon'):
            bad = copy.deepcopy(config)
            if change == 'mode':
                bad['sample_mode'] = 'one_train_sample'
            elif change == 'grid':
                bad['cases'].pop()
            elif change == 'health':
                bad['training']['readout_optimizer_health'] = False
            elif change == 'prefetch':
                bad['training']['prefetch']['workers'] = 0
            elif change == 'targets':
                bad['training']['health_parameter_names'] = ['extra.bias']
            else:
                bad['cases'][0]['model']['config']['feature_norm_eps'] = 1e-5
            with self.assertRaises(ValueError):
                PROFILE.validate_profile(bad)

    def test_compact_health_retains_gate_evidence_and_omits_full_arrays(self):
        target = {'role': 'final_disparity_kernel', 'available': True, 'step': 1,
                  'raw_gradient_energy': 9., 'raw_energy_weighted_denominator_attenuation': .8,
                  'adaptive_delta_excluding_ideal_decay': {'rms': .002, 'quantiles': [1]*50},
                  'adaptive_delta_excluding_rounded_decay': {'rms': .003},
                  'raw_energy_on_changed_stored_coordinates': .99}
        compact = PROFILE.compact_readout_health({'optimizer_moments_all_finite': True,
            'readout_optimizer_health': {'targets': {'head': target, 'early': {'role': 'early_feature_normalization_bias'},
                'coarse_q':{**target,'role':'qkv_projection_kernel'}}}})
        self.assertTrue(compact['optimizer_moments_all_finite'])
        self.assertEqual(list(compact['targets']), ['head','coarse_q'])
        row = compact['targets']['head']
        self.assertEqual(row['raw_energy_weighted_denominator_attenuation'], .8)
        self.assertEqual(row['adaptive_delta_rms_excluding_ideal_decay'], .002)
        self.assertEqual(row['raw_energy_on_changed_stored_coordinates'], .99)
        self.assertNotIn('quantiles', json.dumps(compact))


if __name__ == '__main__':
    unittest.main()
