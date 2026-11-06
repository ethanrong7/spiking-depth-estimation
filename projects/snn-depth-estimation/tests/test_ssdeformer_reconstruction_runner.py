"""Registration/metadata checks only. Does not read Synthetic source payloads."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


PROJECT = Path(__file__).resolve().parents[1]
EXPERIMENT = PROJECT / 'experiments/ssdeformer-spikformer-reconstruction'
SPEC = importlib.util.spec_from_file_location('ssde_reconstruction_runner', EXPERIMENT / 'scripts/run.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class RunnerMetadataTests(unittest.TestCase):
    def fixture(self):
        config = json.loads((EXPERIMENT / 'configs/softmax.json').read_text())
        manifest = PROJECT / 'experiments/spiking-transformer-stereo/configs/manifest.json'
        return config, manifest

    def test_existing_cohort_metadata_and_unregistered_execution_refusal(self):
        config, manifest = self.fixture()
        result = runner.validate_config(config, manifest)
        self.assertEqual(len(result['train']), 960)
        self.assertEqual(len(result['dev']), 60)
        with self.assertRaisesRegex(ValueError, 'Prospective horizon'):
            runner.validate_config(config, manifest, executing=True)

    def test_two_configs_change_only_selected_stack(self):
        a = json.loads((EXPERIMENT / 'configs/softmax.json').read_text())
        b = json.loads((EXPERIMENT / 'configs/spikformer.json').read_text())
        a['model']['variant'] = b['model']['variant']
        self.assertEqual(a, b)

    def test_test_split_horizon_geometry_and_manifest_redirects_refused(self):
        config, manifest = self.fixture()
        for mutate in [lambda c: c.update(validation_split='confirmation'),
                lambda c: c.update(source_shape=[32, 250, 400]),
                lambda c: c['model'].update(neural_steps=4),
                lambda c: c['model'].update(width=64),
                lambda c: c.update(manifest_sha256='0' * 64),
                lambda c: c.update(microbatch=0)]:
            altered = copy.deepcopy(config)
            mutate(altered)
            with self.assertRaises(ValueError):
                runner.validate_config(altered, manifest)

    def test_forbidden_asset_and_scene_leak_rejected_even_with_matching_metadata_digest(self):
        config, manifest = self.fixture()
        source = json.loads(manifest.read_text())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'manifest.json'
            for alter in [lambda d: d['dev'][0].update(spike_l='Test/forbidden.dat'),
                    lambda d: d['dev'][0].update(scene=d['train'][0]['scene'])]:
                changed = copy.deepcopy(source)
                alter(changed)
                path.write_text(json.dumps(changed))
                config['manifest_sha256'] = runner.sha(path)
                with self.assertRaises(ValueError):
                    runner.validate_config(config, path)


if __name__ == '__main__':
    unittest.main()
