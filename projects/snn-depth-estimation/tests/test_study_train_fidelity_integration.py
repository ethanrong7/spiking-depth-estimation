"""Future Slurm preparation bindings; all remote work is mocked."""
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[1]/'experiments/stereo-spiking-transformer-study/scripts'
sys.path.insert(0, str(SCRIPTS))
import cluster_control as control


class FidelityIntegrationTests(unittest.TestCase):
    def fixture(self):
        config = json.loads((SCRIPTS.parent/'configs/train-fidelity-native-v1.example.json').read_text())
        config['registered'] = True
        cluster = dict(run_root='/runs', native_source_root='/native', environment='/env', ema_source='/ema', ema_checkpoint='/weights')
        registry = {'source_files':{'projects/snn-depth-estimation/experiments/flyingthings3d-native-v1/configs/protocol.json':config['legacy_protocol_sha256']}}
        records = dict(panel={'sha256':config['source_panel_sha256'], 'value':dict(test=[], test_values_opened=False,
            sequences=[{'source_id':sid} for sid in config['source_ids']],
            source_roots={'historical_native':'/native', 'new_train_attempt002':'/runs/stereo-spiking-transformer-study/new-train-source-acquisition-v1/attempt-002/source-data'})},
            readback={'sha256':config['source_readback_sha256'], 'value':dict(panel_sha256=config['source_panel_sha256'],
                status='PASS_FIXED92_SOURCE_BYTE_READBACK', files=92, source_arrays_decoded=False, Test_values_opened=False)})
        return cluster, config, registry, records

    def test_exact_bindings_derive_four_fixed_environment_aliases(self):
        cluster, config, registry, records = self.fixture()
        remote = mock.Mock(return_value=json.dumps(records))
        env, receipt = control.train_fidelity_env(cluster, config, registry, remote_fn=remote)
        self.assertEqual(len(env), 4)
        self.assertEqual(env['STUDY_FIDELITY_PANEL'], '/runs/stereo-spiking-transformer-study/train-fidelity-source-readback-v1/source-panel.json')
        self.assertEqual(env['STUDY_EMA_CHECKPOINT'], '/weights')
        self.assertEqual(receipt['sha256'], config['source_readback_sha256'])
        self.assertEqual(remote.call_count, 1)

    def test_unregistered_and_redirected_source_bindings_are_rejected(self):
        cluster, config, registry, records = self.fixture()
        config['registered'] = False
        remote = mock.Mock()
        with self.assertRaisesRegex(ValueError, 'registered=true'):
            control.train_fidelity_env(cluster, config, registry, remote_fn=remote)
        remote.assert_not_called()
        config['registered'] = True
        for change in [lambda r:r['panel'].update(sha256='0'*64),
                lambda r:r['readback']['value'].update(status='STARTED'),
                lambda r:r['readback']['value'].update(files=91),
                lambda r:r['panel']['value']['source_roots'].update(historical_native='/other'),
                lambda r:r['panel']['value'].update(test=[{'sample_id':'TEST/example'}])]:
            altered = copy.deepcopy(records)
            change(altered)
            with self.subTest(change=change), self.assertRaises(ValueError):
                control.train_fidelity_env(cluster, config, registry, remote_fn=mock.Mock(return_value=json.dumps(altered)))


if __name__ == '__main__':
    unittest.main()
