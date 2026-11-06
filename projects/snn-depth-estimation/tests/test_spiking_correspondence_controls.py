"""P3 scalar binding and tiny operator checks; no real checkpoint/model arrays."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile
import unittest

import torch
from core.spiking_stereo_correspondence import SpikingStereoCorrespondence

PROJECT=Path(__file__).resolve().parents[1]
ROOT=PROJECT.parents[1]
STUDY=PROJECT/'experiments/spiking-stereo-correspondence'
spec=importlib.util.spec_from_file_location('controls_runner',STUDY/'scripts/run_experiment.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


class ControlsHardeningTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        archive=ROOT/'storage/experiment-runs/snn-depth-estimation/spiking-stereo-correspondence/source-ssc-main2500-v3/source.tar.gz'
        cls.old_source=None;cls.training_config=None
        if archive.exists():
            with tarfile.open(archive) as capsule:
                prefix='projects/snn-depth-estimation/experiments/spiking-stereo-correspondence/'
                cls.old_source=capsule.extractfile(prefix+'scripts/run_experiment.py').read()
                cls.training_config=json.loads(capsule.extractfile(prefix+'configs/main-seed1234-u2500-p2-v1.json').read())

    def checkpoint_fixture(self):
        if self.training_config is None:self.skipTest('immutable main2500 source fixture unavailable')
        return dict(updates=2500,presentations=20000,config=copy.deepcopy(self.training_config),
            manifest_sha256=runner.MANIFEST_SHA,source_registry_sha256=runner.FINAL2500_BINDING['training_source_registry_sha256'])

    def test_final_checkpoint_scalar_binding_and_configuration_registration(self):
        value=self.checkpoint_fixture()
        result=runner.validate_controls_checkpoint(value,runner.FINAL2500_BINDING,'last.pt')
        self.assertEqual(result['updates'],2500);self.assertEqual(result['presentations'],20000)
        self.assertEqual(result['training_source_identity_sha256'],'31ed4dcc25a0980c96b5df2b57fe090cbb229d378386f83e3b4badd618adcec3')
        config=json.loads((STUDY/'configs/controls-seed1234-p2-v1.json').read_text())
        self.assertEqual(config['expected_checkpoint'],runner.FINAL2500_BINDING)
        self.assertFalse(config['execution_review']['approved'])
        self.assertEqual(config['actual_inference_batch'],1);self.assertFalse(config['gradient_accumulation'])
        self.assertNotIn('checkpoint_sha256',config)  # Still unknown during running P2.

    def test_reject_partial_best_alternative_config_manifest_or_training_registry(self):
        original=self.checkpoint_fixture()
        for key,bad in [('updates',2499),('presentations',19992),('manifest_sha256','0'*64),('source_registry_sha256','1'*64)]:
            value=copy.deepcopy(original);value[key]=bad
            with self.subTest(key=key),self.assertRaises(ValueError):runner.validate_controls_checkpoint(value,runner.FINAL2500_BINDING,'last.pt')
        for key,bad in [('seed',9471),('run_id','other-main'),('updates',1000),('epsilon',1e-7)]:
            value=copy.deepcopy(original);value['config'][key]=bad
            with self.subTest(config=key),self.assertRaises(ValueError):runner.validate_controls_checkpoint(value,runner.FINAL2500_BINDING,'last.pt')
        with self.assertRaises(ValueError):runner.validate_controls_checkpoint(original,runner.FINAL2500_BINDING,'best.pt')
        altered=dict(runner.FINAL2500_BINDING,training_source_identity_sha256='2'*64)
        with self.assertRaises(ValueError):runner.validate_controls_checkpoint(original,altered,'last.pt')

    def probe_fixture(self):
        torch.manual_seed(91)
        model=SpikingStereoCorrespondence(time_steps=3,widths=(4,8,8),heads=2,max_width=24,activation_checkpoint=False).eval()
        with torch.no_grad():
            model.readout[-1].weight.normal_(0,.03)
            for module in model.modules():
                if isinstance(module,torch.nn.GroupNorm):module.bias.uniform_(-.2,.2)
            model.relative_bias.copy_(torch.linspace(-.2,.2,len(model.relative_bias)))
            left,right=torch.rand(1,3,9,17),torch.rand(1,3,9,17)
            outputs=dict(intact=model(left,right),shuffle_right=model(left,right.flip(-1)),
                shift_plus4=model(left,runner.shift_right_zero_fill(right)),
                zero_content_qk=model(left,right,zero_content=True),zero_cross_value_evidence=model(left,right,zero_values=True))
        return model,outputs

    def test_existing_five_forward_probe_invariants_nonzero_readout_and_norm_bias(self):
        model,outputs=self.probe_fixture()
        evidence=runner.probe_invariants(outputs,model.relative_bias,model.max_width)
        self.assertTrue(evidence['passed']);self.assertTrue(all(evidence['exact'].values()))
        self.assertTrue(all(row['passed'] for row in evidence['derived'].values()))
        self.assertEqual(evidence['derived']['content_minus_value_flow']['atol'],1e-4)
        self.assertEqual(evidence['derived']['content_minus_value_flow']['rtol'],1e-5)
        self.assertTrue((outputs['intact']['residual'].abs()>0).any())
        self.assertFalse(outputs['intact']['query_valid'].all())  # Exercises padding mask/prior convention.

    def test_probe_failure_retains_serializable_evidence(self):
        model,outputs=self.probe_fixture()
        outputs['zero_content_qk']['activity_per_time']=outputs['zero_content_qk']['activity_per_time']+.01
        outputs['zero_content_qk']['flow']=outputs['zero_content_qk']['flow']+.1
        with self.assertRaises(runner.ProbeInvariantError) as caught:runner.probe_invariants(outputs,model.relative_bias,model.max_width)
        evidence=caught.exception.evidence
        self.assertIn('content_vs_value:activity_per_time',evidence['failed_checks'])
        self.assertFalse(evidence['derived']['content_minus_value_flow']['passed'])
        self.assertGreater(evidence['derived']['content_minus_value_flow']['max_abs_error'],.01)
        json.dumps(evidence,allow_nan=False)

    def test_functional_band_requires_completed_ten_distinct_nonempty_scenes(self):
        def rows(count):
            return [dict(scene=f'scene_{i}',common_support_pixels=5,shift_response={'final':dict(mean=1.)},
                interventions={'intact':dict(full_gt={'final':dict(epe=2.)}),'shuffle_right':dict(full_gt={'final':dict(epe=3.)})}) for i in range(count)]
        records=rows(10)
        self.assertFalse(runner.summarize_panel(records)['functional_band']['assessable'])
        self.assertTrue(runner.summarize_panel(records,complete=True)['functional_band']['passed'])
        records[0]['common_support_pixels']=0
        self.assertFalse(runner.summarize_panel(records,complete=True)['functional_band']['assessable'])
        self.assertFalse(runner.summarize_panel(rows(9),complete=True)['functional_band']['assessable'])
        records=rows(10);records[-1]['scene']=records[0]['scene']
        self.assertFalse(runner.summarize_panel(records,complete=True)['registered_panel_complete'])

    def test_running_p2_existing_functions_unchanged_except_p3_controls_and_summary(self):
        if self.old_source is None:self.skipTest('immutable main2500 source fixture unavailable')
        old={node.name:ast.dump(node,include_attributes=False) for node in ast.parse(self.old_source).body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))}
        current={node.name:ast.dump(node,include_attributes=False) for node in ast.parse((STUDY/'scripts/run_experiment.py').read_text()).body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))}
        changed=[name for name in old if old[name]!=current.get(name)]
        self.assertEqual(set(changed),{'controls','summarize_panel'})
        for name in ('setup','training','preflight','evaluate','train_batch','load_components','validate_config'):
            self.assertEqual(old[name],current[name])
        self.assertEqual(hashlib.sha256((PROJECT/'core/spiking_stereo_correspondence.py').read_bytes()).hexdigest(),
            'bf47f29af1a13accffaae74487ca48258b4deaf608faface7986681239f14bee')


if __name__=='__main__':unittest.main()
