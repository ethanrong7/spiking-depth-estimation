"""Tiny CPU diagnostic wiring checks; no dataset or neural experiment."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

SCRIPT = Path(__file__).resolve().parents[1]/'experiments/stereo-spiking-transformer-study/scripts/evaluate_functionality.py'
spec = importlib.util.spec_from_file_location('evaluate_functionality', SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def config():
    return dict(schema='stereo-study-functionality-v1', registered=True, purpose='engineering', mode='priors',
        expected_hostname='cluster', seed=9471, max_seconds=120,
        data=dict(root_env='ROOT', manifest_env='MANIFEST', audit_receipt_env='AUDIT',
            manifest_sha256='a'*64, audit_receipt_sha256='b'*64,
            train_samples=320, val_samples=20, train_scenes=40, val_scenes=10))


class FunctionalityTests(unittest.TestCase):
    def test_registered_modes_caps_and_fixed_cohort(self):
        value = config(); self.assertTrue(runner.validate_config(value))
        for key, changed in [('max_seconds', 121), ('purpose', 'final')]:
            bad = copy.deepcopy(value); bad[key] = changed
            with self.assertRaises(ValueError): runner.validate_config(bad)
        bad = copy.deepcopy(value); bad['checkpoint'] = {}
        with self.assertRaises(ValueError): runner.validate_config(bad)
        bad = copy.deepcopy(value); bad['data']['train_samples'] = 1
        with self.assertRaises(ValueError): runner.validate_config(bad)
        bad = copy.deepcopy(value); bad['registered'] = False
        with self.assertRaises(ValueError): runner.validate_config(bad)
        self.assertTrue(runner.validate_config(bad, require_registered=False))

    def test_gt_only_prior_fit_ignores_val_and_never_opens_observations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); output = root/'output'; output.mkdir()
            manifest = dict(schema='stereo-paired-engineering-pilot-v1', evidence_scope='engineering_only',
                            test=[], output_shape=[1, 2], train=[], val=[])
            for split, values in [('train', [2., 6.]), ('val', [100., 200.])]:
                for index, value in enumerate(values):
                    identifier = split+str(index)
                    np.save(root/(identifier+'.npy'), np.array([[value, value]], dtype=np.float32))
                    manifest[split].append(dict(sample_id=identifier, scene_id=identifier, split=split,
                        disparity=identifier+'.npy', spike='FORBIDDEN-spike.npz', event='FORBIDDEN-event.npz'))
            report = dict(binding={'fixture': True}, preselected_val_case_ids=['val0', 'val1'])
            opened = []
            original = np.load
            def gt_only(path, **kwargs):
                opened.append(Path(path).suffix)
                self.assertEqual(Path(path).suffix, '.npy')
                return original(path, **kwargs)
            with mock.patch.object(np, 'load', side_effect=gt_only):
                runner.run_priors(config(), root, manifest, output, report)
            self.assertEqual(report['constant_median'], 4.)
            self.assertEqual(report['prior_fit']['samples'], 2)
            self.assertFalse(report['prior_fit']['fit_includes_val'])
            self.assertFalse(report['model_loaded'])
            self.assertEqual(opened, ['.npy']*4)
            self.assertEqual(len(report['frame_metrics']), 6)
            self.assertEqual(len(report['scene_metrics']), 6)
            constant = [r for r in report['scene_equal_summary'] if r['method']=='train_constant_median'][0]
            zero = [r for r in report['scene_equal_summary'] if r['method']=='zero_disparity'][0]
            self.assertEqual(constant['mean_scene_epe'], 146.)
            self.assertEqual(zero['mean_scene_epe'], 150.)
            artifact = torch.load(output/'train-disparity-priors.pt', weights_only=False)
            torch.testing.assert_close(artifact['pixel_median'], torch.full((1, 1, 2), 4.))
            with original(output/'selected-case-0.npz', allow_pickle=False) as saved:
                self.assertEqual(set(saved.files), {'constant','pixel','zero','target','valid'})

    def test_case_selection_and_equal_scene_summary(self):
        manifest = {'val': [{'sample_id': str(i)} for i in (5,3,2,4,1)]}
        self.assertEqual(runner.selected_cases(manifest), ['1','3','5'])
        rows = [dict(method='x', condition='correct', epe=epe, bad1=0., bad3=0., bad5=0., coverage=1.)
                for epe in (1.,9.)]
        self.assertEqual(runner.scene_equal_summary(rows)[0]['mean_scene_epe'],5.)
        manifest['val'].reverse()
        self.assertEqual(runner.selected_cases(manifest), ['1','3','5'])

    def test_metadata_only_or_incomplete_audit_cannot_unlock_gt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); value=config()
            manifest=dict(schema='stereo-paired-engineering-pilot-v1', evidence_scope='engineering_only',
                          test=[], output_shape=[135,240], configuration_sha256='c'*64, train=[], val=[])
            for split, scenes, frames in [('train',40,8),('val',10,2)]:
                manifest[split]=[dict(sample_id=f'{split}/{scene}/{frame}',scene_id=f'{split}/{scene}',split=split)
                                 for scene in range(scenes) for frame in range(frames)]
            path=root/'manifest.json'; path.write_text(json.dumps(manifest))
            value['data']['manifest_sha256']=runner.file_sha(path)
            audit=dict(status='PASS_ALL_REGISTERED_ASSETS_TRACES_AND_SOURCE_IDENTITIES',mode='all_registered_assets',
                test_access=False,manifest_sha256=value['data']['manifest_sha256'],samples_checked=340,
                source_PFM_equal_samples=340,source_payloads_hashed=1340,source_files_registered=1340,T=50,
                counts={'train':dict(samples=320,scenes=40),'val':dict(samples=20,scenes=10)},
                shape=[135,240],configuration_sha256='c'*64)
            audit_path=root/'audit.json'
            env={'ROOT':str(root),'MANIFEST':str(path),'AUDIT':str(audit_path)}
            for status, count, should_pass in [(audit['status'],340,True),
                    ('PASS_METADATA_ONLY_ARRAYS_NOT_AUDITED',0,False),(audit['status'],339,False)]:
                audit.update(status=status,samples_checked=count)
                audit_path.write_text(json.dumps(audit))
                value['data']['audit_receipt_sha256']=runner.file_sha(audit_path)
                with mock.patch.dict(runner.os.environ,env), mock.patch.object(runner,'validate_paired_manifest'):
                    if should_pass:
                        self.assertEqual(runner.load_bound_data(value)[0],root.resolve())
                    else:
                        with self.assertRaises(ValueError):runner.load_bound_data(value)

    def test_checkpoint_conditions_keep_all_gt_and_separate_shift_geometry(self):
        manifest=dict(schema='stereo-paired-engineering-pilot-v1',evidence_scope='engineering_only',
                      test=[],output_scale=.25,train=[dict(sample_id='t',scene_id='t',split='train')],val=[])
        samples=[]
        for index in range(2):
            row=dict(sample_id='v'+str(index),scene_id='v'+str(index),split='val',disparity='fixture.npy')
            manifest['val'].append(row)
            samples.append(dict(sample_id=row['sample_id'],scene_id=row['scene_id'],metadata=row,
                left=torch.full((50,1,1,16),float(index+1)),right=torch.full((50,1,1,16),float(index+1)),
                target=torch.full((1,1,16),4.),valid=torch.ones((1,1,16),dtype=torch.bool)))
        class Dataset:
            rows=manifest['val']
            def __len__(self):return len(samples)
            def __getitem__(self,index):return samples[index]
        class ToyGeometry:
            def __call__(self,left,right):
                if not bool(right.any()):value=-1. # Raw invalid-direction predictions remain scored.
                elif right.flatten()[0]==0:value=2. # Content +2 => disparity4-2.
                elif left.flatten()[0]!=right.flatten()[0]:value=7.
                else:value=4.
                return {'disparity':torch.full((1,1,1,16),value)}
        value=dict(seed=9471,shift_pixels=2)
        report=dict(preselected_val_case_ids=['v0','v1'])
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(runner,'load_checkpoint',return_value=(ToyGeometry(),{'data':{'modality':'spike'}},{})), \
                    mock.patch('core.study_data.PairedStereoDataset',return_value=Dataset()) as loader:
                runner.run_checkpoint(value,Path(tmp),Path(tmp)/'manifest.json',manifest,Path(tmp),report,torch.device('cpu'))
            loader.assert_called_once_with(Path(tmp),Path(tmp)/'manifest.json','spike','val',purpose='engineering')
        self.assertEqual({r['condition'] for r in report['frame_metrics']},{'correct','shuffled','masked'})
        self.assertTrue(all(r['valid_pixels']==16 for r in report['frame_metrics']))
        masked=[r for r in report['frame_metrics'] if r['condition']=='masked']
        self.assertTrue(all(r['epe']==5. and r['nonpositive_prediction_pixels']==16 for r in masked))
        shifted=[r for r in report['shifted_geometry_frame_metrics'] if r['condition']=='shifted_geometry']
        self.assertTrue(all(r['epe']==0. and r['coverage']==.75 for r in shifted))
        self.assertEqual(report['shift_scene_response_summary']['median_scene_mean_response_ratio'],1.)

    def test_checkpoint_selection_checks_prefix_final_history_and_excludes_initial(self):
        training=dict(total_updates=1000,training=dict(evaluation_updates=[250,500,1000],evaluate_initial=True))
        rows=[dict(update=u,mean_scene_epe=e) for u,e in [(0,.01),(250,4.),(500,5.),(1000,6.)]]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'validation.jsonl'
            def write(items):
                path.write_text(''.join(json.dumps(row)+'\n' for row in items))
                return dict(bytes=path.stat().st_size,sha256=runner.file_sha(path))
            prefix=write(rows[:2]); final=write(rows)
            state=dict(update=250,best_update=250,best_val_epe=4.)
            result=runner.verify_checkpoint_validation_selection(path,prefix,state,training,'best-val.pt',final)
            self.assertEqual(result['best_update'],250)
            self.assertTrue(result['initial_update_excluded_from_selection'])
            # Deliberate endpoint comparison may use latest without calling it best.
            latest=dict(update=1000,best_update=250,best_val_epe=4.)
            result=runner.verify_checkpoint_validation_selection(path,final,latest,training,'latest.pt',final)
            self.assertEqual(result['checkpoint_update'],1000)
            with self.assertRaises(ValueError):
                runner.verify_checkpoint_validation_selection(path,prefix,state,training,'latest.pt',final)
            # Previously best-at250 is no longer eligible after a better final evaluation.
            improved=copy.deepcopy(rows);improved[-1]['mean_scene_epe']=3.
            new_final=write(improved)
            with self.assertRaises(ValueError):
                runner.verify_checkpoint_validation_selection(path,prefix,state,training,'best-val.pt',new_final)
            # A registered full receipt cannot be used as a prefix while ignoring an appended tail.
            write(rows);path.write_bytes(path.read_bytes()+b'\n')
            with self.assertRaises(ValueError):
                runner.verify_checkpoint_validation_selection(path,prefix,state,training,'best-val.pt',final)

    def test_checkpoint_selection_rejects_schedule_and_prefix_corruption(self):
        training=dict(total_updates=1000,training=dict(evaluation_updates=[250,500,1000],evaluate_initial=False))
        rows=[dict(update=u,mean_scene_epe=e) for u,e in [(250,5.),(500,4.),(1000,4.)]]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'validation.jsonl'
            def write(items):
                path.write_text(''.join(json.dumps(row)+'\n' for row in items))
                return dict(bytes=path.stat().st_size,sha256=runner.file_sha(path))
            prefix=write(rows[:2]);final=write(rows)
            state=dict(update=500,best_update=500,best_val_epe=4.)
            runner.verify_checkpoint_validation_selection(path,prefix,state,training,'best-val.pt',final)
            # Equal later EPE does not replace the earlier best under trainer's strict< rule.
            late=dict(update=1000,best_update=1000,best_val_epe=4.)
            with self.assertRaises(ValueError):
                runner.verify_checkpoint_validation_selection(path,final,late,training,'best-val.pt',final)
            broken=dict(prefix,sha256='0'*64)
            with self.assertRaises(ValueError):
                runner.verify_checkpoint_validation_selection(path,broken,state,training,'best-val.pt',final)
            for invalid in (rows[:2],rows+[rows[-1]],list(reversed(rows)),[rows[0],dict(update=750,mean_scene_epe=3.),rows[-1]]):
                receipt=write(invalid)
                with self.assertRaises(ValueError):
                    runner.verify_checkpoint_validation_selection(path,receipt,late,training,'latest.pt',receipt)

    def test_checkpoint_config_requires_complete_final_validation_receipt(self):
        value=config();value.update(mode='checkpoint',max_seconds=600,shift_pixels=2)
        value['checkpoint']=dict(selection='best-val.pt',seed=9471,update=250,
            selection_validation_log_env='FROZEN_VAL_LOG',selection_validation_log_bytes=123,
            selection_validation_log_sha256='c'*64,
            **{name+'_env':name.upper() for name in ('path','metadata','training_config','training_source_registry')},
            **{name+'_sha256':'d'*64 for name in ('path','metadata','training_config','training_source_registry')})
        self.assertTrue(runner.validate_config(value))
        from feature_information import probe_contract
        descriptive=copy.deepcopy(value);descriptive['feature_information']=probe_contract()
        self.assertTrue(runner.validate_config(descriptive))
        descriptive['feature_information']['purpose']='checkpoint_selection'
        with self.assertRaises(ValueError):runner.validate_config(descriptive)
        for name in ('selection_validation_log_env','selection_validation_log_bytes','selection_validation_log_sha256'):
            broken=copy.deepcopy(value);del broken['checkpoint'][name]
            with self.assertRaises(ValueError):runner.validate_config(broken)

    def test_feature_integration_probes_only_fixed_three_correct_presentations(self):
        import feature_information
        manifest=dict(schema='stereo-paired-engineering-pilot-v1',evidence_scope='engineering_only',
            test=[],output_scale=.25,train=[dict(sample_id='t',scene_id='t',split='train')],val=[])
        samples=[]
        for index in range(20):
            identifier='v'+str(index).zfill(2)
            row=dict(sample_id=identifier,scene_id='scene'+str(index//2),split='val',disparity='fixture.npy')
            manifest['val'].append(row)
            samples.append(dict(sample_id=identifier,scene_id=row['scene_id'],metadata=row,
                left=torch.full((50,1,1,16),float(index+1)),right=torch.full((50,1,1,16),float(index+1)),
                target=torch.full((1,1,16),4.),valid=torch.ones((1,1,16),dtype=torch.bool)))
        class Dataset:
            rows=manifest['val']
            def __len__(self):return len(samples)
            def __getitem__(self,index):return samples[index]
        observed=[];active={'probe':False}
        class ProbeFixture:
            def __init__(self,model,input_shape):pass
            def __enter__(self):active['probe']=True;return self
            def __exit__(self,*args):active['probe']=False
            def serialize(self):return {'tiny_fixture':True}
        class Geometry:
            def __call__(self,left,right):
                if not bool(right.any()):condition='masked';value=-1.
                elif right.flatten()[0]==0:condition='shifted';value=2.
                elif left.flatten()[0]!=right.flatten()[0]:condition='shuffled';value=7.
                else:condition='correct';value=4.
                observed.append((int(left.flatten()[0])-1,condition,active['probe']))
                return dict(disparity=torch.full((1,1,1,16),value),aux_disparities_native={4:torch.full((1,1,4,4),value)},
                    match_confidence_native={stride:dict(max_probability=torch.ones((1,1,16//stride,16//stride)),
                        normalized_entropy=torch.zeros((1,1,16//stride,16//stride))) for stride in (4,8,16)})
        value=dict(seed=9471,shift_pixels=2,feature_information=feature_information.probe_contract())
        wanted=['v00','v10','v19'];report=dict(preselected_val_case_ids=wanted)
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(runner,'load_checkpoint',return_value=(Geometry(),{'data':{'modality':'spike'},'model':{'family':'study'}},{})), \
                 mock.patch('core.study_data.PairedStereoDataset',return_value=Dataset()), \
                 mock.patch.object(feature_information,'TemporalFeatureProbe',ProbeFixture):
                runner.run_checkpoint(value,Path(tmp),Path(tmp)/'manifest.json',manifest,Path(tmp),report,torch.device('cpu'))
        self.assertEqual(len(observed),80)
        self.assertEqual([(index,condition) for index,condition,on in observed if on],[(0,'correct'),(10,'correct'),(19,'correct')])
        feature=report['feature_information']
        self.assertEqual(feature['temporal_feature_cases_count'],3)
        self.assertEqual(feature['correct_forward_output_summaries_count'],20)
        self.assertEqual({row['sample_id'] for row in feature['temporal_feature_cases']},set(wanted))
        self.assertEqual(len(report['frame_metrics']),60)


if __name__ == '__main__': unittest.main()
