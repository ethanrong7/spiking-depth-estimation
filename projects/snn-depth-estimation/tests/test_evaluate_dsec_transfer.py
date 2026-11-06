"""Tiny CPU/mocked DSEC evaluator checks; no real arrays or model experiments."""
import copy
import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'experiments/stereo-spiking-transformer-study/scripts/evaluate_dsec_transfer.py'
spec=importlib.util.spec_from_file_location('evaluate_dsec_transfer_test',SCRIPT)
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


def example():
    return json.loads((SCRIPT.parents[1]/'configs/dsec-event-transfer-v1.example.json').read_text())


def fixture():
    config=example();prep=runner.adapter.example_config()
    prep.update(registered=True,max_seconds=600,preflight_manifest_sha256=config['dsec']['preflight_manifest_sha256'],
                adapter_script_sha256=config['dsec']['adapter_script_sha256'])
    targets=[]
    for name in config['dsec']['sample_ids']:
        scene,frame=name.split('/')
        targets.append(dict(sample_id=name,scene_id=scene,original_split='train',target_us=100000,
            native_gt={'member':frame+'.png'},bin_edges_us=list(range(50000,100001,1000))))
    preflight=dict(status='PREFLIGHT_READY_METADATA_ONLY',test_access=False,gt_pixels_opened=False,event_conversion=False,
        contract=runner.adapter.contract(prep),targets=targets,source_files=[dict(path='fixture',bytes=1,sha256='a'*64)])
    preflight['contract_sha256']=runner.digest(preflight['contract'])
    config['dsec']['targets_sha256']=runner.digest(targets)
    config['dsec']['source_files_sha256']=runner.digest(preflight['source_files'])
    prepared=dict(schema='dsec-causal-prepared-v1',status='COMPLETED_PROSPECTIVE_ENGINEERING_PREPARATION',
        protocol_status='prospective_engineering_only',test_access=False,gt_pixels_opened=False,model_inference=False,
        source_manifest_sha256=config['dsec']['preflight_manifest_sha256'],contract=preflight['contract'],
        source_files=preflight['source_files'],binding={
            'config_sha256':config['dsec']['preparation_config_sha256'],'script_sha256':config['dsec']['adapter_script_sha256'],
            'source_registry_sha256':config['dsec']['preparation_source_registry_sha256'],
            'source_identity_sha256':config['dsec']['preparation_source_identity_sha256']},
        conversion_qa=dict(prepared_targets=20,readback_targets=20,ready_for_inference=True,
            all_causal_lower_bound_neighbors_verified=True,all_stored_counts_readback_exact=True,
            source_fault_counts={v:dict(time_outside=0,bad_raw_coordinate=0,bad_polarity=0) for v in ('left','right')}),targets=[])
    for original in targets:
        row=copy.deepcopy(original)
        row.update(sensor_asset=original['sample_id'].replace('/','-')+'.npz',sensor_sha256='a'*64,
            sensor_readback_qa=dict(status='PASS_EXACT_COUNTS_AND_PER_BIN_READBACK',views=['left','right'],
                float32_count_cast_exact=True,ground_truth_opened=False),count_statistics={})
        for view in ('left','right'):
            row['count_statistics'][view]=dict(per_bin_ON_OFF_counts=[[0,0] for _ in range(50)],on_events=0,off_events=0,
                input_events=0,nonfinite_rectification=0,outside_rectified_support=0,
                accepted_events=0,time_outside=0,bad_raw_coordinate=0,bad_polarity=0,
                causal_boundary_neighbors=[dict(boundary_global_us=t,index=0,previous_global_us=t-1,
                    next_global_us=t,lower_bound_verified=True) for t in (50000,100000)])
        prepared['targets'].append(row)
    return config,preflight,prep,prepared


class DSECTransferTests(unittest.TestCase):
    def test_example_unregistered_placeholders_and_no_subset_or_test(self):
        value=example();self.assertTrue(runner.validate_config(value,require_registered=False))
        with self.assertRaises(ValueError):runner.validate_config(value)
        value['registered']=True
        with self.assertRaises(ValueError):runner.validate_config(value)
        for field,changed in [('models',list(runner.FAMILIES[:-1])),('inference_variant','diagnostics_false'),
                              ('max_seconds',601),('scenes',['official_test'])]:
            value=example();value[field]=changed
            with self.assertRaises(ValueError):runner.validate_config(value,require_registered=False)
        value=example();value['dsec']['sample_ids'][0]='test_scene/000002'
        with self.assertRaises(ValueError):runner.validate_config(value,require_registered=False)
        value=example();value['cases'][0]['checkpoint']['selection']='latest.pt'
        with self.assertRaises(ValueError):runner.validate_config(value,require_registered=False)

    def test_prepared_all20_exact_bindings_counts_readback_and_boundaries(self):
        config,preflight,prep,prepared=fixture()
        self.assertTrue(runner.validate_prepared_receipt(config,preflight,prep,prepared))
        for mutate in (
            lambda p:p['targets'].pop(),
            lambda p:p.update(test_access=True),
            lambda p:p['binding'].update(script_sha256='f'*64),
            lambda p:p['conversion_qa'].update(ready_for_inference=False),
            lambda p:p['targets'][0]['sensor_readback_qa'].update(float32_count_cast_exact=False),
            lambda p:p['targets'][0]['count_statistics']['left']['per_bin_ON_OFF_counts'][0].__setitem__(0,1),
            lambda p:p['targets'][0]['count_statistics']['right']['causal_boundary_neighbors'][0].update(previous_global_us=50000),
            lambda p:p['targets'][0].update(original_split='test')):
            bad=copy.deepcopy(prepared);mutate(bad)
            with self.assertRaises(ValueError):runner.validate_prepared_receipt(config,preflight,prep,bad)

    def test_native_mask_thresholds_nonpositive_predictions_and_finiteness(self):
        target=np.array([[[1,2,4,0,np.nan]]],np.float32)
        pred=np.array([[[0,0,1,99,99]]],np.float32)
        row=runner.native_metrics(pred,target)
        self.assertEqual(row['valid_pixels'],3);self.assertEqual(row['total_pixels'],5)
        self.assertEqual(row['epe'],2);self.assertEqual(row['mae'],2)
        self.assertAlmostEqual(row['rmse'],np.sqrt(14/3))
        self.assertEqual(row['one_pe'],2/3);self.assertEqual(row['two_pe'],1/3)
        self.assertEqual(row['coverage'],3/5);self.assertEqual(row['nonpositive_prediction_pixels'],2)
        self.assertEqual(runner.native_metrics(np.array([-2]),np.array([1]))['epe'],3)
        with self.assertRaises(ValueError):runner.native_metrics(np.array([np.nan]),np.array([1]))
        with self.assertRaises(ValueError):runner.native_metrics(np.array([0]),np.array([0]))

    def test_native_units_pixelcenter_bilinear_and_no_clamp(self):
        pred=torch.tensor([[[[-.5,.25],[1.5,2.25]]]])
        native=runner.adapter.predict_to_native(pred,native_shape=(8,8))
        expected=torch.nn.functional.interpolate(pred,size=(8,8),mode='bilinear',align_corners=False)*4
        torch.testing.assert_close(native,expected,rtol=0,atol=0)
        self.assertEqual(native[0,0,0,0].item(),-2)
        constant=runner.adapter.predict_to_native(torch.full((1,1,1,1),.375),native_shape=(4,4))
        torch.testing.assert_close(constant,torch.full((1,1,4,4),1.5))

    def test_pool_within_sequence_then_equal_two_sequences(self):
        rows=[]
        # Unequal target supports must pool within sequence; unequal sequence
        # supports must not alter the equal-sequence final summary.
        for scene in runner.adapter.SCENES:
            for i in range(10):
                target=np.ones(1 if i%2==0 else 3)
                pred=target+(0 if i%2==0 else 4) if scene==runner.adapter.SCENES[0] else target+1
                rows.append(dict(method='fixture',scene_id=scene,**runner.native_metrics(pred,target)))
        sequences,equal=runner.aggregate(rows)
        self.assertEqual(sequences[0]['epe'],3);self.assertEqual(sequences[1]['epe'],1)
        self.assertEqual(equal[0]['epe'],2)
        self.assertAlmostEqual(equal[0]['rmse'],(np.sqrt(12)+1)/2)
        with self.assertRaises(ValueError):runner.aggregate(rows[:-1])

    def test_family_gate_retains_widths_steps_polarity_and_baseline_K(self):
        for name,filename in [('A-event','pilot-A-event-seed9471-v1.json'),
                ('continuous-B-event','pilot-continuous-B-event-seed9471-v1.json'),
                ('ALIF+I-event','pilot-original-I-event-seed9471-v1.json')]:
            training=json.loads((SCRIPT.parents[1]/'configs'/filename).read_text())
            self.assertTrue(runner.validate_training_family(name,training))
            training['model']['config']['input_channels']=1
            with self.assertRaises(ValueError):runner.validate_training_family(name,training)

    def test_selection_failure_prevents_any_real_GT_access(self):
        config,_,_,prepared=fixture();report=dict(frame_metrics=[],models=[],gt_targets_opened=0,predictions_completed=0)
        with mock.patch.object(runner,'precheck_checkpoint',side_effect=[{},ValueError('final log incomplete')]), \
                mock.patch.object(runner.adapter,'load_native_gt') as gt:
            with self.assertRaisesRegex(ValueError,'final log incomplete'):
                runner.run_panel(config,{},prepared,{},Path('unused'),report,torch.device('cpu'))
            gt.assert_not_called()
        self.assertEqual(report['predictions_completed'],0)

    def test_executing_core_mismatch_prevents_GT_access(self):
        member=str((ROOT/'core/study_models.py').relative_to(SCRIPT.parents[5]))
        registry={'source_files':{member:runner.file_sha(ROOT/'core/study_models.py')}}
        self.assertEqual(runner.verify_executing_core(registry)['members_checked'],1)
        registry['source_files'][member]='0'*64
        config,_,_,prepared=fixture();report=dict(frame_metrics=[],models=[],gt_targets_opened=0,predictions_completed=0)
        with mock.patch.object(runner,'precheck_checkpoint',side_effect=lambda *args:runner.verify_executing_core(registry)), \
                mock.patch.object(runner.adapter,'load_native_gt') as gt:
            with self.assertRaisesRegex(ValueError,'Imported core differs'):
                runner.run_panel(config,{},prepared,{},Path('unused'),report,torch.device('cpu'))
            gt.assert_not_called()
        self.assertEqual(report.get('gt_targets_attempted',0),0)

    def test_failed_GT_decode_still_records_attempted_exposure(self):
        config,_,_,prepared=fixture();report=dict(frame_metrics=[],models=[],gt_targets_opened=0,predictions_completed=0)
        with mock.patch.object(runner,'precheck_checkpoint',return_value={}), \
                mock.patch.object(runner.adapter,'load_native_gt',side_effect=RuntimeError('decode failed')):
            with self.assertRaisesRegex(RuntimeError,'decode failed'):
                runner.run_panel(config,{'source_root':Path('unused')},prepared,{},Path('unused'),report,torch.device('cpu'))
        self.assertEqual(report['gt_targets_attempted'],1);self.assertEqual(report['gt_targets_opened'],0)
        self.assertEqual(report['gt_target_attempt_ids'],[prepared['targets'][0]['sample_id']])

    def test_actual_checkpoint_precheck_requires_final_complete_log_and_best_winner(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config=example();case=config['cases'][0];ckpt=case['checkpoint'];ckpt['update']=250
            manifest={split:[dict(sample_id=split+str(i)) for i in range(n)] for split,n in [('train',320),('val',20)]}
            training=json.loads((SCRIPT.parents[1]/'configs/pilot-A-event-seed9471-v1.json').read_text())
            path=root/'training.json';path.write_text(json.dumps(training))
            registry=dict(source_files={'training.json':runner.file_sha(path)},git_commit='c'*40)
            registry['source_identity_sha256']=hashlib.sha256(json.dumps(registry['source_files'],sort_keys=True).encode()).hexdigest()
            registry_path=root/'source-registry.json';registry_path.write_text(json.dumps(registry))
            case['training_source_identity_sha256']=registry['source_identity_sha256']
            case['training_initial_weights_sha256']='a'*64
            ckpt['training_config_sha256']=runner.file_sha(path)
            ckpt['training_source_registry_sha256']=runner.file_sha(registry_path)
            binding=dict(config_sha256=ckpt['training_config_sha256'],config_identity=runner.digest(training),
                source_identity_sha256=registry['source_identity_sha256'],source_registry_sha256=ckpt['training_source_registry_sha256'],
                git_commit=registry['git_commit'],manifest_sha256=config['synthetic_data']['manifest_sha256'],
                train_sample_ids_sha256=runner.digest([r['sample_id'] for r in manifest['train']]),
                val_sample_ids_sha256=runner.digest([r['sample_id'] for r in manifest['val']]),
                training_samples=320,validation_samples=20,total_updates=1000,effective_batch=8)
            metadata=dict(config=training,binding=binding,fresh_seed_model_sha256='a'*64)
            metadata_path=root/'metadata.json';metadata_path.write_text(json.dumps(metadata))
            ckpt['metadata_sha256']=runner.file_sha(metadata_path)
            rows=[dict(update=u,mean_scene_epe=e,untrained_reference=(u==0)) for u,e in [(0,.1),(250,2.),(500,3.),(1000,4.)]]
            raw=b''.join((json.dumps(row)+'\n').encode() for row in rows)
            prefix=b''.join((json.dumps(row)+'\n').encode() for row in rows[:2])
            log=root/'validation.jsonl';log.write_bytes(raw)
            ckpt['selection_validation_log_bytes']=len(raw)
            ckpt['selection_validation_log_sha256']=hashlib.sha256(raw).hexdigest()
            checkpoint=root/'best-val.pt'
            payload=dict(schema=1,binding=binding,state=dict(update=250,presentations=2000,best_update=250,best_val_epe=2.),
                logs={'validation':dict(bytes=len(prefix),sha256=hashlib.sha256(prefix).hexdigest())})
            torch.save(payload,checkpoint);ckpt['path_sha256']=runner.file_sha(checkpoint)
            env={ckpt[key+'_env']:str(p) for key,p in [('path',checkpoint),('metadata',metadata_path),
                ('training_config',path),('training_source_registry',registry_path)]}
            env[ckpt['selection_validation_log_env']]=str(log)
            with mock.patch.dict(runner.os.environ,env):
                with mock.patch.object(runner,'verify_executing_core',wraps=runner.verify_executing_core) as core_guard:
                    selection=runner.precheck_checkpoint(case,config,manifest)['selection']
                    core_guard.assert_called_once()
                self.assertEqual(selection['best_update'],250)  # Val0 .1 cannot compete.
                log.write_bytes(raw+b'\n')
                with self.assertRaises(ValueError):runner.precheck_checkpoint(case,config,manifest)
                # Even an exactly rebound full log cannot hide a later better update.
                rows[-1]['mean_scene_epe']=1.
                changed=b''.join((json.dumps(row)+'\n').encode() for row in rows)
                log.write_bytes(changed);ckpt['selection_validation_log_bytes']=len(changed)
                ckpt['selection_validation_log_sha256']=hashlib.sha256(changed).hexdigest()
                with self.assertRaises(ValueError):runner.precheck_checkpoint(case,config,manifest)

    def test_partial_failure_preserves_counts_and_no_completed_summary(self):
        config,_,_,prepared=fixture();report=dict(frame_metrics=[],models=[],gt_targets_opened=0,predictions_completed=0)
        training=json.loads((SCRIPT.parents[1]/'configs/pilot-A-event-seed9471-v1.json').read_text())
        fake=mock.Mock(side_effect=[{'disparity':torch.ones((1,1,120,160))},RuntimeError('mock forward failure')])
        fake.state_dict.return_value={'weight':torch.zeros(1)}
        with mock.patch.object(runner,'precheck_checkpoint',return_value={}), \
                mock.patch.object(runner.adapter,'load_native_gt',return_value=(np.ones((1,1,2)),np.ones((1,1,2),bool))), \
                mock.patch.object(runner.adapter,'load_prepared_sensor',return_value={v:np.zeros((50,2,1,1),np.float32) for v in ('left','right')}), \
                mock.patch.object(runner.functionality,'load_checkpoint',return_value=(fake,training,{'fixture':True})), \
                mock.patch.object(runner.adapter,'predict_to_native',return_value=torch.ones((1,1,1,2))):
            with self.assertRaisesRegex(RuntimeError,'mock forward failure'):
                runner.run_panel(config,{'source_root':Path('unused'),'prepared_root':Path('unused')},prepared,{},Path('unused'),report,torch.device('cpu'))
        self.assertEqual(report['gt_targets_opened'],20);self.assertEqual(report['predictions_completed'],1)
        self.assertEqual(report['models'][0]['status'],'STARTED')
        self.assertTrue(report['models'][0]['weights_and_buffers_unchanged'])
        self.assertEqual(len(report['frame_metrics']),21);self.assertNotIn('equal_sequence_summary',report)

    def test_mutating_weights_or_buffers_invalidates_panel_completion(self):
        for changed in ('weight','persistent_buffer'):
            with self.subTest(changed=changed):
                config,_,_,prepared=fixture();report=dict(frame_metrics=[],models=[],gt_targets_opened=0,predictions_completed=0)
                training=json.loads((SCRIPT.parents[1]/'configs/pilot-A-event-seed9471-v1.json').read_text())
                state={'weight':torch.zeros(1),'persistent_buffer':torch.zeros(1)}
                def forward(left,right):
                    state[changed].add_(1)
                    return {'disparity':torch.ones((1,1,120,160))}
                fake=mock.Mock(side_effect=forward);fake.state_dict.side_effect=lambda:state
                with mock.patch.object(runner,'precheck_checkpoint',return_value={}), \
                        mock.patch.object(runner.adapter,'load_native_gt',return_value=(np.ones((1,1,2)),np.ones((1,1,2),bool))), \
                        mock.patch.object(runner.adapter,'load_prepared_sensor',return_value={v:np.zeros((50,2,1,1),np.float32) for v in ('left','right')}), \
                        mock.patch.object(runner.functionality,'load_checkpoint',return_value=(fake,training,{'fixture':True})), \
                        mock.patch.object(runner.adapter,'predict_to_native',return_value=torch.ones((1,1,1,2))):
                    with self.assertRaisesRegex(RuntimeError,'Frozen zero-shot weights or persistent buffers changed'):
                        runner.run_panel(config,{'source_root':Path('unused'),'prepared_root':Path('unused')},prepared,{},Path('unused'),report,torch.device('cpu'))
                self.assertEqual(report['predictions_completed'],20)
                self.assertEqual(report['models'][0]['status'],'FAILED_WEIGHTS_OR_BUFFERS_MUTATED')
                self.assertFalse(report['models'][0]['weights_and_buffers_unchanged'])
                self.assertNotIn('equal_sequence_summary',report)


if __name__=='__main__':unittest.main()
