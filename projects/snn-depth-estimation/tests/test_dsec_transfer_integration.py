"""Tiny future-launch wiring tests; SSH/Slurm/files/models are mocked."""
import copy
import hashlib
import json
from pathlib import Path
import shlex
import sys
import unittest
from unittest import mock

SCRIPTS=Path(__file__).resolve().parents[1]/'experiments/stereo-spiking-transformer-study/scripts'
sys.path.insert(0,str(SCRIPTS))
import cluster_control as control
import register_dsec_transfer as registration


def fixture():
    c=json.loads((SCRIPTS.parent/'configs/dsec-event-transfer-v1.example.json').read_text())
    c.update(registered=True,runtime_dependency_run=registration.RUNTIME_RUN,runtime_required_status='COMPLETED_PRIVATE_DSEC_RUNTIME')
    for key in c['dsec']:
        if key.endswith('_sha256'):c['dsec'][key]='a'*64
    data=c['dsec'];data.update(runtime_receipt_env='STUDY_DSEC_RUNTIME_RECEIPT',runtime_receipt_sha256='b'*64,
        runtime_config_sha256='c'*64,runtime_package_files_manifest_sha256='d'*64)
    c['synthetic_data']['audit_receipt_sha256']='e'*64
    c['data_integrity_gate']=dict(run='audit-causal-quarter-pilot-v1',manifest_sha256=c['synthetic_data']['manifest_sha256'])
    c['launch_integration']=dict(prepared_run=registration.PREPARATION_RUN,preparation_source_snapshot='source-preparation-v14',
        runtime_run=registration.RUNTIME_RUN,prepared_manifest_filename='manifest.json',runtime_receipt_filename='result.json',
        runtime_prefix_relative='packages',runtime_prefix_env='STUDY_DSEC_RUNTIME_PREFIX',baseline_capsule_env='STUDY_BASELINE_CAPSULE',
        controller_and_Slurm_wiring_required=True,no_submission_by_registration_helper=True)
    for case in c['cases']:
        run=registration.RUNS[case['case_id']]
        case.update(training_run=run,training_config=run+'.json',training_source_snapshot=registration.TRAIN_SNAPSHOT,
            training_source_identity_sha256=registration.TRAIN_ID,training_initial_weights_sha256='f'*64)
        for key in case['checkpoint']:
            if key.endswith('_sha256'):case['checkpoint'][key]='f'*64
        case['checkpoint']['training_source_registry_sha256']=registration.TRAIN_REGISTRY_SHA
    cluster=dict(run_root='/verified/runs',source_root='/verified/source',environment='/verified/python',
        dsec_source_root='/verified/DSEC',dsec_preflight_manifest='/verified/metadata/preflight-v5.json',baseline_capsule='/verified/capsule')
    source=cluster['source_root']+'/stereo-study/source-preparation-v14'
    entries=[dict(run=run,state='COMPLETED',result_status=status,source=source,
        source_identity=data['preparation_source_identity_sha256'],config_sha256=sha,
        output=cluster['run_root']+'/stereo-spiking-transformer-study/'+run)
        for run,status,sha in [(registration.PREPARATION_RUN,'COMPLETED_PROSPECTIVE_ENGINEERING_PREPARATION',data['preparation_config_sha256']),
                              (registration.RUNTIME_RUN,'COMPLETED_PRIVATE_DSEC_RUNTIME',data['runtime_config_sha256'])]]
    return cluster,c,entries


class DSECIntegrationTests(unittest.TestCase):
    def test_distinct25_checkpoint_aliases_and_complete_data_mapping(self):
        cluster,c,entries=fixture();env=control.dsec_transfer_paths(cluster,c,entries)
        aliases={k:v for k,v in env.items() if k.startswith('STUDY_DSEC_CASE')}
        self.assertEqual(len(aliases),25);self.assertEqual(len(set(aliases)),25)
        self.assertTrue(env['STUDY_DSEC_CASE0_PATH'].endswith('pilot-A-event-seed9471-v1/best-val.pt'))
        self.assertTrue(env['STUDY_DSEC_CASE4_PATH'].endswith('pilot-original-I-event-seed9471-v1/best-val.pt'))
        self.assertTrue(env['STUDY_DSEC_CASE3_TRAINING_CONFIG'].endswith('pilot-continuous-B-event-seed9471-v1.json'))
        self.assertEqual(env['STUDY_DSEC_SOURCE'],cluster['dsec_source_root'])
        self.assertEqual(env['STUDY_DSEC_PREFLIGHT_MANIFEST'],cluster['dsec_preflight_manifest'])
        self.assertEqual(env['STUDY_BASELINE_CAPSULE'],cluster['baseline_capsule'])
        self.assertTrue(env['STUDY_DSEC_PREPARED_MANIFEST'].endswith(registration.PREPARATION_RUN+'/manifest.json'))
        self.assertTrue(env['STUDY_DSEC_RUNTIME_PREFIX'].endswith(registration.RUNTIME_RUN+'/packages'))
        self.assertTrue(env['STUDY_PAIRED_AUDIT'].endswith('audit-causal-quarter-pilot-v1/result.json'))

    def test_unregistered_or_partial_example_cannot_unlock_launch(self):
        cluster,c,entries=fixture()
        example=json.loads((SCRIPTS.parent/'configs/dsec-event-transfer-v1.example.json').read_text())
        with self.assertRaisesRegex(ValueError,'unregistered'):control.dsec_transfer_paths(cluster,example,entries)
        c['cases'].pop()
        with self.assertRaises(ValueError):control.dsec_transfer_paths(cluster,c,entries)

    def test_path_alias_ledger_and_source_redirects_are_rejected(self):
        changes=[lambda c,e:c['cases'][1]['checkpoint'].update(path_env='STUDY_DSEC_CASE0_PATH'),
            lambda c,e:c['cases'][0].update(training_run='../outside'),
            lambda c,e:c['cases'][0].update(training_config='../outside.json'),
            lambda c,e:c['cases'][0].update(training_source_snapshot='other-source'),
            lambda c,e:c['cases'][0].update(training_source_identity_sha256='f'*64),
            lambda c,e:c['dsec'].update(prepared_root_env='STUDY_SOURCE'),
            lambda c,e:c['launch_integration'].update(prepared_manifest_filename='../other.json'),
            lambda c,e:c['launch_integration'].update(preparation_source_snapshot='../other'),
            lambda c,e:e[0].update(output='/outside'),lambda c,e:e[1].update(state='RUNNING'),
            lambda c,e:e[0].update(source_identity='f'*64)]
        for change in changes:
            cluster,c,entries=fixture();change(c,entries)
            with self.subTest(change=change):
                with self.assertRaises(ValueError):control.dsec_transfer_paths(cluster,c,entries)

    def runtime(self,c,root):
        d=c['dsec']
        return dict(sha256=d['runtime_receipt_sha256'],value=dict(status=c['runtime_required_status'],
            config_sha256=d['runtime_config_sha256'],source_identity_sha256=d['preparation_source_identity_sha256'],
            package_prefix=root+'/packages',package_files_manifest_sha256=d['runtime_package_files_manifest_sha256'],
            changes_shared_environment=False,model_inference=False,event_conversion=False,test_access=False,
            runtime={'Blosc_filter_32001':True},packages=['h5py==3.14.0','hdf5plugin==6.0.0'],package_files=129))

    def test_exact_private_runtime_binding(self):
        _,c,_=fixture();root='/runtime';receipt=self.runtime(c,root)
        control.validate_dsec_runtime_binding(c,receipt,root)
        for key,value in [('package_prefix','/other/packages'),('config_sha256','f'*64),
                ('source_identity_sha256','f'*64),('package_files_manifest_sha256','f'*64),
                ('changes_shared_environment',True),('model_inference',True),('test_access',True),
                ('packages',['h5py==9.0.0'])]:
            bad=copy.deepcopy(receipt);bad['value'][key]=value
            with self.subTest(key=key):
                with self.assertRaises(ValueError):control.validate_dsec_runtime_binding(c,bad,root)
        receipt['sha256']='f'*64
        with self.assertRaises(ValueError):control.validate_dsec_runtime_binding(c,receipt,root)

    def test_full_mock_preflight_hashes_before_any_submit(self):
        cluster,c,entries=fixture();env=control.dsec_transfer_paths(cluster,c,entries)
        prefix='projects/snn-depth-estimation/experiments/stereo-spiking-transformer-study/scripts/'
        registry={'source_files':{prefix+n:'9'*64 for n in ('evaluate_dsec_transfer.py','evaluate_functionality.py','setup_dsec_runtime.py','register_dsec_transfer.py')}}
        registry['source_files'][prefix+'dsec_adapter.py']=c['dsec']['adapter_script_sha256']
        prep_files={prefix+'dsec_adapter.py':c['dsec']['adapter_script_sha256'],prefix+'setup_dsec_runtime.py':'8'*64,
            'projects/snn-depth-estimation/experiments/stereo-spiking-transformer-study/configs/dsec-quarter-train-preparation-v1.json':c['dsec']['preparation_config_sha256']}
        prep_id=hashlib.sha256(json.dumps(prep_files,sort_keys=True).encode()).hexdigest()
        # Test one actual metadata map digest rather than bypassing map checks.
        c['dsec']['preparation_source_identity_sha256']=prep_id
        for e in entries:e['source_identity']=prep_id
        d=c['dsec'];records={}
        bindings={'prepared':'prepared_manifest_sha256','preflight':'preflight_manifest_sha256',
            'preparation_config':'preparation_config_sha256','preparation_registry':'preparation_source_registry_sha256',
            'runtime_package_manifest':'runtime_package_files_manifest_sha256'}
        for key,field in bindings.items():records[key]=dict(sha256=d[field],value={})
        records['prepared']['value']={'binding':{'git_commit':'1'*40}}
        records['preparation_registry']['value']=dict(source_files=prep_files,source_identity_sha256=prep_id,git_commit='1'*40)
        records['runtime_verifier']={'sha256':'8'*64};records['runtime_adapter']={'sha256':d['adapter_script_sha256']}
        records['synthetic_manifest']={'sha256':c['synthetic_data']['manifest_sha256']}
        records['synthetic_audit']={'sha256':c['synthetic_data']['audit_receipt_sha256']}
        root=str(Path(env['STUDY_DSEC_RUNTIME_RECEIPT']).parent);records['runtime']=self.runtime(c,root)
        core_member='projects/snn-depth-estimation/core/study_models.py'
        train_files={core_member:'7'*64};registry['source_files'][core_member]='7'*64
        train_id=hashlib.sha256(json.dumps(train_files,sort_keys=True).encode()).hexdigest()
        for i,case in enumerate(c['cases']):
            case['training_source_identity_sha256']=train_id
            for role in ('path','metadata','training_config','training_source_registry','selection_validation_log'):
                records[f'case{i}_{role}']=dict(sha256=case['checkpoint'][role+'_sha256'],bytes=case['checkpoint']['selection_validation_log_bytes'])
            records[f'case{i}_status']={'value':dict(status='COMPLETED_FROZEN_HORIZON',update=1000,presentations=8000,test_access='sealed',best_update=1000)}
            records[f'case{i}_training_source_registry']['value']=dict(source_identity_sha256=train_id,source_files=train_files)
        audit={'path':env['STUDY_PAIRED_AUDIT'],'sha256':c['synthetic_data']['audit_receipt_sha256']}
        calls=[]
        def ssh(cluster,command):
            calls.append(command)
            if '--verify-existing' in command:return json.dumps(dict(status='VERIFIED_PRIVATE_DSEC_RUNTIME',package_prefix=root+'/packages',package_files=129))
            return json.dumps(records)
        with mock.patch('evaluate_dsec_transfer.validate_prepared_receipt',return_value=True), \
                mock.patch.object(registration,'TRAIN_ID',train_id):
            actual,runtime,prepared=control.prepare_dsec_transfer(cluster,c,registry,entries,audit,remote_fn=ssh)
        self.assertEqual(actual['STUDY_DSEC_CASE4_PATH'],env['STUDY_DSEC_CASE4_PATH'])
        self.assertEqual(runtime['package_files_verified'],129);self.assertEqual(prepared['targets'],20)
        self.assertEqual(len(calls),2);self.assertFalse(any('sbatch' in command for command in calls))
        registry['source_files'][core_member]='6'*64;calls.clear()
        with mock.patch('evaluate_dsec_transfer.validate_prepared_receipt',return_value=True), \
                mock.patch.object(registration,'TRAIN_ID',train_id):
            with self.assertRaisesRegex(ValueError,'Executing core differs'):
                control.prepare_dsec_transfer(cluster,c,registry,entries,audit,remote_fn=ssh)
        self.assertEqual(len(calls),1)

    def test_slurm_keeps_private_runtime_and_existing_energy_interfaces(self):
        text=(SCRIPTS.parent/'slurm/stage.sbatch').read_text()
        block=text.split('== dsec_transfer ]]; then',1)[1].split('\nfi',1)[0]
        self.assertLess(block.index('--verify-existing'),block.index('evaluate_dsec_transfer.py'))
        self.assertIn('STUDY_DSEC_RUNTIME_PREFIX',block);self.assertIn('STUDY_DSEC_RUNTIME_RECEIPT_SHA256',block)
        self.assertIn('== energy_interfaces ]]; then',text);self.assertIn('diagnose_energy_interfaces.py',text)


if __name__=='__main__':unittest.main()
