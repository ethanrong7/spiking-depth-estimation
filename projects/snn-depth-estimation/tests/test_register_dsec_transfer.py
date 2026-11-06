"""Tiny metadata/mocked checks; no real arrays, checkpoint tensors or models."""
import copy
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest import mock

SCRIPT=Path(__file__).resolve().parents[1]/'experiments/stereo-spiking-transformer-study/scripts/register_dsec_transfer.py'
spec=importlib.util.spec_from_file_location('register_dsec_transfer_test',SCRIPT)
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


def record(value):
    raw=(json.dumps(value)+'\n').encode()
    return dict(text=raw.decode(),bytes=len(raw),sha256=runner.sha_bytes(raw))


def status():
    return dict(status='COMPLETED_FROZEN_HORIZON',update=1000,presentations=8000,total_horizon_updates=1000,
        test_access='sealed',best_update=500,best_val_epe=2.)


def bundle():
    return dict(cells={name:dict(run=run,status=status()) for name,run in runner.RUNS.items()},
        model_constructed=False,sensor_arrays_opened=False,gt_pixels_opened=False,test_access=False,
        files={'prepared':record(dict(status='COMPLETED_PROSPECTIVE_ENGINEERING_PREPARATION',
            conversion_qa=dict(ready_for_inference=True,readback_targets=20),targets=[{}]*20,gt_pixels_opened=False,test_access=False)),
            'runtime':record(dict(status='COMPLETED_PRIVATE_DSEC_RUNTIME',changes_shared_environment=False,
                test_access=False,runtime={'Blosc_filter_32001':True}))})


def completed_cell(name='A-event'):
    training=json.loads((SCRIPT.parents[1]/'configs'/(runner.RUNS[name]+'.json')).read_text())
    raw=json.dumps(training).encode();synthetic={split:[dict(sample_id=split+str(i)) for i in range(n)] for split,n in [('train',320),('val',20)]}
    registry={'git_commit':'c'*40}
    binding=dict(source_identity_sha256=runner.TRAIN_ID,source_registry_sha256=runner.TRAIN_REGISTRY_SHA,
        config_sha256=runner.sha_bytes(raw),config_identity=runner.evaluator.digest(training),git_commit=registry['git_commit'],
        manifest_sha256=training['data']['manifest_sha256'],total_updates=1000,effective_batch=8,training_samples=320,validation_samples=20,
        train_sample_ids_sha256=runner.evaluator.digest([r['sample_id'] for r in synthetic['train']]),
        val_sample_ids_sha256=runner.evaluator.digest([r['sample_id'] for r in synthetic['val']]))
    metadata=dict(config=training,binding=binding,fresh_seed_model_sha256='a'*64,baseline_provenance=None)
    if name=='ALIF+I-event':
        from core.study_baselines import BASE_ID,ARCHIVE_SHA
        metadata['baseline_provenance']=dict(source_identity_sha256=BASE_ID,archive_sha256=ARCHIVE_SHA,
            architecture='original_I',input_channels=2,event_adaptation=True,
            flow_conversion='d=-legacy_flow, where legacy_flow=x_right-x_left')
    rows=[dict(update=u,mean_scene_epe=e,untrained_reference=(u==0)) for u,e in [(0,.1),(250,3.),(500,2.),(1000,4.)]]
    text=''.join(json.dumps(row)+'\n' for row in rows);prefix=''.join(json.dumps(row)+'\n' for row in rows[:3]).encode()
    validation=dict(text=text,bytes=len(text.encode()),sha256=runner.sha_bytes(text.encode()))
    row=dict(status=status(),checkpoint_schema=1,checkpoint_binding=binding,metadata=record(metadata),validation=validation,
        checkpoint_state=dict(update=500,best_update=500,best_val_epe=2.,presentations=4000),
        checkpoint_logs={'validation':dict(bytes=len(prefix),sha256=runner.sha_bytes(prefix))})
    return row,raw,registry,synthetic,training['data']['manifest_sha256']


class RegistrationTests(unittest.TestCase):
    def test_pending_cells_never_become_baseline_first_registration(self):
        value=bundle();value['cells']['A-event']['status']=dict(status='RUNNING',update=50)
        value['cells']['continuous-A-event']['status']='NOT_STARTED'
        report=runner.readiness(value)
        self.assertEqual(report['status'],'NOT_READY_FULL_FIVE_EVENT_PANEL')
        self.assertFalse(report['registration_written']);self.assertTrue(report['prepared_ready'])
        with mock.patch.object(runner,'read_snapshot') as source:
            with self.assertRaisesRegex(ValueError,'baseline-first/subset'):
                runner.create_candidate(value,Path('unused'),Path('unused'))
            source.assert_not_called()
        value['cells']={'ALIF+I-event':value['cells']['ALIF+I-event']}
        with self.assertRaises(ValueError):runner.readiness(value)

    def test_metadata_integrity_and_no_test_or_model_receipts(self):
        value=record({'x':1});self.assertEqual(runner.record_json(value),{'x':1})
        value['text']+=' '
        with self.assertRaises(ValueError):runner.record_json(value)
        for key in ('model_constructed','sensor_arrays_opened','gt_pixels_opened','test_access'):
            bad=bundle();bad[key]=True
            with self.assertRaises(ValueError):runner.readiness(bad)
        value=bundle();value['cells']['A-event']['run']='pilot-A-spike-seed9471-v1'
        with self.assertRaises(ValueError):runner.readiness(value)

    def test_all_five_families_validate_full_final_logs_and_trained_best(self):
        for name in runner.evaluator.FAMILIES:
            with self.subTest(name=name):
                args=completed_cell(name);metadata,selection=runner.validate_completed_cell(name,*args)
                self.assertEqual(selection['best_update'],500)
                self.assertTrue(selection['initial_update_excluded_from_selection'])
                self.assertEqual(metadata['config']['data']['modality'],'event')

    def test_incomplete_or_changed_final_selection_cannot_register(self):
        for change in ('source','metadata','prefix','final_bytes','later_better','unfinished','schema','polarity'):
            row,raw,registry,synthetic,manifest=completed_cell()
            if change=='source':row['checkpoint_binding']['source_identity_sha256']='f'*64
            if change=='metadata':row['metadata']['sha256']='f'*64
            if change=='prefix':row['checkpoint_logs']['validation']['sha256']='f'*64
            if change=='final_bytes':row['validation']['text']+='\n'
            if change=='later_better':
                rows=[json.loads(line) for line in row['validation']['text'].splitlines()];rows[-1]['mean_scene_epe']=1.
                text=''.join(json.dumps(r)+'\n' for r in rows)
                row['validation']=dict(text=text,bytes=len(text.encode()),sha256=runner.sha_bytes(text.encode()))
            if change=='unfinished':row['status']['update']=500
            if change=='schema':row['checkpoint_schema']=0
            if change=='polarity':
                value=json.loads(raw);value['model']['config']['input_channels']=1;raw=json.dumps(value).encode()
            with self.subTest(change=change):
                with self.assertRaises(ValueError):runner.validate_completed_cell('A-event',row,raw,registry,synthetic,manifest)

    def test_baseline_sign_and_capsule_provenance_cannot_be_replaced(self):
        row,raw,registry,synthetic,manifest=completed_cell('ALIF+I-event')
        metadata=runner.record_json(row['metadata']);metadata['baseline_provenance']['flow_conversion']='positive_already'
        row['metadata']=record(metadata)
        with self.assertRaisesRegex(ValueError,'baseline/sign'):
            runner.validate_completed_cell('ALIF+I-event',row,raw,registry,synthetic,manifest)

    def test_snapshot_archive_and_member_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);raw=b'{"fixture":true}\n';member='config.json'
            with tarfile.open(root/'source.tar.gz','w:gz') as archive:
                info=tarfile.TarInfo(member);info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
            files={member:runner.sha_bytes(raw)}
            registry=dict(source_files=files,source_identity_sha256=runner.sha_bytes(json.dumps(files,sort_keys=True).encode()),
                archive_sha256=runner.evaluator.file_sha(root/'source.tar.gz'))
            (root/'source-registry.json').write_text(json.dumps(registry))
            actual,_=runner.read_snapshot(root);self.assertEqual(runner.frozen_member(root,actual,member),raw)
            with (root/'source.tar.gz').open('ab') as stream:stream.write(b'x')
            with self.assertRaises(ValueError):runner.read_snapshot(root)

    def test_exclusive_output_and_environment_restoration(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'configs').mkdir();path=Path(tmp)/'configs/registration.json'
            with mock.patch.object(runner.evaluator,'validate_config',return_value=True),mock.patch.object(runner,'STUDY',Path(tmp)):
                runner.write_exclusive_config(path,{'fixture':True})
                with self.assertRaises(FileExistsError):runner.write_exclusive_config(path,{'different':True})
                with self.assertRaises(ValueError):runner.write_exclusive_config(Path(tmp)/'forbidden.json',{'fixture':True})
            self.assertEqual(json.loads(path.read_text()),{'fixture':True})
        with mock.patch.dict(runner.os.environ,{'DSEC_TEST_META':'before'}):
            with runner.metadata_environment({'DSEC_TEST_META':'during'}):self.assertEqual(runner.os.environ['DSEC_TEST_META'],'during')
            self.assertEqual(runner.os.environ['DSEC_TEST_META'],'before')


if __name__=='__main__':unittest.main()
