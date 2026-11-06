"""Local command/accounting tests with mocked SSH; never submit a real job."""
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shlex
import sys
import tarfile
import tempfile
import unittest
from unittest import mock


STUDY = Path(__file__).resolve().parents[1] / 'experiments/stereo-spiking-transformer-study'
sys.path.insert(0, str(STUDY/'scripts'))
SPEC = importlib.util.spec_from_file_location('study_training_launch_control_test', STUDY/'scripts/launch_training.py')
LAUNCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LAUNCH)


class LaunchTrainingTests(unittest.TestCase):
    def fixture(self, root):
        snapshot = root/'source-test'
        snapshot.mkdir()
        review = 'projects/snn-depth-estimation/experiments/stereo-spiking-transformer-study/analysis/health.md'
        forecast = 'projects/snn-depth-estimation/experiments/stereo-spiking-transformer-study/analysis/forecast.md'
        cfg = {'schema':'stereo-study-training-v1','registered':True,'scope':'engineering','allocated_gpu_hour_cap':24,
               'test_access':'sealed','pretraining':False,'seed':9471,'total_updates':1000,
               'data':{'dataset_version':'paired-quarter','manifest_sha256':'f'*64},
               'training':{'effective_batch':8,'microbatch':8,'checkpoint_interval':25},
               'scheduler_resources':{'time':'01:00:00'},
               'execution_review':{'health_reviewed':True,'review_record':review,'resource_forecast_record':forecast,
                                   'resource_forecast_sha256':hashlib.sha256(b'measured forecast').hexdigest()}}
        members = {LAUNCH.CONFIG_PREFIX+'run.json':json.dumps(cfg).encode(),review:b'health review',forecast:b'measured forecast'}
        with tarfile.open(snapshot/'source.tar.gz','w:gz') as archive:
            for name, payload in members.items():
                info=tarfile.TarInfo(name);info.size=len(payload);archive.addfile(info,io.BytesIO(payload))
        with tarfile.open(snapshot/'overlay.tar.gz','w:gz'):
            pass
        hashes={name:hashlib.sha256(value).hexdigest() for name,value in members.items()}
        registry={'git_commit':'a'*40,'source_files':hashes,
                  'source_identity_sha256':hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),
                  'archive_sha256':hashlib.sha256((snapshot/'source.tar.gz').read_bytes()).hexdigest(),
                  'overlay_sha256':hashlib.sha256((snapshot/'overlay.tar.gz').read_bytes()).hexdigest()}
        (snapshot/'source-registry.json').write_text(json.dumps(registry))
        cluster={'ssh_host':'mock-only','expected_user':'rongliang','expected_hostname':'cccl3',
                 'source_root':"/source x'$(not-executed)",'run_root':'/raw runs','environment':"/env x'$(not-executed)",
                 'baseline_capsule':'/pinned capsule'}
        return snapshot, cluster

    def test_prepare_commands_are_quoted_and_dry_run_never_calls_ssh(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);snapshot,cluster=self.fixture(root)
            plan=LAUNCH.prepare(cluster,snapshot,'run.json','example-run',200)
            for command in (LAUNCH.probe_command(cluster,plan),LAUNCH.submission_command(cluster,plan,1,'owner')):
                words=shlex.split(command)
                self.assertEqual(words[:2],[cluster['environment']+'/bin/python','-c'])
                compile(words[2],'<mock-remote-command>','exec')
                self.assertIn('not-executed',words[2])
            config=root/'cluster.json';config.write_text(json.dumps(cluster))
            argv=['launch_training.py','--cluster-config',str(config),'--snapshot',str(snapshot),'--config','run.json',
                  '--run','example-run','--stop-at-update','200','--dry-run']
            with mock.patch.object(LAUNCH,'STUDY',root),mock.patch.object(LAUNCH,'remote',side_effect=AssertionError('No SSH in dryrun')) as ssh:
                with mock.patch.object(sys,'argv',argv),contextlib.redirect_stdout(io.StringIO()) as output:
                    LAUNCH.main()
            self.assertFalse(ssh.called)
            self.assertTrue(json.loads(output.getvalue())['no_cluster_calls'])

    def test_strict_accounting_terminal_actual_and_nonterminal_full_reservation(self):
        rows=[{'job_id':'1','state':'COMPLETED','allocated_seconds':'120','resources':{'gpus':2,'limit':'01:00:00'}},
              {'job_id':'2','state':'RUNNING','allocated_seconds':'2','resources':{'gpus':1,'limit':'02:00:00'}}]
        self.assertEqual(LAUNCH.budget_seconds(rows),240+7200)
        missing=copy.deepcopy(rows);missing[0]['allocated_seconds']=None
        with self.assertRaisesRegex(RuntimeError,'actualallocated'):
            LAUNCH.budget_seconds(missing)
        with self.assertRaisesRegex(RuntimeError,'cap'):
            LAUNCH.check_budget(rows,24*3600)

    def test_reviewed_32_update_diagnostic_needs_no_existing_training_outcome(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);snapshot,cluster=self.fixture(root)
            plan=LAUNCH.prepare(cluster,snapshot,'run.json','diagnostic-32',32)
            self.assertEqual(plan['stop_at_update'],32)
            self.assertEqual(plan['total_updates'],1000)
            self.assertEqual(LAUNCH.validate_attempt([],plan),1)

    def test_complete_data_audit_receipt_is_required_before_submission(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);snapshot,cluster=self.fixture(root)
            plan=LAUNCH.prepare(cluster,snapshot,'run.json','diagnostic-32',32)
            cluster['run_root']=str(root/'runs');plan['source']=str(snapshot);plan['output']=str(root/'new-output')
            with tarfile.open(snapshot/'source.tar.gz') as archive:
                payload=archive.extractfile(plan['config_member']).read()
            config_path=snapshot/plan['config_member'];config_path.parent.mkdir(parents=True);config_path.write_bytes(payload)
            plan['config']['data_integrity_gate']={'run':'audit-pilot','manifest_sha256':plan['manifest_sha256'],
                'required_status':'PASS_ALL_REGISTERED_ASSETS_TRACES_AND_SOURCE_IDENTITIES','required_sample_count':340}
            report=root/'runs/stereo-spiking-transformer-study/audit-pilot/result.json'
            report.parent.mkdir(parents=True)
            def probe():
                code=shlex.split(LAUNCH.probe_command(cluster,plan))[2]
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    exec(compile(code,'<local-readonly-probe>','exec'),{})
                return json.loads(output.getvalue())
            with self.assertRaises(FileNotFoundError):probe()
            row={'status':'PASS_METADATA_ONLY_ARRAYS_NOT_AUDITED','manifest_sha256':plan['manifest_sha256'],'samples_checked':0}
            report.write_text(json.dumps(row))
            with self.assertRaisesRegex(AssertionError,'has not passed'):probe()
            row.update(status='PASS_ALL_REGISTERED_ASSETS_TRACES_AND_SOURCE_IDENTITIES',samples_checked=339)
            report.write_text(json.dumps(row))
            with self.assertRaisesRegex(AssertionError,'incomplete'):probe()
            row.update(samples_checked=340,slurm_job_id='42');report.write_text(json.dumps(row))
            answer=probe()
            self.assertEqual(answer['data_integrity_receipt']['slurm_job_id'],'42')
            self.assertEqual(answer['data_integrity_receipt']['sha256'],hashlib.sha256(report.read_bytes()).hexdigest())

    def test_mock_submission_holds_lock_appends_attempt_and_never_retries(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);snapshot,cluster=self.fixture(root)
            plan=LAUNCH.prepare(cluster,snapshot,'run.json','example-run',200)
            ledger=root/'jobs.json'
            calls=[]
            def ssh(c, command):
                calls.append(command)
                if command=='id -un; hostname':return 'rongliang\ncccl3'
                if command.startswith('mkdir '):return ''
                if 'Read-only' in command:return '{}'
                if command.startswith('test '):return ''
                code=shlex.split(command)[2]
                if 'ck=torch.load' in code:return json.dumps({'exists':False})
                if "'sbatch'" in code:
                    self.assertFalse(ledger.exists())
                    self.assertIn('count+1<=1',code)
                    self.assertIn('unreconciled durable',code)
                    return json.dumps({'job_id':'12345','status':'SUBMITTED'})
                raise AssertionError(command)
            entry=LAUNCH.submit(cluster,plan,ledger,remote_fn=ssh)
            self.assertEqual(entry['attempt'],1)
            self.assertEqual(entry['stop_at_update'],200)
            self.assertEqual(json.loads(ledger.read_text())[0]['job_id'],'12345')
            self.assertTrue(calls[-1].startswith('test '))
            with self.assertRaisesRegex(RuntimeError,'already registered'):
                LAUNCH.validate_attempt([entry],plan)

    def test_one_existing_gpu_blocks_training_before_ticket_or_sbatch(self):
        for state in ('RUNNING','PENDING'):
            with self.subTest(state=state),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);snapshot,cluster=self.fixture(root)
                cluster['source_root']=str(root/'source')
                lock=root/'source/stereo-study/submit.lock';lock.mkdir(parents=True)
                (lock/'owner').write_text('owner')
                plan=LAUNCH.prepare(cluster,snapshot,'run.json','example-run',50)
                command=LAUNCH.submission_command(cluster,plan,1,'owner')
                code=shlex.split(command)[2]
                def query(args,**kwargs):
                    if args[0]=='squeue':return f'9|{state}|gres/gpu:rtx5090:1\n'
                    return 'Command='+cluster['source_root']+'/stereo-study/frozen/train.sbatch\n'
                with mock.patch('subprocess.check_output',side_effect=query) as scheduler:
                    with self.assertRaisesRegex(AssertionError,'one GPU'):
                        exec(compile(code,'<mock-one-gpu-training-gate>','exec'),{})
                self.assertEqual(scheduler.call_count,2)
                self.assertEqual(scheduler.call_args_list[0].args[0][0],'squeue')
                self.assertFalse((root/'source/stereo-study/training-submissions').exists())

    def test_resume_and_dependency_require_health_identity_and_terminal_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);snapshot,cluster=self.fixture(root)
            plan=LAUNCH.prepare(cluster,snapshot,'run.json','example-run',400,resume=True)
            last={**plan,'job_id':'12345','state':'COMPLETED','allocated_seconds':300}
            review={'health_reviewed':True,'decision':'continue_same_frozen_recipe','last_job_id':'12345','checkpoint_update':200,
                    **{key:plan[key] for key in ('run','source_identity','config_sha256','manifest_sha256')}}
            self.assertEqual(LAUNCH.validate_attempt([last],plan,review),2)
            changed=copy.deepcopy(plan);changed['config_sha256']='0'*64
            with self.assertRaisesRegex(RuntimeError,'identity'):
                LAUNCH.validate_attempt([last],changed,review)
            with self.assertRaisesRegex(RuntimeError,'terminal'):
                LAUNCH.validate_attempt([{**last,'state':'RUNNING'}],plan,review)
            with self.assertRaisesRegex(ValueError,'health review'):
                LAUNCH.validate_attempt([last],plan)
            dependency=copy.deepcopy(plan);dependency['resume']=False;dependency['run']='new-run';dependency['dependency']='12345'
            with self.assertRaisesRegex(ValueError,'compatible'):
                LAUNCH.validate_attempt([last],dependency)
            dependency['config']['execution_review']['compatible_dependency_job_ids']=['12345']
            self.assertEqual(LAUNCH.validate_attempt([last],dependency),1)


if __name__=='__main__':
    unittest.main()
