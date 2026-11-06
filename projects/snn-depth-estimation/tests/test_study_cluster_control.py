"""Tiny submission transaction tests; SSH and Slurm are replaced completely."""
import contextlib
import copy
import io
import json
from pathlib import Path
import shlex
import sys
import tempfile
import unittest
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[1]/'experiments/stereo-spiking-transformer-study/scripts'
sys.path.insert(0,str(SCRIPTS))
import cluster_control as control
import setup_dsec_runtime as runtime


class StageSubmissionTests(unittest.TestCase):
    def test_branch_cases_bind_distinct_checkpoint_aliases_and_reject_redirects(self):
        config=json.loads((SCRIPTS.parent/'configs/branch-shift-AB-spike-v1.json').read_text())
        cluster={'source_root':'/source','run_root':'/runs'}
        env={}
        for case in config['cases']:env.update(control.branch_checkpoint_env(cluster,case))
        self.assertEqual(len(env),10)
        self.assertTrue(env['STUDY_BRANCH_A_CHECKPOINT'].endswith('pilot-A-spike-seed9471-v1/best-val.pt'))
        self.assertTrue(env['STUDY_BRANCH_B_FINAL_VALIDATION_LOG'].endswith('pilot-B-spike-seed9471-v1/validation.jsonl'))
        broken=copy.deepcopy(config['cases'][1]);broken['checkpoint']['path_env']='STUDY_BRANCH_A_CHECKPOINT'
        with self.assertRaisesRegex(ValueError,'case-specific'):control.branch_checkpoint_env(cluster,broken)
        broken=copy.deepcopy(config['cases'][0]);broken['training_config']='../outside.json'
        with self.assertRaisesRegex(ValueError,'Safe exact'):control.branch_checkpoint_env(cluster,broken)

    def fixture(self, root):
        cluster = dict(source_root=str(root/'source'),environment="/mock env'$(never)",expected_user='mock')
        (root/'source/stereo-study').mkdir(parents=True)
        entry = dict(run='diagnostic',source_identity='a'*64,git_commit='b'*40,
            config_sha256='c'*64,output=str(root/'raw/diagnostic'),dependency=None,
            config={'allocated_gpu_hour_cap':24},resources={'gpus':1,'limit':'00:03:00'})
        ledger = root/'jobs.json'
        return cluster, entry, ledger

    def remote(self, cluster, ledger, *, queue='', foreign=False, lose_receipt=False, mutate_before_return=None):
        submitted=[]; releases=[]
        def scheduler(command, **kwargs):
            if command[0]=='squeue':return queue
            if command[0]=='scontrol':
                return 'Command='+('/separate-sew/train.sbatch' if foreign else cluster['source_root']+'/stereo-study/frozen/stage.sbatch')+'\n'
            self.assertEqual(command[0],'sbatch')
            self.assertEqual(command[-1],'/mock script')
            submitted.append(command)
            return '42;mock-cluster\n'
        def remote_fn(c,command):
            lock=Path(cluster['source_root'])/'stereo-study/submit.lock'
            if command.startswith('mkdir '):
                words=shlex.split(command); lock.mkdir()
                (lock/'owner').write_text(words[words.index('printf')+2]); return ''
            if command.startswith('test '):
                releases.append(json.loads(ledger.read_text()) if ledger.exists() else [])
                (lock/'owner').unlink();lock.rmdir();return ''
            words=shlex.split(command)
            self.assertEqual(words[:2],[cluster['environment']+'/bin/python','-c'])
            with mock.patch('subprocess.check_output',side_effect=scheduler),contextlib.redirect_stdout(io.StringIO()) as stream:
                exec(compile(words[2],'<mock-stage-scheduler>','exec'),{})
            if mutate_before_return:mutate_before_return()
            if lose_receipt:raise ConnectionError('mock transport lost after sbatch')
            return stream.getvalue().strip()
        return remote_fn,submitted,releases

    def test_receipt_is_durable_and_ledger_merged_before_lock_release(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);cluster,entry,ledger=self.fixture(root)
            old=dict(run='old',job_id='1',state='COMPLETED',allocated_seconds=1,
                     resources={'gpus':1,'limit':'00:01:00'})
            ledger.write_text(json.dumps([old]))
            def accounting_refresh():
                changed=copy.deepcopy(old);changed['allocated_seconds']=17
                ledger.write_text(json.dumps([changed]))
            ssh,submitted,releases=self.remote(cluster,ledger,mutate_before_return=accounting_refresh)
            row=control.submit_stage(cluster,entry,{'A':'literal $()'},'/mock script',ledger,remote_fn=ssh)
            rows=json.loads(ledger.read_text())
            self.assertEqual(rows[0]['allocated_seconds'],17)
            self.assertEqual(rows[1]['job_id'],'42')
            self.assertEqual(releases[-1],rows)
            self.assertEqual(row['requested_gpu_seconds'],180)
            self.assertEqual(len(submitted),1)
            ticket=json.loads((root/'source/stereo-study/training-submissions/diagnostic-stage.json').read_text())
            self.assertEqual((ticket['status'],ticket['job_id']),('SUBMITTED','42'))

    def test_one_existing_study_gpu_request_rejects_before_ticket_or_sbatch(self):
      for state in ('RUNNING','PENDING'):
        with self.subTest(state=state),tempfile.TemporaryDirectory() as temp:
            root=Path(temp);cluster,entry,ledger=self.fixture(root)
            ssh,submitted,releases=self.remote(cluster,ledger,queue=f'1|{state}|gres/gpu:1\n')
            with self.assertRaisesRegex(RuntimeError,'durable stage'):
                control.submit_stage(cluster,entry,{},'/mock script',ledger,remote_fn=ssh)
            self.assertEqual(submitted,[])
            self.assertFalse(ledger.exists())
            self.assertEqual(len(releases),1)

    def test_separate_sew_running_and_pending_do_not_occupy_study_slot(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);cluster,entry,ledger=self.fixture(root)
            ssh,submitted,_=self.remote(cluster,ledger,queue='51|RUNNING|gres/gpu:1\n52|PENDING|gres/gpu:1\n',foreign=True)
            control.submit_stage(cluster,entry,{},'/mock script',ledger,remote_fn=ssh)
            self.assertEqual(len(submitted),1)

    def test_multi_gpu_stage_is_rejected_before_remote_work(self):
        with tempfile.TemporaryDirectory() as temp:
            cluster,entry,ledger=self.fixture(Path(temp));entry['resources']['gpus']=2
            remote=mock.Mock()
            with self.assertRaisesRegex(ValueError,'exactly one GPU'):
                control.submit_stage(cluster,entry,{},'/mock script',ledger,remote_fn=remote)
            remote.assert_not_called()

    def test_lost_receipt_is_recoverable_and_cannot_resubmit(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);cluster,entry,ledger=self.fixture(root)
            ssh,submitted,_=self.remote(cluster,ledger,lose_receipt=True)
            with self.assertRaisesRegex(RuntimeError,'durable stage'):
                control.submit_stage(cluster,entry,{},'/mock script',ledger,remote_fn=ssh)
            self.assertEqual(len(submitted),1)
            ticket=root/'source/stereo-study/training-submissions/diagnostic-stage.json'
            self.assertEqual(json.loads(ticket.read_text())['job_id'],'42')
            retry,repeated,_=self.remote(cluster,ledger)
            with self.assertRaisesRegex(RuntimeError,'durable stage'):
                control.submit_stage(cluster,entry,{},'/mock script',ledger,remote_fn=retry)
            self.assertEqual(repeated,[])
            self.assertFalse(ledger.exists())

    def test_terminal_accounting_and_duplicate_runs_fail_before_remote_calls(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);cluster,entry,ledger=self.fixture(root)
            row=dict(run='old',job_id='1',state='COMPLETED',resources={'gpus':1,'limit':'00:01:00'})
            ledger.write_text(json.dumps([row]))
            with self.assertRaisesRegex(RuntimeError,'actualallocated'):
                control.submit_stage(cluster,entry,{},'/mock script',ledger,remote_fn=mock.Mock(side_effect=AssertionError('No SSH')))
            row['run']=entry['run'];ledger.write_text(json.dumps([row]))
            with self.assertRaisesRegex(RuntimeError,'already registered'):
                control.submit_stage(cluster,entry,{},'/mock script',ledger,remote_fn=mock.Mock(side_effect=AssertionError('No SSH')))


class PrivateRuntimeTests(unittest.TestCase):
    def test_config_allows_only_the_two_private_pinned_wheels(self):
        config=json.loads((SCRIPTS.parent/'configs/dsec-runtime-setup-v1.json').read_text())
        runtime.validate_config(config)
        for key,value in [('packages',['numpy==2.0.2']),('max_seconds',3600),('registered',False)]:
            broken={**config,key:value}
            with self.assertRaises(ValueError):runtime.validate_config(broken)

    def test_bound_runtime_manifest_detects_package_changes(self):
        from dsec_adapter import file_sha
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);prefix=root/'packages';prefix.mkdir()
            member=prefix/'module.py';member.write_text('trusted bytes')
            manifest=root/'package-files.json';manifest.write_text(json.dumps({'module.py':file_sha(member)}))
            receipt=root/'result.json';receipt.write_text(json.dumps(dict(status='COMPLETED_PRIVATE_DSEC_RUNTIME',
                package_prefix=str(prefix),runtime={'Blosc_filter_32001':True},package_files_manifest_sha256=file_sha(manifest))))
            sha=file_sha(receipt)
            self.assertEqual(runtime.verify_installed_runtime(receipt,sha)['package_files'],1)
            member.write_text('changed bytes')
            with self.assertRaisesRegex(ValueError,'package file changed'):
                runtime.verify_installed_runtime(receipt,sha)
            with self.assertRaisesRegex(ValueError,'receipt changed'):
                runtime.verify_installed_runtime(receipt,'0'*64)


if __name__=='__main__':
    unittest.main()
