"""Tiny local log projection tests; no cluster calls or model evaluation."""
import importlib.util
import json
from pathlib import Path
import subprocess
import shlex
import sys
import tempfile
import unittest
from unittest import mock

P=Path(__file__).resolve().parents[1]/'experiments/stereo-spiking-transformer-study/scripts/watch_study.py'
spec=importlib.util.spec_from_file_location('study_watch_local_test',P)
watch=importlib.util.module_from_spec(spec);spec.loader.exec_module(watch)


class WatchTests(unittest.TestCase):
    def test_generated_remote_projection_executes_on_tiny_fixture(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'training.jsonl').write_text('{"update":1,"loss":7.0}\n')
            command=watch.training_command({'environment':str(Path(sys.executable).parents[1])},folder)
            args=shlex.split(command)
            output=subprocess.check_output([sys.executable,*args[1:]],text=True)
            self.assertEqual(json.loads(output)['training']['rows'][0]['loss'],7.0)

    def test_successful_refresh_clears_prior_errors_without_losing_new_job(self):
        old={'job_id':'41','training_watch_error':'old failure','other':'retained'}
        added={'job_id':'42','state':'PENDING'}
        merged=watch.merge_refreshed_jobs([old,added],{'41':{'job_id':'41','state':'COMPLETED'}})
        self.assertNotIn('training_watch_error',merged[0])
        self.assertEqual(merged[0]['other'],'retained')
        self.assertEqual(merged[1],added)
        self.assertEqual(watch.refresh_errors(merged),{})

    def test_failed_historical_read_preserves_receipt_and_collects_later_run(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            old={'run':'old','job_id':'41','output':'/raw/old','config':{'driver':'functionality'}}
            new={'run':'new','job_id':'42','output':'/raw/new','config':{'driver':'functionality'}}
            saved=root/'results/old/result.json';saved.parent.mkdir(parents=True)
            saved.write_text('{"status":"PREVIOUS_RECEIPT"}\n');before=saved.read_bytes()
            failing=True
            def remote(command):
                if command.startswith('squeue'):return ''
                if command.startswith('sacct'):
                    jid='41' if ' 41 ' in command else '42'
                    return jid+'|COMPLETED|0:0|88|gres/gpu=1|node\n'
                if '/raw/old/report.json' in command and failing:
                    raise subprocess.CalledProcessError(255,['ssh','fixture'])
                return json.dumps({'status':'CURRENT_RECEIPT'})
            watch.refresh([old,new],{'expected_user':'mock'},remote,study=root)
            self.assertEqual(saved.read_bytes(),before)
            self.assertEqual(json.loads((root/'results/new/result.json').read_text())['status'],'CURRENT_RECEIPT')
            self.assertIn('diagnostic_watch_error',watch.refresh_errors([old,new])['41'])
            failing=False
            watch.refresh([old,new],{'expected_user':'mock'},remote,study=root)
            self.assertEqual(watch.refresh_errors([old,new]),{})
            self.assertEqual(json.loads(saved.read_text())['status'],'CURRENT_RECEIPT')

    def test_accounting_error_is_visible_without_blocking_artifact_read(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            job={'run':'case','job_id':'42','output':'/raw/case','config':{'driver':'functionality'},
                 'state':'COMPLETED','allocated_seconds':'88'}
            def remote(command):
                if command.startswith('squeue'):return ''
                if command.startswith('sacct'):raise subprocess.CalledProcessError(255,['ssh','fixture'])
                return json.dumps({'status':'CURRENT_RECEIPT'})
            watch.refresh([job],{'expected_user':'mock'},remote,study=root)
            self.assertEqual(job['allocated_seconds'],'88')
            self.assertTrue((root/'results/case/result.json').is_file())
            self.assertIn('accounting_watch_error',watch.refresh_errors([job])['42'])

    def test_remote_read_has_connection_and_execution_deadlines(self):
        with mock.patch.object(watch.subprocess,'check_output',return_value=' ok\n') as call:
            self.assertEqual(watch.read_remote({'ssh_host':'fixture'},'readonly'),'ok')
        args,kwargs=call.call_args
        self.assertIn('ServerAliveInterval=15',args[0])
        self.assertIn('ServerAliveCountMax=3',args[0])
        self.assertEqual(kwargs['timeout'],60)

    def test_dsec_report_is_collected_from_actual_driver_filename(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            job=dict(run='dsec-five-events',job_id='42',output='/raw/dsec-five-events',
                     config={'driver':'dsec_transfer'})
            def remote(command):
                if command.startswith('squeue'):return ''
                if command.startswith('sacct'):return '42|COMPLETED|0:0|100|gres/gpu=1|node\n'
                if '/raw/dsec-five-events/report.json' in command:
                    return json.dumps({'status':'COMPLETED_FIXED_FIVE_EVENT_TRANSFER'})
                return ''
            watch.refresh([job],{'expected_user':'mock'},remote,study=root)
            result=json.loads((root/'results/dsec-five-events/result.json').read_text())
            self.assertEqual(result['status'],'COMPLETED_FIXED_FIVE_EVENT_TRANSFER')
            self.assertEqual(job['allocated_seconds'],'100')

    def test_initial_validation_is_retained_labeled_and_partial_line_tolerated(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            rows=[{'update':0,'untrained_reference':True,'mean_scene_epe':8.},
                  {'update':32,'mean_scene_epe':6.}]
            (root/'validation.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows)+'{"update":')
            payload=watch.collect_training(root,1000)
            self.assertEqual([r['update'] for r in payload['validation']['rows']],[0,32])
            self.assertTrue(payload['validation']['rows'][0]['untrained_reference'])
            self.assertEqual(payload['validation']['malformed_or_incomplete_lines'],1)
            # Training must still exclude update0 and malformed records.
            (root/'training.jsonl').write_text(json.dumps({'update':0,'loss':8.})+'\n'+json.dumps({'update':1,'loss':7.})+'\n')
            payload=watch.collect_training(root,1000)
            self.assertEqual([r['update'] for r in payload['training']['rows']],[1])

    def test_stale_attempt_status_is_not_attributed_to_resumed_job(self):
        job={'job_id':'42','submitted_utc':'2026-10-29T00:00:00+00:00'}
        self.assertEqual(watch.current_attempt_status(job,{'status':{'slurm_job_id':'41','status':'PAUSED_STAGE_BOUNDARY'}}),{})
        current={'slurm_job_id':'42','status':'RUNNING'}
        self.assertEqual(watch.current_attempt_status(job,{'status':current}),current)

    def test_sampled_activity_evidence_survives_projection_without_channel_arrays(self):
        row={'update':8,'full_activity_logged':True,'last_microbatch_activity':{
            'sensor_steps':50,'feature_activity':{'4':.2},
            'neuron_activity':{'q:view0':{'mean':.25,'channel_min':0.,'channel_max':.5,
                'temporally_dead_channel_fraction':.01,'temporally_saturated_channel_fraction':0.,
                'channel_rates':[.1,.2,.3]}},
            'neuron_mean_activity_per_time':{'q:view0':[.25]*50}}}
        result=watch.compact_training_row(row)['activity']
        self.assertEqual(result['feature_activity'],{'4':.2})
        self.assertEqual(result['neuron_activity']['q:view0']['temporally_dead_channel_fraction'],.01)
        self.assertNotIn('channel_rates',result['neuron_activity']['q:view0'])
        self.assertEqual(result['neuron_temporal_summary']['q:view0']['nonzero_bins'],50)


if __name__=='__main__':unittest.main()
