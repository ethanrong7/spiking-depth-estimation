"""Per-study scheduler ownership, entirely mocked; no cluster calls."""
import sys
from pathlib import Path
import re
import subprocess
import unittest
from unittest import mock

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/stereo-spiking-transformer-study/scripts'))
from cluster_control import STUDY_GPU_GATE


class StudyGpuGateTests(unittest.TestCase):
    def run_gate(self,rows,commands,ledger=()):
        def query(argv,**kwargs):
            if argv[0]=='squeue':return rows
            return commands[argv[-1]]
        namespace=dict(p={'user':'mock','ledger_job_ids':list(ledger),'study_source_root':'/study/stereo-study'},
                       subprocess=subprocess,re=re)
        with mock.patch('subprocess.check_output',side_effect=query):
            exec(STUDY_GPU_GATE,namespace)
        return namespace['count']

    def test_sew_does_not_count_but_unregistered_study_source_does(self):
        rows='1|RUNNING|gres/gpu:1\n2|PENDING|gres/gpu:1\n'
        self.assertEqual(self.run_gate(rows,{'1':'Command=/sew/train','2':'Command=/sew/train'}),0)
        with self.assertRaisesRegex(AssertionError,'one GPU'):
            self.run_gate(rows,{'1':'Command=/sew/train','2':'Command=/study/stereo-study/v12/train'})

    def test_ledger_ownership_and_unidentifiable_jobs_fail_closed(self):
        with self.assertRaisesRegex(AssertionError,'one GPU'):
            self.run_gate('1|RUNNING|gres/gpu:1\n',{},ledger=['1'])
        with self.assertRaisesRegex(AssertionError,'Cannot identify'):
            self.run_gate('1|RUNNING|gres/gpu:1\n',{'1':'JobId=1'})


if __name__=='__main__':unittest.main()
