"""Metadata-only checks for a fresh, bounded campaign without Test access."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

EXPERIMENT=Path(__file__).resolve().parents[1]/'experiments/spiking-transformer-stereo'
spec=importlib.util.spec_from_file_location('provenance',EXPERIMENT/'scripts/run_provenance.py')
provenance=importlib.util.module_from_spec(spec);spec.loader.exec_module(provenance)


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.registration=json.loads((EXPERIMENT/'campaigns/paired-loss-v1/configs/registration.json').read_text())
        self.config=copy.deepcopy(self.registration['runs'][0]['config']);model=self.registration['frozen_models'][self.config['model']]
        self.source={'source_override':{'parent_source_identity_sha256':model['parent_source_identity_sha256']},'source_files':dict(model['core_source_files'])}
    def validate(self,checkpoint=False):
        provenance.validate_execution_budget(self.config,self.registration,self.registration['manifest_sha256'],self.source,checkpoint)
    def test_registered_fresh_budget_accepts(self):self.validate()
    def test_old_campaign_cap_is_preserved(self):
        provenance.validate_execution_budget({'updates':3000},None,'manifest',{})
        with self.assertRaisesRegex(RuntimeError,'3000'):provenance.validate_execution_budget({'updates':10000},None,'manifest',{})
    def test_registration_required(self):
        with self.assertRaisesRegex(RuntimeError,'registration'):provenance.validate_execution_budget(self.config,None,'manifest',{})
    def test_test_cohort_rejected(self):
        self.config['eval_split']='confirmation'
        with self.assertRaisesRegex(RuntimeError,'configuration'):self.validate()
    def test_budget_schedule_or_seed_changes_rejected(self):
        for key,value in [('updates',10001),('schedule_updates',3000),('seed',9999),('normalize_residual_current',True)]:
            original=self.config[key];self.config[key]=value
            with self.assertRaisesRegex(RuntimeError,'configuration'):self.validate()
            self.config[key]=original
    def test_resume_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'fresh'):self.validate(True)
    def test_core_change_rejected(self):
        key=next(iter(self.source['source_files']));self.source['source_files'][key]='changed'
        with self.assertRaisesRegex(RuntimeError,'core'):self.validate()
    def test_wrong_parent_rejected(self):
        self.source['source_override']['parent_source_identity_sha256']='different'
        with self.assertRaisesRegex(RuntimeError,'parent'):self.validate()


if __name__=='__main__':unittest.main()
