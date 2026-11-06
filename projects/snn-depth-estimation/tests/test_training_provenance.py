"""Metadata-only tests for continuation attribution and rejection."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
MODULE=Path(__file__).resolve().parents[1]/'experiments/spiking-transformer-stereo/scripts/run_provenance.py'
spec=importlib.util.spec_from_file_location('run_provenance',MODULE);provenance=importlib.util.module_from_spec(spec);spec.loader.exec_module(provenance)

class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.data=self.root/'data';self.data.mkdir();self.path=self.root/'last.pt';self.path.write_bytes(b'checkpoint-fixture')
        self.config={'model':'spiking','seed':1234,'normalize_residual_current':True,'lr':.0002,'schedule_updates':3000,'updates':1000}
        self.contract=provenance.training_contract(self.config,'manifest-id','train',self.data)
        self.source={'source_identity_sha256':'source-id','source_files':{'projects/snn-depth-estimation/core/model.py':'core-id'}}
        common={'physical_batch':2,'effective_batch':2,'train_refinements':16,'eval_refinements':32,'hidden_dims':[128]*3,'corr_levels':4,'corr_radius':4,'context_norm':'batch','freeze_batchnorm':True,'optimizer':'AdamW','lr':.0002,'weight_decay':1e-5,'clip_gradient_norm':1.}
        self.meta={'manifest_sha256':'manifest-id','command':['runner','--data-root',str(self.data)],'config':self.config,'source':self.source,'git_protocol':{'common':common,'input':{'bins':50,'height':250,'width':400}}}
        self.ck={'updates':1000,'source':self.source};self.write()
    def tearDown(self):self.temp.cleanup()
    def write(self):(self.root/'metadata.json').write_text(json.dumps(self.meta))
    def validate(self):return provenance.validate_parent_contract(self.path,self.ck,self.contract,self.source)
    def test_legacy_metadata_binds_manifest_and_core(self):self.assertTrue(self.validate()['legacy_contract_validated_from_parent_metadata'])
    def test_manifest_swap_rejected(self):
        self.meta['manifest_sha256']='other';self.write()
        with self.assertRaisesRegex(RuntimeError,'manifest'):self.validate()
    def test_model_semantics_swap_rejected(self):
        self.meta['config']={**self.config,'normalize_residual_current':False};self.write()
        with self.assertRaisesRegex(RuntimeError,'configuration'):self.validate()
    def test_core_swap_rejected(self):
        self.ck['source']=copy.deepcopy(self.source);self.ck['source']['source_files']['projects/snn-depth-estimation/core/model.py']='different'
        with self.assertRaisesRegex(RuntimeError,'core'):self.validate()
    def test_new_contract_rejects_batch_change(self):
        self.ck['training_semantics']={**self.contract,'physical_batch':4}
        with self.assertRaisesRegex(RuntimeError,'semantics'):self.validate()
    def test_source_swap_requires_exact_inherited_best(self):
        self.meta['source']={**self.source,'source_identity_sha256':'different'};self.write()
        with self.assertRaisesRegex(RuntimeError,'inherited-best'):self.validate()
        import hashlib
        (self.root/'inherited-best.json').write_text(json.dumps({'sha256':hashlib.sha256(self.path.read_bytes()).hexdigest()}))
        self.assertEqual(self.validate()['parent_run_source_identity_sha256'],'different')
class ConfirmationTests(unittest.TestCase):
    def setUp(self):
        import hashlib
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.ck=self.root/'best.pt';self.ck.write_bytes(b'frozen-weights');self.prior=self.root/'priors.npz';self.prior.write_bytes(b'training-prior');self.output=self.root/'confirmation';self.selection=self.root/'selection.json'
        rows=[]
        for i in range(10):rows.append({'model':'spiking' if i==0 else 'other'+str(i),'seed':1234,'normalize_residual_current':True,'checkpoint':str(self.ck if i==0 else self.root/str(i)/'best.pt'),'checkpoint_sha256':hashlib.sha256(self.ck.read_bytes()).hexdigest(),'prior_sha256':hashlib.sha256(self.prior.read_bytes()).hexdigest(),'confirmation_output':str(self.output)})
        self.record={'runs':rows,'confirmation_count':80,'primary_api_iters':32,'manifest_sha256':'manifest','preselected_case_ids':['scene_0150-0001','scene_0170-0001','scene_0189-0001']};self.selection.write_text(json.dumps(self.record));self.config={'model':'spiking','seed':1234,'normalize_residual_current':True,'eval_iters':32}
    def tearDown(self):self.temp.cleanup()
    def validate(self):return provenance.validate_confirmation_selection(self.selection,self.ck,self.config,'manifest',self.output)
    def test_bound_selection_accepts(self):self.assertEqual(self.validate()['entry']['model'],'spiking')
    def test_wrong_normalization_rejected(self):
        self.config['normalize_residual_current']=False
        with self.assertRaisesRegex(RuntimeError,'configuration'):self.validate()
    def test_changed_prior_rejected(self):
        self.prior.write_bytes(b'changed')
        with self.assertRaisesRegex(RuntimeError,'prior'):self.validate()
    def test_diagnostic_horizon_cannot_be_primary_test(self):
        self.config['eval_iters']=8
        with self.assertRaisesRegex(RuntimeError,'primary32'):self.validate()
    def test_loaded_checkpoint_matches_selection_claims(self):
        checkpoint={'updates':2800,'source':{'source_identity_sha256':'frozen-source'}}
        entry={'selected_update':2800,'checkpoint_source_identity_sha256':'frozen-source'}
        provenance.validate_loaded_confirmation_checkpoint(checkpoint,entry)
        with self.assertRaisesRegex(RuntimeError,'update'):
            provenance.validate_loaded_confirmation_checkpoint({**checkpoint,'updates':3000},entry)
        with self.assertRaisesRegex(RuntimeError,'source'):
            provenance.validate_loaded_confirmation_checkpoint(checkpoint,{**entry,'checkpoint_source_identity_sha256':'different'})
    def test_frozen_evaluation_configuration_is_exact(self):
        self.config['updates']=0
        self.record['runs'][0]['evaluation_config_identity']={'config':dict(self.config)}
        self.selection.write_text(json.dumps(self.record));self.validate()
        self.config['updates']=1
        with self.assertRaisesRegex(RuntimeError,'frozen evaluation'):self.validate()
if __name__=='__main__':unittest.main()
