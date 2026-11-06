"""Tiny scene-vector/compact-receipt controls; no arrays/model/data experiment."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest
import tempfile
from unittest import mock

import numpy as np

STUDY=Path(__file__).resolve().parents[1]/'experiments/stereo-spiking-transformer-study'
SPEC=importlib.util.spec_from_file_location('pilot_comparison_controls',STUDY/'scripts/summarize_pilot_comparisons.py')
D=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(D)


class Comparisons(unittest.TestCase):
    def test_single_captured_bytes_bind_json_and_hash_despite_later_replacement(self):
        original=b'{"run":"completed","state":"COMPLETED"}\n'
        later=b'{"run":"live","state":"RUNNING"}\n'
        with tempfile.TemporaryDirectory() as raw:
            path=Path(raw)/'jobs.json';path.write_bytes(original)
            read_bytes=Path.read_bytes
            def replace_after_read(source):
                data=read_bytes(source)
                if source == path:path.write_bytes(later)
                return data
            with mock.patch.object(Path,'read_bytes',autospec=True,side_effect=replace_after_read) as captured:
                value,receipt,count=D.captured_json(path)
            captured.assert_called_once_with(path)
            self.assertEqual(value,{'run':'completed','state':'COMPLETED'})
            self.assertEqual(receipt,hashlib.sha256(original).hexdigest());self.assertEqual(count,len(original))
            self.assertEqual(json.loads(path.read_bytes()),{'run':'live','state':'RUNNING'})
            self.assertNotEqual(receipt,hashlib.sha256(path.read_bytes()).hexdigest())

    def test_ratio_of_means_not_mean_scene_ratios(self):
        indices=np.array([[0,1],[1,1],[0,0]])
        estimate,draws=D.paired_ratio_draws([1,9],[1,3],indices)
        self.assertEqual(estimate,2.5)
        self.assertNotEqual(estimate,np.mean(np.array([1,9])/np.array([1,3])))
        np.testing.assert_allclose(draws,[2.5,3,1])

    def test_paired_scene_identity_and_joint_four_arm_cancellation(self):
        indices=D.samples(scenes=4)
        a=np.array([1,10,3,20.]);b=np.array([5,2,4,6.])
        ratio,draws=D.paired_ratio_draws(2*b,b,indices)
        self.assertEqual(ratio,2.);np.testing.assert_array_equal(draws,np.full(2000,2.))
        estimate,interaction=D.joint_interaction_draws(a,b,a,b,indices)
        self.assertEqual(estimate,0.);np.testing.assert_array_equal(interaction,np.zeros(2000))
        # Independent event sampling breaks cancellation and invents variation.
        _,spike=D.paired_ratio_draws(a,b,indices)
        _,event=D.paired_ratio_draws(a,b,indices[::-1])
        self.assertGreater(np.ptp(np.log(spike)-np.log(event)),.1)

    def test_interaction_logs_follow_means_not_scene_log_ratios(self):
        indices=np.array([[0,1],[1,1],[0,0]])
        estimate,draws=D.joint_interaction_draws([1,9],[1,3],[2,4],[1,5],indices)
        expected=np.log(10/4)-np.log(6/6)
        wrong=np.mean(np.log(np.array([1,9])/np.array([1,3]))-np.log(np.array([2,4])/np.array([1,5])))
        self.assertAlmostEqual(estimate,expected);self.assertNotAlmostEqual(estimate,wrong)
        self.assertAlmostEqual(draws[0],expected)

    def test_bootstrap_draws_reproducible_and_invalid_values_rejected(self):
        np.testing.assert_array_equal(D.samples(),D.samples())
        indices=np.array([[0,1],[1,0]])
        for values in ([1,float('nan')],[1,float('inf')],[-1,2]):
            with self.assertRaisesRegex(ValueError,'Invalid'):D.paired_ratio_draws(values,[1,2],indices)
        with self.assertRaisesRegex(ValueError,'denominator'):D.paired_ratio_draws([1,2],[0,0],indices)
        with self.assertRaisesRegex(ValueError,'positive arm'):D.joint_interaction_draws([0,0],[1,2],[1,2],[1,2],indices)
        with self.assertRaisesRegex(ValueError,'mismatch'):D.paired_ratio_draws([1],[1,2],indices)

    def tiny_population(self):
        expected={'scene0':['scene0/a','scene0/b'],'scene1':['scene1/a','scene1/b']}
        frames=[];scenes=[]
        for i,(scene,ids) in enumerate(expected.items()):
            for condition in D.CONDITIONS:
                epe=float(i+1)
                for sid in ids:
                    frames.append(dict(method='checkpoint',condition=condition,sample_id=sid,scene_id=scene,
                        valid_pixels=32400,total_pixels=32400,epe=epe,absolute_error_sum=epe*32400,coverage=1.,bad1=.2,bad3=.1,bad5=.05))
                scenes.append(dict(method='checkpoint',condition=condition,scene_id=scene,sample_ids=ids,frames=2,
                    valid_pixels=64800,total_pixels=64800,epe=epe,coverage=1.,bad1=.2,bad3=.1,bad5=.05))
        return expected,frames,scenes

    def test_unequal_population_duplicate_scene_and_support_rejected(self):
        expected,frames,scenes=self.tiny_population()
        groups,support=D.population(scenes,frames,expected)
        self.assertEqual(list(groups['correct']),['scene0','scene1']);self.assertEqual(len(support),4)
        altered=copy.deepcopy(scenes);altered[1]=copy.deepcopy(altered[0])
        with self.assertRaisesRegex(ValueError,'scene population'):D.population(altered,frames,expected)
        altered=copy.deepcopy(frames);altered[0]['sample_id']='other/a'
        with self.assertRaisesRegex(ValueError,'populations'):D.population(scenes,altered,expected)
        altered=copy.deepcopy(frames);altered[0]['scene_id']='scene1'
        with self.assertRaisesRegex(ValueError,'membership'):D.population(scenes,altered,expected)
        altered=copy.deepcopy(scenes);altered[0]['sample_ids']=['scene0/a','scene0/a']
        with self.assertRaisesRegex(ValueError,'population'):D.population(altered,frames,expected)
        altered=copy.deepcopy(frames);altered[0]['epe']=float('nan')
        with self.assertRaisesRegex(ValueError,'Finite'):D.population(scenes,altered,expected)

    def test_all_ten_cells_preserved_missing_numbers_null_and_no_interaction(self):
        cells=[]
        for model in D.MODEL_NAMES:
            for sensor in ('spike','event'):
                measured=model=='original-I' or (model in ('A','B') and sensor=='spike')
                cell={'model':model,'sensor':sensor,'run':model+'-'+sensor,
                      'status':'MEASURED_COMPLETED_RECEIPT' if measured else 'PENDING_TRAINING'}
                if measured:cell['validated']={'values':{'correct':np.ones(10)*(1 if model=='original-I' else 2)}}
                cells.append(cell)
        rows,interactions,counterparts=D.build_comparisons(cells,D.samples())
        self.assertEqual(len(rows),10);self.assertEqual(sum(x['mean_scene_epe'] is None for x in rows),6)
        self.assertEqual(len(interactions),4);self.assertTrue(all(x['interaction'] is None for x in interactions))
        self.assertEqual(len(counterparts),4);self.assertTrue(all(x['ratio'] is None for x in counterparts))
        self.assertTrue(all(x['ratio_to_ALIF_I'] is None for x in rows if x['status']!='MEASURED_COMPLETED_RECEIPT'))

    def test_complete_counterparts_and_interaction_use_registered_practical_bands(self):
        cells=[]
        for model in D.MODEL_NAMES:
            for sensor in ('spike','event'):
                value=1. if model=='original-I' else 1.1 if sensor=='event' else 1.
                cells.append({'model':model,'sensor':sensor,'run':model+'-'+sensor,'status':'MEASURED_COMPLETED_RECEIPT',
                    'validated':{'values':{'correct':np.full(10,value)}}})
        _,interactions,counterparts=D.build_comparisons(cells,D.samples())
        self.assertAlmostEqual(interactions[0]['interaction']['estimate'],-np.log(1.1))
        self.assertEqual(interactions[0]['practical_point_flag'],'spike_direction_beyond_5pct')
        np.testing.assert_allclose(interactions[0]['practical_band_ratio'],[1/1.05,1.05])
        self.assertTrue(all(x['practical_point_within_10pct_counterpart_margin'] for x in counterparts))

    def test_existing_partial_receipt_and_changed_checkpoint_identity_rejected(self):
        # Existing compact receipt is a schema fixture; no saved pixel assets.
        report=json.loads((STUDY/'results/functionality-pilot-A-spike-seed9471-v1/result.json').read_text())
        queue=json.loads((STUDY/'configs/pilot-first-seed-queue-v1.json').read_text())
        cell=next(x for x in queue['runs'] if x['run']=='pilot-A-spike-seed9471-v1')
        config_path=STUDY/'configs'/cell['config'];config=json.loads(config_path.read_text())
        summary=json.loads((STUDY/'results'/cell['run']/'training-summary.json').read_text())
        jobs=json.loads((STUDY/'results/jobs.json').read_text())
        expected=D.expected_population(json.loads((STUDY/'configs/causal-pilot-scene-selection-v1.json').read_text()))
        def validate(r=report,s=summary):return D.validate_receipt(r,cell,config,D.sha(config_path),s,jobs,expected)
        self.assertEqual(len(validate()['support']),20)
        partial=copy.deepcopy(report);partial['status']='RUNNING'
        with self.assertRaisesRegex(ValueError,'Partial'):validate(partial)
        partial=copy.deepcopy(summary);partial['attempt_status']['update']=500
        with self.assertRaisesRegex(ValueError,'Partial training'):validate(s=partial)
        changed=copy.deepcopy(report);changed['checkpoint']['binding']['source_identity_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'v12'):validate(changed)
        changed=copy.deepcopy(report);changed['checkpoint']['validation_selection']['trained_evaluations'].pop()
        with self.assertRaisesRegex(ValueError,'selection'):validate(changed)


if __name__=='__main__':unittest.main()
