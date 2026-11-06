"""Tiny CPU geometry/median/scene-statistics checks; no neural inference."""
import copy
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core.study_diagnostics import (aggregate_scene_metrics,disparity_metrics,
    fit_train_priors,fixed_val_right_shuffle,gt_support,paired_scene_bootstrap,
    right_view_conditions,shift_right_input,shift_response,shifted_disparity_support)


def manifest():
    return dict(schema='stereo-paired-engineering-pilot-v1',evidence_scope='engineering_only',test=[],
        train=[dict(sample_id='t1',scene_id='train1',split='train'),dict(sample_id='t2',scene_id='train2',split='train')],
        val=[dict(sample_id='v1',scene_id='val1',split='val'),dict(sample_id='v2',scene_id='val2',split='val'),
             dict(sample_id='v3',scene_id='val3',split='val')])


class DiagnosticTests(unittest.TestCase):
    def test_right_inputs_move_content_and_preserve_counts_without_wrap(self):
        right=torch.arange(24,dtype=torch.float32).reshape(2,1,2,6)
        other=right+100
        cases=right_view_conditions(right,other,2)
        torch.testing.assert_close(cases['shifted'][...,2:],right[...,:-2])
        self.assertEqual(int(cases['shifted'][...,:2].sum()),0)
        torch.testing.assert_close(cases['correct'],right)
        torch.testing.assert_close(cases['shuffled'],other)
        self.assertEqual(int(cases['masked'].sum()),0)
        with self.assertRaises(ValueError):shift_right_input(right,1.5)

    def test_positive_disparity_shift_common_support_and_response_sign(self):
        target=torch.full((1,2,8),4.);valid=torch.ones_like(target,dtype=torch.bool)
        # Original xR=xL-4 is in-frame only for xL>=4. Right+2 => d_new=2.
        shifted,common=shifted_disparity_support(target,valid,2)
        self.assertEqual(common.sum().item(),8)
        self.assertFalse(bool(common[...,:4].any()))
        torch.testing.assert_close(shifted[common],torch.full((8,),2.))
        a=disparity_metrics(target,target,common)
        b=disparity_metrics(torch.full_like(target,2.),shifted,common)
        self.assertEqual(a['epe'],0);self.assertEqual(b['epe'],0)
        response=shift_response(target,torch.full_like(target,2.),2,common)
        self.assertEqual(response['response_ratio'],1)
        self.assertEqual(response['positive_response_fraction'],1)
        self.assertTrue(response['majority_positive'])
        self.assertEqual(shift_response(target,target+2,2,common)['response_ratio'],-1)
        # Main metrics retain valid out-of-frame GT; this secondary diagnostic does not.
        self.assertEqual(disparity_metrics(target,target,valid)['valid_pixels'],16)
        low=torch.ones_like(target)
        _,none=shifted_disparity_support(low,valid,2)
        self.assertFalse(bool(none.any()))
        self.assertIsNone(shift_response(low,low,2,none)['response_ratio'])
        trusted=valid.clone();trusted[...,4]=False
        _,limited=shifted_disparity_support(target,valid,2,trusted=trusted)
        self.assertEqual(limited.sum().item(),6)

    def test_train_only_even_medians_pixel_fallback_and_fixed_scene_shuffle(self):
        m=manifest();samples=[]
        for identifier,values in [('t1',[[1.,float('nan')],[3.,float('nan')]]),('t2',[[5.,float('nan')],[7.,float('nan')]])]:
            target=torch.tensor(values)[None]
            samples.append(dict(sample_id=identifier,target=target,valid=torch.isfinite(target),metadata={'split':'train'}))
        priors=fit_train_priors(samples,manifest=m)
        self.assertEqual(priors.constant_median,4.)
        torch.testing.assert_close(priors.predict('pixel'),torch.tensor([[[3.,4.],[5.,4.]]]))
        torch.testing.assert_close(priors.predict('constant'),torch.full((1,2,2),4.))
        self.assertEqual(priors.metadata['per_pixel_fallback_pixels'],2)
        broken=copy.deepcopy(samples);broken[0]['metadata']['split']='val'
        with self.assertRaises(ValueError):fit_train_priors(broken,manifest=m)
        bad_manifest=copy.deepcopy(m);bad_manifest['test']=[{'sample_id':'sealed'}]
        with self.assertRaises(ValueError):fit_train_priors(samples,manifest=bad_manifest)
        mapping=fixed_val_right_shuffle(m,seed=17)
        scenes={r['sample_id']:r['scene_id'] for r in m['val']}
        self.assertTrue(all(scenes[a]!=scenes[b] for a,b in mapping.items()))
        m['val'].reverse()
        self.assertEqual(mapping,fixed_val_right_shuffle(m,seed=17))

    def test_scene_pooling_bootstrap_and_no_prediction_filter(self):
        target=torch.ones((1,2,2));mask=torch.ones_like(target,dtype=torch.bool)
        first=mask.clone();first[0]=torch.tensor([[True,False],[False,False]])
        second=mask.clone();second[0,0,0]=False
        rows=[]
        for sid,pred,valid in [('a',target+2,first),('b',target,second)]:
            rows.append(dict(method='A',condition='correct',scene_id='scene1',sample_id=sid,
                             **disparity_metrics(pred,target,valid)))
        scene=aggregate_scene_metrics(rows)[0]
        self.assertEqual(scene['epe'],.5)
        self.assertEqual(scene['bad1'],.25)
        self.assertEqual(scene['coverage'],.5)
        ci=paired_scene_bootstrap({'s1':2.,'s2':3.,'s3':4.},{'s1':1.,'s2':2.,'s3':3.},draws=200)
        self.assertEqual(ci['descriptive_ci95'],[1.,1.])
        self.assertEqual(ci['training_seeds'],1)
        self.assertEqual(ci['scene_count'],3)
        with self.assertRaises(ValueError):paired_scene_bootstrap({'s1':1.},{'s2':1.})
        bad_prediction=target.clone();bad_prediction[0,0,0]=float('nan')
        with self.assertRaises(ValueError):disparity_metrics(bad_prediction,target,mask)
        # Negative finite predictions remain scored; never silently removed.
        self.assertEqual(disparity_metrics(-target,target,mask)['epe'],2.)

if __name__=='__main__':unittest.main()
