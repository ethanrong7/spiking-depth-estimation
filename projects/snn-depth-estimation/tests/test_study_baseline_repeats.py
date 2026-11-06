"""Tiny synthetic CPU checks; no trained model, real arrays or performance experiment."""
import copy
import json
from pathlib import Path
import sys
import unittest
import torch

S=Path(__file__).resolve().parents[1]/'experiments/stereo-spiking-transformer-study'
sys.path.insert(0,str(S/'scripts'))
import diagnose_baseline_repeats as driver

class RepeatTests(unittest.TestCase):
    def test_actual_configs_and_exact_prospective_contract(self):
        for modality in ('spike','event'):
            c=json.loads((S/f'configs/baseline-repeats-original-I-{modality}-v1.json').read_text())
            self.assertTrue(driver.validate_config(c))
            for key,value in [('repeat_protocol',{**c['repeat_protocol'],'api_warmup_forwards':1}),
                              ('sample_ordinals',[0]),('training_source_snapshot','newer'),('max_seconds',181),
                              ('registered',False)]:
                broken=copy.deepcopy(c);broken[key]=value
                with self.assertRaises(ValueError):driver.validate_config(broken)

    def test_cold_difference_retained_full_outputs_and_three_calls_only(self):
        class Tiny(torch.nn.Module):
            def __init__(self):super().__init__();self.calls=0
            def forward(self,l,r):
                self.calls+=1
                p=l.mean(1)+(2 if self.calls==1 else 0)
                return {'disparity':p,'auxiliary':[p+1],'diagnostics':{'calls':self.calls}}
        model=Tiny();x=torch.ones(1,2,1,2,2)
        primary,evidence=driver.repeat_target(model,x,x)
        self.assertEqual(model.calls,3)
        self.assertEqual(evidence['comparisons']['first_vs_second']['maximum_absolute_difference'],2)
        self.assertTrue(evidence['comparisons']['second_vs_repeat']['exact_equal'])
        self.assertEqual(len(evidence['comparisons']['first_vs_second']['prediction_tensors']),2)
        d=driver.pointwise_difference(primary[0],primary[1],torch.ones(1,2,2,dtype=torch.bool))
        self.assertEqual(d['all_pixels']['changed_pixels'],4)
        self.assertEqual(d['valid_GT_pixels']['mean_absolute_difference'],2)

    def test_stochastic_rng_poststate_matches_without_claiming_zero_consumption(self):
        class Tiny(torch.nn.Module):
            def forward(self,l,r):return {'disparity':l.mean(1)*torch.rand_like(l[:,0])}
        x=torch.ones(1,2,1,2,2)
        _,e=driver.repeat_target(Tiny(),x,x)
        self.assertTrue(e['post_rng_matches_between_calls'])
        self.assertTrue(e['rng_consumption_occurred'])
        self.assertTrue(all(v['exact_equal'] for v in e['comparisons'].values()))

    def test_input_and_weight_mutation_rejected(self):
        class InputMutation(torch.nn.Module):
            def forward(self,l,r):l.add_(1);return {'disparity':l.mean(1)}
        class BufferMutation(torch.nn.Module):
            def __init__(self):super().__init__();self.register_buffer('value',torch.zeros(1))
            def forward(self,l,r):self.value.add_(1);return {'disparity':l.mean(1)}
        x=torch.ones(1,2,1,2,2)
        for model in (InputMutation(),BufferMutation()):
            with self.assertRaisesRegex(ValueError,'Input or trained'):driver.repeat_target(model,x,x)

    def test_later_numerical_instability_is_outcome_not_dropped(self):
        class Tiny(torch.nn.Module):
            def __init__(self):super().__init__();self.calls=0
            def forward(self,l,r):self.calls+=1;return {'disparity':l.mean(1)+self.calls}
        x=torch.ones(1,2,1,2,2)
        _,e=driver.repeat_target(Tiny(),x,x)
        self.assertFalse(e['comparisons']['second_vs_repeat']['within_tolerance'])
        self.assertEqual(e['comparisons']['second_vs_repeat']['maximum_absolute_difference'],1)

    def test_pixel_within_scene_then_equal_scene_aggregation(self):
        from core.study_diagnostics import disparity_metrics
        rows=[]
        for scene,values in [('a',[(1,2),(3,6)]),('b',[(1,10),(1,10)])]:
            for i,(n,error) in enumerate(values):
                target=torch.ones(1,1,n);valid=torch.ones_like(target,dtype=torch.bool)
                for name,scale in [('first',1),('second',2),('repeat',2)]:
                    rows.append(dict(method='tiny',condition=name,scene_id=scene,sample_id=scene+str(i),
                        **disparity_metrics(target+error*scale,target,valid)))
        result=driver.aggregate(rows)
        self.assertEqual(result['equal_scene_epe']['first'],7.5)
        self.assertEqual(result['equal_scene_second_minus_first'],7.5)
        self.assertEqual(result['equal_scene_repeat_minus_second'],0)

    def test_real_slurm_route_executes_only_fresh_driver_process(self):
        import os
        import subprocess
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); bin_dir=root/'bin';bin_dir.mkdir()
            for name,body in [('id','echo rongliang'),('hostname','echo cccl3'),
                              ('python','printf "%s\\n" "$@"')]:
                executable=bin_dir/name
                executable.write_text('#!/bin/sh\n'+body+'\n');executable.chmod(0o755)
            env={**os.environ,'PATH':str(bin_dir)+':'+os.environ['PATH'],
                 'STUDY_SOURCE':'/frozen','STUDY_ENV':str(root),'STUDY_CONFIG':'/config',
                 'STUDY_OUTPUT':'/new-output','STUDY_NATIVE_SOURCE':'/native','STUDY_NATIVE_DERIVED':'/derived',
                 'STUDY_SOURCE_REGISTRY':'/frozen/source-registry.json','STUDY_DRIVER':'baseline_repeats',
                 'SLURM_JOB_ID':'tiny-mock','CUDA_VISIBLE_DEVICES':'0'}
            result=subprocess.run(['bash',str(S/'slurm/stage.sbatch')],env=env,capture_output=True,text=True,check=True)
            self.assertEqual(result.stdout.splitlines(),[
                '/frozen/projects/snn-depth-estimation/experiments/stereo-spiking-transformer-study/scripts/diagnose_baseline_repeats.py',
                '--config','/config','--source-registry','/frozen/source-registry.json','--output','/new-output'])
    def test_independent_saved_map_readback_and_tampering(self):
        import numpy as np
        from verify_baseline_repeat_maps import verify_maps,validate_config
        config=json.loads((S/'configs/baseline-repeat-map-readback-v1.json').read_text());validate_config(config)
        arrays={name:np.full((1,2,2),value,dtype=np.float32) for name,value in
                [('target',1),('first',2),('second',3),('repeat',3)]}
        arrays['valid']=np.ones((1,2,2),dtype=np.bool_)
        metrics={name:dict(absolute_error_sum=error*4,epe=error,valid_pixels=4,
                          bad1_pixels=4 if error>1 else 0,bad3_pixels=0,bad5_pixels=0)
                 for name,error in [('first',1),('second',2),('repeat',2)]}
        evidence={key:{scope:dict(pixels=4,changed_pixels=4 if value else 0,
                    maximum_absolute_difference=value,mean_absolute_difference=value)
                    for scope in ('all_pixels','valid_GT_pixels')}
                    for key,value in [('primary_first_vs_second',1),('primary_second_vs_repeat',0)]}
        self.assertTrue(verify_maps(arrays,metrics,evidence))
        arrays['first'][0,0,0]=10
        with self.assertRaisesRegex(ValueError,'arithmetic disagrees'):verify_maps(arrays,metrics,evidence)


if __name__=='__main__':unittest.main()
