"""Tiny CPU feature-statistic/parity fixtures; no real model/data experiment."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock

import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
SCRIPT=ROOT/'experiments/stereo-spiking-transformer-study/scripts/feature_information.py'
spec=importlib.util.spec_from_file_location('feature_information',SCRIPT)
probe_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe_module)
from core.study_models import TemporalNeuron


class TinyClock(nn.Module):
    def __init__(self,style='clock',fail=False):
        super().__init__();self.style=style;self.fail=fail
        self.stages=nn.ModuleList([nn.ModuleDict({'neuron':TemporalNeuron(True,beta=0.)})])
        self.blocks=nn.ModuleList([nn.ModuleDict(),nn.ModuleDict({'gate':TemporalNeuron(True,beta=0.)})])
        self.weight=nn.Parameter(torch.tensor(.2));self.dropout=nn.Dropout(.25)
        for index,module in enumerate(m for m in self.modules() if isinstance(m,TemporalNeuron)):
            module.state_id=index

    def forward(self,x):
        previous={};result=x.new_zeros(())
        yy,xx=torch.meshgrid(torch.arange(8),torch.arange(8),indexing='ij')
        support=(yy<3)&(xx<4)
        for step in range(50):
            updated={}
            for view in (0,1):
                pattern=(torch.full((8,8),float(step%2)) if self.style=='clock' else ((xx+yy)%2).float())
                current=torch.where(support,pattern*2,torch.full((8,8),2.))[None,None]
                current=current+x[:,step].mean((1,2,3),keepdim=True)+self.weight*0
                z=self.stages[0]['neuron'](current,previous,updated,view)
                gate_current=torch.full((1,2,2,1,1),float(step%2)*2)
                gate=self.blocks[1]['gate'](gate_current,previous,updated,view)
                result=result+self.dropout(z).sum()+gate.sum()*0
            previous=updated
            if self.fail and step==3:raise RuntimeError('fixture forward failure')
        return dict(output=result,state=previous)


class FeatureInformationTests(unittest.TestCase):
    def run_probe(self,style):
        model=TinyClock(style)
        with probe_module.TemporalFeatureProbe(model,(5,7),require_cuda=False) as probe:
            # These operations would synchronize/extract device values. None may
            # occur in the temporal hooks; serialize follows the full forward.
            with mock.patch.object(torch.Tensor,'cpu',side_effect=AssertionError('per-step cpu')), \
                 mock.patch.object(torch.Tensor,'item',side_effect=AssertionError('per-step item')):
                model(torch.zeros((1,50,1,5,7)))
        return probe.serialize()

    def test_uniform_spatial_clock_and_varying_activity_have_same_healthy_looking_mean(self):
        clock=self.run_probe('clock');varying=self.run_probe('spatial')
        a=next(r for r in clock['nodes'] if r['role']=='spatial_feature' and r['view']==0)
        b=next(r for r in varying['nodes'] if r['role']=='spatial_feature' and r['view']==0)
        self.assertEqual(a['activation_mean'],.5);self.assertEqual(b['activation_mean'],.5)
        self.assertEqual(a['mean_within_spatial_channel_variance'],0.)
        self.assertEqual(a['spatially_constant_channel_fraction'],1.)
        self.assertEqual(a['spatially_constant_channel_time_fraction'],1.)
        self.assertEqual(a['temporal_variance_of_channel_spatial_means'],.25)
        self.assertEqual(b['mean_within_spatial_channel_variance'],.25)
        self.assertEqual(b['spatially_constant_channel_fraction'],0.)
        self.assertEqual(b['temporal_variance_of_channel_spatial_means'],0.)
        self.assertEqual(a['calls'],50);self.assertEqual(a['valid_spatial_shape'],[3,4])
        self.assertEqual(clock['padded_shape'],[16,16])
        for gate in (r for r in clock['nodes'] if r['role']=='global_channel_gate'):
            self.assertFalse(gate['spatial_variance_assessable'])
            self.assertIsNone(gate['spatially_constant_channel_fraction'])
            self.assertEqual(gate['temporal_variance_of_channel_spatial_means'],.25)
            self.assertEqual(gate['calls'],50)

    def test_exact_output_input_RNG_explicit_state_parameter_and_gradient_parity(self):
        model=TinyClock('spatial').train();before={k:v.clone() for k,v in model.state_dict().items()}
        first=torch.zeros((1,50,1,5,7),requires_grad=True)
        torch.manual_seed(394);plain=model(first)
        plain_grad=torch.autograd.grad(plain['output'],first)[0];rng=torch.get_rng_state().clone()
        second=first.detach().clone().requires_grad_();unchanged=second.detach().clone();version=second._version
        torch.manual_seed(394)
        with probe_module.TemporalFeatureProbe(model,(5,7),require_cuda=False) as probe:
            observed=model(second)
        probe.serialize()
        grad=torch.autograd.grad(observed['output'],second)[0]
        self.assertTrue(torch.equal(observed['output'],plain['output']))
        self.assertTrue(torch.equal(plain_grad,grad));self.assertTrue(torch.equal(torch.get_rng_state(),rng))
        self.assertTrue(torch.equal(second,unchanged));self.assertEqual(second._version,version)
        self.assertEqual(set(plain['state']),set(observed['state']))
        for key in plain['state']:self.assertTrue(torch.equal(plain['state'][key],observed['state'][key]))
        for key,value in model.state_dict().items():self.assertTrue(torch.equal(value,before[key]))
        self.assertTrue(all(not m._forward_hooks for m in model.modules()))

    def test_hooks_removed_on_failure_and_incomplete_time_not_reported_as_full(self):
        model=TinyClock(fail=True)
        probe=probe_module.TemporalFeatureProbe(model,(5,7),require_cuda=False)
        with self.assertRaises(RuntimeError):
            with probe:model(torch.zeros((1,50,1,5,7)))
        self.assertTrue(all(not m._forward_hooks for m in model.modules()))
        with self.assertRaises(ValueError):probe.serialize()
        with self.assertRaises(ValueError):probe_module.TemporalFeatureProbe(nn.Linear(1,1),(5,7))

    def test_matcher_padding_and_exact_main_matcher_residual_geometry(self):
        confidence={}
        for stride in (4,8,16):
            shape=(16//stride,16//stride)
            ext,_=probe_module.valid_extent((5,7),stride)
            maximum=torch.full((1,1,*shape),.99);entropy=torch.full_like(maximum,.01)
            maximum[...,:ext[0],:ext[1]]=.2;entropy[...,:ext[0],:ext[1]]=.8
            confidence[stride]=dict(max_probability=maximum,normalized_entropy=entropy)
        output=dict(disparity=torch.full((1,1,5,7),2.),aux_disparities_native={4:torch.full((1,1,4,4),5.)},
                    match_confidence_native=confidence)
        summary=probe_module.output_information(output,(5,7))
        branches=summary['disparity_branches']
        self.assertEqual(branches['main']['spatial_mean'],2.)
        self.assertEqual(branches['matcher_upsampled_aux4']['spatial_mean'],5.)
        self.assertEqual(branches['residual_main_minus_matcher']['spatial_mean'],-3.)
        self.assertEqual(branches['residual_main_minus_matcher']['spatial_std'],0.)
        for stride in (4,8,16):
            self.assertAlmostEqual(summary['matcher'][str(stride)+'/max_probability']['spatial_mean'],.2,places=7)
            self.assertAlmostEqual(summary['matcher'][str(stride)+'/normalized_entropy']['spatial_mean'],.8,places=7)


if __name__=='__main__':unittest.main()
