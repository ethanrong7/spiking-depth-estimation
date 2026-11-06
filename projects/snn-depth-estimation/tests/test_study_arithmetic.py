"""Tiny CPU arithmetic formulas and observation parity; no model experiment."""
import importlib.util
from pathlib import Path
import sys
import unittest

import torch
from torch import nn
from torch.nn import functional as F

P=Path(__file__).resolve().parents[1]/'experiments/stereo-spiking-transformer-study/scripts/arithmetic_counter.py'
spec=importlib.util.spec_from_file_location('study_arithmetic_test',P)
counter=importlib.util.module_from_spec(spec);spec.loader.exec_module(counter)

@torch.jit.script
def tiny_script_reset(v:torch.Tensor,s:torch.Tensor):
    return v-s*2


class ArithmeticTests(unittest.TestCase):
    def test_padded_binary_convolution_counts_exact_valid_fanout(self):
        x=torch.ones(1,1,3,3);w=torch.ones(2,1,3,3)
        y=F.conv2d(x,w,padding=1)
        c=counter.convolution_counts((x,w,None,[1,1],[1,1],[1,1],False,[0,0],1),y)
        self.assertEqual(c['dense_mac'],162)
        self.assertEqual(c['potential_event_ac'],98) # (2+3+2)^2 ×2outputs
        self.assertEqual(c['analog_mac'],0)

    def test_grouped_stride_and_nonbinary_inputs(self):
        x=torch.ones(1,2,3,3);w=torch.ones(4,1,3,3)
        y=F.conv2d(x,w,stride=2,padding=1,groups=2)
        c=counter.convolution_counts((x,w,None,[2,2],[1,1],[1,1],False,[0,0],2),y)
        self.assertEqual(c['dense_mac'],144);self.assertEqual(c['potential_event_ac'],64)
        c=counter.convolution_counts((x*2,w,None,[2,2],[1,1],[1,1],False,[0,0],2),y)
        self.assertEqual(c['analog_mac'],144);self.assertEqual(c['potential_event_ac'],0)

    def test_transposed_convolution_crop_fanout(self):
        x=torch.ones(1,1,2,2);w=torch.ones(1,2,3,3)
        y=F.conv_transpose2d(x,w,stride=2,padding=1)
        c=counter.convolution_counts((x,w,None,[2,2],[1,1],[1,1],True,[0,0],1),y)
        self.assertEqual(c['dense_mac'],72);self.assertEqual(c['potential_event_ac'],32)

    def test_sparse_dilated_grouped_and_output_padding_match_direct_sums(self):
        x=torch.tensor([[[[0.,1,0,1],[1,0,1,0],[0,0,1,1]],
                         [[1.,1,0,0],[0,1,0,1],[1,0,1,0]]]])
        for transposed in (False,True):
            weight=torch.ones((2,2,3,2) if transposed else (4,1,3,2))
            stride,padding,dilation=[2,2],[2,1],[2,1]
            output_padding=[1,1] if transposed else [0,0]
            if transposed:
                y=F.conv_transpose2d(x,weight,stride=stride,padding=padding,
                    output_padding=output_padding,groups=2,dilation=dilation)
            else:
                y=F.conv2d(x,weight,stride=stride,padding=padding,groups=2,dilation=dilation)
            counts=counter.convolution_counts((x,weight,None,stride,padding,dilation,
                transposed,output_padding,2),y)
            self.assertEqual(counts['potential_event_ac'],int(y.sum()))

    def test_matrix_binary_tests_and_exact_coincidences(self):
        a=torch.tensor([[1.,0,1],[0,1,0]]);b=torch.tensor([[1.,0],[1,1],[0,1]])
        c=counter.matrix_counts(a,b,a@b)
        self.assertEqual(c['dense_mac'],12);self.assertEqual(c['binary_product_tests'],12)
        self.assertEqual(c['potential_event_ac'],int((a@b).sum()))
        c=counter.matrix_counts(a,b+.5,a@(b+.5))
        self.assertEqual(c['potential_event_ac'],6);self.assertEqual(c['binary_product_tests'],0)

    def test_broadcast_matrix_counts(self):
        a=torch.ones(2,1,3,4);b=torch.ones(1,5,4,2)
        c=counter.matrix_counts(a,b,a@b)
        self.assertEqual(c['dense_mac'],240);self.assertEqual(c['potential_event_ac'],240)

    def test_addmm_nonunit_alpha_beta_are_explicit_scalar_products(self):
        a=torch.full((2,3),.5);b=torch.full((3,2),.5);bias=torch.ones(2)
        watch=counter.ArithmeticCounter()
        with watch:
            output=torch.addmm(bias,a,b,alpha=3,beta=2)
        torch.testing.assert_close(output,torch.full((2,2),4.25))
        counts=watch.report()['totals']
        self.assertEqual(counts['dense_mac'],12)
        self.assertEqual(counts['explicit_bias_add'],4)
        self.assertEqual(counts['element_mul'],8)

    def test_mode_does_not_change_tiny_forward_and_state(self):
        torch.manual_seed(3)
        model=nn.Sequential(nn.Conv2d(1,2,3,padding=1),nn.GroupNorm(1,2),nn.Sigmoid()).eval()
        x=torch.tensor([[[[0.,1],[1,0]]]])
        expected=model(x);state={k:v.clone() for k,v in model.state_dict().items()}
        watch=counter.ArithmeticCounter(model)
        try:
            with watch:
                actual=model(x)
        finally:watch.close()
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        for k,v in model.state_dict().items():torch.testing.assert_close(v,state[k],rtol=0,atol=0)
        report=watch.report();self.assertEqual(report['totals']['dense_mac'],72)
        self.assertIn('0',report['components']);self.assertGreater(report['totals']['logical_input_bytes'],0)
        self.assertEqual(len(watch.handles),0)

    def test_full_api_observer_matches_rng_and_does_not_mutate_inputs(self):
        sys.path.insert(0,str(P.parent))
        try:
            import profile_arithmetic
            class TinyAPI(nn.Module):
                def forward(self,left,right):
                    return {'disparity':(left.mean(1)+right.mean(1))*torch.rand_like(left[:,0])}
            left=torch.ones(1,2,1,2,2);right=torch.zeros_like(left)
            ledger,parity=profile_arithmetic.observe(TinyAPI(),left,right)
            self.assertTrue(parity['input_tensors_unchanged'])
            self.assertTrue(parity['python_numpy_torch_cuda_rng_match_reference'])
            self.assertEqual(parity['maximum_absolute_difference'],0)
            self.assertGreater(ledger['totals']['calls'],0)
        finally:sys.path.pop(0)

    def test_repeat_failure_preserves_audit_and_releases_hooks(self):
        sys.path.insert(0,str(P.parent))
        try:
            import profile_arithmetic
            class UnstableAPI(nn.Module):
                def __init__(self):
                    super().__init__();self.calls=0
                def forward(self,left,right):
                    self.calls+=1
                    return {'disparity':left.mean((1,2),keepdim=False).unsqueeze(1)+self.calls}
            model=UnstableAPI();left=torch.ones(1,2,1,2,2)
            with self.assertRaises(profile_arithmetic.ArithmeticParityError) as caught:
                profile_arithmetic.observe(model,left,left)
            self.assertFalse(caught.exception.audit['reference_repeat_allclose'])
            self.assertEqual(caught.exception.audit['reference_repeat_maximum_absolute_difference'],1)
            self.assertGreater(caught.exception.ledger['totals']['calls'],0)
            self.assertEqual(len(model._forward_hooks),0)
            self.assertEqual(len(model._forward_pre_hooks),0)
        finally:sys.path.pop(0)

    def test_high_level_convolution_boundaries_use_schema_without_double_count(self):
        x=torch.ones(1,2,3,4)
        cases=[(torch.ops.aten.conv2d.default,torch.ones(4,1,2,2),
                {'stride':[2,1],'padding':[1,0],'dilation':[2,1],'groups':2}),
               (torch.ops.aten.conv_transpose2d.input,torch.ones(2,2,2,2),
                {'stride':[2,2],'padding':[1,0],'output_padding':[1,1],'dilation':[2,1],'groups':2})]
        for op,weight,kwargs in cases:
            watch=counter.ArithmeticCounter()
            output=watch.__torch_dispatch__(op,(),(x,weight),kwargs)
            report=watch.report()
            self.assertEqual(set(report['operators']),{str(op)})
            self.assertEqual(report['totals']['calls'],1)
            self.assertEqual(report['totals']['potential_event_ac'],int(output.sum()))
            self.assertGreater(report['totals']['dense_mac'],0)
            self.assertEqual(report['unclassified_operators'],{})

    def test_pinned_einsum_counts_all_row_pairs_and_rejects_other_equations(self):
        a=(torch.arange(24).reshape(2,3,2,2)%2).float()
        b=(torch.arange(48).reshape(2,3,2,4)%3==0).float()
        op=torch.ops.aten.einsum.default;watch=counter.ArithmeticCounter()
        output=watch.__torch_dispatch__(op,(),('aijk,aijh->ajkh',[a,b]),{})
        counts=watch.report()['totals']
        self.assertEqual(counts['dense_mac'],output.numel()*3)
        self.assertEqual(counts['binary_product_tests'],output.numel()*3)
        self.assertEqual(counts['potential_event_ac'],int(output.sum()))
        unsupported=torch.einsum('aijk,aijk->aij',a,a)
        self.assertIsNone(counter.operator_counts(op,('aijk,aijk->aij',[a,a]),{},unsupported))

    def test_actual_high_level_composites_and_inplace_activation_keep_vectors(self):
        x=torch.arange(16.).reshape(1,2,2,4)
        cases=[(torch.ops.aten.group_norm.default,(x,1),{}),
               (torch.ops.aten.instance_norm.default,(x,None,None,None,None,True,.1,1e-5,False),{}),
               (torch.ops.aten.batch_norm.default,(x,None,None,torch.zeros(2),torch.ones(2),False,.1,1e-5,False),{}),
               (torch.ops.aten.grid_sampler.default,(x,torch.zeros(1,2,3,2),0,0,False),{}),
               (torch.ops.aten.relu_.default,(x.clone(),),{})]
        for op,args,kwargs in cases:
            watch=counter.ArithmeticCounter();output=watch.__torch_dispatch__(op,(),args,kwargs)
            report=watch.report();self.assertEqual(report['unclassified_operators'],{})
            name=str(op).split('.')[1]
            key='element_relu_' if name=='relu_' else name+'_output_elements'
            self.assertEqual(report['totals'][key],output.numel())

    def test_auxiliary_only_disagreement_fails_full_output_gate(self):
        sys.path.insert(0,str(P.parent))
        try:
            import profile_arithmetic
            class AuxUnstable(nn.Module):
                def __init__(self):super().__init__();self.calls=0
                def forward(self,left,right):
                    self.calls+=1
                    y=left.mean((1,2)).unsqueeze(1)
                    return {'disparity':y,'refinements':[y+self.calls]}
            model=AuxUnstable();x=torch.ones(1,2,1,2,2)
            with self.assertRaises(profile_arithmetic.ArithmeticParityError) as caught:
                profile_arithmetic.observe(model,x,x)
            rows=caught.exception.audit['comparisons']['reference_repeat']['prediction_tensors']
            self.assertTrue(rows["$['disparity']"]['exact_equal'])
            self.assertFalse(rows["$['refinements'][0]"]['within_tolerance'])
        finally:sys.path.pop(0)

    def test_registered_single_warmup_retains_cold_difference_and_checks_stable_outputs(self):
        sys.path.insert(0,str(P.parent))
        try:
            import profile_arithmetic
            class ColdTransient(nn.Module):
                def __init__(self):super().__init__();self.calls=0
                def forward(self,left,right):
                    self.calls+=1;y=left.mean((1,2)).unsqueeze(1)+(self.calls==1)
                    return {'disparity':y,'refinements':[y*2],'diagnostics':{'call':self.calls}}
            model=ColdTransient();x=torch.ones(1,2,1,2,2)
            _,audit=profile_arithmetic.observe(model,x,x,warmup_forwards=1)
            self.assertEqual(model.calls,4)
            self.assertFalse(audit['cold_vs_reference']['allclose'])
            self.assertEqual(audit['cold_vs_reference']['maximum_absolute_difference'],2)
            self.assertTrue(audit['reference_repeat_allclose'])
            self.assertEqual(audit['maximum_absolute_difference'],0)
            self.assertTrue(audit['input_tensors_unchanged'])
        finally:sys.path.pop(0)

    def test_script_inventory_includes_inherited_static_script_without_forward(self):
        sys.path.insert(0,str(P.parent))
        try:
            import profile_arithmetic
            class Parent(nn.Module):
                reset=staticmethod(tiny_script_reset)
                def forward(self,*args):raise AssertionError('No forward permitted')
            class Child(Parent):pass
            inventory=profile_arithmetic.scripted_source_inventory(Child())
            self.assertEqual(len(inventory['scripts']),1)
            row=next(iter(inventory['scripts'].values()))
            self.assertEqual(row['references'],['<model>.reset'])
            self.assertIn('aten::mul',row['raw_graph_operations'])
            self.assertIn('aten::sub',row['raw_graph_operations'])
        finally:sys.path.pop(0)

    def test_input_mutation_fails_even_when_original_cache_is_protected(self):
        sys.path.insert(0,str(P.parent))
        try:
            import profile_arithmetic
            class MutatingAPI(nn.Module):
                def forward(self,left,right):
                    y=left.mean((1,2)).unsqueeze(1);left.add_(1)
                    return {'disparity':y}
            x=torch.ones(1,2,1,2,2)
            with self.assertRaises(profile_arithmetic.ArithmeticParityError) as caught:
                profile_arithmetic.observe(MutatingAPI(),x,x)
            self.assertFalse(caught.exception.audit['input_tensors_unchanged'])
            self.assertTrue(torch.equal(x,torch.ones_like(x)))
        finally:sys.path.pop(0)

    def test_observed_study_mask_pool_and_expand_boundaries_keep_work(self):
        a=torch.tensor([[True,False],[True,True]]);b=torch.tensor([[False,True],[True,True]])
        op=torch.ops.aten.__and__.Tensor;watch=counter.ArithmeticCounter()
        output=watch.__torch_dispatch__(op,(),(a,b),{})
        torch.testing.assert_close(output,torch.tensor([[False,False],[True,True]]))
        self.assertEqual(watch.report()['totals']['element___and__'],4)
        x=torch.arange(16.).reshape(1,1,4,4);watch=counter.ArithmeticCounter()
        pooled=watch.__torch_dispatch__(torch.ops.aten.max_pool2d.default,(),(x,[2,2],[2,2]),{})
        torch.testing.assert_close(pooled,torch.tensor([[[[5.,7.],[13.,15.]]]]))
        self.assertEqual(watch.report()['totals']['max_pool2d_output_elements'],4)
        watch.__torch_dispatch__(torch.ops.aten.expand_as.default,(),(torch.ones(1),x),{})
        self.assertEqual(watch.report()['unclassified_operators'],{})


if __name__=='__main__':unittest.main()
