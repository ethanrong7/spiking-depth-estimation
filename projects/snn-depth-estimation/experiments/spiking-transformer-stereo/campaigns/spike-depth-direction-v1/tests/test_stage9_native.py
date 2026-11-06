"""Tiny synthetic software checks only; no guarded parent/model/data import."""
import ast
import hashlib
from dataclasses import dataclass
from pathlib import Path
import types
import unittest
import torch
from torch import nn

SCRIPTS=Path(__file__).resolve().parents[1]/"scripts"


def require(value,message):
    if not value:raise RuntimeError(message)


def definitions(filename,names,namespace):
    tree=ast.parse((SCRIPTS/filename).read_text())
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    require(len(nodes)==len(names),"requested source definition absent")
    exec(compile(ast.Module(body=nodes,type_ignores=[]),filename,"exec"),namespace)
    return namespace


BASE=dict(torch=torch,nn=nn,require=require,PREFIXES=(24,32,49),dataclass=dataclass)
D=definitions("stage9_native.py",{"AvailabilityAdapter","visible_input","Features","State","MinimalAvailability"},dict(BASE))
A=definitions("stage9_actions.py",{"equivalent_flow"},dict(BASE,hashlib=hashlib))
R=definitions("stage9_recipe.py",{"prefix_for","pooled_endpoint_loss","sample_order"},dict(BASE))


class NativeSoftwareTests(unittest.TestCase):
    def test_adapter_identity_missingness_and_parent_rng(self):
        torch.manual_seed(7);before=torch.get_rng_state().clone();adapter=D["AvailabilityAdapter"]()
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        x=torch.zeros(1,50,2,3);x[:,1]=1
        self.assertTrue(torch.equal(adapter.encode(x,49),2*x-1))
        self.assertTrue(torch.equal(adapter.encode(x,24)[:,:25],(2*x-1)[:,:25]))
        self.assertEqual(float(adapter.encode(x,24)[:,25:].abs().sum().detach()),0)
        with torch.no_grad():adapter.weight[0,50,0,0]=2
        # Availability channel distinguishes an observed zero from missing.
        self.assertEqual(float(adapter.encode(x,24)[0,0,0,0].detach()),1)

    def test_future_mask_and_validation(self):
        x=torch.zeros(50,2,3);changed=x.clone();changed[25:]=1
        self.assertTrue(torch.equal(D["visible_input"](x,24),D["visible_input"](changed,24)))
        with self.assertRaises(RuntimeError):D["visible_input"](x,23)
        x[0,0,0]=2
        with self.assertRaises(RuntimeError):D["visible_input"](x,24)

    def test_prefix_cycle_and_independent_order(self):
        self.assertEqual([R["prefix_for"]("V",u,m) for u in range(3) for m in (0,1)],[24,32,49,24,32,49])
        self.assertEqual([R["prefix_for"]("F",u,m) for u in range(3) for m in (0,1)],[49]*6)
        before=torch.get_rng_state().clone();a=R["sample_order"]()
        self.assertEqual(len(set(a)),800);self.assertEqual(a,R["sample_order"]())
        self.assertTrue(torch.equal(before,torch.get_rng_state()))

    def test_pooled_valid_pixels_not_micro_mean(self):
        a=torch.tensor([[[[1.,3.]]]],requires_grad=True);b=torch.tensor([[[[9.,0.]]]],requires_grad=True)
        va=torch.tensor([[True,True]]);vb=torch.tensor([[True,False]])
        loss=R["pooled_endpoint_loss"]({k:a for k in (4,8,16)},torch.zeros(1,1,2),va,3)
        loss+=R["pooled_endpoint_loss"]({k:b for k in (4,8,16)},torch.zeros(1,1,2),vb,3)
        self.assertAlmostEqual(float(loss.detach()),13/3,places=6);loss.backward()
        self.assertTrue(torch.allclose(a.grad,torch.full_like(a,1/3)))
        self.assertEqual(float(b.grad[0,0,0,1]),0)

    def test_segmented_state_keeps_recurrent_gradient_and_reset_isolation(self):
        # Exercise the actual staged loop with tiny synthetic arithmetic modules,
        # never the parent model. Coordinates retain native detach semantics.
        class Cell:
            def __init__(self):self.output_lif=types.SimpleNamespace(membrane=None)
        class Update:
            def __init__(self):self.cells=[Cell() for _ in range(3)]
            def reset_state(self,clear_diagnostics=True):
                for c in self.cells:c.output_lif.membrane=None
            def __call__(self,net,context,corr,disp,**kwargs):
                out=[]
                for c,n in zip(self.cells,net):
                    old=0 if c.output_lif.membrane is None else c.output_lif.membrane
                    c.output_lif.membrane=.6*old+n*.1
                    out.append(n*.9+c.output_lif.membrane)
                delta=torch.cat((out[0],torch.ones_like(out[0])),1)
                return out,None,torch.ones_like(delta),delta
        updater=Update()
        native=types.SimpleNamespace(update_block=updater,upsample_disp=lambda value,mask:value)
        host=types.SimpleNamespace(native=native)
        advance=types.MethodType(D["MinimalAvailability"].advance,host)
        def fresh():
            initial=torch.ones(1,1,2,3,requires_grad=True)
            features=types.SimpleNamespace(padder=types.SimpleNamespace(unpad=lambda x:x))
            state=D["State"](features,lambda coords:coords,[initial,initial*2,initial*3],[],
                torch.zeros(1,2,2,3),torch.zeros(1,2,2,3),(None,None,None))
            return state,initial
        state,initial=fresh();advance(state,4);advance(state,8);segmented=advance(state,16)[16]
        segmented.sum().backward();self.assertGreater(float(initial.grad.abs().sum()),0)
        # Final-only loss reaches earliest recurrent state despite coordinate truncation.
        self.assertTrue(all(c.output_lif.membrane is None for c in updater.cells))
        direct,_=fresh();whole=advance(direct,16)[16]
        self.assertTrue(torch.equal(segmented,whole))
        other,_=fresh();advance(other,16)
        again,_=fresh();self.assertTrue(torch.equal(advance(again,16)[16],whole))

    def test_ieee_signed_zero_parity_is_numeric_but_not_bitwise(self):
        a=torch.tensor([[[[0.]]]],dtype=torch.float32);b=-a
        parity=A["equivalent_flow"](a,b)
        self.assertEqual(parity["max_abs_difference"],0)
        self.assertFalse(parity["bitwise_equal"])
        self.assertEqual(parity["differing_pixel_count"],1)
        self.assertNotEqual(parity["actual_output_sha256"],parity["direct_output_sha256"])

    def test_core_and_raw_reader_bindings_static(self):
        import hashlib,json
        camp=SCRIPTS.parent;repo=camp.parents[5]
        binding=json.loads((camp/"configs/stage9-source-bindings.json").read_text())
        core={p:s for p,s in binding["source_files"].items() if "/core/" in p}
        self.assertEqual(len(core),15)
        for relative,expected in binding["source_files"].items():self.assertEqual(hashlib.sha256((repo/relative).read_bytes()).hexdigest(),expected)
        self.assertEqual(hashlib.sha256((camp/"configs/stage9-split-rows.json").read_bytes()).hexdigest(),binding["split_sha256"])


if __name__=="__main__":unittest.main()
