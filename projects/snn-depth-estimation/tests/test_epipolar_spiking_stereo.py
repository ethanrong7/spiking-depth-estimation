"""Tiny synthetic CPU checks, not a dataset experiment or accuracy benchmark."""
import json
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.epipolar_spiking_stereo import EpipolarSpikingDescriptor, EpipolarSpikingStereoNet, signed_rate, epipolar_attention
from core.spikestereonet import SpikeStereoNet
from spikingjelly.activation_based import functional

torch.set_num_threads(2)

class DescriptorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)

    def test_identity_and_live_output_gradient(self):
        block=EpipolarSpikingDescriptor(8,8,2)
        left=torch.randn(1,8,2,7,requires_grad=True);right=torch.randn_like(left)
        out=block(left,right)
        self.assertTrue(torch.equal(out[0],left));self.assertTrue(torch.equal(out[1],right))
        (out[0].square().sum()+out[1].square().sum()).backward()
        for p in (block.output.weight,block.ffn_out.weight):
            self.assertTrue(torch.isfinite(p.grad).all());self.assertGreater(p.grad.abs().sum().item(),0)

    def test_surrogate_and_open_branch_gradients(self):
        x=torch.tensor([-.8,-.1,.2,.8],requires_grad=True)
        z=signed_rate(x);z.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all());self.assertTrue((x.grad.abs()>0).all())
        block=EpipolarSpikingDescriptor(8,8,2)
        torch.nn.init.normal_(block.output.weight,std=.05);torch.nn.init.normal_(block.ffn_out.weight,std=.05)
        left=torch.randn(1,8,2,7,requires_grad=True);right=torch.randn_like(left,requires_grad=True)
        l,r=block(left,right)
        cross_grad=torch.autograd.grad(l.square().mean(),right,retain_graph=True)[0]
        self.assertGreater(cross_grad.abs().sum().item(),0)
        (l.square().mean()+r.square().mean()).backward()
        for p in (block.qkv.weight,block.ffn_in.weight,left,right):
            self.assertTrue(torch.isfinite(p.grad).all());self.assertGreater(p.grad.abs().sum().item(),0)
        self.assertTrue(torch.equal(l.detach(),block(left,right)[0].detach()))

    def test_signed_shift_and_row_isolation(self):
        # Unique synthetic descriptors; shift is key-column minus query-column.
        left=torch.eye(8)[None,None].repeat(1,2,1,1)
        for shift in (-2,2):
            right=torch.zeros_like(left)
            queries=torch.arange(max(0,-shift),min(8,8-shift))
            right[:,:,queries+shift]=left[:,:,queries]
            code=signed_rate(left*2)
            _,p=epipolar_attention(code,signed_rate(right*2),signed_rate(right*2),1)
            expected=queries+shift
            self.assertTrue(torch.equal(p.argmax(-1)[0,:,0,queries],expected[None].expand(2,-1)))
            changed=right.clone();changed[:,1]=torch.randn_like(changed[:,1])
            out,_=epipolar_attention(code,signed_rate(changed*2),signed_rate(changed*2),1)
            baseline,_=epipolar_attention(code,signed_rate(right*2),signed_rate(right*2),1)
            self.assertTrue(torch.equal(out[:,0],baseline[:,0]))

    def test_whole_native_parity_and_backward(self):
        args=SimpleNamespace(hidden_dims=[128]*3,context_norm='batch',shared_backbone=False,n_snn_layers=3,n_downsample=2,corr_levels=4,corr_radius=4,corr_implementation='reg',mixed_precision=False,feature_encoder='cnn',context_encoder='cnn',matched_initialization_seed=1234,slow_fast_gru=False)
        base=SpikeStereoNet(args).eval();new=EpipolarSpikingStereoNet(args).eval();new.load_native_state_dict(base.state_dict())
        left=torch.rand(1,50,32,64);right=torch.rand_like(left)
        with torch.no_grad():
            p=base(left,right,iters=2)[0][-1];functional.reset_net(base)
            q=new(left,right,iters=2)[0][-1];functional.reset_net(new)
        self.assertTrue(torch.equal(p,q));self.assertEqual(tuple(q.shape),(1,1,32,64))
        pred=new(left,right,iters=2)[0][-1];pred.square().mean().backward()
        grad=new.fnet.descriptor.output.weight.grad
        self.assertTrue(torch.isfinite(grad).all());self.assertGreater(grad.abs().sum().item(),0)
        functional.reset_net(new)

if __name__=='__main__':unittest.main()
