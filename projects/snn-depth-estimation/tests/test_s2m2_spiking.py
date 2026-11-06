"""Tiny CPU wiring/geometry checks for the actual vendored S²M² retrofit."""
from pathlib import Path
import sys,unittest,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from core.s2m2_spiking import SpikingS2M2,SpikingOT,fixed_native_initial,s2m2_loss,SpikeProjection
from vendor.s2m2.model.stacked_MRT import StackedMRT

torch.set_num_threads(2)
class S2M2Tests(unittest.TestCase):
    def test_actual_topology_and_native_backward(self):
        torch.manual_seed(1234);m=SpikingS2M2()
        self.assertIsInstance(m.transformer,StackedMRT)
        self.assertEqual(len(m.transformer.uformer_list),1)
        self.assertEqual(m.refine_iter,3)
        self.assertEqual(m.cnn_backbone.conv0[0].in_channels,50)
        self.assertEqual(m.upsample_mask_1x.conv_rgb[0].in_channels,50)
        left=torch.rand(1,50,32,64,requires_grad=True);right=torch.rand_like(left,requires_grad=True)
        out=m(left,right);loss,parts=s2m2_loss(out,torch.full((1,1,32,64),-4.),torch.ones(1,32,64,dtype=torch.bool))
        self.assertTrue(torch.isfinite(loss));loss.backward()
        for tensor in (m.disp_init.q.linear.weight,m.disp_init.k.linear.weight,left,right):
            self.assertTrue(torch.isfinite(tensor.grad).all());self.assertGreater(tensor.grad.abs().sum().item(),0)
        self.assertTrue((left.grad.abs().sum((0,2,3))>0).all())
        self.assertGreater(out['activity'].item(),0);self.assertEqual(out['flow'].shape,(1,1,32,64))
    def test_final_only_gradient_not_detached(self):
        torch.manual_seed(1234);m=SpikingS2M2()
        out=m(torch.rand(1,50,32,64),torch.rand(1,50,32,64))
        out['flow'].square().mean().backward()
        self.assertGreater(m.disp_init.q.linear.weight.grad.norm().item(),0)
    def test_invalid_dustbin_rows_and_neutral_refiner(self):
        m=SpikingOT(8);m.query_valid=torch.ones(1,2,8,dtype=torch.bool);m.query_valid[:,0]=False
        m.key_valid=torch.ones(1,2,8,dtype=torch.bool);m.key_valid[...,:2]=False
        features=torch.randn(2,8,2,8,requires_grad=True)
        d,c,o,cv,down=m(features)
        self.assertTrue((cv[:,0]==0).all());self.assertTrue((cv[...,:2]==0).all())
        self.assertTrue((down[...,:1]==0).all())
        (d.sum()+cv.square().sum()+m.last_probability.sum()).backward()
        self.assertTrue(torch.isfinite(features.grad).all())
    def test_signed_ot_correspondence(self):
        m=SpikingOT(8)
        with torch.no_grad():m.q.linear.weight.copy_(torch.eye(8));m.k.linear.weight.copy_(torch.eye(8))
        left=torch.eye(8).T[None,:,None,:].repeat(1,1,2,1)*4
        for shift in (-2,2):
            right=torch.roll(left,shift,-1)
            d,_,_,_,_=m(torch.cat([left,right]))
            queries=torch.arange(max(0,-shift),min(8,8-shift))
            # Upstream positive disparity is xL−xR; native flow negates it.
            self.assertTrue(torch.allclose(-d[0,0,:,queries],torch.full_like(d[0,0,:,queries],shift),atol=.04))
    def test_initial_lattice_independent_of_refiner(self):
        disp=torch.full((1,1,8,8),999.,requires_grad=True)
        valid=disp.clone();valid[...,1:8,2:6]=2
        fine=fixed_native_initial(valid,[8,8,3,3],26,16)
        self.assertTrue(torch.allclose(fine,torch.full_like(fine,8)))
        fine.sum().backward();self.assertTrue(torch.isfinite(disp.grad).all())

if __name__=='__main__':unittest.main()
