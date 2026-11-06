"""Tiny CPU route/gradient checks; no training or accuracy benchmark."""
from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from core.ssde_direct_attention import local_peak_flow,convex_from_attention,SSDEDirectAttention,direct_loss

torch.set_num_threads(2)
class DirectTests(unittest.TestCase):
    def test_signed_peak_geometry_and_padding(self):
        for shift in (-2,2):
            logits=torch.full((1,1,9,9),-20.)
            query=torch.zeros(1,1,9,dtype=torch.bool);keys=torch.ones_like(query);keys[...,-1]=False
            for x in range(9):
                k=x+shift
                if 0<=k<8:query[...,x]=True;logits[...,x,k]=20
            logits[...,-1]=1000 # padded key must never win
            flow,p,peak=local_peak_flow(logits,keys,query,straight_through=False)
            self.assertTrue(torch.allclose(flow[:,0][query],torch.full_like(flow[:,0][query],shift*4)))
            self.assertTrue((p[...,-1]==0).all())
    def test_global_gradient_beyond_peak(self):
        logits=torch.zeros(1,1,6,6,requires_grad=True)
        mask=torch.ones(1,1,6,dtype=torch.bool)
        flow,p,peak=local_peak_flow(logits,mask,mask)
        flow.sum().backward()
        self.assertGreater(logits.grad[0,0,0,5].abs().item(),0)
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue((peak==0).all()) # explicit flat-content tie, no displacement prior
    def test_convex_valid_only(self):
        flow=torch.full((1,1,3,4),999.)
        support=torch.zeros(1,3,4,dtype=torch.bool);support[:,1,1:3]=True;flow[:,:,1,1:3]=-7
        fine,valid=convex_from_attention(flow,support,torch.randn(1,144,3,4))
        self.assertTrue(torch.allclose(fine[valid],torch.full_like(fine[valid],-7),atol=1e-5))
        with self.assertRaises(ValueError):local_peak_flow(torch.zeros(1,1,2,2),torch.zeros(1,1,2,dtype=torch.bool),torch.ones(1,1,2,dtype=torch.bool))
    def test_whole_route_mask_loss_gradient(self):
        torch.manual_seed(1234);m=SSDEDirectAttention(width=16,heads=2)
        left=torch.rand(1,50,32,64,requires_grad=True);right=torch.rand_like(left,requires_grad=True)
        out=m(left,right);target=torch.full((1,1,32,64),-4.);valid=torch.ones(1,32,64,dtype=torch.bool)
        loss,parts=direct_loss(out,target,valid);loss.backward()
        self.assertEqual(tuple(out['flow'].shape),(1,1,32,64));self.assertTrue(out['native_support'].all())
        self.assertEqual(parts['out_of_frame_count'],32*4)
        for p in (m.final_qk.weight,left,right):
            self.assertTrue(torch.isfinite(p.grad).all());self.assertGreater(p.grad.abs().sum().item(),0)
        with torch.no_grad():
            flat=m(left,right,zero_content=True)
            self.assertTrue((flat['peak']==0).all())
        self.assertFalse(any('relative_bias' in n or 'readout' in n or 'update_block' in n for n,_ in m.named_parameters()))
    def test_odd_geometry(self):
        m=SSDEDirectAttention(width=16,heads=2).eval()
        with torch.no_grad():out=m(torch.rand(1,50,13,19),torch.rand(1,50,13,19))
        self.assertEqual(tuple(out['flow'].shape),(1,1,13,19));self.assertTrue(torch.isfinite(out['flow']).all())

if __name__=='__main__':unittest.main()
