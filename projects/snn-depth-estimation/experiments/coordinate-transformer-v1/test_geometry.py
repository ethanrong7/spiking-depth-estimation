"""Tiny CPU operator geometry and paired parameter checks; no dataset experiment."""
import torch
from core.coordinate_transformer import CoordinateStereo,CoordinateBlock

def main():
    center=torch.tensor([100.,-100.]);offset=torch.arange(-2,3)*4;c=center[:,None]+offset
    valid=torch.ones_like(c,dtype=torch.bool)
    assert torch.equal(CoordinateStereo.expectation(c,torch.ones_like(c)/5,valid),center)
    for i in range(5):
        w=torch.zeros_like(c);w[:,i]=1
        assert torch.equal(CoordinateStereo.expectation(c,w,valid),c[:,i])
    # Signed mirrored candidates, OOV coordinates and duplicate support.
    assert torch.equal(-c[0].flip(0),c[1])
    for d in (-512.,-100.,100.,512.):
        candidates=torch.cat([torch.linspace(-512,512,17),d+offset]).clamp(-512,512)
        eq=(candidates[:,None]-candidates[None,:]).abs()<1e-5
        allowed=~(eq&torch.ones(len(candidates),len(candidates),dtype=torch.bool).tril(-1)).any(-1)
        assert len(candidates[allowed])==len(candidates.unique())
        choose=(candidates==d).nonzero()[0,0];w=torch.zeros_like(candidates);w[choose]=1
        assert CoordinateStereo.expectation(candidates,w,allowed)==d
        width=400;x=80;xr=x-d;mirrored=(width-1-x)-(-d)
        assert (0<=xr<=width-1)==(0<=mirrored<=width-1)
    # Geometry only: right impulse shifting +1 featurecell gives disparity -4px.
    signal=torch.zeros(10);signal[4]=1
    before=40-4*signal.argmax();after=40-4*torch.roll(signal,1).argmax();assert after-before==-4
    block=CoordinateBlock();tokens=torch.randn(1,1,1,5,99,requires_grad=True)
    s=block(tokens,torch.ones(1,1,1,5,dtype=torch.bool));pred=(s.softmax(-1)*c[0]).sum();pred.backward()
    assert block.score.weight.grad.norm()>0 and block.q.weight.grad.norm()>0 and block.k.weight.grad.norm()>0
    torch.manual_seed(1234);a=CoordinateStereo(True);torch.manual_seed(1234);b=CoordinateStereo(False)
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    from core.s2m2_spiking_v2 import SpikingS2M2V2
    torch.manual_seed(1234);reference=SpikingS2M2V2()
    assert all(torch.equal(v,reference.state_dict()[k]) for k,v in a.state_dict().items() if k in reference.state_dict())
    print('PASS signed expectation, mirrored geometry, OOV, duplicates, logits sensitivity, scoring/QK gradients, paired state')
if __name__=='__main__':main()
