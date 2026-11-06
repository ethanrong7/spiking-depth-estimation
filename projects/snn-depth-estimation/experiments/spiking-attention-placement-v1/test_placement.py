import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from core.spiking_attention_placement import CostFilter,DescriptorAttention,RecurrentSources
from core.update import BasicMultiUpdateBlock
from types import SimpleNamespace

def test_cost():
    m=CostFilter();x=torch.randn(1,2,3,1,7,requires_grad=True);y=m(x);assert torch.equal(x,y);y.square().mean().backward();assert m.output.weight.grad.norm()>0
    with torch.no_grad():m.output.weight.add_(.01)
    m.zero_grad();m(x).square().mean().backward();assert m.codes.q.weight.grad.norm()>0 and m.codes.k.weight.grad.norm()>0

def test_descriptors():
    m=DescriptorAttention();l=torch.randn(1,256,2,5,requires_grad=True);r=torch.randn_like(l);a,b=m(l,r);assert torch.equal(a,l) and torch.equal(b,r)
    with torch.no_grad():m.output.weight.add_(.01)
    a,b=m(l,r);(a.square().mean()+b.square().mean()).backward();assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None);assert m.codes.q.weight.grad.norm()>0

def test_recurrent():
    args=SimpleNamespace(corr_levels=4,corr_radius=4,n_snn_layers=3,n_downsample=2)
    m=RecurrentSources(BasicMultiUpdateBlock(args,[128]*3));h=torch.randn(1,2,3,128);c=torch.randn(1,2,3,384);motion=torch.randn(1,2,3,128)
    a=m.compute(h,c,motion);b=m.compute(h+1,c,motion);assert not torch.equal(a,b);a.square().mean().backward()
    for layer in (m.codes.q,m.codes.k,m.codes.v):assert layer.weight.grad.norm()>0 and torch.isfinite(layer.weight.grad).all()

if __name__=='__main__':
    torch.set_num_threads(1);torch.manual_seed(4)
    for f in (test_cost,test_descriptors,test_recurrent):f();print(f.__name__,'PASS')
