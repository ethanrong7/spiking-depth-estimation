"""Tiny FP32 materialization/gradient/geometry and checkpoint-memory tests."""
import copy,json,torch
from core.corr import CorrBlock1D,PytorchAlternateCorrBlock1D
from core.utils.utils import coords_grid
from core.stereo_factor_evidence import OnDemandDot,SpikingEvidence,sample_row_candidates

def main():
 torch.manual_seed(1234);left=torch.randn(1,8,3,19,requires_grad=True);right=torch.randn_like(left,requires_grad=True)
 coords=coords_grid(1,3,19).to(left.device);coords[:,0]+=torch.linspace(-3.25,2.5,19);coords.requires_grad_()
 dense=CorrBlock1D(left,right,num_levels=4,radius=4)(coords);plain=OnDemandDot(left,right,num_levels=4,radius=4,row_chunk=2)(coords);alternate=PytorchAlternateCorrBlock1D(left,right,num_levels=4,radius=4)(coords)
 errors=[]
 for i in range(4):
  e=(dense[:,9*i:9*(i+1)]-plain[:,9*i:9*(i+1)]).abs();errors.append(dict(level=i,max=float(e.max()),mean=float(e.mean())))
 assert torch.allclose(dense,plain,atol=3e-6,rtol=1e-5);assert torch.allclose(dense,alternate,atol=3e-6,rtol=1e-5)
 weights=torch.randn_like(dense);gd=torch.autograd.grad((dense*weights).sum(),(left,right,coords),retain_graph=True);gp=torch.autograd.grad((plain*weights).sum(),(left,right,coords),retain_graph=True)
 gradient=[float((a-b).abs().max()) for a,b in zip(gd,gp)];assert all(e<3e-5 for e in gradient)
 # A must neither allocate width² nor leave evidence when every candidate is invalid.
 a=SpikingEvidence(channels=8,row_chunk=2);b=copy.deepcopy(a);b.use_checkpoint=False
 out=a.prepare(left,right)(coords);other=b.prepare(left,right)(coords);assert torch.equal(out,other)
 torch.autograd.backward((out*weights).sum());torch.autograd.backward((other*weights).sum())
 for name in ('q','k','v'):assert getattr(a,name).weight.grad.norm()>0
 for (name,p),(name2,q) in zip(a.named_parameters(),b.named_parameters()):assert name==name2 and torch.allclose(p.grad,q.grad,atol=3e-6,rtol=3e-5)
 empty=coords.detach().clone();empty[:,0]=10000;assert not a.prepare(left.detach(),right.detach())(empty).any()
 # Native coordinate sign: a right impulse shifted +1 moves its sampled offset +1.
 feature=torch.zeros(1,1,3,19);feature[:,:,:,7]=1;center=coords.detach().clone();center[:,0]=5
 first,_=sample_row_candidates(feature,center[:,:,:1],0,0,4);shift=torch.zeros_like(feature);shift[...,1:]=feature[...,:-1];second,_=sample_row_candidates(shift,center[:,:,:1],0,0,4)
 assert ((second.argmax(-1)-first.argmax(-1))==1).all()
 shift=torch.zeros_like(feature);shift[...,:-1]=feature[...,1:];third,_=sample_row_candidates(shift,center[:,:,:1],0,0,4);assert ((third.argmax(-1)-first.argmax(-1))==-1).all()
 print(json.dumps(dict(status='PASS',per_level_errors=errors,input_gradient_max=gradient,alternate_native_axis_parity=True,checkpoint_gradient_parity=True,empty_support_zero=True,signed_shift=True)))
if __name__=='__main__':main()
