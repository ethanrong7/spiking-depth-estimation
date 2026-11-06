"""Candidate permutation equivariant pointwise/spatial spiking scorers."""
import torch
from torch import nn
from torch.nn import functional as F
from core.epipolar_spiking_stereo import signed_rate

class Selector(nn.Module):
 def __init__(self,spatial=False):
  super().__init__();self.spatial=spatial;self.embed=nn.Linear(4,32);self.q=nn.Linear(32,32,bias=False);self.k=nn.Linear(32,32,bias=False);self.v=nn.Linear(32,32,bias=False);self.out=nn.Linear(32,1);nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)
 def forward(self,features,query_valid=None):
  # B,H,W,K,F. Messages compare relative disparity, never rank channels.
  z=signed_rate(self.embed(features));q=signed_rate(self.q(z));k=signed_rate(self.k(z));v=signed_rate(self.v(z));b,h,w,n,c=z.shape
  if self.spatial:
   def neighbors(t):
    t=t.permute(0,3,4,1,2).reshape(b,n*c,h,w) if t.shape[-1]==c else t.permute(0,3,4,1,2).reshape(b,n,h,w)
    ch=t.shape[1]//n
    return F.unfold(t,3,padding=1).reshape(b,n,ch,9,h,w).permute(0,4,5,3,1,2).reshape(b,h,w,9*n,ch)
   nk,nv=neighbors(k),neighbors(v);nd=neighbors(features[...,:1]);mask=F.unfold((torch.ones(b,1,h,w,device=z.device) if query_valid is None else query_valid[:,None].to(z.dtype)),3,padding=1).reshape(b,9,h,w).permute(0,2,3,1).repeat_interleave(n,-1)
  else:nk,nv,nd=k,v,features[...,:1];mask=torch.ones(b,h,w,n,device=z.device)
  weights=((q[...,None,:]*nk[...,None,:,:]).sum(-1)/32**.5-(features[...,0,None]-nd[...,None,:,0]).abs()).masked_fill(~mask[...,None,:].bool(),-1e4).softmax(-1)
  return features[...,2]*8+self.out(z+(weights[...,None]*nv[...,None,:,:]).sum(-2)).squeeze(-1)

def candidates(output):
 scores,keys=output['logits'].topk(5,-1);x=torch.arange(scores.shape[-2],device=scores.device)[None,None,:,None];disp=(x-keys)*4
 probs=output['probabilities'].gather(-1,keys)
 features=torch.stack([disp/128,scores/8,(scores-scores[...,:1])/8,probs.clamp_min(1e-12).log()/16],-1)
 return features,keys
