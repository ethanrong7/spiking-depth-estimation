"""Opt-in candidate attention correction for the frozen normalized v0 updater."""
import math,types
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

class Spike(torch.autograd.Function):
 @staticmethod
 def forward(ctx,x):ctx.save_for_backward(x);return (x>=0).to(x.dtype)
 @staticmethod
 def backward(ctx,grad):
  x,=ctx.saved_tensors;return grad/(1+(math.pi*x).square())

class GuidedCorrection(nn.Module):
 def __init__(self,hidden,levels=4,radius=4,row_chunk=8):
  super().__init__();self.levels=levels;self.radius=radius;self.row_chunk=row_chunk
  self.query=nn.Linear(hidden,16);self.key=nn.Linear(4,16);self.query_norm=nn.LayerNorm(16);self.key_norm=nn.LayerNorm(16);self.gate=nn.Parameter(torch.zeros(()));self.override=None
  offsets=torch.cat([torch.arange(-radius,radius+1)*2**level for level in range(levels)]).float()
  self.register_buffer('offsets',offsets);self.register_buffer('level_ids',torch.arange(levels).repeat_interleave(2*radius+1).float())
 def _chunk(self,readout,corr,disp,start,width):
  # Native lookup centers are detached by the parent; keep readout/corr attached.
  b,h,w,_=readout.shape;n=len(self.offsets);xx=torch.arange(w,device=readout.device)[None,None,:,None]
  center=xx+disp[:,0].unsqueeze(-1);scale=(2**self.level_ids)[None,None,None,:]
  sample=center/scale+(self.offsets/scale.reshape(-1))[None,None,None,:]
  valid=(sample>=0)&(sample<=torch.div(torch.tensor(width,device=readout.device),scale,rounding_mode='floor')-1)
  content=corr.permute(0,2,3,1)
  if self.override=='permuted':
   # Fixed slot permutation affects only correlation scalars, never native lookup.
   order=torch.arange(n-1,-1,-1,device=content.device);rank=valid.long().cumsum(-1)-1
   # Stable reverse among valid slots only, preserving invalid scalars.
   sorted_ids=torch.argsort(torch.where(valid,torch.arange(n,device=content.device),n+torch.arange(n,device=content.device)),dim=-1)
   count=valid.sum(-1,keepdim=True);donor=sorted_ids.gather(-1,(count-1-rank).clamp(0,n-1));content=torch.where(valid,content.gather(-1,donor),content)
  if self.override=='positional':content=torch.zeros_like(content)
  tokens=torch.stack((content,self.offsets.expand_as(content)/32,self.level_ids.expand_as(content)/3,valid.to(content)),dim=-1)
  q=Spike.apply(self.query_norm(self.query(readout))-.5).reshape(b,h,w,4,4)
  k=Spike.apply(self.key_norm(self.key(tokens))-.5).reshape(b,h,w,n,4,4)
  scores=(q.unsqueeze(3)*k).sum(-1).permute(0,1,2,4,3)/2
  nonempty=valid.any(-1,keepdim=True);masked=scores.masked_fill(~valid.unsqueeze(3),-1e9);weights=masked.softmax(-1)*valid.unsqueeze(3);weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-12)
  mean=(valid*self.offsets).sum(-1)/valid.sum(-1).clamp_min(1)
  correction=((weights*self.offsets).sum(-1)-mean.unsqueeze(-1)).mean(-1)
  return correction.unsqueeze(1)
 def forward(self,readout,corr,disp):
  chunks=[];width=readout.shape[-1]
  for start in range(0,readout.shape[-2],self.row_chunk):
   args=(readout[:,:,start:start+self.row_chunk].permute(0,2,3,1),corr[:,:,start:start+self.row_chunk],disp[:,:,start:start+self.row_chunk],start,width)
   chunks.append(checkpoint(self._chunk,*args,use_reentrant=False) if self.training and torch.is_grad_enabled() else self._chunk(*args))
  branch=torch.cat(chunks,dim=2);scale=0*self.gate if self.override=='off' else self.gate.tanh()
  return scale*branch

def install_guidance(model,seed=2017):
 block=model.update_block
 for cell in block.cells:
  cell.attention_enabled=False;cell.qkv.requires_grad_(False);cell.output_proj.requires_grad_(False);cell.attention_gate.requires_grad_(False)
 with torch.random.fork_rng(devices=[]):
  torch.manual_seed(seed);block.guided_attention=GuidedCorrection(block.disp_head.conv1.in_channels,model.args.corr_levels,model.args.corr_radius)
 original=block.forward
 def forward(self,net,inp,corr=None,disp=None,**kwargs):
  result=original(net,inp,corr,disp,**kwargs)
  if kwargs.get('update',True):
   net,readout,mask,delta=result;correction=self.guided_attention(readout,corr,disp)
   delta=delta+torch.cat((correction,torch.zeros_like(correction)),1)
   return net,readout,mask,delta
  return result
 block.forward=types.MethodType(forward,block);return model
