"""Row-chunked on-demand native control and projected spiking stereo evidence."""
from types import MethodType
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from core.epipolar_spiking_stereo import signed_rate


def sample_row_candidates(feature,coords,start,level,radius):
    b,c,height,width=feature.shape;rows,queries=coords.shape[-2:]
    offsets=torch.arange(-radius,radius+1,device=coords.device,dtype=coords.dtype)
    x=coords[:,0,:,: ,None]/2**level+offsets
    # Put epipolar rows in the batch axis: no fractional-Y rounding or row mixing.
    row_features=feature[:,:,start:start+rows].permute(0,2,1,3).reshape(b*rows,c,1,width)
    grid=torch.stack((2*x/max(width-1,1)-1,torch.zeros_like(x)),-1).reshape(b*rows,1,queries*len(offsets),2)
    values=F.grid_sample(row_features,grid,align_corners=True,padding_mode='zeros').reshape(b,rows,c,queries,len(offsets)).permute(0,2,1,3,4)
    return values,(x>=0)&(x<=width-1)

class OnDemandDot:
    """Linear native dot correlation, pooled-right only; never all-pairs width²."""
    def __init__(self,left,right,num_levels=4,radius=4,row_chunk=4):
        self.left=left;self.right=[right];self.levels=num_levels;self.radius=radius;self.row_chunk=row_chunk
        for _ in range(num_levels-1):self.right.append(F.avg_pool2d(self.right[-1],(1,2)))
    def __call__(self,coords):
        outputs=[];height=self.left.shape[-2]
        for level,right in enumerate(self.right):
            rows=[]
            for start in range(0,height,self.row_chunk):
                end=min(start+self.row_chunk,height);sampled,_=sample_row_candidates(right,coords[:,:,start:end],start,level,self.radius)
                value=(self.left[:,:,start:end,:,None]*sampled).sum(1)/self.left.new_tensor(self.left.shape[1]).sqrt()
                rows.append(value)
            outputs.append(torch.cat(rows,1).permute(0,3,1,2))
        return torch.cat(outputs,1).contiguous().float()

class SpikingEvidence(nn.Module):
    def __init__(self,channels=256,width=32,heads=4,levels=4,radius=4,row_chunk=4):
        super().__init__();self.width=width;self.heads=heads;self.levels=levels;self.radius=radius;self.row_chunk=row_chunk
        self.q=nn.Conv2d(channels,width,1,bias=False);self.k=nn.Conv2d(channels,width,1,bias=False);self.v=nn.Conv2d(channels,width,1,bias=False)
        self.q_norm=nn.GroupNorm(1,width);self.k_norm=nn.GroupNorm(1,width);self.v_norm=nn.GroupNorm(1,width)
        self.readout=nn.Linear(heads,1);self.use_checkpoint=True;self.evidence_evaluations=0;self.trace=[];self._trace_parts=None
    def prepare(self,left,right,num_levels=4,radius=4):
        assert num_levels==self.levels and radius==self.radius
        q=signed_rate(self.q_norm(self.q(left)),4);k=signed_rate(self.k_norm(self.k(right)),4);v=signed_rate(self.v_norm(self.v(right)),4)
        kp,vp=[k],[v]
        for _ in range(self.levels-1):kp.append(F.avg_pool2d(kp[-1],(1,2)));vp.append(F.avg_pool2d(vp[-1],(1,2)))
        self.evidence_evaluations=0;self.trace=[]
        if not self.training:
            self.code_diagnostics={name+'_abs_mean':float(x.detach().abs().mean()) for name,x in [('q',q),('k',k),('v',v)]}
            self.code_diagnostics['query_zero_code_fraction']=float((q.detach()==0).all(1).float().mean())
            self.code_diagnostics['right_adjacent_collision_fraction']=float((k.detach()[...,1:]==k.detach()[...,:-1]).all(1).float().mean())
        module=self
        class Prepared:
            def __call__(self,coords):return module.lookup(q,kp,vp,coords)
        return Prepared()
    def chunk(self,q,k,v,coords,start,level):
        sampled,valid=sample_row_candidates(k,coords,start,level,self.radius);values,_=sample_row_candidates(v,coords,start,level,self.radius)
        b,c,rows,w,n=sampled.shape;d=c//self.heads
        logits=(q.reshape(b,self.heads,d,rows,w,1)*sampled.reshape(b,self.heads,d,rows,w,n)).sum(2)/(d**.5)
        support=valid[:,None];weights=logits.masked_fill(~support,-1e4).softmax(-1)*support
        weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-8)
        weighted=(weights[:,:,None]*values.reshape(b,self.heads,d,rows,w,n)).mean(2)
        evidence=self.readout(weighted.permute(0,2,3,4,1)).squeeze(-1)*valid
        if self._trace_parts is not None:self._trace_parts.append(dict(scores=logits.detach().mean(1),weights=weights.detach().mean(1),valid=valid.detach()))
        return evidence.permute(0,3,1,2).contiguous()
    def lookup(self,q,kp,vp,coords):
        self.evidence_evaluations+=1;record=not self.training and self.evidence_evaluations in (1,8,16,32);levels=[];trace=[]
        for level,(k,v) in enumerate(zip(kp,vp)):
            chunks=[];self._trace_parts=[] if record else None
            for start in range(0,q.shape[-2],self.row_chunk):
                end=min(start+self.row_chunk,q.shape[-2]);query=q[:,:,start:end];position=coords[:,:,start:end]
                def fn(query,key,value,position,start=start,level=level):return self.chunk(query,key,value,position,start,level)
                chunks.append(checkpoint(fn,query,k,v,position,use_reentrant=False) if self.training and self.use_checkpoint else fn(query,k,v,position))
            levels.append(torch.cat(chunks,2))
            if record:trace.append({name:torch.cat([part[name] for part in self._trace_parts],1) for name in ('scores','weights','valid')})
        self._trace_parts=None
        if record:self.trace.append(dict(iteration=self.evidence_evaluations,coords=coords.detach(),levels=trace))
        return torch.cat(levels,1).float()

def install_evidence(model,seed=2017,plain=False):
    from core import spikestereonet as native
    assert model.args.corr_implementation=='reg'
    if not plain:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed);model.spiking_evidence=SpikingEvidence(levels=model.args.corr_levels,radius=model.args.corr_radius)
    original_forward=model.forward
    def forward(self,*args,**kwargs):
        previous=native.CorrBlock1D
        native.CorrBlock1D=OnDemandDot if plain else self.spiking_evidence.prepare
        try:return original_forward(*args,**kwargs)
        finally:native.CorrBlock1D=previous
    model.forward=MethodType(forward,model)
    if not plain:
        def observe(module,args,output):
            evidence=model.spiking_evidence
            if not model.training and evidence.trace and evidence.trace[-1]['iteration']==evidence.evidence_evaluations:
                entry=evidence.trace[-1];coords=entry['coords'];left=torch.arange(coords.shape[-1],device=coords.device)[None,None,:]
                entry['after']=(coords[:,0]-left+output[-1][:,0]).detach()
        model._evidence_trace_hook=model.update_block.register_forward_hook(observe)
    return model
