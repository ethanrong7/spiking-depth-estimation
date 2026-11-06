"""Opt-in direct candidate correction replacing only Original's free DispHead."""
from types import MethodType
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from core.epipolar_spiking_stereo import signed_rate

class OriginalCoordinateHead(nn.Module):
    offsets=(-16,-8,-4,-1,0,1,4,8,16)
    def __init__(self,attention=True,levels=4,radius=4,stride=4):
        super().__init__();self.attention=attention;self.levels=levels;self.radius=radius;self.stride=stride
        self.context=nn.Conv2d(128,16,1);self.input=nn.Linear(16+2*levels+3,16);self.norm1=nn.LayerNorm(16)
        self.q=nn.Linear(16,16);self.k=nn.Linear(16,16);self.v=nn.Linear(16,16);self.proj=nn.Linear(16,16)
        self.norm2=nn.LayerNorm(16);self.ff1=nn.Linear(16,32);self.ff2=nn.Linear(32,16);self.score=nn.Linear(16,1)
        nn.init.normal_(self.score.weight,std=.001);nn.init.zeros_(self.score.bias)
        self.native=(250,400,(8,8,3,3));self.bound=None;self.use_checkpoint=True;self.trace=[];self.override=None
    def set_native_geometry(self,shape,padding):self.native=(shape[-2],shape[-1],tuple(padding));self.trace=[]
    @staticmethod
    def correction(weights,offsets):return (weights*offsets).sum(-1)
    def evidence(self,corr,flow):
        b,_,h,w=corr.shape;offsets=torch.tensor(self.offsets,device=corr.device,dtype=corr.dtype)
        rows=corr.reshape(b,self.levels,2*self.radius+1,h,w).permute(0,3,4,1,2);values=[];support=[]
        for level in range(self.levels):
            index=self.radius+offsets/2**level;ok=(index>=0)&(index<=2*self.radius);lo=index.floor().long().clamp(0,2*self.radius);hi=(lo+1).clamp(0,2*self.radius);alpha=(index-index.floor())
            value=rows[...,level,lo]*(1-alpha)+rows[...,level,hi]*alpha
            values.append(value*ok);support.append(ok)
        values=torch.stack(values,-1);support=torch.stack(support,-1).to(corr.dtype).expand(b,h,w,-1,-1)
        nh,nw,pad=self.native;xx=torch.arange(w,device=corr.device)*self.stride-pad[0];yy=torch.arange(h,device=corr.device)*self.stride-pad[2]
        right=xx[None,None,:,None]+self.stride*(flow[:,0,:,:,None]+offsets)
        visible=(right>=0)&(right<=nw-1)&((yy>=0)&(yy<nh))[None,:,None,None]
        return values*visible[...,None],support,visible,offsets
    def compute(self,state,corr,flow):
        evidence,support,visible,offsets=self.evidence(corr,flow);context=self.context(state).permute(0,2,3,1);n=len(self.offsets)
        context=context[...,None,:].expand(-1,-1,-1,n,-1);f=flow[:,0,:,:,None].expand_as(visible)
        tokens=torch.cat([context,evidence,support,(f*self.stride/512)[...,None],(offsets/16).expand_as(f)[...,None],visible[...,None].float()],-1)
        x=self.input(tokens);z=self.norm1(x);q,k,v=[signed_rate(layer(z),4) for layer in (self.q,self.k,self.v)]
        mix=((q@k.transpose(-1,-2))/4).softmax(-1)@v if self.attention else v
        x=x+self.proj(mix);x=x+self.ff2(signed_rate(self.ff1(self.norm2(x)),4));logits=self.score(x).squeeze(-1)-offsets.abs()/8
        weights=logits.softmax(-1)
        if self.override=='uniform':weights=torch.ones_like(weights)/n
        elif self.override is not None:
            weights=torch.zeros_like(weights);weights[...,self.offsets.index(self.override)]=1
        delta=self.correction(weights,offsets)
        if not self.training:self.trace.append(dict(before=flow[:,0].detach(),after=(flow[:,0]+delta).detach(),weights=weights.detach(),visible=visible.detach(),logits=logits.detach()))
        return torch.stack([delta,torch.zeros_like(delta)],1)
    def forward(self,state):
        assert self.bound is not None
        corr,flow=self.bound
        if self.training and self.use_checkpoint:return checkpoint(self.compute,state,corr,flow,use_reentrant=False)
        return self.compute(state,corr,flow)

def install_coordinate_head(model,attention=True):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2017);head=OriginalCoordinateHead(attention,model.args.corr_levels,model.args.corr_radius,2**model.args.n_downsample)
    model.update_block.disp_head=head
    original=model.update_block.forward
    def wrapped(module,*args,**kwargs):
        corr=kwargs.get('corr',args[2] if len(args)>2 else None);flow=kwargs.get('disp',args[3] if len(args)>3 else None);head.bound=(corr,flow)
        try:return original(*args,**kwargs)
        finally:head.bound=None
    model.update_block.forward=MethodType(wrapped,model.update_block)
    return model
