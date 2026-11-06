"""Three opt-in placements on the native 50-channel SpikeStereoNet control."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from core.epipolar_spiking_stereo import signed_rate
from core.corr import CorrBlock1D

class Codes(nn.Module):
    def __init__(self,width):
        super().__init__(); self.norm=nn.LayerNorm(width);self.q=nn.Linear(width,width);self.k=nn.Linear(width,width);self.v=nn.Linear(width,width)
    def forward(self,x):
        z=self.norm(x);return tuple(signed_rate(layer(z),4) for layer in (self.q,self.k,self.v))

class CostFilter(nn.Module):
    def __init__(self):
        super().__init__();self.input=nn.Linear(2,16);self.codes=Codes(16);self.output=nn.Linear(16,1);nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def compute(self,cost):
        # Each independent (batch,left-y,left-x) row contains all right-x costs.
        shape=cost.shape;w=shape[-1];x=cost.reshape(-1,w,1);position=torch.linspace(-1,1,w,device=x.device).view(1,w,1).expand(x.shape[0],-1,-1)
        q,k,v=self.codes(self.input(torch.cat([x,position],-1)))
        k=F.pad(k.transpose(1,2),(2,2)).unfold(2,5,1).permute(0,2,3,1)
        v=F.pad(v.transpose(1,2),(2,2)).unfold(2,5,1).permute(0,2,3,1)
        valid=(torch.arange(w,device=x.device)[:,None]+torch.arange(-2,3,device=x.device)[None,:]);valid=(valid>=0)&(valid<w)
        logits=(q[:,:,None,:]*k).sum(-1)/4;weights=logits.masked_fill(~valid[None],-torch.inf).softmax(-1)
        return (x+self.output((weights[...,None]*v).sum(-2))).reshape(shape)
    def forward(self,cost):return checkpoint(self.compute,cost,use_reentrant=False) if self.training else self.compute(cost)

class DescriptorAttention(nn.Module):
    def __init__(self):
        super().__init__();self.input=nn.Linear(256,16);self.codes=Codes(16);self.output=nn.Linear(16,256);nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def compute(self,left,right):
        l,r=left.permute(0,2,3,1),right.permute(0,2,3,1);ql,kl,vl=self.codes(self.input(l));qr,kr,vr=self.codes(self.input(r))
        # Both directions use the original, unmodified feature pair; padded native descriptors retained.
        a=l+self.output(((ql@kr.transpose(-1,-2))/4).softmax(-1)@vr)
        b=r+self.output(((qr@kl.transpose(-1,-2))/4).softmax(-1)@vl)
        return a.permute(0,3,1,2).contiguous(),b.permute(0,3,1,2).contiguous()
    def forward(self,left,right):return checkpoint(self.compute,left,right,use_reentrant=False) if self.training else self.compute(left,right)

class RecurrentSources(nn.Module):
    def __init__(self,native):
        super().__init__();self.encoder=native.encoder;self.mask=native.mask;self.disp_head=native.disp_head
        self.state=nn.Linear(128,32);self.context=nn.Linear(384,32);self.motion=nn.Linear(128,32);self.codes=Codes(32);self.output=nn.Linear(32,32);self.norm=nn.LayerNorm(32);self.ff1=nn.Linear(32,64);self.ff2=nn.Linear(64,32);self.voltage=nn.Linear(32,128)
    def compute(self,state,context,motion):
        tokens=torch.stack([self.state(state),self.context(context),self.motion(motion)],-2);q,k,v=self.codes(tokens)
        weights=((q[...,:1,:]@k.transpose(-1,-2))/(32**.5)).softmax(-1)
        x=tokens[...,0,:]+self.output((weights@v).squeeze(-2));x=x+self.ff2(signed_rate(self.ff1(self.norm(x)),4))
        return self.voltage(x)
    def forward(self,net,inp,corr=None,disp=None,**kwargs):
        motion=self.encoder(disp,corr);args=(net[0].permute(0,2,3,1),torch.cat(inp[0],1).permute(0,2,3,1),motion.permute(0,2,3,1))
        v=checkpoint(self.compute,*args,use_reentrant=False) if self.training else self.compute(*args);v=v.permute(0,3,1,2).contiguous();net[0]=torch.tanh(v)
        return net,v,.25*self.mask(v),self.disp_head(v)

def install(model,arm):
    # Native construction precedes optional modules; shared tensors retain exact control initialization.
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(2017)
        if arm=='A':
            model.placement=CostFilter()
            class FilteredCorr(CorrBlock1D):
                def __init__(self,left,right,num_levels=4,radius=4):
                    self.radius=radius;self.num_levels=num_levels;corr=model.placement(CorrBlock1D.corr(left,right));b,h,w,_,r=corr.shape;corr=corr.reshape(b*h*w,1,1,r);self.corr_pyramid=[corr]
                    for _ in range(num_levels):corr=F.avg_pool2d(corr,[1,2],stride=[1,2]);self.corr_pyramid.append(corr)
            # Per-worker process: override the class resolved by native forward, never the native formula.
            import core.spikestereonet as native
            native.CorrBlock1D=FilteredCorr
        elif arm=='B':
            model.placement=DescriptorAttention();model.fnet.register_forward_hook(lambda module,args,pair:model.placement(*pair))
        elif arm=='C':model.update_block=RecurrentSources(model.update_block)
        else:raise ValueError(arm)
    return model
