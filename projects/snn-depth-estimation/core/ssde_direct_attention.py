"""Compact SSDEFormer-inspired matcher: every disparity comes from spiking QK.

50 sensor bins remain CNN input channels. Four descriptor LIF coding steps are
internal static coding, not sensor time. No ALIF updater, correlation volume,
relative displacement bias or independent absolute disparity head exists here.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from core.extractor import BasicEncoder
from core.epipolar_spiking_stereo import EpipolarSpikingDescriptor, signed_rate
from core.utils.utils import InputPadder


def local_peak_flow(logits, key_valid, query_valid, stride=4, straight_through=True):
    """B,H,Wquery,Wkey scores; masked peak±1 expectation, signed key-query flow.

    Ties choose the first valid key (torch.argmax convention). Invalid queries
    return zero with a separate support mask, and may never seed interpolation.
    Training backward uses global expectation; forward is always local peak±1.
    """
    if not key_valid.any(-1).all():
        raise ValueError('each row needs an in-frame key')
    masked = logits.masked_fill(~key_valid[...,None,:],torch.finfo(logits.dtype).min)
    probabilities = masked.softmax(-1)
    peak = probabilities.argmax(-1)
    keys = torch.arange(logits.shape[-1],device=logits.device,dtype=logits.dtype)
    near = (keys-peak[...,None]).abs()<=1
    local = probabilities * near * key_valid[...,None,:]
    local = local / local.sum(-1,keepdim=True).clamp_min(torch.finfo(logits.dtype).tiny)
    query = torch.arange(logits.shape[-2],device=logits.device,dtype=logits.dtype)
    local_flow = stride*((local*keys).sum(-1)-query)
    global_flow = stride*((probabilities*keys).sum(-1)-query)
    flow = local_flow.detach()+global_flow-global_flow.detach() if straight_through else local_flow
    return flow[:,None]*query_valid[:,None], probabilities, peak


def convex_from_attention(flow, support, mask_logits, factor=4):
    """Convex mixing of valid attention-derived flows only; no free flow offset."""
    b,_,h,w=flow.shape
    neighbors=F.unfold(flow,3,padding=1).reshape(b,1,9,1,1,h,w)
    allowed=F.unfold(support[:,None].float(),3,padding=1).reshape(b,1,9,1,1,h,w)>0
    logits=mask_logits.reshape(b,1,9,factor,factor,h,w)
    # An entirely unsupported padded cell is zero, explicitly unsupported.
    weights=logits.masked_fill(~allowed,torch.finfo(logits.dtype).min).softmax(2)*allowed
    weights=weights/weights.sum(2,keepdim=True).clamp_min(torch.finfo(logits.dtype).tiny)
    fine=(weights*neighbors).sum(2).permute(0,1,4,2,5,3).reshape(b,1,h*factor,w*factor)
    valid=allowed.any(2).expand(b,1,factor,factor,h,w).permute(0,1,4,2,5,3).reshape(b,1,h*factor,w*factor)
    return fine,valid


class SSDEDirectAttention(nn.Module):
    def __init__(self,width=64,heads=4,steps=4,seed=1234):
        super().__init__()
        self.heads,self.steps=heads,steps
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+404)
            self.encoder=BasicEncoder(output_dim=256,norm_fn='instance',downsample=2)
            torch.manual_seed(seed+1207)
            self.transformer=EpipolarSpikingDescriptor(256,width,heads,steps)
            self.final_norm=nn.LayerNorm(256)
            self.final_qk=nn.Linear(256,2*width,bias=False)
            self.log_gain=nn.Parameter(torch.tensor(math.log(math.expm1(8.))))
            self.convex_mask=nn.Sequential(nn.Conv2d(256,128,3,padding=1),nn.ReLU(),nn.Conv2d(128,9*16,1))
            nn.init.zeros_(self.convex_mask[-1].weight);nn.init.zeros_(self.convex_mask[-1].bias)

    def forward(self,left,right,*,zero_content=False):
        if left.shape!=right.shape or left.ndim!=4 or left.shape[1]!=50:
            raise ValueError('paired native B,50,H,W sensor stacks required')
        b,_,h,w=left.shape
        padder=InputPadder(left.shape,divis_by=32)
        left,right=padder.pad(2*left-1,2*right-1)
        fl,fr=self.encoder([left,right]);fl,fr=self.transformer(fl,fr)
        def code(x):return signed_rate(self.final_qk(self.final_norm(x.permute(0,2,3,1))),self.steps).chunk(2,-1)
        ql,_=code(fl);_,kr=code(fr)
        hc,wc=fl.shape[-2:];dim=ql.shape[-1]//self.heads
        ql=ql.reshape(b,hc,wc,self.heads,dim).transpose(2,3)
        kr=kr.reshape(b,hc,wc,self.heads,dim).transpose(2,3)
        logits=((ql@kr.transpose(-1,-2))/math.sqrt(dim)).mean(2)*F.softplus(self.log_gain)
        if zero_content:logits=logits*0
        px,_,py,_=padder._pad
        xx=4*torch.arange(wc,device=left.device)-px
        yy=4*torch.arange(hc,device=left.device)-py
        key_valid=((xx>=0)&(xx<w))[None,None].expand(b,hc,-1)
        query_valid=key_valid & ((yy>=0)&(yy<h))[None,:,None]
        coarse,probabilities,peak=local_peak_flow(logits,key_valid,query_valid,straight_through=self.training)
        fine,fine_valid=convex_from_attention(coarse,query_valid,self.convex_mask(fl))
        fine,fine_valid=padder.unpad(fine),padder.unpad(fine_valid)
        if not fine_valid.all():raise RuntimeError('native output has no valid attention-derived inpainting support')
        return dict(flow=fine,coarse_flow=coarse,probabilities=probabilities,logits=logits,
                    query_valid=query_valid,key_valid=key_valid,peak=peak,padding=padder._pad,
                    native_support=fine_valid,route='spikingQK → maskedsoftmax → localpeak±1 → convexvalidmix',
                    attention_entropy=(-(probabilities*probabilities.clamp_min(1e-12).log()).sum(-1))[query_valid].mean())


def direct_loss(output,target,valid,normalization_counts=None):
    """Full valid-depth L1; global CE only when GT key lies inside right frame.

    Out-of-view GT remains in full L1/accuracy. Correspondence supervision is
    sampled at exact padded stride4 query coordinates, not resized label means.
    """
    valid=valid[:,None] if valid.ndim==3 else valid
    valid=valid.bool()&torch.isfinite(target)&(target.abs()<512)
    if not valid.any():raise ValueError('empty full valid mask')
    b,_,h,w=target.shape;px,pr,py,pb=output['padding']
    x=torch.arange(w,device=target.device)[None,None,None]
    inframe=valid&(x+target>=0)&(x+target<=w-1)
    flow=output['flow'];counts=normalization_counts or {}
    l1=(flow-target).abs()[valid].sum()/max(1,counts.get('valid',int(valid.sum())))
    sampled=F.pad(torch.where(valid,target,torch.zeros_like(target)),(px,pr,py,pb))[...,::4,::4][:,0]
    mask=F.pad(inframe,(px,pr,py,pb))[...,::4,::4][:,0]&output['query_valid']
    wc=sampled.shape[-1];keys=torch.arange(wc,device=target.device)[None,None]+sampled/4
    first=output['key_valid'].long().argmax(-1)[...,None]
    last=wc-1-output['key_valid'].flip(-1).long().argmax(-1)[...,None]
    keys=torch.maximum(first,torch.minimum(last,keys))
    lower,upper=keys.floor().long(),keys.ceil().long();frac=keys-lower
    logits=output['logits'].masked_fill(~output['key_valid'][...,None,:],torch.finfo(flow.dtype).min)
    lp=logits.log_softmax(-1)
    ce=-((1-frac)*lp.gather(-1,lower[...,None]).squeeze(-1)+frac*lp.gather(-1,upper[...,None]).squeeze(-1))
    match=ce[mask].sum()/max(1,counts.get('correspondence',int(mask.sum())))
    return l1+.1*match,dict(full_l1=l1,correspondence_ce=match,valid_count=int(valid.sum()),inframe_count=int(inframe.sum()),out_of_frame_count=int((valid&~inframe).sum()),correspondence_count=int(mask.sum()))
