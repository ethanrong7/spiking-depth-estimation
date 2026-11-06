"""Actual S²M² topology, native50-bin input and static-descriptor spike retrofit.

Research-small width32/one MRT/three local refinements. CNNs, projections, softmax,
OT, confidence, refiners and upsampling remain analog. Q/K/V and Transformer FFN
hidden activations use four internal LIF coding steps, not50 sensor-time steps.
Upstream files are unmodified; see vendor/s2m2/UPSTREAM.md and CC-BY-NC4 license.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from core.epipolar_spiking_stereo import signed_rate
from core.utils.utils import InputPadder
from vendor.s2m2.model.s2m2 import S2M2
from vendor.s2m2.model.attentions import SelfAttn,CrossAttn,FFN
from vendor.s2m2.model.submodules import DispInit,CostVolume


class SpikeProjection(nn.Module):
    def __init__(self,linear):
        super().__init__();self.linear=linear;self.last_rate=None
    def forward(self,x):
        out=signed_rate(self.linear(x),4);self.last_rate=out.detach().abs().mean();return out


class SpikeHidden(nn.Module):
    def __init__(self):super().__init__();self.last_rate=None
    def forward(self,x):
        out=signed_rate(x,4);self.last_rate=out.detach().abs().mean();return out


class SpikingOT(DispInit):
    def __init__(self,dim):
        super().__init__(dim,ot_iter=3,use_positivity=False)
        self.q=SpikeProjection(nn.Linear(dim,dim,bias=False))
        self.k=SpikeProjection(nn.Linear(dim,dim,bias=False))
        self.log_gain=nn.Parameter(torch.tensor(math.log(math.expm1(8.))))
        self.key_valid=None;self.query_valid=None

    def scores(self,features):
        left,right=self.layer_norm(features.permute(0,2,3,1)).chunk(2)
        q,k=self.q(left),self.k(right)
        return torch.einsum('bhic,bhjc->bhij',q,k)*(F.softplus(self.log_gain)/math.sqrt(q.shape[-1]))

    def forward(self,features):
        cv=self.scores(features);b,h,w,_=cv.shape
        qvalid,kvalid=self.query_valid,self.key_valid
        if qvalid is None:qvalid=torch.ones(b,h,w,device=cv.device,dtype=torch.bool)
        if kvalid is None:kvalid=torch.ones_like(qvalid)
        pair=qvalid[..., :,None]&kvalid[...,None,:]
        masked_logits=cv.masked_fill(~pair,-1e4)
        # Actual upstream log-Sinkhorn, dustbins and peak±2 confidence regression.
        prob=self._optimal_transport(masked_logits)
        prob=prob*pair
        peak=prob.argmax(-1);keys=torch.arange(w,device=cv.device,dtype=cv.dtype)
        local=prob*((keys-peak[...,None]).abs()<=2)
        conf=local.sum(-1);correspondence=(local*keys).sum(-1)/conf.clamp_min(1e-6)
        disparity=(keys-correspondence)[:,None]
        occ=prob.sum(-1)[:,None]
        # The actual local-refinement two-scale cost volume now uses spiking QK.
        cv_down=self.scores(F.interpolate(features,scale_factor=.5,mode='nearest'))
        down_pair=qvalid[:,::2,::2][..., :,None]&kvalid[:,::2,::2][...,None,:]
        cv_down=cv_down*down_pair
        self.last_logits=masked_logits;self.last_probability=prob;self.last_initial=disparity
        # Refiner features use neutral zero at invalid pairs, never OT sentinels.
        return disparity,conf[:,None],occ,cv*pair,cv_down


def fixed_native_initial(disparity,padding,height,width):
    """Stride4 physical lattice interpolation with valid border extension only."""
    px,_,py,_=padding
    x0,x1=(px+3)//4,(px+width-1)//4
    y0,y1=(py+3)//4,(py+height-1)//4
    crop=disparity[...,y0:y1+1,x0:x1+1]*4
    if not crop.numel():raise ValueError('no native stride4 lattice site')
    yy=(torch.arange(height,device=crop.device,dtype=crop.dtype)-(4*y0-py))/4
    xx=(torch.arange(width,device=crop.device,dtype=crop.dtype)-(4*x0-px))/4
    yy=2*yy/max(1,y1-y0)-1 if y1>y0 else yy*0
    xx=2*xx/max(1,x1-x0)-1 if x1>x0 else xx*0
    gy,gx=torch.meshgrid(yy,xx,indexing='ij')
    grid=torch.stack([gx,gy],-1)[None].expand(crop.shape[0],-1,-1,-1)
    return F.grid_sample(crop,grid,padding_mode='border',align_corners=True)


class SpikingS2M2(S2M2):
    def __init__(self,feature_channels=32,num_transformer=1,refine_iter=3):
        super().__init__(feature_channels,dim_expansion=1,num_transformer=num_transformer,use_positivity=False,output_upsample=False,refine_iter=refine_iter)
        self.cnn_backbone.conv0[0]=nn.Conv2d(50,16,1)
        self.upsample_mask_1x.conv_rgb[0]=nn.ConvTranspose2d(50,16,3,padding=1)
        for module in list(self.modules()):
            if isinstance(module,(SelfAttn,CrossAttn)):
                for name in ('q','k','v'):setattr(module,name,SpikeProjection(getattr(module,name)))
            if isinstance(module,FFN):module.ffn[1]=SpikeHidden()
        self.disp_init=SpikingOT(feature_channels)

    def normalize_img(self,left,right):return 2*left-1,2*right-1

    def spike_activity(self):
        rates=[m.last_rate for m in self.modules() if isinstance(m,(SpikeProjection,SpikeHidden)) and m.last_rate is not None]
        return torch.stack(rates).mean() if rates else torch.tensor(0.)

    def forward(self,left,right):
        if left.shape!=right.shape or left.shape[1]!=50:raise ValueError('nativeB50HW input required')
        b,_,h,w=left.shape;padder=InputPadder(left.shape,divis_by=32);left,right=padder.pad(left,right)
        l,r=self.normalize_img(left,right)
        f4,f2=self.extract_feature(l,r);f02,_=f2.chunk(2)
        pyramid=self.feat_pyramid(f4);tr=self.transformer(*pyramid)
        hc,wc=tr.shape[-2:];px,_,py,_=padder._pad
        xx=4*torch.arange(wc,device=left.device)-px;yy=4*torch.arange(hc,device=left.device)-py
        key_valid=((xx>=0)&(xx<w))[None,None].expand(b,hc,-1)
        query_valid=key_valid&((yy>=0)&(yy<h))[None,:,None]
        self.disp_init.key_valid=key_valid;self.disp_init.query_valid=query_valid
        disp,conf,occ,cv,cv_down=self.disp_init(tr);initial=disp
        f0,_=tr.chunk(2);p0,_=pyramid[0].chunk(2)
        # Training path deliberately removes the upstream inference-only detaches.
        disp=self.global_refiner(f0.contiguous(),disp,conf)
        fused=self.feat_fusion_layer(f0,p0);ctx=self.ctx_feat(fused);hidden=torch.tanh(ctx)
        coords=torch.arange(wc,device=left.device,dtype=f0.dtype)
        cv_fn=CostVolume(cv,cv_down,coords.reshape(1,1,wc,1).repeat(b,hc,1,1),radius=4)
        for _ in range(self.refine_iter):
            hidden,disp,conf,occ=self.refiner(hidden,ctx,disp,conf,occ,cv_fn)
            projected=coords[None,None,None]-disp
            occ=occ*((projected>=px/4)&(projected<=(px+w-1)/4))
        mask=self.upsample_mask_4x_refine(hidden,f02)
        up=self.upsample4x(disp*4,mask);occ_up=self.upsample4x(occ,mask);conf_up=self.upsample4x(conf,mask)
        edge=self.upsample_mask_1x(up,l,f02)
        up=self.upsample1x(up,edge);occ_up=self.upsample1x(occ_up,edge);conf_up=self.upsample1x(conf_up,edge)
        initial_up=fixed_native_initial(initial,padder._pad,h,w)
        return dict(flow=-padder.unpad(up),initial_flow=-initial_up,confidence=padder.unpad(conf_up),occlusion=padder.unpad(occ_up),logits=self.disp_init.last_logits,probabilities=self.disp_init.last_probability,query_valid=query_valid,key_valid=key_valid,padding=padder._pad,activity=self.spike_activity())


def s2m2_loss(output,target,valid,normalization_counts=None):
    from core.ssde_direct_attention import direct_loss
    # Same full-mask final L1 and global correspondence CE as the direct pilot;
    # initial OT L1 is auxiliary to keep matching trainable through confidence gates.
    loss,parts=direct_loss(output,target,valid,normalization_counts)
    v=valid[:,None] if valid.ndim==3 else valid
    v=v.bool()&torch.isfinite(target)&(target.abs()<512)
    counts=normalization_counts or {}
    initial=(output['initial_flow']-target).abs()[v].sum()/max(1,counts.get('valid',int(v.sum())))
    parts['initial_l1']=initial
    return loss+.5*initial,parts
