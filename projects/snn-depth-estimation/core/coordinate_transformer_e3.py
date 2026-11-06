"""Registered E3 range branch: only widen final two candidate sets."""
import torch
from torch.nn import functional as F
from core.coordinate_transformer import CoordinateStereo,coordinate_loss
from core.s2m2_spiking import fixed_native_initial
from core.utils.utils import InputPadder

class RangeCoordinateStereo(CoordinateStereo):
    def forward(self,left,right):
        b,_,h,w=left.shape;padder=InputPadder(left.shape,divis_by=32);left,right=padder.pad(left,right);l,r=self.normalize_img(left,right)
        f4,_=self.extract_feature(l,r);pyramid=self.feat_pyramid(f4);tr=self.transformer(*pyramid)
        hc,wc=tr.shape[-2:];px,_,py,_=padder._pad;xx=4*torch.arange(wc,device=left.device)-px;yy=4*torch.arange(hc,device=left.device)-py
        key_valid=((xx>=0)&(xx<w))[None,None].expand(b,hc,-1);query_valid=key_valid&((yy>=0)&(yy<h))[None,:,None]
        self.disp_init.key_valid=key_valid;self.disp_init.query_valid=query_valid
        initial,*_=self.disp_init(tr);fl,fr=tr.chunk(2);fl=fl.permute(0,2,3,1);fr=fr.permute(0,2,3,1);d=initial[:,0]*4
        predictions=[];diagnostics=[];trace=[]
        for stage in range(3):
            offsets=torch.arange(-2,3,device=d.device)*4 if stage==0 else torch.tensor([-32,-16,-8,-4,0,4,8,16,32],device=d.device)
            local=d[...,None]+offsets
            candidates=torch.cat([torch.linspace(-512,512,17,device=d.device).expand(*d.shape,17),local],-1) if stage==0 else local
            clipped=(candidates < -512)|(candidates>512);candidates=candidates.clamp(-512,512);n=candidates.shape[-1]
            eq=(candidates[..., :,None]-candidates[...,None,:]).abs()<1e-5
            duplicate=(eq&torch.ones(n,n,device=d.device,dtype=torch.bool).tril(-1)).any(-1);valid=~duplicate
            xright=xx[None,None,:,None]-candidates;visible=(xright>=0)&(xright<=w-1)
            gx=2*((xright+px)/4)/max(wc-1,1)-1;gy=2*torch.arange(hc,device=d.device)/max(hc-1,1)-1
            grid=torch.stack([gx,gy[None,:,None,None].expand_as(gx)],-1).reshape(b,hc,wc*n,2)
            sampled=F.grid_sample(fr.permute(0,3,1,2),grid,align_corners=True).reshape(b,32,hc,wc,n).permute(0,2,3,4,1)
            sampled=sampled*visible[...,None];left_tokens=fl[...,None,:].expand(-1,-1,-1,n,-1)
            tokens=torch.cat([left_tokens,sampled,left_tokens*sampled,(candidates/512)[...,None],((candidates-d[...,None])/512)[...,None],visible[...,None].float()],-1)
            logits=self.coordinate(tokens,valid)-(candidates-d[...,None]).abs().div(16).clamp(max=3)
            weights=logits.masked_fill(~valid,-1e4).softmax(-1);next_d=self.expectation(candidates,weights,valid)
            diagnostics.append(dict(mean_delta=float((next_d-d).detach().mean()),saturated_fraction=float(((next_d<=-512)|(next_d>=512)).float().mean())))
            if not self.training:trace.append({k:v.detach() for k,v in dict(before=d,candidates=candidates,weights=weights,after=next_d,valid=valid,visible=visible,clipped=clipped).items()})
            d=next_d;predictions.append(-fixed_native_initial(d[:,None]/4,padder._pad,h,w))
        return dict(flow=predictions[-1],sequence=predictions,initial_flow=-fixed_native_initial(initial,padder._pad,h,w),global_flow=predictions[0],confidence=torch.zeros_like(predictions[-1]),occlusion=torch.zeros_like(predictions[-1]),logits=self.disp_init.last_logits,probabilities=self.disp_init.last_probability,query_valid=query_valid,key_valid=key_valid,padding=padder._pad,activity=self.spike_activity(),coordinate_diagnostics=diagnostics,coordinate_trace=trace)
