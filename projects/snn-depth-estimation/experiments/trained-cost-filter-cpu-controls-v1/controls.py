"""Unlabelled frozen-cost interventions; no training or GT-dependent fit."""
from contextlib import contextmanager
from types import MethodType
import torch
from torch.nn import functional as F

CONDITIONS = ('baseline','bypass','raw_match_Y_moments','smooth5_match_Y_moments','Y_match_X_moments','positive_affine_LS')
EPS = 1e-8


def fit_control(x, y, condition):
    if condition not in CONDITIONS:
        raise ValueError(condition)
    xd, yd = x.double(), y.double()
    mx,my=xd.mean(-1,keepdim=True),yd.mean(-1,keepdim=True)
    dx,dy=xd-mx,yd-my
    vx,vy=dx.square().mean(-1,keepdim=True),dy.square().mean(-1,keepdim=True)
    sx,sy=vx.sqrt(),vy.sqrt()
    both=(sx>EPS)&(sy>EPS)
    slope=(dx*dy).mean(-1,keepdim=True)/vx.clamp_min(EPS**2)
    a=torch.where(sx>EPS,slope.clamp_min(1e-6),torch.full_like(slope,1e-6))
    affine=my+a*dx
    stats=dict(rows=sx.numel(),raw_tiny_std_fraction=float((sx<=EPS).double().mean()),
        learned_tiny_std_fraction=float((sy<=EPS).double().mean()),
        affine_unconstrained_negative_fraction=float((slope<0).double().mean()),
        affine_floor_fraction=float(((slope<1e-6)|(sx<=EPS)).double().mean()),
        affine_fit_MSE=float((affine-yd).square().mean()),learned_variance_mean=float(vy.mean()),
        affine_fit_MSE_over_learned_variance=float((affine-yd).square().mean()/vy.mean().clamp_min(EPS**2)))
    if condition=='baseline':
        return y,stats
    if condition=='bypass':
        return x,stats
    if condition=='positive_affine_LS':
        output=affine
    elif condition=='Y_match_X_moments':
        output=torch.where(both,mx+sx/sy.clamp_min(EPS)*dy,yd)
        stats['fallback_fraction']=float((~both).double().mean())
    else:
        source=xd
        if condition=='smooth5_match_Y_moments':
            width=x.shape[-1]
            flat=xd.reshape(-1,1,width)
            source=(F.avg_pool1d(flat,5,stride=1,padding=2,count_include_pad=False)).reshape(x.shape)
        mean=source.mean(-1,keepdim=True)
        centered=source-mean
        std=centered.square().mean(-1,keepdim=True).sqrt()
        defined=std>EPS
        output=torch.where(defined,my+sy/std.clamp_min(EPS)*centered,my)
        stats['fallback_fraction']=float((~defined).double().mean())
    stats['output_mean_abs_difference_Y']=float((output-yd).abs().mean())
    return output.to(x.dtype),stats


@contextmanager
def intervention(model, condition, sink):
    module=model.placement
    existed='compute' in module.__dict__
    old=module.__dict__.get('compute')
    compute=module.compute
    def changed(self,x):
        y=compute(x)
        output,stats=fit_control(x,y,condition)
        sink.append(stats)
        return output
    module.compute=MethodType(changed,module)
    try:
        yield
    finally:
        if existed:module.compute=old
        else:delattr(module,'compute')
