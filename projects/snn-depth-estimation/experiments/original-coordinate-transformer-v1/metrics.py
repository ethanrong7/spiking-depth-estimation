"""Detached coarse-anchor E4 diagnostics; official full-resolution metrics stay separate."""
import torch
import torch.nn.functional as F


def original_coordinate_metrics(trace, target, valid, padding, stride=4):
    if valid.ndim == 3:valid=valid[:,None]
    gt=F.pad(target,padding)[...,::stride,::stride][:,0]
    mask=F.pad(valid.bool(),padding)[...,::stride,::stride][:,0]
    x=torch.arange(target.shape[-1],device=target.device)[None,None,None]
    visible=valid.bool() & (x+target>=0) & (x+target<=target.shape[-1]-1)
    visible=F.pad(visible,padding)[...,::stride,::stride][:,0]
    rows=[]
    for iteration in (0,1,7,15,31):
        if iteration>=len(trace):continue
        item=trace[iteration]
        before=item['before'].detach().float()
        after=item['after'].detach().float()
        if before.ndim==4 and before.shape[1]==1:before=before[:,0]
        if after.ndim==4 and after.shape[1]==1:after=after[:,0]
        weights=item['weights'].detach().float()
        offsets=torch.tensor([-16,-8,-4,-1,0,1,4,8,16],device=before.device,dtype=before.dtype)
        candidates=(before[...,None]+offsets)*stride
        before=before*stride;after=after*stride
        lo=candidates.min(-1).values;hi=candidates.max(-1).values
        hull=(lo-gt).clamp_min(0)+(gt-hi).clamp_min(0)
        nearest=(candidates-gt[...,None]).abs().min(-1).values
        entropy=-(weights*weights.clamp_min(1e-12).log()).sum(-1)
        values=dict(before_epe=(before-gt).abs(),after_epe=(after-gt).abs(),hull_bound=hull,
                    attainable_excess=(after-gt).abs()-hull,nearest_candidate_distance=nearest,
                    within_one_cell=(nearest<=stride).float(),bracketed=((gt>=lo)&(gt<=hi)).float(),
                    improvement=(before-gt).abs()-(after-gt).abs(),signed_movement=after-before,
                    absolute_movement=(after-before).abs(),entropy=entropy,
                    visible_candidate_mass=(weights*item['visible'].bool()).sum(-1))
        row=dict(iteration=iteration,units='native horizontal pixels',finite=bool(all(torch.isfinite(v[mask]).all() for v in values.values())))
        for label,subset in [('visible',mask&visible),('out_of_view',mask&~visible)]:
            row[label]=dict(count=int(subset.sum()),**{k+'_sum':float(v[subset].sum()) for k,v in values.items()})
        rows.append(row)
    return dict(iterations=rows,support='Official valid targets sampled at padded coarse anchors; true visibility from native signed target. Hull bound describes coarse candidate coordinates before convex spatial upsampling.')
