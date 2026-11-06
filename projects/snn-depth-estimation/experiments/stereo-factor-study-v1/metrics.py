"""Detached per-support stage diagnostics; official full-mask metrics stay separate."""
import torch
import torch.nn.functional as F

def collect_metrics(model,arm,target,valid,padding,stride=4):
    if valid.ndim==3:valid=valid[:,None]
    truth=F.pad(target,padding)[...,::stride,::stride][:,0]/stride
    mask=F.pad(valid.bool(),padding)[...,::stride,::stride][:,0]
    x=torch.arange(target.shape[-1],device=target.device)[None,None,None]
    inview=valid.bool()&(x+target>=0)&(x+target<=target.shape[-1]-1)
    visible=F.pad(inview,padding)[...,::stride,::stride][:,0]
    result=dict(native_valid=int(valid.sum()),native_inview=int(inview.sum()),native_oov=int((valid&~inview).sum()),stages=[])
    entries=[]
    if arm=='A':
        result['codes']=getattr(model.spiking_evidence,'code_diagnostics',{})
        for entry in model.spiking_evidence.trace:
            coords=entry['coords'][:,0];left=torch.arange(coords.shape[-1],device=coords.device)[None,None,:]
            candidates=torch.cat([(coords[...,None]/2**level+torch.arange(-4,5,device=coords.device))*2**level-left[...,None] for level in range(4)],-1)
            weights=torch.cat([v['weights'] for v in entry['levels']],-1)/4
            support=torch.cat([v['valid'] for v in entry['levels']],-1)
            entries.append((entry['iteration'],coords-left,entry.get('after',coords-left),candidates,weights,support,None))
    else:
        c=model.last_components;before=torch.zeros_like(c['coordinate'][:,0]);coordinate=c['coordinate'][:,0];residual=c['residual'][:,0]
        candidates=c['offsets'].expand(*before.shape,-1)
        entries.append((1,before,coordinate+residual,candidates,c['weights'],c['valid'],coordinate))
        result['residual_abs_native_mean']=float(residual.abs()[mask].mean()*stride)
        if 'coordinate_native' in c:
            from core.utils.utils import InputPadder
            padder=InputPadder(target.shape,divis_by=32)
            coord=padder.unpad(c['coordinate_native']);res=padder.unpad(c['residual_native'])
            result['coordinate_native_epe']=float((coord-target).abs()[valid].mean());result['residual_native_abs_mean']=float(res.abs()[valid].mean())
    for iteration,before,after,candidates,weights,support,coordinate in entries:
        weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-9)
        nonempty=support.any(-1)
        low_supported=candidates.masked_fill(~support,float('inf')).min(-1).values
        high_supported=candidates.masked_fill(~support,-float('inf')).max(-1).values
        supported_distance=(low_supported-truth).clamp_min(0)+(truth-high_supported).clamp_min(0)
        supported_nearest=(candidates-truth[...,None]).abs().masked_fill(~support,float('inf')).min(-1).values
        low=candidates.min(-1).values;high=candidates.max(-1).values
        distance=(low-truth).clamp_min(0)+(truth-high).clamp_min(0)
        nearest=(candidates-truth[...,None]).abs().min(-1).values
        chosen=candidates.gather(-1,weights.argmax(-1,keepdim=True)).squeeze(-1)
        stage=dict(iteration=iteration,groups={})
        for name,m in [('valid',mask),('inview',mask&visible),('oov',mask&~visible),('negative_flow',mask&(truth<0)),('positive_flow',mask&(truth>0))]:
            stage['groups'][name]=dict(count=int(m.sum()),before_error_sum=float((before-truth).abs()[m].sum()*stride),after_error_sum=float((after-truth).abs()[m].sum()*stride),candidate_hull_distance_sum=float(distance[m].sum()*stride),nearest_candidate_distance_sum=float(nearest[m].sum()*stride),near_one_cell_count=int(((nearest<=1)&m).sum()),sampled_support_count=int((m&nonempty).sum()),sampled_hull_distance_sum=float(supported_distance[m&nonempty].sum()*stride),sampled_nearest_distance_sum=float(supported_nearest[m&nonempty].sum()*stride),top_weight_error_sum=float((chosen-truth).abs()[m&nonempty].sum()*stride),empty_evidence_count=int((~support.any(-1)&m).sum()),mean_entropy_sum=float((-(weights.clamp_min(1e-9).log()*weights).sum(-1))[m].sum()))
            if coordinate is not None:stage['groups'][name]['coordinate_error_sum']=float((coordinate-truth).abs()[m].sum()*stride)
        result['stages'].append(stage)
    result['bound_scope']='Unmasked geometry and admissible sampled support reported separately. A updater and B residual are unrestricted; B coordinate alone is bounded by sampled support.'
    return result
