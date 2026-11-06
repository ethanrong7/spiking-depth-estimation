"""Detached E2 stage diagnostics; never alter official predictions or masks."""
import torch
import torch.nn.functional as F


def coordinate_metrics(output, target, valid):
    if valid.ndim == 3:valid=valid[:,None]
    pad=output['padding']
    gt=-F.pad(target,pad)[...,::4,::4][:,0]
    mask=F.pad(valid.bool(),pad)[...,::4,::4][:,0]
    x=torch.arange(target.shape[-1],device=target.device)[None,None,None]
    visible=valid.bool() & (x+target>=0) & (x+target<=target.shape[-1]-1)
    vis=F.pad(visible,pad)[...,::4,::4][:,0]
    rows=[]
    for i,trace in enumerate(output['coordinate_trace']):
        candidates=trace['candidates'].detach()
        weights=trace['weights'].detach()
        allowed=trace['valid'].bool()
        before=trace['before'].detach();after=trace['after'].detach()
        if before.ndim==4 and before.shape[1]==1:before=before[:,0]
        if after.ndim==4 and after.shape[1]==1:after=after[:,0]
        lo=candidates.masked_fill(~allowed,torch.inf).min(-1).values
        hi=candidates.masked_fill(~allowed,-torch.inf).max(-1).values
        nearest=(candidates-gt[...,None]).abs().masked_fill(~allowed,torch.inf).min(-1).values
        hull=(lo-gt).clamp_min(0)+(gt-hi).clamp_min(0)
        values=dict(before_epe=(before-gt).abs(),after_epe=(after-gt).abs(),nearest_candidate_distance=nearest,
                    within4=(nearest<=4).float(),bracketed=((gt>=lo)&(gt<=hi)).float(),convex_hull_distance=hull,
                    improvement=(before-gt).abs()-(after-gt).abs(),movement=after-before,
                    absolute_movement=(after-before).abs(),entropy=-(weights*weights.clamp_min(1e-12).log()).sum(-1),
                    rescue_mass=weights[...,:17].sum(-1) if i==0 else torch.zeros_like(gt))
        if 'clipped' in trace:
            clipped=trace['clipped'].bool()
            values['clip_fraction']=clipped.float().mean(-1)
        row=dict(stage=i+1)
        for name,subset in [('visible',mask&vis),('out_of_view',mask&~vis)]:
            row[name]=dict(count=int(subset.sum()),**{k+'_sum':float(v[subset].sum()) for k,v in values.items()})
        rows.append(row)
    return dict(coarse_stage_diagnostics=rows,note='Native signed disparity d=-flow, including mirrored negative-disparity augmentation; coarse common valid anchors. Rescue mass is first 17 signed anchors. Clip fraction uses actual preclip flags.')


"""Reporting-only E1 diagnostics on explicitly common correspondence support."""
import torch
import torch.nn.functional as F


def correspondence_metrics(output, target, valid):
    """Native signed flow; report sums/counts so aggregation preserves support."""
    score = output['logits']
    pad = output['padding']
    h, w = target.shape[-2:]
    x = torch.arange(w, device=target.device)[None, None, None]
    mask = valid.bool()
    if mask.ndim == 3:
        mask = mask[:, None]
    inside = mask & torch.isfinite(target) & (x + target >= 0) & (x + target <= w - 1)
    gt = F.pad(target, pad)[..., ::4, ::4][:, 0]
    support = F.pad(inside, pad)[..., ::4, ::4][:, 0]
    key = torch.arange(score.shape[-1], device=score.device)
    truth_x = key[None, None, :] + gt / 4
    nearest = truth_x.round().long().clamp(0, score.shape[-1] - 1)
    key_valid = output['key_valid']
    support = support & output['query_valid'].bool() & key_valid.gather(-1, nearest).bool()
    truth = score.gather(-1, nearest[..., None]).squeeze(-1)
    ranked = score.masked_fill(~key_valid[..., None, :].bool(), -torch.inf)
    rank = ((ranked > truth[..., None]).sum(-1) + 1).float()
    top1 = (ranked.argmax(-1) - truth_x).abs() * 4
    n = int(support.sum())
    result = dict(count=n, raw_target_rank_sum=float(rank[support].sum()), raw_top1_epe_sum=float(top1[support].sum()))
    for name, field in [('initial', 'initial_flow'), ('global', 'global_flow'), ('final', 'flow')]:
        error = (output[field] - target).abs()
        coarse = F.pad(error, pad)[..., ::4, ::4][:, 0]
        result[name + '_common_epe_sum'] = float(coarse[support].sum())
        result[name + '_valid_epe_sum'] = float(error[mask].sum())
    result['valid_count'] = int(mask.sum())
    return result


class CodingDiagnostics:
    """Observe existing coded-site modules without changing forward or gradients."""
    def __init__(self, model):
        self.model = model
        self.rows = {}
        self.handles = []
        for name, module in model.named_modules():
            if module.__class__.__name__ in ('SpikeProjection', 'SpikeHidden'):
                self.handles.append(module.register_forward_hook(self._hook(name)))

    def _hook(self, name):
        def hook(module, inputs, output):
            z = output.detach().float()
            row = self.rows.setdefault(name, dict(count=0, zero=0, saturated=0, squared_sum=0.0, vector_norm_sum=0.0, vector_count=0))
            row['count'] += z.numel()
            row['zero'] += int((z == 0).sum())
            row['saturated'] += int((z.abs() >= 1).sum())
            row['squared_sum'] += float(z.square().sum())
            norms=z.norm(dim=-1)
            row['vector_norm_sum'] += float(norms.sum())
            row['vector_count'] += norms.numel()
        return hook

    def snapshot(self, reset=True):
        result = {}
        modules = dict(self.model.named_modules())
        for name, row in self.rows.items():
            count = row['count']
            grads = [p.grad.detach().float().square().sum() for p in modules[name].parameters() if p.grad is not None]
            result[name] = dict(count=count, zero_fraction=row['zero']/count, saturated_fraction=row['saturated']/count,
                                rms=(row['squared_sum']/count)**0.5, mean_last_axis_l2=row['vector_norm_sum']/row['vector_count'],
                                gradient_norm=float(torch.stack(grads).sum().sqrt()) if grads else None)
        if reset:
            self.rows = {}
        return result

    def close(self):
        for handle in self.handles:
            handle.remove()
