"""Reporting-only E1 diagnostics on explicitly common correspondence support."""
import torch
import torch.nn.functional as F


def correspondence_metrics(output,target,valid):
    """Literal top5, strict-greater rank, common native stride4 in-view anchors."""
    import hashlib
    score=output['logits'];probability=output['probabilities'];pad=output['padding'];h,w=target.shape[-2:]
    mask=valid[:,None].bool() if valid.ndim==3 else valid.bool()
    x=torch.arange(w,device=target.device)[None,None,None];inside=mask&(x+target>=0)&(x+target<=w-1)
    gt=F.pad(target,pad)[...,::4,::4][:,0];support=F.pad(inside,pad)[...,::4,::4][:,0]
    keys=torch.arange(score.shape[-1],device=score.device);truth_x=keys[None,None,:]+gt/4;nearest=truth_x.round().long().clamp(0,len(keys)-1)
    support=support&output['query_valid'].bool()&output['key_valid'].gather(-1,nearest).bool()
    result=dict(count=int(support.sum()),support_digest=hashlib.sha256(support.detach().cpu().contiguous().numpy().tobytes()).hexdigest())
    for name,values in [('raw',score),('post_ot',probability)]:
        ranked=values.masked_fill(~output['key_valid'][...,None,:].bool(),-torch.inf)
        top=ranked.topk(5,-1).indices;errors=(top-truth_x[...,None]).abs()*4
        gt_score=ranked.gather(-1,nearest[...,None]).squeeze(-1);rank=(ranked>gt_score[...,None]).sum(-1)+1
        result[name+'_top1_epe_sum']=float(((ranked.argmax(-1)-truth_x).abs()*4)[support].sum());result[name+'_target_rank_sum']=float(rank[support].sum());result[name+'_nearest5_epe_sum']=float(errors.min(-1).values[support].sum())
    for name,field in [('initial','initial_flow'),('global','global_flow'),('final','flow')]:
        error=(output[field]-target).abs();coarse=F.pad(error,pad)[...,::4,::4][:,0]
        result[name+'_common_epe_sum']=float(coarse[support].sum());result[name+'_valid_epe_sum']=float(error[mask].sum())
        for group,groupmask in [('visible',inside),('out_of_view',mask&~inside)]:result[name+'_'+group+'_epe_sum']=float(error[groupmask].sum())
    result['valid_count']=int(mask.sum());result['visible_count']=int(inside.sum());result['out_of_view_count']=int((mask&~inside).sum())
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
