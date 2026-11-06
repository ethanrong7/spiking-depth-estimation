"""Cycle diagnostics outside training windows; no optimizer or model changes."""
import math
import statistics


def groups(arm):
    result=dict(feature_encoder=('fnet.',),context_encoder=('cnet.',),disparity_head=('update_block.disp_head.',))
    if arm=='spiking_normalized':
        result.update({f'updater_scale_{i}':(f'update_block.cells.{i}.',) for i in range(3)})
    else:
        result.update({f'updater_scale_{scale}':(f'update_block.snn{scale}.',) for scale in ('08','16','32')})
        if arm=='original_I':result.update({f'current_projection_{scale}':(f'update_block.snn{scale}.current_proj.',) for scale in ('08','16','32')})
    return result


def parameters_cpu(model):return {name:p.detach().cpu().clone() for name,p in model.named_parameters() if p.requires_grad}


def parameter_health(model,before,arm):
    after=parameters_cpu(model);parameters=[]
    for name,p in model.named_parameters():
        if name not in before:continue
        first=before[name].double();last=after[name].double();delta=last-first
        old_l2=float(first.square().sum())**.5;new_l2=float(last.square().sum())**.5;change=float(delta.square().sum())**.5
        grad=p.grad.detach().cpu().double() if p.grad is not None else None
        parameters.append(dict(name=name,numel=p.numel(),delta_l2=change,relative_delta_l2=change/max(old_l2,new_l2,1e-30),
            nonzero_delta_elements=int((delta!=0).sum()),before_l2=old_l2,after_l2=new_l2,
            latest_clipped_gradient_l2=float(grad.square().sum())**.5 if grad is not None else 0.,gradient_present=grad is not None,
            latest_clipped_gradient_nonzero_elements=int((grad!=0).sum()) if grad is not None else 0,latest_clipped_gradient_finite=grad is None or bool(grad.isfinite().all()),
            finite=bool(delta.isfinite().all()) and bool(last.isfinite().all()) and (grad is None or bool(grad.isfinite().all()))))
    required={}
    for group,prefixes in groups(arm).items():
        selected=[p for p in parameters if p['name'].startswith(prefixes)]
        if not selected:raise RuntimeError('required named group missing: '+group)
        change=sum(p['delta_l2']**2 for p in selected)**.5;old=sum(p['before_l2']**2 for p in selected)**.5;new=sum(p['after_l2']**2 for p in selected)**.5
        required[group]=dict(parameter_names=[p['name'] for p in selected],parameter_tensors=len(selected),delta_l2=change,relative_delta_l2=change/max(old,new,1e-30),
            nonzero_delta_elements=sum(p['nonzero_delta_elements'] for p in selected),changed_tensors=sum(p['nonzero_delta_elements']>0 for p in selected),
            latest_clipped_gradient_l2=sum(p['latest_clipped_gradient_l2']**2 for p in selected)**.5,
            finite=all(p['finite'] for p in selected),passes=change>0 and all(p['finite'] for p in selected))
    return dict(status='PASS' if all(p['passes'] for p in required.values()) else 'FAIL',required_groups=required,parameters=parameters,
        relative_norm_definition='deltaL2/max(previousL2,currentL2,1e-30)',gradient_definition='latest update clipped parameter gradients sampled outside the cycle window; every update global norm is measured before clipping'),after


def quantile(values,q):
    x=sorted(values);position=(len(x)-1)*q;lo=int(position);hi=min(lo+1,len(x)-1)
    return x[lo]+(x[hi]-x[lo])*(position-lo)


def correlation(a,b):
    if len(a)!=len(b) or len(a)<2:return None
    ma=statistics.mean(a);mb=statistics.mean(b);aa=sum((x-ma)**2 for x in a);bb=sum((y-mb)**2 for y in b)
    return sum((x-ma)*(y-mb) for x,y in zip(a,b))/(aa*bb)**.5 if aa>0 and bb>0 else None


def learning_health(rows):
    if len(rows)!=480:raise RuntimeError('exactly four complete120-update cycles required')
    cycles=[]
    for cycle in range(4):
        chunk=rows[120*cycle:120*(cycle+1)]
        cycles.append(dict(cycle=cycle+1,start_update=120*cycle+1,end_update=120*(cycle+1),mean_loss=statistics.mean(r['loss'] for r in chunk),mean_train_epe=statistics.mean(r['train_epe'] for r in chunk),loss_sd=statistics.pstdev(r['loss'] for r in chunk),mean_preclip_gradient_norm=statistics.mean(r['gradient_norm'] for r in chunk)))
    first_loss=statistics.mean(c['mean_loss'] for c in cycles[:2]);last_loss=statistics.mean(c['mean_loss'] for c in cycles[2:]);first_epe=statistics.mean(c['mean_train_epe'] for c in cycles[:2]);last_epe=statistics.mean(c['mean_train_epe'] for c in cycles[2:])
    losses=[r['loss'] for r in rows];norms=[r['gradient_norm'] for r in rows];ema=[];value=None
    for x in losses:value=x if value is None else value+2/101*(x-value);ema.append(value)
    logs=[math.log(x) for x in losses] if all(x>0 for x in losses) else None
    phase_profile=[]
    for phase in range(120):phase_profile.append(statistics.mean(rows[cycle*120+phase]['loss']-cycles[cycle]['mean_loss'] for cycle in range(4)))
    return dict(status='PASS' if last_loss<first_loss and last_epe<first_epe else 'FAIL',cycles=cycles,
        first_two_cycle_mean_loss=first_loss,last_two_cycle_mean_loss=last_loss,first_two_cycle_mean_train_epe=first_epe,last_two_cycle_mean_train_epe=last_epe,
        loss_lag120_correlation=correlation(losses[:-120],losses[120:]),log_loss_lag120_correlation=correlation(logs[:-120],logs[120:]) if logs else None,
        phase_centered_loss_profile=phase_profile,phase_profile_sd=statistics.pstdev(phase_profile),raw_loss_sd=statistics.pstdev(losses),ema100=ema,
        preclip_gradient_norm=dict(minimum=min(norms),median=statistics.median(norms),mean=statistics.mean(norms),p95=quantile(norms,.95),maximum=max(norms),clipping_fraction=sum(x>1 for x in norms)/len(norms)),
        monotonic_raw_loss_required=False,criterion='mean cycles3+4 loss and trainingEPE lower than cycles1+2; no extension or tuning on failure')
