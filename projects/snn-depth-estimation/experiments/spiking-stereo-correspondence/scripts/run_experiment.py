"""Registered cluster-only experiment runner; native Synthetic Train/Dev only."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
import time

STARTED = time.monotonic()
PROJECT = Path(__file__).resolve().parents[3]
REPOSITORY = PROJECT.parents[1]
STUDY = Path(__file__).resolve().parents[1]
BASE_ID = 'cb43497fdc02bf612794dd48d516ee71e5658f10969e47ed6f4683591f3e8ec5'
MANIFEST_SHA = 'bad4369b6fd8b0cdf871a20b1727c2d9b59733a16bd33f75904227bdfef4510a'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def write(path, value):
    path = Path(path); temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); temp.replace(path)


def append(path, value):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(value, allow_nan=False) + '\n')


def validate_config(config):
    required = dict(study='spiking-stereo-correspondence', seed=1234, time_steps=50,
        widths=[32, 48, 64], heads=4, window=4, max_width=400,
        effective_batch=8, lr=.0002, weight_decay=1e-5, epsilon=1e-8,
        clip_grad=1., schedule_updates=10000, test_access='sealed')
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError('registered architecture/recipe mismatch: ' + key)
    if config.get('stage') not in ('preflight', 'sanity', 'main', 'controls'):
        raise ValueError('unsupported stage')
    if config.get('physical_batch') not in (1, 2):
        raise ValueError('physical2 or declared physical1 fallback required')
    if config['stage'] == 'main' and config.get('updates') not in (1000, 2500, 5000, 7500, 10000):
        raise ValueError('unregistered main endpoint')
    if config['stage'] == 'sanity' and config.get('updates') != 128:
        raise ValueError('fixed diagnostic exposure required')
    if config.get('model') not in ('correspondence', 'original_I'):
        raise ValueError('unsupported model')
    if config['model'] != 'correspondence':
        raise ValueError('baseline adapter is verification-only; no new baseline training registered')
    return config


def load_components(base_source):
    """Keep all historical dataset/baseline dependencies inside their capsule."""
    recipe_path = PROJECT / 'experiments/spiking-transformer-stereo/campaigns/recipe-aligned-three-way-v1/scripts/recipe.py'
    recipe = load_module('correspondence_historical_recipe', recipe_path)
    driver, mechanism, registry = recipe.load_frozen(base_source)
    if registry['source_identity_sha256'] != BASE_ID:
        raise RuntimeError('exact historical ALIF+I capsule required')
    # Import this standalone new module without routing `core` into the dirty checkout.
    scientific = load_module('registered_correspondence_model', PROJECT / 'core/spiking_stereo_correspondence.py')
    return recipe, driver, mechanism, registry, scientific


def setup(args):
    import numpy as np
    import torch
    config = validate_config(json.loads(args.config.read_text()))
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('all experiment work requires Slurm CUDA')
    if socket.gethostname() != 'cccl3' or torch.cuda.device_count() != 1 or not any(x in torch.cuda.get_device_name(0) for x in ('4090', '5090')):
        raise RuntimeError('verified cccl3 single 4090/5090 required')
    if digest(args.manifest) != MANIFEST_SHA:
        raise RuntimeError('historical dataset manifest changed')
    source = json.loads(args.source_registry.read_text())
    for name, expected in source['source_files'].items():
        if digest(args.source_registry.parent / name) != expected:
            raise RuntimeError('frozen source changed: ' + name)
    config_relative = str(args.config.resolve().relative_to(args.source_registry.parent.resolve()))
    if source['source_files'].get(config_relative) != digest(args.config):
        raise RuntimeError('configuration not registered in source capsule')
    recipe, driver, mechanism, base, scientific = load_components(args.base_source)
    manifest = json.loads(args.manifest.read_text())
    if len(manifest['train']) != 960 or len(manifest['dev']) != 60:
        raise RuntimeError('historical Train960/Dev60 cohort required')
    if args.output.name != config['run_id']:
        raise RuntimeError('output run ID differs from registered configuration')
    args.output.mkdir(parents=True, exist_ok=False)
    seed = config['seed']; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = True
    model = scientific.SpikingStereoCorrespondence(time_steps=50, widths=config['widths'], heads=4,
        window=4, max_width=400, activation_checkpoint=True).cuda()
    train = driver.ManifestDataset(args.data_root, manifest['train'])
    dev = driver.ManifestDataset(args.data_root, manifest['dev'])
    metadata = dict(config=config, command=sys.argv, source=source, base_source=base,
        manifest_sha256=MANIFEST_SHA, config_sha256=digest(args.config), test_access='sealed',
        gpu=torch.cuda.get_device_name(0), torch=torch.__version__, cuda=torch.version.cuda,
        python=sys.version, hostname=socket.gethostname(), user=os.environ.get('USER'),
        slurm={k:v for k,v in os.environ.items() if k.startswith('SLURM_')},
        environment=sys.prefix, started_utc=datetime.now(timezone.utc).isoformat(),
        parameters=sum(p.numel() for p in model.parameters()),
        pip_freeze=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True),
        data_semantics='Historical full50-bin Synthetic sample to paired EXR; timestamp not independently verified; no per-bin GT',
        precision='FP32, cuDNN TF32 enabled and matmul TF32 disabled',
        activity_columns=['encoder_stride2','encoder_stride4','encoder_stride8','cross_query','cross_key','cross_value','cross_output'],
        analog_components=['normalization', 'attention normalization', 'temporal weighting', 'correspondence softmax', 'subpixel expectation', 'null/refinement decoder', 'interpolation'])
    write(args.output/'metadata.json', metadata)
    return config, model, train, dev, recipe, scientific


def stack(items):
    import torch
    return [torch.stack([row[index] for row in items]).cuda() for index in range(4)]


def valid_count(items):
    return sum(int(row[3].sum()) for row in items)


def gradient_summary(model):
    import torch
    names = ('encoder', 'self_qkv', 'cross_qkv', 'temporal_logits', 'readout')
    result = {}
    for prefix in names:
        gradients = [p.grad.detach() for n,p in model.named_parameters() if n.startswith(prefix) and p.grad is not None]
        result[prefix] = float(torch.stack([g.square().sum() for g in gradients]).sum().sqrt()) if gradients else 0.
    return result


def train_batch(model, items, physical, scientific, capture_input_gradient=False, capture_activity=False):
    import torch
    total_valid = valid_count(items)
    total_inframe = total_correspondence = 0
    for row in items:
        truth, mask = row[2][0], row[3]
        x = torch.arange(truth.shape[-1])[None, :]
        inside = mask & torch.isfinite(truth) & (truth.abs() < 512) & (x + truth >= 0) & (x + truth <= truth.shape[-1] - 1)
        total_inframe += int(inside.sum())
        total_correspondence += int(inside[::4, ::4].sum())
    denominators = dict(valid=total_valid, inframe=total_inframe, correspondence=total_correspondence)
    if not total_valid:
        raise RuntimeError('empty effective batch')
    losses = {}; error_sum = 0.; activity = None; activity_per_time = None; temporal_gradients = {}
    for offset in range(0, len(items), physical):
        left, right, target, valid = stack(items[offset:offset+physical])
        if capture_input_gradient:
            left.requires_grad_(True); right.requires_grad_(True)
        output = model(left, right)
        loss, terms = scientific.correspondence_loss(output, target, valid, denominators)
        weight = valid_count(items[offset:offset+physical]) / total_valid
        weighted = loss
        if not torch.isfinite(weighted):
            raise FloatingPointError('nonfinite loss')
        weighted.backward()
        if capture_input_gradient:
            for side, value in (('left',left),('right',right)):
                per_time = value.grad.detach().abs().sum((0,2,3))
                temporal_gradients[side] = per_time if side not in temporal_gradients else temporal_gradients[side] + per_time
        for key, value in terms.items():
            if key.endswith('pixels'): continue
            losses[key] = losses.get(key, 0.) + float(value.detach())
        error_sum += float((output['flow'][:, 0] - target[:, 0]).abs()[valid].sum().detach())
        observation_weight=len(items[offset:offset+physical])/len(items)
        activity = output['activity'].detach() * observation_weight if activity is None else activity + output['activity'].detach() * observation_weight
        if capture_activity:
            rates=output['activity_per_time'].detach()*observation_weight
            activity_per_time=rates if activity_per_time is None else activity_per_time+rates
        del output, loss, weighted, left, right, target, valid
    return dict(loss=losses['final_l1']+.5*losses['coarse_l1']+.1*losses['correspondence_ce']+.1*losses['null_bce'],
                epe=error_sum/total_valid, loss_terms=losses, activity=activity.tolist(),
                spike_rates_per_time=activity_per_time.tolist() if activity_per_time is not None else None,
                input_temporal_gradient_l1={k:v.tolist() for k,v in temporal_gradients.items()},
                loss_supports=dict(valid_pixels=total_valid,inframe_pixels=total_inframe,correspondence_pixels=total_correspondence))


def evaluate(model, dataset, indices=None, intervention=None):
    import numpy as np
    import torch
    indices = list(range(len(dataset))) if indices is None else list(indices)
    rows=[]; diagnostic_rows=[]; model.eval()
    with torch.no_grad():
        for position, index in enumerate(indices):
            item=dataset[index]; left,right,target,valid=stack([item])
            if intervention == 'zero_right': right.zero_()
            elif intervention == 'shuffle_right': right=dataset[indices[(position+len(indices)//2)%len(indices)]][1][None].cuda()
            elif intervention and intervention.startswith('shift_'):
                shift=int(intervention.split('_')[1]); right=torch.roll(right,shift,-1)
                if shift>0: right[...,:shift]=0
                else: right[...,shift:]=0
            output=model(left,right); flow=output['flow']
            if not torch.isfinite(flow).all(): raise FloatingPointError('nonfinite evaluation')
            error=(flow[:,0]-target[:,0]).abs()[valid]
            if not error.numel():raise RuntimeError('empty evaluation frame')
            diagnostic_rows.append(output_diagnostics(output,target,valid))
            source=dataset.rows[index]
            rows.append(dict(scene=source['scene'],frame=source['frame'],valid_pixels=error.numel(),total_pixels=valid.numel(),
                epe=float(error.mean()),error_sum=float(error.sum()),
                mean_flow=float(flow[:,0][valid].mean()),
                **{f'bad{k}':float((error>k).float().mean()*100) for k in (1,3,5)}))
    result=aggregate_metrics(rows)
    result['diagnostics']=aggregate_diagnostics(diagnostic_rows)
    result['diagnostics']['per_frame']=[dict(scene=row['scene'],frame=row['frame'],**diagnostic) for row,diagnostic in zip(rows,diagnostic_rows)]
    return result


def output_diagnostics(output,target,valid):
    """Saved-forward scalar diagnostics; no extra model calls or filtered metrics."""
    import torch
    if valid.ndim==3:valid=valid[:,None]
    valid=valid.bool() & torch.isfinite(target) & (target.abs()<512)
    width=target.shape[-1]
    x=torch.arange(width,device=target.device)[None,None,None,:]
    inframe=valid & (x+target>=0) & (x+target<=width-1)
    outframe=valid & ~inframe
    null=output['null_probability']
    result=dict(valid_pixels=int(valid.sum()),inframe_pixels=int(inframe.sum()),outframe_pixels=int(outframe.sum()),correspondence_pixels=int(inframe[...,::4,::4].sum()),
        match_inframe_error_sum=float((output['coarse_flow']-target).abs()[inframe].sum()),
        weighted_null_abs_sum=float((null*output['null_flow']).abs()[valid].sum()),
        residual_abs_sum=float(output['residual'].abs()[valid].sum()),
        temporal_weights=output['temporal_weights'].detach().tolist(),
        spike_rates_per_time=output['activity_per_time'].detach().tolist() if 'activity_per_time' in output else None)
    for label,mask in (('inframe',inframe),('outframe',outframe)):
        result['null_'+label+'_sum']=float(null[mask].sum())
        result['null_'+label+'_above_half_count']=int((null[mask]>.5).sum())
    probabilities=output['probabilities']
    entropy=-(probabilities*probabilities.clamp_min(1e-12).log()).sum(-1)
    query_valid=output['query_valid']
    result['entropy_sum']=float(entropy[query_valid].sum())
    result['entropy_queries']=int(query_valid.sum())
    return result


def aggregate_diagnostics(rows):
    counts={key:sum(r[key] for r in rows) for key in ('valid_pixels','inframe_pixels','outframe_pixels','correspondence_pixels','entropy_queries')}
    ratio=lambda numerator,denominator:sum(r[numerator] for r in rows)/counts[denominator] if counts[denominator] else None
    result=dict(supports=counts,match_inframe_epe=ratio('match_inframe_error_sum','inframe_pixels'),
        mean_abs_weighted_null_contribution=ratio('weighted_null_abs_sum','valid_pixels'),
        mean_abs_residual_contribution=ratio('residual_abs_sum','valid_pixels'),
        correspondence_entropy_nats=ratio('entropy_sum','entropy_queries'),temporal_weights=rows[0]['temporal_weights'])
    if rows[0].get('spike_rates_per_time') is not None:
        import numpy as np
        result['spike_rates_per_time']=np.mean([r['spike_rates_per_time'] for r in rows],axis=0).tolist()
    for label in ('inframe','outframe'):
        result['null_'+label+'_mean']=ratio('null_'+label+'_sum',label+'_pixels')
        result['null_'+label+'_above_half_fraction']=ratio('null_'+label+'_above_half_count',label+'_pixels')
    return result


def aggregate_metrics(rows):
    """Same equal-frame/equal-scene and valid-pixel aggregates as frozen driver."""
    import numpy as np
    scene={s:float(np.mean([r['epe'] for r in rows if r['scene']==s])) for s in sorted({r['scene'] for r in rows})}
    count=sum(r['valid_pixels'] for r in rows)
    total_pixels=sum(r.get('total_pixels',0) for r in rows)
    return dict(mean_scene_epe=float(np.mean(list(scene.values()))),pooled_epe=sum(r.get('error_sum', r['epe']*r['valid_pixels']) for r in rows)/count,
        valid_pixels=count,valid_fraction=count/total_pixels if total_pixels else None,scene_epe=scene,per_frame=rows,median_scene_epe=float(np.median(list(scene.values()))),
        worst_five_scene_epe=float(np.mean(sorted(scene.values(),reverse=True)[:5])),
        **{f'bad{k}':sum(r[f'bad{k}']*r['valid_pixels'] for r in rows)/count for k in (1,3,5)})


def preflight(args, config, model, train, dev, recipe, scientific):
    import torch
    order=torch.randperm(len(train),generator=torch.Generator().manual_seed(config['seed'])).tolist()
    decode_started=time.monotonic()
    items=[recipe.augment(train[index],*recipe.paired_flags(config['seed'],i)) for i,index in enumerate(order[:8])]
    decode_seconds=time.monotonic()-decode_started
    trials=[]; selected=None
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    for physical in (2,1):
        model.load_state_dict(initial);model.zero_grad(set_to_none=True);model.train()
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=1e-5,eps=1e-8)
        scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=config['lr'],total_steps=10100,pct_start=.01,
            cycle_momentum=False,anneal_strategy='linear',div_factor=25.,final_div_factor=1e4)
        begun=time.monotonic();updates=[]
        try:
            for update in range(2):
                optimizer.zero_grad(set_to_none=True)
                step_started=time.monotonic()
                record=train_batch(model,items,physical,scientific,capture_input_gradient=True)
                gradient=gradient_summary(model)
                norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
                if not math_isfinite(norm) or norm<=0:raise FloatingPointError('invalid preflight gradient')
                before={n:p.detach().clone() for n,p in model.named_parameters() if n.startswith('cross_qkv')}
                optimizer.step();scheduler.step()
                delta=float(torch.stack([(p.detach()-before[n]).square().sum() for n,p in model.named_parameters() if n in before]).sum().sqrt())
                torch.cuda.synchronize()
                state_finite=all(bool(torch.isfinite(v).all()) for state in optimizer.state.values() for v in state.values() if torch.is_tensor(v))
                params_finite=all(bool(torch.isfinite(v).all()) for v in model.state_dict().values())
                if not state_finite or not params_finite:raise FloatingPointError('nonfinite optimizer/model state')
                updates.append(dict(engineering_update=update+1,step_seconds=time.monotonic()-step_started,
                    grad_norm=norm,gradient_groups=gradient,qkv_parameter_update_l2=delta,
                    optimizer_state_finite=state_finite,parameters_finite=params_finite,
                    moments_resident_before_backward=update>0,**record))
                progress=dict(stage='preflight',physical_batch=physical,
                    optimizer_steps=update+1,scheduler_steps=scheduler.last_epoch,
                    elapsed_seconds=time.monotonic()-STARTED,timestamp_utc=datetime.now(timezone.utc).isoformat(),
                    **updates[-1])
                append(args.output/'preflight.jsonl',progress)
                write(args.output/'preflight-summary.json',dict(latest=progress,scientific_training_updates=0))
            train_seconds=(time.monotonic()-begun)/2
            validation_started=time.monotonic();validation=evaluate(model,dev,[0,1]);torch.cuda.synchronize()
            validation_seconds=time.monotonic()-validation_started
            # Check complete predictions, not merely their aggregate metrics.
            model.eval();left,right,_,_=stack([dev[0],dev[1]])
            with torch.no_grad():
                first=model(left,right)['flow'];second=model(left,right)['flow']
            exact=bool(torch.equal(first,second))
            if not exact:raise RuntimeError('full prediction reset/repeat differs')
            counters=[int(v['step'].item()) for v in optimizer.state.values() if 'step' in v]
            if not counters or min(counters)!=2 or max(counters)!=2 or scheduler.last_epoch!=2:
                raise RuntimeError('preflight optimizer/scheduler counters differ')
            trials.append(dict(status='PASS',physical_batch=physical,train_seconds=train_seconds,
                decode_seconds_for_train8=decode_seconds,train_timing_excludes_decode=True,
                dev_two_seconds=validation_seconds,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),engineering_updates=updates,
                post_update_dev2_validation=validation,reset_repeat_exact=True,
                optimizer_steps=2,scheduler_steps=2))
            selected=physical;break
        except torch.cuda.OutOfMemoryError as error:
            trials.append(dict(status='OOM',physical_batch=physical,seconds=time.monotonic()-begun,error=str(error),
                completed_engineering_updates=len(updates),fallback_restores_initial_weights=True))
            model.zero_grad(set_to_none=True)
        finally:
            del optimizer,scheduler
            torch.cuda.empty_cache()
        if time.monotonic()-STARTED>config['max_seconds']:break
    result=dict(status='PASS_PREFLIGHT' if selected else 'INFEASIBLE_REGISTERED_GEOMETRY',
        selected_physical_batch=selected,trials=trials,engineering_optimizer_updates=2 if selected else None,
        scientific_training_updates=0,engineering_presentations_per_successful_trial=16,
        weights_reused_for_training=False,elapsed_seconds=time.monotonic()-STARTED)
    write(args.output/'result.json',result)
    if not selected:raise RuntimeError(result['status'])


def math_isfinite(value):
    import math
    return math.isfinite(value)


def training(args, config, model, train, dev, recipe, scientific):
    import numpy as np
    import torch
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=1e-5,eps=1e-8)
    scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=config['lr'],total_steps=10100,pct_start=.01,
        cycle_momentum=False,anneal_strategy='linear',div_factor=25.,final_div_factor=1e4)
    sanity=config['stage']=='sanity'
    order=torch.randperm(len(train),generator=torch.Generator().manual_seed(config['seed'])).tolist()
    cohort=order[:2] if sanity else order
    eval_indices=order[:2] if sanity else None
    eval_dataset=train if sanity else dev
    write(args.output/'sample-order.json',dict(order=cohort,augmentation='none diagnostic' if sanity else 'historical recipe-hv-v1'))
    def timing(): return dict(elapsed_seconds=time.monotonic()-STARTED,timestamp_utc=datetime.now(timezone.utc).isoformat())
    history=[]; best=float('inf'); best_update=None
    timing_totals=dict(training_step_seconds=0.,decode_seconds=0.,evaluation_seconds=0.,checkpoint_seconds=0.)
    def validate(update):
        nonlocal best,best_update
        begun=time.monotonic();row=dict(update=update,**evaluate(model,eval_dataset,eval_indices),**timing())
        row['evaluation_seconds']=time.monotonic()-begun
        timing_totals['evaluation_seconds']+=row['evaluation_seconds']
        row['split']='train_diagnostic' if sanity else 'dev'
        row['scientific_training_updates']=0 if sanity else update
        row['diagnostics']['positive_logit_gain']=float(torch.nn.functional.softplus(model.logit_gain_raw).detach())
        history.append(row);append(args.output/'validation.jsonl',row);write(args.output/'validation-summary.json',dict(history=history,latest=row))
        if update>0 and row['mean_scene_epe']<best:
            best=row['mean_scene_epe'];best_update=update
            save(update,args.output/'best.pt')
        return row
    def save(update,path):
        checkpoint_started=time.monotonic()
        value=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
            updates=update,presentations=update*8,config=config,best=best,best_update=best_update,
            source_registry_sha256=digest(args.source_registry),manifest_sha256=MANIFEST_SHA,
            rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all()))
        temp=path.with_suffix('.tmp.pt');torch.save(value,temp);temp.replace(path)
        seconds=time.monotonic()-checkpoint_started;timing_totals['checkpoint_seconds']+=seconds
        append(args.output/'checkpoints.jsonl',dict(update=update,path=path.name,bytes=path.stat().st_size,seconds=seconds,**timing()))
    initial=validate(0)
    completed=0;last=None
    for step in range(config['updates']):
        if time.monotonic()-STARTED>config['max_seconds']:
            save(completed,args.output/'last.pt')
            write(args.output/'result.json',dict(status='TIME_CAP_INCOMPLETE',updates=completed,planned_updates=config['updates'],best=best if best_update else None,timing_totals=timing_totals,**timing()))
            return
        begun=time.monotonic();indices=recipe.batch_indices(cohort,step)
        decode_started=time.monotonic()
        flags=[(False,False)]*8 if sanity else [recipe.paired_flags(config['seed'],step*8+i) for i in range(8)]
        items=[recipe.augment(train[index],*flag) for index,flag in zip(indices,flags)]
        decode_seconds=time.monotonic()-decode_started
        model.train();optimizer.zero_grad(set_to_none=True)
        values=train_batch(model,items,config['physical_batch'],scientific,
            capture_activity=step<2 or step+1 in config['eval_updates'] or step+1==config['updates'])
        groups=gradient_summary(model)
        gradient=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
        if not math_isfinite(gradient) or gradient<=0:raise FloatingPointError('invalid gradient')
        before={n:p.detach().clone() for n,p in model.named_parameters() if n.startswith('cross_qkv')}
        lr=optimizer.param_groups[0]['lr'];optimizer.step();scheduler.step()
        qkv_update=float(torch.stack([(p.detach()-before[n]).square().sum() for n,p in model.named_parameters() if n in before]).sum().sqrt())
        completed=step+1
        if scheduler.last_epoch!=completed:raise RuntimeError('scheduler count mismatch')
        last=dict(update=completed,presentations=completed*8,stage=config['stage'],scientific_training_updates=0 if sanity else completed,sample_indices=indices,augmentation_flags=flags,
            lr=lr,lr_next=optimizer.param_groups[0]['lr'],grad_norm=gradient,clip_factor=min(1.,1./(gradient+1e-6)),gradient_groups=groups,
            qkv_parameter_update_l2=qkv_update,decode_seconds=decode_seconds,step_seconds=time.monotonic()-begun,**values,**timing())
        if values['spike_rates_per_time'] is not None:
            last['visit_state']=dict(forward_at_update=completed-1,parameter_snapshot_at_update=completed,
                temporal_weights=model.temporal_logits.detach().softmax(0).tolist(),
                positive_logit_gain=float(torch.nn.functional.softplus(model.logit_gain_raw).detach()))
        timing_totals['training_step_seconds']+=last['step_seconds'];timing_totals['decode_seconds']+=decode_seconds
        append(args.output/'training.jsonl',last)
        write(args.output/'training-summary.json',dict(latest=last,updates=completed,planned_updates=config['updates'],presentations=completed*8))
        if completed in config['eval_updates']:
            validate(completed);save(completed,args.output/'last.pt')
        elif completed%100==0:save(completed,args.output/'last.pt')
    if not history or history[-1]['update']!=completed:validate(completed)
    save(completed,args.output/'last.pt')
    counters=[int(v['step'].item()) for v in optimizer.state.values() if 'step' in v]
    if not counters or min(counters)!=completed or max(counters)!=completed:raise RuntimeError('optimizer count mismatch')
    parameters_finite=all(bool(torch.isfinite(v).all()) for v in model.state_dict().values())
    optimizer_state_finite=all(bool(torch.isfinite(v).all()) for state in optimizer.state.values() for v in state.values() if torch.is_tensor(v))
    if not parameters_finite or not optimizer_state_finite:raise FloatingPointError('nonfinite final model/optimizer state')
    write(args.output/'result.json',dict(status='COMPLETE_DIAGNOSTIC_DISCARD_WEIGHTS' if sanity else 'COMPLETE_MAIN',
        updates=completed,presentations=completed*8,scientific_training_updates=0 if sanity else completed,initial=initial,final=history[-1],best=best,best_update=best_update,
        optimizer_steps=completed,scheduler_steps=scheduler.last_epoch,parameters_finite=parameters_finite,optimizer_state_finite=optimizer_state_finite,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),weights_reused_for_training=False if sanity else None,timing_totals=timing_totals,
        timing_semantics='decode is a subset of training_step; evaluation/checkpoint are disjoint from training_step; process elapsed includes all imports/init/logging/overhead',**timing()))


# Prospective selection anchored to the executing, immutable P2 capsule.
FINAL2500_BINDING = dict(selection='final2500',filename='last.pt',updates=2500,presentations=20000,
    seed=1234,run_id='main-seed1234-u2500-p2-v1',training_physical_batch=2,
    training_source_identity_sha256='31ed4dcc25a0980c96b5df2b57fe090cbb229d378386f83e3b4badd618adcec3',
    training_source_registry_sha256='6057875217739e467a3e7f8dc519bbef779acc5eff62897332b6993ed701a046',
    training_config_canonical_sha256='7f489189bc3db23f6d880dfef68bc3552a93d20174fa7cb26de4348f0f6dcf25',
    training_config_file_sha256='e03e9cecb31d4b77c43aa0293e7f764deeb195e40b077f14982e740c400d83ea',
    manifest_sha256=MANIFEST_SHA)
PROBE_ATOL=1e-4
PROBE_RTOL=1e-5


def validate_controls_checkpoint(value,expected,filename):
    """Validate scalar checkpoint metadata before loading weights/panel forwards."""
    if expected!=FINAL2500_BINDING:raise ValueError('P3 registration must bind the exact prospective final2500 training identity')
    if Path(filename).name!='last.pt':raise ValueError('P3 selects final last.pt, never Dev-best')
    for key in ('updates','presentations'):
        if value.get(key)!=expected[key]:raise ValueError('actual checkpoint '+key+' differs from final2500')
    config=value.get('config',{})
    for key,required in dict(stage='main',seed=1234,run_id=expected['run_id'],updates=2500,effective_batch=8,physical_batch=2).items():
        if config.get(key)!=required:raise ValueError('actual checkpoint training config differs: '+key)
    canonical=hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if canonical!=expected['training_config_canonical_sha256']:raise ValueError('actual complete training configuration digest differs')
    if value.get('manifest_sha256')!=expected['manifest_sha256']:raise ValueError('actual training manifest differs')
    if value.get('source_registry_sha256')!=expected['training_source_registry_sha256']:
        raise ValueError('actual P2 training registry differs; diagnostic registry is a separate identity')
    return dict(expected,actual_training_config_canonical_sha256=canonical,actual_training_registry_sha256=value['source_registry_sha256'])


class ProbeInvariantError(RuntimeError):
    def __init__(self,evidence):
        self.evidence=evidence
        super().__init__('P3 operator invariants failed: '+', '.join(evidence['failed_checks']))


def probe_invariants(outputs,relative_bias,max_width):
    """Existing five forwards only; exact identities and declared FP32 algebra."""
    import torch
    intact=outputs['intact'];content=outputs['zero_content_qk'];value=outputs['zero_cross_value_evidence']
    exact={}
    for key in ('logits','probabilities','coarse_flow'):
        exact['intact_vs_value:'+key]=bool(torch.equal(intact[key],value[key]))
    for key in ('null_logit','null_probability','null_flow','residual','activity','activity_per_time'):
        exact['content_vs_value:'+key]=bool(torch.equal(content[key],value[key]))
    for name,output in outputs.items():
        for key in ('query_valid','temporal_weights'):
            exact[name+':'+key]=bool(torch.equal(intact[key],output[key]))
        exact[name+':analog_components']=intact['analog_components']==output['analog_components']
    width=content['logits'].shape[-1]
    x=torch.arange(width,device=content['logits'].device);relative=x[None,:]-x[:,None]
    prior=relative_bias[relative+max_width//4-1][None,None].expand_as(content['logits'])
    valid=content['query_valid'];geometry=valid[..., :,None]&valid[...,None,:]
    safe=geometry | ~valid[..., :,None]
    prior=prior.masked_fill(~safe,torch.finfo(prior.dtype).min)
    exact['zero_content_retains_relative_prior_and_mask']=bool(torch.equal(content['logits'],prior))
    derived={}
    def algebra(name,actual,expected):
        error=(actual-expected).abs().max()
        finite=bool(torch.isfinite(error))
        derived[name]=dict(passed=bool(torch.allclose(actual,expected,atol=PROBE_ATOL,rtol=PROBE_RTOL)),
            max_abs_error=float(error) if finite else 'nonfinite',atol=PROBE_ATOL,rtol=PROBE_RTOL)
    algebra('content_minus_value_flow',content['flow']-value['flow'],
        (1-value['null_probability'])*(content['coarse_flow']-intact['coarse_flow']))
    for name,output in outputs.items():
        reconstructed=(1-output['null_probability'])*output['coarse_flow']+output['null_probability']*output['null_flow']+output['residual']
        algebra(name+':final_branch_reconstruction',output['flow'],reconstructed)
    failed=[key for key,passed in exact.items() if not passed]+[key for key,row in derived.items() if not row['passed']]
    evidence=dict(exact=exact,derived=derived,failed_checks=failed,passed=not failed,
        tolerance_policy='Exact equality for same tensors; FP32 derived arithmetic atol1e-4+rtol1e-5*abs(expected), prospectively fixed')
    if failed:raise ProbeInvariantError(evidence)
    return evidence


def controls_state_unchanged(model,initial):
    import torch
    return all(torch.equal(initial[k],v.detach().cpu()) for k,v in model.state_dict().items())


def fixed_diagnostic_panel(rows, count=10):
    """First manifest target from each of the first sorted scenes, before outcomes."""
    scenes=sorted({row['scene'] for row in rows})[:count]
    if len(scenes)!=count:raise ValueError('diagnostic panel requires ten distinct Dev scenes')
    indices=[next(i for i,row in enumerate(rows) if row['scene']==scene) for scene in scenes]
    return [dict(position=i+1,index=index,scene=rows[index]['scene'],frame=rows[index]['frame'],
        shuffled_right_index=indices[(i+1)%count],shuffled_right_scene=rows[indices[(i+1)%count]]['scene'],
        save_map=i in (0,4,9)) for i,index in enumerate(indices)]


def shift_right_zero_fill(right, shift=4):
    import torch
    if not 0<shift<right.shape[-1]:raise ValueError('registered shift must be positive and within image')
    result=torch.zeros_like(right);result[...,shift:]=right[...,:-shift]
    return result


def common_shift_support(target,valid,shift=4):
    import torch
    if valid.ndim==3:valid=valid[:,None]
    width=target.shape[-1];x=torch.arange(width,device=target.device)[None,None,None,:]
    right=x+target
    return valid.bool() & torch.isfinite(target) & (target.abs()<512) & (right>=0) & (right<=width-1) & (right+shift>=0) & (right+shift<=width-1)


def branch_fields(output):
    null=output['null_probability']
    return dict(final=output['flow'],conditional_match=output['coarse_flow'],null_flow=output['null_flow'],
        weighted_null=null*output['null_flow'],weighted_match=(1-null)*output['coarse_flow'],residual=output['residual'])


def field_metrics(field,target,mask):
    count=int(mask.sum())
    if not count:return dict(pixels=0,epe=None,mean=None,mean_abs=None)
    return dict(pixels=count,epe=float((field-target).abs()[mask].mean()),mean=float(field[mask].mean()),mean_abs=float(field[mask].abs().mean()))


def shift_response(correct,shifted,mask,shift=4):
    values=(shifted-correct)[mask]/shift
    if not values.numel():return dict(pixels=0,mean=None,median=None)
    return dict(pixels=values.numel(),mean=float(values.mean()),median=float(values.median()))


def summarize_panel(records,complete=False):
    import numpy as np
    result=dict(scenes_completed=len(records),nonempty_common_support_scenes=sum(r.get('common_support_pixels',0)>0 for r in records),
        registered_panel_complete=False,functional_band=dict(assessable=False,passed=None,reason='Ten completed distinct scenes with nonempty common GT support are required'),shift_response={},shuffled_minus_correct={})
    if not records:return result
    for branch in records[0]['shift_response']:
        values=[r['shift_response'][branch]['mean'] for r in records if r['shift_response'][branch]['mean'] is not None]
        result['shift_response'][branch]=dict(equal_scene_mean=float(np.mean(values)) if values else None,
            median_scene_mean=float(np.median(values)) if values else None,positive_scenes=sum(v>0 for v in values),scenes=len(values))
        differences=[r['interventions']['shuffle_right']['full_gt'][branch]['epe']-r['interventions']['intact']['full_gt'][branch]['epe'] for r in records]
        result['shuffled_minus_correct'][branch]=dict(equal_scene_epe_difference=float(np.mean(differences)),per_scene=differences)
    eligible=complete and len(records)==10 and len({r.get('scene') for r in records})==10 and result['nonempty_common_support_scenes']==10
    result['registered_panel_complete']=eligible
    if eligible:
        final=result['shift_response']['final'];shuffled=result['shuffled_minus_correct']['final']['equal_scene_epe_difference']
        result['functional_band']=dict(assessable=True,passed=.5<=final['median_scene_mean']<=1.5 and final['positive_scenes']>=8 and shuffled>0,
            branch='final',median_scene_response=final['median_scene_mean'],positive_scenes=final['positive_scenes'],
            shuffled_minus_correct_epe=shuffled,criterion='median scene-mean response in[.5,1.5],>=8positive scenes,higher shuffled EPE; descriptive only')
    return result


def render_diagnostic_maps(path, arrays):
    """Fixed examples only; array bundle retains every branch/intervention."""
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    valid=arrays['valid'].astype(bool)
    gt=np.where(valid,arrays['truth'],np.nan)
    limits=np.nanquantile(gt,[.01,.99])
    panels=[('Left 50-bin mean',arrays['left_rate'],'gray',None),('GT signed flow',gt,'coolwarm',limits),
        ('Final intact',arrays['intact_final'],'coolwarm',limits),('Conditional match',arrays['intact_conditional_match'],'coolwarm',limits),
        ('Shift +4 final',arrays['shift_plus4_final'],'coolwarm',limits),('Zero QK final',arrays['zero_content_qk_final'],'coolwarm',limits),
        ('Zero cross-value evidence',arrays['zero_cross_value_evidence_final'],'coolwarm',limits),('Null probability',arrays['intact_null_probability'],'viridis',(0,1)),
        ('Residual contribution',arrays['intact_residual'],'coolwarm',(-4,4))]
    fig,axes=plt.subplots(3,3,figsize=(12,7.5),constrained_layout=True)
    for axis,(title,value,cmap,clim) in zip(axes.flat,panels):
        opts={} if clim is None else dict(vmin=clim[0],vmax=clim[1])
        image=axis.imshow(value,cmap=cmap,**opts);axis.set_title(title,fontsize=10);axis.set_axis_off();fig.colorbar(image,ax=axis,fraction=.025)
    fig.savefig(path,dpi=150);plt.close(fig)


def controls(args,config,model,train,dev,recipe,scientific):
    import numpy as np
    import torch
    if not args.checkpoint or not args.checkpoint_sha256 or digest(args.checkpoint)!=args.checkpoint_sha256:
        raise RuntimeError('controls require exact selected checkpoint and SHA256')
    value=torch.load(args.checkpoint,map_location='cuda',weights_only=False)
    try:
        checkpoint_binding=validate_controls_checkpoint(value,config.get('expected_checkpoint'),args.checkpoint.name)
    except ValueError as error:
        write(args.output/'result.json',dict(status='FAILED_CHECKPOINT_BINDING',checkpoint=str(args.checkpoint),
            checkpoint_sha256=args.checkpoint_sha256,error=str(error),elapsed_seconds=time.monotonic()-STARTED))
        raise
    model.load_state_dict(value['model']);model.eval()
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    panel=fixed_diagnostic_panel(dev.rows)
    write(args.output/'panel.json',dict(selection='first manifest target in first ten sorted Dev scenes',
        manifest_sha256=MANIFEST_SHA,entries=panel,shift_native_pixels=4,map_positions=[1,5,10],
        shuffle='next scene cyclically, all50 bins',actual_inference_batch=1,gradient_accumulation=False,
        training_physical_batch=checkpoint_binding.get('training_physical_batch',2),test_access='sealed'))
    items={row['index']:dev[row['index']] for row in panel}
    records=[];maps=args.output/'maps';maps.mkdir()
    with torch.no_grad():
        for entry in panel:
            left,right,target,valid=stack([items[entry['index']]])
            common=common_shift_support(target,valid,4);full=valid[:,None].bool()
            shuffled=items[entry['shuffled_right_index']][1][None].cuda()
            outputs=dict(intact=model(left,right),shuffle_right=model(left,shuffled),
                shift_plus4=model(left,shift_right_zero_fill(right,4)),
                zero_content_qk=model(left,right,zero_content=True),zero_cross_value_evidence=model(left,right,zero_values=True))
            try:
                invariant_evidence=probe_invariants(outputs,model.relative_bias,model.max_width)
            except ProbeInvariantError as error:
                failure=dict(status='FAILED_OPERATOR_INVARIANT',checkpoint_binding=checkpoint_binding,
                    checkpoint=str(args.checkpoint),checkpoint_sha256=args.checkpoint_sha256,failed_scene=entry,
                    invariant_evidence=error.evidence,model_state_unchanged=controls_state_unchanged(model,initial),
                    actual_inference_batch=1,gradient_accumulation=False,records=records,
                    summary=summarize_panel(records),elapsed_seconds=time.monotonic()-STARTED)
                append(args.output/'controls-failures.jsonl',failure);write(args.output/'result.json',failure)
                raise
            interventions={}
            for name,output in outputs.items():
                fields=branch_fields(output)
                comparison_target=target+4 if name=='shift_plus4' else target
                record=dict(common_support={branch:field_metrics(field,comparison_target,common) for branch,field in fields.items()},
                    route_diagnostics=output_diagnostics(output,comparison_target,common[:,0] if name=='shift_plus4' else valid),
                    diagnostic_support='common_gt_intersection' if name=='shift_plus4' else 'full_original_valid',
                    entropy_support='geometry_valid_stride4_queries',
                    branch_sum_max_abs_error=float((fields['final']-fields['weighted_match']-fields['weighted_null']-fields['residual']).abs().max()))
                if name!='shift_plus4':record['full_gt']={branch:field_metrics(field,target,full) for branch,field in fields.items()}
                interventions[name]=record
            correct=branch_fields(outputs['intact']);shifted=branch_fields(outputs['shift_plus4'])
            row=dict(**entry,common_support_pixels=int(common.sum()),full_gt_pixels=int(full.sum()),
                interventions=interventions,shift_response={branch:shift_response(correct[branch],shifted[branch],common,4) for branch in correct},
                zero_value_correspondence_exact=True,invariants=invariant_evidence)
            records.append(row);append(args.output/'controls.jsonl',row)
            write(args.output/'controls-summary.json',dict(latest=row,summary=summarize_panel(records),elapsed_seconds=time.monotonic()-STARTED))
            if entry['save_map']:
                arrays=dict(truth=target[0,0].cpu().numpy(),valid=valid[0].cpu().numpy(),common_support=common[0,0].cpu().numpy(),
                    left_rate=left[0].mean(0).cpu().numpy(),right_rate=right[0].mean(0).cpu().numpy())
                for name,output in outputs.items():
                    arrays.update({name+'_'+branch:field[0,0].cpu().numpy() for branch,field in branch_fields(output).items()})
                    arrays[name+'_null_probability']=output['null_probability'][0,0].cpu().numpy()
                stem=f"panel-{entry['position']:02d}-{entry['scene']}-{entry['frame']}"
                np.savez_compressed(maps/(stem+'.npz'),**arrays);render_diagnostic_maps(maps/(stem+'.png'),arrays)
            if time.monotonic()-STARTED>config['max_seconds']:
                write(args.output/'result.json',dict(status='TIME_CAP_INCOMPLETE',checkpoint_binding=checkpoint_binding,
                    checkpoint=str(args.checkpoint),checkpoint_sha256=args.checkpoint_sha256,checkpoint_update=value['updates'],
                    model_state_unchanged=controls_state_unchanged(model,initial),actual_inference_batch=1,gradient_accumulation=False,
                    summary=summarize_panel(records),records=records,elapsed_seconds=time.monotonic()-STARTED))
                return
    unchanged=controls_state_unchanged(model,initial)
    if not unchanged:raise RuntimeError('diagnostics modified selected model state')
    summary=summarize_panel(records,complete=True)
    write(args.output/'result.json',dict(status='COMPLETE_CONTROLS' if summary['registered_panel_complete'] else 'INCOMPLETE_DIAGNOSTIC_SUPPORT',checkpoint=str(args.checkpoint),
        checkpoint_sha256=args.checkpoint_sha256,checkpoint_update=value['updates'],panel=panel,
        summary=summary,records=records,model_state_unchanged=True,optimizer_updates=0,
        checkpoint_binding=checkpoint_binding,actual_inference_batch=1,gradient_accumulation=False,training_physical_batch=2,
        interventions=dict(zero_content_qk='zero both-direction cross-view content QK scores before value mixing and temporal correspondence; retain relative bias and readout',
            zero_cross_value_evidence='zero attended values before existing cross-output projection/normalization/LIF; preserve fused residual, learned normalization offsets, and correspondence logits'),
        shift_support='valid GT with original and shifted xR in native frame; common across branches/interventions',
        branch_metric_semantics='final and conditional_match are disparity estimates; weighted_null/weighted_match/residual are additive contributions, whose EPE fields compare a contribution to GT and are not standalone model accuracy',
        elapsed_seconds=time.monotonic()-STARTED,test_access='sealed'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('base-source','config','manifest','data-root','output','source-registry'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path);parser.add_argument('--checkpoint-sha256')
    args=parser.parse_args();components=setup(args)
    stage=components[0]['stage']
    if stage=='preflight':preflight(args,*components)
    elif stage=='controls':controls(args,*components)
    else:training(args,*components)


if __name__=='__main__':
    main()
