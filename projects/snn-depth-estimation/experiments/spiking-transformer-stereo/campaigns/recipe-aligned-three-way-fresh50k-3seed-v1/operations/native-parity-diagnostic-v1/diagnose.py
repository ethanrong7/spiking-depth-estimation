"""Bounded zero-optimizer diagnostic of historical CUDA backward repeatability."""
import argparse
import copy
from datetime import datetime,timezone
import importlib.util
import itertools
import json
import os
from pathlib import Path
import sys
import time
import traceback

EXPECTED='1dcaa81f86f130d4945a848b91a290db5d41e4e05d3c49a3f46d22f733b6bba4'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('overlay','parent-overlay','base','data-root','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();sys.path.insert(0,str(a.overlay/'scripts'))
    import fresh
    from prefetch import Prefetch,decode_batch
    source=fresh.read(a.overlay/'source-registry.json')
    if source['source_identity_sha256']!=EXPECTED:raise RuntimeError('diagnostic requires exact frozen r2 source')
    for name,h in source['source_files'].items():
        if fresh.sha(a.overlay/name)!=h:raise RuntimeError('r2 source changed: '+name)
    recipe,parent,driver,mechanism,ps,bs=fresh.load_science(a.parent_overlay,a.base)
    reg=fresh.read(a.overlay/'configs/registration.json');fresh.verify_dependencies(source,reg,ps,bs)
    manifest=a.base/'projects/snn-depth-estimation/experiments/spiking-transformer-stereo/configs/manifest.json'
    if fresh.sha(manifest)!=reg['manifest_sha256']:raise RuntimeError('manifest changed')
    a.output.mkdir(parents=True,exist_ok=False);fresh.setup_cuda(1234)
    import torch
    data=fresh.read(manifest);train=driver.ManifestDataset(a.data_root,data['train'])
    order=torch.randperm(960,generator=torch.Generator().manual_seed(1234)).tolist()
    summaries=[];all_started=time.monotonic()
    # Capture the *unclipped* parameter gradients around the exact unchanged r2 step helper.
    original_clip=torch.nn.utils.clip_grad_norm_
    captured={}
    def capture_clip(parameters,*args,**kwargs):
        parameters=list(parameters)
        captured['gradients']={name:parameter.grad.detach().cpu().clone() for name,parameter in model.named_parameters() if parameter.grad is not None}
        return original_clip(parameters,*args,**kwargs)
    def compare_gradients(first,second):
        if set(first)!=set(second):raise RuntimeError('gradient parameter membership differs')
        parameter=[];d2=0.;a2=0.;b2=0.;maxabs=0.;count=0
        for name in sorted(first):
            x=first[name].double();y=second[name].double();d=x-y
            delta=float(d.square().sum());nx=float(x.square().sum());ny=float(y.square().sum());maximum=float(d.abs().max());different=int((x!=y).sum())
            d2+=delta;a2+=nx;b2+=ny;maxabs=max(maxabs,maximum);count+=different
            parameter.append(dict(name=name,relative_l2=delta**.5/max(nx**.5,ny**.5,1e-30),max_absolute_difference=maximum,l2_difference=delta**.5,different_elements=different,numel=x.numel()))
        return dict(exact=count==0,relative_l2=d2**.5/max(a2**.5,b2**.5,1e-30),l2_difference=d2**.5,first_l2=a2**.5,second_l2=b2**.5,max_absolute_difference=maxabs,different_elements=count,per_parameter=parameter)
    try:
        for arm in fresh.ARMS:
            fresh.setup_cuda(1234)
            model=recipe.build_model(driver,mechanism,arm,1234)
            initialization=parent.initialization_proof(driver,mechanism,model,arm,1234)
            model=model.cuda();opt,sch=fresh.optimizer(model)
            initial_model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};initial_opt=copy.deepcopy(opt.state_dict());initial_sch=copy.deepcopy(sch.state_dict());initial_rng=fresh.rng()
            model_hash=fresh.state_sha(initial_model);opt_hash=fresh.state_sha(initial_opt);rng_hash=fresh.rng_sha(initial_rng)
            def reset():
                driver.reset_model(model);model.load_state_dict(initial_model,strict=True);opt.load_state_dict(copy.deepcopy(initial_opt));sch.load_state_dict(copy.deepcopy(initial_sch));fresh.restore_rng(initial_rng);opt.zero_grad(set_to_none=True)
            output=a.output/arm;output.mkdir()
            warm=decode_batch(train,recipe,order,1234,0);fresh.train_step(model,opt,sch,driver,warm.items,apply=False);del warm
            passes=[];gradients=[];reference_input=None
            for mode in ('serial','prefetch'):
                for index in range(4):
                    reset();before_load=fresh.rng_sha(fresh.rng())
                    if mode=='serial':batch=decode_batch(train,recipe,order,1234,0)
                    else:
                        with Prefetch(train,recipe,order,1234,0,1) as loader:batch=loader.get(0)
                    input_hash=fresh.state_sha(batch.items)
                    if reference_input is None:reference_input=input_hash
                    if input_hash!=reference_input:raise RuntimeError('native diagnostic decoded tensors differ')
                    if before_load!=fresh.rng_sha(fresh.rng()):raise RuntimeError('diagnostic CPU loader consumed RNG')
                    torch.nn.utils.clip_grad_norm_=capture_clip
                    try:row=fresh.train_step(model,opt,sch,driver,batch.items,apply=False)
                    finally:torch.nn.utils.clip_grad_norm_=original_clip
                    gradient=captured.pop('gradients');gradients.append(gradient)
                    record=dict(mode=mode,repeat=index,input_sha256=input_hash,indices=batch.indices,augmentation_hv=batch.flags,scalars=row,
                        gradient_sha256=fresh.state_sha(gradient),model_sha256=fresh.state_sha(model.state_dict()),optimizer_sha256=fresh.state_sha(opt.state_dict()),rng_sha256=fresh.rng_sha(fresh.rng()),optimizer_step_min_max=fresh.optimizer_steps(opt),scheduler_epoch=sch.last_epoch)
                    record['learned_state_optimizer_rng_unchanged']=(record['model_sha256']==model_hash and record['optimizer_sha256']==opt_hash and record['rng_sha256']==rng_hash and record['optimizer_step_min_max']==[0,0] and sch.last_epoch==0)
                    passes.append(record);fresh.write(output/'passes.json',passes);del batch
            pairs=[]
            for i,j in itertools.combinations(range(8),2):
                first=passes[i];second=passes[j]
                comparison=dict(first=i,second=j,first_mode=first['mode'],second_mode=second['mode'],comparison='within_serial' if j<4 else 'within_prefetch' if i>=4 else 'serial_vs_prefetch',
                    scalar_absolute_differences={key:abs(first['scalars'][key]-second['scalars'][key]) for key in ('loss','train_epe','gradient_norm','valid_pixels','lr_used','lr_next')},gradient=compare_gradients(gradients[i],gradients[j]))
                pairs.append(comparison)
            # Save all measured differences before drawing any interpretation or setting a criterion.
            fresh.write(output/'pairwise-differences.json',pairs)
            categories={}
            for category in ('within_serial','within_prefetch','serial_vs_prefetch'):
                group=[x for x in pairs if x['comparison']==category]
                categories[category]=dict(pair_count=len(group),exact_gradient_pairs=sum(x['gradient']['exact'] for x in group),maximum_gradient_relative_l2=max(x['gradient']['relative_l2'] for x in group),maximum_gradient_absolute_difference=max(x['gradient']['max_absolute_difference'] for x in group),maximum_loss_difference=max(x['scalar_absolute_differences']['loss'] for x in group),maximum_train_epe_difference=max(x['scalar_absolute_differences']['train_epe'] for x in group),maximum_gradient_norm_difference=max(x['scalar_absolute_differences']['gradient_norm'] for x in group))
            result=dict(arm=arm,initialization=initialization,inputs_exact=True,all_eight_learned_state_optimizer_rng_unchanged=all(v['learned_state_optimizer_rng_unchanged'] for v in passes),
                initial_model_sha256=model_hash,initial_optimizer_sha256=opt_hash,initial_rng_sha256=rng_hash,categories=categories,
                zero_optimizer_backward_passes=9,optimizer_updates=0,scientific_optimizer_updates=0,diagnostic_only_no_performance_gate_change=True)
            fresh.write(output/'summary.json',result);summaries.append(result)
            del gradients,passes,pairs,model,opt,sch,initial_model,initial_opt,initial_sch;torch.cuda.empty_cache()
    finally:torch.nn.utils.clip_grad_norm_=original_clip
    # Isolated constant-tensor CUDA operator probe, after all native data comparisons.
    # It avoids cuBLAS and cannot change any historical learned state or runtime setting.
    enabled=torch.are_deterministic_algorithms_enabled();warn=torch.is_deterministic_algorithms_warn_only_enabled()
    probe=dict(operation='F.grid_sample CUDA backward',constant_tensor_only=True,optimizer_updates=0)
    before_probe_rng=fresh.rng_sha(fresh.rng())
    try:
        torch.use_deterministic_algorithms(True)
        x=torch.arange(9,device='cuda',dtype=torch.float32).reshape(1,1,3,3).requires_grad_()
        grid=torch.zeros((1,2,2,2),device='cuda',dtype=torch.float32)
        torch.nn.functional.grid_sample(x,grid,align_corners=True).sum().backward();torch.cuda.synchronize()
        probe['outcome']='COMPLETED_WITH_DETERMINISTIC_ALGORITHMS_TRUE'
    except Exception as error:probe.update(outcome='RAISED',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
    finally:torch.use_deterministic_algorithms(enabled,warn_only=warn)
    probe.update(historical_flags_restored=torch.are_deterministic_algorithms_enabled()==enabled and torch.is_deterministic_algorithms_warn_only_enabled()==warn,rng_unchanged=fresh.rng_sha(fresh.rng())==before_probe_rng)
    fresh.write(a.output/'grid-sample-determinism-probe.json',probe)
    fresh.write(a.output/'summary.json',dict(status='DIAGNOSTIC_COMPLETE_NO_TOLERANCE_DECISION',source_identity_sha256=EXPECTED,diagnostic_script_sha256=fresh.sha(Path(__file__)),diagnostic_slurm_sha256=fresh.sha(Path(__file__).with_name('diagnostic.sbatch')),
        command=sys.argv,slurm={k:v for k,v in os.environ.items() if k.startswith('SLURM_')},torch=torch.__version__,cuda=torch.version.cuda,
        job_id=os.environ['SLURM_JOB_ID'],gpu=torch.cuda.get_device_name(0),
        native_zero_optimizer_backward_passes=27,constant_operator_backward_probes=1,engineering_optimizer_updates=0,scientific_optimizer_updates=0,arms=summaries,probe=probe,
        elapsed_seconds=time.monotonic()-all_started,timestamp_utc=datetime.now(timezone.utc).isoformat()))


if __name__=='__main__':main()
