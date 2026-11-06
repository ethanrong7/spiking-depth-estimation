"""Fresh300k schedule; execute only an explicitly released bounded stage."""
import argparse
from datetime import datetime,timezone
import getpass
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from stage import *
from prefetch import Prefetch,decode_batch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('overlay','runtime-overlay','parent-overlay','base','config','data-root','output','stage-release'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--stage-target',type=int,required=True);p.add_argument('--resume-checkpoint',type=Path);a=p.parse_args()
    started=time.monotonic();started_utc=datetime.now(timezone.utc).isoformat()
    manifest=a.base/'projects/snn-depth-estimation/experiments/spiking-transformer-stereo/configs/manifest.json'
    c,reg,source=verify_source(a.overlay,a.config,manifest)
    rt,recipe,parent,driver,mechanism,rr,pr,br=load_runtime(a.runtime_overlay,a.parent_overlay,a.base)
    verify_dependencies(source,reg,rr,pr,br)
    release=read(a.stage_release);start=release['start_update'];target=release['target_update']
    if not release['root_reviewed'] or release['source_identity_sha256']!=source['source_identity_sha256'] or release['registration_sha256']!=source['registration_sha256'] or c['run_id'] not in release['runs']:raise RuntimeError('stage release identity differs')
    if not (0<=start<target<=300000) or a.stage_target!=target:raise RuntimeError('unreleased stage target')
    if a.output.name!=c['run_id']:raise RuntimeError('output run differs')
    grid=release.get('eval_updates',GRID)
    if target not in grid or sorted(set(grid))!=grid or any(not (0<=u<=target) for u in grid):raise RuntimeError('invalid released evaluation grid')
    if start==0 and (target!=50000 or a.resume_checkpoint is not None or grid!=GRID):raise RuntimeError('initial stage must be fresh0 to50000')
    if start>0 and a.resume_checkpoint is None:raise RuntimeError('later stage requires final boundary checkpoint')
    rt.setup_cuda(c['seed'])
    import torch
    from torch.utils.data import DataLoader
    model=recipe.build_model(driver,mechanism,c['model'],c['seed'])
    proof=parent.initialization_proof(driver,mechanism,model,c['model'],c['seed']);model=model.cuda();opt,sch=optimizer(model)
    ctx=context(c,source,reg,proof,'scientific_stage');data=read(manifest)
    if (len(data['train']),len(data['dev']))!=(960,60):raise RuntimeError('fixed cohort changed')
    train=driver.ManifestDataset(a.data_root,data['train']);dev=driver.ManifestDataset(a.data_root,data['dev'])
    dev_loader=DataLoader(dev,batch_size=2,num_workers=0,generator=torch.Generator().manual_seed(c['seed']+991))
    order=torch.randperm(960,generator=torch.Generator().manual_seed(c['seed'])).tolist()
    def timing():return dict(stage_start=start,stage_target=target,elapsed_seconds=time.monotonic()-started,timestamp_utc=datetime.now(timezone.utc).isoformat())
    if start==0:
        a.output.mkdir(parents=True,exist_ok=False)
        for name in ('train.jsonl','loss-components.jsonl','evaluations.jsonl'):(a.output/name).touch()
        write(a.output/'initialization.json',proof);write(a.output/'sample-order.json',dict(order=order,effective_batch=8,updates_per_cycle=120))
        fresh_hash=rt.state_sha(model.state_dict());fresh_rng=rt.rng()
        warm=next(iter(dev_loader));model.eval()
        with torch.no_grad():driver.forward(model,warm[0].cuda(),warm[1].cuda(),32)
        driver.reset_model(model);del warm
        began=time.monotonic();initial=dict(update=0,**driver.evaluate(model,dev_loader,32),**timing());initial['evaluation_seconds']=time.monotonic()-began
        if rt.state_sha(model.state_dict())!=fresh_hash or rt.optimizer_steps(opt)!=[0,0] or sch.last_epoch!=0:raise RuntimeError('initial warmed Dev changed fresh state')
        rt.restore_rng(fresh_rng)
        append(a.output/'evaluations.jsonl',initial);write(a.output/'initial.json',initial);history=[initial];best=None;best_update=None
        write(a.output/'history.json',history)
        metadata=dict(config=c,source=source,registration=reg,runtime_source=rr,parent_recipe_source=pr,base_source=br,initialization=proof,
            fresh_zero_step_model_sha256=fresh_hash,initial_rng_sha256=rt.rng_sha(fresh_rng),manifest_sha256=sha(manifest),command=sys.argv,
            hostname=socket.gethostname(),user=getpass.getuser(),gpu=torch.cuda.get_device_name(0),torch=torch.__version__,cuda=torch.version.cuda,
            python=sys.version,slurm={k:v for k,v in os.environ.items() if k.startswith('SLURM_')},pip_freeze=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True),
            started_utc=started_utc,planned_updates=300000,initial_stage_target=50000,
            precision='FP32 cuDNN TF32 enabled/matmul disabled',timing_definition='step includes consumer wait+transfer+training; loader work overlaps and is not added to elapsed')
        write(a.output/'metadata.json',metadata)
        checkpoint(a.output/'last.pt',rt,ctx,model,opt,sch,0,order,history,best,best_update)
    else:
        if a.resume_checkpoint.name!='last.pt' or a.resume_checkpoint.resolve().parent!=a.output.resolve():raise RuntimeError('resume must use this run final last.pt, never selected-best')
        previous=read(a.output/'result.json')
        if previous['status']!='PAUSED_AT_STAGE_BOUNDARY' or previous['updates_completed']!=start or previous['stage_boundary_checkpoint_sha256']!=sha(a.resume_checkpoint):raise RuntimeError('previous final stage boundary differs')
        if release['checkpoint_sha256'][c['run_id']]!=sha(a.resume_checkpoint):raise RuntimeError('unreleased resume lineage')
        if previous['best_checkpoint_sha256']!=sha(a.output/'best.pt'):raise RuntimeError('historical selected-best checkpoint changed')
        ck=restore(a.resume_checkpoint,rt,ctx,model,opt,sch,expected_update=start);strict_log_boundary(a.output,ck)
        if ck['order']!=order:raise RuntimeError('seeded resumed sample order differs')
        history=ck['history'];best=ck['best'];best_update=ck['best_update']
        next_batch=decode_batch(train,recipe,order,c['seed'],start)
        signature=forward_signature(model,driver,rt,next_batch.items);del next_batch
        # Warm/signature inspection must not consume the resumed scientific RNG stream.
        rt.restore_rng(ck['rng'])
        if rt.optimizer_steps(opt)!=[start,start] or sch.last_epoch!=start:raise RuntimeError('resume warmup changed clocks')
        preserved=a.output/'stages'/f'{previous["stage_start"]:06d}-{start:06d}';preserved.mkdir(parents=True,exist_ok=False)
        for name in ('result.json','last.pt','best.pt','history.json'):shutil.copy2(a.output/name,preserved/name)
        write(preserved/'resume-lineage.json',dict(checkpoint_sha256=sha(a.resume_checkpoint),next_forward_signature=signature,new_release_sha256=sha(a.stage_release)))
        del ck
    append(a.output/'stage-launches.jsonl',dict(stage_start=start,stage_target=target,stage_release_sha256=sha(a.stage_release),command=sys.argv,job_id=os.environ['SLURM_JOB_ID'],started_utc=started_utc))
    with Prefetch(train,recipe,order,c['seed'],start,target) as stream:
        for step in range(start,target):
            began=time.monotonic();batch=stream.get(step);wait=time.monotonic()-began
            row=rt.train_step(model,opt,sch,driver,batch.items)
            row.update(update=step+1,samples_seen=(step+1)*8,indices=batch.indices,augmentation_hv=batch.flags,consumer_wait_seconds=wait,
                loader_work_seconds=batch.loader_work_seconds,seconds=time.monotonic()-began,**timing())
            if sch.last_epoch!=step+1:raise RuntimeError('global scheduler clock differs')
            append(a.output/'train.jsonl',row);append(a.output/'loss-components.jsonl',{key:row[key] for key in ('update','loss','disparity_loss','neuron_loss','valid_pixels')});del batch
            improved=False
            if step+1 in grid:
                began=time.monotonic();evaluation=dict(update=step+1,**driver.evaluate(model,dev_loader,32),**timing());evaluation['evaluation_seconds']=time.monotonic()-began
                history.append(evaluation);append(a.output/'evaluations.jsonl',evaluation);write(a.output/'history.json',history)
                improved=best is None or evaluation['mean_scene_epe']<best
                if improved:best=evaluation['mean_scene_epe'];best_update=step+1
                print('EVAL',step+1,evaluation['mean_scene_epe'],flush=True)
            if step+1 in grid or (step+1)%1000==0 or step+1==target:
                checkpoint(a.output/'last.pt',rt,ctx,model,opt,sch,step+1,order,history,best,best_update)
                if improved:shutil.copy2(a.output/'last.pt',a.output/'best.tmp.pt');(a.output/'best.tmp.pt').replace(a.output/'best.pt')
    ck=torch.load(a.output/'last.pt',map_location='cpu',weights_only=False);strict_log_boundary(a.output,ck)
    if rt.optimizer_steps(opt)!=[target,target] or sch.last_epoch!=target or sch.total_steps!=300100:raise RuntimeError('stage final clocks differ')
    rows=[json.loads(line) for line in (a.output/'train.jsonl').read_text().splitlines()]
    if not all(rt.finite(r) for r in rows):raise FloatingPointError('nonfinite saved training records')
    write(a.output/'result.json',dict(status='COMPLETED_PLANNED_HORIZON' if target==300000 else 'PAUSED_AT_STAGE_BOUNDARY',planned_updates=300000,
        updates_completed=target,stage_updates_completed=target-start,samples_seen=target*8,stage_presentations=(target-start)*8,
        stage_primary=history[-1],best_dev_secondary=min((r for r in history if r['update']>0),key=lambda r:r['mean_scene_epe']),best_update=best_update,
        stage_boundary_checkpoint_sha256=sha(a.output/'last.pt'),best_checkpoint_sha256=sha(a.output/'best.pt'),
        optimizer_step_min_max=[target,target],scheduler_last_epoch=target,scheduler_total_steps=300100,physical_batch=4,effective_batch=8,
        source_identity_sha256=source['source_identity_sha256'],registration_sha256=source['registration_sha256'],stage_release_sha256=sha(a.stage_release),
        contiguous_updates=True,all_training_records_finite=True,all_checkpoint_tensors_finite=rt.finite(ck['model']) and rt.finite(ck['optimizer']),
        complete120_update_cycles=target//120,partial_cycle_updates=target%120,
        stage_training_step_seconds=sum(r['seconds'] for r in rows if r['update']>start),stage_consumer_wait_seconds=sum(r['consumer_wait_seconds'] for r in rows if r['update']>start),**timing()))


if __name__=='__main__':main()
