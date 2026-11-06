"""Nine fresh, prefetched 50k runs; no checkpoint continuation or LR restart."""
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
from fresh import *
from prefetch import Prefetch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('overlay','parent-overlay','base','config','data-root','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();begun=time.monotonic();started_utc=datetime.now(timezone.utc).isoformat()
    manifest=a.base/'projects/snn-depth-estimation/experiments/spiking-transformer-stereo/configs/manifest.json'
    c,reg,source=verify_source(a.overlay/'source-registry.json',a.config,a.overlay/'configs/registration.json',manifest)
    if a.output.name!=c['run_id']:raise RuntimeError('output run identity differs')
    recipe,parent,driver,mechanism,parent_source,base_source=load_science(a.parent_overlay,a.base)
    verify_dependencies(source,reg,parent_source,base_source)
    setup_cuda(c['seed'])
    import torch
    from torch.utils.data import DataLoader
    model=recipe.build_model(driver,mechanism,c['model'],c['seed'])
    proof=parent.initialization_proof(driver,mechanism,model,c['model'],c['seed']);model=model.cuda()
    opt,scheduler=optimizer(model);fresh_rng=rng();fresh_model=state_sha(model.state_dict())
    if optimizer_steps(opt)!=[0,0] or scheduler.last_epoch!=0 or opt.param_groups[0]['lr']!=8e-6:raise RuntimeError('fresh zero-step optimizer/scheduler required')
    data=read(manifest)
    if len(data['train'])!=960 or len(data['dev'])!=60:raise RuntimeError('fixed Train/Dev membership changed')
    train=driver.ManifestDataset(a.data_root,data['train']);dev=driver.ManifestDataset(a.data_root,data['dev'])
    loader=DataLoader(dev,batch_size=2,num_workers=0,generator=torch.Generator().manual_seed(c['seed']+991))
    order=torch.randperm(960,generator=torch.Generator().manual_seed(c['seed'])).tolist()
    a.output.mkdir(parents=True,exist_ok=False)
    write(a.output/'initialization.json',proof)
    write(a.output/'sample-order.json',dict(order=order,effective_batch=8,updates_per_cycle=120,full_cycles=416,partial_cycle_updates=80))
    metadata=dict(config=c,source=source,parent_recipe_source=parent_source,base_source=base_source,registration=reg,
        initialization=proof,manifest_sha256=sha(manifest),fresh_zero_step_model_sha256=fresh_model,initial_rng_sha256=rng_sha(fresh_rng),
        command=sys.argv,started_utc=started_utc,hostname=socket.gethostname(),user=getpass.getuser(),gpu=torch.cuda.get_device_name(0),
        torch=torch.__version__,cuda=torch.version.cuda,python=sys.version,slurm={k:v for k,v in os.environ.items() if k.startswith('SLURM_')},
        pip_freeze=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True),
        selection='final50k primary; best trained Dev among registered positive update grid secondary; initial0 separate',
        precision='FP32 historical cuDNN TF32 enabled; matmul TF32 disabled',
        timing_definition='seconds includes consumer waiting+transfer+forward/backward/clip/AdamW/scheduler; CPU loader_work overlaps GPU compute and must not be added to elapsed')
    write(a.output/'metadata.json',metadata)
    def timing():return dict(elapsed_seconds=time.monotonic()-begun,timestamp_utc=datetime.now(timezone.utc).isoformat())
    # Match the historical warmed native32 Dev convention, retaining fresh weights/RNG.
    warm=next(iter(loader));model.eval()
    with torch.no_grad():driver.forward(model,warm[0].cuda(),warm[1].cuda(),32)
    driver.reset_model(model);del warm
    evaluation_started=time.monotonic();initial=dict(update=0,**driver.evaluate(model,loader,32),**timing())
    initial['evaluation_seconds']=time.monotonic()-evaluation_started
    if state_sha(model.state_dict())!=fresh_model or optimizer_steps(opt)!=[0,0] or scheduler.last_epoch!=0:raise RuntimeError('warmed initial Dev changed fresh scientific state')
    restore_rng(fresh_rng)
    write(a.output/'initial.json',initial);append(a.output/'evaluations.jsonl',initial)
    history=[initial];best=float('inf');best_update=None
    def checkpoint(step):
        if optimizer_steps(opt)!=[step,step] or scheduler.last_epoch!=step:raise RuntimeError('checkpoint optimizer/scheduler clocks differ')
        if not finite(model.state_dict()) or not finite(opt.state_dict()):raise FloatingPointError('nonfinite checkpoint state')
        state=dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),updates=step,samples_seen=step*8,
            config=c,order=order,best=best,best_update=best_update,source=source,base_source_identity_sha256=base_source['source_identity_sha256'],
            manifest_sha256=sha(manifest),initialization=proof,rng=rng())
        torch.save(state,a.output/'last.tmp.pt');(a.output/'last.tmp.pt').replace(a.output/'last.pt')
    # No RNG/CUDA operation in producer; it emits fixed FIFO batches with stateless paired flags.
    # An initial last.pt uses None for best until the first trained evaluation.
    best=None;checkpoint(0);best=float('inf')
    with Prefetch(train,recipe,order,c['seed'],0,50000) as prefetch:
        for step in range(50000):
            start=time.monotonic();batch=prefetch.get(step);wait=time.monotonic()-start
            row=train_step(model,opt,scheduler,driver,batch.items)
            row.update(update=step+1,samples_seen=(step+1)*8,indices=batch.indices,augmentation_hv=batch.flags,
                consumer_wait_seconds=wait,loader_work_seconds=batch.loader_work_seconds,seconds=time.monotonic()-start,**timing())
            if scheduler.last_epoch!=step+1:raise RuntimeError('scheduler update count differs')
            append(a.output/'train.jsonl',row)
            append(a.output/'loss-components.jsonl',{key:row[key] for key in ('update','loss','disparity_loss','neuron_loss','valid_pixels')});del batch
            if step+1 in GRID:
                evaluation_started=time.monotonic();evaluation=dict(update=step+1,**driver.evaluate(model,loader,32),**timing())
                evaluation['evaluation_seconds']=time.monotonic()-evaluation_started
                history.append(evaluation);append(a.output/'evaluations.jsonl',evaluation);write(a.output/'history.json',history)
                improved=evaluation['mean_scene_epe']<best
                if improved:best=evaluation['mean_scene_epe'];best_update=step+1
                checkpoint(step+1)
                if improved:shutil.copy2(a.output/'last.pt',a.output/'best.tmp.pt');(a.output/'best.tmp.pt').replace(a.output/'best.pt')
                print('EVAL',step+1,evaluation['mean_scene_epe'],flush=True)
            elif (step+1)%1000==0:checkpoint(step+1)
    rows=[json.loads(line) for line in (a.output/'train.jsonl').read_text().splitlines()]
    if [r['update'] for r in rows]!=list(range(1,50001)) or rows[-1]['samples_seen']!=400000:raise RuntimeError('final count continuity failed')
    if optimizer_steps(opt)!=[50000,50000] or scheduler.last_epoch!=50000 or not all(finite(r) for r in rows):raise RuntimeError('final optimizer/finite audit failed')
    write(a.output/'result.json',dict(status='COMPLETED',updates_completed=50000,samples_seen=400000,physical_batch=4,effective_batch=8,
        final_primary=history[-1],best_dev_secondary=min(history[1:],key=lambda r:r['mean_scene_epe']),best_update=best_update,initial=initial,
        optimizer_step_min_max=[50000,50000],scheduler_last_epoch=50000,scheduler_total_steps=50100,
        contiguous_updates=True,all_training_records_finite=True,all_checkpoint_tensors_finite=True,
        training_step_seconds=sum(r['seconds'] for r in rows),consumer_wait_seconds=sum(r['consumer_wait_seconds'] for r in rows),
        overlapping_loader_work_seconds=sum(r['loader_work_seconds'] for r in rows),dev_seconds=sum(r['evaluation_seconds'] for r in history),**timing()))


if __name__=='__main__':main()
