"""CPU preflight, CUDA capacity benchmark, and bounded SDSA-NMRF pilot.

Outputs and datasets must be outside the source checkout. GPU simulation does
not establish an inference energy saving. Uses the native NMRF training loss.
"""
import argparse
import json
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from evaluate_nmrf import input_tensors,metric_counts,summarize
from nmrf_runtime import prepare_batch,weighted_loss,UPSTREAM_REVISION
from run_nmrf import record_run,emit
from runtime import NativeDataset
from spiking_nmrf import build_spiking_nmrf,ARCHITECTURE,MODEL_NAME

ROOT=Path(__file__).resolve().parent


def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['preflight','benchmark','train'])
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--manifest',type=Path,default=ROOT/'configs/synthetic-split.json')
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--steps',type=int,default=2000)
    p.add_argument('--crop',type=int,nargs=2)
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--accumulation',type=int,default=1)
    p.add_argument('--learning-rate',type=float,default=2e-4)
    p.add_argument('--workers',type=int,default=0)
    p.add_argument('--seed',type=int,default=1234)
    p.add_argument('--save-every',type=int,default=200)
    p.add_argument('--log-every',type=int,default=20)
    p.add_argument('--eval-every',type=int,default=200)
    p.add_argument('--eval-pairs',type=int,default=6)
    p.add_argument('--benchmark-steps',type=int,default=6)
    p.add_argument('--benchmark-warmup',type=int,default=2)
    p.add_argument('--memory-fraction',type=float,default=.92)
    p.add_argument('--resume',type=Path)
    a=p.parse_args()
    if a.crop is None:a.crop=[32,64] if a.command=='preflight' and a.device=='cpu' else [128,256]
    if min(a.steps,a.batch_size,a.accumulation,a.save_every,a.log_every,a.eval_every,a.eval_pairs,a.benchmark_steps,a.benchmark_warmup)<1:p.error('Positive counts required')
    if a.workers<0 or not 0<a.memory_fraction<=1:p.error('Invalid worker/memory setting')
    if a.command!='preflight' and a.device!='cuda':p.error('Training/benchmarks require the authorized GPU')
    if a.output.resolve().is_relative_to(ROOT):p.error('Outputs must be outside Git')
    if a.device=='cuda' and not torch.cuda.is_available():p.error('CUDA unavailable')
    return a


@torch.no_grad()
def evaluate(model,dataset,rows,device):
    model.eval();counts=[]
    for sample in dataset:
        left,right,target,valid=input_tensors(sample,device,None)
        prediction=model(left,right)['disp'][0]
        counts.append(metric_counts(prediction,target,valid & (target>0) & (target<192)))
    model.train()
    return dict(**summarize(counts),pairs=len(rows),resolution='native',max_disparity=192)


def checkpoint(a,model,optimizer,scheduler,step):
    temporary=a.output/'checkpoint.tmp'
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                    step=step,model_name=MODEL_NAME,architecture=ARCHITECTURE,upstream_revision=UPSTREAM_REVISION,
                    batch_size=a.batch_size,accumulation=a.accumulation),temporary)
    temporary.replace(a.output/'checkpoint-last.pt')


def main():
    a=arguments()
    torch.set_num_threads(2)
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    if a.device=='cuda':torch.cuda.set_per_process_memory_fraction(a.memory_fraction)
    manifest=json.loads(a.manifest.read_text())
    if set(r['scene'] for r in manifest['train']) & set(r['scene'] for r in manifest['dev']):raise ValueError('Scene split overlap')
    dataset=NativeDataset(a.data_root,manifest['train']);sample=dataset[0]
    if not all(bool(((x==0)|(x==1)).all()) for x in sample[:2]):raise ValueError('Expected binary native sensor input')
    # One first-listed pair per evenly spaced dev scene for periodic diagnostics.
    first={}
    for row in manifest['dev']:first.setdefault(row['scene'],row)
    scene_rows=list(first.values())
    periodic=[scene_rows[i] for i in np.linspace(0,len(scene_rows)-1,min(a.eval_pairs,len(scene_rows)),dtype=int)]
    model,criterion,cfg=build_spiking_nmrf(a.seed)
    model.to(a.device);criterion.to(a.device)
    cfg.SEED=a.seed;cfg.SOLVER.MAX_ITER=a.steps;cfg.SOLVER.BASE_LR=a.learning_rate
    cfg.SOLVER.IMS_PER_BATCH=a.batch_size*a.accumulation
    cfg.DATASETS.CROP_SIZE=a.crop;cfg.DATASETS.TRAIN=['native-synthetic-manifest'];cfg.DATASETS.TEST=[]
    record_run(a,cfg,manifest)
    config=json.loads((a.output/'config.json').read_text())
    config.update(model=MODEL_NAME,architecture=ARCHITECTURE,precision='FP32',periodic_dev_rows=periodic,
                  final_dev_rows=manifest['dev'],initialization='fresh seed; no trained ANN decoder transfer',
                  runtime='dense torch simulation of binary spike operations; no energy claim')
    (a.output/'config.json').write_text(json.dumps(config,indent=2))
    for name in ['spiking_nmrf.py','run_spiking_nmrf.py','evaluate_nmrf.py']:
        shutil.copyfile(ROOT/name,a.output/name)
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,weight_decay=1e-5)
    scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=a.learning_rate,total_steps=a.steps,pct_start=.05,cycle_momentum=False)
    start=0
    if a.resume:
        state=torch.load(a.resume,map_location='cpu',weights_only=False)
        if state['model_name']!=MODEL_NAME or state['architecture']!=ARCHITECTURE or state['upstream_revision']!=UPSTREAM_REVISION:raise ValueError('Incompatible checkpoint')
        if state['scheduler']['total_steps']!=a.steps:raise ValueError('Resume must preserve scheduler horizon')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler']);start=state['step']
        del state
        if start>=a.steps:raise ValueError('Run already complete')
    loader=DataLoader(dataset,batch_size=a.batch_size,shuffle=a.command!='preflight',num_workers=a.workers,
                      persistent_workers=a.workers>0,pin_memory=a.device=='cuda',drop_last=True)
    iterator=iter(loader);model.train()
    if a.device=='cuda':torch.cuda.reset_peak_memory_stats()
    started=time.monotonic()
    final=start+1 if a.command=='preflight' else (start+a.benchmark_warmup+a.benchmark_steps if a.command=='benchmark' else a.steps)
    for step in range(start+1,final+1):
        if a.command=='benchmark' and step==start+a.benchmark_warmup+1:
            torch.cuda.synchronize();measured=time.monotonic();torch.cuda.reset_peak_memory_stats()
        collect=step==start+1 or step%a.log_every==0 or step==final
        model.parent.backbone.collect_activity(collect)
        optimizer.zero_grad(set_to_none=True);total_loss=0.
        for micro in range(a.accumulation):
            try:batch=next(iterator)
            except StopIteration:iterator=iter(loader);batch=next(iterator)
            left,right,target,valid,_=prepare_batch(batch,a.device,a.crop,True)
            if a.command=='preflight':left.requires_grad_(True);right.requires_grad_(True)
            outputs=model(left,right)
            loss,parts=weighted_loss(criterion,outputs,target,valid)
            (loss/a.accumulation).backward();total_loss+=float(loss.detach())/a.accumulation
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        if a.command=='preflight':
            channel_gradient=(left.grad.abs().sum((0,2,3))+right.grad.abs().sum((0,2,3))).detach()
            if not bool((channel_gradient>0).all()):raise RuntimeError('Some sensor bins lack a gradient')
            qk_grad={n:float(m.qkv.conv.weight.grad.abs().sum()) for n,m in model.parent.backbone.named_modules() if hasattr(m,'qkv')}
            if not all(v>0 and np.isfinite(v) for v in qk_grad.values()):raise RuntimeError('Missing spiking attention gradients')
            before=model.parent.backbone.stem1.conv.weight.detach().clone()
        optimizer.step()
        if a.command!='preflight':scheduler.step()
        if collect:
            emit(a,dict(event='train',step=step,loss=total_loss,grad_norm=float(grad),seconds=time.monotonic()-started,
                        learning_rate=scheduler.get_last_lr()[0],batch_size=a.batch_size,
                        peak_cuda_bytes=torch.cuda.max_memory_allocated() if a.device=='cuda' else None,
                        reserved_cuda_bytes=torch.cuda.memory_reserved() if a.device=='cuda' else None,
                        spike_rates=model.parent.backbone.activity()))
        model.parent.backbone.collect_activity(False)
        if a.command=='preflight':
            if torch.equal(before,model.parent.backbone.stem1.conv.weight):raise RuntimeError('Stem failed to update')
            model.eval()
            with torch.no_grad():pred=model(left[...,:-1,:-1],right[...,:-1,:-1])['disp']
            if pred.shape!=target[...,:-1,:-1].shape or not torch.isfinite(pred).all():raise RuntimeError('Evaluation/padding failed')
            result=dict(status='passed',all_50_sensor_bins_receive_gradient=True,qkv_gradient_sums=qk_grad,
                        parameter_count=sum(p.numel() for p in model.parameters()),seconds=time.monotonic()-started,
                        peak_cuda_bytes=torch.cuda.max_memory_allocated() if a.device=='cuda' else None)
            (a.output/'preflight.json').write_text(json.dumps(result,indent=2));emit(a,result);return
        del outputs,loss,parts,left,right,target,valid,batch
        if a.command=='train' and (step%a.save_every==0 or step==final):checkpoint(a,model,optimizer,scheduler,step)
        if a.command=='train' and (step%a.eval_every==0 or step==final):
            rows=manifest['dev'] if step==final else periodic
            metrics=evaluate(model,NativeDataset(a.data_root,rows),rows,a.device)
            emit(a,dict(event='validation',step=step,seconds=time.monotonic()-started,**metrics))
    if a.command=='benchmark':
        torch.cuda.synchronize();elapsed=time.monotonic()-measured
        training_peak=torch.cuda.max_memory_allocated()
        # Exercise full-frame validation with live optimizer state before admitting batch size.
        validation=evaluate(model,NativeDataset(a.data_root,periodic[:1]),periodic[:1],a.device)
        result=dict(batch_size=a.batch_size,workers=a.workers,steps=a.benchmark_steps,
                    examples_per_second=a.benchmark_steps*a.batch_size*a.accumulation/elapsed,
                    seconds=elapsed,training_peak_cuda_bytes=training_peak,
                    peak_cuda_bytes=torch.cuda.max_memory_allocated(),reserved_cuda_bytes=torch.cuda.memory_reserved(),
                    free_cuda_bytes=torch.cuda.mem_get_info()[0],validation=validation)
        (a.output/'benchmark.json').write_text(json.dumps(result,indent=2));emit(a,result);return
    completed=dict(steps=final,seconds=time.monotonic()-started,status='completed')
    (a.output/'completed.json').write_text(json.dumps(completed,indent=2));emit(a,completed)


if __name__=='__main__':main()
