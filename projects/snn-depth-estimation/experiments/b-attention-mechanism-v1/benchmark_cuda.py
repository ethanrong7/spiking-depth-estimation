"""Short standalone CUDA training-utilization benchmark; no scientific EPE result."""
import argparse
import ast
import gc
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
sys.path[:0]=[str(HERE),str(PROJECT)]
import torch
from core.spikestereonet import SpikeStereoNet
from core.attention_encoder_controls import install
from core.utils.utils import InputPadder
from spikingjelly.activation_based import functional
from speed_runtime import SpeedRuntime


def legacy_loss():
    path=HERE.parent/'spiking-transformer-stereo/scripts/run_experiment.py'
    tree=ast.parse(path.read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='disparity_loss')
    scope={'torch':torch}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),scope)
    return scope['disparity_loss']


class Monitor:
    def __init__(self):self.samples=[]
    def start(self):
        self.process=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu',
            '--format=csv,noheader,nounits','-lms','500'],stdout=subprocess.PIPE,text=True)
        def consume():
            for line in self.process.stdout:
                try:
                    fields=line.strip().split(', ')
                    self.samples.append(dict(time=time.perf_counter(),timestamp=fields[0],gpu_util=float(fields[1]),
                        memory_util=float(fields[2]),memory_mib=float(fields[3]),power_w=float(fields[4]),temperature_c=float(fields[5])))
                except (ValueError,IndexError):pass
        self.thread=threading.Thread(target=consume,daemon=True);self.thread.start()
    def stop(self):
        self.process.terminate();self.process.wait();self.thread.join(timeout=3)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch',type=int,default=1)
    p.add_argument('--warmup',type=int,default=8)
    p.add_argument('--steps',type=int,default=30)
    p.add_argument('--modes',nargs='+',choices=['legacy','speed'],default=['legacy','speed'])
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    assert torch.cuda.is_available()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic=False;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    config=dict(batch=a.batch,accumulation=2,height=250,width=400,bins=50,refinements=16,
                warmup=a.warmup,steps=a.steps,seed=1234,variant='signed-surrogate',dtype='float32',
                input='fixed synthetic cached uint8 host tensors; no disk/EXR loading',
                gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,
                scope='training compute/host-transfer loop; excludes full corpus I/O, validation and checkpoints')
    registry=PROJECT.parents[1]/'benchmark-source.json'
    if registry.exists():config['source']=json.loads(registry.read_text())
    (a.output/'config.json').write_text(json.dumps(config,indent=2))
    g=torch.Generator().manual_seed(91)
    left=torch.randint(0,2,(a.batch,50,250,400),generator=g,dtype=torch.uint8)
    right=torch.randint(0,2,left.shape,generator=g,dtype=torch.uint8)
    target=torch.full((a.batch,1,250,400),-8.)
    valid=torch.ones(a.batch,250,400,dtype=torch.bool)
    padder=InputPadder(left.shape,divis_by=32)
    old_loss=legacy_loss();results=[]
    for mode in a.modes:
        torch.manual_seed(1234);torch.cuda.manual_seed_all(1234)
        args=SimpleNamespace(hidden_dims=[128]*3,context_norm='batch',shared_backbone=False,n_snn_layers=3,
            n_downsample=2,corr_levels=4,corr_radius=4,corr_implementation='reg',mixed_precision=False,
            feature_encoder='cnn',context_encoder='cnn',matched_initialization_seed=1234,slow_fast_gru=False)
        model=install(SpikeStereoNet(args),'B',variant='signed-surrogate',seed=1234).cuda()
        model.train();model.freeze_bn()
        optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-5,eps=1e-8)
        scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=2e-4,total_steps=30000,pct_start=.01,cycle_momentum=False,anneal_strategy='linear')
        speed=SpeedRuntime();monitor=Monitor();monitor.start();timings=[];start=None
        torch.cuda.reset_peak_memory_stats()
        print('BEGIN',mode,flush=True)
        try:
            for step in range(a.warmup+a.steps):
                tick=time.perf_counter()
                if step==a.warmup:start=tick
                optimizer.zero_grad(set_to_none=True);losses=[];checks=[]
                for _ in range(2):
                    l,r,t,v=[x.cuda() for x in (left,right,target,valid)]
                    l,r=padder.pad(l.float(),r.float());functional.reset_net(model)
                    preds,spikes,volts=model(l,r,iters=16)
                    preds=[padder.unpad(x) for x in preds]
                    if mode=='speed':
                        loss,okay=speed.loss(preds,t,v);checks.append(okay)
                    else:
                        loss=old_loss(preds,t,v)
                        assert torch.isfinite(loss)
                    loss=loss/2;loss.backward()
                    losses.append(loss.detach() if mode=='speed' else float(loss.detach()))
                    functional.reset_net(model)
                    del l,r,t,v,preds,spikes,volts,loss
                if mode=='speed':
                    branch={n:x.grad.norm() for n,x in model.placement.named_parameters() if x.grad is not None} if step==0 or (step+1)%100==0 else {}
                    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                    total,norm,gradients=speed.summarize(losses,checks,norm,branch)
                else:
                    gradients={n:float(x.grad.norm()) for n,x in model.placement.named_parameters() if x.grad is not None}
                    norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.));total=sum(losses)
                optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize()
                dt=time.perf_counter()-tick
                if step>=a.warmup:timings.append(dt)
                if (step+1)%5==0:print(mode,step+1,'seconds',round(dt,4),'loss',round(total,4),flush=True)
            end=time.perf_counter()
        finally:monitor.stop()
        samples=[x for x in monitor.samples if start<=x['time']<=end]
        result=dict(mode=mode,mean_step_seconds=statistics.mean(timings),median_step_seconds=statistics.median(timings),
            measured_seconds=end-start,updates_per_second=len(timings)/(end-start),
            avg_gpu_util=statistics.mean(x['gpu_util'] for x in samples) if samples else None,
            min_gpu_util=min((x['gpu_util'] for x in samples),default=None),max_gpu_util=max((x['gpu_util'] for x in samples),default=None),
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
            sample_count=len(samples),step_seconds=timings,samples=samples)
        results.append(result)
        (a.output/'results.json').write_text(json.dumps(results,indent=2))
        print('RESULT',json.dumps({k:v for k,v in result.items() if k not in ('samples','step_seconds')}),flush=True)
        del model,optimizer,scheduler,losses,checks,speed
        gc.collect();torch.cuda.empty_cache()
    print('DONE',flush=True)


if __name__=='__main__':main()
