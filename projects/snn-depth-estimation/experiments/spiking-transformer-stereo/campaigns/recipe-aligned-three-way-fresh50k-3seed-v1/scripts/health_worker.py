"""Disposable native GPU throughput/parity test; never creates a scientific checkpoint."""
import argparse
import copy
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import resource
import statistics
import subprocess
import threading
import time
from fresh import *
from prefetch import Prefetch,decode_batch
from native_parity import native_parity,tensor_deltas
import hashlib

STEPS=128
WARMUP=4


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('overlay','parent-overlay','base','data-root','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--wave',choices=('1','2'),required=True);a=p.parse_args()
    rank=int(os.environ['SLURM_PROCID']);assert rank in (0,1)
    arms={'1':('original','original_I'),'2':('spiking_normalized','original')}[a.wave];arm=arms[rank]
    run=arm.lower()+'-seed1234-fresh50k-v1'
    config_path=a.overlay/'configs'/(run+'.json');manifest=a.base/'projects/snn-depth-estimation/experiments/spiking-transformer-stereo/configs/manifest.json'
    c,reg,source=verify_source(a.overlay/'source-registry.json',config_path,a.overlay/'configs/registration.json',manifest)
    recipe,parent,driver,mechanism,parent_source,base_source=load_science(a.parent_overlay,a.base)
    verify_dependencies(source,reg,parent_source,base_source)
    setup_cuda(1234)
    import torch
    out=a.output/('wave'+a.wave)/('rank'+str(rank)+'-'+arm);out.mkdir(parents=True,exist_ok=False)
    def barrier(name):
        directory=a.output/('wave'+a.wave)/('barrier-'+name);directory.mkdir(parents=True,exist_ok=True)
        write(directory/(str(rank)+'.json'),{'ready_utc':datetime.now(timezone.utc).isoformat()})
        begun=time.monotonic()
        while not all((directory/(str(i)+'.json')).exists() for i in (0,1)):
            if time.monotonic()-begun>900:raise RuntimeError('paired health worker barrier timeout')
            time.sleep(.1)
    model=recipe.build_model(driver,mechanism,arm,1234)
    proof=parent.initialization_proof(driver,mechanism,model,arm,1234);model=model.cuda();opt,sch=optimizer(model)
    initial_hash=state_sha(model.state_dict());initial_rng=rng();data=read(manifest)
    train=driver.ManifestDataset(a.data_root,data['train']);order=torch.randperm(960,generator=torch.Generator().manual_seed(1234)).tolist()
    write(out/'initialization.json',proof)
    # Explicit all-H/V actual sample equality uses the identical frozen augmentation.
    original=train[order[0]];hv={}
    saved_rng=rng_sha(rng())
    for h,v in ((False,False),(False,True),(True,False),(True,True)):
        first=recipe.augment(original,h,v);second=recipe.augment(train[order[0]],h,v)
        if not all(torch.equal(x,y) for x,y in zip(first[:4],second[:4])):raise RuntimeError('actual all-H/V decoding differs')
        hv[str((h,v))]=state_sha(first[:4])
    sync=decode_batch(train,recipe,order,1234,0)
    with Prefetch(train,recipe,order,1234,0,1) as loader:prefetched=loader.get(0)
    if (sync.indices,sync.flags)!=(prefetched.indices,prefetched.flags):raise RuntimeError('prefetch order/flags differ')
    if not all(torch.equal(x,y) for left,right in zip(sync.items,prefetched.items) for x,y in zip(left[:4],right[:4])):raise RuntimeError('actual prefetched tensors differ')
    if rng_sha(rng())!=saved_rng:raise RuntimeError('actual CPU loading consumed RNG')
    del original,sync,prefetched
    parity=native_parity(model,opt,sch,driver,recipe,train,order,1234,out)
    if initial_hash!=state_sha(model.state_dict()) or optimizer_steps(opt)!=[0,0] or sch.last_epoch!=0:raise RuntimeError('parity altered fresh initialization')
    # Four explicit engineering optimizer steps warm allocations including real AdamW moments.
    for step in range(WARMUP):train_step(model,opt,sch,driver,decode_batch(train,recipe,order,1234,step).items)
    warm_model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    warm_opt=copy.deepcopy(opt.state_dict());warm_sch=copy.deepcopy(sch.state_dict());warm_rng=rng()
    assert optimizer_steps(opt)==[WARMUP,WARMUP] and sch.last_epoch==WARMUP
    def reset():
        driver.reset_model(model);model.load_state_dict(warm_model,strict=True);opt.load_state_dict(copy.deepcopy(warm_opt));sch.load_state_dict(copy.deepcopy(warm_sch));restore_rng(warm_rng)
        opt.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
    # Resolve this process's physical GPU UUID without assuming global device ordinal.
    applications=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True)
    uuids=[line.split(',')[1].strip() for line in applications.splitlines() if line.split(',')[0].strip()==str(os.getpid())]
    if len(set(uuids))!=1:raise RuntimeError('cannot identify uniquely assigned GPU UUID')
    uuid=uuids[0];results={};trajectories={};serial_final_model=None;serial_final_optimizer=None
    for mode in ('serial','prefetch'):
        reset();barrier(mode);samples=[];stop=threading.Event()
        def telemetry():
            while not stop.is_set():
                try:
                    values=subprocess.check_output(['nvidia-smi','-i',uuid,'--query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu,clocks.sm,pstate','--format=csv,noheader,nounits'],text=True,timeout=5).strip().split(',')
                    samples.append(dict(monotonic=time.monotonic(),gpu_util=float(values[0]),memory_util=float(values[1]),memory_mib=float(values[2]),power_w=float(values[3]),temperature_c=float(values[4]),sm_clock_mhz=float(values[5]),pstate=values[6].strip()))
                except Exception as error:samples.append(dict(monotonic=time.monotonic(),error=str(error)))
                stop.wait(1.)
        thread=threading.Thread(target=telemetry,daemon=True);thread.start()
        # Queue/thread startup is charged to the measured prefetch window.
        started=time.monotonic();cpu_started=time.process_time();rows=[]
        loader=Prefetch(train,recipe,order,1234,WARMUP,WARMUP+STEPS) if mode=='prefetch' else None
        try:
            for step in range(WARMUP,WARMUP+STEPS):
                begun=time.monotonic();batch=loader.get(step) if loader else decode_batch(train,recipe,order,1234,step)
                wait=time.monotonic()-begun
                row=train_step(model,opt,sch,driver,batch.items)
                identity=dict(manifest_sha256=reg['manifest_sha256'],seed=1234,global_index=step,indices=batch.indices,augmentation_hv=batch.flags,rows=[data['train'][i] for i in batch.indices])
                input_identity=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
                row.update(update=step+1,indices=batch.indices,augmentation_hv=batch.flags,input_identity_sha256=input_identity,consumer_wait_seconds=wait,loader_work_seconds=batch.loader_work_seconds,seconds=time.monotonic()-begun)
                rows.append(row);del batch
            torch.cuda.synchronize();ended=time.monotonic();cpu_seconds=time.process_time()-cpu_started
        finally:
            if loader:loader.close()
            stop.set();thread.join()
        if optimizer_steps(opt)!=[WARMUP+STEPS,WARMUP+STEPS] or sch.last_epoch!=WARMUP+STEPS:raise RuntimeError('engineering optimizer clocks differ')
        # Fingerprints happen outside the throughput window.
        trajectory=[{key:r[key] for key in ('update','indices','augmentation_hv','input_identity_sha256','valid_pixels','lr_used','lr_next')} for r in rows]
        trajectories[mode]=trajectory
        util=[v['gpu_util'] for v in samples if 'gpu_util' in v and started<=v['monotonic']<=ended]
        if not util:raise RuntimeError('missing assigned-GPU telemetry')
        total=ended-started
        results[mode]=dict(window_seconds=total,logical_updates=STEPS,presentations=STEPS*8,seconds_per_update=total/STEPS,updates_per_second=STEPS/total,
            consumer_wait_fraction=sum(r['consumer_wait_seconds'] for r in rows)/total,loader_work_seconds=sum(r['loader_work_seconds'] for r in rows),
            gpu_util_mean=statistics.mean(util),gpu_util_median=statistics.median(util),gpu_util_low_under10_fraction=sum(v<10 for v in util)/len(util),gpu_util_zero_fraction=sum(v==0 for v in util)/len(util),telemetry_samples=len(util),
            temperature_c_mean=statistics.mean(v['temperature_c'] for v in samples if 'temperature_c' in v),temperature_c_max=max(v['temperature_c'] for v in samples if 'temperature_c' in v),sm_clock_mhz_mean=statistics.mean(v['sm_clock_mhz'] for v in samples if 'sm_clock_mhz' in v),pstates=sorted({v['pstate'] for v in samples if 'pstate' in v}),
            process_cpu_seconds=cpu_seconds,mean_cpu_cores_used=cpu_seconds/total,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            peak_cuda_memory_mib=torch.cuda.max_memory_allocated()/2**20,final_model_sha256=state_sha(model.state_dict()),final_optimizer_sha256=state_sha(opt.state_dict()),final_rng_sha256=rng_sha(rng()),final_model_optimizer_finite=finite(model.state_dict()) and finite(opt.state_dict()))
        if mode=='serial':
            serial_final_model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            serial_final_optimizer=copy.deepcopy(opt.state_dict())
        else:
            results[mode]['descriptive_final_model_drift']=tensor_deltas(serial_final_model,model.state_dict())
            results[mode]['descriptive_final_optimizer_drift']=tensor_deltas(serial_final_optimizer,opt.state_dict())
        write(out/(mode+'-rows.json'),rows);write(out/(mode+'-telemetry.json'),samples)
        barrier(mode+'-finished')
    trajectory_equal=trajectories['serial']==trajectories['prefetch']
    state_equal=all(results['serial'][k]==results['prefetch'][k] for k in ('final_model_sha256','final_optimizer_sha256','final_rng_sha256'))
    speedup=results['serial']['seconds_per_update']/results['prefetch']['seconds_per_update']
    enough=all(v['window_seconds']>=60 for v in results.values())
    rng_equal=results['serial']['final_rng_sha256']==results['prefetch']['final_rng_sha256']
    passed=enough and trajectory_equal and rng_equal and all(r['final_model_optimizer_finite'] for r in results.values()) and speedup>=1.15 and results['prefetch']['consumer_wait_fraction']<=.15
    result=dict(status='PASS' if passed else 'INSUFFICIENT_WINDOW' if not enough else 'FAIL',arm=arm,wave=int(a.wave),rank=rank,source_identity_sha256=source['source_identity_sha256'],registration_sha256=sha(a.overlay/'configs/registration.json'),config_sha256=sha(config_path),
        initial_model_sha256=initial_hash,initialization=proof,actual_tensor_and_all_hv_equal=True,loading_rng_unchanged=True,zero_update_forward_exact=True,zero_update_model_optimizer_rng_unchanged=True,gradients_within_registered_native_variability=True,
        parity=parity,all_hv_hashes=hv,results=results,speedup_ratio=speedup,all128_input_identity_order_hv_lr_validity_exact=trajectory_equal,final_benchmark_state_bitwise_equal_descriptive=state_equal,final_rng_exact=rng_equal,
        final_model_optimizer_bitwise_equality_is_gate=False,input_identity_definition='SHA256 immutable manifest identity + selected rows + indices + seed + global index + H/V; not a full tensor hash; native9passes verify full tensors outside timing',
        mode_order=['serial','prefetch'],mode_order_limitation='fixed serial then prefetch; thermal/clock/cache order effects are reported, not separately randomized',
        scientific_optimizer_updates=0,engineering_optimizer_updates=WARMUP+2*STEPS,zero_step_native_backward_passes=9,production_reuses_no_benchmark_state=True,gpu_uuid=uuid,gpu=torch.cuda.get_device_name(0),
        slurm_job_id=os.environ['SLURM_JOB_ID'],cpu_affinity=sorted(os.sched_getaffinity(0)),physical_batch=4,effective_batch=8,timestamp_utc=datetime.now(timezone.utc).isoformat())
    write(out/'result.json',result)
    if not passed:raise RuntimeError('prelaunch health gate did not pass: '+result['status'])


if __name__=='__main__':main()
