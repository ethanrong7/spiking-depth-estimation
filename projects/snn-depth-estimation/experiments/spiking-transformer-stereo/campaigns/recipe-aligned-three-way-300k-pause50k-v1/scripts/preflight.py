"""Four complete cycles per disposable worker: updates, learning, resume and GPU health."""
import argparse
from contextlib import ExitStack
from datetime import datetime,timezone
import os
from pathlib import Path
import resource
import statistics
import time
from stage import *
from prefetch import Prefetch,decode_batch
from health import learning_health,parameter_health,parameters_cpu
from telemetry import Sampler


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('overlay','runtime-overlay','parent-overlay','base','data-root','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--wave',choices=('1','2'),required=True);a=p.parse_args();begun=time.monotonic()
    rank=int(os.environ['SLURM_PROCID']);assert rank in (0,1)
    arm={'1':('original','original_I'),'2':('spiking_normalized','original')}[a.wave][rank];run=arm.lower()+'-seed1234-300k-v1'
    manifest=a.base/'projects/snn-depth-estimation/experiments/spiking-transformer-stereo/configs/manifest.json'
    c,reg,source=verify_source(a.overlay,a.overlay/'configs'/(run+'.json'),manifest)
    rt,recipe,parent,driver,mechanism,rr,pr,br=load_runtime(a.runtime_overlay,a.parent_overlay,a.base);verify_dependencies(source,reg,rr,pr,br)
    rt.setup_cuda(1234)
    import torch
    model=recipe.build_model(driver,mechanism,arm,1234);proof=parent.initialization_proof(driver,mechanism,model,arm,1234);model=model.cuda();opt,sch=optimizer(model)
    if rt.optimizer_steps(opt)!=[0,0] or sch.last_epoch!=0 or opt.param_groups[0]['lr']!=8e-6:raise RuntimeError('preflight must be fresh0')
    ctx=context(c,source,reg,proof,'engineering_resume_probe');data=read(manifest);train=driver.ManifestDataset(a.data_root,data['train'])
    order=torch.randperm(960,generator=torch.Generator().manual_seed(1234)).tolist()
    out=a.output/('wave'+a.wave)/('rank'+str(rank)+'-'+arm);out.mkdir(parents=True,exist_ok=False)
    write(out/'initialization.json',proof);write(out/'sample-order.json',dict(order=order,effective_batch=8));write(out/'history.json',[])
    for name in ('train.jsonl','loss-components.jsonl','evaluations.jsonl'):(out/name).touch()
    ready=a.output/('wave'+a.wave)/'ready';ready.mkdir(parents=True,exist_ok=True);write(ready/(str(rank)+'.json'),dict(timestamp_utc=datetime.now(timezone.utc).isoformat()))
    wait_start=time.monotonic()
    while not all((ready/(str(i)+'.json')).exists() for i in (0,1)):
        if time.monotonic()-wait_start>900:raise RuntimeError('paired worker startup timeout')
        time.sleep(.1)
    sampler=Sampler();windows=[];inspections=[];rows=[];named=[];before=parameters_cpu(model);resume_proof=None
    try:
        with ExitStack() as stack:
            stream=stack.enter_context(Prefetch(train,recipe,order,1234,0,480));pending=None
            for cycle in range(1,5):
                window_start=time.monotonic();cpu_start=time.process_time();torch.cuda.reset_peak_memory_stats()
                for step in range((cycle-1)*120,cycle*120):
                    start=time.monotonic()
                    prepared_during_resume=pending is not None
                    if pending is not None:batch=pending;pending=None
                    else:batch=stream.get(step)
                    wait=time.monotonic()-start
                    if batch.global_index!=step:raise RuntimeError('resumed prefetched batch offset differs')
                    row=rt.train_step(model,opt,sch,driver,batch.items)
                    row.update(update=step+1,samples_seen=(step+1)*8,indices=batch.indices,augmentation_hv=batch.flags,consumer_wait_seconds=wait,
                        loader_work_seconds=batch.loader_work_seconds,prepared_during_resume_inspection=prepared_during_resume,seconds=time.monotonic()-start,cycle=cycle,timestamp_utc=datetime.now(timezone.utc).isoformat())
                    if sch.last_epoch!=step+1:raise RuntimeError('preflight global scheduler clock differs')
                    rows.append(row);append(out/'train.jsonl',row);append(out/'loss-components.jsonl',{key:row[key] for key in ('update','loss','disparity_loss','neuron_loss','valid_pixels')});del batch
                torch.cuda.synchronize();window_end=time.monotonic();cpu_seconds=time.process_time()-cpu_start
                windows.append(dict(cycle=cycle,started_monotonic=window_start,ended_monotonic=window_end,seconds=window_end-window_start,updates=120,cpu_seconds=cpu_seconds,
                    peak_cuda_memory_mib=torch.cuda.max_memory_allocated()/2**20))
                inspect_start=time.monotonic();update=cycle*120
                if rt.optimizer_steps(opt)!=[update,update] or sch.last_epoch!=update or sch.total_steps!=300100:raise RuntimeError('cycle clocks differ')
                finite=rt.finite(model.state_dict()) and rt.finite(opt.state_dict())
                if not finite:raise FloatingPointError('cycle model/AdamW state nonfinite')
                health,before=parameter_health(model,before,arm);health.update(cycle=cycle,update=update,optimizer_step_min_max=rt.optimizer_steps(opt),scheduler_epoch=sch.last_epoch,model_optimizer_finite=finite)
                named.append(health);write(out/'named-parameter-health.json',named)
                if update==240:
                    file=out/'engineering-checkpoint240.pt';checkpoint_sha=checkpoint(file,rt,ctx,model,opt,sch,240,order,[],None,None)
                    next_batch=stream.get(240);signature_before=forward_signature(model,driver,rt,next_batch.items)
                    with stream.queue.mutex:discarded_before_close=[batch.global_index for batch in stream.queue.queue if hasattr(batch,'global_index')]
                    stream.close()
                    old_producer_closed=not stream.thread.is_alive()
                    model_hash=rt.state_sha(model.state_dict());opt_hash=rt.state_sha(opt.state_dict());rng_hash=rt.rng_sha(rt.rng());lr=opt.param_groups[0]['lr']
                    rebuilt=recipe.build_model(driver,mechanism,arm,1234).cuda();new_opt,new_sch=optimizer(rebuilt)
                    ck=restore(file,rt,ctx,rebuilt,new_opt,new_sch,expected_update=240);strict_log_boundary(out,ck)
                    if ck['order']!=order:raise RuntimeError('engineering resume order differs')
                    stream=stack.enter_context(Prefetch(train,recipe,order,1234,240,480));pending=stream.get(240)
                    queue_input_equal=(pending.global_index==240 and pending.indices==next_batch.indices and pending.flags==next_batch.flags and all(torch.equal(x,y) for a,b in zip(next_batch.items,pending.items) for x,y in zip(a[:4],b[:4])))
                    signature_after=forward_signature(rebuilt,driver,rt,pending.items)
                    exact=dict(reconstructed_queue_input=queue_input_equal,old_producer_closed=old_producer_closed,model=rt.state_sha(rebuilt.state_dict())==model_hash,optimizer=rt.state_sha(new_opt.state_dict())==opt_hash,rng=rt.rng_sha(rt.rng())==rng_hash,
                        scheduler=new_sch.state_dict()==sch.state_dict(),lr=new_opt.param_groups[0]['lr']==lr==new_sch.get_last_lr()[0],input_tensors=signature_before['input_tensor_sha256']==signature_after['input_tensor_sha256'],forward_outputs=signature_before['micro_prediction_sha256']==signature_after['micro_prediction_sha256'])
                    resume_proof=dict(status='PASS' if all(exact.values()) else 'FAIL',checkpoint_sha256=checkpoint_sha,exact=exact,signature_before=signature_before,signature_after=signature_after,
                        queue_restarted_global_index=240,discarded_unconsumed_prefetch_data=True,discarded_queued_indices_before_close=discarded_before_close,producer_inflight_may_have_been_discarded=True,first_reconstructed_batch_retained_for_actual_update241=True,
                        restored_update=240,next_update=241,next_presentation=1920,optimizer_step_min_max=rt.optimizer_steps(new_opt),scheduler_epoch=new_sch.last_epoch,scheduler_total_steps=new_sch.total_steps,lr_next_update=lr,source_identity_sha256=source['source_identity_sha256'],config_sha256=sha(a.overlay/'configs'/(run+'.json')),context_exact=ck['context']==ctx,log_prefix_exact=True,scientific_optimizer_updates=0)
                    write(out/'resume240.json',resume_proof)
                    if not all(exact.values()):raise RuntimeError('native production-helper240 resume mismatch; raw evidence saved')
                    old_model=model;old_opt=opt;old_sch=sch;model=rebuilt;opt=new_opt;sch=new_sch
                    rt.restore_rng(ck['rng']);del old_model,old_opt,old_sch,ck,next_batch;torch.cuda.empty_cache()
                inspections.append(dict(cycle=cycle,seconds=time.monotonic()-inspect_start))
    finally:sampler.close();write(out/'telemetry.json',sampler.samples);write(out/'windows.json',windows);write(out/'inspections.json',inspections)
    learning=learning_health(rows);write(out/'learning-health.json',learning)
    active=windows[1:];duration=sum(w['seconds'] for w in active);active_rows=rows[120:]
    mean_step=duration/360;wait_fraction=sum(r['consumer_wait_seconds'] for r in active_rows)/duration;telemetry=sampler.summary(active)
    reference=reg['preflight']['reference_prefetch_seconds_per_update'][arm];limit=1.25*reference
    exact_counts=[r['update'] for r in rows]==list(range(1,481)) and all(r['samples_seen']==r['update']*8 for r in rows)
    throughput_pass=duration>=180 and wait_fraction<=.15 and mean_step<=limit and telemetry['gpu_util_mean']>=50
    named_pass=all(h['status']=='PASS' for h in named);resume_pass=resume_proof is not None and resume_proof['status']=='PASS'
    passed=throughput_pass and learning['status']=='PASS' and named_pass and resume_pass and exact_counts and all(rt.finite(r) and r['gradient_norm']>0 for r in rows)
    result=dict(status='PASS' if passed else 'FAIL',arm=arm,wave=int(a.wave),rank=rank,source_identity_sha256=source['source_identity_sha256'],registration_sha256=source['registration_sha256'],
        engineering_optimizer_updates=480,scientific_optimizer_updates=0,initialization=proof,planned_scheduler_updates=300000,scheduler_total_steps=300100,optimizer_step_min_max=rt.optimizer_steps(opt),scheduler_epoch=sch.last_epoch,
        exact_update_sample_counts=exact_counts,named_parameter_gate='PASS' if named_pass else 'FAIL',resume_gate='PASS' if resume_pass else 'FAIL',learning_gate=learning['status'],throughput_gate='PASS' if throughput_pass else 'FAIL',
        active_training_seconds=duration,active_seconds_per_update=mean_step,reference_prefetch_seconds_per_update=reference,maximum_allowed_seconds_per_update=limit,foreground_wait_fraction=wait_fraction,telemetry=telemetry,
        windows=windows,inspection_seconds=sum(v['seconds'] for v in inspections),cycle1_warmup_training_seconds=windows[0]['seconds'],
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,cpu_affinity=sorted(os.sched_getaffinity(0)),physical_batch=4,effective_batch=8,
        final_model_optimizer_finite=rt.finite(model.state_dict()) and rt.finite(opt.state_dict()),preflight_elapsed_seconds=time.monotonic()-begun,job_id=os.environ['SLURM_JOB_ID'],timestamp_utc=datetime.now(timezone.utc).isoformat(),production_reuses_no_engineering_state=True)
    write(out/'result.json',result)
    if not passed:raise RuntimeError('registered preflight gate failed; no extension, tuning or retry')


if __name__=='__main__':main()
