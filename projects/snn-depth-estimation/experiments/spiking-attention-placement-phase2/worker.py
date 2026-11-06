"""Independent fresh/native worker, shared assigned GPU, complete safe recovery."""
from pathlib import Path
import argparse,fcntl,hashlib,importlib.util,json,os,random,signal,sys,time
from datetime import datetime,timezone
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/"shared-gpu-30k-v1"))
from common import atomic_json,append,reconcile,phase_lock
from protocol import Prefetch,SEEDS,ARMS,presentation,expected_lr,validate_row,audit_history,tensor_hash,sha256
PROJECT=Path(__file__).resolve().parents[2]
MANIFEST_SHA='bad4369b6fd8b0cdf871a20b1727c2d9b59733a16bd33f75904227bdfef4510a'
EVALS=[0,250,500,1000,1500,2500]+list(range(5000,30001,2500))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('data-root','manifest','root'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--stop-at',type=int,choices=(7500,30000),default=7500);p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--seed',type=int,choices=tuple(SEEDS),required=True);p.add_argument('--branch-seed',type=int);p.add_argument('--preflight',action='store_true');a=p.parse_args()
    a.branch_seed=SEEDS[a.seed] if a.branch_seed is None else a.branch_seed
    assert a.branch_seed==SEEDS[a.seed]
    import numpy as np
    import torch
    import cv2
    from torch.utils.data import DataLoader
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available() and torch.cuda.device_count()==1
    torch.set_num_threads(6);cv2.setNumThreads(1);assert hashlib.sha256(a.manifest.read_bytes()).hexdigest()==MANIFEST_SHA
    manifest=json.loads(a.manifest.read_text());assert len(manifest['train'])==960 and len(manifest['dev'])==60
    sys.path.insert(0,str(PROJECT));driver_path=PROJECT/'experiments/spiking-transformer-stereo/scripts/run_experiment.py';sys.path.insert(0,str(driver_path.parent))
    spec=importlib.util.spec_from_file_location('shared_native_driver',driver_path);driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
    from core.spiking_attention_phase2 import install
    registry=json.loads((PROJECT.parents[1]/'source-registry.json').read_text())
    for rel,digest in registry['source_files'].items():assert hashlib.sha256((PROJECT.parents[1]/rel).read_bytes()).hexdigest()==digest
    a.root.mkdir(parents=True,exist_ok=True);outdir=a.root/f'seed-{a.seed}'/a.arm;outdir.mkdir(parents=True,exist_ok=True)
    lease=(outdir/'worker.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    assert str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-')==os.environ['PLACEMENT_GPU_UUID'].removeprefix('GPU-')
    assert os.environ['PHASE2_SLURM_JOB_ID']==os.environ['SLURM_JOB_ID']
    assert any(tag in torch.cuda.get_device_name(0) for tag in ('RTX 4090','RTX 5090'))
    phase_root=a.root/'gpu-locks'/os.environ['SLURM_JOB_ID']/os.environ['PLACEMENT_GPU_UUID'];phase_root.mkdir(parents=True,exist_ok=True)
    lock_name=f'seed-{a.seed}-{a.arm}-{os.getpid()}'
    model=driver.build_model('original',a.seed)
    native_initial=tensor_hash(model.state_dict())
    if a.arm!='Original':model=install(model,cell=a.arm,seed=a.branch_seed)
    common_initial=tensor_hash({n:v for n,v in model.state_dict().items() if not n.startswith('placement.')})
    assert common_initial==native_initial,'optional branch changed common initial weights'
    model=model.cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-5,eps=1e-8)
    scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=2e-4,total_steps=30000,pct_start=.01,cycle_momentum=False,anneal_strategy='linear')
    train=driver.ManifestDataset(a.data_root,manifest['train']);dev=driver.ManifestDataset(a.data_root,manifest['dev'][:2] if a.preflight else manifest['dev'])
    loader=DataLoader(dev,batch_size=2,num_workers=0,generator=torch.Generator().manual_seed(a.seed+991))
    order=torch.randperm(960,generator=torch.Generator().manual_seed(a.seed)).tolist();steps=3 if a.preflight else 30000;evals=[0,2] if a.preflight else EVALS
    prior_peak_reserved=0.;prior_peak_allocated=0.;completed=0;evaluated=set();stop=[False];checkpoint_path=outdir/'last.pt';started=time.monotonic()
    signal.signal(signal.SIGTERM,lambda *args:stop.__setitem__(0,True));signal.signal(signal.SIGUSR1,lambda *args:stop.__setitem__(0,True))
    def state_hash():
        h=hashlib.sha256()
        for n,v in sorted(model.state_dict().items()):h.update(n.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    def shared_hash():
        h=hashlib.sha256()
        for n,v in sorted(model.state_dict().items()):
            if n.startswith(('fnet.','cnet.','context_zqr_convs.')):h.update(n.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    config=dict(study='spiking-attention-placement-phase2',coding='native' if a.arm=='Original' else ('signed_T4_threshold1_leak0.9_full_BPTT' if a.arm in ('SC','SP') else 'tanh'),arm=a.arm,seed=a.seed,branch_seed=a.branch_seed,updates=steps,schedule_total=30000,effective_batch=8,physical_batch=4,augmentation='hv',eval_updates=evals,prefetch_depth=2,prefetch_threads=1,manifest_sha256=MANIFEST_SHA)
    def status(phase):atomic_json(outdir/'status.json',dict(phase=phase,update=completed,pid=os.getpid(),updated_utc=datetime.now(timezone.utc).isoformat()))
    def checkpoint():
        driver.reset_model(model);optimizer.zero_grad(set_to_none=True)
        state=dict(peak_reserved_mib=max(prior_peak_reserved,torch.cuda.max_memory_reserved()/2**20),peak_allocated_mib=max(prior_peak_allocated,torch.cuda.max_memory_allocated()/2**20),model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),updates=completed,samples_seen=completed*8,evaluated=sorted(evaluated),config=config,source_identity=registry['source_identity_sha256'],order=order,rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),loader=loader.generator.get_state()))
        temp=outdir/'last.tmp.pt';torch.save(state,temp);temp.replace(checkpoint_path)
        atomic_json(outdir/'checkpoint.json',dict(update=completed,scheduler_last_epoch=scheduler.last_epoch,evaluated=sorted(evaluated),bytes=checkpoint_path.stat().st_size,complete=True))
    if checkpoint_path.exists():
        ck=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
        prior_peak_reserved=ck.get('peak_reserved_mib',0.);prior_peak_allocated=ck.get('peak_allocated_mib',0.)
        assert ck['config']==config and ck['source_identity']==registry['source_identity_sha256'] and ck['order']==order
        model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler']);completed=ck['updates'];evaluated=set(ck['evaluated'])
        for state in optimizer.state.values():
            for k,v in state.items():
                if torch.is_tensor(v) and k!='step':state[k]=v.cuda()
        for name,value in model.state_dict().items():assert torch.equal(value.detach().cpu(),ck['model'][name]),name
        random.setstate(ck['rng']['python']);np.random.set_state(ck['rng']['numpy']);torch.set_rng_state(ck['rng']['torch']);torch.cuda.set_rng_state_all(ck['rng']['cuda']);loader.generator.set_state(ck['rng']['loader']);del ck
        assert scheduler.last_epoch==completed
        saved_steps=[int(s['step']) for s in optimizer.state.values() if 'step' in s]
        assert not saved_steps or min(saved_steps)==max(saved_steps)==completed
        reconcile(outdir/'train.jsonl',lambda r:r['update']<=completed);reconcile(outdir/'evaluations.jsonl',lambda r:r['update'] in evaluated)
        append(outdir/'recoveries.jsonl',dict(restored_update=completed,time_utc=datetime.now(timezone.utc).isoformat(),scheduler=scheduler.last_epoch))
    else:
        if (outdir/'metadata.json').exists():
            old=json.loads((outdir/'metadata.json').read_text())
            assert not (outdir/'train.jsonl').exists() and old['config']==config and old['source_registry']==registry and old['initial_state_sha256']==state_hash(),'incomplete output is not an untouched initialization'
        else:atomic_json(outdir/'metadata.json',dict(config=config,source_registry=registry,initial_state_sha256=state_hash(),shared_state_sha256=shared_hash(),native_initial_state_sha256=native_initial,common_initial_state_sha256=common_initial,branch_initial_state_sha256=tensor_hash(model.placement.state_dict()) if a.arm!='Original' else None,order=order,gpu_uuid=str(getattr(torch.cuda.get_device_properties(0),'uuid','unavailable')),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),slurm={k:v for k,v in os.environ.items() if k.startswith('SLURM_')},torch=torch.__version__,started_utc=datetime.now(timezone.utc).isoformat()))
        checkpoint()
    @torch.no_grad()
    def evaluate():
        status('waiting_validation')
        py,nprng=random.getstate(),np.random.get_state()
        with phase_lock(phase_root,lock_name,True,torch.cuda.empty_cache),torch.random.fork_rng(devices=[0]):
            status('validation');model.eval();frames=[]
            for left,right,target,valid,indices in loader:
                left,right,target,valid=[v.cuda() for v in (left,right,target,valid)]
                pred=driver.forward(model,left,right,32)
                assert torch.isfinite(pred).all()
                for b,index in enumerate(indices.tolist()):
                    row=dev.rows[index];mask=valid[b];error=(pred[b,0]-target[b,0]).abs()[mask]
                    frame=dict(scene=row['scene'],frame=row['frame'],valid_pixels=int(mask.sum()),**driver.metric_values(error))
                    frames.append(frame)
                driver.reset_model(model);del left,right,target,valid,pred
        random.setstate(py);np.random.set_state(nprng)
        assert len(frames)==len(dev.rows) and all(frame['valid_pixels']>0 for frame in frames),'incomplete validation collection'
        scenes={s:float(np.mean([r['epe'] for r in frames if r['scene']==s])) for s in sorted({r['scene'] for r in frames})};count=sum(r['valid_pixels'] for r in frames)
        result=dict(update=completed,samples_seen=completed*8,mean_scene_epe=float(np.mean(list(scenes.values()))),scene_epe=scenes,per_frame=frames,pooled_epe=sum(r['epe']*r['valid_pixels'] for r in frames)/count,**{k:sum(r[k]*r['valid_pixels'] for r in frames)/count for k in ('bad1','bad3','bad5')})
        append(outdir/'evaluations.jsonl',result);evaluated.add(completed);checkpoint();print(a.arm,'DEV',completed,result['mean_scene_epe'],flush=True)
    def preserve_endpoint():
        import shutil
        assert completed in evaluated and scheduler.last_epoch==completed
        assert all(torch.isfinite(v).all() for v in model.state_dict().values())
        assert all(torch.isfinite(v).all() for state in optimizer.state.values() for v in state.values() if torch.is_tensor(v))
        rows=[json.loads(line) for line in (outdir/'evaluations.jsonl').read_text().splitlines()]
        evaluation=next(row for row in rows if row['update']==completed)
        assert len(evaluation['per_frame'])==60 and len(evaluation['scene_epe'])==30
        assert all(row['valid_pixels']>0 for row in evaluation['per_frame'])
        audit=audit_history(outdir/'train.jsonl',order,a.seed,completed)
        target=outdir/f'step-{completed}.pt'
        if not target.exists():
            temporary=target.with_suffix('.tmp.pt');shutil.copyfile(checkpoint_path,temporary);temporary.replace(target)
        prior_receipt=outdir/f'endpoint-{completed}.json'
        if prior_receipt.exists():
            receipt=json.loads(prior_receipt.read_text());assert receipt['checkpoint_sha256']==sha256(target)
            assert receipt['config']==config and receipt['source_identity']==registry['source_identity_sha256']
            return
        atomic_json(outdir/f'endpoint-{completed}.json',dict(update=completed,scheduler_last_epoch=scheduler.last_epoch,checkpoint=str(target),checkpoint_sha256=sha256(target),config=config,source_identity=registry['source_identity_sha256'],native_initial_state_sha256=native_initial,common_initial_state_sha256=common_initial,evaluation=evaluation,stream_audit=audit,slurm_job_id=os.environ['SLURM_JOB_ID'],gpu_uuid=str(torch.cuda.get_device_properties(0).uuid)))
    audit_history(outdir/'train.jsonl',order,a.seed,completed)
    prefetch=Prefetch(train,order,completed,steps,a.seed)
    try:
        if completed in evals and completed not in evaluated:evaluate()
        torch.cuda.reset_peak_memory_stats()
        while completed<min(steps,a.stop_at):
            if stop[0] or (a.root/'pause.request').exists():checkpoint();status('paused');return 75
            wait=time.monotonic();batch=prefetch.get(completed);consumer_wait=time.monotonic()-wait;status('training')
            with phase_lock(phase_root,lock_name,False,lambda:torch.cuda.empty_cache() if list(phase_root.glob('validation-*.request')) else None):
                tick=time.monotonic();model.train()
                model.freeze_bn()
                optimizer.zero_grad(set_to_none=True);items=batch['items']
                target=torch.stack([r[2] for r in items]);valid=torch.stack([r[3] for r in items])[:,None]
                counts=dict(valid=int(valid.sum()))
                total=0.;parts_sum={};activity=None
                for offset in (0,4):
                    micro=items[offset:offset+4];left,right,target,valid=[torch.stack([r[j] for r in micro]).cuda() for j in range(4)]
                    preds,spikes,volts=driver.forward(model,left,right,16,True);loss=driver.disparity_loss(preds,target,valid)*int(valid.sum())/counts['valid'];del preds,spikes,volts
                    assert torch.isfinite(loss);loss.backward();total+=float(loss.detach());driver.reset_model(model);del left,right,target,valid,loss,micro
                q_gradient=None
                gradients={n:float(v.grad.norm()) for n,v in model.placement.named_parameters() if v.grad is not None} if a.arm!='Original' else {}
                assert all(np.isfinite(v) for v in gradients.values())
                if a.preflight and a.arm!='Original' and completed>0:
                    assert all(gradients.get('codes.'+name+'.weight',0)>0 for name in ('q','k','v')),gradients
                norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.));assert np.isfinite(norm) and norm>0
                lr=optimizer.param_groups[0]['lr'];optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();completed+=1
                row=dict(update=completed,samples_seen=completed*8,indices=batch['indices'],augmentation_hv=batch['flags'],loss=total,**parts_sum,gradient_norm=norm,final_qk_gradient=q_gradient,placement_gradients=gradients,lr_used=lr,lr_next=scheduler.get_last_lr()[0],spike_activity=activity,seconds=time.monotonic()-tick,consumer_wait_seconds=consumer_wait,decode_seconds=batch['decode_seconds'],peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,resident_allocated_mib=torch.cuda.memory_allocated()/2**20)
                validate_row(row,order,a.seed);append(outdir/'train.jsonl',row)
                del items,batch
            if completed%50==0:checkpoint()
            if completed in evals and completed not in evaluated:evaluate()
            if not a.preflight and completed in (7500,30000):preserve_endpoint()
            if a.preflight and completed==int(os.environ.get('SHARED_PREFLIGHT_PAUSE_AT','-1')):
                checkpoint();status('preflight_resume_probe');return 75
            if completed%25==0:print(a.arm,'UPDATE',completed,'LOSS',total,flush=True)
        checkpoint();status('screen_completed' if completed<steps else 'completed')
        if not a.preflight and completed in (7500,30000):preserve_endpoint()
        assert all(torch.isfinite(v).all() for v in model.state_dict().values())
        assert all(torch.isfinite(v).all() for state in optimizer.state.values() for v in state.values() if torch.is_tensor(v))
        atomic_json(outdir/'result.json',dict(status='SCREEN_COMPLETED' if completed<steps else 'COMPLETED',updates=completed,samples_seen=completed*8,scheduler_last_epoch=scheduler.last_epoch,evaluated=sorted(evaluated),finite_model_optimizer=True,peak_allocated_mib=max(prior_peak_allocated,torch.cuda.max_memory_allocated()/2**20),peak_reserved_mib=max(prior_peak_reserved,torch.cuda.max_memory_reserved()/2**20),process_seconds=time.monotonic()-started))
        return 0
    except torch.cuda.OutOfMemoryError as error:
        atomic_json(outdir/'memory-failure.json',dict(update_attempt=completed+1,error=str(error),last_complete_checkpoint=json.loads((outdir/'checkpoint.json').read_text()),time_utc=datetime.now(timezone.utc).isoformat()));return 86
    finally:
        prefetch.close()
        for f in phase_root.glob('validation-'+lock_name+'.request'):f.unlink(missing_ok=True)

if __name__=='__main__':
    try:sys.exit(main())
    except Exception as error:
        import torch
        if isinstance(error,torch.cuda.OutOfMemoryError):
            root=Path(sys.argv[sys.argv.index('--root')+1]);arm=sys.argv[sys.argv.index('--arm')+1];seed=int(sys.argv[sys.argv.index('--seed')+1]);folder=root/f'seed-{seed}'/arm;folder.mkdir(parents=True,exist_ok=True)
            atomic_json(folder/'memory-failure.json',dict(error=str(error),phase='initialization_or_restore',time_utc=datetime.now(timezone.utc).isoformat()));sys.exit(86)
        raise
