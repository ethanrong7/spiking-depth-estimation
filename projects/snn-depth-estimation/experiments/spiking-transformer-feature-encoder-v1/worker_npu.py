"""Encoder study: native NPU data/loss/solver loop, with new fnet adapters."""
import argparse
import copy
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from contextlib import nullcontext

os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
import numpy as np
import torch

EXP=Path(__file__).resolve().parent
PROJECT=EXP.parents[1]
ROOT=PROJECT.parents[1]
sys.path[:0]=[str(EXP),str(PROJECT),str(EXP.parent/'b-attention-mechanism-v1'),str(EXP.parent/'shared-gpu-30k-v1')]
from common import atomic_json, append, reconcile
from settings import load, add_input_arguments
from input_pipeline import InputPipeline
from input_pipeline_npu import RankPreparedPrefetch
from npu_backend import configure, environment, rng_state, restore_rng, preserve_rng, cpu_tree
from encoders import install, VARIANTS
from neuron import MultiStepLIFNode
from upstream import ensure_sources
contract=load('npu_presentation_contract',EXP.parent/'spiking-attention-placement-phase2/protocol.py')
MANIFEST_SHA=contract.MANIFEST_SHA


def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()


class SingleDevice:
    rank=0
    world=1
    main=True
    def __init__(self):
        assert int(os.environ.get('WORLD_SIZE','1'))==1
        self.device=configure(0)
        assert torch.npu.device_count()==1
    def barrier(self):pass
    def gather(self,obj):return [obj]
    def maximum(self,value):return float(value)



class Prefetch:
    """Original absolute presentation addresses; decode only this rank's share."""
    def __init__(self,dataset,order,start,stop,seed,rank,world):
        self.dataset,self.order,self.stop,self.seed=dataset,order,stop,seed
        self.rank,self.world,self.next=rank,world,start
        self.pending=deque();self.pool=ThreadPoolExecutor(max_workers=1)
        self.fill()
    def decode(self,step):
        begin=time.monotonic()
        indices,flags=contract.presentation(self.order,step,self.seed)
        width=8//self.world
        slots=list(range(self.rank*width,(self.rank+1)*width))
        items=[]
        for slot in slots:
            item=list(self.dataset[indices[slot]])
            h,v=flags[slot]
            if h:
                item[:4]=[x.flip(-1) for x in item[:4]];item[2]=-item[2]
            if v:item[:4]=[x.flip(-2) for x in item[:4]]
            items.append(item)
        return dict(indices=indices,flags=flags,items=items,slots=slots,
                    decode_seconds=time.monotonic()-begin)
    def fill(self):
        while len(self.pending)<2 and self.next<self.stop:
            step=self.next;self.next+=1
            self.pending.append((step,self.pool.submit(self.decode,step)))
    def get(self,step):
        expected,future=self.pending.popleft();assert step==expected
        result=future.result();self.fill();return result
    def close(self):
        for _,future in self.pending:future.cancel()
        self.pool.shutdown(wait=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,choices=(1234,2345),required=True)
    parser.add_argument('--variant',choices=VARIANTS,required=True)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--source-dir',type=Path,required=True)
    parser.add_argument('--physical-batch',type=int,choices=(1,2),default=2)
    for name in ('data-root','manifest','root','source-registry','initialization'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--stop-at',type=int,required=True)
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--pause-at',type=int)
    add_input_arguments(parser)
    parser.set_defaults(input_mode='preload',preload_gib=12.,input_workers=2,prefetch_depth=2)
    args=parser.parse_args()
    args.model='transformer-encoder'
    assert args.stop_at==(200 if args.preflight else 30000)
    assert args.pause_at is None or (args.preflight and args.pause_at==2)
    assert (args.root/'checkpoints'/'last.pt').exists()==args.resume, 'Resume must be explicit'
    assert args.input_mode=='preload' and args.preload_gib>0
    assert args.input_workers==2 and args.prefetch_depth==2
    group=SingleDevice();device=group.device
    if not args.preflight:
        admission=json.loads((args.source_registry.parent/'port-validation.json').read_text())
        assert admission['status']=='passed'
        expected=admission['cells'][0]['environment'];actual=environment()
        for key in ('python','torch','torch_npu','npu_name','precision'):
            assert actual[key]==expected[key],('NPU environment differs from preflight',key)

    physical_batch=args.physical_batch;effective_batch=8
    span=lambda name:nullcontext()
    import cv2
    cv2.setNumThreads(1)
    driver_path=PROJECT/'experiments/spiking-transformer-stereo/scripts/run_experiment.py'
    sys.path.insert(0,str(driver_path.parent))
    driver=load('npu_native_driver',driver_path)
    registry=json.loads(args.source_registry.read_text())
    for name,expected in registry['source_files'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
    assert hashlib.sha256(args.manifest.read_bytes()).hexdigest()==MANIFEST_SHA
    manifest=json.loads(args.manifest.read_text())
    assert len(manifest['train'])==960 and len(manifest['dev'])==60
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.npu.manual_seed(args.seed)
    ensure_sources(args.source_dir,offline=True)
    model=driver.build_model('original',args.seed)
    initial=torch.load(args.initialization,map_location='cpu',weights_only=False)
    assert initial['seed']==args.seed
    assert contract.tensor_hash(initial['model'])==initial['sha256']
    model.load_state_dict(initial['model'],strict=True)
    shared={n:v for n,v in model.state_dict().items() if not n.startswith('fnet.')}
    shared_hash=contract.tensor_hash(shared)
    model=install(model,args.variant,args.seed,args.source_dir)
    assert contract.tensor_hash({n:v for n,v in model.state_dict().items() if not n.startswith('fnet.')})==shared_hash
    initial_hash=contract.tensor_hash(model.state_dict())
    native_hash=initial['sha256']
    encoder_config=model.fnet.metadata()
    encoder_config['total_params']=sum(p.numel() for p in model.parameters())
    del initial,shared
    model=model.to(device)
    torch.manual_seed(args.seed);torch.npu.manual_seed(args.seed)
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-5,eps=1e-8,foreach=False,fused=False)
    scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=2e-4,total_steps=30000,
                    pct_start=.01,cycle_momentum=False,anneal_strategy='linear')
    dev_rows=manifest['dev'][:2] if args.preflight else manifest['dev']
    pipeline=None
    if args.input_mode=='legacy':
        train=driver.ManifestDataset(args.data_root,manifest['train'])
        dev=driver.ManifestDataset(args.data_root,dev_rows)
    else:
        pipeline=InputPipeline(args.input_mode,args.preload_gib,args.input_workers,args.prefetch_depth,args.seed)
        train,dev=pipeline.datasets(args.data_root,manifest['train'],dev_rows,args.preflight)
    subset=torch.utils.data.Subset(dev,list(range(group.rank,len(dev),group.world)))
    loader=torch.utils.data.DataLoader(subset,batch_size=2,num_workers=0,
                    generator=torch.Generator().manual_seed(args.seed+991))
    order=torch.randperm(960,generator=torch.Generator().manual_seed(args.seed)).tolist()
    evals=[0,2,200] if args.preflight else [0,1000]+list(range(2500,30001,2500))
    steps=200 if args.preflight else 30000
    config=dict(study='encoder-screen-npu-v1',backend='npu',model=args.model,
                arm='fnet',seed=args.seed,variant=args.variant,encoder=encoder_config,
                encoder_seed=args.seed+783,augmentation_seed=args.seed,
                updates=steps,schedule_total=30000,effective_batch=effective_batch,
                physical_batch=physical_batch,
                accumulation=effective_batch//(physical_batch*group.world),world_size=group.world,
                gradient_reduction='none_single_device_valid_count_weighted_accumulation',
                augmentation='hv',eval_updates=evals,prefetch_depth=2,prefetch_threads=1,
                manifest_sha256=MANIFEST_SHA,initial_state_sha256=initial_hash,
                train_iterations=16,dev_iterations=32,dtype='float32',
                optimizer=dict(name='AdamW',max_lr=2e-4,betas=[.9,.999],amsgrad=False,weight_decay=1e-5,eps=1e-8,
                               foreach=False,fused=False,gradient_clip=1.0),
                scheduler=dict(name='OneCycleLR',total_steps=30000,pct_start=.01,
                    cycle_momentum=False,anneal_strategy='linear',div_factor=25.,final_div_factor=10000.))
    config['execution']=dict(deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_deterministic=torch.backends.cudnn.deterministic,cudnn_benchmark=torch.backends.cudnn.benchmark,
        hccl_deterministic=os.environ['HCCL_DETERMINISTIC'])
    assert config['execution']==dict(deterministic_algorithms=False,cudnn_deterministic=False,cudnn_benchmark=False,hccl_deterministic='false')
    config['loss_implementation']='native_boolean_indexed_masked_mean'
    config['diagnostic_every']=100;config['checkpoint_every']=1000;config['status_every']=25
    config['initial_native_state_sha256']=native_hash
    config['shared_state_sha256']=shared_hash
    config['device_count']=1
    config['bad3_unit']='percent'
    if pipeline is not None:
        config['input_pipeline']={k:pipeline.metadata[k] for k in ('mode','budget_bytes','workers','prefetch_depth','spike_host_dtype')}
        config['prefetch_threads']=args.input_workers
        config['prefetch_depth']=args.prefetch_depth
    out=args.root
    if group.main:out.mkdir(parents=True,exist_ok=True)
    group.barrier()
    if pipeline is not None:
        pipeline_meta=group.gather(dict(rank=group.rank,**pipeline.metadata))
        if group.main:atomic_json(out/'input-pipeline.json',dict(ranks=pipeline_meta,**pipeline.metadata))
    lineage=[]
    (out/'checkpoints').mkdir(exist_ok=True)
    ckpath=out/'checkpoints'/'last.pt';completed=0;evaluated=set();stop=[False]
    prior_peak_allocated=0.;prior_peak_reserved=0.;started=time.monotonic()
    for signum in (signal.SIGTERM,signal.SIGUSR1):signal.signal(signum,lambda *_:stop.__setitem__(0,True))
    def status(phase):
        if group.main:atomic_json(out/'status.json',dict(phase=phase,update=completed,pid=os.getpid(),
                        updated_utc=utc(),stage_budget=args.stop_at,world_size=group.world))
    def checkpoint():
        checkpoint_started=time.monotonic()
        driver.reset_model(model);optimizer.zero_grad(set_to_none=True)
        all_rng=group.gather(rng_state(loader))
        hashes=group.gather(contract.tensor_hash(model.state_dict()))
        assert len(set(hashes))==1,'Rank model divergence'
        peak_allocated=group.maximum(torch.npu.max_memory_allocated()/2**20)
        peak_reserved=group.maximum(torch.npu.max_memory_reserved()/2**20)
        if group.main:
            state=dict(model=cpu_tree(model.state_dict()),optimizer=cpu_tree(optimizer.state_dict()),
                scheduler=scheduler.state_dict(),updates=completed,samples_seen=completed*effective_batch,
                evaluated=sorted(evaluated),config=config,order=order,rng_by_rank=all_rng,
                source_identity=registry['source_identity_sha256'],lineage=lineage,model_sha256=hashes[0],
                peak_allocated_mib=max(prior_peak_allocated,peak_allocated),
                peak_reserved_mib=max(prior_peak_reserved,peak_reserved))
            temp=ckpath.with_suffix('.tmp.pt');torch.save(state,temp);temp.replace(ckpath)
            atomic_json(out/'checkpoint.json',dict(update=completed,scheduler_last_epoch=scheduler.last_epoch,
                evaluated=sorted(evaluated),bytes=ckpath.stat().st_size,complete=True,model_sha256=hashes[0],
                rng_ranks=group.world,updated_utc=utc()))
        group.barrier()
        if group.main:append(out/'performance.jsonl',dict(kind='checkpoint',update=completed,seconds=time.monotonic()-checkpoint_started))
    if ckpath.exists():
        ck=torch.load(ckpath,map_location='cpu',weights_only=False)
        lineage=ck.get('lineage',[])
        assert ck['source_identity']==registry['source_identity_sha256']
        assert ck['config']==config
        assert ck['order']==order
        model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler'])
        completed=ck['updates'];evaluated=set(ck['evaluated'])
        assert completed<=args.stop_at
        for state in optimizer.state.values():
            for key,value in state.items():
                if torch.is_tensor(value) and key!='step':state[key]=value.to(device)
        assert contract.tensor_hash(model.state_dict())==ck['model_sha256']
        prior_peak_allocated=ck['peak_allocated_mib'];prior_peak_reserved=ck['peak_reserved_mib']
        restore_rng(ck['rng_by_rank'][group.rank],loader)
        assert scheduler.last_epoch==completed
        optimizer_steps=[int(s['step']) for s in optimizer.state.values() if 'step' in s]
        assert not optimizer_steps or min(optimizer_steps)==max(optimizer_steps)==completed
        def exact_restore(a,b,name):
            if torch.is_tensor(a):
                assert torch.is_tensor(b) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.cpu(),b.cpu()),name
            elif isinstance(a,np.ndarray):assert np.array_equal(a,b),name
            elif isinstance(a,dict):
                assert a.keys()==b.keys(),name
                for k in a:exact_restore(a[k],b[k],name+'/'+str(k))
            elif isinstance(a,(list,tuple)):
                assert type(a)==type(b) and len(a)==len(b),name
                for i,(x,y) in enumerate(zip(a,b)):exact_restore(x,y,name+'/'+str(i))
            else:assert a==b,(name,a,b)
        exact_restore(cpu_tree(optimizer.state_dict()),ck['optimizer'],'optimizer')
        exact_restore(scheduler.state_dict(),ck['scheduler'],'scheduler')
        exact_restore(rng_state(loader),ck['rng_by_rank'][group.rank],'rng')
        restored=group.gather(dict(rank=group.rank,model_hash_exact=True,optimizer_exact=True,scheduler_exact=True,rng_exact=True))
        if group.main:atomic_json(out/f'restore-audit-{completed}.json',dict(status='passed',update=completed,ranks=restored,order_exact=True,source_identity=registry['source_identity_sha256']))
        if group.main:
            reconcile(out/'train.jsonl',lambda row:row['update']<=completed)
            reconcile(out/'eval.jsonl',lambda row:row['update'] in evaluated)
            contract.audit_history(out/'train.jsonl',order,args.seed,completed)
            append(out/'recoveries.jsonl',dict(restored_update=completed,scheduler=scheduler.last_epoch,time_utc=utc()))
        del ck
    else:
        assert not (out/'metadata.json').exists(),'Existing incomplete run needs inspection'
        environments=group.gather(environment())
        if group.main:atomic_json(out/'metadata.json',dict(config=config,source_registry=registry,
            initial_state_sha256=initial_hash,initialization_file=str(args.initialization),order=order,
            rank_environments=environments,started_utc=utc(),job_id=os.environ.get('ENCODER_JOB_ID'),
            command=sys.argv))
        checkpoint()
    @torch.no_grad()
    def evaluate():
        status('validation')
        with preserve_rng():
            model.eval();frames=[]
            for left,right,target,valid,indices in loader:
                left,right,target,valid=[v.to(device) for v in (left,right,target,valid)]
                left,right=left.float(),right.float()
                pred=driver.forward(model,left,right,32)
                assert torch.isfinite(pred).all()
                for b,index in enumerate(indices.tolist()):
                    row=dev.rows[index];mask=valid[b];error=(pred[b,0]-target[b,0]).abs()[mask]
                    frames.append(dict(scene=row['scene'],frame=row['frame'],valid_pixels=int(mask.sum()),
                                       bad3_pixels=int((error>3).sum()),**driver.metric_values(error)))
                driver.reset_model(model)
                del left,right,target,valid,pred
            frames=[frame for rank_frames in group.gather(frames) for frame in rank_frames]
            frames.sort(key=lambda row:(row['scene'],row['frame']))
            assert len(frames)==len(dev_rows)
            assert len({(r['scene'],r['frame']) for r in frames})==len(dev_rows)
            scenes={s:float(np.mean([r['epe'] for r in frames if r['scene']==s]))
                    for s in sorted({r['scene'] for r in frames})}
            count=sum(r['valid_pixels'] for r in frames)
            result=dict(update=completed,samples_seen=completed*effective_batch,mean_scene_epe=float(np.mean(list(scenes.values()))),
                scene_epe=scenes,per_frame=frames,pooled_epe=sum(r['epe']*r['valid_pixels'] for r in frames)/count,
                **{k:sum(r[k]*r['valid_pixels'] for r in frames)/count for k in ('bad1','bad3','bad5')})
        if group.main:
            append(out/'eval.jsonl',result)
            print('DEV',completed,result['mean_scene_epe'],flush=True)
        evaluated.add(completed);checkpoint()
    if pipeline is None:
        prefetch=Prefetch(train,order,completed,min(steps,args.stop_at),args.seed,group.rank,group.world)
    else:
        prefetch=RankPreparedPrefetch(train,order,completed,min(steps,args.stop_at),args.seed,group.rank,group.world,args.input_workers,args.prefetch_depth,global_batch=effective_batch,micro_batch=physical_batch)
    try:
        if completed in evals and completed not in evaluated:evaluate()
        torch.npu.reset_peak_memory_stats()
        while completed<min(steps,args.stop_at):
            update_started=time.monotonic()
            if group.maximum(stop[0] or (args.root/'pause.request').exists()):
                checkpoint();status('paused');return 75
            wait=time.monotonic()
            with span('input_wait'):batch=prefetch.get(completed)
            consumer_wait=time.monotonic()-wait
            torch.npu.synchronize()
            tick=time.monotonic()
            if args.preflight or completed%25==0:status('training')
            model.train();model.freeze_bn();optimizer.zero_grad(set_to_none=True)
            diagnostic = args.preflight and (completed in (0,1,199))
            for m in model.fnet.modules():
                if isinstance(m,MultiStepLIFNode):m.collect_activity=diagnostic
            items=batch.get('items')
            local_count=batch['valid_count'] if pipeline is not None else sum(int(item[3].sum()) for item in items)
            count=int(local_count);assert count>0
            total=0.
            activity_values=[]
            micro_size=physical_batch
            for offset in range(0,effective_batch//group.world,micro_size):
                with span('host_to_device'):
                    if pipeline is None:
                        micro=items[offset:offset+micro_size]
                        left,right,target,valid=[torch.stack([r[j] for r in micro]).to(device) for j in range(4)]
                        micro_count=int(valid.sum())
                    else:
                        micro=batch['microbatches'][offset//micro_size]
                        left,right,target,valid=[v.to(device) for v in micro['tensors']]
                        left,right=left.float(),right.float()
                        micro_count=micro['valid_count']
                with span('model_forward'):preds,spikes,volts=driver.forward(model,left,right,16,True)
                with span('disparity_loss'):
                    loss=driver.disparity_loss(preds,target,valid)*int(micro_count)/count
                assert torch.isfinite(loss)
                with span('backward'):loss.backward()
                total+=float(loss.detach())
                if diagnostic:
                    activity_values.extend(float(m.activity) for m in model.fnet.modules() if isinstance(m,MultiStepLIFNode) and m.activity is not None)
                driver.reset_model(model)
                del left,right,target,valid,preds,spikes,volts,loss,micro
            detailed=args.preflight or completed==0 or (completed+1)%100==0
            gradients={}
            if detailed:
                for label,submodule in [('encoder',model.fnet.backbone),('readout',model.fnet.readout),('solver',model.update_block)]:
                    norms=[p.grad.detach().square().sum() for p in submodule.parameters() if p.grad is not None]
                    gradients[label]=float(torch.stack(norms).sum().sqrt()) if norms else 0.
                assert all(np.isfinite(v) and v>0 for v in gradients.values()),gradients
            norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
            total_value=total
            assert np.isfinite(norm) and norm>0
            lr=optimizer.param_groups[0]['lr']
            with span('optimizer_step'):optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True)
            with span('end_step_sync'):torch.npu.synchronize()
            completed+=1
            seconds=group.maximum(time.monotonic()-tick)
            allocated=group.maximum(torch.npu.max_memory_allocated()/2**20)
            resident=group.maximum(torch.npu.memory_allocated()/2**20)
            row=dict(update=completed,samples_seen=completed*effective_batch,indices=batch['indices'],augmentation_hv=batch['flags'],
                loss=total_value,gradient_norm=norm,module_gradients=gradients,final_qk_gradient=None,
                lr_used=lr,lr_next=scheduler.get_last_lr()[0],spike_activity=float(np.mean(activity_values)) if activity_values else None,seconds=seconds,
                consumer_wait_seconds=consumer_wait,phase_wait_seconds=None,phase_wait_status='not_applicable_disjoint_npu_allocations',
                update_wall_seconds=time.monotonic()-update_started,decode_seconds=batch['decode_seconds'],
                input_prepare_seconds=batch.get('input_prepare_seconds'),host_batch_bytes=batch.get('host_batch_bytes'),
                peak_allocated_mib=allocated,resident_allocated_mib=resident,world_size=1,physical_batch=physical_batch,accumulation=effective_batch//physical_batch)
            contract.validate_row(row,order,args.seed)
            if group.main:append(out/'train.jsonl',row)
            del items,batch,total
            if completed%1000==0 and completed not in evals:checkpoint()
            if completed in evals and completed not in evaluated:evaluate()
            if completed in {10000,20000,30000,args.stop_at} and completed in evaluated:
                if group.main:
                    temp=ckpath.parent/f'step-{completed}.tmp.pt';shutil.copyfile(ckpath,temp);temp.replace(ckpath.parent/f'step-{completed}.pt')
            if args.pause_at==completed:
                checkpoint();status('preflight_resume_probe');return 75
            if completed%25==0 and group.main:print('UPDATE',completed,'LOSS',row['loss'],flush=True)
        checkpoint()
        assert completed in evaluated
        assert all(torch.isfinite(v).all() for v in model.state_dict().values())
        assert all(torch.isfinite(v).all() for state in optimizer.state.values() for v in state.values() if torch.is_tensor(v))
        timing = None
        if not args.preflight:
            with torch.no_grad(), preserve_rng():
                model.eval()
                batch=next(iter(loader))
                left,right=[v[:1].to(device).float() for v in batch[:2]]
                for _ in range(5):
                    driver.forward(model,left,right,32);driver.reset_model(model)
                torch.npu.synchronize();begin=time.monotonic()
                for _ in range(20):
                    driver.forward(model,left,right,32);driver.reset_model(model)
                torch.npu.synchronize();timing=(time.monotonic()-begin)*1000/20
                del left,right,batch
        status('preflight_completed' if args.preflight else 'completed')
        peak_allocated=group.maximum(torch.npu.max_memory_allocated()/2**20)
        peak_reserved=group.maximum(torch.npu.max_memory_reserved()/2**20)
        if group.main:
            atomic_json(out/'result.json',dict(status='PREFLIGHT_COMPLETED' if args.preflight else 'COMPLETED',
                updates=completed,samples_seen=completed*effective_batch,scheduler_last_epoch=scheduler.last_epoch,
                evaluated=sorted(evaluated),finite_model_optimizer=True,process_seconds=time.monotonic()-started,
                endpoint_kind='preflight' if args.preflight else 'full_schedule',
                inference_ms_per_pair=timing,encoder=encoder_config,
                peak_allocated_mib=max(prior_peak_allocated,peak_allocated),
                peak_reserved_mib=max(prior_peak_reserved,peak_reserved),
                inference_timing=dict(batch=1,warmup=5,repeats=20,refinements=32) if timing is not None else None,
                world_size=group.world,source_identity=registry['source_identity_sha256']))
        return 0
    except Exception as error:
        atomic_json(out/f'failure-rank-{group.rank}.json',dict(error=repr(error),update_attempt=completed+1,
                last_checkpoint_update=json.loads((out/'checkpoint.json').read_text())['update'],time_utc=utc()))
        raise
    finally:
        prefetch.close()


if __name__=='__main__':raise SystemExit(main())
