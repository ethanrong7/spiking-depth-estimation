"""Operational continuation handover: unchanged science, current clocks, bounded CPU prefetch."""
import argparse,copy,importlib.util,json,os,shutil,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
from prefetch import Prefetch,decode_batch
STARTED=time.monotonic();STARTED_UTC=datetime.now(timezone.utc).isoformat()


def main():
    p=argparse.ArgumentParser()
    for key in ('scientific-overlay','operational-overlay','handover','run-id','parent-overlay','base-source','parent-root','data-root','output'):
        p.add_argument('--'+key,required=True,type=str if key=='run-id' else Path)
    a=p.parse_args();source=json.loads((a.scientific_overlay/'source-registry.json').read_text());operational_source=json.loads((a.operational_overlay/'source-registry.json').read_text())
    sys.path.insert(0,str(a.scientific_overlay/'scripts'))
    from continuation import read,sha,load_parent,build_scheduler,optimizer_steps,current_rng,restore_rng,rng_fingerprint,GRID
    for capsule,registry in ((a.scientific_overlay,source),(a.operational_overlay,operational_source)):
        for name,digest in registry['source_files'].items():
            if sha(capsule/name)!=digest:raise RuntimeError('capsule mismatch: '+name)
    release=read(a.handover)
    if not release['root_reviewed'] or release['operational_source_identity_sha256']!=operational_source['source_identity_sha256']:raise RuntimeError('handover release mismatch')
    route=release['runs'][a.run_id];start=route['checkpoint_updates'];checkpoint_path=Path(route['checkpoint_path'])
    if sha(checkpoint_path)!=route['checkpoint_sha256']:raise RuntimeError('durable handover checkpoint changed')
    config=read(a.scientific_overlay/'configs'/f'{a.run_id}.json');registration=read(a.scientific_overlay/'configs/registration.json')
    recipe,parent,driver,mechanism,parent_source,base_source=load_parent(a.parent_overlay,a.base_source)
    import torch,numpy as np
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available() and torch.cuda.device_count()==1 and '5090' in torch.cuda.get_device_name(0)
    torch.set_num_threads(8);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    ck=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    if ck['updates']!=start or ck['samples_seen']!=start*8 or not 10000<=start<50000:raise RuntimeError('handover counters')
    initial_parent=route['checkpoint_kind']=='parent10k'
    if initial_parent:
        expected=read(a.parent_overlay/'configs'/registration['parents'][a.run_id]['parent_config_file'])
        if start!=10000 or ck['config']!=expected or ck['source']!=parent_source:raise RuntimeError('unstarted-model parent identity')
    elif ck['config']!=config or ck['source']!=source:raise RuntimeError('current continuation identity')
    model=recipe.build_model(driver,mechanism,config['model'],1234);model.load_state_dict(ck['model'],strict=True);loaded_hash=parent.tensor_hash(model.state_dict());model=model.cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=.0002,weight_decay=1e-5,eps=1e-8);opt.load_state_dict(ck['optimizer'])
    scheduler=build_scheduler(opt,config)
    if not initial_parent:
        # OneCycle construction resets LR: restore BOTH groups/moments and current schedule.
        opt.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler'])
        if opt.state_dict()['param_groups']!=ck['optimizer']['param_groups']:raise RuntimeError('current optimizer LR/groups not restored')
        if scheduler.state_dict()!=ck['scheduler']:raise RuntimeError('current scheduler not restored')
        if opt.param_groups[0]['lr']!=ck['optimizer']['param_groups'][0]['lr'] or opt.param_groups[0]['lr']!=scheduler.get_last_lr()[0]:raise RuntimeError('current LR was reset')
    if optimizer_steps(opt)!=(start,start) or scheduler.last_epoch!=start-10000:raise RuntimeError('restored global/segment clocks')
    def opt_hash():return parent.tensor_hash({f'{identifier}.{key}':value for identifier,state in opt.state_dict()['state'].items() for key,value in state.items() if torch.is_tensor(value)})
    expected_opt=parent.tensor_hash({f'{identifier}.{key}':value for identifier,state in ck['optimizer']['state'].items() for key,value in state.items() if torch.is_tensor(value)})
    if opt_hash()!=expected_opt:raise RuntimeError('restored moments differ')
    order=ck['order']
    if order!=torch.randperm(960,generator=torch.Generator().manual_seed(1234)).tolist():raise RuntimeError('restored order differs')
    saved_rng=ck['rng'];restore_rng(saved_rng);rng_hash=rng_fingerprint(current_rng());old_scheduler=copy.deepcopy(scheduler.state_dict())
    manifest_path=a.base_source/'projects/snn-depth-estimation/experiments/spiking-transformer-stereo/configs/manifest.json'
    if sha(manifest_path)!=registration['manifest_sha256']:raise RuntimeError('manifest mismatch')
    manifest=read(manifest_path);train=driver.ManifestDataset(a.data_root,manifest['train']);dev=driver.ManifestDataset(a.data_root,manifest['dev'])
    loader=torch.utils.data.DataLoader(dev,batch_size=2,num_workers=0,generator=torch.Generator().manual_seed(1234+991))
    a.output.mkdir(parents=True,exist_ok=False)
    sync=decode_batch(train,recipe,order,1234,start)
    with Prefetch(train,recipe,order,1234,start,start+1) as test:prefetched=test.get(start)
    if sync.indices!=prefetched.indices or sync.flags!=prefetched.flags:raise RuntimeError('prefetch index/augmentation mismatch')
    for synchronous,prepared in zip(sync.items,prefetched.items):
        for left,right in zip(synchronous,prepared):
            if torch.is_tensor(left):
                if left.dtype!=right.dtype or left.shape!=right.shape or not torch.equal(left,right):raise RuntimeError('prefetched tensor differs')
            elif left!=right:raise RuntimeError('prefetched sample identity differs')
    def gradient_pass(batch):
        model.train();model.freeze_bn();opt.zero_grad(set_to_none=True);count=sum(int(item[3].sum()) for item in batch.items);loss_value=0.
        for offset in (0,4):
            micro=batch.items[offset:offset+4];left,right,target,valid=[torch.stack([item[j] for item in micro]).cuda() for j in range(4)]
            preds,spikes,volts=driver.forward(model,left,right,16,True);loss=driver.disparity_loss(preds,target,valid)*(int(valid.sum())/count);loss.backward();loss_value+=float(loss.detach());driver.reset_model(model)
            del micro,left,right,target,valid,preds,spikes,volts,loss
        gradients={name:p.grad.detach().cpu().clone() for name,p in model.named_parameters() if p.grad is not None}
        return loss_value,gradients
    gradient_pass(sync)  # Numerical kernel warmup only; no optimizer step.
    reference_loss,reference_grad=gradient_pass(sync);candidate_loss,candidate_grad=gradient_pass(prefetched)
    if reference_loss!=candidate_loss or set(reference_grad)!=set(candidate_grad) or not all(torch.equal(reference_grad[name],candidate_grad[name]) for name in reference_grad):raise RuntimeError('synchronous/prefetch loss or gradient parity failed')
    opt.zero_grad(set_to_none=True);driver.reset_model(model)
    if parent.tensor_hash(model.state_dict())!=loaded_hash or opt_hash()!=expected_opt or scheduler.state_dict()!=old_scheduler:raise RuntimeError('engineering validation changed learned/optimizer/scheduler state')
    if rng_fingerprint(current_rng())!=rng_hash:raise RuntimeError('loader/engineering validation advanced RNG')
    restore_rng(saved_rng)
    validation=dict(status='PASS',checkpoint_updates=start,optimizer_updates=0,engineering_gradient_passes=3,decoded_tensors_exact=True,loss_exact=True,all_gradients_bitwise_equal=True,model_optimizer_scheduler_unchanged=True,rng_unchanged_and_restored=True,restored_model_sha256=loaded_hash,optimizer_tensor_sha256=expected_opt,rng_sha256=rng_hash,current_lr=opt.param_groups[0]['lr'],scheduler_local_epoch=scheduler.last_epoch,global_AdamW_steps=list(optimizer_steps(opt)))
    parent.write(a.output/'loading-validation.json',validation);del sync,prefetched,reference_grad,candidate_grad
    lineage=registration['parents'][a.run_id];parent_schedule=ck['scheduler'] if initial_parent else ck['parent_scheduler'];parent_initialization=ck['initialization']
    prefix_root=Path(route['prefix_root']) if route.get('prefix_root') else None
    prefix_rows=[json.loads(line) for line in (prefix_root/'train.jsonl').read_text().splitlines() if json.loads(line)['update']<=start] if prefix_root else []
    if [row['update'] for row in prefix_rows]!=list(range(10001,start+1)):raise RuntimeError('canonical durable prefix gap/duplication')
    if prefix_root:
        history=[row for row in [json.loads(line) for line in (prefix_root/'evaluations.jsonl').read_text().splitlines()] if row['update']<=start]
        best=ck['best'];best_update=ck['best_update']
    else:
        baseline=read(a.parent_root/lineage['parent_run_id']/'result.json')['final_primary'];history=[dict(baseline,segment_update=0)];best=baseline['mean_scene_epe'];best_update=10000
    if min(history,key=lambda row:row['mean_scene_epe'])['update']!=best_update:raise RuntimeError('carried historical best mismatch')
    best_checkpoint=Path(route['best_checkpoint_path'])
    if sha(best_checkpoint)!=route['best_checkpoint_sha256']:raise RuntimeError('historical best checkpoint changed')
    shutil.copy2(best_checkpoint,a.output/'best.pt');shutil.copy2(checkpoint_path,a.output/'last.pt')
    def timing():return dict(elapsed_seconds=time.monotonic()-STARTED,timestamp_utc=datetime.now(timezone.utc).isoformat())
    metadata=dict(config=config,scientific_source=source,operational_source=operational_source,source=source,parent_source=parent_source,base_source=base_source,lineage=lineage,handover=route,handover_release_sha256=sha(a.handover),loading_validation=validation,slurm={k:v for k,v in os.environ.items() if k.startswith('SLURM_')},gpu=torch.cuda.get_device_name(0),torch=torch.__version__,started_utc=STARTED_UTC,initialization=parent_initialization,manifest_sha256=registration['manifest_sha256'],loader=dict(producer_threads=1,queue_depth=2,pinning=False,cuda_in_worker=False),timing_definition='decode_seconds=producer work overlapping other steps; loader_wait_seconds=foreground stall; neither producer work nor wait is added again to seconds')
    parent.write(a.output/'metadata.json',metadata);parent.write(a.output/'sample-order.json',dict(order=order,effective_batch=8,start_global_index=start));parent.write(a.output/'history.json',history)
    def checkpoint(step):
        if optimizer_steps(opt)!=(step,step) or scheduler.last_epoch!=step-10000:raise RuntimeError('checkpoint clocks')
        state=dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),updates=step,global_updates=step,segment_updates=step-10000,samples_seen=step*8,additional_samples_seen=(step-10000)*8,config=config,order=order,best=best,best_update=best_update,source=source,operational_source=operational_source,parent_source=parent_source,parent_checkpoint_sha256=lineage['checkpoint_sha256'],handover_checkpoint_sha256=route['checkpoint_sha256'],parent_scheduler=parent_schedule,manifest_sha256=registration['manifest_sha256'],initialization=parent_initialization,rng=current_rng())
        torch.save(state,a.output/'last.tmp.pt');(a.output/'last.tmp.pt').replace(a.output/'last.pt')
    with Prefetch(train,recipe,order,1234,start,50000) as producer:
      for step in range(start,50000):
          model.train();model.freeze_bn();opt.zero_grad(set_to_none=True)
          begun=time.monotonic();prepared=producer.get(step);loader_wait_seconds=time.monotonic()-begun
          indices=prepared.indices;flags=prepared.flags;items=prepared.items;decode_seconds=prepared.loader_work_seconds
          valid_count=sum(int(item[3].sum()) for item in items)
          if not valid_count:raise RuntimeError('no valid effective8 pixels')
          disparity=0.;train_error=0.;diagnostics=[]
          for offset in range(0,8,4):
              micro=items[offset:offset+4]
              left,right,target,valid=[torch.stack([item[j] for item in micro]).cuda() for j in range(4)]
              preds,spikes,volts=driver.forward(model,left,right,16,True)
              loss=driver.disparity_loss(preds,target,valid)*(int(valid.sum())/valid_count)
              if not torch.isfinite(loss):raise FloatingPointError('nonfinite disparity loss')
              loss.backward();disparity+=float(loss.detach());train_error+=float((preds[-1]-target).abs()[:,0][valid].sum())/valid_count
              if hasattr(model.update_block,'diagnostics'):diagnostics.append(model.update_block.diagnostics())
              driver.reset_model(model);del micro,left,right,target,valid,preds,spikes,volts,loss
          grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
          if not np.isfinite(grad) or grad<=0:raise FloatingPointError('invalid gradient')
          lr_used=opt.param_groups[0]['lr'];opt.step();scheduler.step();torch.cuda.synchronize()
          if scheduler.last_epoch!=step+1-10000:raise RuntimeError('segment scheduler count')
          row=dict(update=step+1,segment_update=step+1-10000,samples_seen=(step+1)*8,additional_samples_seen=(step+1-10000)*8,
              indices=indices,augmentation_hv=flags,loss=disparity,disparity_loss=disparity,neuron_loss=0.,train_epe=train_error,
              valid_pixels=valid_count,gradient_norm=grad,lr_used=lr_used,lr_next=scheduler.get_last_lr()[0],lr=scheduler.get_last_lr()[0],
              seconds=time.monotonic()-begun,decode_seconds=decode_seconds,loader_wait_seconds=loader_wait_seconds,loader_work_seconds=decode_seconds,peak_memory_mib=torch.cuda.max_memory_allocated()/2**20,diagnostics=diagnostics,**timing())
          parent.append(a.output/'train.jsonl',row);parent.append(a.output/'loss-components.jsonl',{key:row[key] for key in ('update','segment_update','disparity_loss','neuron_loss','loss','valid_pixels')});del items,prepared
          if step+1 in GRID:
              evaluation_started=time.monotonic();result=dict(update=step+1,segment_update=step+1-10000,**driver.evaluate(model,loader,32),**timing());result['evaluation_seconds']=time.monotonic()-evaluation_started
              history.append(result);parent.append(a.output/'evaluations.jsonl',result)
              improved=result['mean_scene_epe']<best
              if improved:best=result['mean_scene_epe'];best_update=step+1
              checkpoint(step+1)
              if improved:shutil.copy2(a.output/'last.pt',a.output/'best.tmp.pt');(a.output/'best.tmp.pt').replace(a.output/'best.pt')
              parent.write(a.output/'history.json',history);print('EVAL',step+1,result['mean_scene_epe'],flush=True)
          elif (step+1)%1000==0:checkpoint(step+1)
    if optimizer_steps(opt)!=(50000,50000) or scheduler.last_epoch!=40000:raise RuntimeError('final global/segment counters')
    execution_rows=[json.loads(line) for line in (a.output/'train.jsonl').read_text().splitlines()]
    rows=prefix_rows+execution_rows
    if [row['update'] for row in rows]!=list(range(10001,50001)) or rows[-1]['samples_seen']!=400000:raise RuntimeError('continuation count continuity')
    finite_model=all(bool(torch.isfinite(value).all()) for value in model.state_dict().values())
    finite_optimizer=all(bool(torch.isfinite(value).all()) for state in opt.state.values() for value in state.values() if torch.is_tensor(value))
    if not finite_model or not finite_optimizer:raise FloatingPointError('nonfinite final checkpoint tensors')
    parent.write(a.output/'result.json',dict(status='COMPLETED',updates_completed=50000,additional_updates_completed=40000,samples_seen=400000,additional_samples_seen=320000,
        physical_batch=4,effective_batch=8,final_primary=history[-1],best_dev_secondary=min(history,key=lambda row:row['mean_scene_epe']),best_update=best_update,
        optimizer_step_min_max=[50000,50000],scheduler_segment_last_epoch=40000,scheduler_total_steps=40100,
        contiguous_updates=True,all_training_records_finite=True,all_checkpoint_tensors_finite=True,
        training_step_seconds=sum(row['seconds'] for row in rows),train_decode_seconds=sum(row['decode_seconds'] for row in rows),
        dev_seconds=sum(row['evaluation_seconds'] for row in history),parent_checkpoint_sha256=lineage['checkpoint_sha256'],operational_source_identity_sha256=operational_source['source_identity_sha256'],execution_start_updates=start,loader_wait_seconds=sum(row.get('loader_wait_seconds',0.) for row in execution_rows),**timing()))

if __name__=='__main__':main()
