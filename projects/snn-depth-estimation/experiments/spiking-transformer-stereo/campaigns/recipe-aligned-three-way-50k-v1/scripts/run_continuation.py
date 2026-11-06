"""Resume the three audited 10k endpoints to exactly50k, with a distinct LR segment."""
import argparse
from datetime import datetime,timezone
import getpass
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from continuation import GRID,SCHEDULE,build_scheduler,compare_baseline,current_rng,load_parent,optimizer_steps,read,restore_rng,rng_fingerprint,sha,validate_config
STARTED=time.monotonic()
STARTED_UTC=datetime.now(timezone.utc).isoformat()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('preflight','train'),required=True)
    for key in ('config','registration','source-registry','parent-overlay','base-source','parent-root','data-root','output'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();config=read(a.config);validate_config(config);registration=read(a.registration)
    source=read(a.source_registry);overlay=a.source_registry.parent
    for name,expected in source['source_files'].items():
        if sha(overlay/name)!=expected:raise RuntimeError('continuation source mismatch: '+name)
    if sha(a.config)!=registration['config_sha256'][config['run_id']]:raise RuntimeError('unregistered config')
    if sha(a.registration)!=source['registration_sha256']:raise RuntimeError('unregistered campaign')
    recipe,parent,driver,mechanism,parent_source,base_source=load_parent(a.parent_overlay,a.base_source)
    if source['parent_source_identity_sha256']!=parent_source['source_identity_sha256']:raise RuntimeError('parent source binding')
    import random,numpy as np,torch
    from torch.utils.data import DataLoader
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available() and torch.cuda.device_count()==1
    assert '5090' in torch.cuda.get_device_name(0)
    torch.set_num_threads(8);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    lineage=registration['parents'][config['run_id']];parent_root=a.parent_root/lineage['parent_run_id']
    checkpoint_path=parent_root/'last.pt'
    if sha(checkpoint_path)!=lineage['checkpoint_sha256']:raise RuntimeError('parent checkpoint changed')
    if sha(parent_root/'result.json')!=lineage['result_sha256']:raise RuntimeError('parent result changed')
    parent_config_path=a.parent_overlay/'configs'/lineage['parent_config_file']
    if sha(parent_config_path)!=lineage['parent_config_sha256']:raise RuntimeError('parent configuration binding')
    expected_parent=read(parent_config_path)
    if expected_parent['model']!=config['model']:raise RuntimeError('parent arm identity differs')
    ck=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    if ck['config']!=expected_parent or ck['updates']!=10000 or ck['samples_seen']!=80000:raise RuntimeError('parent checkpoint contract')
    if ck['source']!=parent_source or ck['manifest_sha256']!=registration['manifest_sha256']:raise RuntimeError('parent source/manifest')
    if ck['initialization']!=lineage['initialization']:raise RuntimeError('parent initialization lineage')
    seed=1234;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=recipe.build_model(driver,mechanism,config['model'],seed);model.load_state_dict(ck['model'],strict=True)
    loaded_hash=parent.tensor_hash(model.state_dict())
    if loaded_hash!=parent.tensor_hash(ck['model']):raise RuntimeError('model restore differs')
    model=model.cuda();opt=torch.optim.AdamW(model.parameters(),lr=.0002,weight_decay=1e-5,eps=1e-8)
    opt.load_state_dict(ck['optimizer'])
    if optimizer_steps(opt)!=(10000,10000):raise RuntimeError('parent AdamW clock')
    # Match every restored moment tensor before and after the numerical resume check.
    def verify_moments():
        actual=opt.state_dict()['state'];expected=ck['optimizer']['state']
        if set(actual)!=set(expected):raise RuntimeError('restored optimizer state membership')
        for identifier,original in expected.items():
            for key,value in original.items():
                if torch.is_tensor(value) and not torch.equal(value.cpu(),actual[identifier][key].cpu()):raise RuntimeError('restored optimizer tensor differs')
    verify_moments()
    order=ck['order'];expected_order=torch.randperm(960,generator=torch.Generator().manual_seed(seed)).tolist()
    if order!=expected_order:raise RuntimeError('parent cyclic order')
    scheduler=build_scheduler(opt,config)
    if scheduler.last_epoch!=0 or optimizer_steps(opt)!=(10000,10000):raise RuntimeError('new segment/global clocks')
    parent_schedule=ck['scheduler'];saved_rng=ck['rng'];parent_initialization=ck['initialization']
    parent_final=read(parent_root/'result.json')['final_primary']
    manifest_path=a.base_source/'projects/snn-depth-estimation/experiments/spiking-transformer-stereo/configs/manifest.json'
    if sha(manifest_path)!=registration['manifest_sha256']:raise RuntimeError('manifest changed')
    manifest=read(manifest_path);assert len(manifest['train'])==960 and len(manifest['dev'])==60
    train=driver.ManifestDataset(a.data_root,manifest['train']);dev=driver.ManifestDataset(a.data_root,manifest['dev'])
    loader=DataLoader(dev,batch_size=2,num_workers=0,generator=torch.Generator().manual_seed(seed+991))
    a.output.mkdir(parents=True,exist_ok=False)
    def timing():return dict(elapsed_seconds=time.monotonic()-STARTED,timestamp_utc=datetime.now(timezone.utc).isoformat())
    warm=next(iter(loader));model.eval()
    with torch.no_grad():driver.forward(model,warm[0].cuda(),warm[1].cuda(),32)
    driver.reset_model(model);del warm
    if parent.tensor_hash(model.state_dict())!=loaded_hash:raise RuntimeError('warmup changed model/buffers')
    restore_rng(saved_rng)
    evaluation_started=time.monotonic();baseline=driver.evaluate(model,loader,32)
    baseline['evaluation_seconds']=time.monotonic()-evaluation_started
    reproduction=compare_baseline(baseline,parent_final)
    if parent.tensor_hash(model.state_dict())!=loaded_hash:raise RuntimeError('resume evaluation changed model/buffers')
    verify_moments()
    # Construction, numerical warmup and Dev reproduction never consume the resumed stream.
    restore_rng(saved_rng)
    if rng_fingerprint(current_rng())!=rng_fingerprint(saved_rng):raise RuntimeError('parent RNG restore differs')
    if optimizer_steps(opt)!=(10000,10000) or scheduler.last_epoch!=0:raise RuntimeError('zero-update resume changed clocks')
    baseline=dict(update=10000,segment_update=0,**baseline,**timing())
    check=dict(status='PASS',model=config['model'],global_updates=10000,segment_updates=0,scientific_optimizer_updates=0,
        parent_checkpoint_sha256=lineage['checkpoint_sha256'],restored_model_state_sha256=loaded_hash,
        parent_optimizer_tensors_exactly_restored=True,optimizer_step_min_max=[10000,10000],scheduler_local_epoch=0,
        state_exactly_unchanged=True,parent_rng_restored=True,parent_rng_sha256=rng_fingerprint(saved_rng),
        baseline_reproduction=reproduction,baseline=baseline,gpu=torch.cuda.get_device_name(0),job_id=os.environ['SLURM_JOB_ID'],**timing())
    parent.write(a.output/'resume-check.json',check)
    if a.mode=='preflight':print(json.dumps(check,allow_nan=False),flush=True);return
    if a.output.name!=config['run_id']:raise RuntimeError('wrong output run identity')
    metadata=dict(config=config,source=source,parent_source=parent_source,base_source=base_source,registration=registration,
        lineage=lineage,command=sys.argv,hostname=socket.gethostname(),user=getpass.getuser(),gpu=torch.cuda.get_device_name(0),
        torch=torch.__version__,cuda=torch.version.cuda,python=sys.version,slurm={k:v for k,v in os.environ.items() if k.startswith('SLURM_')},
        pip_freeze=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True),
        started_utc=STARTED_UTC,parent_scheduler=parent_schedule,initialization=parent_initialization,
        manifest_sha256=registration['manifest_sha256'],resume_check=check,
        clock_definition='AdamW global10000..50000; new OneCycle local0..40000, total_steps40100; LR restart, not uninterrupted50k',
        timing_definition='decode is included in training_step_seconds, not added twice; process clock includes resume warmup/evaluation/checkpoint I/O')
    parent.write(a.output/'metadata.json',metadata);parent.write(a.output/'sample-order.json',dict(order=order,effective_batch=8,start_global_index=10000))
    parent.write(a.output/'initial.json',baseline);parent.append(a.output/'evaluations.jsonl',baseline)
    history=[baseline];best=baseline['mean_scene_epe'];best_update=10000
    def checkpoint(step):
        if optimizer_steps(opt)!=(step,step) or scheduler.last_epoch!=step-10000:raise RuntimeError('atomic checkpoint global/segment clocks')
        state=dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),updates=step,
            global_updates=step,segment_updates=step-10000,samples_seen=step*8,additional_samples_seen=(step-10000)*8,
            config=config,order=order,best=best,best_update=best_update,source=source,parent_source=parent_source,
            parent_checkpoint_sha256=lineage['checkpoint_sha256'],parent_scheduler=parent_schedule,
            manifest_sha256=registration['manifest_sha256'],initialization=parent_initialization,rng=current_rng())
        torch.save(state,a.output/'last.tmp.pt');(a.output/'last.tmp.pt').replace(a.output/'last.pt')
    import shutil
    checkpoint(10000);shutil.copy2(a.output/'last.pt',a.output/'best.pt')
    for step in range(10000,50000):
        model.train();model.freeze_bn();opt.zero_grad(set_to_none=True)
        indices=recipe.batch_indices(order,step);flags=[recipe.paired_flags(seed,step*8+pos) for pos in range(8)]
        begun=time.monotonic();items=[recipe.augment(train[index],*flag) for index,flag in zip(indices,flags)]
        decode_seconds=time.monotonic()-begun;valid_count=sum(int(item[3].sum()) for item in items)
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
            seconds=time.monotonic()-begun,decode_seconds=decode_seconds,peak_memory_mib=torch.cuda.max_memory_allocated()/2**20,diagnostics=diagnostics,**timing())
        parent.append(a.output/'train.jsonl',row);parent.append(a.output/'loss-components.jsonl',{key:row[key] for key in ('update','segment_update','disparity_loss','neuron_loss','loss','valid_pixels')});del items
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
    rows=[json.loads(line) for line in (a.output/'train.jsonl').read_text().splitlines()]
    if [row['update'] for row in rows]!=list(range(10001,50001)) or rows[-1]['samples_seen']!=400000:raise RuntimeError('continuation count continuity')
    finite_model=all(bool(torch.isfinite(value).all()) for value in model.state_dict().values())
    finite_optimizer=all(bool(torch.isfinite(value).all()) for state in opt.state.values() for value in state.values() if torch.is_tensor(value))
    if not finite_model or not finite_optimizer:raise FloatingPointError('nonfinite final checkpoint tensors')
    parent.write(a.output/'result.json',dict(status='COMPLETED',updates_completed=50000,additional_updates_completed=40000,samples_seen=400000,additional_samples_seen=320000,
        physical_batch=4,effective_batch=8,final_primary=history[-1],best_dev_secondary=min(history,key=lambda row:row['mean_scene_epe']),best_update=best_update,
        optimizer_step_min_max=[50000,50000],scheduler_segment_last_epoch=40000,scheduler_total_steps=40100,
        contiguous_updates=True,all_training_records_finite=True,all_checkpoint_tensors_finite=True,
        training_step_seconds=sum(row['seconds'] for row in rows),train_decode_seconds=sum(row['decode_seconds'] for row in rows),
        dev_seconds=sum(row['evaluation_seconds'] for row in history),parent_checkpoint_sha256=lineage['checkpoint_sha256'],**timing()))


if __name__=='__main__':main()
