"""Bounded native Synthetic actual S2M2 spiking retrofit; Slurm GPU only, no Test."""
import argparse,hashlib,importlib.util,json,os,random,sys,time,subprocess
from datetime import datetime,timezone
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[2]
MANIFEST_SHA='bad4369b6fd8b0cdf871a20b1727c2d9b59733a16bd33f75904227bdfef4510a'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def append(p,v):
    with p.open('a') as f:f.write(json.dumps(v,allow_nan=False)+'\n')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('data-root','manifest','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--preflight',action='store_true');a=parser.parse_args();started=time.monotonic()
    assert sha(a.manifest)==MANIFEST_SHA
    manifest=json.loads(a.manifest.read_text());assert len(manifest['train'])==960 and len(manifest['dev'])==60
    assert len({r['scene'] for r in manifest['dev']})==30
    assert not {r['scene'] for r in manifest['train']}&{r['scene'] for r in manifest['dev']}
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available() and torch.cuda.device_count()==1
    assert '5090' in torch.cuda.get_device_name(0) or '4090' in torch.cuda.get_device_name(0)
    sys.path.insert(0,str(PROJECT));driver_path=PROJECT/'experiments/spiking-transformer-stereo/scripts/run_experiment.py'
    sys.path.insert(0,str(driver_path.parent));spec=importlib.util.spec_from_file_location('native_pilot_driver',driver_path);driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
    from core.s2m2_spiking import SpikingS2M2,s2m2_loss as direct_loss
    from core.utils.utils import InputPadder
    registry_path=PROJECT.parents[1]/'source-registry.json';registry=json.loads(registry_path.read_text())
    for rel,digest in registry['source_files'].items():assert sha(PROJECT.parents[1]/rel)==digest
    a.output.mkdir(parents=True,exist_ok=False);random.seed(1234);np.random.seed(1234);torch.manual_seed(1234);torch.cuda.manual_seed_all(1234)
    torch.set_num_threads(8);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    model=SpikingS2M2().cuda();opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-5,eps=1e-8)
    scheduler=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=2e-4,total_steps=10100,pct_start=.01,cycle_momentum=False,anneal_strategy='linear')
    train=driver.ManifestDataset(a.data_root,manifest['train']);dev=driver.ManifestDataset(a.data_root,manifest['dev'])
    loader=DataLoader(dev,batch_size=2,num_workers=0,generator=torch.Generator().manual_seed(2225))
    order=torch.randperm(960,generator=torch.Generator().manual_seed(1234)).tolist();updates=2 if a.preflight else 2500
    write(a.output/'metadata.json',dict(config=dict(model='s2m2_spiking',seed=1234,updates=updates,effective_batch=8,physical_batch=4,augmentation='hv',schedule_total=10100,eval_updates=[] if a.preflight else [0,250,1000,2500],loss='final fullmaskL1 +0.5fixedOTinitialL1 +0.1globalCE; actualOT/global+localrefinement',spike_coding='4 internal static-descriptor steps; native50binsCNNchannels',test_access='sealed'),source_registry=registry,manifest_sha256=MANIFEST_SHA,order=order,parameters=sum(p.numel() for p in model.parameters()),gpu_inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name','--format=csv,noheader'],text=True),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),logical_gpu_uuid=str(getattr(torch.cuda.get_device_properties(0),'uuid','unavailable')),gpu=torch.cuda.get_device_name(0),torch=torch.__version__,slurm={k:v for k,v in os.environ.items() if k.startswith('SLURM_')},command=sys.argv,started_utc=datetime.now(timezone.utc).isoformat()))
    history=[]
    @torch.no_grad()
    def evaluate(update):
        model.eval();frames=[]
        py,npstate=random.getstate(),np.random.get_state()
        with torch.random.fork_rng(devices=[0]):
            for left,right,target,valid,indices in loader:
                left,right,target,valid=[x.cuda() for x in (left,right,target,valid)]
                out=model(left,right);pred=out['flow'];assert torch.isfinite(pred).all()
                for b,index in enumerate(indices.tolist()):
                    err=(pred[b,0]-target[b,0]).abs();mask=valid[b]
                    x=torch.arange(target.shape[-1],device=pred.device)[None]
                    inside=mask&(x+target[b,0]>=0)&(x+target[b,0]<=target.shape[-1]-1)
                    outside=mask&~inside;row=loader.dataset.rows[index]
                    frames.append(dict(scene=row['scene'],frame=row['frame'],valid_pixels=int(mask.sum()),initial_epe=float((out['initial_flow'][b,0]-target[b,0]).abs()[mask].mean()),mean_confidence=float(out['confidence'][b,0][mask].mean()),**driver.metric_values(err[mask]),inframe_pixels=int(inside.sum()),out_of_frame_pixels=int(outside.sum()),inframe_epe=float(err[inside].mean()) if inside.any() else None,out_of_frame_epe=float(err[outside].mean()) if outside.any() else None))
        random.setstate(py);np.random.set_state(npstate)
        scenes={s:float(np.mean([f['epe'] for f in frames if f['scene']==s])) for s in sorted({f['scene'] for f in frames})};count=sum(f['valid_pixels'] for f in frames)
        row=dict(update=update,samples_seen=update*8,mean_scene_epe=float(np.mean(list(scenes.values()))),scene_epe=scenes,per_frame=frames,pooled_epe=sum(f['epe']*f['valid_pixels'] for f in frames)/count,**{k:sum(f[k]*f['valid_pixels'] for f in frames)/count for k in ('bad1','bad3','bad5')})
        initial_scenes={s:float(np.mean([f['initial_epe'] for f in frames if f['scene']==s])) for s in scenes};row['initial_scene_epe']=initial_scenes;row['initial_mean_scene_epe']=float(np.mean(list(initial_scenes.values())))
        append(a.output/'evaluations.jsonl',row);history.append(row);print('DEV',update,row['mean_scene_epe'],flush=True)
    if not a.preflight:evaluate(0)
    torch.cuda.reset_peak_memory_stats()
    for step in range(updates):
        torch.cuda.synchronize();tick=time.monotonic();model.train();opt.zero_grad(set_to_none=True)
        indices=[order[(step*8+j)%960] for j in range(8)];items=[];flags=[]
        for j,index in enumerate(indices):
            item=list(train[index]);digest=hashlib.sha256(f'recipe-hv-v1:1234:{step*8+j}'.encode()).digest();h,v=digest[0]<128,digest[1]<128
            if h:item[:4]=[x.flip(-1) for x in item[:4]];item[2]=-item[2]
            if v:item[:4]=[x.flip(-2) for x in item[:4]]
            items.append(item);flags.append([h,v])
        target=torch.stack([item[2] for item in items]);valid=torch.stack([item[3] for item in items])[:,None]
        x=torch.arange(target.shape[-1])[None,None,None];inside=valid&(x+target>=0)&(x+target<=target.shape[-1]-1)
        padding=InputPadder(target.shape,divis_by=32)._pad
        counts=dict(valid=int(valid.sum()),correspondence=int(F.pad(inside,padding)[...,::4,::4].sum()))
        total_loss=0.;parts_sum={}
        for offset in (0,4):
            micro=items[offset:offset+4];left,right,target,valid=[torch.stack([item[j] for item in micro]).cuda() for j in range(4)]
            out=model(left,right);loss,parts=direct_loss(out,target,valid,counts)
            assert torch.isfinite(loss);loss.backward();total_loss+=float(loss.detach())
            for key in ('full_l1','correspondence_ce','initial_l1'):parts_sum[key]=parts_sum.get(key,0)+float(parts[key].detach())
        qk=float(model.disp_init.q.linear.weight.grad.norm());assert math_isfinite(qk) and qk>0
        key_grad=float(model.disp_init.k.linear.weight.grad.norm());assert math_isfinite(key_grad) and key_grad>0
        mrt_gradients={name:float(parameter.grad.norm()) for name,parameter in model.named_parameters() if parameter.grad is not None and name.startswith('transformer.') and name.endswith(('.q.linear.weight','.k.linear.weight'))}
        assert mrt_gradients and all(math_isfinite(value) for value in mrt_gradients.values()) and sum(mrt_gradients.values())>0
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm) and norm>0
        lr=opt.param_groups[0]['lr'];opt.step();scheduler.step();torch.cuda.synchronize()
        append(a.output/'train.jsonl',dict(update=step+1,samples_seen=(step+1)*8,indices=indices,augmentation_hv=flags,loss=total_loss,**parts_sum,gradient_norm=float(norm),final_qk_gradient=qk,final_key_gradient=key_grad,mrt_qk_gradient_sum=sum(mrt_gradients.values()),mrt_qk_gradients=mrt_gradients if step<2 or step+1 in (250,1000,2500) else {},lr_used=lr,lr_next=scheduler.get_last_lr()[0],spike_activity=float(out['activity']),seconds=time.monotonic()-tick,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,resident_allocated_mib=torch.cuda.memory_allocated()/2**20))
        print('UPDATE',step+1,'LOSS',total_loss,'QK_GRAD',qk,flush=True)
        if not a.preflight and step+1 in (250,1000,2500):evaluate(step+1)
    assert all(torch.isfinite(v).all() for v in model.state_dict().values())
    assert all(torch.isfinite(v).all() for state in opt.state.values() for v in state.values() if torch.is_tensor(v))
    if not a.preflight:
        # Predetermined first frame of each of the first10 sorted Dev scenes.
        # Same final checkpoint; inference-only diagnostics, never selection.
        panel=[]
        for scene in sorted({r['scene'] for r in manifest['dev']})[:10]:
            panel.append(min((r for r in manifest['dev'] if r['scene']==scene),key=lambda r:r['frame']))
        ds=driver.ManifestDataset(a.data_root,panel);measurements=[];model.eval()
        with torch.no_grad():
            for i in range(len(ds)):
                l,r,t,v,_=ds[i];other=ds[(i+1)%len(ds)][1]
                l,r,t,v=[x[None].cuda() for x in (l,r,t,v)]
                shifted=F.pad(r[...,:-4],(4,0))
                correct=model(l,r)['flow'];shift=model(l,shifted)['flow']
                shuffled=model(l,other[None].cuda())['flow']
                x=torch.arange(t.shape[-1],device=t.device)[None,None,None]
                common=v[:,None]&(x+t>=0)&(x+t+4<=t.shape[-1]-1)
                full=v[:,None]
                measurements.append(dict(scene=panel[i]['scene'],frame=panel[i]['frame'],correct_epe=float((correct-t).abs()[full].mean()),shuffled_epe=float((shuffled-t).abs()[full].mean()),shift_response_median=float(((shift-correct)/4)[common].median()) if common.any() else None,common_pixels=int(common.sum())))
        write(a.output/'geometric-panel.json',dict(checkpoint_update=2500,selection=panel,right_shift_native_pixels=4,shift_fill='zero/no-wrap',frames=measurements))
    for rel,digest in registry['source_files'].items():assert sha(PROJECT.parents[1]/rel)==digest
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),updates=updates,source_registry=registry,order=order,rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())),a.output/'last.pt')
    write(a.output/'result.json',dict(status='COMPLETED',updates=updates,samples_seen=updates*8,equivalent_train_passes=updates*8/960,finite_model_optimizer=True,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,process_seconds=time.monotonic()-started,history=history))

def math_isfinite(x):
    import math
    return math.isfinite(x)
if __name__=='__main__':main()
