"""Exact counted feasibility runner; all scientific execution requires Slurm CUDA."""
import argparse, hashlib, json, os, random, sys, time, subprocess, socket, getpass
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime, timezone
PROCESS_STARTED=time.monotonic()
PROCESS_STARTED_UTC=datetime.now(timezone.utc).isoformat()
os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
PROJECT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(PROJECT),str(PROJECT/'core')]
from core.utils.reader import get_block_spikes, read_depth
from core.utils.utils import InputPadder
from core.spikestereonet import SpikeStereoNet
from core.spiking_transformer_stereo import SpikingTransformerStereo
from spikingjelly.activation_based import functional
from run_provenance import training_contract, validate_parent_contract, validate_confirmation_selection, validate_loaded_confirmation_checkpoint, validate_execution_budget

def reset_model(model):
    if hasattr(model,'update_block') and hasattr(model.update_block,'reset_state'):
        model.update_block.reset_state(clear_diagnostics=False)
    else:
        functional.reset_net(model)

class ManifestDataset(Dataset):
    def __init__(self,root,rows): self.root=Path(root);self.rows=rows
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        row=self.rows[i]; depth=read_depth(str(self.root/row['depth'])).astype(np.float32)
        valid=np.isfinite(depth)&(depth>0)
        disp=np.zeros_like(depth);disp[valid]=-(289.740625*0.08)/depth[valid]
        valid &= np.abs(disp)<512
        left=get_block_spikes(str(self.root/row['spike_l'])).astype(np.float32)
        right=get_block_spikes(str(self.root/row['spike_r'])).astype(np.float32)
        assert left.shape==right.shape==(50,250,400)
        return torch.from_numpy(left),torch.from_numpy(right),torch.from_numpy(disp[None]),torch.from_numpy(valid),i

def build_model(name,seed,normalize_residual_current=False):
    args=SimpleNamespace(hidden_dims=[128]*3,context_norm='batch',shared_backbone=False,n_snn_layers=3,n_downsample=2,corr_levels=4,corr_radius=4,corr_implementation='reg',mixed_precision=False,feature_encoder='cnn',context_encoder='cnn',matched_initialization_seed=seed,slow_fast_gru=False)
    if name=='ssde_inspired':
        from core.ssdeformer_inspired import SSDEFormerInspired
        torch.manual_seed(seed);return SSDEFormerInspired(args)
    torch.manual_seed(seed); reference=SpikeStereoNet(args)
    if name.startswith('original'): return reference
    torch.manual_seed(seed+777)
    normalization_options={'normalize_residual_current':True} if normalize_residual_current else {}
    model=SpikingTransformerStereo(args,mode='spiking',spatial_mixing=name!='center',**normalization_options)
    model.load_common_state(reference)
    if name=='analog':
        analog=SpikingTransformerStereo(args,mode='analog',**normalization_options);analog.load_state_dict(model.state_dict());return analog
    return model

def forward(model,left,right,iters,training=False):
    if hasattr(model,'update_block') and hasattr(model.update_block,'reset_state'):model.update_block.reset_state()
    else:reset_model(model)
    for module in model.modules():
        if hasattr(module,'v') and isinstance(module.v,(float,int)) and module.v!=0:raise RuntimeError('nonzero reset membrane')
    padder=InputPadder(left.shape,divis_by=32);left,right=padder.pad(left,right)
    if training:
        preds,spikes,volts=model(left,right,iters=iters)
        return [padder.unpad(p) for p in preds],spikes,volts
    _,pred,_=model(left,right,iters=iters,test_mode=True)
    return padder.unpad(pred)

def disparity_loss(preds,target,valid):
    if not all(torch.isfinite(p).all() for p in preds):raise FloatingPointError('nonfinite disparity predictions')
    mask=valid[:,None] & torch.isfinite(target)&(target.abs()<700)
    if not mask.any():raise RuntimeError('no valid disparity pixels')
    return sum((.9**(15/max(1,len(preds)-1)))**(len(preds)-i-1)*(p-target).abs()[mask].mean() for i,p in enumerate(preds))

def metric_values(error):
    return {'epe':float(error.mean()),**{f'bad{k}':float((error>k).float().mean()*100) for k in [1,3,5]}}

@torch.no_grad()
def evaluate(model,loader,iters,output=None):
    model.eval(); per=[];total_error=0.;total_count=0;bad={k:0 for k in [1,3,5]}
    for left,right,target,valid,indices in loader:
        left,right,target,valid=[x.cuda() for x in [left,right,target,valid]]
        pred=forward(model,left,right,iters)
        if not torch.isfinite(pred).all():raise FloatingPointError('nonfinite evaluation prediction')
        for b,index in enumerate(indices.tolist()):
            e=(pred[b,0]-target[b,0]).abs()[valid[b]]
            if not e.numel():raise RuntimeError('empty valid evaluation frame')
            row=loader.dataset.rows[index]; per.append({'scene':row['scene'],'frame':row['frame'],'valid_pixels':e.numel(),**metric_values(e)})
            total_error+=float(e.sum());total_count+=e.numel()
            for k in bad:bad[k]+=int((e>k).sum())
            if output:
                np.savez_compressed(output/f"{row['scene']}-{row['frame']}.npz",prediction=pred[b,0].cpu().numpy(),truth=target[b,0].cpu().numpy(),valid=valid[b].cpu().numpy(),input_rate=left[b].mean(0).cpu().numpy())
    scenes=sorted(set(p['scene'] for p in per));scene_values={s:float(np.mean([p['epe'] for p in per if p['scene']==s])) for s in scenes}
    reset_model(model)
    return {'mean_scene_epe':float(np.mean(list(scene_values.values()))),'pooled_epe':total_error/total_count,'valid_pixels':total_count,'valid_fraction':total_count/(len(per)*250*400),**{f'bad{k}':v/total_count*100 for k,v in bad.items()},'scene_epe':scene_values,'per_frame':per}

@torch.no_grad()
def fit_priors(dataset,device):
    # L1-optimal priors use the lower median, matching torch.median convention.
    maps=[]
    for row in dataset.rows:
        depth=read_depth(str(dataset.root/row['depth'])).astype(np.float32)
        valid=np.isfinite(depth)&(depth>0)
        disp=np.full_like(depth,np.nan);disp[valid]=-(289.740625*0.08)/depth[valid]
        disp[np.abs(disp)>=512]=np.nan;maps.append(torch.from_numpy(disp))
    values=torch.stack(maps).to(device);count=torch.isfinite(values).sum(0)
    scalar=values[torch.isfinite(values)].median();pixel=values.nanmedian(0).values
    pixel=torch.where(count>0,pixel,scalar)
    return scalar,pixel,count

@torch.no_grad()
def evaluate_priors(scalar,pixel,loader):
    per={'scalar':[],'pixel':[]}
    for _,_,truth,valid,indices in loader:
        truth,valid=truth.cuda()[:,0],valid.cuda()
        for name,pred in [('scalar',scalar),('pixel',pixel)]:
            error=(pred-truth).abs()
            for b,index in enumerate(indices.tolist()):
                row=loader.dataset.rows[index];e=error[b][valid[b]]
                per[name].append({'scene':row['scene'],'frame':row['frame'],'valid_pixels':e.numel(),'error_sum':float(e.sum()),**{f'bad{k}_count':int((e>k).sum()) for k in [1,3,5]},**metric_values(e)})
    result={}
    for name,rows in per.items():
        scenes=sorted(set(r['scene'] for r in rows));scene={s:np.mean([r['epe'] for r in rows if r['scene']==s]) for s in scenes}
        total=sum(r['valid_pixels'] for r in rows)
        result[name]={'mean_scene_epe':float(np.mean(list(scene.values()))),'scene_epe':scene,'pooled_epe':sum(r['error_sum'] for r in rows)/total,'valid_pixels':total,**{f'bad{k}':sum(r[f'bad{k}_count'] for r in rows)/total*100 for k in [1,3,5]},'definition':'training valid pixels lower median; global or per pixel'}
    return result

def main():
    run_started=PROCESS_STARTED;started_utc=PROCESS_STARTED_UTC
    def timing():return {'elapsed_seconds':time.monotonic()-run_started,'timestamp_utc':datetime.now(timezone.utc).isoformat()}
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--source-registry',type=Path,required=True);p.add_argument('--checkpoint',type=Path);p.add_argument('--confirmation-authorized',action='store_true');p.add_argument('--selection-record',type=Path);p.add_argument('--campaign-registration',type=Path);a=p.parse_args()
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available(),'scientific execution requires Slurm CUDA'
    assert '5090' in torch.cuda.get_device_name(0);assert torch.cuda.device_count()==1
    c=json.loads(a.config.read_text());manifest=json.loads(a.manifest.read_text());source=json.loads(a.source_registry.read_text())
    for relative,expected in source['source_files'].items():
        if hashlib.sha256((PROJECT.parents[1]/relative).read_bytes()).hexdigest()!=expected:raise RuntimeError(f'source mismatch: {relative}')
    for field,expected in [('physical_batch',2),('effective_batch',2),('train_refinements',16),('eval_refinements',32)]:
        if c.get(field,expected)!=expected:raise RuntimeError('unsupported fixed training protocol: '+field)
    registration=json.loads(a.campaign_registration.read_text()) if a.campaign_registration else None
    validate_execution_budget(c,registration,hashlib.sha256(a.manifest.read_bytes()).hexdigest(),source,a.checkpoint is not None)
    a.output.mkdir(parents=True,exist_ok=False)
    seed=c.get('seed',1234);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    train_key=c.get('train_split','overfit' if c['stage'] in ['smoke','overfit'] else 'train');eval_key=c.get('eval_split','overfit' if train_key=='overfit' else 'dev')
    assert eval_key!='confirmation' or a.confirmation_authorized
    confirmation_selection=None
    if eval_key=='confirmation':
        if not a.selection_record or not a.checkpoint:raise RuntimeError('confirmation requires frozen selection record and checkpoint')
        confirmation_selection=validate_confirmation_selection(a.selection_record,a.checkpoint,c,hashlib.sha256(a.manifest.read_bytes()).hexdigest(),a.output)
    train=ManifestDataset(a.data_root,manifest[train_key]);ev=ManifestDataset(a.data_root,manifest[eval_key]);evloader=DataLoader(ev,batch_size=2,num_workers=0,generator=torch.Generator().manual_seed(c.get('seed',1234)+991))
    model=build_model(c['model'],seed,c.get('normalize_residual_current',False)).cuda();parameter_count=sum(p.numel() for p in model.parameters())
    opt=torch.optim.AdamW(model.parameters(),lr=c.get('lr',.0002),weight_decay=1e-5,eps=1e-8)
    updates=c['updates'];schedule_updates=c.get('schedule_updates',updates)
    scheduler=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=c.get('lr',.0002),total_steps=schedule_updates+100,pct_start=.01,cycle_momentum=False,anneal_strategy='linear')
    manifest_sha=hashlib.sha256(a.manifest.read_bytes()).hexdigest()
    contract=training_contract(c,manifest_sha,train_key,a.data_root)
    parent_lineage=None
    start=0;best=float('inf')
    previous_checkpoint=None
    if a.checkpoint:
        previous_checkpoint=a.checkpoint.parent/'best.pt'
        if not previous_checkpoint.exists():raise RuntimeError('resume requires sibling best.pt')
        ck=torch.load(a.checkpoint,map_location='cuda',weights_only=False)
        if confirmation_selection is not None:validate_loaded_confirmation_checkpoint(ck,confirmation_selection['entry'])
        for key,default in [('model',None),('seed',1234),('normalize_residual_current',False),('lr',.0002),('schedule_updates',updates)]:
            if c.get(key,default)!=ck['config'].get(key,default):raise RuntimeError(f'restore configuration mismatch: {key}')
        parent_lineage=validate_parent_contract(a.checkpoint,ck,contract,source)
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler']);start=ck['updates'];best=ck['best'];
        optimizer_steps=[int(state['step'].item()) for state in opt.state.values() if 'step' in state]
        if optimizer_steps and (min(optimizer_steps)!=start or max(optimizer_steps)!=start):raise RuntimeError('optimizer counters differ from checkpoint update')
        if sorted(ck['order'])!=list(range(len(train))):raise RuntimeError('restored training order is not cohort permutation')
        old_history=json.loads((a.checkpoint.parent/'history.json').read_text())
        if c['stage'] not in ['evaluate','diagnostic'] and any(row['update']>start for row in old_history):raise RuntimeError('history extends beyond restored checkpoint')
        old_best=torch.load(previous_checkpoint,map_location='cpu',weights_only=False)
        if old_best['updates']>start:raise RuntimeError('best checkpoint is newer than restored checkpoint')
        if abs(min(row['mean_scene_epe'] for row in old_history)-best)>1e-6:raise RuntimeError('best metric/history mismatch')
        del old_best
        if scheduler.last_epoch!=start or scheduler.total_steps!=schedule_updates+100:raise RuntimeError('restored schedule/counter mismatch')
        random.setstate(ck['rng']['python']);np.random.set_state(ck['rng']['numpy']);torch.set_rng_state(ck['rng']['torch'].cpu());torch.cuda.set_rng_state_all([v.cpu() for v in ck['rng']['cuda']])
    metadata={'config':c,'command':sys.argv,'source':source,'manifest_sha256':manifest_sha,'training_semantics':contract,'parent_lineage':parent_lineage,'confirmation_selection':confirmation_selection,'git_protocol':json.loads((Path(__file__).parents[1]/'protocol.json').read_text()),'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'cuda':torch.version.cuda,'python':sys.version,'slurm':{k:v for k,v in os.environ.items() if k.startswith('SLURM_')},'parameter_count':parameter_count,'restore_checkpoint':str(a.checkpoint) if a.checkpoint else None,'restore_checkpoint_sha256':hashlib.sha256(a.checkpoint.read_bytes()).hexdigest() if a.checkpoint else None,'output':str(a.output),'hostname':socket.gethostname(),'user':getpass.getuser(),'pip_freeze':subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True),'conda_explicit':(Path(os.environ['STEREO_ENV']).parent/'environment-20261023'/'conda-explicit.txt').read_text()}
    metadata.update(started_utc=started_utc,elapsed_time_definition='monotonic seconds since script entry before heavy imports, including initialization, decoding, evaluation and checkpoint I/O; per-step seconds remains separate',cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),campaign_registration=registration,campaign_registration_sha256=hashlib.sha256(a.campaign_registration.read_bytes()).hexdigest() if a.campaign_registration else None)
    (a.output/'metadata.json').write_text(json.dumps(metadata,indent=2))
    if a.checkpoint and c['stage'] not in ['evaluate','diagnostic']:
        import shutil;shutil.copy2(previous_checkpoint,a.output/'best.pt')
        (a.output/'inherited-best.json').write_text(json.dumps({'checkpoint':str(previous_checkpoint),'sha256':hashlib.sha256(previous_checkpoint.read_bytes()).hexdigest(),'best_metric':best},indent=2))
    if c['stage']=='diagnostic':
        if not a.checkpoint:raise RuntimeError('diagnostic requires checkpoint')
        sweep={str(iters):evaluate(model,evloader,iters) for iters in [8,16,24,32]}
        if c['model']=='spiking':
            for name,options in [('center_only',{'center_only':True}),('attention_disabled',{'disable_attention':True})]:
                with model.update_block.attention_intervention(**options):sweep[name]=evaluate(model,evloader,32)
        (a.output/'result.json').write_text(json.dumps({'refinement_response':sweep,'checkpoint':str(a.checkpoint),'checkpoint_update':start},indent=2));return
    if c['stage']=='evaluate':
        if not a.checkpoint:raise RuntimeError('evaluation requires frozen checkpoint')
        predout=a.output/'predictions';predout.mkdir();final=evaluate(model,evloader,c.get('eval_iters',32),predout)
        prior_path=a.checkpoint.parent/'priors.npz'
        if not prior_path.exists():raise RuntimeError('frozen training priors required for evaluation')
        with np.load(prior_path) as prior:
            scalar=torch.as_tensor(prior['scalar'],device='cuda');pixel=torch.as_tensor(prior['pixel'],device='cuda')
        (a.output/'prior-results.json').write_text(json.dumps(evaluate_priors(scalar,pixel,evloader),indent=2))
        import shutil;shutil.copy2(prior_path,a.output/'priors.npz')
        (a.output/'result.json').write_text(json.dumps({'selected':final,'checkpoint':str(a.checkpoint),'checkpoint_update':start},indent=2));return
    if c['stage']=='smoke':
        geometry=[]
        for left,right,target,valid,index in DataLoader(ev,batch_size=1):
            l,r=left.mean(1).cuda(),right.mean(1).cuda();flow=target[:,0].cuda();mask=valid.cuda();h,w=flow.shape[-2:];yy,xx=torch.meshgrid(torch.arange(h,device='cuda'),torch.arange(w,device='cuda'),indexing='ij');common=mask & (xx+flow>=0)&(xx+flow<=w-1)&(xx-flow>=0)&(xx-flow<=w-1)
            errors={}
            for name,shift in [('none',torch.zeros_like(flow)),('correct',flow),('opposite',-flow)]:
                grid=torch.stack([2*(xx+shift)/(w-1)-1,2*yy.expand_as(flow)/(h-1)-1],-1);warped=F.grid_sample(r[:,None],grid,align_corners=True)[:,0];errors[name]=float((l-warped).abs()[common].mean())
            geometry.append({'frame':ev.rows[int(index[0])],'valid_fraction':float(mask.float().mean()),'depth_quantiles_in_upstream_units':torch.quantile((289.740625*.08)/(-flow[mask]),torch.tensor([0.,.25,.5,.75,1.],device='cuda')).tolist(),'positive_disparity_quantiles':torch.quantile(-flow[mask],torch.tensor([0.,.25,.5,.75,1.],device='cuda')).tolist(),'rate_warp_mae_common_mask':errors,'common_pixels':int(common.sum())})
        (a.output/'geometry.json').write_text(json.dumps(geometry,indent=2));del left,right,target,valid,l,r,flow,mask,common,grid,warped
        batch=next(iter(evloader));left,right=[v.cuda() for v in batch[:2]];model.eval()
        with torch.no_grad():
            one=forward(model,left,right,2);two=forward(model,left,right,2)
            repeat={'exact':bool(torch.equal(one,two)),'max_abs_difference':float((one-two).abs().max()),'mean_abs_difference':float((one-two).abs().mean())};(a.output/'reset-repeat.json').write_text(json.dumps(repeat,indent=2))
            if not torch.allclose(one,two,atol=1e-4,rtol=1e-4):raise RuntimeError(f'cross-batch repeat mismatch {repeat}')
        del batch,left,right,one,two
    initial={**evaluate(model,evloader,32),**timing()}
    with (a.output/'evaluations.jsonl').open('a') as f:f.write(json.dumps({'update':start,**initial})+'\n')
    if a.checkpoint and (a.checkpoint.parent/'initial.json').exists():
        (a.output/'resume-initial.json').write_text(json.dumps(initial,indent=2));initial=json.loads((a.checkpoint.parent/'initial.json').read_text())
    (a.output/'initial.json').write_text(json.dumps(initial,indent=2));print('INITIAL',initial['mean_scene_epe'],flush=True)
    order=ck['order'] if a.checkpoint else torch.randperm(len(train),generator=torch.Generator().manual_seed(seed)).tolist();history=json.loads((a.checkpoint.parent/'history.json').read_text()) if a.checkpoint and (a.checkpoint.parent/'history.json').exists() else []
    for step in range(start,updates):
        model.train();
        if hasattr(model,'freeze_bn'):model.freeze_bn()
        opt.zero_grad(set_to_none=True)
        indices=[order[(step*2+i)%len(order)] for i in range(2)]
        items=[train[i] for i in indices];left,right,target,valid=[torch.stack([z[j] for z in items]).cuda() for j in range(4)]
        begun=time.monotonic();preds,spikes,volts=forward(model,left,right,16,True);loss=F.smooth_l1_loss(preds[-1][:,0][valid],target[:,0][valid]) if c['model']=='ssde_inspired' else disparity_loss(preds,target,valid)
        disparity_component=float(loss);neuron_component=0.
        if c['model']=='original_regularized':
            neuron_loss=1e-4*(torch.stack(spikes,1).mean((0,1))-.05).square().sum()+1e-3*torch.stack(volts,1).square().mean();neuron_component=float(neuron_loss);loss=loss+neuron_loss;del neuron_loss
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
        loss.backward();grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
        if not np.isfinite(grad) or grad==0:raise FloatingPointError(f'invalid gradient {grad}')
        diagnostics=model.update_block.diagnostics() if hasattr(model,'update_block') and hasattr(model.update_block,'diagnostics') else {}
        if c['model'].startswith('original'):
            diagnostics={'output_mean':float(torch.stack([x.detach().mean() for x in spikes]).mean()),'voltage_rms':float(torch.stack([x.detach().square().mean() for x in volts]).mean().sqrt())}
        opt.step();scheduler.step()
        if scheduler.last_epoch!=step+1:raise RuntimeError('scheduler counter differs from exact update count')
        reset_model(model);torch.cuda.synchronize()
        train_epe=float((preds[-1]-target).abs()[:,0][valid].mean())
        row={'disparity_loss':disparity_component,'neuron_loss':neuron_component,'train_epe':train_epe,'update':step+1,'samples_seen':(step+1)*2,'loss':float(loss),'gradient_norm':grad,'lr':scheduler.get_last_lr()[0],'seconds':time.monotonic()-begun,'peak_memory_mib':torch.cuda.max_memory_allocated()/2**20,'diagnostics':diagnostics,**timing()}
        with (a.output/'train.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        del preds,spikes,volts,loss,left,right,target,valid,items
        if (step+1)%c.get('eval_every',100)==0 or step+1==updates:
            result={**evaluate(model,evloader,32),**timing()};history.append({'update':step+1,**result});print('EVAL',step+1,result['mean_scene_epe'],flush=True)
            with (a.output/'evaluations.jsonl').open('a') as f:f.write(json.dumps({'update':step+1,**result})+'\n')
            state={'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'updates':step+1,'samples_seen':(step+1)*2,'manifest_sha256':manifest_sha,'training_semantics':contract,'parent_lineage':parent_lineage,'source':source,'config':c,'order':order,'best':min(best,result['mean_scene_epe']),'rng':{'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}}
            torch.save(state,a.output/'last.pt')
            if result['mean_scene_epe']<best:best=result['mean_scene_epe'];torch.save(state,a.output/'best.pt')
            history_tmp=a.output/'history.tmp';history_tmp.write_text(json.dumps(history,indent=2));history_tmp.replace(a.output/'history.json')
    if not (a.output/'best.pt').exists():
        if previous_checkpoint is None:raise RuntimeError('no selected checkpoint')
        import shutil;shutil.copy2(previous_checkpoint,a.output/'best.pt')
    ck=torch.load(a.output/'best.pt',map_location='cuda',weights_only=False);model.load_state_dict(ck['model']);predout=a.output/'predictions';predout.mkdir();final=evaluate(model,evloader,32,predout)
    if c['stage']!='smoke' and c.get('final_extras',True):
        scalar,pixel,count=fit_priors(train,'cuda');np.savez_compressed(a.output/'priors.npz',scalar=scalar.cpu().numpy(),pixel=pixel.cpu().numpy(),count=count.cpu().numpy())
        (a.output/'prior-results.json').write_text(json.dumps(evaluate_priors(scalar,pixel,evloader),indent=2))
    if c['model']=='spiking' and c['stage']!='smoke' and c.get('final_extras',True):
        interventions={}
        for name,options in [('center_only',{'center_only':True}),('attention_disabled',{'disable_attention':True})]:
            with model.update_block.attention_intervention(**options):interventions[name]=evaluate(model,evloader,32)
        (a.output/'interventions.json').write_text(json.dumps(interventions,indent=2))
    (a.output/'result.json').write_text(json.dumps({'best_update':ck['updates'],'initial':initial,'selected':final,'relative_epe_reduction':1-final['mean_scene_epe']/initial['mean_scene_epe'],'updates_completed':updates,'updates_completed_this_run':updates-start,'cumulative_samples_seen':updates*2,'samples_seen_this_run':(updates-start)*2,'train_samples':len(train),'parameter_count':parameter_count,**timing()},indent=2))
if __name__=='__main__':main()
