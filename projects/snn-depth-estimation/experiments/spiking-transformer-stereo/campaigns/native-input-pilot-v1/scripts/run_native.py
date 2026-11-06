"""Fresh native-input runs; all scientific execution gated by audited paired data."""
import argparse,datetime,hashlib,json,os,random,socket,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from native_data import NativeDataset,apply_crop,crop_coordinates,validate_manifest
from models import build_model,forward,reset_model,PROJECT
from objective import train_pair
from metrics import evaluate


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
 path=Path(path);temp=path.with_suffix(path.suffix+'.part');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def validate_registration(config,registration,manifest_sha,data_audit,geometry_receipt,source_identity):
 if registration['status']!='registered' or registration['manifest_sha256']!=manifest_sha or config['manifest_sha256']!=manifest_sha:raise RuntimeError('unbound dataset/registration')
 if data_audit['status']!='AUDIT_PASS' or data_audit['manifest_sha256']!=manifest_sha or data_audit['train_samples']!=960 or data_audit['dev_samples']!=60:raise RuntimeError('paired data not audited')
 if geometry_receipt['status']!='PASS' or geometry_receipt['manifest_sha256']!=manifest_sha or geometry_receipt['selected_training_geometry']!=config['training_geometry']:raise RuntimeError('unverified common geometry')
 if geometry_receipt['source_identity_sha256']!=source_identity or registration['source_identity_sha256']!=source_identity:raise RuntimeError('preflight/code source identity mismatch')
 if set(geometry_receipt['models_passed'])!={'original','raft_stereo','selective_raft'}:raise RuntimeError('preflight must cover ALLthree')
 if len(registration['runs'])!=3 or {r['config']['model'] for r in registration['runs']}!={'original','raft_stereo','selective_raft'}:raise RuntimeError('exactlythree registered models required')
 if any(r['config']['training_geometry']!=geometry_receipt['selected_training_geometry'] or r['config']['manifest_sha256']!=manifest_sha for r in registration['runs']):raise RuntimeError('ALLthree configs must share selectedgeometry and manifest')
 if not any(r['config']==config for r in registration['runs']):raise RuntimeError('unregistered config')
 expected={'updates':10000,'schedule_updates':10000,'seed':1234,'physical_batch':1,'accumulation_steps':2,'effective_batch':2,'train_refinements':16,'eval_refinements':32,'lr':.0002,'primary_validity':'all_source_valid_finite_positive_no_disparity_cutoff'}
 if any(config.get(k)!=v for k,v in expected.items()) or any(r['config'].get(k)!=v for r in registration['runs'] for k,v in expected.items()):raise RuntimeError('unauthorized protocol')
 if registration['max_allocated_account_gpus']!=1:raise RuntimeError('maxONEGPU required')
 if config['evaluation_updates']!=[0,1000,2500,5000,7500,10000] or config['training_geometry'] not in ['native','crop384x768']:raise RuntimeError('unauthorized grid/geometry')


def main():
 start=time.monotonic();p=argparse.ArgumentParser()
 for n in ['config','registration','manifest','data-audit','geometry-receipt','source-registry','source-root','derived-root','output']:p.add_argument('--'+n,required=True)
 a=p.parse_args()
 if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available() or torch.cuda.device_count()!=1 or '5090' not in torch.cuda.get_device_name(0):raise RuntimeError('one Slurm RTX5090 required')
 c=json.loads(Path(a.config).read_text());registration=json.loads(Path(a.registration).read_text());manifest=json.loads(Path(a.manifest).read_text());source=json.loads(Path(a.source_registry).read_text());data_audit=json.loads(Path(a.data_audit).read_text());geometry=json.loads(Path(a.geometry_receipt).read_text());manifest_sha=digest(a.manifest)
 validate_registration(c,registration,manifest_sha,data_audit,geometry,source['source_identity_sha256'])
 for relative,expected in source['source_files'].items():
  if digest(PROJECT.parents[1]/relative)!=expected:raise RuntimeError('frozen source changed: '+relative)
 validate_manifest(manifest)
 out=Path(a.output);out.mkdir(parents=True,exist_ok=False);seed=c['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
 model=build_model(c['model'],seed).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=.0002,weight_decay=1e-5,eps=1e-8);scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=.0002,total_steps=10100,pct_start=.01,cycle_momentum=False,anneal_strategy='linear',div_factor=25,final_div_factor=10000)
 train=NativeDataset(a.source_root,a.derived_root,manifest['train'],c['model']);dev=NativeDataset(a.source_root,a.derived_root,manifest['dev'],c['model']);order=torch.randperm(960,generator=torch.Generator().manual_seed(seed)).tolist()
 contract=dict(input_modality='50binaryspikeplanes_then_original2sminus1' if c['model']=='original' else 'native3channelRGB0255_then_upstream_normalization',primary_validity=c['primary_validity'],secondary_validity='0<disparity<512_only_secondary',target='negative_actual_left_PFM_positive_disparity_no_fB',training_geometry=c['training_geometry'],evaluation_geometry='native540x960',optimizer='AdamW_2e-4_wd1e-5_eps1e-8',scheduler_total_steps=10100,sequence_loss='16predictions_sum_.9^(15-i)_pooled_allvalid_L1',physical_batch=1,accumulation_steps=2,effective_batch=2,freeze_bn=True,reset_each_sample=True,training_refinements=16,evaluation_refinements=32,no_neuron_regularizer=True,FP32=True,torch_matmul_tf32=False,cudnn_tf32=True,max_allocated_account_gpus=1)
 metadata=dict(config=c,registration_sha256=digest(a.registration),manifest_sha256=manifest_sha,data_audit_sha256=digest(a.data_audit),geometry_receipt_sha256=digest(a.geometry_receipt),source=source,training_semantics=contract,restore_checkpoint=None,parameter_count=sum(p.numel() for p in model.parameters()),command=sys.argv,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),hostname=socket.gethostname(),slurm={k:v for k,v in os.environ.items() if k.startswith('SLURM_')},pip_freeze=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 save(out/'metadata.json',metadata)
 def timing():return dict(elapsed_seconds=time.monotonic()-start,timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 before={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};warm=dev[0];model.eval()
 with torch.no_grad():forward(model,c['model'],warm['left'][None].cuda(),warm['right'][None].cuda(),32,False)
 reset_model(model)
 if not all(torch.equal(before[k],v.detach().cpu()) for k,v in model.state_dict().items()):raise RuntimeError('warmup altered persistent state')
 save(out/'warmup.json',dict(frames=1,refinements=32,training=False,state_tensors_exactly_unchanged=True));del warm,before
 initial={**evaluate(model,c['model'],dev,forward,reset_model),**timing()};save(out/'initial.json',initial)
 with (out/'evaluations.jsonl').open('a') as f:f.write(json.dumps(dict(update=0,**initial))+'\n')
 history=[];best=float('inf');best_update=None
 for update in range(1,10001):
  begun=time.monotonic();indices=[order[((update-1)*2+j)%960] for j in range(2)];items=[]
  for micro,index in enumerate(indices):
   item=apply_crop(train[index],crop_coordinates(seed,update,micro,c['training_geometry']));items.append({**item,**{k:item[k].cuda() for k in ['left','right','target','valid']}})
  crops=[item['crop'] for item in items];coverage=[]
  for item in items:
   y,x,h,w=item['crop'];d=-item['target'][0];v=item['valid'];xx=torch.arange(w,device='cuda')[None]
   coverage.append(dict(valid_pixels=int(v.sum()),right_target_outside_crop=int((v&((xx-d<0)|(xx-d>w-1))).sum()),right_target_outside_native=int((v&((xx+x-d<0)|(xx+x-d>959))).sum())))
  values=train_pair(model,c['model'],items,optimizer,scheduler,forward,reset_model)
  if scheduler.last_epoch!=update:raise RuntimeError('scheduler counter mismatch')
  torch.cuda.synchronize();row=dict(update=update,samples_seen=2*update,indices=indices,crops=crops,coverage=coverage,**values,seconds=time.monotonic()-begun,peak_memory_mib=torch.cuda.max_memory_allocated()/2**20,**timing())
  with (out/'train.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
  del items
  if update in c['evaluation_updates']:
   metrics={**evaluate(model,c['model'],dev,forward,reset_model),**timing()};entry=dict(update=update,**metrics);history.append(entry)
   with (out/'evaluations.jsonl').open('a') as f:f.write(json.dumps(entry,allow_nan=False)+'\n')
   state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),updates=update,samples_seen=2*update,manifest_sha256=manifest_sha,source=source,config=c,training_semantics=contract,order=order,best=min(best,metrics['mean_scene_epe']),rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all()))
   torch.save(state,out/'last.pt')
   if metrics['mean_scene_epe']<best:best=metrics['mean_scene_epe'];best_update=update;torch.save(state,out/'best.pt')
   save(out/'history.json',history);print('EVAL',update,metrics['mean_scene_epe'],flush=True)
 save(out/'result.json',dict(updates_completed=10000,cumulative_samples_seen=20000,primary_final=history[-1],best_update=best_update,best_mean_scene_epe=best,primary_checkpoint='last.pt',secondary_checkpoint='best.pt',**timing()))
if __name__=='__main__':main()
