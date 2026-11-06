"""S3 standalone sensor-time training; source/registration-bound Slurm CUDA only.

Only DAT/depth parsing, the cyclic cohort and metric formulas are inherited.
No original model, refinement loop, normalized 2S-1 input or sequence loss runs.
"""
import argparse,datetime,hashlib,importlib.util,json,os,random,socket,sys,time
from pathlib import Path
os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
EXPERIMENT=Path(__file__).resolve().parents[3];PROJECT=EXPERIMENT.parents[1];REPO=PROJECT.parents[1]
READER=PROJECT/'core/utils/reader.py'
READER_SHA='e5edc76c5ff13a26975154c16b7cb5a4a763902960b8d10eccc2a9877978b9af'
CONTRACTS={'stage3-sensor-time-core-contract-astra.md':'fc5e1bd185605ea59a433bb6c27df12eec7887da22030a45345d03c2abf5be29','stage3-readout-source-amendment-astra.md':'afaf610dacf433a58359e0199ddde3873313dbf98ea1fba4bda29e4b3b73d9a2','stage3-readout-diagnostics-astra.md':'0fb036aba34989abe37d71b35402bf4b59ed7988fbda9a39e2858267dc2ef965'}
MANIFEST_SHA='d8f07b9dc3a17fcc640777e71d55884f17fb116d29fc5e8c9a23c1892f181d71'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
if sha(READER)!=READER_SHA:raise RuntimeError('frozen dataset reader changed')
spec=importlib.util.spec_from_file_location('sensor_time_frozen_reader',READER);reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
from sensor_time_ssde import build_model
from sensor_readout_diagnostics import frame_summaries,aggregate as aggregate_readout

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(path,value):
 path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def state_hash(state):
 h=hashlib.sha256()
 for name,value in sorted(state.items()):h.update(name.encode());h.update(str(value.dtype).encode());h.update(str(tuple(value.shape)).encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def setup(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True

def sample_order(seed=1234):return torch.randperm(960,generator=torch.Generator().manual_seed(seed)).tolist()
def step_indices(order,update):
 start=(update*2)%len(order);return [order[start],order[(start+1)%len(order)]]

class ManifestDataset(Dataset):
 def __init__(self,root,rows):self.root=Path(root);self.rows=rows
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  row=self.rows[i];depth=reader.read_depth(str(self.root/row['depth'])).astype(np.float32);valid=np.isfinite(depth)&(depth>0);disp=np.zeros_like(depth);disp[valid]=-(289.740625*.08)/depth[valid];valid&=np.abs(disp)<512
  left=reader.get_block_spikes(str(self.root/row['spike_l'])).astype(np.float32);right=reader.get_block_spikes(str(self.root/row['spike_r'])).astype(np.float32)
  if left.shape!=right.shape or left.shape!=(50,250,400):raise RuntimeError('native50-bin geometry changed')
  return torch.from_numpy(left),torch.from_numpy(right),torch.from_numpy(disp[None]),torch.from_numpy(valid),i

def load_batch(dataset,indices):
 items=[dataset[i]for i in indices];values=[torch.stack([item[k]for item in items])for k in range(4)]
 if values[0].shape!=values[1].shape or tuple(values[0].shape)!=(2,50,250,400):raise RuntimeError('native B2 required')
 if not all(bool(((x==0)|(x==1)).all())for x in values[:2]):raise RuntimeError('binary0/1 sensor input required')
 return [x.cuda()for x in values]

def final_loss(prediction,target,valid):
 if prediction.shape!=target.shape or prediction.dtype!=torch.float32 or not torch.isfinite(prediction).all():raise FloatingPointError('invalid final native FP32 flow')
 mask=valid[:,None]&torch.isfinite(target)&(target.abs()<512)
 if not bool(mask.any()):raise RuntimeError('empty valid disparity support')
 return F.smooth_l1_loss(prediction[mask],target[mask],beta=1.,reduction='mean')

def diagnostics(values):
 result={}
 for name,v in values.items():
  if isinstance(v,torch.Tensor):
   if v.numel()!=1 or v.requires_grad:raise RuntimeError('diagnostics must be detached scalars')
   v=float(v)
  if not isinstance(v,(int,float))or not np.isfinite(v):raise RuntimeError('invalid diagnostic '+name)
  result[name]=float(v)
 return result

def forward(model,left,right,with_diagnostics=True):
 # The new model owns its50sensor steps, padding, functional zero states/head.
 if with_diagnostics:
  prediction,values=model(left,right,checkpointed=True,return_diagnostics=True);return prediction,diagnostics(values)
 return model(left,right,checkpointed=True,return_diagnostics=False)

@torch.no_grad()
def evaluate(model,loader,readout_fields=False):
 model.eval();per=[];total_error=0.;total_count=0;bad={k:0 for k in [1,3,5]};diag={};readout=[];batches=0;started=time.perf_counter()
 for left,right,target,valid,indices in loader:
  if tuple(left.shape)!=(2,50,250,400):raise RuntimeError('native B2 eval required')
  left,right,target,valid=[x.cuda()for x in [left,right,target,valid]]
  if readout_fields:
   prediction,raw_values,fields=model(left,right,checkpointed=True,return_diagnostics=True,return_fields=True);values=diagnostics(raw_values);readout.extend(frame_summaries(prediction,target,valid,fields,[loader.dataset.rows[i]for i in indices.tolist()]));del fields
  else:prediction,values=forward(model,left,right)
  if prediction.shape!=target.shape or not torch.isfinite(prediction).all():raise FloatingPointError('invalid evaluation flow')
  for name,v in values.items():diag[name]=diag.get(name,0.)+v
  batches+=1
  for b,index in enumerate(indices.tolist()):
   error=(prediction[b,0]-target[b,0]).abs()[valid[b]]
   if not error.numel():raise RuntimeError('empty evaluation frame')
   source=loader.dataset.rows[index];per.append({'scene':source['scene'],'frame':source['frame'],'valid_pixels':error.numel(),'epe':float(error.mean()),**{f'bad{k}':float((error>k).float().mean()*100)for k in [1,3,5]}});total_error+=float(error.sum());total_count+=error.numel()
   for k in bad:bad[k]+=int((error>k).sum())
 scene={s:float(np.mean([r['epe']for r in per if r['scene']==s]))for s in sorted({r['scene']for r in per})};torch.cuda.synchronize()
 result={'mean_scene_epe':float(np.mean(list(scene.values()))),'pooled_epe':total_error/total_count,'valid_pixels':total_count,'valid_fraction':total_count/(len(per)*250*400),**{f'bad{k}':v/total_count*100 for k,v in bad.items()},'scene_epe':scene,'per_frame':per,'diagnostics_batch_mean':{k:v/batches for k,v in diag.items()},'evaluation_seconds':time.perf_counter()-started}
 if readout_fields:
  result['readout_diagnostics']=aggregate_readout(readout)
  if result['readout_diagnostics']['frame_count']!=60 or result['readout_diagnostics']['scene_count']!=30:raise RuntimeError('fixed readout diagnostic cohort incomplete')
 result['evaluation_seconds']=time.perf_counter()-started
 return result

def optimizer(model):return torch.optim.AdamW(model.parameters(),lr=2e-4,betas=(.9,.999),eps=1e-8,weight_decay=1e-4)
def scheduler(opt):return torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=2e-4,total_steps=10100,pct_start=.01,anneal_strategy='linear',cycle_momentum=False,div_factor=25.,final_div_factor=1e4)
def require_cuda():
 if not os.environ.get('SLURM_JOB_ID')or not torch.cuda.is_available()or torch.cuda.device_count()!=1 or '5090'not in torch.cuda.get_device_name(0):raise RuntimeError('one allocated Slurm RTX5090 required')

def bind(args,kind):
 require_cuda();source=json.loads(Path(args.source_registry).read_text());registration=json.loads(Path(args.registration).read_text());manifest=json.loads(Path(args.manifest).read_text())
 for name,h in source['source_files'].items():
  if sha(REPO/name)!=h:raise RuntimeError('source mismatch '+name)
 if registration['status']!='registered' or registration['source_identity_sha256']!=source['source_identity_sha256'] or registration['manifest_sha256']!=sha(args.manifest) or sha(args.manifest)!=MANIFEST_SHA:raise RuntimeError('source/registration/manifest binding mismatch')
 for name,h in CONTRACTS.items():
  if sha(Path(__file__).parent.parent/'analysis'/name)!=h:raise RuntimeError('scientific contract changed')
 if set(manifest)!={'schema_version','source_manifest_sha256','train','dev'}or len(manifest['train'])!=960 or len(manifest['dev'])!=60:raise RuntimeError('Train/Dev cohort changed')
 if not all(str(row[k]).startswith('Train/')for split in ['train','dev']for row in manifest[split]for k in ['depth','spike_l','spike_r']):raise RuntimeError('non-Train source path')
 if sample_order()[:2]!=[495,243]:raise RuntimeError('paired cyclic order changed')
 if registration['phase']!=kind:raise RuntimeError('registration phase mismatch')
 return source,registration,manifest

def validate_train(config,registration):
 fixed={'campaign_id':'spike-depth-direction-v1-S3-raw','seed':1234,'updates':5000,'samples':10000,'physical_batch':2,'effective_batch':2,'sensor_steps_train':50,'sensor_steps_eval':50,'checkpoint_chunk_steps':10,'input_code':'raw','lr':.0002,'weight_decay':.0001,'schedule_updates':10000,'schedule_total_steps':10100,'loss':'final_masked_smooth_l1_beta1','precision':'fp32','geometry':[250,400],'eval_updates':[1000,2500,5000]}
 if any(config.get(k)!=v for k,v in fixed.items())or config['mode']not in ['hard','continuous']:raise RuntimeError('unauthorized S3 train configuration')
 if config not in [row['config']for row in registration['runs']]or len(registration['runs'])!=2:raise RuntimeError('unregistered arm/budget')

def main():
 parser=argparse.ArgumentParser()
 for name in ['config','registration','source-registry','manifest','data-root','output','preflight-receipt','training-review']:parser.add_argument('--'+name,required=True,type=Path)
 args=parser.parse_args();source,registration,manifest=bind(args,'training');config=json.loads(args.config.read_text());validate_train(config,registration)
 preflight=json.loads(args.preflight_receipt.read_text());review=json.loads(args.training_review.read_text())
 if preflight['status']!='PASS_ENGINEERING_ONLY_REQUIRES_TRAINING_REVIEW'or not review['training_authorized']or review['source_identity_sha256']!=source['source_identity_sha256']or review['registration_sha256']!=sha(args.registration)or review['preflight_sha256']!=sha(args.preflight_receipt):raise RuntimeError('training release gate missing')
 if args.output.exists():raise RuntimeError('output exists; fresh runs only/no automatic resume')
 args.output.mkdir(parents=True);started=time.perf_counter();setup(config['seed']);train=ManifestDataset(args.data_root,manifest['train']);dev=ManifestDataset(args.data_root,manifest['dev']);loader=DataLoader(dev,batch_size=2,shuffle=False,num_workers=0,generator=torch.Generator().manual_seed(2225));model=build_model(config['mode'],config['seed']).cuda()
 if any(isinstance(m,torch.nn.modules.batchnorm._BatchNorm)for m in model.modules()):raise RuntimeError('BatchNorm is forbidden')
 initial_hash=state_hash(model.state_dict());opt=optimizer(model);schedule=scheduler(opt);order=sample_order();params=sum(p.numel()for p in model.parameters());metadata={'config':config,'source':source,'manifest_sha256':sha(args.manifest),'registration_sha256':sha(args.registration),'parameter_count':params,'active_parameter_count':sum(p.numel()for p in model.parameters()if p.requires_grad),'initial_state_sha256':initial_hash,'fresh_initialization':True,'restore_checkpoint':None,'torch':torch.__version__,'cuda':torch.version.cuda,'python':sys.version,'hostname':socket.gethostname(),'slurm':{k:v for k,v in os.environ.items()if k.startswith('SLURM_')},'started_utc':now(),'training_contract':{'name':'ssde_sensor_time_hybrid_v1','sensor_steps':50,'gradients':'full temporal BPTT; nonreentrant chunks10; no detaches','input':'binary0/1; no2S-1; all50 supplied bins once','state':'functional perview/perunit zero eachforward','loss':'one final smooth_l1 beta1 over inherited finitepositive depth/absdisp<512 includingrightoutofview','flow_sign':'negative native disparity','optimizer':{'name':'AdamW','max_lr':.0002,'betas':[.9,.999],'eps':1e-8,'weight_decay':1e-4,'clip_norm':1.},'scheduler':{'name':'OneCycleLR','total_steps':10100,'pct_start':.01,'anneal_strategy':'linear','cycle_momentum':False,'div_factor':25.,'final_div_factor':1e4},'mode':config['mode'],'contracts':CONTRACTS,'not_official_SSDEFormer':True,'no_energy_or_motion_claim':True}}
 save(args.output/'metadata.json',metadata)
 # Fixed numerical warmup only, then a fresh functional state on every call.
 l,r,target,valid=load_batch(dev,[0,1]);model.eval()
 with torch.no_grad():
  for _ in range(2):forward(model,l,r)
 if state_hash(model.state_dict())!=initial_hash:raise RuntimeError('warmup changed tensors')
 del l,r,target,valid
 save(args.output/'warmup.json',{'batches':2,'pairs':4,'sensor_steps':50,'no_grad':True,'optimizer_updates':0,'state_tensors_unchanged':True});initial=evaluate(model,loader);save(args.output/'initial.json',initial);history=[];best=None
 with(args.output/'evaluations.jsonl').open('a')as f:f.write(json.dumps({'update':0,**initial})+'\n')
 for update in range(1,5001):
  begun=time.perf_counter();model.train();opt.zero_grad(set_to_none=True);left,right,target,valid=load_batch(train,step_indices(order,update-1));prediction,diag=forward(model,left,right);loss=final_loss(prediction,target,valid);loss.backward();grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
  if not np.isfinite(grad)or any(not torch.isfinite(p.grad).all()for p in model.parameters()if p.grad is not None):raise FloatingPointError('nonfinite gradients')
  used=float(opt.param_groups[0]['lr']);opt.step();schedule.step();torch.cuda.synchronize();mask=valid[:,None];train_epe=float((prediction.detach()-target).abs()[mask].mean());row={'update':update,'samples_seen':2*update,'loss':float(loss.detach()),'disparity_loss':float(loss.detach()),'train_epe':train_epe,'gradient_norm':grad,'used_lr':used,'lr':schedule.get_last_lr()[0],'seconds':time.perf_counter()-begun,'elapsed_seconds':time.perf_counter()-started,'timestamp_utc':now(),'peak_memory_mib':torch.cuda.max_memory_allocated()/2**20,'diagnostics':diag}
  with(args.output/'train.jsonl').open('a')as f:f.write(json.dumps(row,allow_nan=False)+'\n')
  del left,right,target,valid,prediction,loss
  if update in config['eval_updates']:
   values=evaluate(model,loader,readout_fields=update==5000)
   if update==5000:
    save(args.output/'final-readout-diagnostics.json',values.pop('readout_diagnostics'));values['readout_diagnostics_file']='final-readout-diagnostics.json'
   entry={'update':update,'elapsed_seconds':time.perf_counter()-started,'timestamp_utc':now(),**values};history.append(entry)
   with(args.output/'evaluations.jsonl').open('a')as f:f.write(json.dumps(entry,allow_nan=False)+'\n')
   state={'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':schedule.state_dict(),'updates':update,'samples_seen':2*update,'source':source,'config':config,'manifest_sha256':sha(args.manifest),'training_contract':metadata['training_contract'],'initial_state_sha256':initial_hash,'order':order,'rng':{'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}}
   checkpoint=args.output/'last.pt.tmp';torch.save(state,checkpoint);checkpoint.replace(args.output/'last.pt')
   if best is None or values['mean_scene_epe']<best['mean_scene_epe']:
    best=entry;checkpoint=args.output/'best.pt.tmp';torch.save(state,checkpoint);checkpoint.replace(args.output/'best.pt')
   save(args.output/'history.json',history)
 final=history[-1];save(args.output/'result.json',{'status':'COMPLETED','primary_final_update':5000,'primary_final':final,'selected_best_secondary':best,'best_update':best['update'],'updates_completed':5000,'samples_seen':10000,'initial_state_sha256':initial_hash,'elapsed_seconds':time.perf_counter()-started,'no_selected_checkpoint_reevaluation':True,'no_automatic_next_stage':True})
if __name__=='__main__':main()
