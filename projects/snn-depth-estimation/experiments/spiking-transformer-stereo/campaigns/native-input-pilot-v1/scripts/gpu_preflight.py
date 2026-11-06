"""Serial ONE-GPU native/crop engineering gate, with zero optimizer updates."""
import argparse,gc,json,os,time
from pathlib import Path
import torch
from native_data import NativeDataset,crop_coordinates,apply_crop,validate_manifest
from models import build_model,forward,reset_model,PROJECT
from objective import sequence_sum
from run_native import digest,save
NAMES=['original','raft_stereo','selective_raft']


def trial(name,seed,dataset,geometry,training):
 model=None;optimizer=None;temporary_margin=None;items=[];left=right=preds=loss=None;started=time.monotonic();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
 try:
  model=build_model(name,seed).cuda();parameters=sum(p.numel() for p in model.parameters())
  if training:
   model.train();model.freeze_bn()
   # Mirror resident AdamW FP32 moments, without an optimizer update. Reserve
   # one additional parameter-sized foreach buffer plus64MiB conservatively.
   optimizer=torch.optim.AdamW(model.parameters(),lr=.0002,weight_decay=1e-5,eps=1e-8)
   for parameter in model.parameters():
    if parameter.requires_grad:optimizer.state[parameter]={'step':torch.tensor(0.),'exp_avg':torch.zeros_like(parameter),'exp_avg_sq':torch.zeros_like(parameter)}
   temporary_margin=torch.zeros(parameters+16*1024*1024,device='cuda',dtype=torch.float32)
   items=[]
   for i in range(2):
    sample=apply_crop(dataset[i],crop_coordinates(seed,1,i,geometry));items.append({**sample,**{k:sample[k].cuda() for k in ['left','right','target','valid']}})
   total=sum(int(x['valid'].sum()) for x in items)
   if not total:raise RuntimeError('preflight pair no valid GT')
   loss_value=0.;microbatch_seconds=[]
   for item in items:
    torch.cuda.synchronize();micro_started=time.monotonic()
    left,right,target,valid=[item[k][None] for k in ['left','right','target','valid']];preds=forward(model,name,left,right,16,True);loss=sequence_sum(preds,target,valid)/total
    if not torch.isfinite(loss):raise FloatingPointError('nonfinite preflight loss')
    loss.backward();loss_value+=float(loss.detach());reset_model(model);del target,valid;left=right=preds=loss=None
    torch.cuda.synchronize();microbatch_seconds.append(time.monotonic()-micro_started)
   clip_started=time.monotonic();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
   if not torch.isfinite(torch.tensor(norm)) or norm<=0:raise FloatingPointError('invalid preflight gradients')
   assert all(float(state['step'])==0 for state in optimizer.state.values());torch.cuda.synchronize();clip_seconds=time.monotonic()-clip_started
  else:
   model.eval();item=dataset[0];left,right=[item[k][None].cuda() for k in ['left','right']]
   with torch.no_grad():preds=forward(model,name,left,right,32,False)
   if preds.shape!=(1,1,540,960) or not torch.isfinite(preds).all():raise RuntimeError('native eval output invalid')
   reset_model(model)
  torch.cuda.synchronize()
  return dict(status='PASS',model=name,training=training,geometry=geometry,seconds=time.monotonic()-started,peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(),parameters=parameters,optimizer_updates=0,loss=loss_value if training else None,optimizer_moment_bytes=parameters*8 if training else 0,optimizer_temporary_margin_bytes=(parameters+16*1024*1024)*4 if training else 0,resident_GPU_examples=2 if training else 1,microbatch_forward_backward_seconds=microbatch_seconds if training else [],clip_seconds=clip_seconds if training else None,optimizer_memory_reservation='FP32AdamW2moments+1paramforeachbuffer+64MiB;conservativeconcurrentreservation' if training else None)
 except torch.cuda.OutOfMemoryError as exc:
  return dict(status='OOM',model=name,training=training,geometry=geometry,message=str(exc),optimizer_updates=0)
 finally:
  model=None;optimizer=None;temporary_margin=None;items=[];left=right=preds=loss=None;gc.collect();torch.cuda.empty_cache()


def main():
 p=argparse.ArgumentParser()
 for k in ['registration','manifest','data-audit','source-root','derived-root','source-registry','output']:p.add_argument('--'+k,required=True)
 a=p.parse_args()
 if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available() or torch.cuda.device_count()!=1 or '5090' not in torch.cuda.get_device_name(0):raise RuntimeError('oneGPU Slurm allocation required')
 manifest=json.loads(Path(a.manifest).read_text());audit=json.loads(Path(a.data_audit).read_text());sha=digest(a.manifest);source=json.loads(Path(a.source_registry).read_text())
 registration=json.loads(Path(a.registration).read_text())
 if registration['status']!='registered' or registration['mode']!='native-memory-preflight' or registration['source_identity_sha256']!=source['source_identity_sha256'] or registration['manifest_sha256']!=sha or registration['data_audit_sha256']!=digest(a.data_audit) or registration['models']!=NAMES or registration['max_allocated_account_gpus']!=1 or registration['optimizer_updates']!=0:raise RuntimeError('preflightregistration mismatch')
 fixed={'seed':1234,'train_refinements':16,'eval_refinements':32,'physical_batch':1,'resident_accumulated_examples':2,'optimizer_updates':0,'max_allocated_account_gpus':1}
 if any(registration.get(k)!=v for k,v in fixed.items()) or registration['candidate_geometries']!=['native','common_crop384x768_only_if_ANY_native_train_OOM']:raise RuntimeError('preflightrecipe mismatch')
 if audit['status']!='AUDIT_PASS' or audit['manifest_sha256']!=sha or len(manifest['train'])!=960 or len(manifest['dev'])!=60:raise RuntimeError('data audit gate not passed')
 for path,value in source['source_files'].items():
  if digest(PROJECT.parents[1]/path)!=value:raise RuntimeError('source changed: '+path)
 validate_manifest(manifest)
 output=Path(a.output)
 if output.exists():raise RuntimeError('preflight receipt exists; no blind repeated execution')
 torch.manual_seed(1234);torch.cuda.manual_seed_all(1234);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
 records=[];fallback=False
 for name in NAMES:
  train=NativeDataset(a.source_root,a.derived_root,manifest['train'],name);dev=NativeDataset(a.source_root,a.derived_root,manifest['dev'],name)
  train_result=trial(name,1234,train,'native',True);records.append(train_result);fallback|=train_result['status']=='OOM'
  eval_result=trial(name,1234,dev,'native',False);records.append(eval_result)
  if eval_result['status']!='PASS':raise RuntimeError('fullnativeevaluationfailed; stop, no fallback permitted')
  save(output.with_suffix('.progress.json'),dict(status='ENGINEERING_CHECK',records=records,scientific_training_started=False))
 selected='crop384x768' if fallback else 'native'
 if fallback:
  for name in NAMES:
   train=NativeDataset(a.source_root,a.derived_root,manifest['train'],name);result=trial(name,1234,train,'crop384x768',True);records.append(result)
   if result['status']!='PASS':raise RuntimeError('commoncropfailed; stop, no further fallback')
 save(output,dict(status='PASS',registration_sha256=digest(a.registration),data_audit_sha256=digest(a.data_audit),torch=torch.__version__,cudnn=torch.backends.cudnn.version(),manifest_sha256=sha,source_identity_sha256=source['source_identity_sha256'],selected_training_geometry=selected,native_training_any_OOM=fallback,models_passed=NAMES,records=records,job_id=os.environ['SLURM_JOB_ID'],gpu=torch.cuda.get_device_name(0),max_allocated_account_gpus=1,optimizer_updates=0,scientific_training_started=False,evaluation_geometry='native540x960'))
if __name__=='__main__':main()
