"""Registered <=1GPUh S3 zero-scientific-update engineering preflight.

Scratch optimizer timing never updates or feeds back into the scientific model.
All source-data forwards, full50BPTT and timings require one Slurm RTX5090.
"""
import argparse,hashlib,json,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import run_sensor_time as s
from input_codes import transform_stereo

def resident_states(opt,model):
 for parameter in model.parameters():opt.state[parameter]={'step':torch.tensor(0.),'exp_avg':torch.zeros_like(parameter),'exp_avg_sq':torch.zeros_like(parameter)}

def backward(model,opt,dataset,indices):
 torch.cuda.synchronize();started=time.perf_counter();model.train();opt.zero_grad(set_to_none=True);left,right,target,valid=s.load_batch(dataset,indices);prediction,diag=s.forward(model,left,right);loss=s.final_loss(prediction,target,valid);loss.backward();grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
 if not np.isfinite(grad)or any(not torch.isfinite(p.grad).all()for p in model.parameters()if p.grad is not None):raise FloatingPointError('nonfinite full50BPTT gradient')
 train_epe=float((prediction.detach()-target).abs()[valid[:,None]].mean());value=float(loss.detach());torch.cuda.synchronize();seconds=time.perf_counter()-started;del left,right,target,valid,prediction,loss
 return {'decode_transfer_forward50_backward50_clip_seconds':seconds,'finite_loss':value,'train_epe_engineering_only':train_epe,'finite_gradient_norm':grad,'diagnostics':diag,'scientific_optimizer_updates':0}

def scratch_optimizer(model):
 # No reference/alias to model parameter or gradient storage is allowed.
 originals=list(model.parameters());copies=[torch.nn.Parameter(p.detach().clone())for p in originals]
 for original,copy in zip(originals,copies):
  assert original.data_ptr()!=copy.data_ptr()
  copy.grad=None if original.grad is None else original.grad.detach().clone()
 opt=torch.optim.AdamW(copies,lr=2e-4,betas=(.9,.999),eps=1e-8,weight_decay=1e-4)
 for parameter in copies:opt.state[parameter]={'step':torch.tensor(0.),'exp_avg':torch.zeros_like(parameter),'exp_avg_sq':torch.zeros_like(parameter)}
 times=[]
 for _ in range(3):
  torch.cuda.synchronize();started=time.perf_counter();opt.step();torch.cuda.synchronize();times.append(time.perf_counter()-started)
 state=opt.state_dict();return copies,opt,state,times

def main():
 parser=argparse.ArgumentParser()
 for name in ['registration','source-registry','manifest','data-root','output']:parser.add_argument('--'+name,type=Path,required=True)
 args=parser.parse_args();source,reg,manifest=s.bind(args,'preflight');spec=reg['preflight']
 expected={'scientific_optimizer_updates':0,'modes':['hard','continuous'],'warmup_repetitions':2,'timed_forward_backward_repetitions':10,'engineering_scratch_optimizer_steps_per_mode':3,'CPU_count_timing_repetitions':3,'full_Dev_timing_passes_per_mode':1,'max_allocated_GPU_hours':1,'reserved_memory_fraction_max':.9,'forecast_updates_per_run':5000,'forecast_evaluations_per_run':4,'forecast_checkpoint_writes_per_run_max':6,'conditional_runs_forecast':6,'forecast_contingency':1.2,'max_allocated_hours_per_training_run':5,'max_two_run_block_GPU_hours':10,'study_GPU_hours_cap':40,'prior_study_allocated_GPU_seconds':4911}
 if spec!=expected:raise RuntimeError('preflight differs from exact engineering registration')
 if args.output.exists():raise RuntimeError('preflight output exists: inspect/no retry')
 args.output.mkdir(parents=True);started=time.perf_counter();s.setup(1234);train=s.ManifestDataset(args.data_root,manifest['train']);dev=s.ManifestDataset(args.data_root,manifest['dev']);indices=s.step_indices(s.sample_order(),0);assert indices==[495,243]
 # CPU timing of the canonical code is engineering allowance for conditional
 # controls only; no canonical-input neural forward or training is performed.
 items=[train[i]for i in indices];left=torch.stack([x[0]for x in items]);right=torch.stack([x[1]for x in items]);count_times=[]
 for _ in range(3):
  t=time.perf_counter();coded=transform_stereo(left,right,'count_only');count_times.append(time.perf_counter()-t)
  assert all(torch.equal(c.sum(1),r.sum(1))for c,r in zip(coded,[left,right]))
 del items,left,right,coded
 modes={};common_hash=None;total_device=torch.cuda.get_device_properties(0).total_memory
 for mode in ['hard','continuous']:
  if time.perf_counter()-started>=3300:raise RuntimeError('preflight one-hour ceiling approaching; no extension')
  s.setup(1234);model=s.build_model(mode,1234).cuda();initial=s.state_hash(model.state_dict())
  if common_hash is None:common_hash=initial
  if initial!=common_hash:raise RuntimeError('hard/continuous initialization differs')
  if any(isinstance(m,torch.nn.modules.batchnorm._BatchNorm)for m in model.modules()):raise RuntimeError('BatchNorm forbidden')
  opt=s.optimizer(model);schedule=s.scheduler(opt);resident_states(opt,model);parameter_values=sum(p.numel()for p in model.parameters());optimizer_temp=torch.empty(2*parameter_values,device='cuda',dtype=torch.float32)
  torch.cuda.reset_peak_memory_stats();warm=[backward(model,opt,train,indices)for _ in range(2)];repetitions=[backward(model,opt,train,indices)for _ in range(10)];training_peak_alloc=torch.cuda.max_memory_allocated();training_peak_reserved=torch.cuda.max_memory_reserved();assert s.state_hash(model.state_dict())==initial
  assert all(int(v['step'])==0 for v in opt.state.values())and schedule.last_epoch==0
  copies,scratch,scratch_state,optimizer_times=scratch_optimizer(model);assert s.state_hash(model.state_dict())==initial
  # Matching-shape payload is explicitly engineering scratch, not a checkpoint:
  # no source/config/update metadata and it is never restored into the model.
  payload={'engineering_only_NOT_checkpoint':True,'model_shape_payload':model.state_dict(),'optimizer_shape_payload':scratch_state};temporary=args.output/(mode+'-engineering-NOT-checkpoint.pt');torch.cuda.synchronize();t=time.perf_counter();torch.save(payload,temporary);io_write=time.perf_counter()-t;payload_bytes=temporary.stat().st_size;t=time.perf_counter();loaded=torch.load(temporary,map_location='cpu',weights_only=True);io_read=time.perf_counter()-t
  assert loaded['engineering_only_NOT_checkpoint']is True;payload_sha=hashlib.sha256(temporary.read_bytes()).hexdigest();temporary.unlink();del loaded,payload,copies,scratch,scratch_state
  assert s.state_hash(model.state_dict())==initial
  loader=DataLoader(dev,batch_size=2,shuffle=False,num_workers=0,generator=torch.Generator().manual_seed(2225));values=s.evaluate(model,loader,readout_fields=True);engineering_peak_alloc=torch.cuda.max_memory_allocated();engineering_peak_reserved=torch.cuda.max_memory_reserved();assert s.state_hash(model.state_dict())==initial
  modes[mode]={'initial_state_sha256':initial,'parameter_count':parameter_values,'active_parameter_count':sum(p.numel()for p in model.parameters()if p.requires_grad),'warmups':warm,'timed_repetitions':repetitions,'engineering_scratch_optimizer_steps':3,'scratch_AdamW_seconds':optimizer_times,'scientific_optimizer_updates':0,'scientific_optimizer_steps_all_zero':True,'scheduler_total_steps':schedule.total_steps,'scheduler_last_epoch':schedule.last_epoch,'actual_model_tensors_unchanged':True,'resident_AdamW_moment_bytes':2*4*parameter_values,'explicit_optimizer_temporary_reserve_bytes':2*4*parameter_values,'training_peak_allocated_bytes':training_peak_alloc,'training_peak_reserved_bytes':training_peak_reserved,'engineering_peak_allocated_bytes':engineering_peak_alloc,'engineering_peak_reserved_bytes':engineering_peak_reserved,'full_Dev_evaluation_seconds':values['evaluation_seconds'],'preflight_accuracy_engineering_only_NOT_study_result':values,'engineering_IO':{'not_a_checkpoint':True,'bytes':payload_bytes,'sha256':payload_sha,'write_seconds':io_write,'read_verification_seconds':io_read,'deleted_only_own_temporary_payload':True}}
  del model,opt,schedule,optimizer_temp,loader;torch.cuda.empty_cache()
 max_step=max(r['decode_transfer_forward50_backward50_clip_seconds']for m in modes.values()for r in m['timed_repetitions']);max_optimizer=max(x for m in modes.values()for x in m['scratch_AdamW_seconds']);max_eval=max(m['full_Dev_evaluation_seconds']for m in modes.values());max_IO=max(m['engineering_IO']['write_seconds']for m in modes.values());max_warm=sum(max(m['warmups'][i]['decode_transfer_forward50_backward50_clip_seconds']for m in modes.values())for i in range(2));max_count=max(count_times)
 forecast_run=1.2*(5000*(max_step+max_optimizer+max_count)+4*max_eval+6*max_IO+max_warm);memory_ok=all(m['engineering_peak_reserved_bytes']<=.9*total_device for m in modes.values());block_ok=2*forecast_run<=10*3600;run_ok=forecast_run<=5*3600;total_reserved=4911+3600+6*forecast_run;total_ok=total_reserved<=40*3600;elapsed=time.perf_counter()-started
 result={'status':'PASS_ENGINEERING_ONLY_REQUIRES_TRAINING_REVIEW'if all([memory_ok,block_ok,run_ok,total_ok,elapsed<=3600])else 'FAIL_RESOURCE_GATE_NO_FALLBACK_OR_TRAINING','job_id':__import__('os').environ['SLURM_JOB_ID'],'source_identity_sha256':source['source_identity_sha256'],'registration_sha256':s.sha(args.registration),'manifest_sha256':s.sha(args.manifest),'contracts':s.CONTRACTS,'actual_geometry':[2,50,250,400],'steps':50,'chunk_steps':10,'checkpointed_full_temporal_BPTT':True,'first_train_indices':indices,'modes':modes,'common_initial_state_sha256':common_hash,'scientific_optimizer_updates':0,'engineering_scratch_optimizer_steps':6,'CPU_count_transform_seconds':count_times,'device_total_memory_bytes':total_device,'memory_fraction_limit':.9,'gates':{'memory':memory_ok,'each_run_within5h':run_ok,'two_run_block_within10GPUh':block_ok,'all6conditional_runs_plus1hpreflight_reserve_within40h':total_ok,'preflight_within1h':elapsed<=3600},'forecast':{'worst_decode_transfer_forward_backward_clip_seconds':max_step,'worst_scratch_AdamW_seconds':max_optimizer,'worst_conditional_CPU_count_transform_seconds':max_count,'full_Dev_evaluation_seconds_max':max_eval,'checkpoint_write_seconds_max':max_IO,'max_checkpoint_writes_per_run':6,'scheduled_evaluations_per_run':4,'contingency':1.2,'seconds_per5k_run':forecast_run,'two_run_block_GPU_hours':2*forecast_run/3600,'six_run_GPU_hours':6*forecast_run/3600,'prior_allocated_GPU_seconds':4911,'preflight_reserved_seconds':3600,'complete_plan_reserved_GPU_hours':total_reserved/3600,'timing_scope':'firstfixedbatchaftertwo numericalwarmups; actualdecode+transfer+full50BPTT+clip, isolatedscratchAdamW, full60Devandmatching-shapecheckpointIO; notcontrolledthroughputcomparison'},'runtime_seconds':elapsed,'fresh_scientific_training_process_required':True,'no_automatic_training_or_conditional_stages':True}
 s.save(args.output/'result.json',result)
 if result['status']!='PASS_ENGINEERING_ONLY_REQUIRES_TRAINING_REVIEW':raise RuntimeError('resource gate failed; retain/no fallback')
if __name__=='__main__':
 try:main()
 except BaseException:
  # Preserve original traceback; do not revise scientific settings or resubmit.
  traceback.print_exc();raise
