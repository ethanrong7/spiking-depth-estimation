"""One native-batch, zero-update GPU check inside the first S1 allocation."""
import argparse,json,os,time,hashlib
from pathlib import Path
import numpy as np
import torch
import run_input_screen as s

def save(path,value):
 path=Path(path);temporary=path.with_suffix('.part');temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)

def main():
 p=argparse.ArgumentParser()
 for name in ['manifest','data-root','source-registry','registration','output']:p.add_argument('--'+name,required=True)
 a=p.parse_args();out=Path(a.output)
 if out.exists():raise RuntimeError('preflight receipt exists; inspect/no retry')
 if not os.environ.get('SLURM_JOB_ID')or not torch.cuda.is_available()or torch.cuda.device_count()!=1 or '5090'not in torch.cuda.get_device_name(0):raise RuntimeError('one5090SlurmGPU required')
 started=time.monotonic();source=json.loads(Path(a.source_registry).read_text());registration=json.loads(Path(a.registration).read_text());manifest=json.loads(Path(a.manifest).read_text());manifest_sha=hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest()
 for rel,h in source['source_files'].items():
  if hashlib.sha256((s.driver.PROJECT.parents[1]/rel).read_bytes()).hexdigest()!=h:raise RuntimeError('source mismatch '+rel)
 if registration['source_identity_sha256']!=source['source_identity_sha256']or registration['manifest_sha256']!=manifest_sha:raise RuntimeError('registration/source/data mismatch')
 torch.manual_seed(1234);torch.cuda.manual_seed_all(1234);np.random.seed(1234);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
 order=torch.randperm(960,generator=torch.Generator().manual_seed(1234)).tolist();indices=order[:2];parent=s.base_dataset(a.data_root,manifest['train']);parent_items=[parent[i]for i in indices]
 def batch(items):return [torch.stack([item[k]for item in items]).cuda()for k in range(4)]
 s.set_mode('raw');dataset=s.InputDataset(a.data_root,manifest['train']);raw_items=[dataset[i]for i in indices]
 assert all(torch.equal(raw_items[j][k],parent_items[j][k])for j in range(2)for k in range(4))
 model=s.build_model('original_I_raw',1234).cuda();initial=s.state_digest(model.state_dict());parent_model=s.base_parent_build('original_I',1234);assert initial==s.state_digest(parent_model.state_dict());del parent_model
 l,r,target,valid=batch(raw_items);model.eval()
 with torch.no_grad():
  s.driver.forward(model,l,r,32);s.driver.reset_model(model)
  reference=s.driver.forward(model,*batch(parent_items)[:2],32);adapted=s.driver.forward(model,l,r,32)
 assert torch.equal(reference,adapted),'raw adapter prediction differs from exactparent';del reference,adapted,l,r,target,valid
 checks=[]
 for alias,mode in s.ALIASES.items():
  s.set_mode(mode);config=next(x['config']for x in registration['runs']if x['config']['model']==alias);s.validate_budget(config,registration,manifest_sha,source,False)
  current=s.build_model(alias,1234);assert s.state_digest(current.state_dict())==initial;del current
  items=[dataset[i]for i in indices]
  if mode=='count_only':assert all(torch.equal(items[j][view].sum(0),parent_items[j][view].sum(0))for j in range(2)for view in [0,1])
  l,r,target,valid=batch(items);model.train();model.freeze_bn();model.zero_grad(set_to_none=True);predictions,_,_=s.driver.forward(model,l,r,16,True);loss=s.driver.disparity_loss(predictions,target,valid)
  if not torch.isfinite(loss):raise FloatingPointError('nonfinite '+mode+' preflight loss')
  loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
  if not np.isfinite(norm)or norm<=0 or any(not torch.isfinite(x.grad).all()for x in model.parameters()if x.grad is not None):raise FloatingPointError('invalid '+mode+' gradient')
  checks.append({'input_code':mode,'finite_loss':float(loss.detach()),'finite_gradient_norm':norm,'optimizer_updates':0,'target_and_geometry_unchanged':True});s.driver.reset_model(model);model.zero_grad(set_to_none=True);del l,r,target,valid,predictions,loss,items
 assert s.state_digest(model.state_dict())==initial,'preflight changed persistent state'
 torch.cuda.synchronize();save(out,{'status':'PASS','job_id':os.environ['SLURM_JOB_ID'],'source_identity_sha256':source['source_identity_sha256'],'manifest_sha256':manifest_sha,'registration_sha256':hashlib.sha256(Path(a.registration).read_bytes()).hexdigest(),'initial_state_sha256':initial,'initial_state_exact_frozen_parent_and_both_modes':True,'raw_dataset_target_identity':True,'raw_parent_prediction_exact_after_numerical_warmup':True,'no_persistent_tensor_change':True,'optimizer_updates':0,'fresh_model_and_optimizer_reconstructed_in_new_training_process':True,'indices':indices,'mode_checks':checks,'runtime_seconds':time.monotonic()-started,'peak_memory_allocated_bytes':torch.cuda.max_memory_allocated(),'gpu':torch.cuda.get_device_name(0),'geometry':[2,50,250,400],'training_refinements':16,'evaluation_refinements':32})
if __name__=='__main__':main()
