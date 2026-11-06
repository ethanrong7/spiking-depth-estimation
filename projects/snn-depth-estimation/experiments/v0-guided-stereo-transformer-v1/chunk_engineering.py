"""One discarded-state comparison; never writes the official model or optimizer."""
import argparse,copy,hashlib,importlib.util,json,os,random,shutil,sys,time
from pathlib import Path

def main():
 p=argparse.ArgumentParser()
 for name in ('source','run-root','data-root','output'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args()
 import torch,numpy as np,cv2
 assert os.environ['SLURM_JOB_ID']=='2757' and torch.cuda.device_count()==1
 assert str(torch.cuda.get_device_properties(0).uuid)=='8d9598da-dfec-a680-9b2f-975c955e5130'
 torch.cuda.set_per_process_memory_fraction(.80);torch.set_num_threads(6);cv2.setNumThreads(1)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
 project=a.source/'projects/snn-depth-estimation';exp=project/'experiments/v0-guided-stereo-transformer-v1';sys.path[:0]=[str(exp),str(project/'experiments/shared-gpu-30k-v1')]
 import recipe
 from common import decode,phase_lock
 driver,mechanism,_=recipe.load_frozen(a.source/'historical')
 spec=importlib.util.spec_from_file_location('guided',project/'core/guided_stereo_attention.py');guided=importlib.util.module_from_spec(spec);spec.loader.exec_module(guided)
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with phase_lock(a.run_root/'train','C-chunk-engineering',True,torch.cuda.empty_cache):
  assert torch.cuda.mem_get_info()[0]>24*1024**3,'Existing worker did not release cache; stop engineering only'
  checkpoint=a.output.parent/'engineering-input.pt';shutil.copyfile(a.run_root/'train/C/last.pt',checkpoint);ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
  checkpoint_hash=hashlib.sha256(checkpoint.read_bytes()).hexdigest();initial_step=ck['updates']
  manifest=json.loads((project/'experiments/spiking-transformer-stereo/configs/manifest.json').read_text());dataset=driver.ManifestDataset(a.data_root,manifest['train'])
  batches=[decode(dataset,ck['order'],initial_step+i) for i in range(4)]
  reference=None;results={};spike_reference=None
  for chunk in (8,64):
   model=guided.install_guidance(recipe.build_model(driver,mechanism,'spiking_normalized',1234)).cuda();model.load_state_dict(ck['model']);model.update_block.guided_attention.row_chunk=chunk
   optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-5,eps=1e-8);optimizer.load_state_dict(copy.deepcopy(ck['optimizer']))
   for state in optimizer.state.values():
    for name,value in state.items():
     if torch.is_tensor(value) and name!='step':state[name]=value.cuda()
   random.setstate(ck['rng']['python']);np.random.set_state(ck['rng']['numpy']);torch.set_rng_state(ck['rng']['torch']);torch.cuda.set_rng_state_all(ck['rng']['cuda'])
   model.train();model.freeze_bn();head=model.update_block.guided_attention;collect={'forward':False};spikes=[]
   def begin(module,args):
    if collect['forward']:collect['q']=[];collect['k']=[]
   def q_capture(module,args,value):
    if collect['forward']:collect['q'].append((value.detach()>=.5).cpu())
   def k_capture(module,args,value):
    if collect['forward']:collect['k'].append((value.detach()>=.5).cpu())
   def end(module,args,value):
    if collect['forward']:spikes.append({key:hashlib.sha256(torch.cat(collect[key],dim=1).numpy().tobytes()).hexdigest() for key in ('q','k')})
   hooks=[head.register_forward_pre_hook(begin),head.query_norm.register_forward_hook(q_capture),head.key_norm.register_forward_hook(k_capture),head.register_forward_hook(end)]
   timings=[];output_first=[];losses=[];peak=0;reserved=0
   for run,batch in enumerate(batches):
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();tick=time.perf_counter();loss_sum=0.
    items=batch['items'];denominator=sum(int(item[3].sum()) for item in items)
    for start in (0,4):
     micro=items[start:start+4];left,right,target,valid=[torch.stack([item[j] for item in micro]).cuda() for j in range(4)]
     collect['forward']=run==0;preds,ss,vv=driver.forward(model,left,right,16,True);collect['forward']=False
     if run==0:output_first.extend(pred.detach().cpu() for pred in preds)
     loss=driver.disparity_loss(preds,target,valid)*int(valid.sum())/denominator;loss_sum+=float(loss.detach());loss.backward();driver.reset_model(model);del preds,ss,vv,left,right,target,valid,loss
    if run==0:
     gradients={name:parameter.grad.detach().cpu().clone() for name,parameter in model.named_parameters() if parameter.grad is not None}
     if reference is None:reference=(output_first.copy(),gradients);spike_reference=spikes.copy();parity=None
     else:
      errors=[float((x-y).abs().max()) for x,y in zip(reference[0],output_first)];gradient_errors={name:float((gradient-reference[1][name]).abs().max()) for name,gradient in gradients.items()};close=True
      try:
       for x,y in zip(reference[0],output_first):torch.testing.assert_close(x,y,atol=1e-4,rtol=1e-5)
       for name,gradient in gradients.items():torch.testing.assert_close(gradient,reference[1][name],atol=1e-4,rtol=1e-4)
      except AssertionError:close=False
      parity=dict(max_native_output_error=max(errors),max_parameter_gradient_error=max(gradient_errors.values()),worst_gradient_parameter=max(gradient_errors,key=gradient_errors.get),assert_close_passed=close,all_forward_query_key_spikes_identical=spikes==spike_reference,forward_invocations=len(spikes))
     for hook in hooks:hook.remove()
     hooks=[];del gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();torch.cuda.synchronize();elapsed=time.perf_counter()-tick;peak=max(peak,torch.cuda.max_memory_allocated()/2**20);reserved=max(reserved,torch.cuda.max_memory_reserved()/2**20)
    if run>0:timings.append(elapsed)
    losses.append(loss_sum);del output_first[:]
   results[str(chunk)]=dict(measured_updates=3,seconds=timings,mean_seconds=sum(timings)/len(timings),peak_allocated_mib=peak,peak_reserved_mib=reserved,losses=losses,parity=parity)
   del model,optimizer;torch.cuda.empty_cache()
  report=dict(status='COMPLETED',engineering_only=True,official_training_changed=False,job=os.environ['SLURM_JOB_ID'],step=os.environ.get('SLURM_STEP_ID'),uuid=str(torch.cuda.get_device_properties(0).uuid),input_checkpoint_update=initial_step,input_checkpoint_sha256=checkpoint_hash,physical_batch=4,effective_batch=8,train_iterations=16,rows=64,chunks=results,source_identity=json.loads((a.source/'source-registry.json').read_text())['source_identity_sha256'],script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),timing_scope='Three end-to-end native4x2 train updates per independent state after one parity warmup; includes stack/transfer/forward/loss/backward/clip/Adam, excludes decoding/checkpoints/validation; official worker held at safe phase lock')
  a.output.write_text(json.dumps(report,indent=2,allow_nan=False));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
