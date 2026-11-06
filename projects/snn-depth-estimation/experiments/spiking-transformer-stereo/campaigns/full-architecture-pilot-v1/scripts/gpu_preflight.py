"""Serial synthetic critical-path checks in the first ONE-GPU allocation."""
import argparse,json,os,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_pilot as pilot
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available() and torch.cuda.device_count()==1 and '5090' in torch.cuda.get_device_name(0)
torch.set_num_threads(8);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
import opt_einsum
rows=[]
for name in ['original','raft_stereo','selective_raft']:
 model=pilot.build_model(name,1234).cuda();model.train();model.freeze_bn();before={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};torch.cuda.reset_peak_memory_stats()
 generator=torch.Generator().manual_seed(1234);left=torch.randint(0,2,(2,50,250,400),generator=generator).float().cuda();right=torch.randint(0,2,(2,50,250,400),generator=generator).float().cuda();target=torch.full((2,1,250,400),-1.,device='cuda');valid=torch.ones(2,250,400,dtype=torch.bool,device='cuda')
 preds,spikes,volts=pilot.driver.forward(model,left,right,16,True);assert len(preds)==16 and all(x.shape==target.shape for x in preds)
 loss=pilot.common_disparity_loss(preds,target,valid);assert torch.isfinite(loss);loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(grad) and grad>0
 if name!='original':assert spikes==[] and volts==[]
 del preds,spikes,volts,loss;model.zero_grad(set_to_none=True);pilot.driver.reset_model(model)
 with torch.no_grad():pred=pilot.driver.forward(model.eval(),left,right,32)
 assert pred.shape==target.shape and torch.isfinite(pred).all();assert all(torch.equal(before[k],v.detach().cpu()) for k,v in model.state_dict().items())
 rows.append(dict(model=name,synthetic_batch=2,input_shape=[50,250,400],train_refinements=16,eval_refinements=32,finite_backward=True,preclip_gradient_norm=float(grad),weights_and_buffers_unchanged=True,peak_memory_mib=torch.cuda.max_memory_allocated()/2**20,parameters=sum(p.numel() for p in model.parameters()),optimizer_updates=0,dataset_presentations=0))
 print(json.dumps(rows[-1]),flush=True);del model,before,left,right,target,valid,pred,grad;torch.cuda.empty_cache()
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(dict(status='PASS',job_id=os.environ['SLURM_JOB_ID'],gpu=torch.cuda.get_device_name(0),torch=torch.__version__,opt_einsum=opt_einsum.__version__,opt_einsum_path=opt_einsum.__file__,checks=rows,scope='syntheticnativegeometrychecks only; zero optimizerupdates andzero scientificdatasetpresentations'),indent=2)+'\n')
