"""Tiny CPU synthetic wiring checks, not experimental validation/performance."""
import json,sys,hashlib
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_pilot as pilot

torch.set_num_threads(1);rows=[]
for name in ['original','raft_stereo','selective_raft']:
 model=pilot.build_model(name,1234);state={k:v.clone() for k,v in model.state_dict().items()};del model
 model=pilot.build_model(name,1234);assert all(torch.equal(v,model.state_dict()[k]) for k,v in state.items());del state
 native=model if name=='original' else model.model
 assert native.fnet.conv1.in_channels==native.cnet.conv1.in_channels==50
 model.eval();seen={};handles=[]
 for key,module in [('feature',native.fnet.conv1),('context',native.cnet.conv1)]:
  def hook(m,args,key=key):seen.setdefault(key,args[0].detach().clone())
  handles.append(module.register_forward_pre_hook(hook))
 left=torch.zeros(1,50,32,64);left[:,::2,:,::2]=1;right=left.flip(1).clone()
 with torch.no_grad():
  preds,spikes,volts=model(left,right,iters=2);_,final,_=model(left,right,iters=2,test_mode=True)
 assert len(preds)==2 and all(p.shape==(1,1,32,64) and torch.isfinite(p).all() for p in preds) and final.shape==(1,1,32,64) and torch.isfinite(final).all()
 assert torch.equal(seen['context'],2*left-1) and torch.equal(seen['feature'][:1],2*left-1)
 if name!='original':assert spikes==[] and volts==[]
 for h in handles:h.remove()
 model.train();model.freeze_bn();assert all(not m.training for m in model.modules() if isinstance(m,torch.nn.BatchNorm2d))
 pilot.driver.reset_model(model)
 if name=='selective_raft':assert hasattr(native,'cam') and hasattr(native,'sam') and type(native.update_block).__name__=='BasicSelectiveMultiUpdateBlock'
 rows.append(dict(model=name,parameters=sum(p.numel() for p in model.parameters()),initialization_repeat_exact=True,all50channels_2sminus1=True,train_eval_shape_finite=True,frozen_bn=True,auxiliary_spike_voltage='original' if name=='original' else 'notapplicable'))
 del model
# Explicit negative disparity convention and strict512 common mask.
target=torch.tensor([[[[-2.,-600.],[-3.,-512.]]]]);valid=torch.ones(1,2,2,dtype=torch.bool);pred=target.clone().requires_grad_(True)
assert pilot.common_disparity_loss([pred]*16,target,valid)==0
positive=(-target).requires_grad_(True);loss=pilot.common_disparity_loss([positive]*16,target,valid);assert loss>0;loss.backward();assert positive.grad[0,0,0,1]==positive.grad[0,0,1,1]==0
manual=sum(.9**(15-i) for i in range(16))*5
assert abs(float(loss)-manual)<1e-4
assert '/vendor/' not in str(sys.modules['core.update'].__file__)
result=dict(models=rows,common_negative_disparity_and_mask512=True,common_sequence_loss_scalar_and_masked_gradients=True,isolated_import_namespaces=True,scope='tinyCPUonly; GPUbackward/nativegeometryverificationpending')
out=Path(__file__).parents[1]/'results/cpu-checks.json';out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
