"""Tiny changed-path CPU controls, not an experiment or native memory claim."""
import json,sys
from pathlib import Path
import numpy as np
import torch
from native_data import input_tensor,target_tensor,crop_coordinates,apply_crop
from objective import sequence_sum,train_pair
from metrics import frame_metrics
from run_native import validate_registration


def main():
 torch.set_num_threads(1)
 rgb=np.array([[[17,93,241],[0,255,4]]],dtype=np.uint8);assert torch.equal(input_tensor(rgb,'raft_stereo'),torch.tensor([[[17.,0.]],[[93.,255.]],[[241.,4.]]]))
 spikes=np.zeros((50,2,3),dtype=bool);spikes[49,1,2]=True;assert input_tensor(spikes,'original')[49,1,2]==1 and input_tensor(spikes,'original').sum()==1
 d=np.array([[1.,512.,700.,900.,np.inf,np.nan,-3.,0.]],dtype=np.float32);target,valid=target_tensor(d);assert valid.tolist()==[[True,True,True,True,False,False,False,False]] and target[0,0,:4].tolist()==[-1.,-512.,-700.,-900.]
 pred=target.clone();pred[0,0,3]+=1000;primary=frame_metrics(pred,target,valid[None]);secondary=frame_metrics(pred,target,valid[None]&(target.abs()<512));assert primary['valid_pixels']==4 and primary['epe']==250. and secondary['valid_pixels']==1 and secondary['epe']==0.
 coords=crop_coordinates(1234,19,1,'crop384x768');assert coords==crop_coordinates(1234,19,1,'crop384x768')
 # Coordinate identity across RGB/50planes/GT; no resize or disparity scaling.
 grid=torch.arange(540*960).reshape(540,960);item=dict(left=grid[None].expand(3,-1,-1),right=grid[None].expand(3,-1,-1),target=grid[None],valid=grid>0)
 spike_crop=apply_crop({**item,'left':grid[None].expand(50,-1,-1),'right':grid[None].expand(50,-1,-1)},coords);assert spike_crop['left'].shape==(50,384,768) and torch.equal(spike_crop['left'][49],spike_crop['target'][0])
 cropped=apply_crop(item,coords);assert cropped['left'].shape==(3,384,768) and torch.equal(cropped['left'][0],cropped['target'][0]) and torch.equal(cropped['right'],cropped['left'])
 class Tiny(torch.nn.Module):
  def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.tensor(.4));self.bn=torch.nn.BatchNorm2d(1);self.resets=0;self.frozen=0
  def freeze_bn(self):self.bn.eval();self.frozen+=1
 model=Tiny();items=[]
 for value,mask in [(1.,[[True,False],[False,False]]),(3.,[[True,True],[True,False]])]:items.append(dict(left=torch.full((1,2,2),value),right=torch.zeros(1,2,2),target=torch.full((1,2,2),-2.),valid=torch.tensor(mask)))
 manual=sum(sequence_sum([model.weight*it['left'][None] for _ in range(16)],it['target'][None],it['valid'][None]) for it in items)/4
 expected=float(manual.detach());manual.backward();gradient=model.weight.grad.clone();model.zero_grad()
 optimizer=torch.optim.AdamW([model.weight],lr=2e-4,weight_decay=1e-5);scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=2e-4,total_steps=10100,pct_start=.01,cycle_momentum=False,anneal_strategy='linear');calls={'clip':0,'optimizer':0,'scheduler':0};raw_step=optimizer.step;raw_scheduler=scheduler.step
 def step(*a,**kw):calls['optimizer']+=1;return raw_step(*a,**kw)
 def schedule(*a,**kw):calls['scheduler']+=1;return raw_scheduler(*a,**kw)
 step._wrapped_by_lr_sched=True
 optimizer.step=step;scheduler.step=schedule
 def forward(m,name,l,r,iters,training):assert not m.bn.training and iters==16 and training;return [m.weight*l for _ in range(16)]
 def reset(m):m.resets+=1
 def clip(parameters,norm):calls['clip']+=1;assert norm==1. and torch.equal(model.weight.grad,gradient);return torch.nn.utils.clip_grad_norm_(parameters,norm)
 before={k:v.clone() for k,v in model.bn.state_dict().items()};result=train_pair(model,'original',items,optimizer,scheduler,forward,reset,clip)
 assert abs(result['loss']-expected)<1e-5 and result['microbatch_valid_pixels']==[1,3] and calls=={'clip':1,'optimizer':1,'scheduler':1} and scheduler.last_epoch==1 and model.resets==2 and model.frozen==1
 assert all(torch.equal(v,model.bn.state_dict()[k]) for k,v in before.items())
 # Equal per-image averaging would be wrong with counts1/3; the test detects it.
 means=[float(sequence_sum([torch.tensor(.4)*it['left'][None] for _ in range(16)],it['target'][None],it['valid'][None]))/int(it['valid'].sum()) for it in items];assert abs(sum(means)/2-expected)>.1
 # Binding must match the source used for the all-model preflight.
 configs=[json.loads((Path(__file__).parents[1]/'configs'/name).read_text()) for name in ['original-seed1234-10k-v1.json','raft-seed1234-10k-v1.json','selective-raft-seed1234-10k-v1.json']]
 for config in configs:config.update(training_geometry='native',manifest_sha256='manifest')
 reg=dict(status='registered',source_identity_sha256='source',manifest_sha256='manifest',max_allocated_account_gpus=1,runs=[dict(config=c) for c in configs]);audit=dict(status='AUDIT_PASS',manifest_sha256='manifest',train_samples=960,dev_samples=60);geometry=dict(status='PASS',manifest_sha256='manifest',selected_training_geometry='native',models_passed=['original','raft_stereo','selective_raft'],source_identity_sha256='source')
 validate_registration(configs[0],reg,'manifest',audit,geometry,'source')
 for wrong in ['source','geometry']:
  if wrong=='source':geometry['source_identity_sha256']='mismatch'
  else:geometry['source_identity_sha256']='source';configs[-1]['training_geometry']='crop384x768'
  try:validate_registration(configs[0],reg,'manifest',audit,geometry,'source');raise ValueError('invalid binding must fail')
  except RuntimeError:pass
 print(json.dumps(dict(preflight_exact_source_and_ALL3_common_geometry_bound=True,scope='tiny synthetic CPU only; noGPU/nativegeometry/dataexperiment',native_RGB_channels_values_exact=True,all50_binary_planes_preserved=True,negative_actual_PFM_target=True,finite_positive_includes512_700_900=True,no_fB_conversion=True,primary_allpositive_and_secondary512_metrics_distinct=True,paired_deterministic_crop_no_rescale=True,pooled_unequal_valid_loss_and_gradient_exact=True,clip_optimizer_scheduler_once_per2examples=True,reset_each_microbatch=True,frozen_BN_buffers_exact=True),indent=2))
if __name__=='__main__':main()
