"""Unique tiny CPU checks: unchanged parent initialization and loader placement."""
import json,hashlib
from pathlib import Path
import torch
import run_input_screen as s

def main():
 checks={};states=[];rng=[]
 for name in ['original_I_raw','original_I_count_only']:
  model=s.build_model(name,1234);states.append(s.state_digest(model.state_dict()));rng.append(torch.get_rng_state().clone());assert sum(p.numel()for p in model.parameters())==12427926;assert all(p.requires_grad for p in model.parameters());del model
 parent=s.base_parent_build('original_I',1234);assert s.state_digest(parent.state_dict())==states[0]==states[1];assert torch.equal(torch.get_rng_state(),rng[0])and torch.equal(rng[0],rng[1]);checks['both_modes_initial_tensors_and_RNG_exact_parent']=True;checks['all12427926_parameters_active_parent_identical']=True;del parent
 original=s.base_dataset.__getitem__;left=(torch.arange(2*50*4*4).reshape(2,50,4,4)%3==0).float();target=torch.zeros(1,4,4);valid=torch.ones(4,4,dtype=torch.bool);returned=(left[0],left[1],target,valid,0)
 try:
  s.base_dataset.__getitem__=lambda self,index:returned
  dataset=s.InputDataset('.',[]);before=torch.get_rng_state().clone();s.set_mode('raw');raw=dataset[0];assert raw[0].data_ptr()==returned[0].data_ptr()and torch.equal(raw[0],returned[0])and torch.equal(raw[1],returned[1]);s.set_mode('count_only');coded=dataset[0];assert torch.equal(coded[0].sum(0),left[0].sum(0))and torch.equal(coded[1].sum(0),left[1].sum(0));assert coded[2]is target and coded[3]is valid and coded[4]==0;assert torch.equal(before,torch.get_rng_state());assert torch.equal(torch.unique(2*coded[0]-1),torch.tensor([-1.,1.]));checks['loader_transform_before_normalization_targets_unchanged_no_RNG']=True
 finally:s.base_dataset.__getitem__=original
 p=torch.nn.Parameter(torch.ones(1));opt=torch.optim.AdamW([p],lr=.0002,weight_decay=1e-5,eps=1e-8);scheduler=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=.0002,total_steps=10100,pct_start=.01,cycle_momentum=False,anneal_strategy='linear');assert scheduler.total_steps==10100and scheduler.last_epoch==0;checks['parent_OneCycle10100_initialcounter0']=True
 result={'status':'PASS_TINY_CPU_ONLY','checks':checks,'initial_state_sha256':states[0],'parent_identity':s.PARENT_ID,'no_dataset_payload_no_model_forward_no_GPU':True,'source_sha256':hashlib.sha256(Path(s.__file__).read_bytes()).hexdigest()};path=Path(__file__).resolve().parents[1]/'results/s1-cpu-checks.json';path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
