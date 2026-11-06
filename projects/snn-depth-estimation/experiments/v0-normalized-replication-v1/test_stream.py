"""Tiny synthetic proof of seed-specific paired data order and signed HV."""
import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).parents[1]/'shared-gpu-30k-v1'))
import common,recipe
class Toy:
 def __getitem__(self,index):
  x=torch.arange(6).reshape(2,3).float()+index
  return x,x+10,x[None]+20,torch.ones(2,3,dtype=torch.bool),index
original_flags=common.flags
for seed in (2345,3456):
 order=torch.randperm(960,generator=torch.Generator().manual_seed(seed)).tolist();common.flags=lambda position:original_flags(position,seed)
 for step in (0,1,119,120,29999):
  batch=common.decode(Toy(),order,step)
  assert batch['indices']==recipe.batch_indices(order,step,8)
  for j,item in enumerate(batch['items']):
   h,v=recipe.paired_flags(seed,step*8+j);expected=recipe.augment(Toy()[batch['indices'][j]],h,v)
   assert batch['flags'][j]==[h,v] and all(torch.equal(a,b) for a,b in zip(item[:4],expected[:4]))
common.flags=original_flags
print('PASS bothseedorders/absoluteHVflags/signedtensorsacrossepochboundariesand30000scheduleend')
