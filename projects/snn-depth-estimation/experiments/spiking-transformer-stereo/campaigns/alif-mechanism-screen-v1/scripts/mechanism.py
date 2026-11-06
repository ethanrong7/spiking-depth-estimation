"""Within-original-scaffold O/I/F diagnostic adapters; no shared core edits."""
from types import MethodType
import torch
from torch import nn


def current(cell,h,cz,cr,cq,*inputs):
 return cell.current_norm(cell.current_proj(torch.cat((h,cz,cr,cq,*inputs),dim=1)))


def injected_charge(node,s,h,alpha):
 node._original_charge(s,h,alpha)
 node.v=node.v+node._independent_current


def independent_forward(cell,s,h,cz,cr,cq,*inputs):
 cell.snn._independent_current=current(cell,h,cz,cr,cq,*inputs)
 try:return cell._original_forward(s,h,cz,cr,cq,*inputs)
 finally:del cell.snn._independent_current


def fixed_forward(cell,s,h,cz,cr,cq,*inputs):
 node=cell.snn
 node._independent_current=current(cell,h,cz,cr,cq,*inputs)
 try:
  alpha=torch.full_like(h,.5)
  node.v_float_to_tensor(alpha)
  node.v_threshold=torch.full_like(h,.5)
  node.neuronal_charge(s,h,alpha)
  spike=node.neuronal_fire()  # The unchanged original ATan surrogate.
  node.neuronal_reset(spike,torch.ones_like(h))
  return (spike,node.v) if cell.layer=='08' else spike
 finally:del node._independent_current


def adapt(model,arm,seed):
 if arm not in ('O','I','F'):raise ValueError(arm)
 cells=[model.update_block.snn08,model.update_block.snn16,model.update_block.snn32]
 # Isolated deterministic extra-module initialization never advances base RNG.
 with torch.random.fork_rng(devices=[]):
  torch.manual_seed(seed+19001)
  for cell in cells:
   width=128+3*128+cell.input_dim
   cell.current_proj=nn.Conv2d(width,128,1,bias=True)
   cell.current_norm=nn.GroupNorm(1,128,affine=True)
 for cell in cells:
  if arm=='O':
   cell.current_proj.requires_grad_(False);cell.current_norm.requires_grad_(False)
   continue
  cell._original_forward=cell.forward
  cell.snn._original_charge=cell.snn.neuronal_charge
  cell.snn.neuronal_charge=MethodType(injected_charge,cell.snn)
  cell.forward=MethodType(independent_forward if arm=='I' else fixed_forward,cell)
  if arm=='F':
   for module in (cell.convz,cell.convr,cell.convq,cell.parametric_sigmoid):module.requires_grad_(False)
 return model
