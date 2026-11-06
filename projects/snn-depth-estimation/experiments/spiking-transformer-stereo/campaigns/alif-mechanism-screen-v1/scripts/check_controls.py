"""CPU-only changed-path checks; no experimental data or training."""
import json,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from mechanism import adapt,current
import run_screen as screen
from core.update import ALIFModel

torch.set_num_threads(1);seed=1234
base=screen.base_build('original',seed,False);base_state={k:v.clone() for k,v in base.state_dict().items()};base_rng=torch.get_rng_state().clone();del base
counts={};states={}
for arm in ['O','I','F']:
 model=screen.build_model('original_'+arm,seed,False)
 assert all(torch.equal(v,model.state_dict()[k]) for k,v in base_state.items())
 assert torch.equal(base_rng,torch.get_rng_state())
 states[arm]={k:v.clone() for k,v in model.state_dict().items() if 'current_' in k}
 counts[arm]=dict(total=sum(p.numel() for p in model.parameters()),active=sum(p.numel() for p in model.parameters() if p.requires_grad),dormant=sum(p.numel() for p in model.parameters() if not p.requires_grad))
 assert [cell.current_proj.in_channels for cell in [model.update_block.snn08,model.update_block.snn16,model.update_block.snn32]]==[768,768,640]
 del model
assert all(torch.equal(v,states[arm][k]) for arm in ['I','F'] for k,v in states['O'].items())
assert all(row['total']==12427926 for row in counts.values())

# All three original scale callsites on tiny shapes, preserving convf cropping.
checks=[]
for layer,width,size,s_size in [('08',256,3,2),('16',256,2,1),('32',128,1,None)]:
 for arm in ['O','I','F']:
  torch.manual_seed(99);cell=ALIFModel(128,width,layer=layer)
  class Container(torch.nn.Module):pass
  model=Container();model.update_block=Container()
  # Adapter expects all three cells; use independently initialized identicaltypes.
  model.update_block.snn08=ALIFModel(128,256,layer='08');model.update_block.snn16=ALIFModel(128,256,layer='16');model.update_block.snn32=ALIFModel(128,128,layer='32')
  name={'08':'snn08','16':'snn16','32':'snn32'}[layer];setattr(model.update_block,name,cell)
  reference=ALIFModel(128,width,layer=layer);reference.load_state_dict(cell.state_dict());adapt(model,arm,seed)
  h=torch.randn(1,128,size,4 if layer=='08' else size,requires_grad=True);ctx=[torch.randn_like(h) for _ in range(3)];inputs=[torch.randn_like(h) for _ in range(width//128)];s=0. if s_size is None else torch.randn(1,128,s_size,s_size)
  previous=0.
  for step in range(3):
   if arm=='F':
    c=current(cell,h,*ctx,*inputs);r=cell.snn.convw(h)
    if s_size is not None:
     term=cell.snn.convf(s);r=r+(term[:,:,:size,:] if layer=='08' else term)
    u=.5*previous+.5*r+c;z=cell.snn.surrogate_function(u-.5);expected=u-.5*z
   out=cell(s,h,*ctx,*inputs)
   if arm=='O':
    ref=reference(s,h,*ctx,*inputs);left=out if isinstance(out,tuple) else (out,);right=ref if isinstance(ref,tuple) else (ref,);assert all(torch.equal(a,b) for a,b in zip(left,right))
   if arm=='F':assert torch.equal(cell.snn.v,expected);previous=cell.snn.v
   if arm=='I':assert not hasattr(cell.snn,'_independent_current')
  if arm=='O':
   grad_a=torch.autograd.grad(cell.snn.v.square().mean(),[h,cell.snn.convw.weight],retain_graph=True);grad_b=torch.autograd.grad(reference.snn.v.square().mean(),[h,reference.snn.convw.weight],retain_graph=True);assert all(torch.equal(a,b) for a,b in zip(grad_a,grad_b))
  loss=cell.snn.v.square().mean();loss.backward();assert h.grad is not None and torch.isfinite(h.grad).all()
  if arm=='O':assert all(p.grad is None for p in cell.current_proj.parameters())
  if arm=='F':assert all(p.grad is None for p in cell.convz.parameters())
  if arm in ['I','F']:assert cell.current_proj.weight.grad is not None and torch.isfinite(cell.current_proj.weight.grad).all()
  cell.snn.reset();assert cell.snn.v==0.;checks.append(layer+arm)
# Zero-current I is exactly original O, including gradients, on a tiny middlecell.
models=[]
for arm in ['O','I']:
 model=screen.build_model('original_'+arm,seed,False);cell=model.update_block.snn16
 with torch.no_grad():cell.current_proj.weight.zero_();cell.current_proj.bias.zero_()
 models.append((model,cell))
h0=torch.randn(1,128,2,2);ctx=[torch.randn_like(h0) for _ in range(3)];inputs=[torch.randn_like(h0) for _ in range(2)];s=torch.randn(1,128,1,1);outputs=[];gradients=[]
for model,cell in models:
 h=h0.clone().requires_grad_(True)
 for _ in range(3):out=cell(s,h,*ctx,*inputs)
 outputs.append(cell.snn.v.detach().clone());cell.snn.v.square().mean().backward();gradients.append(h.grad.clone())
assert torch.equal(outputs[0],outputs[1]) and torch.equal(gradients[0],gradients[1])
result=dict(zero_current_I_equals_O_forward_and_gradient=True,seed=seed,base_initial_tensors_exact=True,extra_current_tensors_exact_across_arms=True,base_rng_restored=True,widths=[768,768,640],counts=counts,tiny_checks=checks,original_forward_exact=True,fixed_charge_threshold_reset_equation_exact=True,finite_recurrent_gradients=True,dormant_gradients_absent=True,reset=True)
path=Path(__file__).parents[1]/'results/cpu-checks.json';path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
