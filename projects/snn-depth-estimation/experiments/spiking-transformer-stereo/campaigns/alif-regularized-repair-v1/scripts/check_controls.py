"""Tiny CPU checks for the changed loss/contract and unchanged initialization."""
import json,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_repair as repair
from core.loss import neuron_regularization
from core.ALIF import ALIFNode08

torch.set_num_threads(1)
checks={}
for seed in [1234,2345]:
 reference=repair.driver.build_model('original',seed,False);state={k:v.clone() for k,v in reference.state_dict().items()};rng=torch.get_rng_state().clone();del reference
 model=repair.driver.build_model('original_regularized',seed,False)
 assert state.keys()==model.state_dict().keys() and all(torch.equal(v,model.state_dict()[k]) for k,v in state.items())
 assert torch.equal(rng,torch.get_rng_state());assert sum(p.numel() for p in model.parameters())==12148246
 checks[str(seed)]=dict(initial_state_exact=True,rng_exact=True,parameters=12148246);del model,state

spikes=[torch.randn(2,2,2,2,requires_grad=True) for _ in range(3)]
volts=[torch.randn(2,2,2,2,requires_grad=True) for _ in range(3)]
rate,voltage=repair.regularizer_components(spikes,volts);joint=neuron_regularization(spikes,volts)
assert torch.equal(rate+voltage,joint)
grad1=torch.autograd.grad(rate+voltage,spikes+volts,retain_graph=True);grad2=torch.autograd.grad(joint,spikes+volts)
assert all(torch.equal(a,b) and torch.isfinite(a).all() and a.abs().sum()>0 for a,b in zip(grad1,grad2))
# Analytic constant fixture distinguishes sum(C,H,W) from a global mean.
r,v=repair.regularizer_components([torch.full((2,2,2,2),.25) for _ in range(3)],[torch.full((2,2,2,2),2.) for _ in range(3)])
assert torch.allclose(r,torch.tensor(.000032),atol=1e-10) and torch.allclose(v,torch.tensor(.004),atol=1e-10)
node=ALIFNode08();args=[torch.randn(1,128,1,1),torch.randn(1,128,2,2)]+[torch.full((1,128,2,2),.5) for _ in range(3)]
a=node(*args);node.reset();b=node(*args);assert all(torch.equal(x,y) for x,y in zip(a,b));node.reset();assert node.v==0.
checks.update(regularizer_scalar_exact_upstream=True,regularizer_gradients_exact_finite=True,analytic_reductions=True,tiny_neuron_reset_repeat_exact=True)
reg=json.loads((Path(__file__).parents[1]/'configs/registration.json').read_text())
for row in reg['runs']:
 c=row['config'];assert c['model']=='original_regularized' and c['updates']==30000 and c['schedule_updates']==50000 and c['train_refinements']==16 and c['eval_refinements']==32
 contract=repair.training_contract(c,reg['manifest_sha256'],'train',Path('.'));assert contract['scheduler']['total_steps']==50100 and contract['repair_identity']['detach'] is False
checks['configuration_contract']='pass'
out=Path(__file__).parents[1]/'results/cpu-checks.json';out.write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks,indent=2))
