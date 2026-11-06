"""Tiny CPU geometry, init and native reset/parity checks (not an experiment)."""
import importlib.util,json,sys,hashlib,copy
from pathlib import Path
import torch
p=Path(__file__).resolve();project=p.parents[2]
spec=importlib.util.spec_from_file_location('guided',project/'core/guided_stereo_attention.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
torch.set_num_threads(2);torch.manual_seed(3)
head=g.GuidedCorrection(8,row_chunk=2);v=torch.randn(2,8,3,19,requires_grad=True);corr=torch.randn(2,36,3,19,requires_grad=True);disp=torch.zeros(2,2,3,19)
assert torch.equal(head(v,corr,disp),torch.zeros(2,1,3,19))
head(v,corr,disp).square().sum().backward();head.zero_grad();(head(v,corr,disp)*torch.randn(2,1,3,19)).sum().backward();assert head.gate.grad.abs()>0
with torch.no_grad():head.gate.fill_(.1)
head.zero_grad();out=head(v,corr,disp);out.square().mean().backward();assert head.query.weight.grad.norm()>0 and head.key.weight.grad.norm()>0 and corr.grad.norm()>0 and v.grad.norm()>0
head.eval();assert torch.equal(head(v,corr,disp),head(v,corr,disp))
far=torch.full_like(disp,10000);assert torch.equal(head(v,corr,far),torch.zeros_like(out))
# Uniform attention is neutral on asymmetric borders and symmetric interior.
with torch.no_grad():head.query.weight.zero_();head.query.bias.zero_();head.key.weight.zero_();head.key.bias.zero_()
assert head(v,corr,disp).abs().max()<1e-6
assert torch.equal(head.offsets,torch.cat([torch.arange(-4,5)*2**i for i in range(4)]).float());assert head.offsets.min()==-32 and head.offsets.max()==32
# Force positive/negative signed offset preference on an interior anchor.
forced=g.GuidedCorrection(8).eval()
with torch.no_grad():
 forced.gate.fill_(.5);forced.query.weight.zero_();forced.query.bias.copy_(torch.tensor([1.,-1.]*8));forced.key.weight.zero_();forced.key.bias.zero_();forced.key.weight[:,0]=torch.tensor([1.,-1.]*8)
 vv=torch.zeros(1,8,1,128);dd=torch.zeros(1,2,1,128);cc=torch.zeros(1,36,1,128);cc[:,8]=2
 plus=forced(vv,cc,dd)[0,0,0,64];cc.zero_();cc[:,0]=2;minus=forced(vv,cc,dd)[0,0,0,64]
 assert plus>0 and minus<0;torch.testing.assert_close(plus,-minus,atol=1e-6,rtol=1e-6)
sys.path.insert(0,str(p.parent));import recipe
driver,mechanism,_=recipe.load_frozen(Path(sys.argv[1])/'historical');base=recipe.build_model(driver,mechanism,'spiking_normalized',1234)
for cell in base.update_block.cells:
 cell.attention_enabled=False;cell.qkv.requires_grad_(False);cell.output_proj.requires_grad_(False);cell.attention_gate.requires_grad_(False)
model=g.install_guidance(copy.deepcopy(base));assert all(torch.equal(t,model.state_dict()[name]) for name,t in base.state_dict().items())
x=torch.randn(1,50,32,64);y=torch.randn_like(x)
with torch.no_grad():
 for training,iterations in ((True,16),(False,32)):
  base.train(training);model.train(training);base.freeze_bn();model.freeze_bn();a=driver.forward(base,x,y,iterations,training);b=driver.forward(model,x,y,iterations,training)
  aa=a[0] if training else [a];bb=b[0] if training else [b];assert all(torch.equal(u,v) for u,v in zip(aa,bb));driver.reset_model(base);driver.reset_model(model)
h=hashlib.sha256()
for name,t in sorted(model.state_dict().items()):h.update(name.encode());h.update(t.contiguous().numpy().tobytes())
r=dict(status='PASS',initial_state_sha256=h.hexdigest(),parameters=sum(x.numel() for x in model.parameters()),trainable_parameters=sum(x.numel() for x in model.parameters() if x.requires_grad),native_zero_gate_parity='bit exact all16train and32eval on tiny32x64 CPU',branch_rng=2017)
(p.parent/'tiny-cpu-verification.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
