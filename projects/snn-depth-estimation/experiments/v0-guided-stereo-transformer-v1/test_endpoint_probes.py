"""Tiny CPU tests only; no data decoding or experiment evaluation."""
import importlib.util,sys,types,json
from pathlib import Path
import torch
p=Path(__file__).resolve();sys.path.insert(0,str(p.parent));import endpoint_probes as probes
spec=importlib.util.spec_from_file_location('guided',p.parents[2]/'core/guided_stereo_attention.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
torch.set_num_threads(2);torch.manual_seed(44)
content=torch.randn(2,3,19,36,requires_grad=True);valid=torch.rand(content.shape)>.4;valid[0,0,0]=False
permuted=probes.permutation(content,valid,9107);assert torch.equal(permuted[~valid],content[~valid]);assert torch.equal(permuted,probes.permutation(content,valid,9107))
for values,mask,changed in zip(content.reshape(-1,36),valid.reshape(-1,36),permuted.reshape(-1,36)):assert torch.equal(values[mask].sort().values,changed[mask].sort().values)
permuted.sum().backward();assert torch.equal(content.grad,torch.ones_like(content))
class Block(torch.nn.Module):
 def __init__(self):super().__init__();self.guided_attention=g.GuidedCorrection(8,row_chunk=2);self.disp_head=torch.nn.Conv2d(8,2,1)
 def forward(self,v,inp,corr,disp):
  native=self.disp_head(v);extra=self.guided_attention(v,corr,disp);return None,v,None,native+torch.cat((extra,torch.zeros_like(extra)),1)
model=torch.nn.Module();model.update_block=Block().eval();model.update_block.guided_attention.gate.data.fill_(.3)
v=torch.randn(2,8,3,19);corr=torch.randn(2,36,3,19);saved=corr.clone();disp=torch.zeros(2,2,3,19);disp[...,0]=10000
truth=torch.randn(2,3,19);groups={'valid':torch.ones_like(truth,dtype=torch.bool),'empty_group':torch.zeros_like(truth,dtype=torch.bool)}
with torch.no_grad():
 reference=model.update_block(v,None,corr,disp)[3]
 with probes.Recorder(model,truth,groups,'native',0) as recorder:observed=model.update_block(v,None,corr,disp)[3]
 assert torch.equal(observed,reference);assert recorder.rows['1']['valid']['count']==114;json.dumps(recorder.rows,allow_nan=False)
 with probes.Recorder(model,truth,groups,'off',0):off=model.update_block(v,None,corr,disp)[3]
 assert torch.equal(off,model.update_block.disp_head(v))
 with probes.Recorder(model,truth,groups,'permuted',0) as recorder:changed=model.update_block(v,None,corr,disp)[3]
 assert torch.equal(corr,saved);assert torch.isfinite(changed).all();json.dumps(recorder.rows,allow_nan=False)
print('PASS valid-only fixed-seed bijection, untouched invalid/native content, gradients, bit-exact instrumented native/off, finite empty support/groups, hooks restored')
if len(sys.argv)>1:
 import recipe
 driver,mechanism,_=recipe.load_frozen(Path(sys.argv[1])/'historical');native_model=g.install_guidance(recipe.build_model(driver,mechanism,'spiking_normalized',1234)).eval();native_model.update_block.guided_attention.gate.data.fill_(.1)
 left=torch.randn(2,50,32,64);right=torch.randn_like(left);gt=torch.zeros(2,8,16);group={'valid':torch.ones_like(gt,dtype=torch.bool)}
 with torch.no_grad():
  reference=driver.forward(native_model,left,right,32);driver.reset_model(native_model)
  with probes.Recorder(native_model,gt,group,'native',0) as r:observed=driver.forward(native_model,left,right,32)
  assert torch.equal(reference,observed);assert len(r.rows)==32;driver.reset_model(native_model)
  with probes.Recorder(native_model,gt,group,'permuted',0) as r:changed=driver.forward(native_model,left,right,32)
  assert torch.isfinite(changed).all() and len(r.rows)==32;driver.reset_model(native_model)
 print('PASS tiny native batch2/32iter adapter instrumented parity EXACT; all32iteration statistics and seeded intervention finite')
