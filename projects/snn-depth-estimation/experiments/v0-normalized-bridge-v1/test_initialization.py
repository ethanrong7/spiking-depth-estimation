"""CPU-only historical constructor and current Original compatibility audit."""
import hashlib,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).parent))
import recipe

def digest(model,typed=False):
 h=hashlib.sha256()
 for n,v in sorted(model.state_dict().items()):
  h.update(n.encode())
  if typed:h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode())
  h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def main():
 root=Path(sys.argv[1]);driver,mechanism,registry=recipe.load_frozen(root/'historical')
 full=recipe.build_model(driver,mechanism,'spiking_normalized',1234);assert digest(full,True)=='07bc89397666f2382b3ad11c3ccf6e9c702c8a03b336a353b5a49938d38cd397';assert sum(p.numel() for p in full.parameters())==8398902
 original=recipe.build_model(driver,mechanism,'original',1234);assert digest(original)=='8ceac15b09ce188f9ee52ad5a8eca8e68727cbbb2887a012874d7246dc6495ca'
 for path in ('fnet','cnet','context_zqr_convs','update_block.encoder','update_block.disp_head','update_block.mask'):
  a,b=full.get_submodule(path),original.get_submodule(path);assert all(torch.equal(v,b.state_dict()[n]) for n,v in a.state_dict().items())
 control=recipe.build_model(driver,mechanism,'spiking_normalized',1234)
 for cell in control.update_block.cells:
  cell.attention_enabled=False;cell.qkv.requires_grad_(False);cell.output_proj.requires_grad_(False);cell.attention_gate.requires_grad_(False)
 assert digest(control,True)==digest(full,True);assert sum(p.numel() for p in control.parameters() if p.requires_grad)==8349075
 assert all(cell.normalize_residual_current for cell in full.update_block.cells)
 print('PASS historicaldriverhash/corecapsule/modeltypedhash+8398902params/normalizedcurrent/Original8cea+commoninitializedmodules/pairednodrift+49827frozendormantparams')
if __name__=='__main__':main()
