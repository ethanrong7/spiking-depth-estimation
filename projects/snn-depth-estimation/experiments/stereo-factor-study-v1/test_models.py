"""Tiny synthetic native API, common tensors and feedforward invariance checks."""
import sys,importlib.util,torch,hashlib
from pathlib import Path
p=Path(__file__).parents[2];sys.path[:0]=[str(p),str(p/'experiments/spiking-transformer-stereo/scripts')]
import run_experiment as driver
from core.stereo_factor_evidence import install_evidence
from core.stereo_factor_feedforward import install_feedforward

def main():
 torch.set_num_threads(2);original=driver.build_model('original',1234);state={k:v.clone() for k,v in original.state_dict().items()};a=install_evidence(original)
 assert all(torch.equal(v,a.state_dict()[k]) for k,v in state.items())
 b=install_feedforward(driver.build_model('original',1234));assert not hasattr(b,'update_block')
 for name in ('fnet','cnet'):assert all(torch.equal(v,state[name+'.'+k]) for k,v in b.get_submodule(name).state_dict().items())
 b.eval();left=torch.randint(0,2,(1,50,32,64)).float();right=torch.randint(0,2,left.shape).float();b.set_native_geometry(left.shape,(0,0,0,0))
 with torch.no_grad():
  one=driver.forward(b,left,right,1);many=driver.forward(b,left,right,32);assert torch.equal(one,many);assert one.shape==(1,1,32,64)
  from core.corr import CorrBlock1D
  old_corr=CorrBlock1D.corr
  CorrBlock1D.corr=staticmethod(lambda *args: (_ for _ in ()).throw(AssertionError('A attempted dense all-pairs')))
  try:
   a.eval();driver.forward(a,left,right,2)
  finally:CorrBlock1D.corr=staticmethod(old_corr)
  native=driver.build_model('original',1234).eval();plain=install_evidence(driver.build_model('original',1234),plain=True).eval();base=driver.forward(native,left,right,2);other=driver.forward(plain,left,right,2)
 print('PASS commonOriginalparameters/nativeBAPI/noALIF/Biterationinvariance;plainOriginaltinyreplaymax',float((base-other).abs().max()))
if __name__=='__main__':main()
