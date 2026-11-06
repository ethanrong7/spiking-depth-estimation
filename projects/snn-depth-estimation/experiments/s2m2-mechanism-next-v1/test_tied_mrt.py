"""Tiny CPU relay/one-pass/gradient and exact archived initialization checks."""
import torch,hashlib
from core.s2m2_spiking_v2 import SpikingS2M2V2
from core.s2m2_tied_mrt import apply_tied_mrt

def main():
 torch.manual_seed(1234);model=SpikingS2M2V2();before={k:v.clone() for k,v in model.state_dict().items()};original=model.transformer.forward
 dims=model.transformer.uformer_list[0].dims
 pyramid=[torch.randn(2,c,s,s,requires_grad=True) for c,s in zip([dims[0],dims[1],dims[2],dims[2]],(16,8,4,2))]
 reference=original(*pyramid);apply_tied_mrt(model,passes=1);observed=model.transformer(*pyramid);assert torch.equal(reference,observed)
 model.transformer.tied_mrt_passes=2;calls=[]
 def record(module,inputs,output):calls.append((inputs,output))
 handle=model.transformer.uformer_list[0].register_forward_hook(record);out=model.transformer(*pyramid);handle.remove()
 assert len(calls)==2 and len(calls[0][1])==4
 assert all(calls[1][0][i] is calls[0][1][i] for i in range(4));assert out.shape==reference.shape
 out.square().mean().backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0 for p in pyramid)
 assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items());h=hashlib.sha256()
 for n,v in sorted(model.state_dict().items()):h.update(n.encode());h.update(v.numpy().tobytes())
 assert h.hexdigest()=='4b450134aca35e055f6e139861327feb3a2ea11983e804603e685b6d860918db'
 print('PASS exactonepass,all4tensorrelay,bothpassesfinitegradients,unchangedarchivedparameters/hash')
if __name__=='__main__':main()
