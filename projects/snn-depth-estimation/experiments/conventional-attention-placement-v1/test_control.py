"""Tiny synthetic mathematical and initialization checks; no dataset/benchmark."""
import argparse, hashlib, importlib.util, json, sys
from pathlib import Path
import torch
from torch.nn import functional as F
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from core.conventional_attention_placement import install,LinearCodes
from core.spiking_attention_placement import CostFilter,DescriptorAttention,install as install_spiking

def digest(state):
    h=hashlib.sha256()
    for n,v in sorted(state.items()):h.update(n.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--require-historical',action='store_true');args=parser.parse_args()
    torch.set_num_threads(1)
    # Identical learned tensors and no coding-range cap or hidden tanh.
    codes=LinearCodes(16);x=torch.randn(2,7,16)
    with torch.no_grad():codes.q.weight.mul_(100)
    q,k,v=codes(x);assert q.abs().max()>1
    for actual,layer in zip((q,k,v),(codes.q,codes.k,codes.v)):torch.testing.assert_close(actual,layer(codes.norm(x)),rtol=0,atol=0)
    for cls,channels in ((CostFilter,1),(DescriptorAttention,256)):
        torch.manual_seed(2017);spike=cls();torch.manual_seed(2017);analog=cls();analog.codes.__class__=LinearCodes
        assert digest(spike.state_dict())==digest(analog.state_dict())
        assert sum(p.numel() for p in analog.parameters())==(913 if channels==1 else 9312)
        if channels==1:
            identity=torch.randn(1,1,3,1,7);torch.testing.assert_close(analog.compute(identity),identity,atol=0,rtol=0)
        else:
            identity=(torch.randn(1,256,2,7),torch.randn(1,256,2,7))
            for actual,original in zip(analog.compute(*identity),identity):torch.testing.assert_close(actual,original,atol=0,rtol=0)
        with torch.no_grad():analog.output.weight.normal_(std=.02)
        if channels==1:
            cost=torch.randn(1,1,3,1,7,requires_grad=True)
            z=cost.reshape(-1,7,1);position=torch.linspace(-1,1,7).view(1,7,1).expand(3,-1,-1)
            q,k,v=analog.codes(analog.input(torch.cat((z,position),-1)))
            i=torch.arange(7);mask=(i[:,None]-i[None,:]).abs()<=2
            expected=z+analog.output(F.scaled_dot_product_attention(q[:,None],k[:,None],v[:,None],attn_mask=mask).squeeze(1))
            actual=analog.compute(cost);torch.testing.assert_close(actual,expected.reshape_as(cost),atol=1e-6,rtol=1e-5)
            actual.sum().backward()
        else:
            left,right=(torch.randn(1,256,2,7,requires_grad=True) for _ in range(2))
            l,r=left.permute(0,2,3,1),right.permute(0,2,3,1);ql,kl,vl=analog.codes(analog.input(l));qr,kr,vr=analog.codes(analog.input(r))
            expected=(l+analog.output(F.scaled_dot_product_attention(ql,kr,vr)),r+analog.output(F.scaled_dot_product_attention(qr,kl,vl)))
            actual=analog.compute(left,right)
            for a,b in zip(actual,expected):torch.testing.assert_close(a,b.permute(0,3,1,2),atol=1e-6,rtol=1e-5)
            sum(a.square().sum() for a in actual).backward()
        for name in ('q','k','v'):
            gradient=getattr(analog.codes,name).weight.grad;assert torch.isfinite(gradient).all() and gradient.norm()>0
    # Construct only (no forward) from frozen scientific dependencies.
    path=PROJECT/'experiments/spiking-transformer-stereo/scripts/run_experiment.py';sys.path.insert(0,str(path.parent))
    spec=importlib.util.spec_from_file_location('driver',path);driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
    expected={'A':'988bf97b1129ba65476bad0332307d301a77d3d86e0a51350e0c6b42c7473853','B':'b37335857fd34f9f5aaa1cf203f906304f7edbd965447ff7dad3953172d47bd4'}
    hashes={}
    for arm in ('A','B'):
        model=install(driver.build_model('original',1234),arm);hashes[arm]=digest(model.state_dict())
        reference=install_spiking(driver.build_model('original',1234),arm);assert hashes[arm]==digest(reference.state_dict());del reference
        if args.require_historical:assert hashes[arm]==expected[arm],(arm,hashes[arm])
        del model
    print(json.dumps(dict(status='PASSED',initial_state_hashes=hashes,checks=['linear_unbounded_QKV','identical_module_initialization_and_parameter_counts','A_masked_SDPA_equivalence','B_bidirectional_SDPA_equivalence','QKV_gradients_open_readout','historical_full_model_initialization'])))
if __name__=='__main__':main()
