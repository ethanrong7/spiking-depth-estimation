"""Tiny CPU checks for direct signed flow and named evidence interpolation."""
import torch,hashlib
from types import SimpleNamespace
from core.original_coordinate_transformer import OriginalCoordinateHead,install_coordinate_head
from core.spikestereonet import SpikeStereoNet

def args():return SimpleNamespace(hidden_dims=[128]*3,context_norm='batch',shared_backbone=False,n_snn_layers=3,n_downsample=2,corr_levels=4,corr_radius=4,corr_implementation='reg',mixed_precision=False,feature_encoder='cnn',context_encoder='cnn',matched_initialization_seed=1234,slow_fast_gru=False)
def main():
    torch.manual_seed(2017);head=OriginalCoordinateHead();head.use_checkpoint=False;head.set_native_geometry((5,29),(0,0,0,0))
    state=torch.randn(1,128,2,8);corr=torch.randn(1,36,2,8);flow=torch.zeros(1,2,2,8);flow[:,0]=.3
    for forced in (-1,1,'uniform'):
        head.override=forced;delta=head.compute(state,corr,flow);assert torch.equal(delta[:,1],torch.zeros_like(delta[:,1]));assert torch.allclose(delta[:,0],torch.full_like(delta[:,0],0 if forced=='uniform' else forced),atol=1e-6)
    head.override=None
    evidence,support,visible,_=head.evidence(corr,flow)
    mirrored_corr=corr.reshape(1,4,9,2,8).flip(2).flip(-1).reshape_as(corr);mirrored_flow=-flow.flip(-1)
    ev2,su2,vis2,_=head.evidence(mirrored_corr,mirrored_flow)
    assert torch.allclose(ev2,evidence.flip(2).flip(3),atol=1e-6);assert torch.equal(vis2,visible.flip(2).flip(3))
    # Outside-view proposals still have finite nonzero logit gradients.
    captured=[]
    def record(module,inputs,output):output.retain_grad();captured.append(output)
    handle=head.score.register_forward_hook(record);flow[:,0]=-1000;state.requires_grad_();delta=head.compute(state,corr,flow);delta[:,0].sum().backward();handle.remove()
    assert torch.isfinite(captured[0].grad).all() and captured[0].grad.abs().sum()>0
    assert head.q.weight.grad.norm()>0 and head.k.weight.grad.norm()>0 and head.score.weight.grad.norm()>0
    import copy
    plain=OriginalCoordinateHead();checked=copy.deepcopy(plain);plain.use_checkpoint=False;checked.use_checkpoint=True
    x=torch.randn(1,128,2,8);c=torch.randn(1,36,2,8);f=torch.zeros(1,2,2,8)
    for m in (plain,checked):
        m.bound=(c,f);m(x.clone().requires_grad_()).square().sum().backward()
    assert all(torch.allclose(v.grad,dict(checked.named_parameters())[k].grad,atol=1e-6,rtol=1e-5) for k,v in plain.named_parameters())
    torch.manual_seed(1234);a=SpikeStereoNet(args());before={k:v.clone() for k,v in a.state_dict().items()};h=hashlib.sha256()
    for n,v in sorted(before.items()):h.update(n.encode());h.update(v.numpy().tobytes())
    assert h.hexdigest()=='8ceac15b09ce188f9ee52ad5a8eca8e68727cbbb2887a012874d7246dc6495ca'
    install_coordinate_head(a,True);torch.manual_seed(1234);b=install_coordinate_head(SpikeStereoNet(args()),False)
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    assert all(torch.equal(v,before[k]) for k,v in a.state_dict().items() if not k.startswith('update_block.disp_head.'))
    print('PASS forced±1/uniform/verticalzero, mirroredlookup/visibility, OOVlogitgradient,QKgradient, archivedOriginalcommonstate and pairedinitial equality')
if __name__=='__main__':main()
