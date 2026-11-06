"""Tiny CPU-only parameter, equivalence, reset and gradient checks."""
import torch
from spikingjelly.activation_based import functional
import run_bridge as run
from current_bridge import install_current_bridge


def tree_equal(a, b):
    if isinstance(a, tuple):
        return all(tree_equal(x, y) for x, y in zip(a, b))
    return torch.equal(a, b)


def cell_run(cell, seed, scale):
    torch.manual_seed(seed)
    side = 4
    h = torch.randn(1, 128, side, side, requires_grad=True)
    contexts = [torch.randn_like(h, requires_grad=True) for _ in range(3)]
    inputs = [torch.randn(1, cell.input_dim, side, side, requires_grad=True)]
    s = torch.randn(1, 128, 2, 2) if scale != 'snn32' else 0.
    values=[]
    for _ in range(2):
        result=cell(s,h,*contexts,*inputs)
        voltage=cell.snn.v
        values.append((result,voltage))
        h=result[0] if isinstance(result,tuple) else result
    loss=sum(value[1].square().mean() for value in values)
    loss.backward()
    grads={name:parameter.grad.clone() for name,parameter in cell.named_parameters() if parameter.grad is not None}
    return values,grads,inputs[0].grad


for seed in (1234,2345,3456):
    original=run.base_build('original',seed,False)
    low=run.build_model('original_low_lr',seed,False)
    bridge=run.build_model('original_current_bridge',seed,False)
    zero=install_current_bridge(run.base_build('original',seed,False),strength=0.)
    for model in (low,bridge,zero):
        assert model.state_dict().keys()==original.state_dict().keys()
        assert all(torch.equal(v,model.state_dict()[k]) for k,v in original.state_dict().items())
        assert sum(p.numel() for p in model.parameters())==12148246
    for scale in ('snn32','snn16','snn08'):
        a=getattr(original.update_block,scale);b=getattr(zero.update_block,scale)
        av,ag,ax=cell_run(a,seed,scale);bv,bg,bx=cell_run(b,seed,scale)
        assert tree_equal(tuple(av),tuple(bv)),scale
        assert ag.keys()==bg.keys() and all(torch.equal(v,bg[k]) for k,v in ag.items()),scale
        assert torch.equal(ax,bx),scale
        cell=getattr(bridge.update_block,scale)
        first,grad,inp_grad=cell_run(cell,seed,scale)
        assert torch.isfinite(inp_grad).all() and inp_grad.abs().sum()>0
        assert torch.isfinite(grad['convq.weight']).all() and grad['convq.weight'].abs().sum()>0
        assert not hasattr(cell.snn,'_bridge_current')
        functional.reset_net(cell)
        cell.zero_grad(set_to_none=True)
        second,_,_=cell_run(cell,seed,scale)
        assert tree_equal(tuple(first),tuple(second)),scale
    print(seed,'identical12148246 initialparameters;zero-bridge forward/backward exact all3scales;active bridge gradient/reset clean')
print('CPU smoke only; no dataset, scientific evaluation or GPU workload')
