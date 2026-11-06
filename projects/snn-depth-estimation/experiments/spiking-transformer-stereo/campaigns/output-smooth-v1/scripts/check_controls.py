"""Tiny CPU initialization, selective wiring, derivative, gradient and reset checks."""
import math
import torch
import run_output_smooth as run


def collect_calls(cell, hidden, context, extra):
    captured={name:[] for name in ['q','k','v','output']};handles=[]
    for name, module in zip(captured,[cell.q_lif,cell.k_lif,cell.v_lif,cell.output_lif]):
        def hook(module,inputs,result,name=name):captured[name].append(result[0].detach().clone())
        handles.append(module.register_forward_hook(hook))
    values=[cell(hidden,context,extra) for _ in range(3)]
    for handle in handles:handle.remove()
    return values,captured


def main():
    torch.set_num_threads(1)
    x=torch.tensor([-2.,-.5,0.,.5,2.],dtype=torch.float64,requires_grad=True)
    smooth=.5+(math.pi*x).atan()/math.pi;smooth.sum().backward()
    hard_x=x.detach().clone().requires_grad_();run.HARD_SPIKE.apply(hard_x).sum().backward()
    assert torch.allclose(x.grad,hard_x.grad,atol=1e-14,rtol=1e-14)
    neuron=run.SmoothOutputLIF();currents=torch.tensor([.1,.5,-.3,.8],dtype=torch.float64,requires_grad=True);state=0.;values=[]
    for current in currents:
        voltage=.5*state+current;signal=.5+(math.pi*(voltage-.5)).atan()/math.pi;state=voltage-.5*signal
        actual=neuron(current);assert torch.equal(actual[0],signal) and torch.equal(actual[1],state);values.append(actual[0])
    torch.stack(values).sum().backward();assert torch.isfinite(currents.grad).all() and currents.grad.abs().min()>0
    neuron.reset();assert neuron.membrane is None
    for seed in [1234,2345,3456]:
        hard=run.base_build('spiking',seed,True)
        mixed=run.base_build('spiking',seed,True);rng=torch.get_rng_state().clone();run.install_output_smooth(mixed)
        assert torch.equal(rng,torch.get_rng_state())
        assert run.attention._Spike is run.HARD_SPIKE
        assert hard.state_dict().keys()==mixed.state_dict().keys()
        assert all(torch.equal(v,mixed.state_dict()[n]) for n,v in hard.state_dict().items())
        assert all(p.requires_grad==dict(mixed.named_parameters())[n].requires_grad for n,p in hard.named_parameters())
        assert sum(p.numel() for p in mixed.parameters())==sum(p.numel() for p in mixed.parameters() if p.requires_grad)==8398902
        for a,b in zip(hard.update_block.cells,mixed.update_block.cells):
            assert all(type(n) is run.attention.RefinementLIF for n in [b.q_lif,b.k_lif,b.v_lif])
            assert isinstance(b.output_lif,run.SmoothOutputLIF)
            width=a.input_proj.in_channels-4*128;hidden=torch.randn(1,128,2,2);context=[torch.randn_like(hidden) for _ in range(3)];extra=torch.randn(1,width,2,2)
            av,ac=collect_calls(a,hidden,context,extra);bv,bc=collect_calls(b,hidden,context,extra)
            for name in ['q','k','v']:
                assert all(torch.equal(x,y) for x,y in zip(ac[name],bc[name]))
                assert all(((v==0)|(v==1)).all() for v in bc[name])
            assert all(((v>0)&(v<1)).all() for v in bc['output'])
            loss=sum(s.square().mean()+v.square().mean() for s,v in bv);loss.backward()
            for module in [b.input_proj,b.qkv,b.output_proj,b.ffn[0]]:
                assert module.weight.grad is not None and torch.isfinite(module.weight.grad).all() and module.weight.grad.abs().sum()>0
            diagnostics=b.diagnostics();assert all(k in diagnostics for k in ['q_rate','k_rate','v_rate','output_activation_mean','output_signal_derivative_mean']) and 'output_rate' not in diagnostics
            b.reset_state();again,_=collect_calls(b,hidden,context,extra)
            assert all(torch.equal(x,y) for pair,repeat in zip(bv,again) for x,y in zip(pair,repeat))
            b.reset_state();assert all(n.membrane is None for n in [b.q_lif,b.k_lif,b.v_lif,b.output_lif])
        print(seed,'exact8398902initialparameters/buffers+RNG;QKVhardbinary/unmodified;outputcontinuous;all3scalegradient/reset/labels passed')
    print('CPU smoke only; no dataset/model evaluation/training experiment/GPU workload')


if __name__=='__main__':main()
