"""Tiny CPU-only minimal-LIF wiring, initial tensors, gradient and reset checks."""
import torch
import run_minimal as run

for seed in (1234,2345,3456):
    reference=run.parent.build_model('no_attention',seed,True)
    model=run.build_model('minimal_lif',seed,True)
    assert model.state_dict().keys()==reference.state_dict().keys()
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in reference.state_dict().items())
    counts=(sum(p.numel() for p in model.parameters()),sum(p.numel() for p in model.parameters() if p.requires_grad),sum(p.numel() for p in model.parameters() if not p.requires_grad))
    assert counts==(8398902,8150544,248358),counts
    for cell in model.update_block.cells:
        input_dim=cell.input_proj.in_channels-4*128
        torch.manual_seed(seed)
        hidden=torch.randn(1,128,2,2)
        context=[torch.randn_like(hidden) for _ in range(3)]
        extra=torch.randn(1,input_dim,2,2)
        expected=cell.norm(cell.input_proj(torch.cat((hidden,*context,extra),dim=1)))
        captures=[]
        handle=cell.output_lif.register_forward_pre_hook(lambda module,inputs:captures.append(inputs[0].detach().clone()))
        first=[]
        for _ in range(3):first.append(cell(hidden,context,extra))
        handle.remove()
        assert len(captures)==3 and all(torch.equal(v,expected) for v in captures)
        assert cell.norm.weight.requires_grad and cell.input_proj.weight.requires_grad
        loss=sum(v.square().mean()+s.mean() for s,v in first);loss.backward()
        assert cell.input_proj.weight.grad.abs().sum()>0 and cell.norm.weight.grad.abs().sum()>0
        for name,p in cell.named_parameters():
            if not p.requires_grad:assert p.grad is None,name
        cell.reset_state();cell.zero_grad(set_to_none=True)
        second=[cell(hidden,context,extra) for _ in range(3)]
        assert all(torch.equal(a,b) for x,y in zip(first,second) for a,b in zip(x,y))
        assert cell.q_lif.membrane is None and cell.k_lif.membrane is None and cell.v_lif.membrane is None
        assert cell.diagnostics()['ffn_enabled'] is False
        cell.reset_state()
        assert cell.output_lif.membrane is None
    print(seed,'exactinitialtensors;total/active/dormant',counts,'; r=c all3scales; dormantgradientsabsent;activecurrentgrad/resetpassed')
print('CPU smoke only; no dataset access, inference experiment, or GPU work')
