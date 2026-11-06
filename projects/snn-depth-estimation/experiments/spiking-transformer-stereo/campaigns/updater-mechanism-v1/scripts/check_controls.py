"""Tiny CPU initialization/gradient/reset smoke; no scientific evaluation."""
import importlib.util
from pathlib import Path
import torch
import run_controller as controller

spec = importlib.util.spec_from_file_location('mechanism_probe', Path(__file__).with_name('probe_checkpoints.py'))
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)

for seed in (1234, 2345, 3456):
    full = controller.base_build('spiking', seed, True)
    control = controller.build_model('no_attention', seed, True)
    assert full.state_dict().keys() == control.state_dict().keys()
    assert all(torch.equal(value, control.state_dict()[key]) for key, value in full.state_dict().items())
    total = sum(p.numel() for p in control.parameters())
    active = sum(p.numel() for p in control.parameters() if p.requires_grad)
    assert (total, active) == (8398902, 8349075), (total, active)
    # Tiny cell inputs preserve real channel dimensions and test learned paths.
    cell = control.update_block.cells[2]
    hidden = torch.randn(1, 128, 2, 2)
    context = [torch.randn_like(hidden) for _ in range(3)]
    extra = torch.randn_like(hidden)
    _, voltage = cell(hidden, context, extra)
    voltage.square().mean().backward()
    assert cell.input_proj.weight.grad.abs().sum() > 0
    assert cell.ffn[0].weight.grad.abs().sum() > 0
    assert cell.qkv.weight.grad is None and cell.output_proj.weight.grad is None
    assert cell.attention_gate.grad is None
    control.update_block.reset_state()
    with probe.reset_each_cell_call(control):
        first = cell(hidden, context, extra)
        second = cell(hidden, context, extra)
    assert all(torch.equal(a, b) for a, b in zip(first, second))
    assert not cell._forward_pre_hooks
    print(seed, 'exact initial tensors; total', total, 'active', active, '; tiny gradient/reset passed')
print('CPU smoke only; no dataset, training experiment, or GPU validation')
