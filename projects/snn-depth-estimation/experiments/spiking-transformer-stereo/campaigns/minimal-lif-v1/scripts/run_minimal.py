"""Frozen no-attention controller with its complete FFN branch removed."""
import importlib.util
import math
from pathlib import Path
from types import MethodType
import torch

EXPERIMENT = Path(__file__).resolve().parents[3]
parent_path = EXPERIMENT / 'campaigns/updater-mechanism-v1/scripts/run_controller.py'
spec = importlib.util.spec_from_file_location('minimal_parent_controller', parent_path)
parent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent)
driver = parent.driver


def minimal_forward(cell, hidden, context, *inputs):
    raw_current = cell.input_proj(torch.cat((hidden, *context, *inputs), dim=1))
    current = cell.norm(raw_current)
    # The complete projected attention and FFN branches are absent. r=c
    # exactly before the unchanged soft-reset output LIF, with no state detach.
    output, readout = cell.output_lif(current)
    with torch.no_grad():
        pre_reset = readout.detach() + cell.output_lif.threshold * output.detach()
        derivative = (1 + (math.pi * (pre_reset - cell.output_lif.threshold)).square()).reciprocal().mean()
        cell._diagnostics.append(torch.stack((current.detach().square().mean().sqrt(),
            output.detach().mean(), readout.detach().square().mean().sqrt(), derivative)))
    return output, readout


def minimal_diagnostics(cell):
    if not cell._diagnostics:
        return {}
    values = torch.stack(cell._diagnostics).mean(0).cpu().tolist()
    result = dict(zip(('residual_current_rms', 'output_rate', 'output_post_reset_voltage_rms',
                       'output_surrogate_derivative_mean'), values))
    result.update(steps=len(cell._diagnostics), attention_enabled=False, ffn_enabled=False)
    for name, parameter in [('input_proj', cell.input_proj.weight), ('input_norm', cell.norm.weight)]:
        if parameter.grad is not None:
            result[name + '_grad_norm'] = parameter.grad.detach().norm().item()
    return result


def build_model(name, seed, normalize_residual_current=False):
    if name != 'minimal_lif' or not normalize_residual_current:
        raise ValueError('requires registered normalized minimal_lif configuration')
    model = parent.build_model('no_attention', seed, True)
    for cell in model.update_block.cells:
        cell.ffn.requires_grad_(False)
        cell.ffn_norm.requires_grad_(False)
        cell.ffn_gate.requires_grad_(False)
        cell.forward = MethodType(minimal_forward, cell)
        cell.diagnostics = MethodType(minimal_diagnostics, cell)
    return model


driver.build_model = build_model
if __name__ == '__main__':
    driver.main()
