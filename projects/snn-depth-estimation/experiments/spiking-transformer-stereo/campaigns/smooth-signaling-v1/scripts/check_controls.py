"""Tiny CPU-only signal/gradient, seeded-tensor and dynamics wiring checks."""
import math
import torch
import run_smooth as control


def main():
    torch.set_num_threads(1)
    x = torch.tensor([-2., -.5, 0., .5, 2.], dtype=torch.float64, requires_grad=True)
    soft = control.SmoothSignal.apply(x)
    soft.sum().backward()
    derivative = 1 / (1 + (math.pi * x.detach()).square())
    assert torch.allclose(x.grad, derivative, atol=1e-14, rtol=1e-14)
    hard_x = x.detach().clone().requires_grad_()
    hard = control.hard_spike.apply(hard_x)
    hard.sum().backward()
    assert torch.allclose(hard_x.grad, x.grad, atol=1e-14, rtol=1e-14)
    assert torch.all((soft > 0) & (soft < 1)) and set(hard.tolist()) == {0., 1.}
    control.attention._Spike = control.SmoothSignal
    neuron = control.attention.RefinementLIF()
    currents = torch.tensor([.1, .5, -.3, .8], dtype=torch.float64, requires_grad=True)
    manual = 0.
    outputs = []
    for current in currents:
        voltage = .5 * manual + current
        expected = control.SmoothSignal.apply(voltage - .5)
        manual = voltage - .5 * expected
        signal, state = neuron(current)
        assert torch.equal(signal, expected) and torch.equal(state, manual)
        outputs.append(signal)
    torch.stack(outputs).sum().backward()
    assert torch.isfinite(currents.grad).all() and (currents.grad.abs() > 0).all()
    neuron.reset()
    assert neuron.membrane is None
    for seed in (1234, 2345, 3456):
        control.attention._Spike = control.hard_spike
        hard_model = control.base_build('spiking', seed, True)
        smooth_model = control.build_model('smooth_coincidence', seed, True)
        hard_parameters, smooth_parameters = dict(hard_model.named_parameters()), dict(smooth_model.named_parameters())
        assert hard_parameters.keys() == smooth_parameters.keys()
        assert all(torch.equal(value, smooth_parameters[name]) and value.requires_grad == smooth_parameters[name].requires_grad for name, value in hard_parameters.items())
        assert hard_model.state_dict().keys() == smooth_model.state_dict().keys()
        assert all(torch.equal(value, smooth_model.state_dict()[name]) for name, value in hard_model.state_dict().items())
        assert sum(p.numel() for p in smooth_model.parameters()) == 8398902
        cell = smooth_model.update_block.cells[2]
        hidden = torch.randn(1, 128, 2, 2)
        context = [torch.randn_like(hidden) for _ in range(3)]
        extra = torch.randn_like(hidden)
        first = cell(hidden, context, extra)
        cell.reset_state()
        second = cell(hidden, context, extra)
        assert all(torch.equal(a, b) for a, b in zip(first, second))
        (second[0].square().mean() + second[1].square().mean()).backward()
        for layer in [cell.input_proj, cell.qkv, cell.output_proj, cell.ffn[0]]:
            assert layer.weight.grad is not None and torch.isfinite(layer.weight.grad).all() and layer.weight.grad.abs().sum() > 0
        diagnostics = cell.diagnostics()
        assert 'q_activation_mean' in diagnostics and 'output_signal_derivative_mean' in diagnostics
        assert not any(key.endswith('_rate') for key in diagnostics)
        assert all(lif.membrane is not None for lif in [cell.q_lif, cell.k_lif, cell.v_lif, cell.output_lif])
        cell.reset_state()
        assert all(lif.membrane is None for lif in [cell.q_lif, cell.k_lif, cell.v_lif, cell.output_lif])
        print(seed, 'all learned/buffer tensors equal at initialization; tiny-cell gradients/reset/diagnostic labels passed')
    print('CPU smoke only; no dataset, model evaluation, training experiment or GPU workload')


if __name__ == '__main__':
    main()
