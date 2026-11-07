"""Independent forward/backward controls for native placement B.

The diagonal cells execute the original implementations. Crossed cells specify
first-order surrogate derivatives, not derivatives of their forward functions.
No learned tensors, attention operations or state-dict keys are added.
"""
import torch
from torch.autograd.function import once_differentiable

from core.epipolar_spiking_stereo import signed_rate
from core.spiking_attention_placement import Codes, DescriptorAttention

VARIANTS = {
    'linear-identity': ('linear', 'identity'),
    'signed-surrogate': ('signed', 'surrogate'),
    'linear-surrogate': ('linear', 'surrogate'),
    'signed-identity': ('signed', 'identity'),
}


def signed_value_and_derivative(current):
    """Exact T4 values and local surrogate-BPTT Jacobian, including soft reset.

Each current is independent, so the local Jacobian is diagonal. This is not
the downstream attention Jacobian, which ordinary autograd still computes.
"""
    with torch.no_grad():
        voltage = torch.zeros_like(current)
        dv = torch.zeros_like(current)
        total = torch.zeros_like(current)
        derivative = torch.zeros_like(current)
        for _ in range(4):
            pre = .9 * voltage + current
            dpre = .9 * dv + 1
            spike = (pre >= 1).to(current.dtype) - (pre <= -1).to(current.dtype)
            slope = (1 + (pre - 1).abs()).square().reciprocal()
            slope = slope + (1 + (-pre - 1).abs()).square().reciprocal()
            ds = slope * dpre
            voltage = pre - spike
            dv = dpre - ds
            total = total + spike
            derivative = derivative + ds
        return total / 4, derivative / 4


class _CrossedCode(torch.autograd.Function):
    @staticmethod
    def forward(ctx, current, forward_rule, backward_rule):
        signed, derivative = signed_value_and_derivative(current)
        ctx.backward_rule = backward_rule
        if backward_rule == 'surrogate':
            ctx.save_for_backward(derivative)
        return signed if forward_rule == 'signed' else current.clone()

    @staticmethod
    @once_differentiable
    def backward(ctx, incoming):
        if ctx.backward_rule == 'surrogate':
            incoming = incoming * ctx.saved_tensors[0]
        return incoming, None, None


def controlled_code(current, forward_rule, backward_rule):
    if (forward_rule, backward_rule) not in VARIANTS.values():
        raise ValueError((forward_rule, backward_rule))
    if current.dtype not in (torch.float32, torch.float64):
        raise TypeError('This protocol validates FP32/FP64 only; no mixed precision.')
    if forward_rule == 'linear' and backward_rule == 'identity':
        return current
    if forward_rule == 'signed' and backward_rule == 'surrogate':
        return signed_rate(current, 4)
    return _CrossedCode.apply(current, forward_rule, backward_rule)


class ControlledCodes(Codes):
    def forward(self, x):
        z = self.norm(x)
        return tuple(controlled_code(layer(z), self.forward_rule, self.backward_rule)
                     for layer in (self.q, self.k, self.v))


def configure_codes(codes, variant):
    forward_rule, backward_rule = VARIANTS[variant]
    codes.__class__ = ControlledCodes
    codes.forward_rule = forward_rule
    codes.backward_rule = backward_rule
    return codes


def install(model, arm, *, variant, seed):
    if arm != 'B':
        raise ValueError('This registered mechanism experiment is placement B only.')
    if variant not in VARIANTS:
        raise ValueError(variant)
    # 1234 -> 2017 reproduces the historical module exactly; paired seed 2345
    # also varies branch initialization, using the established seed+783 rule.
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed + 783)
        model.placement = DescriptorAttention()
    configure_codes(model.placement.codes, variant)
    model.fnet.register_forward_hook(lambda module, args, pair: model.placement(*pair))
    return model
