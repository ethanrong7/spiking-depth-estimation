"""Isolated scalar activation controls; no model installation or training.

First-order derivatives only. Signed means the original constant-current T4
LIF code with leak .9 and differentiable soft reset, not sensor-time processing.
"""
import torch
from torch.autograd.function import once_differentiable


def signed_value_and_derivative(current):
    """Return hard forward value and original local surrogate-BPTT derivative.

    This computes the specified first-order derivative explicitly; it does not
    build a graph for higher-order differentiation of that derivative.
    """
    with torch.no_grad():
        voltage = torch.zeros_like(current)
        voltage_derivative = torch.zeros_like(current)
        total = torch.zeros_like(current)
        total_derivative = torch.zeros_like(current)
        for _ in range(4):
            pre_reset = .9 * voltage + current
            pre_reset_derivative = .9 * voltage_derivative + 1
            spike = ((pre_reset - 1 >= 0).to(current.dtype) -
                     (-pre_reset - 1 >= 0).to(current.dtype))
            threshold_derivative = ((1 + (pre_reset - 1).abs()).square().reciprocal() +
                                    (1 + (-pre_reset - 1).abs()).square().reciprocal())
            spike_derivative = threshold_derivative * pre_reset_derivative
            voltage = pre_reset - spike
            voltage_derivative = (1 - threshold_derivative) * pre_reset_derivative
            total = total + spike
            total_derivative = total_derivative + spike_derivative
        return total / 4, total_derivative / 4


class _ForwardBackwardControl(torch.autograd.Function):
    @staticmethod
    def forward(ctx, current, forward_rule, backward_rule):
        if forward_rule not in ('signed', 'tanh') or backward_rule not in ('signed', 'tanh'):
            raise ValueError('rules must each be signed or tanh')
        if not current.is_floating_point():
            raise TypeError('floating-point current required')
        signed, signed_derivative = signed_value_and_derivative(current)
        continuous = current.tanh()
        derivative = signed_derivative if backward_rule == 'signed' else 1 - continuous.square()
        ctx.save_for_backward(derivative)
        return signed if forward_rule == 'signed' else continuous

    @staticmethod
    @once_differentiable
    def backward(ctx, incoming):
        (derivative,) = ctx.saved_tensors
        return incoming * derivative, None, None


def controlled_code(current, forward_rule='signed', backward_rule='signed'):
    """Choose forward values and local first-order derivative independently.

    Crossed tanh/signed uses a surrogate derivative for tanh, not its mathematical
    derivative. This prototype is not admitted for mixed precision, checkpoint
    recomputation, model integration, or reuse of archived diagonal training.
    """
    return _ForwardBackwardControl.apply(current, forward_rule, backward_rule)
