"""Torch/NPU LIF port of the pinned authors' SpikingJelly clock-driven defaults.

Full-sequence local state avoids cross-example leakage and recomputation side
 effects. Forward hard reset is unchanged; reset derivatives follow detach_reset.
"""
import torch
from torch import nn


class SigmoidSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        s = torch.sigmoid(4.0 * x)
        return grad * (1.0 - s) * s * 4.0


class MultiStepLIFNode(nn.Module):
    def __init__(self, tau=2., decay_input=True, v_threshold=1., v_reset=0.,
                 surrogate_function=None, detach_reset=False, backend='torch', **kwargs):
        super().__init__()
        if surrogate_function is not None or kwargs:
            raise ValueError('Unsupported upstream neuron option; do not silently approximate it')
        self.tau, self.decay_input = tau, decay_input
        self.v_threshold, self.v_reset = v_threshold, v_reset
        self.detach_reset = detach_reset
        self.backend = 'torch'
        self.collect_activity = False
        self.activity = None

    def reset(self):
        self.activity = None

    def forward(self, x):
        v = torch.zeros_like(x[0]) + (self.v_reset or 0.)
        outputs = []
        for current in x.unbind(0):
            base = self.v_reset or 0.
            if self.decay_input:
                v = v + (current - (v - base)) / self.tau
            else:
                v = v - (v - base) / self.tau + current
            spike = SigmoidSpike.apply(v - self.v_threshold)
            reset_spike = spike.detach() if self.detach_reset else spike
            v = v - reset_spike * self.v_threshold if self.v_reset is None else (1 - reset_spike) * v + reset_spike * self.v_reset
            outputs.append(spike)
        result = torch.stack(outputs)
        if self.collect_activity:
            self.activity = result.detach().mean()
        return result


class DropPath(nn.Identity):
    def __init__(self, drop_prob=0.):
        if drop_prob != 0:
            raise ValueError('This study fixes stochastic depth to zero')
        super().__init__()
