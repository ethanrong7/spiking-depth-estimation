"""Opt-in native SpikeStereoNet with residual epipolar descriptor attention.

Static frame descriptors are encoded for four INTERNAL neuron steps. These steps
are not sensor time: the unchanged CNN consumes all 50 sensor bins as channels.
Q/K/V and FFN hidden codes use signed LIF spikes; projections, LayerNorm, softmax,
rate averaging, residual addition and native correlation/refinement remain analog.
"""
import math
import torch
from torch import nn
from core.spikestereonet import SpikeStereoNet


class Threshold(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        return grad / (1 + x.abs()).square()


def signed_rate(current, steps=4, leak=.9):
    """Signed spikes {-1,0,1}, soft reset and full BPTT; state local to call."""
    if steps < 1:
        raise ValueError('positive internal coding steps required')
    voltage = torch.zeros_like(current)
    total = torch.zeros_like(current)
    for _ in range(steps):
        voltage = leak * voltage + current
        spike = Threshold.apply(voltage - 1) - Threshold.apply(-voltage - 1)
        voltage = voltage - spike
        total = total + spike
    return total / steps


def epipolar_attention(q, k, value, heads):
    """BHWC codes; both horizontal signs allowed, no cross-row interaction."""
    b, h, w, c = q.shape
    def split(x):
        return x.reshape(b, h, w, heads, c // heads).transpose(2, 3)
    q, k, value = map(split, (q, k, value))
    weights = ((q @ k.transpose(-1, -2)) / math.sqrt(c // heads)).softmax(-1)
    mixed = (weights @ value).transpose(2, 3).reshape(b, h, w, c)
    return mixed, weights


class EpipolarSpikingDescriptor(nn.Module):
    def __init__(self, channels=256, width=64, heads=4, steps=4):
        super().__init__()
        if heads < 1 or width < 1 or width % heads or steps < 1:
            raise ValueError('invalid coding/head geometry')
        self.heads, self.steps = heads, steps
        self.norm = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, 3 * width, bias=False)
        self.output = nn.Linear(width, channels, bias=False)
        self.ffn_norm = nn.LayerNorm(channels)
        self.ffn_in = nn.Linear(channels, 2 * width)
        self.ffn_out = nn.Linear(2 * width, channels, bias=False)
        # Exact identity, but these output projections receive gradients at step1.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.ffn_out.weight)

    def codes(self, x):
        return signed_rate(self.qkv(self.norm(x)), self.steps).chunk(3, -1)

    def forward(self, left, right):
        if left.shape != right.shape or left.ndim != 4:
            raise ValueError('equal BCHW descriptor maps required')
        l, r = (x.permute(0, 2, 3, 1) for x in (left, right))
        ql, kl, vl = self.codes(l)
        qr, kr, vr = self.codes(r)
        ml, _ = epipolar_attention(ql, kr, vr, self.heads)
        mr, _ = epipolar_attention(qr, kl, vl, self.heads)
        def residual(x, mixed):
            x = x + self.output(mixed)
            x = x + self.ffn_out(signed_rate(self.ffn_in(self.ffn_norm(x)), self.steps))
            return x.permute(0, 3, 1, 2).contiguous()
        return residual(l, ml), residual(r, mr)


class StereoDescriptorEncoder(nn.Module):
    def __init__(self, encoder, descriptor):
        super().__init__()
        self.encoder, self.descriptor = encoder, descriptor

    def forward(self, images):
        if not isinstance(images, (list, tuple)) or len(images) != 2:
            raise ValueError('paired feature encoding required')
        return self.descriptor(*self.encoder(images))


class EpipolarSpikingStereoNet(SpikeStereoNet):
    """Native forward unchanged; only feature maps entering correlation change."""
    def __init__(self, args):
        if args.shared_backbone or any(getattr(args, key, 'cnn') != 'cnn' for key in ('feature_encoder', 'context_encoder')):
            raise ValueError('this variant retains separate native CNN encoders')
        super().__init__(args)
        # Isolate added initialization from subsequent native data RNG consumers.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(getattr(args, 'descriptor_seed', 2011))
            descriptor = EpipolarSpikingDescriptor(
                width=getattr(args, 'descriptor_width', 64),
                heads=getattr(args, 'descriptor_heads', 4),
                steps=getattr(args, 'descriptor_steps', 4))
        self.fnet = StereoDescriptorEncoder(self.fnet, descriptor)

    def load_native_state_dict(self, state):
        """Load unwrapped original model weights; only new descriptor keys missing."""
        remapped = {('fnet.encoder.' + k[5:] if k.startswith('fnet.') else k): v for k, v in state.items()}
        result = self.load_state_dict(remapped, strict=False)
        if result.unexpected_keys or any(not k.startswith('fnet.descriptor.') for k in result.missing_keys):
            raise ValueError('incomplete or incompatible native checkpoint')
        return result
