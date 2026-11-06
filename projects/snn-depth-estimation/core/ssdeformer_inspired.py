"""Compact SSDEFormer-inspired ANN control; not official or faithful SSDEFormer.

Public source: https://doi.org/10.1109/ICME52920.2022.9859975
Choices absent/different in paper: width64; normalized cumulative endpoints across
all50 sensor bins; quarter-resolution row attention and bilinear upsampling;
explicit nonnegative disparity support; +/-3 full-resolution-pixel local readout.
The sensor bins are channels, not neural simulation timesteps. See prior-art.md.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class CumulativeSpikeCode(nn.Module):
    def __init__(self, bins=50, channels=32):
        super().__init__()
        endpoints = torch.linspace(1, bins, channels).round().long()
        self.bins = bins
        self.register_buffer("endpoints", endpoints)

    def forward(self, spikes):
        if spikes.ndim != 4 or spikes.shape[1] != self.bins:
            raise ValueError(f"expected [B,{self.bins},H,W] sensor bins")
        cumulative = spikes.float().cumsum(dim=1)
        return cumulative[:, self.endpoints - 1] / self.endpoints[None, :, None, None]


def _conv(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1),
                         nn.GroupNorm(8, cout), nn.GELU())


class PyramidFeatures(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.quarter = nn.Sequential(_conv(32, width // 2, 2), _conv(width // 2, width, 2))
        self.eighth = _conv(width, width, 2)
        self.sixteenth = _conv(width, width, 2)
        self.fuse = nn.Sequential(_conv(width * 3, width), nn.Conv2d(width, width, 1))

    def forward(self, x):
        q = self.quarter(x)
        e = self.eighth(q)
        s = self.sixteenth(e)
        return self.fuse(torch.cat([q, F.interpolate(e, size=q.shape[-2:], mode="bilinear", align_corners=False),
                                   F.interpolate(s, size=q.shape[-2:], mode="bilinear", align_corners=False)], dim=1))


class RowAttentionLayer(nn.Module):
    """Shared-view standard softmax attention, LN, residual and GELU FFN."""
    def __init__(self, width, heads, cross):
        super().__init__()
        self.cross = cross
        self.norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=0.0, batch_first=True)
        self.ffnorm = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width))

    def forward(self, left, right):
        l, r = self.norm(left), self.norm(right)
        lv, rv = (r, l) if self.cross else (l, r)
        # Both directions read the same previous-layer state, not an updated left.
        lo = left + self.attention(l, lv, lv, need_weights=False)[0]
        ro = right + self.attention(r, rv, rv, need_weights=False)[0]
        return lo + self.ff(self.ffnorm(lo)), ro + self.ff(self.ffnorm(ro))


def local_disparity_readout(logits, scale=4, radius_pixels=3):
    """Row matching [B,H,W_left,W_right] -> positive disparity in coarse units.

    Select the best valid right position, then linearly interpolate its logits at
    seven positions spaced one full-resolution pixel apart (+/-3px). Softmax
    within this support is differentiable w.r.t. matching scores; peak selection
    itself is discrete. Invalid negative/out-of-image disparities receive no mass.
    """
    width_left, width_right = logits.shape[-2:]
    x = torch.arange(width_left, device=logits.device, dtype=logits.dtype)
    right_x = torch.arange(width_right, device=logits.device, dtype=logits.dtype)
    valid = right_x[None, :] <= x[:, None]
    masked = logits.masked_fill(~valid, float("-inf"))
    peak = masked.argmax(dim=-1).to(logits.dtype)
    offsets = torch.arange(-radius_pixels, radius_pixels + 1, device=logits.device,
                           dtype=logits.dtype) / scale
    locations = peak[..., None] + offsets
    support = (locations >= 0) & (locations <= width_right - 1) & (locations <= x[None, None, :, None])
    lo = locations.floor().long().clamp(0, width_right - 1)
    hi = (lo + 1).clamp(max=width_right - 1)
    fraction = locations - locations.floor()
    scores = logits.gather(-1, lo) * (1 - fraction) + logits.gather(-1, hi) * fraction
    weights = scores.masked_fill(~support, float("-inf")).softmax(dim=-1)
    disparity = ((x[None, None, :, None] - locations) * weights).sum(dim=-1)
    return disparity[:, None]


class SSDEFormerInspired(nn.Module):
    """One-shot compact conventional transformer with SpikeStereoNet API."""
    def __init__(self, args=None, width=64, layers=6, heads=8):
        super().__init__()
        if width % heads or width % 16:
            raise ValueError("width must be divisible by heads and 16")
        self.args = args
        self.code = CumulativeSpikeCode()
        self.features = PyramidFeatures(width)
        self.layers = nn.ModuleList([RowAttentionLayer(width, heads, bool(i % 2)) for i in range(layers)])
        self.match_left = nn.Linear(width, width)
        self.match_right = nn.Linear(width, width)
        self.width = width

    def freeze_bn(self):
        # GroupNorm/LayerNorm have no batch-running statistics.
        return None

    @staticmethod
    def _position(width, channels, device, dtype):
        x = torch.arange(width, device=device, dtype=torch.float32)[:, None]
        frequency = torch.exp(torch.arange(0, channels, 2, device=device).float() * (-math.log(10000.0) / channels))
        position = torch.zeros(width, channels, device=device)
        position[:, 0::2] = torch.sin(x * frequency)
        position[:, 1::2] = torch.cos(x * frequency)
        return position.to(dtype)[None]

    def forward(self, spike1, spike2, iters=16, disp_init=None, test_mode=False):
        if spike1.shape != spike2.shape:
            raise ValueError("left/right sensor shapes must match")
        height, width = spike1.shape[-2:]
        if height % 16 or width % 16:
            raise ValueError("pad sensor inputs to multiples of 16 before forwarding")
        if disp_init is not None:
            raise ValueError("one-shot inspired control does not accept a refinement initialization")
        left, right = self.features(self.code(spike1)), self.features(self.code(spike2))
        b, c, h, w = left.shape
        left = left.permute(0, 2, 3, 1).reshape(b * h, w, c)
        right = right.permute(0, 2, 3, 1).reshape(b * h, w, c)
        position = self._position(w, c, left.device, left.dtype)
        left, right = left + position, right + position
        for layer in self.layers:
            left, right = layer(left, right)
        logits = (self.match_left(left) @ self.match_right(right).transpose(-1, -2)) / math.sqrt(c)
        logits = logits.reshape(b, h, w, w)
        disparity = local_disparity_readout(logits.float())
        low_flow = -disparity
        prediction = F.interpolate(low_flow, size=(height, width), mode="bilinear", align_corners=False) * 4
        # Exactly one prediction: no duplicated refinement outputs for loss weighting.
        if test_mode:
            return low_flow, prediction, []
        return [prediction], [], []
