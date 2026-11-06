"""Hybrid spatial attention updater; simulation steps are disparity refinements.

Q/K/V and output LIF states live only within a stereo forward. Attention uses
overlapping 3x3 spatial neighborhoods, never sensor bins or cross-view tokens.
Convolutional projections, pointwise FFN, residuals and decoder remain analog.
"""
import math
from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F


class _Spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        return grad / (1 + (math.pi * x).square())


class RefinementLIF(nn.Module):
    """Soft-reset LIF with full gradients through all refinement steps."""
    def __init__(self, decay=0.5, threshold=0.5):
        super().__init__()
        self.decay, self.threshold = decay, threshold
        self.membrane = None

    def reset(self):
        self.membrane = None

    def forward(self, current):
        previous = 0.0 if self.membrane is None else self.membrane
        voltage = self.decay * previous + current
        spike = _Spike.apply(voltage - self.threshold)
        self.membrane = voltage - self.threshold * spike
        return spike, self.membrane


class LocalAttentionRefinement(nn.Module):
    def __init__(self, hidden_dim, input_dim, mode="spiking", heads=4,
                 head_dim=8, window=3, spatial_mixing=True,
                 normalize_residual_current=False):
        super().__init__()
        if mode not in ("spiking", "analog"):
            raise ValueError("mode must be spiking or analog")
        if window < 1 or window % 2 != 1:
            raise ValueError("window must be positive and odd")
        self.mode, self.heads, self.head_dim = mode, heads, head_dim
        self.window, self.spatial_mixing = window, spatial_mixing
        self.attention_enabled = True
        self.normalize_residual_current = normalize_residual_current
        width = heads * head_dim
        self.input_proj = nn.Conv2d(hidden_dim + input_dim + 3 * hidden_dim,
                                    hidden_dim, 1)
        self.norm = nn.GroupNorm(1, hidden_dim)
        self.qkv = nn.Conv2d(hidden_dim, 3 * width, 1)
        self.output_proj = nn.Conv2d(width, hidden_dim, 1)
        self.ffn_norm = nn.GroupNorm(1, hidden_dim)
        self.ffn = nn.Sequential(nn.Conv2d(hidden_dim, 2 * hidden_dim, 1),
                                 nn.GELU(), nn.Conv2d(2 * hidden_dim, hidden_dim, 1))
        self.attention_gate = nn.Parameter(torch.tensor(math.log(0.1 / 0.9)))
        self.ffn_gate = nn.Parameter(torch.tensor(math.log(0.1 / 0.9)))
        self.q_lif, self.k_lif, self.v_lif = (RefinementLIF() for _ in range(3))
        self.output_lif = RefinementLIF()
        self._diagnostics = []

    def reset_state(self, clear_diagnostics=True):
        for lif in (self.q_lif, self.k_lif, self.v_lif, self.output_lif):
            lif.reset()
        if clear_diagnostics:
            self._diagnostics = []

    def diagnostics(self):
        if not self._diagnostics:
            return {}
        values = torch.stack(self._diagnostics).mean(0).cpu().tolist()
        result = dict(zip(("q_rate", "k_rate", "v_rate", "output_rate",
                           "nonzero_attention_fraction", "output_readout_rms",
                           "output_above_threshold_fraction", "output_voltage_excursion_proxy_fraction",
                           "output_surrogate_derivative_mean"), values))
        if self.mode == "spiking":
            result["output_post_reset_voltage_rms"] = result["output_readout_rms"]
            result["output_post_reset_above_threshold_fraction"] = result["output_above_threshold_fraction"]
        else:
            for name in ("output_above_threshold_fraction", "output_voltage_excursion_proxy_fraction",
                         "output_surrogate_derivative_mean"):
                result.pop(name)
        result["steps"] = len(self._diagnostics)
        result["attention_gate"] = self.attention_gate.detach().sigmoid().item()
        result["ffn_gate"] = self.ffn_gate.detach().sigmoid().item()
        for name, parameter in (("qkv", self.qkv.weight),
                                ("output_proj", self.output_proj.weight)):
            if parameter.grad is not None:
                result[name + "_grad_norm"] = parameter.grad.detach().norm().item()
        if self.qkv.weight.grad is not None:
            for name, gradient in zip(("q", "k", "v"), self.qkv.weight.grad.chunk(3)):
                result[name + "_grad_norm"] = gradient.detach().norm().item()
        return result

    def _attention(self, q, k, v):
        b, _, height, width = q.shape
        n, d, neighbors = self.heads, self.head_dim, self.window ** 2
        q = q.reshape(b, n, d, 1, height * width)
        def patches(x):
            return F.unfold(x, self.window, padding=self.window // 2).reshape(
                b, n, d, neighbors, height * width)
        keys, values = patches(k), patches(v)
        valid = F.unfold(q.new_ones(1, 1, height, width), self.window,
                         padding=self.window // 2).bool().reshape(1, 1, neighbors, -1)
        if not self.spatial_mixing:
            center = torch.arange(neighbors, device=q.device) == neighbors // 2
            valid = valid & center.reshape(1, 1, -1, 1)
        scores = (q * keys).sum(2) / math.sqrt(d)
        if self.mode == "spiking":
            # Nonnegative binary coincidence, bounded mass, zero for silent Q/K.
            scores = scores * valid
            weights = scores / (1 + scores.sum(2, keepdim=True))
        else:
            weights = scores.masked_fill(~valid, -torch.inf).softmax(2)
        mixed = (weights.unsqueeze(2) * values).sum(3)
        return mixed.reshape(b, n * d, height, width), weights

    def forward(self, hidden, context, *inputs):
        raw_current = self.input_proj(torch.cat((hidden, *context, *inputs), dim=1))
        normalized = self.norm(raw_current)
        q, k, v = self.qkv(normalized).chunk(3, dim=1)
        current = normalized if self.normalize_residual_current else raw_current
        if self.mode == "spiking":
            q, _ = self.q_lif(q)
            k, _ = self.k_lif(k)
            v, _ = self.v_lif(v)
        mixed, weights = self._attention(q, k, v)
        attention_scale = self.attention_gate.sigmoid() if self.attention_enabled else 0.0
        readout = current + attention_scale * self.output_proj(mixed)
        readout = readout + self.ffn_gate.sigmoid() * self.ffn(self.ffn_norm(readout))
        if self.mode == "spiking":
            output, readout = self.output_lif(readout)
        else:
            output = torch.sigmoid(readout)
        pre_reset = readout.detach() + self.output_lif.threshold * output.detach()
        derivative = (1 + (math.pi * (pre_reset - self.output_lif.threshold)).square()).reciprocal().mean()
        self._diagnostics.append(torch.stack((q.detach().mean(), k.detach().mean(),
            v.detach().mean(), output.detach().mean(), (weights.detach() > 0).float().mean(),
            readout.detach().square().mean().sqrt(), (readout.detach() >= self.output_lif.threshold).float().mean(),
            (readout.detach().abs() > 5 * self.output_lif.threshold).float().mean(), derivative)))
        return output, readout


class SpikingAttentionMultiUpdateBlock(nn.Module):
    """Drop-in fixed-three-scale interface for SpikeStereoNet's refinement loop."""
    def __init__(self, args, hidden_dims, mode="spiking", spatial_mixing=True,
                 heads=4, head_dim=8, window=3, normalize_residual_current=False):
        super().__init__()
        from core.update import BasicMotionEncoder, DispHead
        if args.n_snn_layers != 3 or len(set(hidden_dims)) != 1:
            raise ValueError("experiment requires three equally wide recurrent scales")
        self.args = args
        hidden = hidden_dims[0]
        self.encoder = BasicMotionEncoder(args)
        self.cells = nn.ModuleList([
            LocalAttentionRefinement(hidden, input_dim, mode, heads, head_dim,
                                     window, spatial_mixing, normalize_residual_current)
            for input_dim in (128 + hidden, 2 * hidden, hidden)])
        self.disp_head = DispHead(hidden, hidden_dim=256, output_dim=2)
        factor = 2 ** args.n_downsample
        self.mask = nn.Sequential(nn.Conv2d(hidden, 256, 3, padding=1),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(256, factor ** 2 * 9, 1))

    def reset_state(self, clear_diagnostics=True):
        for cell in self.cells:
            cell.reset_state(clear_diagnostics)

    def diagnostics(self):
        return {f"scale_{i}": cell.diagnostics() for i, cell in enumerate(self.cells)}

    @contextmanager
    def attention_intervention(self, center_only=False, disable_attention=False):
        """Reversible inference intervention on a trained model, no weight edits."""
        if self.training:
            raise ValueError("attention interventions require eval mode")
        previous = [(cell.spatial_mixing, cell.attention_enabled) for cell in self.cells]
        try:
            for cell in self.cells:
                if center_only:
                    cell.spatial_mixing = False
                if disable_attention:
                    cell.attention_enabled = False
            yield
        finally:
            for cell, (mixing, enabled) in zip(self.cells, previous):
                cell.spatial_mixing, cell.attention_enabled = mixing, enabled

    def forward(self, net, inp, corr=None, disp=None, iter08=True, iter16=True,
                iter32=True, update=True):
        def pool(x):
            return F.avg_pool2d(x, 3, stride=2, padding=1)
        def resize(x, target):
            return F.interpolate(x, target.shape[-2:], mode="bilinear", align_corners=True)
        if iter32:
            net[2], _ = self.cells[2](net[2], inp[2], pool(net[1]))
        if iter16:
            net[1], _ = self.cells[1](net[1], inp[1], pool(net[0]), resize(net[2], net[1]))
        if iter08:
            net[0], readout = self.cells[0](net[0], inp[0], self.encoder(disp, corr),
                                           resize(net[1], net[0]))
        if not update:
            return net
        if not iter08:
            raise ValueError("decoder update requires a fine-scale update")
        return net, readout, 0.25 * self.mask(readout), self.disp_head(readout)
