"""S9 guarded minimal-current graph with explicit availability and resumable state.

The15 historical core files remain unchanged. Native coordinate detachment is
preserved; no additional hidden/membrane detachment occurs during training.
Eval branches may clone State only under no_grad. No data is read on import.
"""
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

CAMP = Path(__file__).resolve().parents[1]
REPO = CAMP.parents[5]
EXPERIMENT = CAMP.parents[1]
CONTRACT_SHA = "217c7f7d1e0947fab351d2068577421a67c93f1936ba8612e250d58391975d2d"
PREFIXES, HORIZONS = (24, 32, 49), (4, 8, 16)
ACTIONS = tuple(f"p{p}-r{r}" for p in PREFIXES for r in HORIZONS)
FEATURE_NAMES = ("left_rate", "right_rate", "left_zero_fraction", "right_zero_fraction", "left_count_gradient", "right_count_gradient",
                 "disparity_mean", "disparity_std", "increment3to4_absmean", "increment2to3_absmean", "increment3to4_gt1_fraction",
                 "fine_entropy_mean", "fine_entropy_std", "fine_zmargin_mean")


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def guard_sources():
    binding = json.loads((CAMP / "configs/stage9-source-bindings.json").read_text())
    require(binding["scientific_contract_sha256"] == CONTRACT_SHA
            and sha(CAMP / "analysis/stage9-action-value-contract-astra.md") == CONTRACT_SHA, "S9 contract changed")
    require(sha(CAMP / "analysis/stage9-prospective-parity-clarification.md") == binding["parity_addendum_sha256"] == "1fb91653cce57e8c8f4264e2bbe4444cc2f3d02f856cdff837b4aa3e1c36b826", "parity contract changed")
    for name, expected in binding["source_files"].items():
        require(sha(REPO / name) == expected, "historical/native source changed: " + name)
    require(sha(CAMP / "configs/stage9-split-rows.json") == binding["split_sha256"], "certified split changed")
    return binding


BINDINGS = guard_sources()
parent_path = EXPERIMENT / "campaigns/minimal-lif-v1/scripts/run_minimal.py"
sys.path.insert(0, str(parent_path.parent))
spec = importlib.util.spec_from_file_location("s9_historical_minimal", parent_path)
parent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent)
from core.corr import CorrBlock1D
from core.utils.utils import InputPadder


def visible_input(spikes, prefix):
    """CPU masking before any model access; never exposes future values to device."""
    require(prefix in PREFIXES and spikes.device.type == "cpu" and spikes.ndim == 3 and spikes.shape[0] == 50, "invalid visible-input boundary")
    out = torch.zeros_like(spikes)
    out[:prefix+1] = spikes[:prefix+1]
    require(bool(((out == 0) | (out == 1)).all()), "binary sensor input required")
    return out


def digest_state(model, active_only=False):
    items = ((n, p) for n, p in model.named_parameters() if p.requires_grad) if active_only else model.state_dict().items()
    h = hashlib.sha256()
    for name, value in sorted(items):
        h.update(name.encode());h.update(str(value.dtype).encode());h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class AvailabilityAdapter(nn.Conv2d):
    def __init__(self):
        # Conv's otherwise discarded random initialization cannot perturb parent RNG.
        with torch.random.fork_rng(devices=[]):
            super().__init__(100, 50, 1, bias=True)
        with torch.no_grad():
            self.weight.zero_();self.bias.zero_()
            self.weight[:, :50, 0, 0].copy_(torch.eye(50))

    def encode(self, visible, prefix):
        mask = (torch.arange(50, device=visible.device) <= prefix).to(visible.dtype)[None, :, None, None]
        mask = mask.expand_as(visible)
        return self(torch.cat((mask * (2*visible-1), mask), dim=1))


@dataclass
class Features:
    prefix: int
    padder: object
    adapted_left: torch.Tensor
    adapted_right: torch.Tensor
    fine: torch.Tensor
    native_hw: tuple


@dataclass
class State:
    features: Features
    corr: object
    net: list
    context: list
    coords0: torch.Tensor
    coords1: torch.Tensor
    membranes: tuple
    iteration: int = 0


class MinimalAvailability(nn.Module):
    def __init__(self, seed=1234):
        super().__init__()
        self.native = parent.build_model("minimal_lif", seed, True)
        require(not self.native.args.mixed_precision and not self.native.args.shared_backbone
                and not self.native.args.slow_fast_gru and self.native.args.n_snn_layers == 3
                and self.native.args.corr_levels == 4 and self.native.args.corr_radius == 4
                and self.native.args.corr_implementation == "reg", "historical native graph flags changed")
        self.adapter = AvailabilityAdapter()
        counts = (sum(p.numel() for p in self.parameters()), sum(p.numel() for p in self.parameters() if p.requires_grad))
        require(counts == (8403952, 8155594), "S9 active/stored parameter mismatch")

    def freeze_bn(self):
        self.native.freeze_bn()

    def reset(self):
        self.native.update_block.reset_state()

    def features(self, left, right, prefix):
        require(left.ndim == 4 and left.shape == right.shape and left.shape[0] == 1 and left.shape[1] == 50, "physicalB1/full50channels required")
        require(prefix in PREFIXES, "unregistered prefix")
        padder = InputPadder(left.shape, divis_by=32)
        left, right = padder.pad(left, right)
        adapted_left, adapted_right = self.adapter.encode(left, prefix), self.adapter.encode(right, prefix)
        fmap1, fmap2 = self.native.fnet([adapted_left, adapted_right])
        fine = CorrBlock1D.corr(fmap1.float(), fmap2.float())
        return Features(prefix, padder, adapted_left, adapted_right, fine, (padder.ht, padder.wd))

    def initialize(self, features):
        self.reset()
        # Preserve the exact eager native pyramid, including its unused fifth level.
        corr = CorrBlock1D.__new__(CorrBlock1D)
        corr.num_levels, corr.radius = 4, 4
        b, h, w, _, keys = features.fine.shape
        value = features.fine.reshape(b*h*w, 1, 1, keys)
        corr.corr_pyramid = [value]
        for _ in range(corr.num_levels):
            value = F.avg_pool2d(value, [1, 2], stride=[1, 2])
            corr.corr_pyramid.append(value)
        cnet = self.native.cnet(features.adapted_left, num_layers=3)
        net = [torch.tanh(x[0]) for x in cnet]
        context = [torch.relu(x[1]) for x in cnet]
        context = [list(conv(x).split(conv.out_channels//3, dim=1)) for x, conv in zip(context, self.native.context_zqr_convs)]
        coords0, coords1 = self.native.initialize_disp(net[0])
        return State(features, corr, net, context, coords0, coords1, (None, None, None))

    def advance(self, state, endpoint, decode_at=None):
        require(state.iteration <= endpoint <= 16, "invalid continuation horizon")
        decode_at = {endpoint} if decode_at is None else set(decode_at)
        require(all(state.iteration < k <= endpoint for k in decode_at), "decode outside continuation segment")
        for cell, membrane in zip(self.native.update_block.cells, state.membranes):
            cell.output_lif.membrane = membrane
        predictions = {}
        for iteration in range(state.iteration+1, endpoint+1):
            # This is the native coordinate truncation; hidden/membrane histories remain connected.
            state.coords1 = state.coords1.detach()
            corr = state.corr(state.coords1)
            disp = state.coords1-state.coords0
            state.net, _, mask, delta = self.native.update_block(state.net, state.context, corr, disp, iter32=True, iter16=True)
            delta[:, 1] = 0.0
            state.coords1 = state.coords1+delta
            if iteration in decode_at:
                require(mask is not None, "native minimal convex upsampler required")
                prediction = self.native.upsample_disp(state.coords1-state.coords0, mask)[:, :1]
                predictions[iteration] = state.features.padder.unpad(prediction)
            state.iteration = iteration
        state.membranes = tuple(cell.output_lif.membrane for cell in self.native.update_block.cells)
        # State owns graph tensors; mutable module attributes cannot leak across actions/samples.
        self.native.update_block.reset_state(clear_diagnostics=False)
        return predictions

    def direct(self, left, right, prefix, endpoint, decode_at=None):
        state = self.initialize(self.features(left, right, prefix))
        return self.advance(state, endpoint, decode_at), state


def build_model(seed=1234):
    return MinimalAvailability(seed)


def clone_state(state):
    require(not torch.is_grad_enabled(), "detached branch cloning is evaluation-only")
    from copy import deepcopy
    def clone(x):
        if isinstance(x, torch.Tensor):return x.detach().clone()
        if isinstance(x, list):return [clone(v) for v in x]
        if isinstance(x, tuple):return tuple(clone(v) for v in x)
        return deepcopy(x)
    f = state.features
    features = Features(f.prefix, deepcopy(f.padder), clone(f.adapted_left), clone(f.adapted_right), clone(f.fine), f.native_hw)
    corr = CorrBlock1D.__new__(CorrBlock1D)
    corr.num_levels, corr.radius = state.corr.num_levels, state.corr.radius
    corr.corr_pyramid = clone(state.corr.corr_pyramid)
    return State(features, corr, clone(state.net), clone(state.context), clone(state.coords0), clone(state.coords1), clone(state.membranes), state.iteration)


def correlation_zscores(features):
    fine = features.fine[:, :, :, 0, :]
    h, w = features.native_hw
    pad = features.padder._pad
    qy = torch.arange(fine.shape[1], device=fine.device)*4-pad[2]
    qx = torch.arange(fine.shape[2], device=fine.device)*4-pad[0]
    kx = torch.arange(fine.shape[3], device=fine.device)*4-pad[0]
    scores = fine[:, (qy >= 0) & (qy < h)][:, :, (qx >= 0) & (qx < w)][:, :, :, (kx >= 0) & (kx < w)]
    require(scores.shape[-1] >= 2 and scores.numel() > 0, "insufficient native feature centers")
    z = (scores-scores.mean(-1, keepdim=True))/(scores.std(-1, unbiased=False, keepdim=True)+1e-6)
    return z


def correlation_entropy(features):
    z = correlation_zscores(features)
    prob = z.softmax(-1)
    return (-(prob*prob.clamp_min(torch.finfo(prob.dtype).tiny).log()).sum(-1)/torch.log(z.new_tensor(z.shape[-1]))).mean()


def correlation_statistics(features):
    z = correlation_zscores(features)
    prob = z.softmax(-1)
    entropy = -(prob*prob.clamp_min(torch.finfo(prob.dtype).tiny).log()).sum(-1)/torch.log(z.new_tensor(z.shape[-1]))
    top = z.topk(2, dim=-1).values
    return entropy.mean(), entropy.std(unbiased=False), (top[..., 0]-top[..., 1]).mean()


def probe_features(left, right, prefix, predictions, features):
    values, gradients, zero = [], [], []
    for spikes in (left, right):
        rate = spikes[:, :prefix+1].mean(1)
        values.append(spikes[:, :prefix+1].sum(dtype=torch.int64).to(torch.float64)/(spikes.shape[0]*(prefix+1)*spikes.shape[-2]*spikes.shape[-1]));zero.append((rate == 0).float().mean())
        gradients.append(.5*((rate[..., 1:]-rate[..., :-1]).abs().mean()+(rate[..., 1:, :]-rate[..., :-1, :]).abs().mean()))
    d2, d3, d4 = (-predictions[k] for k in (2, 3, 4))
    entropy_mean, entropy_std, margin = correlation_statistics(features)
    return torch.stack((*values, *zero, *gradients, d4.mean(), d4.std(unbiased=False), (d4-d3).abs().mean(),
                        (d3-d2).abs().mean(), ((d4-d3).abs() > 1).float().mean(), entropy_mean, entropy_std, margin))
