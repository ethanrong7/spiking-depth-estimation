"""Prospective full temporal stereo systems; see architecture-spec-v1.md.

These are new adaptations, not reproductions of STTR, S2M2 or SDSA models.
Every sensor bin is a simulation step. State is local to forward and remains
in the autograd graph, including soft reset and checkpoint recomputation.
All systems have analog normalization, correspondence and dense readout.
"""
from dataclasses import dataclass
import math
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class _Spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, gradient):
        (x,) = ctx.saved_tensors
        return gradient / (1 + x.abs()).square()


class TemporalNeuron(nn.Module):
    """Explicit-state LIF with differentiated reset, or its smooth counterpart."""
    def __init__(self, spiking=True, beta=.9, threshold=1.):
        super().__init__()
        self.spiking, self.beta, self.threshold = spiking, beta, threshold
        self.state_id = None
        self.collect_diagnostics = True

    def forward(self, current, previous, updated, view):
        key = (self.state_id, view)
        pre = self.beta * previous.get(key, 0.) + current
        z = _Spike.apply(pre - self.threshold) if self.spiking else torch.sigmoid(4 * (pre - self.threshold))
        updated[key] = pre - self.threshold * z
        # Detached telemetry does not replace/detach the differentiable state.
        if self.collect_diagnostics:
            activity = z.detach().reshape(z.shape[0], -1, *z.shape[-2:]).mean((0, 2, 3)) if z.ndim >= 4 else z.detach().reshape(-1)
            updated[('activity', self.state_id, view)] = activity
            updated[('voltage', self.state_id, view)] = pre.detach().abs().mean()
        return z


def norm(channels):
    return nn.GroupNorm(math.gcd(8, channels), channels)


class SpikeConv(nn.Module):
    def __init__(self, ci, co, kernel=3, stride=1, spiking=True):
        super().__init__()
        self.conv = nn.Conv2d(ci, co, kernel, stride, kernel // 2, bias=False)
        self.norm = norm(co)
        self.neuron = TemporalNeuron(spiking)

    def forward(self, x, previous, updated, view):
        return self.neuron(self.norm(self.conv(x)), previous, updated, view)


class ResidualUnit(nn.Module):
    def __init__(self, channels, spiking):
        super().__init__()
        self.first = SpikeConv(channels, channels, spiking=spiking)
        self.second = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm = norm(channels)
        self.out = TemporalNeuron(spiking)

    def forward(self, x, previous, updated, view):
        y = self.first(x, previous, updated, view)
        return self.out(x + self.norm(self.second(y)), previous, updated, view)


def _heads(x, heads):
    b, c, h, w = x.shape
    return x.reshape(b, heads, c // heads, h, w)


def epipolar_mask(width, max_disparity, device, reverse=False):
    """Rows are queries, columns keys; positive d=x_left-x_right."""
    q = torch.arange(width, device=device)[:, None]
    k = torch.arange(width, device=device)[None, :]
    disparity = k - q if reverse else q - k
    return (disparity >= 0) & (disparity <= max_disparity)


def normalize_counts(scores, dim, normalization='count_floor_v2', eps=1e-6):
    if normalization == 'count_floor_v2':
        return scores / scores.sum(dim, keepdim=True).clamp_min(1.)
    if normalization == 'epsilon_v1':
        return scores / (scores.sum(dim, keepdim=True) + eps)
    raise ValueError('Unknown attention normalization version')


def row_attention(q, k, v, valid, max_disparity, reverse=False, eps=1e-6, normalization='count_floor_v2'):
    """Query-specific nonnegative spike coincidence normalized on valid keys.

    Computes row matrices only, never all-image 2D matrices. The disparity
    restriction masks score/aggregation support and is included in cost logs.
    q/k/v shapes are B,heads,channels,H,W. valid is B,1,H,W.
    """
    scale = q.shape[2] ** -.5 if normalization == 'epsilon_v1' else 1.
    qh, kh, vh = [x.permute(0, 3, 1, 4, 2) for x in (q, k, v)]
    scores = torch.matmul(qh, kh.transpose(-1, -2)) * scale
    mask = epipolar_mask(q.shape[-1], max_disparity, q.device, reverse)
    key_valid = valid[:, 0, :, None, None, :]
    scores = scores * (mask[None, None, None] & key_valid).to(scores.dtype)
    weights = normalize_counts(scores, -1, normalization, eps)
    result = torch.matmul(weights, vh).permute(0, 2, 4, 1, 3)
    return result * valid[:, None]


def local_attention(q, k, v, valid, eps=1e-6, normalization='count_floor_v2'):
    b, heads, channels, h, w = q.shape
    ku = F.unfold(k.reshape(b, heads * channels, h, w), 3, padding=1)
    vu = F.unfold(v.reshape(b, heads * channels, h, w), 3, padding=1)
    ku = ku.reshape(b, heads, channels, 9, h, w)
    vu = vu.reshape(b, heads, channels, 9, h, w)
    mask = F.unfold(valid.float(), 3, padding=1).reshape(b, 1, 9, h, w)
    scale = channels ** -.5 if normalization == 'epsilon_v1' else 1.
    scores = (q.unsqueeze(3) * ku).sum(2) * scale * mask
    weights = normalize_counts(scores, 2, normalization, eps)
    return (weights.unsqueeze(2) * vu).sum(3) * valid[:, None]


class QKV(nn.Module):
    def __init__(self, channels, spiking):
        super().__init__()
        self.q, self.k, self.v = [SpikeConv(channels, channels, 1, spiking=spiking) for _ in range(3)]

    def forward(self, x, previous, updated, view):
        return tuple(layer(x, previous, updated, view) for layer in (self.q, self.k, self.v))


class AttentionSublayer(nn.Module):
    def __init__(self, channels, kind, spiking, heads=4, attention_normalization='count_floor_v2'):
        super().__init__()
        self.qkv = QKV(channels, spiking)
        self.kind, self.heads = kind, heads
        self.attention_normalization = attention_normalization
        self.project = nn.Conv2d(channels, channels, 1, bias=False)
        self.norm = norm(channels)
        self.output = TemporalNeuron(spiking)
        self.gate = TemporalNeuron(spiking) if kind == 'gate' else None

    def _finish(self, x, mixed, previous, updated, view):
        b, _, _, h, w = mixed.shape
        return self.output(x + self.norm(self.project(mixed.reshape(b, -1, h, w))), previous, updated, view)

    def forward_pair(self, left, right, previous, updated, valid, max_disparity):
        # Both directions use pre-update features and independent view states.
        ql, kl, vl = [_heads(x, self.heads) for x in self.qkv(left, previous, updated, 0)]
        qr, kr, vr = [_heads(x, self.heads) for x in self.qkv(right, previous, updated, 1)]
        if self.kind == 'cross':
            ml = row_attention(ql, kr, vr, valid, max_disparity, normalization=self.attention_normalization)
            mr = row_attention(qr, kl, vl, valid, max_disparity, reverse=True, normalization=self.attention_normalization)
        elif self.kind == 'local':
            ml = local_attention(ql, kl, vl, valid, normalization=self.attention_normalization)
            mr = local_attention(qr, kr, vr, valid, normalization=self.attention_normalization)
        else:
            mixed = []
            count = valid.sum((-2, -1), keepdim=True).clamp_min(1).sqrt()
            for view, (q, k, v) in enumerate(((ql, kl, vl), (qr, kr, vr))):
                gate_current = (k * v * valid[:, None]).sum((-2, -1), keepdim=True) / count[:, None]
                gate = self.gate(gate_current, previous, updated, view)
                mixed.append(q * gate * valid[:, None])
            ml, mr = mixed
        return self._finish(left, ml, previous, updated, 0), self._finish(right, mr, previous, updated, 1)


class FeatureBlock(nn.Module):
    def __init__(self, channels, architecture, spiking, attention_normalization='count_floor_v2'):
        super().__init__()
        self.architecture = architecture
        if architecture == 'conv':
            self.mixers = nn.ModuleList([ResidualUnit(channels, spiking) for _ in range(2)])
        else:
            kinds = ('local', 'cross') if architecture == 'A' else ('gate', 'gate')
            self.mixers = nn.ModuleList([AttentionSublayer(channels, kind, spiking, attention_normalization=attention_normalization) for kind in kinds])
        self.ffn_first = SpikeConv(channels, 2 * channels, 1, spiking=spiking)
        self.ffn_second = nn.Conv2d(2 * channels, channels, 1, bias=False)
        self.ffn_norm = norm(channels)
        self.ffn_out = TemporalNeuron(spiking)

    def forward(self, left, right, previous, updated, valid, max_disparity):
        for mixer in self.mixers:
            if self.architecture == 'conv':
                left, right = [mixer(x, previous, updated, view) for view, x in enumerate((left, right))]
            else:
                left, right = mixer.forward_pair(left, right, previous, updated, valid, max_disparity)
        result = []
        for view, x in enumerate((left, right)):
            current = self.ffn_first(x, previous, updated, view)
            result.append(self.ffn_out(x + self.ffn_norm(self.ffn_second(current)), previous, updated, view) * valid)
        return tuple(result)


class StereoMatcher(nn.Module):
    """Analog shared cosine descriptor, masked softmax and subpixel readout."""
    def __init__(self, channels, descriptor_channels=32, temperature=.1):
        super().__init__()
        self.descriptor = nn.Sequential(nn.Conv2d(channels, descriptor_channels, 1, bias=False), norm(descriptor_channels))
        self.temperature = temperature

    def forward(self, own, other, valid, stride, max_disparity, reverse=False):
        q = F.normalize(self.descriptor(own), dim=1, eps=1e-6)
        k = F.normalize(self.descriptor(other), dim=1, eps=1e-6)
        b, c, h, w = q.shape
        # B,H,query,key. Analog matching cost remains in every model family.
        logits = torch.matmul(q.permute(0, 2, 3, 1), k.permute(0, 2, 1, 3)) / self.temperature
        mask = epipolar_mask(w, max_disparity // stride, q.device, reverse)[None, None]
        mask = mask & valid[:, 0, :, None, :]
        # Padded query pixels are never used by the decoder/metric; give them
        # one finite candidate to avoid all-minus-infinity softmax/NaNs.
        fallback = torch.eye(w, dtype=torch.bool, device=q.device)[None, None]
        mask = torch.where(valid[:, 0, :, :, None], mask, fallback)
        probabilities = logits.masked_fill(~mask, -torch.inf).softmax(-1)
        query = torch.arange(w, device=q.device)[:, None]
        key = torch.arange(w, device=q.device)[None, :]
        disparity = ((key - query) if reverse else (query - key)).to(q.dtype) * stride
        estimate = (probabilities * disparity).sum(-1)[:, None] * valid
        maximum = probabilities.amax(-1)[:, None] * valid
        entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(-1)[:, None]
        count = mask.sum(-1).clamp_min(2).to(q.dtype).log()[:, None]
        entropy = entropy / count * valid
        return estimate, maximum, entropy


def analog_block(ci, co):
    return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1, bias=False), norm(co), nn.GELU(),
                         nn.Conv2d(co, co, 3, padding=1, bias=False), norm(co), nn.GELU())


class DenseDecoder(nn.Module):
    def __init__(self, widths):
        super().__init__()
        self.blocks = nn.ModuleDict()
        previous = 0
        for index in (3, 2, 1, 0):
            self.blocks[str(index)] = analog_block(widths[index] + previous + (3 if index else 0), widths[index])
            previous = widths[index]
        self.residual = nn.Sequential(nn.Conv2d(widths[0], 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, 1, 3, padding=1))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        self.occlusion = nn.Sequential(nn.Conv2d(widths[1] + 2, 32, 3, padding=1, bias=False), norm(32), nn.GELU(), nn.Conv2d(32, 1, 1))

    def forward(self, features, matches, shape, max_disparity):
        current = None
        for index in (3, 2, 1, 0):
            x = features[index]
            pieces = [x]
            if current is not None:
                pieces.append(F.interpolate(current, size=x.shape[-2:], mode='bilinear', align_corners=False))
            if index:
                d, confidence, entropy = matches[2 ** (index + 1)]
                pieces.extend((d / max_disparity, confidence, entropy))
            current = self.blocks[str(index)](torch.cat(pieces, 1))
        residual = self.residual(F.interpolate(current, size=shape, mode='bilinear', align_corners=False))
        disparity = F.interpolate(matches[4][0], size=shape, mode='bilinear', align_corners=False) + residual
        occ = self.occlusion(torch.cat((features[1], matches[4][1], matches[4][2]), 1))
        return disparity, F.interpolate(occ, size=shape, mode='bilinear', align_corners=False)


@dataclass(frozen=True)
class StudyConfig:
    architecture: str = 'A'
    spiking: bool = True
    widths: tuple = (32, 64, 96, 128)
    time_steps: int = 50
    max_disparity: int = 256
    input_channels: int = 1
    checkpoint: bool = True
    temporal_chunk_size: int = 5
    attention_normalization: str = 'count_floor_v2'
    collect_diagnostics: bool = True
    feature_norm_eps: float = 1e-5
    blocks_per_scale: int = 2


class StudyStereo(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.architecture not in ('A', 'B', 'conv'):
            raise ValueError('architecture must be A, B or conv')
        if config.attention_normalization not in ('count_floor_v2', 'epsilon_v1'):
            raise ValueError('Unknown attention normalization version')
        if not math.isfinite(config.feature_norm_eps) or config.feature_norm_eps <= 0:
            raise ValueError('Feature normalization epsilon must be finite and positive')
        if len(config.widths) != 4 or any(c % 4 for c in config.widths):
            raise ValueError('four widths divisible by four are required')
        if config.time_steps < 1 or config.max_disparity < 1 or config.input_channels not in (1, 2) or config.temporal_chunk_size < 1:
            raise ValueError('invalid temporal, disparity or modality configuration')
        self.stages = nn.ModuleList()
        self.blocks = nn.ModuleList()
        ci = config.input_channels
        for index, co in enumerate(config.widths):
            self.stages.append(nn.ModuleList((SpikeConv(ci, co, 3, 2, config.spiking), ResidualUnit(co, config.spiking))))
            self.blocks.append(nn.ModuleList([FeatureBlock(co, config.architecture, config.spiking, config.attention_normalization) for _ in range(config.blocks_per_scale)] if index else []))
            ci = co
        self.matchers = nn.ModuleDict({str(2 ** (i + 1)): StereoMatcher(config.widths[i]) for i in (1, 2, 3)})
        self.decoder = DenseDecoder(config.widths)
        # Feature-current normalization recipe only. Analog descriptor and
        # dense decoder normalization retain their original1e-5 epsilon.
        for container in (self.stages, self.blocks):
            for module in container.modules():
                if isinstance(module, nn.GroupNorm):
                    module.eps = config.feature_norm_eps
        for index, module in enumerate(m for m in self.modules() if isinstance(m, TemporalNeuron)):
            module.state_id = index
            # Static for the entire forward/backward, including recomputation.
            module.collect_diagnostics = config.collect_diagnostics
        self.neuron_names = {module.state_id: name for name, module in self.named_modules() if isinstance(module, TemporalNeuron)}

    def _step(self, left, right, previous, masks):
        updated, features_left, features_right = {}, [], []
        for index, (stage, blocks) in enumerate(zip(self.stages, self.blocks)):
            outputs = []
            for view, x in enumerate((left, right)):
                x = stage[0](x, previous, updated, view)
                x = stage[1](x, previous, updated, view) * masks[index]
                outputs.append(x)
            left, right = outputs
            for block in blocks:
                left, right = block(left, right, previous, updated, masks[index], self.config.max_disparity // (2 ** (index + 1)))
            features_left.append(left)
            features_right.append(right)
        return tuple(features_left), tuple(features_right), updated

    def _chunk(self, left, right, previous, masks):
        """Pure temporal segment; only boundary state survives checkpointing.

        Internal simulation state remains differentiable through every step.
        Nonreentrant checkpointing recomputes the complete segment in backward,
        rather than retaining one full state dictionary per sensor bin.
        """
        states, summed_left, summed_right = previous, None, None
        activities, voltages, per_time = {}, {}, {}
        for step in range(left.shape[1]):
            fl, fr, states = self._step(left[:, step], right[:, step], states, masks)
            summed_left = list(fl) if summed_left is None else [a + z for a, z in zip(summed_left, fl)]
            summed_right = list(fr) if summed_right is None else [a + z for a, z in zip(summed_right, fr)]
            for key, value in states.items():
                if key[0] == 'activity':
                    activities[key[1:]] = activities.get(key[1:], 0.) + value
                    per_time.setdefault(key[1:], []).append(value.mean())
                elif key[0] == 'voltage':
                    voltages[key[1:]] = voltages.get(key[1:], 0.) + value
        # Detached telemetry never enters the recurrent state gradient path.
        boundary = {key: value for key, value in states.items() if isinstance(key[0], int)}
        return tuple(summed_left), tuple(summed_right), boundary, activities, voltages, {
            key: torch.stack(values) for key, values in per_time.items()}

    def forward(self, left, right):
        if left.shape != right.shape or left.ndim != 5:
            raise ValueError('left/right require identical B,T,C,H,W shapes')
        b, steps, channels, h, w = left.shape
        if steps != self.config.time_steps or channels != self.config.input_channels:
            raise ValueError('input must preserve configured sensor steps and polarity channels')
        ph, pw = (-h) % 16, (-w) % 16
        left, right = F.pad(left, (0, pw, 0, ph)), F.pad(right, (0, pw, 0, ph))
        support = left.new_ones((b, 1, h, w))
        support = F.pad(support, (0, pw, 0, ph))
        masks = []
        for _ in range(4):
            support = F.max_pool2d(support, 2, 2)
            masks.append(support.bool())
        states, summed_left, summed_right = {}, None, None
        neuron_activity, neuron_voltage, activity_per_time = {}, {}, {}
        chunk_size = self.config.temporal_chunk_size
        for start in range(0, steps, chunk_size):
            stop = min(steps, start + chunk_size)
            if self.config.checkpoint and self.training and torch.is_grad_enabled():
                fl, fr, states, activities, voltages, times = checkpoint(
                    self._chunk, left[:, start:stop], right[:, start:stop], states, masks, use_reentrant=False)
            else:
                fl, fr, states, activities, voltages, times = self._chunk(
                    left[:, start:stop], right[:, start:stop], states, masks)
            for key, value in activities.items():
                neuron_activity[key] = neuron_activity.get(key, 0.) + value / steps
                activity_per_time.setdefault(key, []).append(times[key])
            for key, value in voltages.items():
                neuron_voltage[key] = neuron_voltage.get(key, 0.) + value / steps
            summed_left = list(fl) if summed_left is None else [a + z for a, z in zip(summed_left, fl)]
            summed_right = list(fr) if summed_right is None else [a + z for a, z in zip(summed_right, fr)]
        rates_left, rates_right = [[x / steps for x in sums] for sums in (summed_left, summed_right)]
        ml, mr = {}, {}
        for index in (1, 2, 3):
            stride = 2 ** (index + 1)
            matcher = self.matchers[str(stride)]
            ml[stride] = matcher(rates_left[index], rates_right[index], masks[index], stride, self.config.max_disparity)
            mr[stride] = matcher(rates_right[index], rates_left[index], masks[index], stride, self.config.max_disparity, reverse=True)
        disparity, occlusion = self.decoder(rates_left, ml, left.shape[-2:], self.config.max_disparity)
        rdisparity, rocc = self.decoder(rates_right, mr, left.shape[-2:], self.config.max_disparity)
        crop = lambda x: x[..., :h, :w]
        diagnostics = {'sensor_steps': steps, 'refinement_iterations': 0, 'hybrid': True,
                       'collect_diagnostics': self.config.collect_diagnostics,
                       'feature_norm_eps': self.config.feature_norm_eps,
                       'spiking': self.config.spiking, 'state_tensors': sum(isinstance(key[0], int) for key in states),
                       'neuron_activity': {f'{self.neuron_names[i]}:view{v}': {
                           'mean': x.mean(), 'channel_min': x.min(), 'channel_max': x.max(),
                           'temporally_dead_channel_fraction': (x == 0).float().mean(),
                           'temporally_saturated_channel_fraction': (x == 1).float().mean(),
                           'channel_rates': x} for (i, v), x in neuron_activity.items()},
                       'neuron_abs_pre_voltage': {f'{self.neuron_names[i]}:view{v}': x for (i, v), x in neuron_voltage.items()},
                       'neuron_mean_activity_per_time': {f'{self.neuron_names[i]}:view{v}': torch.cat(values) for (i, v), values in activity_per_time.items()},
                       'temporal_checkpoint_chunk_size': chunk_size,
                       'attention_compute': 'dense_row_W_squared_masked_to_disparity_support' if self.config.architecture == 'A' else 'global_channel_gates' if self.config.architecture == 'B' else 'local_convolution',
                       'attention_normalization': self.config.attention_normalization,
                       'matcher_compute': 'dense_row_W_squared_masked_to_disparity_support',
                       'feature_activity': {str(2 ** (i + 1)): x.detach().mean() for i, x in enumerate(rates_left)},
                       'analog_components': ['input_projection', 'GroupNorm', 'attention_normalization', 'temporal_rate', 'cosine_matching', 'softmax', 'subpixel_expectation', 'dense_decoder', 'interpolation', 'occlusion_head']}
        return {'disparity': crop(disparity), 'right_disparity': crop(rdisparity),
                'occlusion_logits': crop(occlusion), 'right_occlusion_logits': crop(rocc),
                'aux_disparities_native': {s: values[0] for s, values in ml.items()},
                'right_aux_disparities_native': {s: values[0] for s, values in mr.items()},
                'match_confidence_native': {s: {'max_probability': values[1], 'normalized_entropy': values[2]} for s, values in ml.items()},
                'aux_disparities': {s: crop(F.interpolate(values[0], size=left.shape[-2:], mode='bilinear', align_corners=False)) for s, values in ml.items()},
                'right_aux_disparities': {s: crop(F.interpolate(values[0], size=left.shape[-2:], mode='bilinear', align_corners=False)) for s, values in mr.items()},
                'diagnostics': diagnostics}


def build_model(config: Mapping):
    """Build a complete architecture with explicit sensor/modality contract."""
    accepted = set(StudyConfig.__dataclass_fields__)
    unknown = set(config) - accepted
    if unknown:
        raise ValueError('unknown model configuration fields: ' + ', '.join(sorted(unknown)))
    values = dict(config)
    if 'widths' in values:
        values['widths'] = tuple(values['widths'])
    return StudyStereo(StudyConfig(**values))
