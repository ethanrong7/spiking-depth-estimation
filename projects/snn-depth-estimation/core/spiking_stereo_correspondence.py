"""Ordered-bin spiking correspondence Transformer, with an analog disparity readout.

All neuronal state is explicit, local to one forward, and differentiated through
soft reset. The score supervised by correspondence CE is the cross-attention
QK score itself; there is no parallel descriptor/correlation matching branch.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, gradient):
        (x,) = ctx.saved_tensors
        return gradient / (1 + x.abs()).square()


def lif(current, previous, beta=.9):
    membrane = beta * previous + current
    spike = SurrogateSpike.apply(membrane - 1.)
    return spike, membrane - spike  # Reset remains in the temporal graph.


def norm(channels):
    return nn.GroupNorm(math.gcd(8, channels), channels)


class Current(nn.Sequential):
    def __init__(self, ci, co, kernel=1, stride=1):
        super().__init__(nn.Conv2d(ci, co, kernel, stride, kernel // 2, bias=False), norm(co))


def row_scores(q, k, heads):
    """B,C,H,W -> B,H,heads,W(query),W(key), with no cross-row pairs."""
    b, c, h, w = q.shape
    q, k = [x.reshape(b, heads, c // heads, h, w).permute(0, 3, 1, 4, 2) for x in (q, k)]
    return (q @ k.transpose(-1, -2)) / math.sqrt(c // heads)


def row_values(scores, value, heads, valid):
    b, c, h, w = value.shape
    mask = valid[:, 0, :, None, :, None] & valid[:, 0, :, None, None, :]
    counts = scores * mask.to(scores.dtype)
    weights = counts / counts.sum(-1, keepdim=True).clamp_min(1.)
    values = value.reshape(b, heads, c // heads, h, w).permute(0, 3, 1, 4, 2)
    return (weights @ values).permute(0, 2, 4, 1, 3).reshape(b, c, h, w)


def window_values(q, k, value, heads, valid, window=4):
    b, c, h, w = q.shape
    ph, pw = (-h) % window, (-w) % window
    hp, wp = h + ph, w + pw
    def partition(x):
        x = F.pad(x, (0, pw, 0, ph))
        return x.reshape(b, heads, c // heads, hp // window, window, wp // window, window).permute(0, 3, 5, 1, 4, 6, 2).reshape(-1, heads, window * window, c // heads)
    q, k, v = [partition(x) for x in (q, k, value)]
    mask = F.pad(valid, (0, pw, 0, ph)).reshape(b, 1, hp // window, window, wp // window, window).permute(0, 2, 4, 1, 3, 5).reshape(-1, 1, window * window)
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(c // heads)
    scores = scores * (mask[..., :, None] & mask[..., None, :]).to(scores.dtype)
    mixed = (scores / scores.sum(-1, keepdim=True).clamp_min(1.)) @ v
    return mixed.reshape(b, hp // window, wp // window, heads, window, window, c // heads).permute(0, 3, 6, 1, 4, 2, 5).reshape(b, c, hp, wp)[..., :h, :w]


def native_lattice(value, height, width, stride=4):
    """Upsample sites at native coordinates 0,stride,... with border extension."""
    value = value[..., :math.ceil(height / stride), :math.ceil(width / stride)]
    hc, wc = value.shape[-2:]
    y = torch.arange(height, device=value.device, dtype=value.dtype) / stride
    x = torch.arange(width, device=value.device, dtype=value.dtype) / stride
    y = 2 * y / max(1, hc - 1) - 1 if hc > 1 else y * 0
    x = 2 * x / max(1, wc - 1) - 1 if wc > 1 else x * 0
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    grid = torch.stack([xx, yy], -1)[None].expand(value.shape[0], -1, -1, -1)
    return F.grid_sample(value, grid, mode='bilinear', padding_mode='border', align_corners=True)


class SpikingStereoCorrespondence(nn.Module):
    def __init__(self, time_steps=50, widths=(32, 48, 64), heads=4, window=4,
                 max_width=400, activation_checkpoint=True):
        super().__init__()
        a, b, c = widths
        if b % heads or c % heads or time_steps < 1 or max_width % 4:
            raise ValueError('invalid architecture geometry')
        self.time_steps, self.widths, self.heads, self.window = time_steps, widths, heads, window
        self.max_width, self.activation_checkpoint = max_width, activation_checkpoint
        self.encoder = nn.ModuleList([Current(1, a, 3, 2), Current(a, b, 3, 2), Current(b, c, 3, 2)])
        self.self_qkv = nn.ModuleList([Current(c, c) for _ in range(3)])
        self.self_output = Current(c, c)
        self.ffn_expand = Current(c, 2 * c)
        self.ffn_contract = Current(2 * c, c)
        self.top_down = Current(c, b)
        self.cross_qkv = nn.ModuleList([Current(b, b) for _ in range(3)])
        self.cross_output = Current(b, b)
        self.temporal_logits = nn.Parameter(torch.zeros(time_steps))
        self.logit_gain_raw = nn.Parameter(torch.tensor(math.log(math.expm1(8.))))
        self.relative_bias = nn.Parameter(torch.zeros(2 * (max_width // 4) - 1))
        self.readout = nn.Sequential(nn.Conv2d(2 * b, b, 3, padding=1), norm(b), nn.GELU(), nn.Conv2d(b, 3, 3, padding=1))
        nn.init.zeros_(self.readout[-1].weight)
        with torch.no_grad():
            self.readout[-1].bias.copy_(torch.tensor([-4., 0., 0.]))

    def _step(self, sensor, valid4, valid8, *previous, zero_content=False, zero_values=False):
        states, index = [], 0
        def neuron(current):
            nonlocal index
            z, state = lif(current, previous[index]); index += 1; states.append(state)
            return z
        e2 = neuron(self.encoder[0](sensor))
        e4 = neuron(self.encoder[1](e2))
        e8 = neuron(self.encoder[2](e4))
        q, k, v = [neuron(module(e8)) for module in self.self_qkv]
        local = window_values(q, k, v, self.heads, valid8, self.window)
        enriched = neuron(e8 + self.self_output(local))
        hidden = neuron(self.ffn_expand(enriched))
        enriched = neuron(enriched + self.ffn_contract(hidden))
        fused = neuron(e4 + native_lattice(self.top_down(enriched), *e4.shape[-2:], stride=2))
        q, k, v = [neuron(module(fused)) for module in self.cross_qkv]
        ql, qr = q.chunk(2); kl, kr = k.chunk(2); vl, vr = v.chunk(2)
        scores = row_scores(ql, kr, self.heads)
        reverse = row_scores(qr, kl, self.heads)
        if zero_content:
            scores = scores * 0
            reverse = reverse * 0
        mixed = torch.cat([row_values(scores, vr, self.heads, valid4[:len(ql)]), row_values(reverse, vl, self.heads, valid4[:len(ql)])])
        if zero_values:
            mixed = torch.zeros_like(mixed)
        cross = neuron(fused + self.cross_output(mixed))
        # The left QK score is both the correspondence evidence and attention score.
        return (*states, scores.mean(2), fused[:len(ql)], cross[:len(ql)],
                torch.stack([e2.mean(), e4.mean(), e8.mean(), q.mean(), k.mean(), v.mean(), cross.mean()]))

    def _initial(self, sensor):
        b, _, h, w = sensor.shape
        a, c4, c8 = self.widths
        shapes = [(a, h // 2, w // 2), (c4, h // 4, w // 4)] + [(c8, h // 8, w // 8)] * 5 + [(2 * c8, h // 8, w // 8), (c8, h // 8, w // 8)] + [(c4, h // 4, w // 4)] * 5
        return tuple(sensor.new_zeros((b, *shape)) for shape in shapes)

    def forward(self, left, right, *, zero_content=False, zero_values=False):
        if left.shape != right.shape or left.ndim != 4 or left.shape[1] != self.time_steps:
            raise ValueError('paired B,T,H,W inputs with all registered time bins required')
        batch, _, height, width = left.shape
        if width > self.max_width:
            raise ValueError('input exceeds registered maximum width')
        sensor = F.pad(torch.cat([left, right]), (0, (-width) % 8, 0, (-height) % 8))
        mask = sensor.new_zeros((2 * batch, 1, *sensor.shape[-2:]), dtype=torch.bool)
        mask[..., :height, :width] = True
        valid4 = F.interpolate(mask.float(), scale_factor=.25, mode='nearest').bool()
        valid8 = F.interpolate(mask.float(), scale_factor=.125, mode='nearest').bool()
        state = self._initial(sensor[:, :1]); count = len(state)
        weights = self.temporal_logits.softmax(0)
        score_sum = fused_sum = cross_sum = activity = None
        activity_per_time = []
        for t in range(self.time_steps):
            args = (sensor[:, t:t + 1], valid4, valid8, *state)
            output = checkpoint(self._step, *args, use_reentrant=False, zero_content=zero_content, zero_values=zero_values) if self.activation_checkpoint and self.training and torch.is_grad_enabled() else self._step(*args, zero_content=zero_content, zero_values=zero_values)
            state = output[:count]
            score, fused, cross, rates = output[count:]
            score_sum = weights[t] * score if score_sum is None else score_sum + weights[t] * score
            fused_sum = weights[t] * fused if fused_sum is None else fused_sum + weights[t] * fused
            cross_sum = weights[t] * cross if cross_sum is None else cross_sum + weights[t] * cross
            activity = rates.detach() / self.time_steps if activity is None else activity + rates.detach() / self.time_steps
            activity_per_time.append(rates.detach())
        wc = score_sum.shape[-1]
        x = torch.arange(wc, device=left.device)
        relative = x[None, :] - x[:, None]
        bias = self.relative_bias[relative + self.max_width // 4 - 1]
        logits = F.softplus(self.logit_gain_raw) * score_sum + bias
        query_valid = valid4[:batch, 0]
        geometry = query_valid[..., :, None] & query_valid[..., None, :]
        # Invalid query rows get a finite placeholder before softmax, then zero output.
        safe = geometry | ~query_valid[..., :, None]
        logits = logits.masked_fill(~safe, torch.finfo(logits.dtype).min)
        probabilities = logits.softmax(-1) * query_valid[..., :, None]
        coarse = (probabilities * (4 * relative)[None, None]).sum(-1)[:, None]
        readout = self.readout(torch.cat([fused_sum, cross_sum], 1))
        null_logit, null_flow_raw, residual_raw = readout.split(1, 1)
        p_null = null_logit.sigmoid()
        null_flow = 512. * null_flow_raw.tanh()
        native = lambda value: native_lattice(value, height, width)
        match = native(coarse)
        null = native(null_flow)
        null_probability = native(null_logit).sigmoid()
        residual = 4. * native(residual_raw).tanh()
        flow = (1 - null_probability) * match + null_probability * null + residual
        return dict(flow=flow, coarse_flow=match, logits=logits, probabilities=probabilities,
                    null_logit=null_logit, null_probability=null_probability, null_flow=null,
                    residual=residual, query_valid=query_valid, temporal_weights=weights,
                    activity=activity, activity_per_time=torch.stack(activity_per_time), analog_components=('GroupNorm', 'score normalization', 'temporal softmax',
                    'matching softmax/expectation', 'interpolation', 'null/refinement readout'))


def correspondence_loss(output, target, valid, normalization_counts=None):
    """Full-mask signed-flow L1 plus in-frame correspondence/null supervision.

    Coarse correspondence targets are sampled on the stride-four query lattice.
    A target between key sites uses linearly interpolated CE; the last partial
    key interval clamps to the final valid site. No per-bin ground truth is used.
    """
    if valid.ndim == 3:
        valid = valid[:, None]
    valid = valid.bool() & torch.isfinite(target) & (target.abs() < 512)
    if not valid.any():
        raise ValueError('empty valid training mask')
    h, w = target.shape[-2:]
    x = torch.arange(w, device=target.device)[None, None, None, :]
    projected = x + target
    inframe = valid & (projected >= 0) & (projected <= w - 1)
    counts = normalization_counts or {}
    final = (output['flow'] - target).abs()[valid].sum() / max(1, counts['valid'] if 'valid' in counts else int(valid.sum()))
    coarse = (output['coarse_flow'] - target).abs()[inframe].sum() / max(1, counts['inframe'] if 'inframe' in counts else int(inframe.sum()))
    null_native = native_lattice(output['null_logit'], h, w)
    null_bce = F.binary_cross_entropy_with_logits(null_native[valid], (~inframe)[valid].float(), reduction='sum') / max(1, counts['valid'] if 'valid' in counts else int(valid.sum()))
    logits = output['logits']
    hc, wc = logits.shape[1:3]
    # Sampling includes padded sites, which remain invalid in the CE mask.
    pad = (0, (-w) % 8, 0, (-h) % 8)
    index_target = torch.where(valid, target, torch.zeros_like(target))
    coarse_target = F.pad(index_target, pad)[..., ::4, ::4][:, 0]
    coarse_mask = F.pad(inframe, pad)[..., ::4, ::4][:, 0] & output['query_valid']
    key = (torch.arange(wc, device=target.device)[None, None, :] + coarse_target / 4).clamp(0, min(wc - 1, (w - 1) // 4))
    lower, upper = key.floor().long(), key.ceil().long()
    fraction = key - lower
    logp = logits.log_softmax(-1)
    ce = -((1 - fraction) * logp.gather(-1, lower[..., None]).squeeze(-1) + fraction * logp.gather(-1, upper[..., None]).squeeze(-1))
    correspondence = ce[coarse_mask].sum() / max(1, counts['correspondence'] if 'correspondence' in counts else int(coarse_mask.sum()))
    total = final + .5 * coarse + .1 * correspondence + .1 * null_bce
    return total, dict(final_l1=final, coarse_l1=coarse, correspondence_ce=correspondence,
                       null_bce=null_bce, valid_pixels=valid.sum(), inframe_pixels=inframe.sum())
