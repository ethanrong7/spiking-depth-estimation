"""Independent SSDEFormer paper reconstruction and matched Spikformer block swap.

Not official SSDEFormer, and not the historical 426k inspired control. See the
experiment's paper-contract.json for specified versus reconstructed decisions.
Both variants share cumulative coding, full-resolution FPN/tokenizer, position,
epipolar layout and the same analog disparity readout. Spikformer uses repeated
encoded features over neural time, not raw-bin sensor-time feature extraction.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class _ATanSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, margin):
        ctx.save_for_backward(margin)
        return (margin >= 0).to(margin.dtype)

    @staticmethod
    def backward(ctx, gradient):
        margin, = ctx.saved_tensors
        return gradient / (1 + (math.pi * margin).square())


def lif_sequence(current, threshold=1., detach_reset=True):
    """SpikingJelly-style tau2, decay_input=True, hard reset0, ATan alpha2.

    State is local to this complete [T,...] call. No state detachment or truncated
    BPTT. detach_reset is the upstream reset-derivative convention, not detaching
    the membrane between time steps. Distinct packed samples/views never share it.
    """
    membrane = torch.zeros_like(current[0])
    outputs = []
    for item in current.unbind(0):
        membrane = membrane + (item - membrane) / 2
        signal = _ATanSpike.apply(membrane - threshold)
        reset = signal.detach() if detach_reset else signal
        membrane = membrane * (1 - reset)
        outputs.append(signal)
    return torch.stack(outputs)


class CumulativeCode(nn.Module):
    """Paper Eq6 unnormalized prefix sums; retain all50 supplied bins."""
    def forward(self, values):
        if values.ndim != 4 or values.shape[1] != 50:
            raise ValueError('Expected all50 binary sensor bins [B,50,H,W]')
        return values.float().cumsum(1)


def conv(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1),
                         nn.GroupNorm(8, cout), nn.GELU())


class FullResolutionFeatures(nn.Module):
    """Declared FPN reconstruction: scales4/8/16, tokenizer returns to HxW.

    Paper does not specify the convolution inventory, widths or norm. GN/GELU,
    concatenation, bilinear resizing and these widths are explicit choices.
    """
    def __init__(self, width, pyramid=(64, 128, 256, 512)):
        super().__init__()
        a, b, c, d = pyramid
        self.down = nn.ModuleList([conv(50, a, 2), conv(a, b, 2),
                                   conv(b, c, 2), conv(c, d, 2)])
        self.fuse8 = conv(d + c, c)
        self.fuse4 = conv(c + b, b)
        self.fuse1 = nn.Conv2d(b + 50, width, 1)

    def forward(self, code):
        features, value = [], code
        for layer in self.down:
            value = layer(value)
            features.append(value)
        value = self.fuse8(torch.cat([F.interpolate(features[3], size=features[2].shape[-2:],
            mode='bilinear', align_corners=False), features[2]], 1))
        value = self.fuse4(torch.cat([F.interpolate(value, size=features[1].shape[-2:],
            mode='bilinear', align_corners=False), features[1]], 1))
        return self.fuse1(torch.cat([F.interpolate(value, size=code.shape[-2:],
            mode='bilinear', align_corners=False), code], 1))


def position_2d(height, width, channels, device):
    if channels % 4:
        raise ValueError('Position width must be divisible by4')
    frequency = torch.exp(torch.arange(0, channels // 2, 2, device=device).float()
                          * (-math.log(10000.) / (channels // 2)))
    def axis(length):
        phase = torch.arange(length, device=device).float()[:, None] * frequency
        return torch.stack([phase.sin(), phase.cos()], -1).flatten(-2)
    return torch.cat([axis(height)[:, None].expand(-1, width, -1),
                      axis(width)[None].expand(height, -1, -1)], -1)


class SoftmaxBlock(nn.Module):
    def __init__(self, width, heads, mlp_ratio):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, width * mlp_ratio), nn.GELU(),
                                 nn.Linear(width * mlp_ratio, width))

    def forward(self, own, source, valid):
        q, kv = self.norm1(own), self.norm1(source)
        mixed = self.attention(q, kv, kv, key_padding_mask=~valid, need_weights=False)[0]
        value = own + mixed
        value = value + self.ffn(self.norm2(value))
        return value * valid[..., None]


class SpikeProjection(nn.Module):
    def __init__(self, cin, cout, *, bias=False, threshold=1., detach_reset=True):
        super().__init__()
        self.conv = nn.Conv1d(cin, cout, 1, bias=bias)
        self.bn = nn.BatchNorm1d(cout)
        self.threshold, self.detach_reset = threshold, detach_reset

    def forward(self, value):
        # [T, packed_view_and_rows, W, C]. BN pools time as upstream does.
        t, b, w, c = value.shape
        current = self.bn(self.conv(value.reshape(t * b, w, c).transpose(1, 2)))
        current = current.transpose(1, 2).reshape(t, b, w, -1)
        return lif_sequence(current, self.threshold, self.detach_reset)


class SpikformerBlock(nn.Module):
    """Original SSA+spiking MLP adapted to row self/cross attention.

    Reference: ZK-Zhou/spikformer@621268981827af47aa666c1b5f774de4281d90ad,
    imagenet/model.py. Copyright(c)2022 Zhaokun Zhou, MIT; see experiment
    third_party/Spikformer-LICENSE. No SPS/classifier substitution. Cross-view querying,
    valid-key masking and stereo row grouping are our adaptations.
    """
    def __init__(self, width, heads, mlp_ratio, detach_reset=True):
        super().__init__()
        self.heads, self.scale = heads, .125
        self.q = SpikeProjection(width, width, detach_reset=detach_reset)
        self.k = SpikeProjection(width, width, detach_reset=detach_reset)
        self.v = SpikeProjection(width, width, detach_reset=detach_reset)
        self.project = SpikeProjection(width, width, bias=True, detach_reset=detach_reset)
        self.fc1 = SpikeProjection(width, width * mlp_ratio, bias=True, detach_reset=detach_reset)
        self.fc2 = SpikeProjection(width * mlp_ratio, width, bias=True, detach_reset=detach_reset)
        self.detach_reset = detach_reset

    def forward(self, own, source, valid):
        t, b, w, c = own.shape
        def heads(value):
            return value.reshape(t, b, w, self.heads, c // self.heads).transpose(2, 3)
        q = heads(self.q(own))
        k = heads(self.k(source)) * valid[None, :, None, :, None]
        v = heads(self.v(source)) * valid[None, :, None, :, None]
        # Associative SSA, no softmax or normalization substituted into it.
        mixed = (q @ (k.transpose(-1, -2) @ v)) * self.scale
        mixed = mixed.transpose(2, 3).reshape(t, b, w, c)
        mixed = lif_sequence(mixed, .5, self.detach_reset)
        value = own + self.project(mixed)
        value = value + self.fc2(self.fc1(value))
        return value * valid[None, ..., None]


def local_disparity(probability, valid, radius=1):
    """Full-pixel WTA followed by normalized three-candidate expectation.

    W3 is interpreted as a three-pixel total window (peak-1,peak,peak+1), not
    the historical inspired model's seven-candidate +/-3 support. Paper wording
    is ambiguous; this choice is bound explicitly in the reconstruction contract.
    """
    width = probability.shape[-1]
    x = torch.arange(width, device=probability.device)
    disparity = x[:, None] - x[None, :]
    support = (disparity >= 0) & valid[:, None, :]
    masked = probability * support
    peak = masked.argmax(-1)
    local = (x[None, None, :] - peak[..., None]).abs() <= radius
    mass = masked * local
    mass = mass / mass.sum(-1, keepdim=True).clamp_min(1e-12)
    return (mass * disparity).sum(-1) * valid


class SSDEFormerReconstruction(nn.Module):
    """Two configurations of one shared stereo system; positive disparity output.

    Default1024 is a prospective width choice, not a recovered paper value.
    Small widths are supported solely for synthetic wiring tests. Scientific
    geometry/width/time must be frozen separately before cluster execution.
    """
    def __init__(self, variant='softmax', width=1024, heads=8, layers=6,
                 mlp_ratio=4, neural_steps=50, pyramid=(64, 128, 256, 512),
                 row_chunk=8, detach_reset=True):
        super().__init__()
        if variant not in ('softmax', 'spikformer') or width % heads or width % 4:
            raise ValueError('Explicit variant and compatible width/heads required')
        if layers <= 0 or layers % 2 or neural_steps != 50 or row_chunk <= 0:
            raise ValueError('Even self/cross stack and all50 neural steps required')
        self.variant, self.width = variant, width
        self.neural_steps, self.row_chunk = neural_steps, row_chunk
        self.code = CumulativeCode()
        self.features = FullResolutionFeatures(width, pyramid)
        # Construct common modules before variant modules: equal initialization
        # seed gives byte-identical feature/readout tensors in both variants.
        self.match_query = nn.Linear(width, width)
        self.match_key = nn.Linear(width, width)
        cls = SoftmaxBlock if variant == 'softmax' else SpikformerBlock
        self.blocks = nn.ModuleList([cls(width, heads, mlp_ratio) if variant == 'softmax'
            else cls(width, heads, mlp_ratio, detach_reset) for _ in range(layers)])
        # Common analog correspondence readout. Explicit independent decision:
        # raw final SSA weights cannot substitute for calibrated probabilities.

    def _interaction(self, packed, valid):
        # packed=[two_views,rows,W,C], same pre-update pair in both directions.
        if self.variant == 'spikformer':
            packed = packed.unsqueeze(0).expand(self.neural_steps, -1, -1, -1, -1)
        for index, block in enumerate(self.blocks):
            own = packed.flatten(0, 1) if self.variant == 'softmax' else packed.flatten(1, 2)
            source = packed.flip(0) if self.variant == 'softmax' else packed.flip(1)
            source = source.flatten(0, 1) if self.variant == 'softmax' else source.flatten(1, 2)
            if not index % 2:
                source = own
            value = block(own, source, valid.repeat(2, 1))
            packed = value.reshape_as(packed)
        return packed if self.variant == 'softmax' else packed.mean(0)

    def forward(self, left, right):
        if left.shape != right.shape:
            raise ValueError('Matched stereo shapes required')
        b, _, h, w = left.shape
        if min(h, w) < 16:
            raise ValueError('At least16x16 source geometry required')
        code = self.code(torch.cat([left, right], 0))
        padding = (0, (-w) % 16, 0, (-h) % 16)
        feature = self.features(F.pad(code, padding))[:, :, :h, :w]
        feature = feature.permute(0, 2, 3, 1)
        feature = feature + position_2d(h, w, self.width, feature.device)
        pair = feature.reshape(2, b * h, w, self.width)
        valid = torch.ones(b * h, w, dtype=torch.bool, device=left.device)
        # Eval row chunking preserves sample results. Training BN must see all
        # rows jointly, so training always executes the unchunked common batch.
        chunk = b * h if self.training else self.row_chunk
        results = []
        for offset in range(0, b * h, chunk):
            mask = valid[offset:offset + chunk]
            lo, ro = self._interaction(pair[:, offset:offset + chunk], mask).unbind(0)
            logits = (self.match_query(lo) @ self.match_key(ro).transpose(-1, -2)) / math.sqrt(self.width)
            columns = torch.arange(w, device=left.device)
            logits = logits.masked_fill(columns[None, :] > columns[:, None], -torch.inf)
            probability = logits.softmax(-1)
            results.append(local_disparity(probability, mask))
        disparity = torch.cat(results).reshape(b, 1, h, w)
        return {'disparity': disparity}


def build_from_config(config):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config['initialization_seed'])
        return SSDEFormerReconstruction(**config['model'])
