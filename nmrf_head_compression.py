"""Opt-in reduced-width spatial heads for the native 50-bin NMRF parent.

Adapted from the vendored NMRF SwinNMP and WindowAttention (CVPR 2024).
The upstream license and attribution remain in the vendored directory.
"""
import torch
from einops import rearrange, repeat
from torch import nn
from torch.nn import functional as F

import nmrf_runtime  # Registers the pinned upstream module path.
from nmrf.models.NMP import SwinNMP, WindowAttention
from nmrf_selective_updates import SelectiveSwinNMP

FORMAT = 'nmrf-head-compression-v1'
ARCHITECTURE = 'full_spatial_reduced_heads_nmrf_v1'


class ReducedHeadSwinNMP(SwinNMP):
    """Narrow only attention heads while preserving the original state width."""

    encode = SelectiveSwinNMP.encode

    def __init__(self, original, heads, coding='continuous', projection=None):
        nn.Module.__init__(self)
        if type(original) is not SwinNMP or not original.normalize_before:
            raise ValueError('Expected an original pre-norm spatial NMRF block')
        if coding not in ('continuous', 'ternary'):
            raise ValueError('Head compression supports continuous and ternary coding')
        heads = tuple(heads)
        if (not heads or any(type(i) is not int or not 0 <= i < original.num_heads for i in heads)
                or tuple(sorted(set(heads))) != heads):
            raise ValueError('Retained heads must be unique and in original order')

        for name in ('dim', 'qkv_dim', 'window_size', 'shift_size', 'mlp_ratio',
                     'normalize_before', 'H', 'W'):
            setattr(self, name, getattr(original, name))
        for name in ('norm1', 'norm2', 'mlp', 'proj_drop', 'drop_path'):
            setattr(self, name, getattr(original, name))
        self.retained_heads = heads
        self.head_dim = original.attn.dim // original.num_heads
        self.num_heads = len(heads)
        self.attention_dim = self.num_heads*self.head_dim
        self.coding = coding
        self.threshold = .5

        device = original.qkv.weight.device
        dtype = original.qkv.weight.dtype
        ids = torch.tensor(heads, device=device)
        head_columns = (ids[:, None]*self.head_dim + torch.arange(self.head_dim, device=device)).flatten()
        qkv_rows = (torch.arange(3, device=device)[:, None]*original.attn.dim + head_columns).flatten()
        if projection is not None:
            weight, bias = projection['weight'], projection['bias']
            if (weight.shape != (self.dim, self.attention_dim) or bias.shape != (self.dim,)
                    or weight.dtype != torch.float32 or bias.dtype != torch.float32
                    or not torch.isfinite(weight).all() or not torch.isfinite(bias).all()):
                raise ValueError('Fitted projection must contain finite FP32 weight and bias of the reduced shape')
        else:
            weight = original.proj.weight.index_select(1, head_columns)
            bias = original.proj.bias

        # New linear constructors consume RNG before their weights are copied.
        # Keep the caller's RNG state for other model construction.
        devices = [device.index] if device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            self.qkv = nn.Linear(self.qkv_dim, 3*self.attention_dim, original.qkv.bias is not None,
                                 device=device, dtype=dtype)
            self.proj = nn.Linear(self.attention_dim, self.dim, device=device, dtype=dtype)
            self.attn = WindowAttention(self.attention_dim, original.attn.window_size, self.shift_size,
                                        self.num_heads, qk_scale=original.attn.scale,
                                        attn_drop=original.attn.attn_drop.p).to(device=device, dtype=dtype)
        with torch.no_grad():
            self.qkv.weight.copy_(original.qkv.weight.index_select(0, qkv_rows))
            if self.qkv.bias is not None:
                self.qkv.bias.copy_(original.qkv.bias.index_select(0, qkv_rows))
            self.proj.weight.copy_(weight)
            self.proj.bias.copy_(bias)
            relative = original.attn.relative_position_enc_table.reshape(-1, original.num_heads, 3*self.head_dim)
            self.attn.relative_position_enc_table.copy_(relative.index_select(1, ids).flatten(1))
            self.attn.relative_position_index.copy_(original.attn.relative_position_index)
        self.train(original.training)

    def _window_heads(self, projection):
        projection = rearrange(projection, '(b h w) n c -> b h w n c', h=self.H, w=self.W)
        if self.shift_size:
            projection = torch.roll(projection, (-self.shift_size, -self.shift_size), (1, 2))
        return [self.attn.window_partition(part) for part in projection.chunk(3, dim=-1)]

    def _signed_message(self, label, geometry, mask):
        content = F.linear(self.norm1(label), self.qkv.weight[:, :self.dim], self.qkv.bias)
        positional = F.linear(geometry, self.qkv.weight[:, self.dim:])
        qc, kc, vc = self._window_heads(content)
        qg, kg, vg = self._window_heads(positional)
        q, k, v = qc+qg, kc+kg, vc+vg
        coded_q, coded_k = self.encode(qc)+qg, self.encode(kc)+kg

        pixels = self.window_size**2
        candidates = label.shape[1]
        relative = self.attn.relative_position_enc_table[self.attn.relative_position_index]
        relative = relative.reshape(pixels, pixels, self.num_heads, 3*self.head_dim)
        qe, ke, ve = repeat(relative, 'i j h c -> (i a) (j b) h c',
                            a=candidates, b=candidates).chunk(3, dim=-1)
        scale = self.attn.scale
        scores = (coded_q*scale) @ coded_k.transpose(-2, -1)
        scores = scores + torch.einsum('bhic,ijhc->bhij', q*scale, ke)
        scores = scores + torch.einsum('bhjc,ijhc->bhij', k, qe*scale)
        if mask is not None:
            windows = (self.H//self.window_size)*(self.W//self.window_size)
            scores = rearrange(scores, '(b w) h i j -> b w h i j', w=windows)
            scores = scores + mask[None, :, None]
            scores = rearrange(scores, 'b w h i j -> (b w) h i j')
        probs = self.attn.attn_drop(scores.softmax(dim=-1))
        message = probs@v + torch.einsum('bhij,ijhc->bhic', probs, ve)
        message = rearrange(message, '(b i j) h (p q n) d -> b (i p) (j q) n (h d)',
                            i=self.H//self.window_size, j=self.W//self.window_size,
                            p=self.window_size, q=self.window_size)
        if self.shift_size:
            message = torch.roll(message, (self.shift_size, self.shift_size), (1, 2))
        return rearrange(message, 'b h w n c -> (b h w) n c')

    def forward_pre(self, label_rep, abs_encoding, attn_mask):
        if self.coding == 'continuous':
            return super().forward_pre(label_rep, abs_encoding, attn_mask)
        message = self._signed_message(label_rep, abs_encoding, attn_mask)
        message = self.proj_drop(self.proj(message))
        updated = label_rep + self.drop_path(message)
        return updated + self.drop_path(self.mlp(self.norm2(updated)))


def _valid_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def install_head_compression(model, head_pairs, *, coding, projections=None, projection_sha256=None):
    """Install two heads in each native spatial block before optimizer construction."""
    blocks = model.parent.inference.layers
    pairs = [tuple(pair) for pair in head_pairs]
    if len(blocks) != 5 or len(pairs) != 5 or any(len(pair) != 2 for pair in pairs):
        raise ValueError('Supply one two-head pair for each of the five spatial inference blocks')
    if model.parent.num_proposals != 4 or any(
            block.nmp.dim != 128 or block.nmp.num_heads != 4 or block.nmp.qkv_dim != 159
            or block.nmp.window_size != 6 or block.nmp.attn.scale != 32**-.5 for block in blocks):
        raise ValueError('Head compression expects native four-head, 128-state spatial inference')
    if projections is not None:
        if len(projections) != 5 or not _valid_digest(projection_sha256):
            raise ValueError('Fitted projection state requires five layers and its SHA-256 identity')
    elif projection_sha256 is not None:
        raise ValueError('Projection identity requires fitted projections')
    replacements = [
        ReducedHeadSwinNMP(block.nmp, pair, coding, projections[i] if projections is not None else None)
        for i, (block, pair) in enumerate(zip(blocks, pairs))]
    for block, replacement in zip(blocks, replacements):
        block.nmp = replacement
    return dict(architecture=ARCHITECTURE, layers=list(range(5)), head_pairs=[list(pair) for pair in pairs],
                retained_head_order='original', head_dim=32, attention_dim=64, state_dim=128,
                candidates=4, spatial_support='full_original_windows', scale=32**-.5,
                coding=coding, threshold=.5, geometry='continuous_qkv_with_mixed_content_terms',
                projection_initialization='ridge_fit' if projections is not None else 'teacher_columns',
                projection_sha256=projection_sha256)


def restore_head_compression(model, checkpoint):
    """Recreate reduced shapes and strictly restore without fitting data."""
    if checkpoint.get('format') != FORMAT:
        raise ValueError('Expected a head-compression checkpoint')
    config = checkpoint['architecture']
    if config.get('architecture') != ARCHITECTURE:
        raise ValueError('Unknown head-compression architecture')
    projection = config.get('projection_initialization')
    identity = config.get('projection_sha256')
    if not ((projection == 'teacher_columns' and identity is None)
            or (projection == 'ridge_fit' and _valid_digest(identity))):
        raise ValueError('Invalid head-compression projection provenance')
    restored = install_head_compression(model, config['head_pairs'], coding=config['coding'])
    restored.update(projection_initialization=projection, projection_sha256=identity)
    if restored != config:
        raise ValueError('Incompatible head-compression architecture')
    model.load_state_dict(checkpoint['model'], strict=True)
    return restored
