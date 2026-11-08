"""Opt-in compact spatial NMRF updates; upstream source remains unchanged.

Adapted from the vendored NMRF SwinNMP / WindowAttention implementation (CVPR
2024); its attribution and license remain in projects/snn-depth-estimation/vendor/nmrf.
This is a trainable hybrid signed/quantized prototype, not a temporal SNN.
Ternary floating tensors do not imply sparse or integer GPU execution.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, repeat

LEGACY_SELECTOR = "fixed_even_spatial_phase_by_layer"
ALTERNATING_SELECTOR = "shift_aware_alternating_global_column_parity"
COMPACT_SUPPORT = "selected_queries_selected_keys_values"
FULL_KV_SUPPORT = "selected_queries_all_keys_values"


class SelectiveSwinNMP(nn.Module):
    """Gather whole query pixels before Q and FFN, then scatter their updates.

    All candidates of a selected pixel stay active. The fixed schedule is shared
    across windows so relative-position tensors need not be duplicated per window.
    Coding applies to the content Q/K projections. Geometric projections and
    values stay continuous. Mixed content–geometry terms in QK change when
    coding is enabled. Positional dot products use the original continuous Q/K.
    The default gathers K/V at the same pixels. Full K/V support keeps all
    inputs available as context while still updating only selected queries.
    Config is external; state_dict keys are unchanged.
    """

    def __init__(self, original, keep_fraction=0.5, coding="continuous", phase=0,
                 threshold=0.5, attention_support=COMPACT_SUPPORT):
        super().__init__()
        if not math.isfinite(keep_fraction) or not 0 < keep_fraction <= 1:
            raise ValueError("keep_fraction must be in (0, 1]")
        if coding not in ("continuous", "ternary", "tanh"):
            raise ValueError("coding must be continuous, ternary or tanh")
        if not math.isfinite(threshold) or threshold <= 0:
            raise ValueError("threshold must be finite and positive")
        if attention_support not in (COMPACT_SUPPORT, FULL_KV_SUPPORT):
            raise ValueError("Unknown selective attention support")
        if isinstance(original, SelectiveSwinNMP):
            raise ValueError("Block already has selective updates")
        for name, module in original.named_children():
            self.add_module(name, module)
        for name in ("dim", "qkv_dim", "num_heads", "window_size", "shift_size",
                     "mlp_ratio", "normalize_before", "H", "W"):
            setattr(self, name, getattr(original, name))
        self.keep_fraction = keep_fraction
        self.coding = coding
        self.phase = int(phase)
        self.threshold = threshold
        self.attention_support = attention_support
        self.train(original.training)

    def selected_pixels(self, device):
        pixels = self.window_size ** 2
        count = max(1, math.ceil(pixels * self.keep_fraction))
        # Evenly spread, deterministic, and independent of labels/activations.
        return ((torch.arange(count, device=device) * pixels // count
                 + self.phase) % pixels).sort().values

    def partition(self, x):
        x = rearrange(x, '(b h w) n c -> b h w n c', h=self.H, w=self.W)
        if self.shift_size:
            x = torch.roll(x, (-self.shift_size, -self.shift_size), (1, 2))
        return rearrange(x, 'b (i h) (j w) n c -> (b i j) (h w) n c',
                         h=self.window_size, w=self.window_size)

    def unpartition(self, x):
        x = rearrange(x, '(b i j) (h w) n c -> b (i h) (j w) n c',
                         i=self.H // self.window_size, j=self.W // self.window_size,
                         h=self.window_size, w=self.window_size)
        if self.shift_size:
            x = torch.roll(x, (self.shift_size, self.shift_size), (1, 2))
        return rearrange(x, 'b h w n c -> (b h w) n c')

    def encode(self, x):
        if self.coding == "continuous":
            return x
        # Per-token/head detached RMS: no Dev labels or oracle output statistics.
        # This scaling is part of the intervention, not a free energy operation.
        scale = x.detach().square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
        if self.coding == "tanh":
            return scale * torch.tanh(x / scale)
        decoded = scale * x.sign() * (x.abs() >= self.threshold * scale)
        # Explicit identity STE; a training comparison cannot attribute any gain
        # uniquely to discreteness without a forward/backward factorial.
        return x + (decoded - x).detach()

    def forward(self, label_rep, abs_encoding, attn_mask):
        if self.H is None or self.W is None or self.H % self.window_size or self.W % self.window_size:
            raise ValueError("H/W must be set and divisible by window_size")
        if label_rep.shape[:2] != abs_encoding.shape[:2]:
            raise ValueError("Label and absolute encoding shapes must agree")
        if label_rep.shape[-1] != self.dim or label_rep.shape[-1] + abs_encoding.shape[-1] != self.qkv_dim:
            raise ValueError("Unexpected label/geometry dimensions")
        all_labels = self.partition(label_rep)
        pixels = self.selected_pixels(label_rep.device)
        labels = all_labels.index_select(1, pixels)
        nw, kept, candidates, _ = labels.shape
        full_kv = self.attention_support == FULL_KV_SUPPORT

        def project(content, geometry, weight, bias):
            # The two contractions have the same total MACs as a concatenated
            # projection while keeping content available for coding.
            return F.linear(content, weight[:, :self.dim], bias), F.linear(geometry, weight[:, self.dim:])

        def heads(x):
            return rearrange(x, 'b p n (h d) -> b h (p n) d', h=self.num_heads)

        if full_kv:
            all_geometry = self.partition(abs_encoding)
            geometry = all_geometry.index_select(1, pixels)
            all_content = self.norm1(all_labels) if self.normalize_before else all_labels
            content = all_content.index_select(1, pixels) if self.normalize_before else labels
            q_weight, kv_weight = self.qkv.weight.split((self.dim, 2*self.dim))
            q_bias, kv_bias = (None, None) if self.qkv.bias is None else self.qkv.bias.split((self.dim, 2*self.dim))
            qc, qg = project(content, geometry, q_weight, q_bias)
            kv_c, kv_g = project(all_content, all_geometry, kv_weight, kv_bias)
            qc, kc, vc = map(heads, (qc, *kv_c.chunk(2, -1)))
            qg, kg, vg = map(heads, (qg, *kv_g.chunk(2, -1)))
        else:
            geometry = self.partition(abs_encoding).index_select(1, pixels)
            content = self.norm1(labels) if self.normalize_before else labels
            projected_content, projected_geometry = project(content, geometry, self.qkv.weight, self.qkv.bias)
            qc, kc, vc = map(heads, projected_content.chunk(3, -1))
            qg, kg, vg = map(heads, projected_geometry.chunk(3, -1))
        q, k, v = qc + qg, kc + kg, vc + vg
        coded_q, coded_k = self.encode(qc) + qg, self.encode(kc) + kg

        relative_index = self.attn.relative_position_index.index_select(0, pixels)
        if not full_kv:
            relative_index = relative_index.index_select(1, pixels)
        key_pixels = relative_index.shape[1]
        rpe = self.attn.relative_position_enc_table[relative_index].reshape(kept, key_pixels, self.num_heads, -1)
        qe, ke, ve = repeat(rpe, 'i j h c -> (i a) (j b) h c',
                            a=candidates, b=candidates).chunk(3, -1)
        scale = self.attn.scale
        scores = (coded_q * scale) @ coded_k.transpose(-2, -1)
        scores = scores + torch.einsum('bhic,ijhc->bhij', q * scale, ke)
        scores = scores + torch.einsum('bhjc,ijhc->bhij', k, qe * scale)
        if attn_mask is not None:
            token_ids = (pixels[:, None] * candidates
                         + torch.arange(candidates, device=pixels.device)).flatten()
            compact_mask = attn_mask.index_select(-2, token_ids)
            if not full_kv:
                compact_mask = compact_mask.index_select(-1, token_ids)
            num_windows = (self.H // self.window_size) * (self.W // self.window_size)
            if (compact_mask.ndim != 3 or compact_mask.shape[0] not in (1, num_windows)
                    or compact_mask.shape[1:] != (kept*candidates, key_pixels*candidates)):
                raise ValueError("Expected upstream [windows, tokens, tokens] attention mask")
            if compact_mask.shape[0] != 1:
                compact_mask = compact_mask.repeat(nw // num_windows, 1, 1)
            scores = scores + compact_mask[:, None]
        probs = self.attn.attn_drop(scores.softmax(-1))
        message = probs @ v + torch.einsum('bhij,ijhc->bhic', probs, ve)
        message = rearrange(message, 'b h (p n) d -> (b p) n (h d)', p=kept, n=candidates)
        message = self.proj_drop(self.proj(message))
        shortcut = labels.reshape(nw * kept, candidates, self.dim)
        updated = shortcut + self.drop_path(message)
        if self.normalize_before:
            updated = updated + self.drop_path(self.mlp(self.norm2(updated)))
        else:
            updated = self.norm1(updated)
            updated = self.norm2(updated + self.drop_path(self.mlp(updated)))
        result = all_labels.index_copy(1, pixels, updated.reshape_as(labels))
        return self.unpartition(result)


def install_selective_updates(model, keep_fraction=0.5, coding="continuous",
                              layers=(1, 2, 3), threshold=0.5, selector=LEGACY_SELECTOR,
                              attention_support=COMPACT_SUPPORT):
    """Opt in before optimizer construction; keep entry/exit inference dense.

    Return serializable configuration to store alongside checkpoints. Reload the
    architecture with this configuration before loading its unchanged weight keys.
    Does not modify the CNN, DPN, per-pixel candidate mixer or refinement.
    """
    indices = tuple(layers)
    blocks = model.parent.inference.layers
    if not indices or len(set(indices)) != len(indices) or any(i < 0 or i >= len(blocks) for i in indices):
        raise ValueError("layers must be unique, valid inference-layer indices")
    if selector not in (LEGACY_SELECTOR, ALTERNATING_SELECTOR):
        raise ValueError("Unknown selective-update policy")
    spatial = [blocks[i].nmp for i in indices]
    phases = indices
    selector_config = {}
    architecture = "selective_spatial_nmrf_v1"
    if selector == ALTERNATING_SELECTOR:
        windows = [block.window_size for block in spatial]
        if keep_fraction != .5 or len(set(windows)) != 1 or windows[0] % 2:
            raise ValueError("Alternating columns require half retention and a shared even window size")
        if indices != tuple(sorted(indices)):
            raise ValueError("Alternating columns require layers in increasing order")
        shifts = [block.shift_size for block in spatial]
        parities = [i % 2 for i in range(len(spatial))]
        # The negative roll places global column c at local c - shift.
        phases = [(parity - shift) % 2 for parity, shift in zip(parities, shifts)]
        selector_config = dict(phases=phases, global_column_parities=parities,
                               window_size=windows[0], shifts=shifts)
        architecture = "selective_spatial_nmrf_v2"
    if attention_support == FULL_KV_SUPPORT:
        architecture = "query_selective_full_kv_nmrf_v1"
        selector_config["attention_support"] = attention_support
    # Validate all replacements before changing the model.
    replacements = [SelectiveSwinNMP(block, keep_fraction, coding, phase, threshold, attention_support)
                    for block, phase in zip(spatial, phases)]
    for i, replacement in zip(indices, replacements):
        blocks[i].nmp = replacement
    return dict(architecture=architecture, layers=list(indices),
                keep_fraction=keep_fraction, coding=coding, threshold=threshold,
                selector=selector, surrogate="identity_STE_for_ternary",
                normalization="detached_per_token_head_RMS", temporal_steps=1,
                description="hybrid signed/quantized messages; not a temporal SNN",
                **selector_config)
