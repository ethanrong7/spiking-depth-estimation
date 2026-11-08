"""Opt-in local binary agreement with a continuous NMRF state update.

Spatial partitioning and inherited block parameters derive from the vendored
NMRF SwinNMP implementation (CVPR 2024). See the vendored attribution and
license under projects/snn-depth-estimation/vendor/nmrf. This is a single-pass
hybrid. The reference path represents bits with floating tensors.
"""
import torch
from torch.nn import functional as F

from nmrf_selective_updates import SelectiveSwinNMP

ARCHITECTURE = 'nmrf_cardinal_feature_agreement_v1'
LAYERS = (1, 2, 3)
SPATIAL_SHIFTS = (0, 3, 0, 3, 0)
NORMALIZER = 3.14 / 64
WINDOW = 6
CANDIDATES = 4
CHANNELS = 128
HEADS = 4
HEAD_WIDTH = 32


def local_pool():
    """Return fixed self, up, down, left, right slots in local window order."""
    positions, exists = [], []
    for p in range(WINDOW**2):
        row, col = divmod(p, WINDOW)
        adjacent = [(row, col), (row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]
        legal = [0 <= y < WINDOW and 0 <= x < WINDOW for y, x in adjacent]
        positions.append([y * WINDOW + x if valid else p for (y, x), valid in zip(adjacent, legal)])
        exists.append(legal)
    return torch.tensor(positions), torch.tensor(exists)


def normalize_content(projected):
    scale = projected.detach().square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
    return projected / scale


def support_features(normalized, mode, training):
    if mode == 'binary':
        hard = (normalized >= 1).to(normalized.dtype)
        if not training:
            return hard
        soft = torch.sigmoid(normalized - 1)
        return soft + (hard - soft).detach()
    return torch.sigmoid(normalized - 1)


class LocalSupportSwinNMP(SelectiveSwinNMP):
    """Replace only the spatial operator; retain the 128-wide residual and FFN."""

    def __init__(self, original, *, support_mode):
        if support_mode not in ('binary', 'continuous'):
            raise ValueError('Support mode must be binary or continuous')
        if (original.window_size != WINDOW or original.dim != CHANNELS or original.num_heads != HEADS
                or original.qkv_dim != CHANNELS + 31 or not original.normalize_before
                or original.attn.attn_drop.p != 0):
            raise ValueError('Expected the native pre-norm four-head spatial block without attention dropout')
        super().__init__(original, keep_fraction=1, coding='continuous')
        self.support_mode = support_mode
        pixels, exists = local_pool()
        device = self.qkv.weight.device
        self.register_buffer('pool_pixels', pixels.to(device), persistent=False)
        self.register_buffer('pool_exists', exists.to(device), persistent=False)
        token_ids = pixels[..., None] * CANDIDATES + torch.arange(CANDIDATES)
        self.register_buffer('pool_tokens', token_ids.flatten(1).to(device), persistent=False)
        relative = self.attn.relative_position_index.gather(1, self.pool_pixels)
        self.register_buffer('pool_relative', relative, persistent=False)
        self._activity = None
        self._fused = None
        self._observe_support = False
        self._support_totals = None

    def enable_fused_inference(self, kernel):
        """Attach a previously prepared evaluation-only CUDA implementation."""
        self._fused = kernel

    def _projection(self, labels, geometry):
        content = self.norm1(labels)
        cut = 2 * self.dim
        bias = self.qkv.bias
        qk = F.linear(content, self.qkv.weight[:cut, :self.dim], None if bias is None else bias[:cut])
        value = F.linear(torch.cat((content, geometry), dim=-1), self.qkv.weight[cut:],
                         None if bias is None else bias[cut:])
        shape = (*labels.shape[:-1], HEADS, HEAD_WIDTH)
        q, k = [normalize_content(part.reshape(shape)) for part in qk.chunk(2, dim=-1)]
        return q, k, value.reshape(shape)

    def _legal_prior(self, geometry, attn_mask):
        windows, pixels, candidates, _ = geometry.shape
        windows_per_image = self.H // WINDOW * (self.W // WINDOW)
        if (attn_mask is None or attn_mask.ndim != 3
                or attn_mask.shape[0] not in (1, windows_per_image)
                or attn_mask.shape[1:] != (pixels * candidates, pixels * candidates)):
            raise ValueError('Expected the original spatial attention mask')
        pool = self.pool_tokens[None, :, None].expand(attn_mask.shape[0], -1, candidates, -1)
        legal = attn_mask.reshape(attn_mask.shape[0], pixels, candidates, -1).gather(-1, pool).isfinite()
        if legal.shape[0] != 1:
            legal = legal.repeat(windows // windows_per_image, 1, 1, 1)
        # All-zero Fourier geometry denotes centered NMP window padding.
        valid = geometry.ne(0).any(dim=-1)
        neighbor_valid = valid[:, self.pool_pixels].flatten(-2)
        pool_exists = self.pool_exists.repeat_interleave(candidates, dim=-1)
        legal = legal & valid[..., None] & neighbor_valid[:, :, None] & pool_exists[None, :, None]
        disparity = geometry[..., -1] * (1 / NORMALIZER)
        local_disparity = disparity[:, self.pool_pixels].flatten(-2)
        difference = (disparity[..., None] - local_disparity[:, :, None]).abs()
        prior = (1 - difference).clamp_min(0).masked_fill(~legal, 0)
        return prior, valid, legal

    def _reference(self, q, k, value, prior, relative_values, valid, legal):
        shape = (*q.shape[:2], -1, HEADS, HEAD_WIDTH)
        local_keys = k[:, self.pool_pixels].reshape(shape)
        local_values = value[:, self.pool_pixels].reshape(shape)
        # Relative values depend on pixel pairs and are shared across candidates.
        relative = relative_values[self.pool_relative].repeat_interleave(CANDIDATES, dim=1)
        local_values = local_values + relative[None]
        agreement = q.transpose(2, 3) @ local_keys.permute(0, 1, 3, 4, 2)
        weights = agreement * prior[:, :, None]
        mass = weights.sum(dim=-1, keepdim=True)
        if self._observe_support:
            self._record_support(q, k, prior, agreement, weights, mass, valid, legal)
        mass = mass.clamp_min(1)
        return ((weights @ local_values.transpose(2, 3)) / mass).transpose(2, 3)

    def _record_support(self, q, k, prior, agreement, weights, mass, valid, legal):
        mask = valid[..., None, None]
        stats = torch.stack((
            (q * mask).sum(), (k * mask).sum(), valid.sum() * HEADS * HEAD_WIDTH,
            legal.sum() * HEADS, (prior > 0).sum() * HEADS,
            ((agreement > 0) & legal[:, :, None]).sum(), (weights > 0).sum(),
            valid.sum() * HEADS, ((mass.squeeze(-1) == 0) & valid[:, :, None]).sum(),
        )).detach().double()
        self._support_totals = stats if self._support_totals is None else self._support_totals + stats

    def forward(self, label_rep, abs_encoding, attn_mask):
        if (self.H is None or self.W is None or self.H % WINDOW or self.W % WINDOW
                or label_rep.ndim != 3 or label_rep.shape[1:] != (CANDIDATES, CHANNELS)
                or abs_encoding.shape != (*label_rep.shape[:-1], 31)):
            raise ValueError('Expected the native padded four-candidate spatial inputs')
        labels = self.partition(label_rep)
        geometry = self.partition(abs_encoding)
        q, k, value = self._projection(labels, geometry)
        prior, valid, legal = self._legal_prior(geometry, attn_mask)
        table = self.attn.relative_position_enc_table.reshape(-1, HEADS, 3, HEAD_WIDTH)
        relative_values = table[:, :, 2]
        if self._fused is not None and not self.training:
            if torch.is_grad_enabled():
                raise RuntimeError('Fused support inference requires no_grad or inference_mode')
            message = self._fused.aggregate(self.support_mode, q, k, value, prior, relative_values,
                                            self.pool_tokens, self.pool_relative)
        else:
            q = support_features(q, self.support_mode, self.training)
            k = support_features(k, self.support_mode, self.training)
            if self.training:
                count = 2 * valid.sum() * HEADS * HEAD_WIDTH
                mask = valid[..., None, None]
                self._activity = ((q * mask).sum() + (k * mask).sum(), count)
            message = self._reference(q, k, value, prior, relative_values, valid, legal)
        projected = self.proj(message.reshape_as(labels))
        hidden = label_rep + self.drop_path(self.proj_drop(self.unpartition(projected)))
        return hidden + self.drop_path(self.mlp(self.norm2(hidden)))

    def take_activity(self):
        activity, self._activity = self._activity, None
        if activity is None:
            raise RuntimeError('No training activity was recorded for this block')
        return activity


def support_metadata(support_mode, layers=LAYERS):
    if support_mode not in ('binary', 'continuous'):
        raise ValueError('Support mode must be binary or continuous')
    layers = tuple(layers)
    if (not layers or any(type(layer) is not int or not 0 <= layer < len(SPATIAL_SHIFTS)
                          for layer in layers) or layers != tuple(sorted(set(layers)))):
        raise ValueError('Support layers must be a nonempty increasing subset of 0–4')
    return dict(architecture=ARCHITECTURE, support_mode=support_mode, layers=list(layers),
                shifts=[SPATIAL_SHIFTS[i] for i in layers],
                window_size=WINDOW, candidates=CANDIDATES, head_width=HEAD_WIDTH,
                neighbor_pool='self_cardinal_original_mask', geometry='unit_tent_in_proposal_units', agreement='raw_head_coincidence_count',
                threshold_rms=1., mass_denominator='max_1_sum_weights', continuous_values=True,
                relative_value=True, dense_own_ffn=True)


def install_binary_support(model, *, support_mode, layers=LAYERS):
    """Install the fixed local operator before optimizer construction or restore."""
    metadata = support_metadata(support_mode, layers)
    stack = model.parent.inference.layers
    indices = metadata['layers']
    if (len(stack) != len(SPATIAL_SHIFTS) or model.parent.num_proposals != CANDIDATES
            or [stack[i].nmp.shift_size for i in indices] != metadata['shifts']):
        raise ValueError('Expected the native five-block inference stack')
    blocks = [LocalSupportSwinNMP(stack[i].nmp, support_mode=support_mode) for i in indices]
    for index, block in zip(indices, blocks):
        stack[index].nmp = block
    return metadata


def take_support_activity(model):
    """Average Q and K activity across the installed local blocks."""
    sums, counts = zip(*(block.take_activity() for block in support_blocks(model)))
    return torch.stack(sums).sum() / torch.stack(counts).sum().clamp_min(1)


SUPPORT_COUNT_KEYS = ('q_active_sum', 'k_active_sum', 'event_denominator', 'legal_edges',
                      'compatible_edges', 'positive_coincidences', 'positive_support_edges',
                      'valid_query_heads', 'null_query_heads')


def support_block_items(model):
    return [(i, layer.nmp) for i, layer in enumerate(model.parent.inference.layers)
            if isinstance(layer.nmp, LocalSupportSwinNMP)]


def support_blocks(model):
    return [block for _, block in support_block_items(model)]


def begin_support_observation(model):
    for block in support_blocks(model):
        if block._fused is not None:
            raise ValueError('Collect edge statistics on the untimed reference path')
        block._support_totals = None
        block._observe_support = True


def end_support_observation(model):
    blocks = support_block_items(model)
    if not blocks or any(block._support_totals is None for _, block in blocks):
        raise RuntimeError('Expected observed reference forwards in every local block')
    values = torch.stack([block._support_totals for _, block in blocks]).cpu().tolist()
    result = {}
    for (layer, block), row in zip(blocks, values):
        counts = dict(zip(SUPPORT_COUNT_KEYS, row))
        for key in SUPPORT_COUNT_KEYS[2:]:
            counts[key] = int(counts[key])
        events, edges, queries = [counts[k] for k in ('event_denominator','legal_edges','valid_query_heads')]
        result[str(layer)] = dict(**counts,
            q_activity=counts['q_active_sum'] / events if events else None,
            k_activity=counts['k_active_sum'] / events if events else None,
            positive_coincidence_density=counts['positive_coincidences'] / edges if edges else None,
            positive_support_density=counts['positive_support_edges'] / edges if edges else None,
            null_query_fraction=counts['null_query_heads'] / queries if queries else None)
        block._observe_support = False
        block._support_totals = None
    return result
