"""ssde_sensor_time_hybrid_v1: the prospective S3 hard/CDF pair.

New compact SSDE-inspired architecture, not published SSDEFormer reproduction.
Both modes execute the entire feature and six-layer row-attention graph on all
50 ordered raw binary planes. All 17 neuronal states are explicit, with separate
view/sample entries. No module stores a membrane, temporal sum or diagnostics.
Hard spikes and the actual smooth ATan CDF share a local derivative, not a
forward trajectory. Conventional softmax attention and the analog completion
head are common; this implementation establishes no hardware/energy claim.

Scientific contract SHA256: fc5e1bd185605ea59a433bb6c27df12eec7887da22030a45345d03c2abf5be29
Readout amendment SHA256: afaf610dacf433a58359e0199ddde3873313dbf98ea1fba4bda29e4b3b73d9a2
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


MODES = ("hard", "continuous")
SENSOR_STEPS = 50
CHUNK_STEPS = 10
WIDTH = 64
HEADS = 8
STATE_COUNT = 17


class _HardATan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, margin):
        ctx.save_for_backward(margin)
        return (margin >= 0).to(margin.dtype)

    @staticmethod
    def backward(ctx, gradient):
        (margin,) = ctx.saved_tensors
        return gradient / (1 + (math.pi * margin).square())


def state_step(current, previous, mode):
    """Fixed leak .9, threshold 1, fully differentiated subtractive reset."""
    if mode not in MODES:
        raise ValueError("mode must be hard or continuous")
    if current.dtype != torch.float32 or previous.dtype != torch.float32:
        raise ValueError("S3 currents and states must be float32")
    if current.shape != previous.shape or current.device != previous.device:
        raise ValueError("functional state shape/device mismatch")
    voltage = .9 * previous + current
    margin = voltage - 1
    signal = _HardATan.apply(margin) if mode == "hard" else .5 + torch.atan(math.pi * margin) / math.pi
    return signal, voltage - signal


def column_position(columns, device):
    index = torch.arange(columns, dtype=torch.float32, device=device)[:, None]
    frequency = torch.exp(torch.arange(0, WIDTH, 2, dtype=torch.float32, device=device)
                          * (-math.log(10000.) / WIDTH))
    code = torch.empty(columns, WIDTH, dtype=torch.float32, device=device)
    code[:, 0::2] = torch.sin(index * frequency)
    code[:, 1::2] = torch.cos(index * frequency)
    return code[None]


def global_disparity_expectation(logits, return_probability=False):
    """All j<=i hypotheses, one-key-safe softmax, quarter-cell disparity."""
    if logits.dtype != torch.float32 or logits.shape[-2] != logits.shape[-1]:
        raise ValueError("expected float32 square epipolar matching rows")
    columns = logits.shape[-1]
    index = torch.arange(columns, dtype=logits.dtype, device=logits.device)
    distances = index[:, None] - index[None, :]
    probability = logits.masked_fill(distances < 0, -torch.inf).softmax(-1)
    disparity = (probability * distances).sum(-1)
    return (disparity, probability) if return_probability else disparity


class FeatureStep(nn.Module):
    def __init__(self):
        super().__init__()
        channels = ((1, 32), (32, 64), (64, 64), (64, 64))
        self.convs = nn.ModuleList([nn.Conv2d(cin, cout, 3, stride=2, padding=1, bias=True)
                                    for cin, cout in channels])
        self.norms = nn.ModuleList([nn.GroupNorm(8, cout, eps=1e-5, affine=True) for _, cout in channels])
        self.fuse = nn.Conv2d(192, 64, 1, bias=True)
        self.fuse_norm = nn.GroupNorm(8, 64, eps=1e-5, affine=True)

    def forward(self, plane, states, mode, collect_diagnostics=False):
        if len(states) != 5:
            raise ValueError("five feature states required")
        new_states, signals = [], []
        signal = plane
        for conv, norm, previous in zip(self.convs, self.norms, states[:4]):
            signal, membrane = state_step(norm(conv(signal)), previous, mode)
            new_states.append(membrane)
            signals.append(signal)
        quarter = signals[1]
        fusion = torch.cat((quarter,
                            F.interpolate(signals[2], size=quarter.shape[-2:], mode="bilinear", align_corners=False),
                            F.interpolate(signals[3], size=quarter.shape[-2:], mode="bilinear", align_corners=False)), 1)
        signal, membrane = state_step(self.fuse_norm(self.fuse(fusion)), states[4], mode)
        new_states.append(membrane)
        rates = tuple(value.detach().mean() for value in signals + [signal]) if collect_diagnostics else ()
        return signal, tuple(new_states), rates


class RowStateLayer(nn.Module):
    """One self/cross layer, simultaneous directions, two functional states."""
    def __init__(self, cross):
        super().__init__()
        self.cross = cross
        self.pre_attention_norm = nn.LayerNorm(64, eps=1e-5, elementwise_affine=True)
        self.attention = nn.MultiheadAttention(64, 8, dropout=0., bias=True, batch_first=True)
        self.attention_current_norm = nn.LayerNorm(64, eps=1e-5, elementwise_affine=True)
        self.pre_ff_norm = nn.LayerNorm(64, eps=1e-5, elementwise_affine=True)
        self.ff1 = nn.Linear(64, 128, bias=True)
        self.ff2 = nn.Linear(128, 64, bias=True)
        self.ff_current_norm = nn.LayerNorm(64, eps=1e-5, elementwise_affine=True)

    def forward(self, left, right, states, mode, collect_diagnostics=False):
        normalized_left = self.pre_attention_norm(left)
        normalized_right = self.pre_attention_norm(right)
        key_left, key_right = ((normalized_right, normalized_left) if self.cross
                               else (normalized_left, normalized_right))
        # Both directions read the incoming pair. No mask, including padding.
        attention_left = self.attention(normalized_left, key_left, key_left, need_weights=False)[0]
        attention_right = self.attention(normalized_right, key_right, key_right, need_weights=False)[0]
        current = self.attention_current_norm(torch.cat((left + attention_left, right + attention_right), 0))
        attention_signal, attention_state = state_step(current, states[0], mode)
        feedforward = self.ff2(F.gelu(self.ff1(self.pre_ff_norm(attention_signal)), approximate="none"))
        current = self.ff_current_norm(attention_signal + feedforward)
        signal, feedforward_state = state_step(current, states[1], mode)
        left_next, right_next = signal.split(left.shape[0], dim=0)
        rates = (attention_signal.detach().mean(), signal.detach().mean()) if collect_diagnostics else ()
        return left_next, right_next, (attention_state, feedforward_state), rates


class MatchingCompletionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.match_left = nn.Linear(64, 64, bias=True)
        self.match_right = nn.Linear(64, 64, bias=True)
        self.residual_conv1 = nn.Conv2d(65, 64, 3, stride=1, padding=1, bias=True)
        self.residual_norm = nn.GroupNorm(8, 64, eps=1e-5, affine=True)
        self.residual_conv2 = nn.Conv2d(64, 1, 3, stride=1, padding=1, bias=True)
        nn.init.zeros_(self.residual_conv2.weight)
        nn.init.zeros_(self.residual_conv2.bias)

    def forward(self, left_mean, right_mean, batch, rows, return_fields=False):
        columns = left_mean.shape[1]
        if columns <= 1:
            raise ValueError("completion head requires at least two columns")
        logits = (self.match_left(left_mean) @ self.match_right(right_mean).transpose(-1, -2)) / math.sqrt(64)
        d0, probability = global_disparity_expectation(logits, return_probability=True)
        d0 = d0.reshape(batch, 1, rows, columns)
        left = left_mean.reshape(batch, rows, columns, 64).permute(0, 3, 1, 2)
        residual_input = torch.cat((left, d0 / (columns - 1)), 1)
        residual = self.residual_conv2(F.gelu(self.residual_norm(self.residual_conv1(residual_input)), approximate="none"))
        disparity = d0 + residual
        result = (disparity, d0, residual)
        if return_fields:
            # Same head tensors, no second matching calculation or retained state.
            fields = {"d0": d0.detach(), "r": residual.detach(), "d": disparity.detach(),
                      "p": probability.reshape(batch, rows, columns, columns).detach()}
            return result + (fields,)
        return result


class SensorTimeSSDE(nn.Module):
    def __init__(self, mode):
        super().__init__()
        if mode not in MODES:
            raise ValueError("mode must be hard or continuous")
        self.mode = mode
        self.features = FeatureStep()
        self.layers = nn.ModuleList([RowStateLayer(cross=bool(index % 2)) for index in range(6)])
        self.head = MatchingCompletionHead()

    @staticmethod
    def _padding(height, width):
        extra_h, extra_w = (-height) % 32, (-width) % 32
        return (extra_w // 2, extra_w - extra_w // 2, extra_h // 2, extra_h - extra_h // 2)

    @staticmethod
    def _initial_states(packed):
        views_batch, _, height, width = packed.shape
        feature_shapes = ((32, height // 2, width // 2), (64, height // 4, width // 4),
                          (64, height // 8, width // 8), (64, height // 16, width // 16),
                          (64, height // 4, width // 4))
        features = tuple(packed.new_zeros(views_batch, *shape) for shape in feature_shapes)
        attention = tuple(packed.new_zeros(views_batch * (height // 4), width // 4, 64) for _ in range(12))
        return features + attention

    def _run_chunk(self, planes, *states, collect_diagnostics=False):
        if len(states) != STATE_COUNT:
            raise ValueError("all 17 functional states must cross every chunk boundary")
        batch = planes.shape[0] // 2
        rows, columns = planes.shape[-2] // 4, planes.shape[-1] // 4
        left_sum = planes.new_zeros(batch * rows, columns, 64)
        right_sum = torch.zeros_like(left_sum)
        activity_sum = planes.new_zeros(STATE_COUNT) if collect_diagnostics else None
        position = column_position(columns, planes.device)
        current_states = tuple(states)
        for time_index in range(planes.shape[1]):
            feature, feature_states, rates = self.features(planes[:, time_index:time_index + 1], current_states[:5],
                                                          self.mode, collect_diagnostics)
            tokens = feature.permute(0, 2, 3, 1).reshape(2 * batch * rows, columns, 64)
            left, right = tokens.split(batch * rows, dim=0)
            left, right = left + position, right + position
            next_states = list(feature_states)
            all_rates = list(rates)
            for index, layer in enumerate(self.layers):
                offset = 5 + 2 * index
                left, right, layer_states, layer_rates = layer(left, right, current_states[offset:offset + 2],
                                                              self.mode, collect_diagnostics)
                next_states.extend(layer_states)
                all_rates.extend(layer_rates)
            current_states = tuple(next_states)
            left_sum = left_sum + left
            right_sum = right_sum + right
            if collect_diagnostics:
                activity_sum = activity_sum + torch.stack(all_rates)
        result = (left_sum, right_sum) + current_states
        return result + (activity_sum,) if collect_diagnostics else result

    def forward(self, left, right, checkpointed=True, return_diagnostics=False, return_fields=False):
        """Return native negative flow, optional scalars and eval-only head fields.

        return_fields requires eval mode with gradients disabled and returns
        (flow, scalar_diagnostics, fields). d0/r/d are [B,1,Hq,Wq] in
        quarter-grid disparity cells; p is [B,Hq,Wq,Wq] on the padded
        matching lattice, with j>i probability zero. Nothing is cached.
        """
        if return_fields and (self.training or torch.is_grad_enabled()):
            raise ValueError("head fields require model.eval() and gradients disabled")
        collect_diagnostics = return_diagnostics or return_fields
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            raise TypeError("stereo sensor inputs must be tensors")
        if left.shape != right.shape or left.ndim != 4 or left.shape[1] != SENSOR_STEPS:
            raise ValueError("expected matching complete [B,50,H,W] stereo inputs")
        if any(size == 0 for size in left.shape):
            raise ValueError("batch and spatial dimensions must be nonempty")
        if left.dtype != torch.float32 or right.dtype != torch.float32 or left.device != right.device:
            raise ValueError("stereo inputs must share a device and float32 dtype")
        if next(self.parameters()).dtype != torch.float32:
            raise ValueError("S3 parameters must remain float32")
        if not bool(torch.all((left == 0) | (left == 1))) or not bool(torch.all((right == 0) | (right == 1))):
            raise ValueError("expected raw binary 0/1 planes; do not apply 2*S-1")
        batch, _, height, width = left.shape
        padding = self._padding(height, width)
        with torch.autocast(device_type=left.device.type, enabled=False):
            packed = F.pad(torch.cat((left, right), 0), padding, mode="replicate")
            states = self._initial_states(packed)
            rows, columns = packed.shape[-2] // 4, packed.shape[-1] // 4
            left_sum = packed.new_zeros(batch * rows, columns, 64)
            right_sum = torch.zeros_like(left_sum)
            activity_sum = packed.new_zeros(STATE_COUNT) if collect_diagnostics else None
            for start in range(0, SENSOR_STEPS, CHUNK_STEPS):
                chunk = packed[:, start:start + CHUNK_STEPS]
                if checkpointed and torch.is_grad_enabled():
                    output = checkpoint(self._run_chunk, chunk, *states, use_reentrant=False,
                                        preserve_rng_state=True, collect_diagnostics=collect_diagnostics)
                else:
                    output = self._run_chunk(chunk, *states, collect_diagnostics=collect_diagnostics)
                left_sum, right_sum = left_sum + output[0], right_sum + output[1]
                states = output[2:2 + STATE_COUNT]
                if collect_diagnostics:
                    activity_sum = activity_sum + output[-1]
            head_output = self.head(left_sum / SENSOR_STEPS, right_sum / SENSOR_STEPS, batch, rows,
                                    return_fields=return_fields)
            disparity, d0, residual = head_output[:3]
            padded_flow = 4 * F.interpolate(-disparity, size=packed.shape[-2:], mode="bilinear", align_corners=False)
            flow = padded_flow[..., padding[2]:padding[2] + height, padding[0]:padding[0] + width]
            if not collect_diagnostics:
                return flow
            prefix = "hard_firing_rate" if self.mode == "hard" else "continuous_activation_mean"
            names = [f"feature_F{i}" for i in range(1, 6)] + [f"layer_{i}_{unit}" for i in range(1, 7) for unit in ("A", "F")]
            rates = (activity_sum / SENSOR_STEPS).detach().cpu().tolist()
            diagnostics = {prefix + "_" + name: value for name, value in zip(names, rates)}
            diagnostics.update(sensor_steps=SENSOR_STEPS, d0_mean_coarse=float(d0.detach().mean()),
                               d0_max_coarse=float(d0.detach().max()), residual_abs_mean_coarse=float(residual.detach().abs().mean()),
                               residual_abs_max_coarse=float(residual.detach().abs().max()))
            return (flow, diagnostics, head_output[3]) if return_fields else (flow, diagnostics)


def build_model(mode, seed):
    """Fresh common PyTorch initialization; preserve caller CPU RNG state."""
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(torch.Generator(device="cpu").manual_seed(seed).get_state())
        return SensorTimeSSDE(mode)
