"""Functional-state SEW-ADD encoders for ordered spike streams.

The modules in this file mirror the spatial widths, strides, residual-block
counts, and output interfaces of ``extractor.py``.  A temporal input has shape
``[batch, time, height, width]`` and is consumed in increasing time order.

The default consumes one sensor bin per simulation step. The opt-in
``static_channels`` mode repeats the complete channel stack at each simulation
step, allowing the number of sensor bins and neuron updates to differ.

Every LIF uses the following recurrence (v_reset = 0)::

    h[t] = v[t-1] + (x[t] - v[t-1]) / tau
    s[t] = ATan_alpha_2(h[t] - 1)
    v[t] = h[t] * (1 - s[t])

``tau`` is 2, input decay is enabled, the reset is hard and participates in
autograd, and the forward spike is binary.  Membrane tensors are explicit
function inputs and outputs rather than module buffers.  This makes temporal
checkpoint replay stateless and preserves the full temporal gradient.
"""

from typing import Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from spikingjelly.activation_based import surrogate


def _normalization(norm_fn: str, channels: int) -> nn.Module:
    if norm_fn == "group":
        return nn.GroupNorm(num_groups=channels // 8, num_channels=channels)
    if norm_fn == "batch":
        return nn.BatchNorm2d(channels)
    if norm_fn == "instance":
        return nn.InstanceNorm2d(channels)
    if norm_fn == "none":
        return nn.Sequential()
    raise ValueError(f"Unknown normalization: {norm_fn}")


class FunctionalLIF(nn.Module):
    """Single functional LIF step with no persistent membrane buffer."""

    def __init__(self):
        super().__init__()
        self.tau = 2.0
        self.v_threshold = 1.0
        self.v_reset = 0.0
        self.decay_input = True
        self.detach_reset = False
        self.surrogate_function = surrogate.ATan(alpha=2.0)

    def forward(self, x: torch.Tensor, v: torch.Tensor):
        h = v + (x - v) / self.tau
        spike = self.surrogate_function(h - self.v_threshold)
        v_next = h * (1.0 - spike)
        return spike, v_next


class _StateCursor:
    def __init__(self, states: Sequence[torch.Tensor]):
        self.states = states
        self.index = 0
        self.output = []

    def take(self, like: torch.Tensor) -> torch.Tensor:
        if len(self.states) == 0:
            value = torch.zeros_like(like)
        else:
            if self.index >= len(self.states):
                raise RuntimeError("SEW membrane-state tuple is incomplete")
            value = self.states[self.index]
            if value.shape != like.shape:
                raise RuntimeError(
                    f"SEW membrane-state shape {tuple(value.shape)} does not match "
                    f"activation shape {tuple(like.shape)}"
                )
        self.index += 1
        return value

    def put(self, value: torch.Tensor) -> None:
        self.output.append(value)

    def finish(self) -> Tuple[torch.Tensor, ...]:
        if len(self.states) != 0 and self.index != len(self.states):
            raise RuntimeError(
                f"SEW membrane-state tuple has {len(self.states) - self.index} unused tensors"
            )
        return tuple(self.output)


class SEWResidualBlock(nn.Module):
    """Two spiking convs and a SEW-ADD shortcut, with no spike after ADD."""

    def __init__(self, in_planes, planes, norm_fn="group", stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, padding=1, stride=stride
        )
        self.norm1 = _normalization(norm_fn, planes)
        self.lif1 = FunctionalLIF()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.norm2 = _normalization(norm_fn, planes)
        self.lif2 = FunctionalLIF()

        if stride == 1 and in_planes == planes:
            self.shortcut = None
        else:
            self.shortcut = nn.Conv2d(
                in_planes, planes, kernel_size=1, stride=stride
            )
            self.shortcut_norm = _normalization(norm_fn, planes)
            self.shortcut_lif = FunctionalLIF()
        self.state_count = 2 + int(self.shortcut is not None)

    @staticmethod
    def _lif_step(lif, value, cursor):
        spike, membrane = lif(value, cursor.take(value))
        cursor.put(membrane)
        return spike

    def step(self, x: torch.Tensor, cursor: _StateCursor) -> torch.Tensor:
        y = self._lif_step(self.lif1, self.norm1(self.conv1(x)), cursor)
        y = self._lif_step(self.lif2, self.norm2(self.conv2(y)), cursor)

        if self.shortcut is None:
            shortcut = x
        else:
            shortcut = self._lif_step(
                self.shortcut_lif,
                self.shortcut_norm(self.shortcut(x)),
                cursor,
            )
        return shortcut + y


class _SEWLayer(nn.Module):
    def __init__(self, in_planes, planes, norm_fn, stride):
        super().__init__()
        self.block1 = SEWResidualBlock(in_planes, planes, norm_fn, stride)
        self.block2 = SEWResidualBlock(planes, planes, norm_fn, 1)

    def step(self, x, cursor):
        return self.block2.step(self.block1.step(x, cursor), cursor)


class _TemporalSEWMixin:
    timesteps: int
    checkpoint_chunk_size: int
    _temporal_output_count: int

    def _apply_stateful(self, step_function, x, states, offset, count):
        if len(states) == 0:
            input_states = ()
        else:
            input_states = states[offset : offset + count]
            if len(input_states) != count:
                raise RuntimeError("SEW membrane-state tuple is incomplete")

        def run(value, *membranes):
            cursor = _StateCursor(membranes)
            output = step_function(value, cursor)
            return (output,) + cursor.finish()

        use_spatial_checkpoint = (
            self.spatial_checkpoint and self.training and torch.is_grad_enabled()
        )
        if use_spatial_checkpoint:
            result = checkpoint(
                run,
                x,
                *input_states,
                use_reentrant=False,
                preserve_rng_state=True,
            )
        else:
            result = run(x, *input_states)
        return result[0], tuple(result[1:]), offset + count

    @staticmethod
    def _finish_stateful_step(states, offset, outputs):
        if len(states) != 0 and offset != len(states):
            raise RuntimeError(
                f"SEW membrane-state tuple has {len(states) - offset} unused tensors"
            )
        return tuple(outputs)

    def _check_input(self, x: torch.Tensor) -> None:
        if x.ndim != 4:
            raise ValueError(f"Expected [B,T,H,W], received {tuple(x.shape)}")
        expected = self.input_bins if self.input_mode == "static_channels" else self.timesteps
        if x.shape[1] != expected:
            raise ValueError(
                f"SEW encoder requires exactly {expected} input bins; "
                f"received {x.shape[1]}"
            )

    def _check_checkpoint_safety(self) -> None:
        training_bn = [
            name
            for name, module in self.named_modules()
            if isinstance(module, nn.BatchNorm2d) and module.training
        ]
        if training_bn:
            raise RuntimeError(
                "Temporal checkpoint replay requires frozen BatchNorm modules; "
                f"training modules: {training_bn[:3]}"
            )

    def _run_chunk(self, x_chunk, *states):
        sums = None
        current_states = states
        for time_index in range(x_chunk.shape[1]):
            frame = (x_chunk[:, time_index] if self.input_mode == "static_channels"
                     else x_chunk[:, time_index : time_index + 1])
            outputs, current_states = self._single_step(
                frame, current_states
            )
            if sums is None:
                sums = list(outputs)
            else:
                sums = [total + value for total, value in zip(sums, outputs)]
        return tuple(sums) + tuple(current_states)

    def _temporal_forward(self, x):
        self._check_input(x)
        if self.input_mode == "static_channels":
            # A view: all input bins are reused at each neuron step, never dropped.
            x = x.unsqueeze(1).expand(-1, self.timesteps, -1, -1, -1)
        use_checkpoint = (
            self.checkpoint_chunk_size > 0
            and self.training
            and torch.is_grad_enabled()
        )
        if use_checkpoint or (
            self.spatial_checkpoint and self.training and torch.is_grad_enabled()
        ):
            self._check_checkpoint_safety()
        # Keep floating-point summation association identical between the
        # no-grad sufficient-statistics pass and checkpointed gradient replay.
        # Checkpointing itself still runs only when gradients are enabled.
        if self.checkpoint_chunk_size > 0 and self.training:
            chunk_size = self.checkpoint_chunk_size
        else:
            chunk_size = self.timesteps

        states = ()
        output_sums = None
        for start in range(0, self.timesteps, chunk_size):
            x_chunk = x[:, start : start + chunk_size]
            if use_checkpoint:
                result = checkpoint(
                    self._run_chunk,
                    x_chunk,
                    *states,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
            else:
                result = self._run_chunk(x_chunk, *states)
            chunk_outputs = result[: self._temporal_output_count]
            states = result[self._temporal_output_count :]
            if output_sums is None:
                output_sums = list(chunk_outputs)
            else:
                output_sums = [
                    total + value for total, value in zip(output_sums, chunk_outputs)
                ]
        return tuple(value / self.timesteps for value in output_sums)


class SpikingBasicEncoder(_TemporalSEWMixin, nn.Module):
    """SEW counterpart of ``BasicEncoder`` with the same output interface."""

    def __init__(
        self,
        output_dim=128,
        norm_fn="batch",
        dropout=0.0,
        downsample=3,
        timesteps=50,
        checkpoint_chunk_size=0,
        spatial_checkpoint=False,
        input_mode="sensor_time",
        input_bins=50,
    ):
        super().__init__()
        self.norm_fn = norm_fn
        self.downsample = downsample
        self.timesteps = timesteps
        if input_mode not in ("sensor_time", "static_channels") or timesteps < 1:
            raise ValueError("invalid encoder input mode or simulation length")
        self.input_mode = input_mode
        self.input_bins = input_bins
        self.checkpoint_chunk_size = checkpoint_chunk_size
        self.spatial_checkpoint = spatial_checkpoint
        self._temporal_output_count = 1

        self.conv1 = nn.Conv2d(
            input_bins if input_mode == "static_channels" else 1,
            64, kernel_size=7, stride=1 + (downsample > 2), padding=3
        )
        self.norm1 = _normalization(norm_fn, 64)
        self.lif1 = FunctionalLIF()
        self.layer1 = _SEWLayer(64, 64, norm_fn, 1)
        self.layer2 = _SEWLayer(64, 96, norm_fn, 1 + (downsample > 1))
        self.layer3 = _SEWLayer(96, 128, norm_fn, 1 + (downsample > 0))
        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else None
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(
                module, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)
            ):
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def _stem_step(self, x, cursor):
        stem = self.norm1(self.conv1(x))
        x, membrane = self.lif1(stem, cursor.take(stem))
        cursor.put(membrane)
        return x

    def _single_step(self, x, states):
        offset = 0
        output_states = []
        x, new_states, offset = self._apply_stateful(
            self._stem_step, x, states, offset, 1
        )
        output_states.extend(new_states)
        for block in (
            self.layer1.block1,
            self.layer1.block2,
            self.layer2.block1,
            self.layer2.block2,
            self.layer3.block1,
            self.layer3.block2,
        ):
            x, new_states, offset = self._apply_stateful(
                block.step, x, states, offset, block.state_count
            )
            output_states.extend(new_states)
        x = self.conv2(x)
        if self.training and self.dropout is not None:
            x = self.dropout(x)
        return (x,), self._finish_stateful_step(states, offset, output_states)

    def forward(self, x, dual_inp=False):
        if dual_inp:
            raise ValueError("SpikingBasicEncoder does not implement dual_inp")
        is_list = isinstance(x, (tuple, list))
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)
        (x,) = self._temporal_forward(x)
        if is_list:
            x = x.split(split_size=batch_dim, dim=0)
        return x


class _SEWOutputHead(nn.Module):
    def __init__(self, output_dim, norm_fn):
        super().__init__()
        self.residual = SEWResidualBlock(128, 128, norm_fn, stride=1)
        self.projection = nn.Conv2d(128, output_dim, 3, padding=1)

    def step(self, x, cursor):
        return self.projection(self.residual.step(x, cursor))


class SpikingMultiBasicEncoder(_TemporalSEWMixin, nn.Module):
    """SEW counterpart of ``MultiBasicEncoder`` with multiscale outputs."""

    def __init__(
        self,
        output_dim=((128, 128, 128),),
        norm_fn="batch",
        dropout=0.0,
        downsample=3,
        timesteps=50,
        checkpoint_chunk_size=0,
        spatial_checkpoint=False,
        input_mode="sensor_time",
        input_bins=50,
    ):
        super().__init__()
        self.norm_fn = norm_fn
        self.downsample = downsample
        self.timesteps = timesteps
        if input_mode not in ("sensor_time", "static_channels") or timesteps < 1:
            raise ValueError("invalid encoder input mode or simulation length")
        self.input_mode = input_mode
        self.input_bins = input_bins
        self.checkpoint_chunk_size = checkpoint_chunk_size
        self.spatial_checkpoint = spatial_checkpoint
        self._temporal_output_count = 3 * len(output_dim)

        self.conv1 = nn.Conv2d(
            input_bins if input_mode == "static_channels" else 1,
            64, kernel_size=7, stride=1 + (downsample > 2), padding=3
        )
        self.norm1 = _normalization(norm_fn, 64)
        self.lif1 = FunctionalLIF()
        self.layer1 = _SEWLayer(64, 64, norm_fn, 1)
        self.layer2 = _SEWLayer(64, 96, norm_fn, 1 + (downsample > 1))
        self.layer3 = _SEWLayer(96, 128, norm_fn, 1 + (downsample > 0))
        self.layer4 = _SEWLayer(128, 128, norm_fn, 2)
        self.layer5 = _SEWLayer(128, 128, norm_fn, 2)
        self.outputs08 = nn.ModuleList(
            [_SEWOutputHead(dim[2], norm_fn) for dim in output_dim]
        )
        self.outputs16 = nn.ModuleList(
            [_SEWOutputHead(dim[1], norm_fn) for dim in output_dim]
        )
        self.outputs32 = nn.ModuleList(
            [nn.Conv2d(128, dim[0], 3, padding=1) for dim in output_dim]
        )
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else None
        self._dual_inp = False
        self._num_layers = 3
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(
                module, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)
            ):
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    @staticmethod
    def _lif_step(lif, value, cursor):
        spike, membrane = lif(value, cursor.take(value))
        cursor.put(membrane)
        return spike

    def _stem_step(self, x, cursor):
        return self._lif_step(self.lif1, self.norm1(self.conv1(x)), cursor)

    def _single_step(self, x, states):
        offset = 0
        output_states = []
        x, new_states, offset = self._apply_stateful(
            self._stem_step, x, states, offset, 1
        )
        output_states.extend(new_states)
        for block in (
            self.layer1.block1,
            self.layer1.block2,
            self.layer2.block1,
            self.layer2.block2,
            self.layer3.block1,
            self.layer3.block2,
        ):
            x, new_states, offset = self._apply_stateful(
                block.step, x, states, offset, block.state_count
            )
            output_states.extend(new_states)
        if self._dual_inp:
            shared = x
            x = x[: x.shape[0] // 2]
        outputs08 = []
        for head in self.outputs08:
            value, new_states, offset = self._apply_stateful(
                head.step, x, states, offset, head.residual.state_count
            )
            output_states.extend(new_states)
            outputs08.append(value)

        if self._num_layers >= 2:
            x16 = x
            for block in (self.layer4.block1, self.layer4.block2):
                x16, new_states, offset = self._apply_stateful(
                    block.step, x16, states, offset, block.state_count
                )
                output_states.extend(new_states)
            outputs16 = []
            for head in self.outputs16:
                value, new_states, offset = self._apply_stateful(
                    head.step, x16, states, offset, head.residual.state_count
                )
                output_states.extend(new_states)
                outputs16.append(value)
        else:
            outputs16 = [torch.zeros_like(value) for value in outputs08]

        if self._num_layers >= 3:
            x32 = x16
            for block in (self.layer5.block1, self.layer5.block2):
                x32, new_states, offset = self._apply_stateful(
                    block.step, x32, states, offset, block.state_count
                )
                output_states.extend(new_states)
            outputs32 = [head(x32) for head in self.outputs32]
        else:
            outputs32 = [torch.zeros_like(value) for value in outputs08]

        outputs = outputs08 + outputs16 + outputs32
        if self._dual_inp:
            outputs.append(shared)
        return tuple(outputs), self._finish_stateful_step(
            states, offset, output_states
        )

    def forward(self, x, dual_inp=False, num_layers=3):
        if num_layers != 3:
            raise ValueError("SEW feasibility encoder currently requires num_layers=3")
        self._dual_inp = dual_inp
        self._num_layers = num_layers
        expected_outputs = self._temporal_output_count + int(dual_inp)
        original_count = self._temporal_output_count
        self._temporal_output_count = expected_outputs
        try:
            outputs = self._temporal_forward(x)
        finally:
            self._temporal_output_count = original_count

        group = len(self.outputs08)
        outputs08 = list(outputs[:group])
        outputs16 = list(outputs[group : 2 * group])
        outputs32 = list(outputs[2 * group : 3 * group])
        result = (outputs08, outputs16, outputs32)
        if dual_inp:
            result = result + (outputs[-1],)
        return result
