"""Prospective S1 input controls on binary sensor planes, before ``2*S-1``.

Both modes accept complete [B, 50, H, W] binary tensors. ``raw`` returns the
original tensor, including its gradient path. ``count_only`` replaces timing by
the canonical binary sequence K_t = ((t+1)*C)//50 - (t*C)//50, where C is the
per-pixel count. Shape, per-pixel counts, dtype and device are preserved; no
input tensor or source data is written.

Canonical coding creates artificial count-dependent flicker and stereo
synchrony. A train-matched comparison measures the utility of information beyond
counts under this representation/conditioning choice; it does not isolate
chronological order or establish a benefit of spiking neurons. This is a
discrete data transform, not a differentiable model operation. Value checks
reject nonbinary/normalized inputs; an all-ones tensor alone cannot establish
whether normalization already happened, so callers must enforce the boundary.
"""

import torch


MODES = ("raw", "count_only")
SENSOR_BINS = 50


def _validate_mode(mode):
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")


def _validate_input(spikes):
    if not isinstance(spikes, torch.Tensor):
        raise TypeError("sensor input must be a torch.Tensor")
    if spikes.layout != torch.strided or spikes.is_quantized or spikes.is_complex():
        raise TypeError("sensor input must be a dense real numeric or bool tensor")
    if spikes.ndim != 4 or spikes.shape[1] != SENSOR_BINS:
        raise ValueError("expected complete binary [B,50,H,W] sensor input")
    if any(size == 0 for size in spikes.shape):
        raise ValueError("sensor batch and spatial dimensions must be nonempty")
    if not bool(torch.all((spikes == 0) | (spikes == 1))):
        raise ValueError("expected binary 0/1 sensor input before 2*S-1 normalization")


def _transform_validated(spikes, mode):
    if mode == "raw":
        return spikes
    counts = spikes.sum(dim=1, keepdim=True, dtype=torch.int64)
    steps = torch.arange(SENSOR_BINS, device=spikes.device, dtype=torch.int64)
    steps = steps.reshape(1, SENSOR_BINS, 1, 1)
    code = ((steps + 1) * counts) // SENSOR_BINS - (steps * counts) // SENSOR_BINS
    return code.to(dtype=spikes.dtype)


def transform_input(spikes: torch.Tensor, mode: str) -> torch.Tensor:
    """Validate and encode a complete binary sensor batch without mutation.

    Validation reads all values; on a GPU this entails synchronization. Apply
    this data transform before device transfer when the driver loads CPU data.
    """
    _validate_mode(mode)
    _validate_input(spikes)
    return _transform_validated(spikes, mode)


def transform_stereo(left: torch.Tensor, right: torch.Tensor, mode: str):
    """Encode a stereo pair with identical batch, geometry, dtype and device."""
    _validate_mode(mode)
    _validate_input(left)
    _validate_input(right)
    if left.shape != right.shape:
        raise ValueError("left/right sensor shapes must match exactly")
    if left.dtype != right.dtype or left.device != right.device:
        raise ValueError("left/right sensor dtype and device must match")
    return _transform_validated(left, mode), _transform_validated(right, mode)
