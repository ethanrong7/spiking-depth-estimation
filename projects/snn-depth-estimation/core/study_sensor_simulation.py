"""Idealized paired sensor models for normalized-time engineering data.

Intensity model: SpikeStereoNet arXiv:2505.19487v3 A.2.3 (V+=I,
threshold 5, subtractive reset). With dt=1/50 source-frame interval this
is equivalently normalized charge q+=I*dt and charge threshold 5/50.
No measured irradiance/current gain, physical FPS or sensor calibration is
available; encoded-RGB grayscale is a declared proxy, not photon flux.

Event model: independent implementation of log-intensity level crossings,
following the reference semantics in uzh-rpg/rpg_vid2e at commit
 ecbb11a9345bb9d31b4b691e7d82965da4401345, esim_py/src/esim.cpp
(SHA256 05a12180c08bc29d67890ab0e2e7217e11605951579b378bc73bda31e5cd699d).
The source reference is GPL-3.0. No upstream source is vendored here.
This version supports ideal zero-noise/zero-refractory count histograms only;
without refractory, all crossings between adjacent bin edges belong to that
same bin, so storing sub-bin timestamps is unnecessary for these model inputs.
"""
from dataclasses import asdict, dataclass
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SensorConfig:
    T: int = 50
    spike_threshold: float = 5.0
    event_threshold_on: float = 0.2
    event_threshold_off: float = 0.2
    log_epsilon: float = 1e-3
    event_tolerance: float = 1e-6
    noise: float = 0.0
    refractory: float = 0.0

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not math.isfinite(value):
                raise ValueError('Finite sensor parameter required: ' + name)
        if self.T != 50 or self.spike_threshold < 1:
            raise ValueError('All 50 steps and binary-safe spike threshold>=1 required')
        if min(self.event_threshold_on, self.event_threshold_off, self.log_epsilon) <= 0:
            raise ValueError('Positive log epsilon and event thresholds required')
        if self.event_tolerance < 0 or self.noise != 0 or self.refractory != 0:
            raise ValueError('This implementation supports ideal zero-noise/zero-refractory only')

    def metadata(self):
        return {**asdict(self), 'dt_source_frame_interval': 1/self.T,
                'spike_charge_threshold_normalized': self.spike_threshold/self.T,
                'physical_time_calibrated': False, 'radiometric_calibration': False,
                'radiometry': 'encoded_RGB_BT601_no_gamma_linearization',
                'initialization': 'zero_spike_accumulator; event_reference=first_source_log_intensity',
                'state_lifetime': 'continuous_per_sequence_per_view; no_per_window_reset',
                'event_channels': ['ON', 'OFF'],
                'event_reference_commit': 'ecbb11a9345bb9d31b4b691e7d82965da4401345',
                'event_reference_source_sha256': '05a12180c08bc29d67890ab0e2e7217e11605951579b378bc73bda31e5cd699d'}


def rgb_to_gray(rgb):
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError('Expected H,W,RGB float input')
    if not rgb.is_floating_point() or not torch.isfinite(rgb).all() or torch.any((rgb < 0) | (rgb > 1)):
        raise ValueError('Expected finite floating RGB in [0,1]')
    return .299*rgb[..., 0] + .587*rgb[..., 1] + .114*rgb[..., 2]


def registered_output_shape(shape, output_scale):
    """Registered common sensor resolutions; no independent view/modality resize."""
    if output_scale not in (1.0, 0.5, 0.25):
        raise ValueError('Registered output scales are 1, 0.5 and 0.25')
    dimensions = tuple(int(v * output_scale) for v in shape)
    if (len(shape) != 2 or min(dimensions) <= 0
            or any(new != old * output_scale for new, old in zip(dimensions, shape))):
        raise ValueError('Source dimensions must exactly support the registered scale')
    return dimensions


def resample_float_gray(gray, output_scale):
    """Area-average common irradiance proxy BEFORE either sensor nonlinearity.

    Applying this to already binary spikes or ON/OFF events is forbidden by the
    caller contract; both simulators must receive this same floating gray image.
    """
    if gray.ndim != 2 or gray.dtype != torch.float32:
        raise ValueError('Spatial sensor resampling requires FP32 grayscale H,W')
    shape = registered_output_shape(gray.shape, output_scale)
    if output_scale == 1.0:
        return gray
    return F.interpolate(gray[None, None], size=shape, mode='area')[0, 0]


def resample_disparity_nearest(disparity, output_scale):
    """Nearest source target + horizontal pixel-unit scaling, no GT averaging.

    nearest-exact follows source/output pixel-center coordinates. Nonfinite or
    nonpositive values remain available to the loader's fixed validity rule.
    """
    if disparity.ndim != 2 or disparity.dtype != torch.float32:
        raise ValueError('Disparity resampling requires FP32 H,W')
    shape = registered_output_shape(disparity.shape, output_scale)
    if output_scale == 1.0:
        return disparity
    selected = F.interpolate(disparity[None, None], size=shape, mode='nearest-exact')[0, 0]
    return selected * (shape[1] / disparity.shape[1])


class PairedSensorState:
    """One state per sequence/view. step() never resets or detaches neural data.

    This is offline data preparation under no_grad, not a neural encoder. Its
    state is independent of the model's full-gradient sensor-time simulation.
    """
    @torch.no_grad()
    def __init__(self, initial_gray, config=SensorConfig()):
        self.config = config
        self._check(initial_gray)
        self.voltage = torch.zeros_like(initial_gray)
        self.previous_log = torch.log(initial_gray + config.log_epsilon)
        self.reference_log = self.previous_log.clone()
        self.steps = 0

    def _check(self, gray):
        if (gray.ndim != 2 or gray.dtype != torch.float32
                or not torch.isfinite(gray).all() or torch.any((gray < 0) | (gray > 1))):
            raise ValueError('Expected finite FP32 grayscale H,W in [0,1]')
        if hasattr(self, 'voltage') and (gray.shape != self.voltage.shape or gray.device != self.voltage.device):
            raise ValueError('Sensor state shape/device changed')

    @torch.no_grad()
    def step(self, gray):
        self._check(gray)
        config = self.config
        charged = self.voltage + gray
        spikes = charged >= config.spike_threshold
        self.voltage = charged - config.spike_threshold * spikes.to(charged.dtype)
        current = torch.log(gray + config.log_epsilon)
        upward = current >= self.previous_log
        delta = torch.where(upward, config.event_threshold_on, -config.event_threshold_off)
        active = (current - self.previous_log).abs() > config.event_tolerance
        count = torch.zeros_like(gray, dtype=torch.int32)
        # Explicit FP32 threshold additions preserve the reference crossing
        # boundaries and direction/reversal semantics; no signed cancellation.
        bound = math.ceil(math.log((1+config.log_epsilon)/config.log_epsilon)
                          / min(config.event_threshold_on, config.event_threshold_off)) + 2
        for _ in range(bound):
            proposed = self.reference_log + delta
            inside = torch.where(upward,
                                 (proposed > self.previous_log) & (proposed <= current),
                                 (proposed < self.previous_log) & (proposed >= current))
            active = active & inside
            if not bool(active.any()):
                break
            self.reference_log = torch.where(active, proposed, self.reference_log)
            count += active.to(count.dtype)
        else:
            raise RuntimeError('Crossing loop bound exceeded; inspect parameters/state')
        on = torch.where(upward, count, torch.zeros_like(count))
        off = torch.where(upward, torch.zeros_like(count), count)
        self.previous_log = current
        self.steps += 1
        return spikes[None].to(torch.uint8), torch.stack((on, off))
