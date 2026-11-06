"""Portable entry points for the extracted models; no historical run dependencies."""
import os
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
import sys
from pathlib import Path
from types import SimpleNamespace
import random
import numpy as np
import torch
from torch.utils.data import Dataset

PROJECT = Path(__file__).resolve().parent / 'projects/snn-depth-estimation'
sys.path.insert(0, str(PROJECT))
from core.utils.utils import InputPadder
from core.utils.reader import get_block_spikes, read_depth

RECURRENT = ('spikestereonet', 'v0', 'v0-no-attention', 'v0-guided',
             'placement-a', 'placement-b', 'placement-c', 'epipolar', 'ssde-inspired')
DIRECT = ('s2m2-spiking', 's2m2-v2', 's2m2-continuous', 'coordinate',
          'coordinate-e3', 'ssde-direct', 'spiking-correspondence')
MODELS = RECURRENT + DIRECT


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(name, seed=1234):
    if name not in MODELS:
        raise ValueError(f'Unknown model: {name}')
    seed_all(seed)
    if name in RECURRENT:
        from core.spikestereonet import SpikeStereoNet
        args = SimpleNamespace(hidden_dims=[128]*3, context_norm='batch',
            shared_backbone=False, n_snn_layers=3, n_downsample=2,
            corr_levels=4, corr_radius=4, corr_implementation='reg',
            mixed_precision=False, feature_encoder='cnn', context_encoder='cnn',
            matched_initialization_seed=seed, slow_fast_gru=False)
        # Placement A installs a process-local correlation class. Reset it when
        # constructing another model; the CLI owns one model per process.
        import core.spikestereonet as native
        from core.corr import CorrBlock1D
        native.CorrBlock1D = CorrBlock1D
        if name == 'epipolar':
            from core.epipolar_spiking_stereo import EpipolarSpikingStereoNet
            return EpipolarSpikingStereoNet(args)
        if name == 'ssde-inspired':
            from core.ssdeformer_inspired import SSDEFormerInspired
            return SSDEFormerInspired(args)
        reference = SpikeStereoNet(args)
        if name.startswith('placement-'):
            from core.spiking_attention_placement import install
            return install(reference, name[-1].upper())
        if name == 'spikestereonet':
            return reference
        from core.spiking_transformer_stereo import SpikingTransformerStereo
        torch.manual_seed(seed+777)
        model = SpikingTransformerStereo(args, normalize_residual_current=True)
        model.load_common_state(reference)
        if name == 'v0-guided':
            from core.guided_stereo_attention import install_guidance
            return install_guidance(model)
        if name == 'v0-no-attention':
            for cell in model.update_block.cells:
                cell.attention_enabled = False
                cell.qkv.requires_grad_(False)
                cell.output_proj.requires_grad_(False)
                cell.attention_gate.requires_grad_(False)
        return model
    if name == 's2m2-spiking':
        from core.s2m2_spiking import SpikingS2M2
        return SpikingS2M2()
    if name in ('s2m2-v2', 's2m2-continuous'):
        from core.s2m2_spiking_v2 import SpikingS2M2V2
        model = SpikingS2M2V2()
        if name.endswith('continuous'):
            from core.s2m2_continuous_twin import apply_continuous_codes
            model = apply_continuous_codes(model)
        return model
    if name == 'coordinate':
        from core.coordinate_transformer import CoordinateStereo
        return CoordinateStereo()
    if name == 'coordinate-e3':
        from core.coordinate_transformer_e3 import RangeCoordinateStereo
        return RangeCoordinateStereo()
    if name == 'ssde-direct':
        from core.ssde_direct_attention import SSDEDirectAttention
        return SSDEDirectAttention(seed=seed)
    from core.spiking_stereo_correspondence import SpikingStereoCorrespondence
    return SpikingStereoCorrespondence()


def reset(model):
    from spikingjelly.activation_based import functional
    update = getattr(model, 'update_block', None)
    if hasattr(update, 'reset_state'):
        update.reset_state()
    else:
        functional.reset_net(model)


def forward(model, name, left, right, iterations=32, training=False):
    """Return native-resolution signed flow x_right-x_left plus loss auxiliaries."""
    reset(model)
    if name in RECURRENT:
        padder = InputPadder(left.shape, divis_by=32)
        left, right = padder.pad(left, right)
        out = model(left, right, iters=iterations, test_mode=not training)
        if training:
            predictions = [padder.unpad(p) for p in out[0]]
            return predictions[-1], predictions
        return padder.unpad(out[1]), None
    out = model(left, right)
    return out['flow'], out


def loss(name, auxiliaries, target, valid):
    if name in RECURRENT:
        mask = valid[:, None].bool() & torch.isfinite(target) & (target.abs() < 700)
        if not mask.any():
            raise ValueError('No valid training pixels')
        predictions = auxiliaries
        gamma = .9**(15/max(1, len(predictions)-1))
        return sum(gamma**(len(predictions)-i-1)*(p-target).abs()[mask].mean()
                   for i, p in enumerate(predictions))
    if name in ('coordinate', 'coordinate-e3'):
        from core.coordinate_transformer import coordinate_loss as function
    elif name in ('s2m2-v2', 's2m2-continuous'):
        from core.s2m2_spiking_v2 import s2m2_v2_loss as function
    elif name == 'spiking-correspondence':
        from core.spiking_stereo_correspondence import correspondence_loss as function
    else:
        from core.ssde_direct_attention import direct_loss as function
    return function(auxiliaries, target, valid)[0]


def load_spikes(path):
    path = Path(path)
    values = np.load(path, allow_pickle=False) if path.suffix == '.npy' else get_block_spikes(str(path))
    if values.ndim != 3 or values.shape[0] != 50:
        raise ValueError('Expected [50,H,W] spike input; raw .dat uses native 250x400 geometry')
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite spike input')
    return torch.from_numpy(values.astype(np.float32))


class NativeDataset(Dataset):
    """Native Synthetic files, calibration and signed-flow convention."""
    def __init__(self, root, rows):
        self.root, self.rows = Path(root), rows
        if not rows:
            raise ValueError('Empty dataset split')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        left = load_spikes(self.root/row['spike_l'])
        right = load_spikes(self.root/row['spike_r'])
        depth = read_depth(str(self.root/row['depth'])).astype(np.float32)
        valid = np.isfinite(depth) & (depth > 0)
        flow = np.zeros_like(depth)
        flow[valid] = -(289.740625*.08)/depth[valid]
        valid &= np.abs(flow) < 512
        if left.shape != right.shape or left.shape[-2:] != depth.shape:
            raise ValueError('Spike/depth shapes disagree')
        return left, right, torch.from_numpy(flow[None]), torch.from_numpy(valid), index
