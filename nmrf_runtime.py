"""Native 50-bin input adaptation for the pinned, continuous NMRF parent."""
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

VENDOR = Path(__file__).resolve().parent / 'projects/snn-depth-estimation/vendor/nmrf'
sys.path.insert(0, str(VENDOR))
from nmrf.config import get_cfg
from nmrf.models import build_model

UPSTREAM_REVISION = 'a50bd8f407ae481c38f95cf141e1b5256e962cb7'


class SpikeInputNMRF(nn.Module):
    """Widen only the CNN stem; retain all 50 observations as input channels.

    This is an ANN receiving sensor spikes, not a conversion to spiking neurons.
    Scaling by 255 lets the untouched upstream encoder normalize to 2*s - 1.
    """
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        old = parent.backbone.conv1
        stem = nn.Conv2d(50, old.out_channels, old.kernel_size, old.stride,
                         old.padding, bias=False)
        nn.init.kaiming_normal_(stem.weight, mode='fan_out', nonlinearity='relu')
        parent.backbone.conv1 = stem

    def forward(self, left, right):
        if left.shape != right.shape or left.ndim != 4 or left.shape[1] != 50:
            raise ValueError('Expected equal [B,50,H,W] left/right spike tensors')
        if self.training and (left.shape[-2] % 8 or left.shape[-1] % 8):
            raise ValueError('Training dimensions must be divisible by 8; use prepare_batch')
        return self.parent({'img1': left.float() * 255, 'img2': right.float() * 255})


def build_native_nmrf(seed=1234):
    torch.manual_seed(seed)
    cfg = get_cfg()
    parent, criterion = build_model(cfg)
    # Upstream build() calls this weight 'proposal_disp', but Criterion emits
    # 'loss_prop'. Correct the lookup in our runner without editing upstream.
    criterion.weight_dict['loss_prop'] = criterion.weight_dict.pop('proposal_disp')
    return SpikeInputNMRF(parent), criterion, cfg


def weighted_loss(criterion, outputs, target, valid):
    mask = valid & (target > 0) & (target < criterion.max_disp)
    if not mask.any():
        raise ValueError('No supervised pixels inside NMRF loss disparity range')
    losses = criterion(outputs, {'disp': target.clone(), 'valid': valid}, log=False)
    missing = set(losses) - set(criterion.weight_dict)
    if missing:
        raise ValueError(f'Unweighted NMRF losses: {sorted(missing)}')
    total = sum(value * criterion.weight_dict[key] for key, value in losses.items())
    if not torch.isfinite(total):
        raise FloatingPointError('Nonfinite NMRF loss')
    return total, losses


def prepare_batch(batch, device, crop=None, random_crop=False):
    """Convert native negative flow to positive NMRF disparity; no resizing.

    Apply identical spatial crops to both views; padding has no valid labels.
    Sensor-time order and all 50 bins are preserved.
    """
    left, right, flow, valid, indices = batch
    target = -flow[:, 0]
    if crop:
        h, w = left.shape[-2:]
        ch, cw = crop
        if not (8 <= ch <= h and 8 <= cw <= w and ch % 8 == cw % 8 == 0):
            raise ValueError('Crop must fit the native image and be divisible by 8')
        y = int(torch.randint(h-ch+1, ()).item()) if random_crop else (h-ch)//2
        x = int(torch.randint(w-cw+1, ()).item()) if random_crop else (w-cw)//2
        left, right = left[..., y:y+ch, x:x+cw], right[..., y:y+ch, x:x+cw]
        target, valid = target[..., y:y+ch, x:x+cw], valid[..., y:y+ch, x:x+cw]
    h, w = target.shape[-2:]
    padding = (0, (-w) % 8, 0, (-h) % 8)
    left = F.pad(left, padding, mode='replicate')
    right = F.pad(right, padding, mode='replicate')
    target = F.pad(target, padding)
    valid = F.pad(valid, padding, value=False)
    return left.to(device), right.to(device), target.to(device), valid.to(device), indices
