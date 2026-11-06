"""Disparity-local attention over scalar row-wise projected evidence."""
import torch
from torch import nn
from torch.nn import functional as F


class LocalAttentionEvidence(nn.Module):
    def __init__(self, channels=256, heads=4, levels=4, radius=4, attention=True):
        super().__init__()
        if channels % heads:
            raise ValueError('channels must divide heads')
        self.heads, self.levels, self.radius = heads, levels, radius
        self.attention = attention
        self.query = nn.Conv2d(channels, channels, 1, bias=False)
        self.key = nn.Conv2d(channels, channels, 1, bias=False)
        self.value = nn.Conv2d(channels, channels, 1, bias=False)
        self.readout = nn.Conv2d(heads, 1, 1)

    def prepare(self, left, right):
        """Project once; retain scalar QK/QV volumes, never sampled descriptors."""
        b, c, h, w = left.shape
        def project(layer, feature):
            return layer(feature).reshape(b, self.heads, c // self.heads, h, -1)
        q = project(self.query, left)
        volumes = [torch.einsum('bndhx,bndhy->bnhxy', q, project(layer, right)) /
                   ((c // self.heads) ** 0.5) for layer in (self.key, self.value)]
        pyramid = []
        for level in range(self.levels):
            pyramid.append(tuple(volumes))
            if level + 1 < self.levels:
                volumes = [F.avg_pool2d(v.reshape(-1, 1, 1, v.shape[-1]), (1, 2)).reshape(*v.shape[:-1], -1) for v in volumes]
        return pyramid

    def lookup(self, pyramid, coords):
        b, _, h, w = coords.shape
        offsets = torch.arange(-self.radius, self.radius + 1, device=coords.device, dtype=coords.dtype)
        outputs = []
        for level, (qk, qv) in enumerate(pyramid):
            rw = qk.shape[-1]
            x = coords[:, 0, :, :, None] / (2 ** level) + offsets
            x = x[:, None].expand(-1, self.heads, -1, -1, -1)
            grid = torch.stack((2*x/max(rw-1, 1)-1, torch.zeros_like(x)), -1).reshape(-1, 1, offsets.numel(), 2)
            def sample(volume):
                return F.grid_sample(volume.reshape(-1, 1, 1, rw), grid, align_corners=True).reshape(b, self.heads, h, w, -1)
            logits, values = sample(qk), sample(qv)
            weights = logits.softmax(-1) if self.attention else torch.ones_like(logits)/logits.shape[-1]
            # Every signed offset channel requires attention-weighted value evidence.
            evidence = (weights * values).permute(0, 4, 1, 2, 3).reshape(-1, self.heads, h, w)
            outputs.append(self.readout(evidence).reshape(b, -1, h, w))
        return torch.cat(outputs, 1).float()

    def forward(self, left, right, coords):
        return self.lookup(self.prepare(left, right), coords)
