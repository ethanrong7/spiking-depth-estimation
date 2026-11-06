"""Matched site-A coding × routing controls; native placement stays opt-in.

SC is the historical signed/content cost filter. SP uses position-only Q/K,
CC substitutes tanh for signed-rate codes, and CP combines the substitutions.
All cells retain the historical parameter names, shapes, and initialization.
"""
import torch
from torch.nn import functional as F
from core.corr import CorrBlock1D
from core.epipolar_spiking_stereo import signed_rate
from core.spiking_attention_placement import CostFilter

CELLS = ('SC', 'SP', 'CC', 'CP')


class MatchedCostFilter(CostFilter):
    def __init__(self, cell='SC'):
        if cell not in CELLS:
            raise ValueError(f'unknown phase-2 cell: {cell}')
        super().__init__()
        self.cell = cell

    def projected_codes(self, cost):
        """Return rowwise Q/K/V; positional Q/K never consume sample costs."""
        width = cost.shape[-1]
        x = cost.reshape(-1, width, 1)
        position = torch.linspace(-1, 1, width, device=x.device).view(1, width, 1).expand(x.shape[0], -1, -1)
        value_features = self.codes.norm(self.input(torch.cat([x, position], -1)))
        if self.cell in ('SP', 'CP'):
            routing_features = self.codes.norm(self.input(torch.cat([torch.zeros_like(x), position], -1)))
        else:
            routing_features = value_features
        encode = (lambda current: signed_rate(current, 4)) if self.cell in ('SC', 'SP') else torch.tanh
        return (encode(self.codes.q(routing_features)),
                encode(self.codes.k(routing_features)),
                encode(self.codes.v(value_features)))

    @staticmethod
    def _window(x):
        return F.pad(x.transpose(1, 2), (2, 2)).unfold(2, 5, 1).permute(0, 2, 3, 1)

    @staticmethod
    def _weights(q, k):
        width = q.shape[1]
        indices = torch.arange(width, device=q.device)[:, None] + torch.arange(-2, 3, device=q.device)[None, :]
        valid = (indices >= 0) & (indices < width)
        logits = (q[:, :, None, :] * MatchedCostFilter._window(k)).sum(-1) / 4
        return logits.masked_fill(~valid[None], -torch.inf).softmax(-1)

    def routing_weights(self, cost):
        """Diagnostic rows × right-x × five weights with absent candidates zero."""
        q, k, _ = self.projected_codes(cost)
        return self._weights(q, k)

    def compute(self, cost):
        q, k, v = self.projected_codes(cost)
        weights = self._weights(q, k)
        message = (weights[..., None] * self._window(v)).sum(-2)
        x = cost.reshape(-1, cost.shape[-1], 1)
        return (x + self.output(message)).reshape(cost.shape)


def install(model, cell, seed=2017):
    """Install A before pyramid pooling, preserving native state and caller RNG.

    This overrides native CorrBlock1D process-wide just as historical A did;
    workers must run each model in its own process. Paired branch seeds are
    2017, 3128 and 4239 for training seeds 1234, 2345 and 3456.
    """
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        model.placement = MatchedCostFilter(cell)

    class FilteredCorr(CorrBlock1D):
        def __init__(self, left, right, num_levels=4, radius=4):
            self.radius = radius
            self.num_levels = num_levels
            corr = model.placement(CorrBlock1D.corr(left, right))
            b, h, w, _, r = corr.shape
            corr = corr.reshape(b * h * w, 1, 1, r)
            self.corr_pyramid = [corr]
            for _ in range(num_levels):
                corr = F.avg_pool2d(corr, [1, 2], stride=[1, 2])
                self.corr_pyramid.append(corr)

    import core.spikestereonet as native
    native.CorrBlock1D = FilteredCorr
    return model
