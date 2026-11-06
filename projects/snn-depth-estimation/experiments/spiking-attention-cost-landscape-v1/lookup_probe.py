"""Passive, external instrumentation of native decoder-facing cost lookup.

Install after the frozen model's placement installer. No model/state changes,
dataset access, decoder interventions, or training are performed here.
"""
from contextlib import contextmanager

import torch


def _moments(values, support):
    selected = values.detach()[support]
    if not selected.numel():
        return dict(count=0, mean=None, abs_mean=None, rms=None)
    selected = selected.double()
    return dict(count=selected.numel(), mean=selected.mean().item(),
                abs_mean=selected.abs().mean().item(),
                rms=selected.square().mean().sqrt().item())


def summarize_lookup(block, coords, center, minus, plus, mask, epsilon, lower_coords, upper_coords):
    """Aggregate by pyramid level; epsilon is in base feature-cell units.

    mask is an evaluator-supplied B,H,W boolean support. This helper cannot
    infer visibility, native valid support, or a GT correspondence from it.
    Finite-difference support requires every contributing sample in bounds.
    """
    b, _, h, w = center.shape
    if mask.shape != (b, h, w) or mask.dtype != torch.bool:
        raise ValueError('explicit boolean B,H,W coarse support required')
    if mask.device != center.device:
        raise ValueError('support and lookup must share a device')
    k = 2 * block.radius + 1
    expected = (b, block.num_levels * k, h, w)
    if any(t.shape != expected for t in (center, minus, plus)):
        raise ValueError('lookup channel layout differs from native levels/windows')
    if not all(torch.isfinite(t).all() for t in (center, minus, plus)):
        raise ValueError('nonfinite lookup')
    profile = [t.detach().reshape(b, block.num_levels, k, h, w)
               for t in (center, minus, plus)]
    c, lo, hi = profile
    # FP rounding can make the two actual offsets asymmetric or zero. Use
    # measured offsets in base feature units; zero steps have empty FD support.
    a = (coords[:, :1].detach().double() - lower_coords[:, :1].double())[:, None]
    z = (upper_coords[:, :1].double() - coords[:, :1].detach().double())[:, None]
    positive_steps = (a > 0) & (z > 0)
    safe_a = torch.where(positive_steps, a, torch.ones_like(a))
    safe_z = torch.where(positive_steps, z, torch.ones_like(z))
    denominator = safe_a + safe_z
    # Nonuniform three-point first/second differences at the center.
    slope = ((safe_a / safe_z) * (hi.double() - c) +
             (safe_z / safe_a) * (c.double() - lo)) / denominator
    curvature = 2 * ((hi.double() - c) / safe_z - (c.double() - lo) / safe_a) / denominator
    offsets = torch.arange(-block.radius, block.radius + 1,
                           device=center.device).view(1, k, 1, 1)
    rows = []
    for level in range(block.num_levels):
        scale = 2**level
        x = coords[:, :1].detach() / scale + offsets
        width = block.corr_pyramid[level].shape[-1]
        support = mask[:, None].expand(-1, k, -1, -1)
        interior = (support & positive_steps[:, 0] &
                    (x - a[:, 0] / scale >= 0) & (x + z[:, 0] / scale <= width - 1))
        whole_profile = interior.all(1)
        rows.append(dict(
            level=level, width=width, epsilon_base_feature_cells=epsilon,
            lookup=_moments(c[:, level], support),
            inbounds_lookup=_moments(c[:, level], interior),
            slope_per_base_feature_cell=_moments(slope[:, level], interior),
            curvature_per_base_feature_cell_squared=_moments(curvature[:, level], interior),
            profile_std=_moments(c[:, level].std(1, unbiased=False), whole_profile),
            profile_range=_moments(c[:, level].amax(1) - c[:, level].amin(1), whole_profile),
            support_note='explicit coarse support; no inferred GT rank or visibility'))
    return rows


@contextmanager
def observe_lookups(native_module, support_provider, sink, epsilon=0.25):
    """Temporarily instrument the resolved CorrBlock1D, preserving its output.

    support_provider(coords) must return the frozen evaluator's coarse support.
    sink receives one record per actual lookup call, not one per decoder prefix.
    The caller must label frame/batch/iteration identity and aggregate scenes.
    One model per process, as required by the existing placement installer.
    """
    if not epsilon > 0 or not torch.isfinite(torch.tensor(epsilon)):
        raise ValueError('finite positive epsilon required')
    original = native_module.CorrBlock1D
    call = original.__call__

    class ObservedCorr(original):
        def __call__(self, coords):
            output = call(self, coords)
            with torch.no_grad():
                lower, upper = coords.detach().clone(), coords.detach().clone()
                lower[:, 0] -= epsilon
                upper[:, 0] += epsilon
                minus, plus = call(self, lower), call(self, upper)
                mask = support_provider(coords)
                levels = summarize_lookup(self, coords, output, minus, plus,
                                          mask, epsilon, lower, upper)
                sink(dict(levels=levels,
                          coordinates_base_feature_cells=coords.detach().cpu().clone(),
                          actual_lower_delta=(coords[:, :1] - lower[:, :1]).detach().cpu(),
                          actual_upper_delta=(upper[:, :1] - coords[:, :1]).detach().cpu(),
                          coarse_support=mask.detach().cpu().clone()))
            return output

    native_module.CorrBlock1D = ObservedCorr
    try:
        yield
    finally:
        native_module.CorrBlock1D = original
