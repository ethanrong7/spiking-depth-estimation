"""Controlled v2b loss change: supervise conditional post-Sinkhorn matching."""
from core.s2m2_spiking_v2 import s2m2_v2_loss as native_loss


def post_ot_loss(output, target, valid, normalization_counts=None):
    # direct_loss applies log_softmax, hence log probability yields exactly the
    # conditional row posterior. Preserve native interpolation and denominator.
    calibrated = dict(output)
    calibrated['logits'] = output['probabilities'].clamp_min(1e-12).log()
    return native_loss(calibrated, target, valid, normalization_counts)
