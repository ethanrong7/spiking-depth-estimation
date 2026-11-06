"""Registered positive-disparity supervision with explicit support boundaries."""
import torch
from torch.nn import functional as F

AUXILIARY_WEIGHTS = {4: .5, 8: .25, 16: .125}


def _masked_smooth_l1(prediction, target, valid):
    counts = valid.flatten(1).sum(1)
    nonempty = counts > 0
    if not nonempty.any():
        return prediction.sum() * 0
    error = F.smooth_l1_loss(prediction, torch.where(valid, target, torch.zeros_like(target)), reduction='none')
    per_sample = (error * valid).flatten(1).sum(1) / counts.clamp_min(1)
    return per_sample[nonempty].mean()


def disparity_loss(output, target, valid, max_disparity, *, right_target=None,
                   right_valid=None, occlusion_target=None, occlusion_valid=None):
    """Final loss keeps all trusted GT, including unmatchable/out-of-view pixels.

    Auxiliary matcher losses use native-scale estimates in FULL-image pixel
    disparity units. Nearest target samples retain original disparity units;
    pad is invalid, never geometry-rescaled. Only matcher-representable trusted
    correspondences supervise the structured auxiliary heads.

    No right GT or trusted occlusion labels are inferred from validity.
    """
    if output['disparity'].shape != target.shape or valid.shape != target.shape:
        raise ValueError('Disparity, target and validity must share B,1,H,W')
    trusted = valid.bool() & torch.isfinite(target) & (target > 0)
    terms = {'final_left': _masked_smooth_l1(output['disparity'], target, trusted)}
    total = terms['final_left']
    height, width = target.shape[-2:]
    for stride, estimate in output.get('aux_disparities_native', {}).items():
        if stride not in AUXILIARY_WEIGHTS:
            raise ValueError('Unregistered auxiliary scale')
        hh, ww = estimate.shape[-2:]
        ph, pw = hh * stride - height, ww * stride - width
        if min(ph, pw) < 0:
            raise ValueError('Auxiliary grid smaller than padded input geometry')
        gt = F.pad(torch.where(trusted, target, torch.zeros_like(target)), (0, pw, 0, ph))[..., ::stride, ::stride]
        mask = F.pad(trusted, (0, pw, 0, ph), value=False)[..., ::stride, ::stride]
        x = torch.arange(ww, device=target.device, dtype=target.dtype)[None, None, None] * stride
        mask = mask & (gt <= (max_disparity // stride) * stride) & (x - gt >= 0) & (x - gt < width)
        term = _masked_smooth_l1(estimate, gt, mask)
        terms['aux_left_' + str(stride)] = term
        total = total + AUXILIARY_WEIGHTS[stride] * term
    if right_target is not None:
        if right_valid is None or output['right_disparity'].shape != right_target.shape:
            raise ValueError('True right-reference labels require matching validity/shape')
        right_mask = right_valid.bool() & torch.isfinite(right_target) & (right_target > 0)
        terms['final_right'] = _masked_smooth_l1(output['right_disparity'], right_target, right_mask)
        total = total + terms['final_right']
    if occlusion_target is not None:
        if occlusion_valid is None or occlusion_target.shape != target.shape:
            raise ValueError('Occlusion supervision requires trusted labels and validity')
        mask = occlusion_valid.bool()
        if mask.any():
            terms['occlusion'] = F.binary_cross_entropy_with_logits(output['occlusion_logits'][mask], occlusion_target.float()[mask])
            total = total + .1 * terms['occlusion']
    return total, terms
