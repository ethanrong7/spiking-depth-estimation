"""Geometry and scene-level functionality diagnostics for the bounded pilot.

No model inference occurs here. Dataset-wide prior fitting/evaluation belongs
in the registered cluster diagnostic stage; local use is tiny synthetic tests.
Every prediction uses positive disparity d=x_left-x_right in output pixels.
Main error metrics keep all valid GT, including out-of-view correspondence;
only the separately reported shift-response diagnostic uses a trusted common
in-frame subset. Confidence never changes support.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
from typing import Mapping
import warnings

import numpy as np
import torch
import torch.nn.functional as F


def require_engineering_manifest(manifest):
    if (manifest.get('schema') != 'stereo-paired-engineering-pilot-v1'
            or manifest.get('evidence_scope') != 'engineering_only'
            or manifest.get('test') != []):
        raise ValueError('Diagnostics require the engineering-only paired manifest with no Test')
    groups = {}
    for split in ('train', 'val'):
        rows = manifest.get(split, [])
        if not rows:
            raise ValueError('Engineering Train and Val identities are required')
        for row in rows:
            if row.get('split') != split:
                raise ValueError('Manifest split field disagrees with its partition')
            scene = row['scene_id']
            if scene.split('/')[0].upper() == 'TEST' or row.get('original_split') == 'TEST':
                raise ValueError('Official Test identities cannot enter engineering diagnostics')
            if scene in groups and groups[scene] != split:
                raise ValueError('Train/Val scene leakage')
            groups[scene] = split
    return True


def gt_support(target, valid):
    if target.shape != valid.shape:
        raise ValueError('GT and validity shapes must match')
    return valid.bool() & torch.isfinite(target) & (target > 0)


def shift_right_input(right, delta):
    """Move right-view content +delta output pixels, zero-fill left, no wrap.

    Integer shifts preserve binary spikes and event multiplicity; fractional
    sensor-tensor interpolation is deliberately unsupported in this diagnostic.
    Leading dimensions (B,T,C,H or T,C,H) are unchanged.
    """
    if not isinstance(delta, int) or isinstance(delta, bool) or not 0 < delta < right.shape[-1]:
        raise ValueError('An integer shift in [1,width) is required')
    return F.pad(right[..., :-delta], (delta, 0), mode='constant', value=0)


def right_view_conditions(right, other_scene_right, delta):
    """Four fixed right inputs; caller keeps the original left and records IDs."""
    if (other_scene_right.shape != right.shape or other_scene_right.dtype != right.dtype
            or other_scene_right.device != right.device):
        raise ValueError('Shuffled right view must share temporal/spatial layout, dtype and device')
    return dict(correct=right, shuffled=other_scene_right,
                masked=torch.zeros_like(right), shifted=shift_right_input(right, delta))


def shifted_disparity_support(target, valid, delta, *, trusted=None):
    """For right content +delta, d_new=d_old-delta on fixed common GT support.

    Both original xR=xL-d and shifted xR+delta must lie within pixel-center
    bounds [0,W-1]. Positive finite original/shifted GT and any independently
    supplied trusted-GT mask are required. This does not infer occlusions.
    No prediction/confidence tensor participates in selecting support.
    """
    if not isinstance(delta, int) or isinstance(delta, bool) or not 0 < delta < target.shape[-1]:
        raise ValueError('An integer shift in [1,width) is required')
    support = gt_support(target, valid)
    if trusted is not None:
        if trusted.shape != target.shape:
            raise ValueError('Trusted GT mask shape mismatch')
        support = support & trusted.bool()
    x = torch.arange(target.shape[-1], dtype=target.dtype, device=target.device)
    right_original = x - target
    changed_target = target - delta
    right_shifted = right_original + delta
    common = (support & (changed_target > 0) & (right_original >= 0)
              & (right_original <= target.shape[-1]-1) & (right_shifted >= 0)
              & (right_shifted <= target.shape[-1]-1))
    return torch.where(common, changed_target, torch.zeros_like(target)), common


def disparity_metrics(prediction, target, valid):
    """Fixed GT-only support; nonfinite predictions fail rather than disappear."""
    if prediction.shape != target.shape:
        raise ValueError('Prediction and target shapes must match')
    support = gt_support(target, valid)
    count = int(support.sum())
    if not torch.isfinite(prediction[support]).all():
        raise ValueError('Nonfinite prediction on valid GT; cannot filter it from metrics')
    errors = (prediction[support] - target[support]).abs().double()
    total = target.numel()
    sums = {f'bad{threshold}_pixels': int((errors > threshold).sum()) for threshold in (1,3,5)}
    return dict(valid_pixels=count, total_pixels=total,
                coverage=count/total if total else 0.,
                epe=float(errors.mean()) if count else None,
                absolute_error_sum=float(errors.sum()),
                **sums,
                **{f'bad{threshold}': sums[f'bad{threshold}_pixels']/count if count else None
                   for threshold in (1,3,5)})


def shift_response(correct_prediction, shifted_prediction, delta, common_support):
    """Expected positive response: (correct_pred-shifted_pred)/delta ~= 1.

    Return additive quantities so frames aggregate within scenes before scene
    comparisons. majority_positive describes pixels in this diagnostic subset;
    it is not an independence assumption or a quantitative binocular gain.
    """
    if delta <= 0 or correct_prediction.shape != shifted_prediction.shape or correct_prediction.shape != common_support.shape:
        raise ValueError('Positive delta and matching response tensors required')
    support = common_support.bool()
    a, b = correct_prediction[support], shifted_prediction[support]
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError('Nonfinite prediction in the common shift-response support')
    response = (a.double()-b.double())/delta
    n = response.numel();positive = int((response > 0).sum())
    return dict(response_pixels=n, response_sum=float(response.sum()),
                response_ratio=float(response.mean()) if n else None,
                positive_response_pixels=positive,
                positive_response_fraction=positive/n if n else None,
                majority_positive=positive > n/2 if n else None)


def fixed_val_right_shuffle(manifest, *, seed=1234):
    """A stable cyclic pairing of OTHER Val scenes, fixed before inference.

    Within a scene, pair the sample's sorted ordinal with the same ordinal
    modulo peer length. This uses identities only, never GT or model outputs.
    """
    require_engineering_manifest(manifest)
    grouped = defaultdict(list)
    for row in manifest['val']:
        grouped[row['scene_id']].append(row)
    if len(grouped) < 2:
        raise ValueError('At least two Val scenes are needed for other-scene shuffling')
    order = sorted(grouped, key=lambda scene: (
        hashlib.sha256(('stereo-right-shuffle-v1\0'+str(seed)+'\0'+scene).encode()).hexdigest(), scene))
    pairing = {}
    for index, scene in enumerate(order):
        own = sorted(grouped[scene], key=lambda row: row['sample_id'])
        peer = sorted(grouped[order[(index+1)%len(order)]], key=lambda row: row['sample_id'])
        for ordinal, row in enumerate(own):
            pairing[row['sample_id']] = peer[ordinal%len(peer)]['sample_id']
    return pairing


@dataclass
class TrainDisparityPriors:
    constant_median: float
    pixel_median: torch.Tensor
    metadata: dict

    def predict(self, kind, *, device=None):
        if kind == 'constant':
            value = torch.full_like(self.pixel_median, self.constant_median)
        elif kind == 'pixel':
            value = self.pixel_median.clone()
        else:
            raise ValueError('Prior kind must be constant or pixel')
        return value.to(device) if device is not None else value


def fit_train_priors(samples, *, manifest):
    """Exact medians of raw, unaugmented Train GT; no Val labels are fitted.

    Even-size medians average the central two values. Pixel locations with no
    valid Train GT fall back to the global valid-Train-pixel constant median.
    Preserve shape and output-pixel units; no resize, median scale alignment,
    confidence mask or pretraining occurs. Materialize GT on cluster CPU only
    for real data; local tests use tiny fixtures.
    """
    require_engineering_manifest(manifest)
    allowed = {row['sample_id'] for row in manifest['train']}
    identifiers, arrays, valid_arrays = [], [], []
    shape = None
    for sample in samples:
        metadata = sample['metadata']
        identifier = sample['sample_id']
        if metadata.get('split') != 'train' or identifier not in allowed or identifier in identifiers:
            raise ValueError('Prior fitting accepts each registered Train sample exactly once')
        target = sample['target'].detach().cpu()
        if target.ndim != 3 or target.shape[0] != 1 or metadata.get('crop') is not None:
            raise ValueError('Prior fitting requires unaugmented [1,H,W] disparity targets')
        valid = gt_support(target, sample['valid'].detach().cpu())
        if shape is None:
            shape = tuple(target.shape)
        if tuple(target.shape) != shape:
            raise ValueError('Prior fitting requires one unaugmented target shape')
        arrays.append(target.numpy().astype(np.float32, copy=True))
        valid_arrays.append(valid.numpy().copy())
        identifiers.append(identifier)
    if set(identifiers) != allowed:
        raise ValueError('Fit all registered Train samples; no missing or extra identities')
    stack, masks = np.stack(arrays), np.stack(valid_arrays)
    if not masks.any():
        raise ValueError('No valid Train GT to fit disparity priors')
    constant = float(np.median(stack[masks]))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)  # documented all-invalid pixel fallback
        pixel = np.nanmedian(np.where(masks, stack, np.nan), axis=0)
    missing = ~np.isfinite(pixel)
    pixel[missing] = constant
    return TrainDisparityPriors(constant, torch.from_numpy(pixel.astype(np.float32)),
        dict(fit_split='train', sample_ids=sorted(identifiers), samples=len(identifiers),
             target_shape=list(shape), disparity_units='output_horizontal_pixels',
             valid_train_pixels=int(masks.sum()), per_pixel_fallback_pixels=int(missing.sum()),
             fallback='Train_constant_median_where_no_valid_Train_label',
             median_convention='average_central_two_for_even_count',
             fit_includes_val=False, fit_includes_test=False))


def aggregate_scene_metrics(frame_rows):
    """Pool valid pixels WITHIN each scene; retain separate scene observations.

    Frame rows supply method, condition, scene_id, sample_id and the additive
    outputs of disparity_metrics, optionally shift_response. No confidence
    support changes or cross-scene pixel pooling is performed.
    """
    groups = defaultdict(list)
    for row in frame_rows:
        groups[(row['method'],row['condition'],row['scene_id'])].append(row)
    output = []
    for (method, condition, scene), rows in sorted(groups.items()):
        if len({r['sample_id'] for r in rows}) != len(rows):
            raise ValueError('Duplicate sample within a method/condition/scene')
        n = sum(r['valid_pixels'] for r in rows);total = sum(r['total_pixels'] for r in rows)
        bad = {f'bad{t}': sum(r[f'bad{t}_pixels'] for r in rows)/n if n else None for t in (1,3,5)}
        result = dict(method=method,condition=condition,scene_id=scene,
            sample_ids=sorted(r['sample_id'] for r in rows),frames=len(rows),valid_pixels=n,total_pixels=total,
            coverage=n/total if total else 0.,
            epe=sum(r['absolute_error_sum'] for r in rows)/n if n else None,**bad)
        if any('response_pixels' in r for r in rows):
            if not all('response_pixels' in r for r in rows):
                raise ValueError('Mixed response availability within one scene/condition')
            response_n=sum(r['response_pixels'] for r in rows)
            positive=sum(r['positive_response_pixels'] for r in rows)
            result.update(response_pixels=response_n,
                response_ratio=sum(r['response_sum'] for r in rows)/response_n if response_n else None,
                positive_response_fraction=positive/response_n if response_n else None,
                majority_positive=positive>response_n/2 if response_n else None)
        output.append(result)
    return output


def paired_scene_bootstrap(first: Mapping[str,float], second: Mapping[str,float], *, seed=1234, draws=2000):
    """Descriptive paired scene uncertainty for ONE trained seed, no pixel IID.

    first/second must have the same independent scene keys and finite values;
    no missing scene can silently disappear. Difference is first minus second.
    This is not uncertainty over possible training runs or an equivalence test.
    """
    if set(first) != set(second) or not first or draws < 100:
        raise ValueError('Identical nonempty scene identities and >=100 draws required')
    scenes=sorted(first)
    a=np.array([first[s] for s in scenes],dtype=np.float64)
    b=np.array([second[s] for s in scenes],dtype=np.float64)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Every registered scene needs a finite metric')
    delta=a-b;rng=np.random.default_rng(seed)
    indices=rng.integers(0,len(scenes),size=(draws,len(scenes)))
    bootstrap=delta[indices].mean(axis=1)
    lo,hi=np.quantile(bootstrap,[.025,.975])
    return dict(scenes=scenes,scene_count=len(scenes),first_scene_mean=float(a.mean()),
        second_scene_mean=float(b.mean()),paired_difference=float(delta.mean()),
        descriptive_ci95=[float(lo),float(hi)],bootstrap_draws=draws,bootstrap_seed=seed,
        resampling_unit='paired_independent_scene',training_seeds=1,
        interpretation='descriptive_scene_uncertainty_for_one_training_seed_not_training_seed_inference')
