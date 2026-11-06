"""S4 float64 dense-labelled-warp measurements; no model or dataset imports."""
import hashlib
import numpy as np
import cv2

OFFSETS = (-8, -4, -2, -1, 0, 1, 2, 4, 8)
PATCH = 7


def box(a):
    return cv2.boxFilter(np.asarray(a, np.float64), -1, (PATCH, PATCH),
                         normalize=False, borderType=cv2.BORDER_CONSTANT) / 49.


def warp(right, g, delta, right_valid=None):
    """Explicit horizontal interpolation; only nonzero-weight indices count."""
    right = np.asarray(right, np.float64)
    g = np.asarray(g, np.float64)
    h, w = right.shape
    assert g.shape == right.shape
    q = np.arange(w, dtype=np.float64)[None, :] - g + delta
    finite = np.isfinite(q)
    qsafe = np.where(finite, q, 0.)
    lo = np.floor(qsafe).astype(np.int64)
    weight = qsafe - lo
    hi = lo + 1
    valid = finite & (lo >= 0) & (lo < w) & ((weight == 0) | (hi < w))
    lc, hc = np.clip(lo, 0, w-1), np.clip(hi, 0, w-1)
    y = np.arange(h)[:, None]
    if right_valid is not None:
        rv = np.asarray(right_valid, bool)
        valid &= rv[y, lc] & ((weight == 0) | rv[y, hc])
    value = (1-weight)*right[y, lc] + weight*right[y, hc]
    return np.where(valid, value, 0.), valid


def moments(left, warped, lm=None, lv=None):
    if lm is None:
        lm = box(left)
    if lv is None:
        lv = np.maximum(box(left*left)-lm*lm, 0.)
    wm = box(warped)
    wv = np.maximum(box(warped*warped)-wm*wm, 0.)
    cov = box(left*warped)-lm*wm
    return lm, lv, wm, wv, cov


def describe(values):
    v = np.asarray(values, np.float64).ravel()
    assert np.isfinite(v).all()
    if not v.size:
        return dict(count=0, mean=None, median=None, q25=None, q75=None, minimum=None, maximum=None,
                    fraction_strictly_positive=None)
    return dict(count=int(v.size), mean=float(v.mean()), median=float(np.median(v)), minimum=float(v.min()), maximum=float(v.max()),
                q25=float(np.quantile(v, .25)), q75=float(np.quantile(v, .75)),
                fraction_strictly_positive=float((v > 0).mean()))


def measure(left, right, g, original_valid, right_valid=None):
    """Two-pass common-N support, then sequential candidates; O(HW) memory.

    P uses common geometric support for every dense warped sample in a patch.
    N excludes any candidate's degenerate variance before either N comparison.
    """
    left, right, g = [np.asarray(x, np.float64) for x in (left, right, g)]
    assert left.shape == right.shape == g.shape == np.shape(original_valid)
    assert np.isfinite(left).all() and np.isfinite(right).all()
    m = np.asarray(original_valid, bool).copy()
    for delta in OFFSETS:
        _, valid = warp(right, g, delta, right_valid)
        m &= valid
    p = box(m.astype(np.float64)) == 1.
    lm = box(left)
    lv = np.maximum(box(left*left)-lm*lm, 0.)
    n = p & (lv > 1e-12)
    for delta in OFFSETS:
        w, _ = warp(right, g, delta, right_valid)
        _, _, _, wv, _ = moments(left, w, lm, lv)
        n &= wv > 1e-12
    safe_g = np.where(np.isfinite(g), g, 0.)
    g_range = cv2.dilate(safe_g, np.ones((7, 7), np.uint8)) - cv2.erode(safe_g, np.ones((7, 7), np.uint8))
    support = dict(original_valid=int(np.count_nonzero(original_valid)), common_point=int(m.sum()),
                   common_patch=int(p.sum()), common_N=int(n.sum()),
                   patch_excluded_non_native_or_not_common=int(p.size-p.sum()),
                   N_excluded_numerical_degeneracy=int(p.sum()-n.sum()))

    def costs(delta):
        w, _ = warp(right, g, delta, right_valid)
        _, _, _, wv, cov = moments(left, w, lm, lv)
        mae = box(np.abs(left-w))
        disagreement = np.zeros_like(mae)
        disagreement[n] = 1.-np.clip(cov[n]/np.sqrt(lv[n]*wv[n]), -1., 1.)
        return mae, disagreement

    baseline = costs(0)
    result = dict(support=support, costs={})
    for name, mask, index in [('rate_MAE', p, 0), ('normalized_disagreement', n, 1)]:
        result['costs'][name] = {
            'ambiguity': {'left_rate': describe(left[mask]), 'left_patch_variance': describe(lv[mask]),
                          'GT_patch_range_pixels': describe(g_range[mask])},
            'candidates': {}, 'paired_margins_vs_zero': {}}
    for delta in OFFSETS:
        pair = baseline if delta == 0 else costs(delta)
        for name, mask, index in [('rate_MAE', p, 0), ('normalized_disagreement', n, 1)]:
            dest = result['costs'][name]
            dest['candidates'][str(delta)] = describe(pair[index][mask])
            if delta:
                dest['paired_margins_vs_zero'][str(delta)] = describe((pair[index]-baseline[index])[mask])
    return result


def flatten(value, prefix=''):
    for key, v in value.items():
        path = f'{prefix}.{key}' if prefix else key
        if isinstance(v, dict):
            yield from flatten(v, path)
        else:
            yield path, v


def aggregate(rows):
    """Equal-record means of summaries, never pooled pixel quantiles."""
    keys = sorted({k for row in rows for k, _ in flatten(row)})
    flat = [dict(flatten(row)) for row in rows]
    out = {}
    for k in keys:
        values = [r[k] for r in flat if isinstance(r.get(k), (float, int))]
        out[k] = {'mean_of_record_summaries': float(np.mean(values)) if values else None,
                  'contributing_records': len(values), 'total_records': len(rows)}
    return out


def overlap_report(a_bytes, b_bytes, plane_bytes=12500, planes=50):
    """Byte-only window alignment audit with explicit repeated-plane ambiguity."""
    expected = plane_bytes*planes
    base = {'length_A': len(a_bytes), 'length_B': len(b_bytes), 'expected_length': expected}
    if len(a_bytes) != expected or len(b_bytes) != expected:
        return {**base, 'status': 'INVALID_LENGTH', 'observed_relative_bin_displacement': None}
    def planes_info(data):
        chunks = [data[i*plane_bytes:(i+1)*plane_bytes] for i in range(planes)]
        hashes = [hashlib.sha256(c).hexdigest() for c in chunks]
        return hashes, dict(distinct_plane_hashes=len(set(hashes)),
                            all_zero_planes=sum(c == b'\x00'*plane_bytes for c in chunks),
                            all_one_planes=sum(c == b'\xff'*plane_bytes for c in chunks),
                            plane_sha256=hashes)
    a, ai = planes_info(a_bytes)
    b, bi = planes_info(b_bytes)
    matches = [r for r in range(1, planes+1) if a[-r:] == b[:r]]
    unique_matches = [{'A_index': i, 'B_index': b.index(h), 'B_minus_A_index': b.index(h)-i}
                      for i, h in enumerate(a) if a.count(h) == b.count(h) == 1]
    r = matches[0] if len(matches) == 1 else None
    interpretable = r is not None and 2 <= r < planes and len(set(a[-r:])) >= 2 and all(a.count(h) == b.count(h) == 1 for h in a[-r:])
    return {**base, 'status': 'COMPLETE', 'A': ai, 'B': bi, 'suffix_prefix_matching_lengths': matches,
            'unique_hash_relative_index_matches': unique_matches, 'full_window_identity': a == b,
            'repeated_hashes_A_or_B': len(set(a)) < planes or len(set(b)) < planes,
            'multiple_alignments': len(matches) > 1,
            'observed_relative_bin_displacement': planes-r if interpretable else None,
            'interpretation': 'consistent_with_overlap_not_physical_cadence' if interpretable else 'ambiguous_or_no_unique_overlap'}
