"""Fixed-support spatial ZNCC; NumPy CPU only, no learned model."""
import numpy as np


def chunk_rates(spikes, length):
    if length < 1 or spikes.ndim != 3 or spikes.shape[0] % length:
        raise ValueError('T,H,W spikes with exactly divisible positive chunk length required')
    return spikes.reshape(-1, length, *spikes.shape[1:]).mean(1, dtype=np.float64)


def zncc(left, right, queries, disparities, radius=4, epsilon=1e-12):
    """Return N,D scores; all patch pixels and candidates must be in image.

    Queries are integer y,x left patch centers. Positive disparity samples the
    right patch at x-d. Constant patches receive score zero, never a new mask.
    """
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError('equal H,W images required')
    queries = np.asarray(queries, dtype=np.int64)
    disparities = np.asarray(disparities, dtype=np.int64)
    y, x = queries.T
    height, width = left.shape
    if not ((y >= radius).all() and (y+radius < height).all() and
            (x[:, None]-disparities[None] >= radius).all() and
            (x[:, None]-disparities[None]+radius < width).all() and
            (x+radius < width).all()):
        raise ValueError('all candidate patches must fit without padding')
    side = 2*radius+1
    l = np.lib.stride_tricks.sliding_window_view(left, (side, side))[y-radius, x-radius].reshape(len(y), -1).astype(np.float64)
    r = np.lib.stride_tricks.sliding_window_view(right, (side, side))[
        y[:, None]-radius, x[:, None]-disparities[None]-radius].reshape(len(y), len(disparities), -1).astype(np.float64)
    l -= l.mean(-1, keepdims=True)
    r -= r.mean(-1, keepdims=True)
    denominator = np.sqrt((l*l).sum(-1)[:, None] * (r*r).sum(-1))
    scores = np.divide((l[:, None]*r).sum(-1), denominator,
                       out=np.zeros_like(denominator), where=denominator > epsilon)
    return scores, denominator <= epsilon


def aggregate_scores(left_chunks, right_chunks, queries, disparities, radius=4):
    if left_chunks.shape != right_chunks.shape:
        raise ValueError('equal chunk stacks required')
    scores = np.zeros((len(queries), len(disparities)), dtype=np.float64)
    zero_counts = np.zeros_like(scores, dtype=np.int64)
    for left, right in zip(left_chunks, right_chunks):
        score, zeros = zncc(left, right, queries, disparities, radius)
        scores += score
        zero_counts += zeros
    return scores / len(left_chunks), zero_counts


def score_metrics(scores, truth, disparities):
    """Smaller-disparity argmax tie break; competition rank at nearest GT bin."""
    disparities = np.asarray(disparities)
    predicted = disparities[scores.argmax(1)]
    nearest = np.abs(disparities[None]-truth[:, None]).argmin(1)
    truth_scores = scores[np.arange(len(truth)), nearest]
    rank = 1 + (scores > truth_scores[:, None]).sum(1)
    error = np.abs(predicted-truth)
    near = np.abs(disparities[None]-truth[:, None]) <= 1
    near_best = np.where(near, scores, -np.inf).max(1)
    near_rank = 1 + (scores > near_best[:, None]).sum(1)
    far = np.abs(disparities[None]-truth[:, None]) > 3
    gt_margin = truth_scores - np.where(far, scores, -np.inf).max(1)
    sorted_scores = np.sort(scores, axis=1)
    return dict(count=len(truth), epe=float(error.mean()) if len(error) else None,
        bad1=float((error > 1).mean()*100) if len(error) else None,
        bad3=float((error > 3).mean()*100) if len(error) else None,
        bad5=float((error > 5).mean()*100) if len(error) else None,
        mean_gt_rank=float(rank.mean()) if len(rank) else None,
        top1_fraction=float((rank == 1).mean()) if len(rank) else None,
        argmax_tie_fraction=float(((scores == scores.max(1, keepdims=True)).sum(1) > 1).mean()) if len(rank) else None,
        mean_gt_near1_rank=float(near_rank.mean()) if len(rank) else None,
        gt_nearest_vs_far3_margin=float(gt_margin.mean()) if len(rank) else None,
        top1_top2_margin=float((sorted_scores[:, -1]-sorted_scores[:, -2]).mean()) if len(rank) else None,
        _predicted=predicted, _error=error, _rank=rank, _near1_rank=near_rank, _gt_margin=gt_margin)
