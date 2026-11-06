"""Tensor geometry for the prospective S2 packet; no dataset/model operations."""
import torch


def nearest_native_lattice(target, valid, feature_shape, stride=4, padding=(8, 8, 3, 3)):
    """Nearest-pixel GT at actual convolution centers; never average depth."""
    import torch.nn.functional as F
    b, _, height, width = target.shape
    h, w = feature_shape
    if (height, width, h, w, stride, tuple(padding)) != (250, 400, 64, 104, 4, (8, 8, 3, 3)):
        raise ValueError("unregistered feature geometry")
    yy = torch.arange(h, device=target.device) * stride - padding[2]
    xx = torch.arange(w, device=target.device) * stride - padding[0]
    inside = (yy[:, None] >= 0) & (yy[:, None] < height) & (xx[None] >= 0) & (xx[None] < width)
    y, x = yy.clamp(0, height - 1), xx.clamp(0, width - 1)
    flow = target[:, 0][:, y[:, None], x[None]]
    labeled = inside[None] & valid[:, y[:, None], x[None]] & torch.isfinite(flow)
    right_native = xx[None, None, :] + flow
    inbounds = labeled & (right_native >= 0) & (right_native <= width - 1)
    qgt = torch.arange(w, device=target.device)[None, None, :] + flow / stride
    # S0 convention: Sobel /8, >1 pixel per pixel, eroded valid neighborhood.
    disp = -target[:, :1]
    kx = disp.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])[None, None] / 8
    ky = kx.transpose(-1, -2)
    padded = F.pad(disp, (1, 1, 1, 1), mode="reflect")
    gradient = torch.sqrt(F.conv2d(padded, kx).square() + F.conv2d(padded, ky).square())[:, 0]
    # Erode with outside treated valid, matching OpenCV default erosion border.
    valid_neighborhood = -F.max_pool2d(F.pad(-valid[:, None].float(), (1, 1, 1, 1), value=-1), 3, stride=1)[:, 0] > .5
    nonboundary = inbounds & valid_neighborhood[:, y[:, None], x[None]] & (gradient[:, y[:, None], x[None]] <= 1)
    return dict(qgt=qgt, valid=inbounds, labeled=labeled, center_inside=inside[None].expand(b, -1, -1),
                nonboundary=nonboundary, key_valid=(xx >= 0) & (xx < width), native_y=y, native_x=x,
                native_center_y=yy, native_center_x=xx)


def score_evidence(rows, qgt, key_valid):
    """All tensor outputs retain continuous values, including undefined rows."""
    keys = torch.arange(rows.shape[-1], device=rows.device)
    valid = key_valid.expand_as(rows)
    group = valid & ((keys - qgt[..., None]).abs() <= 1)
    distractors = valid & ~group
    count = valid.sum(-1)
    mean = rows.masked_fill(~valid, 0).sum(-1) / count.clamp_min(1)
    std = ((rows - mean[..., None]).square().masked_fill(~valid, 0).sum(-1) / count.clamp_min(1)).sqrt()
    best_g = rows.masked_fill(~group, -torch.inf).amax(-1)
    best_b = rows.masked_fill(~distractors, -torch.inf).amax(-1)
    defined = group.any(-1) & distractors.any(-1)
    margin = ((best_g - best_b) / (std + 1e-6)).masked_fill(~defined, torch.nan)
    peak = rows.masked_fill(~valid, -torch.inf).argmax(-1)
    unrestricted = rows.argmax(-1)
    nearest = (keys - qgt[..., None]).abs().masked_fill(~valid, torch.inf).argmin(-1)
    score = rows.gather(-1, nearest[..., None])[..., 0]
    rank = 1 + ((rows > score[..., None]) & valid).sum(-1) + .5 * (((rows == score[..., None]) & valid).sum(-1) - 1)
    maxima = rows.masked_fill(~valid, -torch.inf).amax(-1)
    peak_unique = ((rows == maxima[..., None]) & valid).sum(-1) == 1
    correct = (peak - qgt).abs() <= 1
    return dict(group=group, group_size=group.sum(-1), margin=margin, row_std=std,
                best_gt_score=best_g, best_distractor_score=best_b, nearest_gt_rank=rank,
                nearest_gt_score=score,
                nearest_gt_index=nearest, valid_peak_index=peak, unrestricted_peak_index=unrestricted,
                unrestricted_peak_padded=~key_valid[unrestricted], peak_correct=correct,
                peak_unique=peak_unique, score_defined=defined,
                evidence_available=defined & (std > 1e-6) & (margin >= .25) & correct & peak_unique)


def lookup_support(query, group, level_widths, radius=4):
    """Expand actual positive bilinear taps and recursively pooled cells.

    Returned coverage: 0=none, 1=partial, 2=full; empty G is undefined (-1).
    Intermediate positive tap sets are also returned for audit/storage.
    """
    if len(level_widths) != 4:
        raise ValueError("requires exactly four used correlation levels")
    import torch.nn.functional as F
    fine_width = group.shape[-1]
    if query.dtype != torch.float32:
        raise ValueError("frozen native lookup requires float32 coordinates/correlation")
    per_level, taps = [], []
    for level, width in enumerate(level_widths):
        if width != fine_width // (2 ** level):
            raise ValueError("pool width does not match floor/truncation")
        centers = query[..., None] / (2 ** level) + torch.arange(-radius, radius + 1, device=query.device)
        # Probe the actual kernel with three periodic basis channels. Adjacent
        # taps have distinct colors; candidates floor(x)-1/floor(x)/floor(x)+1
        # cover normalization roundoff on either side of nominal integers.
        # This retains tiny positive weights instead of inventing an epsilon.
        colors = torch.arange(width, device=query.device) % 3
        basis = (torch.arange(3, device=query.device)[:, None] == colors[None]).to(query.dtype)[None, :, None]
        xgrid = 2 * centers / (width - 1) - 1
        grid = torch.stack((xgrid, torch.zeros_like(xgrid)), -1).reshape(1, -1, 2 * radius + 1, 2)
        sampled = F.grid_sample(basis, grid, align_corners=True).permute(0, 2, 3, 1).reshape(*centers.shape, 3)
        indices = centers.floor().long()[..., None] + torch.tensor([-1, 0, 1], device=query.device)
        weights = sampled.gather(-1, indices.remainder(3)).reshape(*query.shape, -1)
        indices = indices.reshape(*query.shape, -1)
        active = (weights > 0) & (indices >= 0) & (indices < width)
        support = torch.zeros((*query.shape, fine_width), dtype=torch.bool, device=query.device)
        group_mass = torch.zeros_like(weights)
        for offset in range(2 ** level):
            fine = indices * (2 ** level) + offset
            support.scatter_add_(-1, fine.clamp(0, fine_width - 1), active)
            group_mass += group.gather(-1, fine.clamp(0, fine_width - 1)) * active * weights / (2 ** level)
        per_level.append(support)
        taps.append(dict(indices=indices, weights=weights, active=active,
                         GT_mass_per_sample=group_mass.reshape(*query.shape, 2 * radius + 1, 3).sum(-1),
                         GT_total_mass=group_mass.sum(-1)))
    def coverage(support):
        hits = (support & group).sum(-1)
        size = group.sum(-1)
        result = torch.where(hits == 0, 0, torch.where(hits == size, 2, 1))
        return result.masked_fill(size == 0, -1)
    pooled = torch.stack(per_level[1:]).any(0)
    union = torch.stack(per_level).any(0)
    result = {f"support_level_{i}": coverage(s) for i, s in enumerate(per_level)}
    result.update(support_pooled_union=coverage(pooled), support_all_union=coverage(union))
    result.update(fine_full=result["support_level_0"] == 2,
                  pooled_only_full=(result["support_level_0"] != 2) & (result["support_pooled_union"] == 2),
                  outside_all_support=result["support_all_union"] == 0)
    return result, taps
