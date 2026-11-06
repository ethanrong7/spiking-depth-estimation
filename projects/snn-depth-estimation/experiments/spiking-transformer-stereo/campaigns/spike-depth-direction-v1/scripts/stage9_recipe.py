"""Shared S9 paired-update semantics; no data/model construction on import."""
import torch
from stage9_native import require


def prefix_for(arm, update, micro):
    require(arm in ("V", "F") and micro in (0, 1) and update >= 0, "invalid paired presentation")
    return (24, 32, 49)[(2*update+micro)%3] if arm == "V" else 49


def pooled_endpoint_loss(predictions, target, valid, combined_valid_pixels):
    require(set(predictions) == {4, 8, 16} and combined_valid_pixels >= int(valid.sum()) > 0, "three endpoints/source-valid count required")
    return sum((predictions[k][0,0]-target[0]).abs()[valid].sum() for k in (4, 8, 16))/(3*combined_valid_pixels)


def optimizer_schedule(model):
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4, weight_decay=1e-5, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-4, total_steps=5100, pct_start=.05,
        anneal_strategy="linear", cycle_momentum=False, div_factor=25, final_div_factor=10000)
    return opt, scheduler


def sample_order():
    return torch.randperm(800, generator=torch.Generator().manual_seed(1234)).tolist()


def backward_pair(model, items, arm, update):
    """Two independent connected16-step graphs, pooled pixels, one external update."""
    model.train();model.freeze_bn();model.zero_grad(set_to_none=True)
    count = sum(int(item["valid"].sum()) for item in items)
    totals = [];endpoint_gradients = []
    from stage9_data import device_pairs
    for micro, item in enumerate(items):
        p = prefix_for(arm, update, micro)
        pairs, _ = device_pairs(item, (p,))
        target, valid = item["target"].cuda(), item["valid"].cuda()
        model.reset()
        predictions, state = model.direct(*pairs[p], p, 16, {4,8,16})
        for prediction in predictions.values():prediction.retain_grad()
        loss = pooled_endpoint_loss(predictions, target, valid, count)
        require(bool(torch.isfinite(loss)), "nonfinite pooled endpoint loss")
        loss.backward()
        endpoint_gradients.append({str(k): bool(predictions[k].grad is not None and torch.isfinite(predictions[k].grad).all() and predictions[k].grad.abs().sum()>0) for k in (4,8,16)})
        totals.append(float(loss.detach()))
        del predictions, state, pairs, loss, target, valid
        model.reset()
    missing = [n for n,p in model.named_parameters() if p.requires_grad and p.grad is None]
    dormant = [n for n,p in model.named_parameters() if not p.requires_grad and p.grad is not None]
    require(not missing and not dormant, "active/dormant gradient census failed")
    require(all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None), "nonfinite parameter gradient")
    require(all(all(row.values()) for row in endpoint_gradients), "missing endpoint loss gradient")
    return {"loss_finite": True, "endpoint_gradients": endpoint_gradients, "missing_active_gradients": missing, "dormant_gradients": dormant,
            "valid_pixels_combined": count, "prefixes": [prefix_for(arm,update,m) for m in (0,1)], "loss": sum(totals)}
