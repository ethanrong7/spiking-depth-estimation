"""S9 actual direct/probe/restart paths, costs and same-frame counterfactual tables.

CUDA segments exclude CPU selectors/I/O/transfers. Complete wall/component times
are separate. No GT reaches execute_path; metrics are computed only afterward.
"""
import time
import hashlib
import torch
from stage9_native import (ACTIONS, PREFIXES, HORIZONS, correlation_entropy,
                           probe_features, require)


def parse_action(action):
    p, r = action.split("-")
    return int(p[1:]), int(r[1:])


def timed_cuda(function):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    value = function()
    end.record();end.synchronize()
    return value, float(start.elapsed_time(end))


def cpu_density(pair):
    result = []
    for spikes in pair:
        result.append(int(spikes[:25].sum(dtype=torch.int64))/(25*spikes.shape[-2]*spikes.shape[-1]))
    return result


@torch.no_grad()
def execute_path(model, device_pairs, cpu_pair25, kind, fixed_action=None, selector=None, prefix=24):
    """Exactly one decision from currently available features, then one terminal.

    Counterfactual calls pass a fixed action, never labels. Actual deployed calls
    pass only a frozen current-feature selector. Both follow the same device path.
    """
    require((fixed_action is None) != (selector is None), "provide one action source")
    begun = time.perf_counter();segments = {};cpu = {"density_ms": 0., "feature_transfer_ms": 0., "selector_ms": 0.}
    vector = [0.] * 14
    def segment(name, fn):
        value, elapsed = timed_cuda(fn)
        segments[name] = elapsed
        return value
    def decide():
        if fixed_action is not None:return fixed_action
        started = time.perf_counter();selected = selector(vector)
        cpu["selector_ms"] = 1000*(time.perf_counter()-started)
        require(selected in ACTIONS, "selector returned an unregistered action")
        return selected
    if kind in ("direct", "density"):
        if kind == "density":
            started = time.perf_counter();vector[:2] = cpu_density(cpu_pair25)
            cpu["density_ms"] = 1000*(time.perf_counter()-started)
        action = decide();p, r = parse_action(action)
        flow = segment("direct", lambda: model.direct(*device_pairs[p], p, r)[0][r])
    elif kind == "entropy":
        def entropy_probe():
            features = model.features(*device_pairs[24], 24)
            return features, correlation_entropy(features)
        features, scalar = segment("feature_fine_entropy", entropy_probe)
        started = time.perf_counter();vector[11] = float(scalar.cpu())
        cpu["feature_transfer_ms"] = 1000*(time.perf_counter()-started)
        action = decide();p, r = parse_action(action)
        if p == 24:
            flow = segment("reuse_context_pyramid_updates_decode", lambda: model.advance(model.initialize(features), r)[r])
        else:
            del features, scalar
            flow = segment("restart", lambda: model.direct(*device_pairs[p], p, r)[0][r])
    elif kind in ("joint", "iteration"):
        probe_prefix = prefix if kind == "iteration" else 24
        def joint_probe():
            features = model.features(*device_pairs[probe_prefix], probe_prefix)
            state = model.initialize(features)
            predictions = model.advance(state, 4, {2, 3, 4})
            vector = probe_features(*device_pairs[probe_prefix], probe_prefix, predictions, features)
            return state, predictions, vector
        state, predictions, values = segment("four_update_probe_decodes234_features", joint_probe)
        started = time.perf_counter();vector = values.cpu().double().tolist()
        cpu["feature_transfer_ms"] = 1000*(time.perf_counter()-started)
        action = decide();p, r = parse_action(action)
        if kind == "iteration":require(p == probe_prefix, "iteration selector changed support")
        if p == probe_prefix:
            flow = predictions[4] if r == 4 else segment("continue_updates_terminal_decode", lambda: model.advance(state, r)[r])
        else:
            del state, predictions, values
            flow = segment("restart", lambda: model.direct(*device_pairs[p], p, r)[0][r])
    else:
        raise ValueError(kind)
    model.reset()
    return flow, {"action": action, "gpu_ms": sum(segments.values()), "CUDAsegments": segments,
                  "CPU_selector_ms": cpu["selector_ms"], "CPU_components_ms": cpu,
                  "wall_ms": 1000*(time.perf_counter()-begun), "current_features": vector}


@torch.no_grad()
def metrics(flow, target, valid):
    error = (flow[0, 0].double()-target[0].double()).abs()[valid]
    require(error.numel() > 0 and bool(torch.isfinite(error).all()), "invalid source-valid evaluation")
    count = int(error.numel())
    bad = {k: int((error > k).sum()) for k in (1, 3, 5)}
    return {"epe": float(error.mean()), "bad1_percent": 100*bad[1]/count,
            "bad3_percent": 100*bad[3]/count, "bad5_percent": 100*bad[5]/count,
            "bad3_count": bad[3], "valid_pixels": count}


def equivalent_flow(a, b, target=None, valid=None):
    require(a.dtype == b.dtype == torch.float32 and a.shape == b.shape and bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()), "path output shape/finite mismatch")
    difference = float((a-b).abs().max())
    require(difference <= 1e-5, "native unpadded absolute parity threshold failed")
    def identity(x):return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    flip = None
    if target is not None:
        flip = int(((((a[0,0].double()-target[0].double()).abs()>3) != ((b[0,0].double()-target[0].double()).abs()>3)) & valid).sum())
    return {"status": "PASS", "both_finite": True, "atol_pixels": 1e-5, "rtol": 0,
            "max_abs_difference": difference, "bitwise_equal": bool(torch.equal(a.contiguous().view(torch.int32),b.contiguous().view(torch.int32))),
            "differing_pixel_count": int((a.contiguous().view(torch.int32) != b.contiguous().view(torch.int32)).sum()), "actual_output_sha256": identity(a),
            "direct_output_sha256": identity(b), "bad3_flip_count": flip}


@torch.no_grad()
def extract_frame(model, item, device_pairs, cpu_pair25, split, arm, policies=None, T_gpu_ms=None, policy_sha=None):
    """Full fixed action table; outputs are retained only for this single frame."""
    model.eval()
    frame_started = time.perf_counter()
    target, valid = item["target"].cuda(), item["valid"].cuda()
    row = {"split": split, "arm": arm, "sample_id": item["row"]["sample_id"], "source_id": item["row"]["source_id"],
           "manifest_index": item["manifest_ordinal"], "valid_pixels": int(valid.sum()),
           "actions": {}, "features": {}, "policy_gpu_ms": {}, "timing_receipts": {},
           "I_O_decode_ms": item["I_O_decode_ms"], "host_device_transfer_ms": item["host_device_transfer_ms"]}
    direct = {}
    for p in PREFIXES if arm == "V" else (49,):
        for r in HORIZONS:
            action = f"p{p}-r{r}"
            execute_path(model, device_pairs, cpu_pair25, "direct", fixed_action=action)  # one symmetric numerical warmup
            flow, receipt = execute_path(model, device_pairs, cpu_pair25, "direct", fixed_action=action)
            direct[action] = flow
            row["actions"][action] = {**metrics(flow, target, valid), "gpu_ms_direct": receipt["gpu_ms"],
                "output_sha256": hashlib.sha256(flow.detach().cpu().contiguous().numpy().tobytes()).hexdigest()}
            row["timing_receipts"]["direct:"+action] = receipt
    row["numerical_protocol"] = "one reset numerical warmup of each exact path, then its measured execution; warmup not charged to deployed GPU axis; complete extraction wall includes it"
    if arm == "F":
        row["complete_extraction_wall_ms"] = 1000*(time.perf_counter()-frame_started)
        return row
    density_started = time.perf_counter();row["CPU_density_rates"] = cpu_density(cpu_pair25)
    row["CPU_density_ms"] = 1000*(time.perf_counter()-density_started)
    # Every counterfactual path executes its true computation. No clone/fork or
    # extra terminal decode is silently included in a different action's cost.
    for kind in ("joint", "entropy"):
        row["policy_gpu_ms"][kind] = {}
        for action in ACTIONS:
            execute_path(model, device_pairs, cpu_pair25, kind, fixed_action=action)
            flow, receipt = execute_path(model, device_pairs, cpu_pair25, kind, fixed_action=action)
            receipt["parity"] = {**equivalent_flow(flow, direct[action], target, valid), "sample_id": row["sample_id"], "action": action}
            receipt["parity_against_direct_max_abs"] = receipt["parity"]["max_abs_difference"]
            row["policy_gpu_ms"][kind][action] = receipt["gpu_ms"]
            row["timing_receipts"][kind+":"+action] = receipt
            if kind == "joint":row["features"]["p24"] = receipt["current_features"]
    row["policy_gpu_ms"]["density"] = {a: row["actions"][a]["gpu_ms_direct"] for a in ACTIONS}
    for p in PREFIXES:
        key = f"iteration_p{p}";row["policy_gpu_ms"][key] = {}
        for r in HORIZONS:
            action = f"p{p}-r{r}"
            execute_path(model, device_pairs, cpu_pair25, "iteration", fixed_action=action, prefix=p)
            flow, receipt = execute_path(model, device_pairs, cpu_pair25, "iteration", fixed_action=action, prefix=p)
            receipt["parity"] = {**equivalent_flow(flow, direct[action], target, valid), "sample_id": row["sample_id"], "action": action}
            receipt["parity_against_direct_max_abs"] = receipt["parity"]["max_abs_difference"]
            row["policy_gpu_ms"][key][action] = receipt["gpu_ms"]
            row["features"][f"p{p}"] = receipt["current_features"]
            row["timing_receipts"][key+":"+action] = receipt
    if policies is not None:
        from stage9_analyze import predict_action
        row["deployed_policies"] = {};row["policy_packet_sha256"] = policy_sha
        for cell in policies["cells"]:
            row["deployed_policies"][cell["cell_id"]] = {}
            for policy in cell["policies"]:
                if not policy["feasible"]:continue
                kind = policy["name"]
                if kind == "window":kind = "joint"
                chosen = lambda features, policy=policy: predict_action(policy, features, T_gpu_ms)
                execute_path(model, device_pairs, cpu_pair25, kind, selector=chosen, prefix=policy["prefix"])
                flow, receipt = execute_path(model, device_pairs, cpu_pair25, kind, selector=chosen, prefix=policy["prefix"])
                action = receipt["action"]
                receipt["parity"] = {**equivalent_flow(flow, direct[action], target, valid), "sample_id": row["sample_id"], "action": action}
                receipt["parity_against_direct_max_abs"] = receipt["parity"]["max_abs_difference"]
                row["deployed_policies"][cell["cell_id"]][policy["name"]] = {**metrics(flow, target, valid), **receipt}
    del direct
    row["complete_extraction_wall_ms"] = 1000*(time.perf_counter()-frame_started)
    model.reset()
    return row
