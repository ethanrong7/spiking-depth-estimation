"""S5 fixed model-free target-time observation; payload execution requires Slurm.

Only NumPy/Pillow and two unchanged, source-bound FlyingThings utilities are
used. Static streams are hypothetical target-image controls. No learned model,
calibration, outside-window observations or censored-value fallback is present.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import platform
import resource
import socket
import sys
import time
import unittest
from collections import defaultdict

import numpy as np

CONTRACT_SHA = "8382968ec156def131fec15cffea1729785323825f462722696c777bcd01d264"
MANIFEST_SHA = "eb52ec22890423efc2d5455904c3155d6aaa8c8a96199eb8fcd371305d62534d"
CONDITIONS = ("Actual", "Static0", "StaticHalf")
VIEWS = ("L", "R")
CAMPAIGN = Path(__file__).resolve().parents[1]
PREP = Path(__file__).resolve().parents[4] / "flyingthings3d-native-v1/scripts"


def load_utility(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


codec = load_utility("s5_frozen_ft_codec", PREP / "spike_codec.py")
source_io = load_utility("s5_frozen_ft_source_io", PREP / "source_io.py")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def target_gray(rgb):
    require(rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[-1] == 3,
            "target must be decoded uint8 RGB")
    gray = codec.rgb_to_gray(rgb.astype(np.float32) / np.float32(255))
    require(gray.dtype == np.float32 and np.isfinite(gray).all()
            and gray.min() >= 0 and gray.max() <= 1, "invalid target grayscale")
    return gray


def event_summary(count, previous, following):
    bracket = (previous >= 0) & (following >= 0)
    gap = following.astype(np.int16) - previous.astype(np.int16)
    require(np.all(gap[bracket] > 0), "nonpositive observed interval")
    interval = np.full(count.shape, np.nan, dtype=np.float64)
    interval[bracket] = np.float64(5) / gap[bracket].astype(np.float64)
    count_estimate = np.float64(5) * count.astype(np.float64) / np.float64(50)
    require(np.isfinite(count_estimate).all() and np.isfinite(interval[bracket]).all(),
            "nonfinite estimator on declared support")
    return {"count": count, "a": previous, "b": following, "B": bracket,
            "C": count_estimate, "T": interval}


def summarize_spikes(spikes):
    require(spikes.ndim == 3 and spikes.shape[0] == 50
            and np.isin(spikes, [0, 1]).all(), "complete binary50-bin array required")
    count = spikes.sum(axis=0, dtype=np.int16)
    previous = np.full(count.shape, -1, dtype=np.int16)
    following = np.full(count.shape, -1, dtype=np.int16)
    for k in range(24):
        previous[spikes[k].astype(bool)] = k
    for k in range(24, 50):
        following[(following < 0) & spikes[k].astype(bool)] = k
    return event_summary(count, previous, following)


def simulate_static(gray, phase):
    require(gray.dtype == np.float32 and np.isfinite(gray).all()
            and gray.min() >= 0 and gray.max() <= 1, "FP32 normalized gray required")
    require(phase in (0., 2.5), "only two registered static phases")
    voltage = np.full(gray.shape, phase, dtype=np.float32)
    count = np.zeros(gray.shape, dtype=np.int16)
    previous = np.full(gray.shape, -1, dtype=np.int16)
    following = np.full(gray.shape, -1, dtype=np.int16)
    for k in range(50):
        spike, voltage = codec.integrate_numpy(voltage, gray)
        count += spike.astype(np.int16)
        if k < 24:
            previous[spike] = k
        else:
            following[(following < 0) & spike] = k
    return event_summary(count, previous, following)


def common_masks(streams):
    return {v: np.logical_and.reduce([streams[z][v]["B"] for z in CONDITIONS]) for v in VIEWS}


def horizontal_support(q, base, left_mask, right_mask):
    """Positive-weight native row taps; also well-defined at the last column."""
    require(q.shape == base.shape == left_mask.shape == right_mask.shape, "geometry shape mismatch")
    h, w = q.shape
    base = base & np.isfinite(q) & (q >= 0) & (q <= w - 1)
    safe = np.where(base, q, 0)
    i0 = np.floor(safe).astype(np.int64)
    w1 = safe - i0
    i1 = np.minimum(i0 + 1, w - 1)
    rows = np.broadcast_to(np.arange(h)[:, None], (h, w))
    joint = base & left_mask & right_mask[rows, i0]
    positive = w1 > 0
    joint &= ~positive | right_mask[rows, i1]
    return {"F": base, "J": joint, "rows": rows, "i0": i0, "i1": i1, "w1": w1}


def stereo_geometry(disparity, left_mask, right_mask):
    """Native positive disparity with no maximum-disparity or visibility mask."""
    q = np.arange(disparity.shape[1], dtype=np.float64)[None, :] - disparity.astype(np.float64)
    return horizontal_support(q, np.isfinite(disparity) & (disparity > 0), left_mask, right_mask)


def warp_selected(image, geometry):
    """Gather only J and only strictly positive-weight taps; no NaN filling."""
    selected = geometry["J"]
    rows = geometry["rows"][selected]
    i0 = geometry["i0"][selected]
    i1 = geometry["i1"][selected]
    w1 = geometry["w1"][selected]
    value = (1 - w1) * image[rows, i0].astype(np.float64)
    positive = w1 > 0
    value[positive] += w1[positive] * image[rows[positive], i1[positive]].astype(np.float64)
    require(np.isfinite(value).all(), "attempt to interpolate invalid/placeholder values")
    return value


def scalar_record(value, n):
    require(value is None or math.isfinite(float(value)), "nonfinite saved scalar")
    return {"value": None if value is None else float(value), "support_count": int(n)}


def describe(values, quantiles=(.1, .5, .9)):
    values = np.asarray(values, dtype=np.float64)
    require(np.isfinite(values).all(), "nonfinite metric population")
    if not values.size:
        return {"mean": None, **{f"q{round(q*100)}": None for q in quantiles}}
    result = {"mean": float(values.mean(dtype=np.float64))}
    result.update({f"q{round(q*100)}": float(v) for q, v in zip(quantiles, np.quantile(values, quantiles, method="linear"))})
    return result


def add_description(destination, prefix, values, quantiles=(.1, .5, .9)):
    for key, value in describe(values, quantiles).items():
        destination[prefix + "." + key] = scalar_record(value, np.size(values))


def add_error(destination, prefix, errors, tails=False):
    stats = describe(errors, (.9,) if tails else ())
    destination[prefix + ".mae"] = scalar_record(stats["mean"], np.size(errors))
    if tails:
        destination[prefix + ".q90"] = scalar_record(stats["q90"], np.size(errors))
    return stats["mean"]


def add_gains(metrics, prefix, errors, n):
    gains = {}
    for condition in CONDITIONS:
        c, t = errors[condition]
        gains[condition] = None if c is None or t is None else c - t
        metrics[prefix + ".gain." + condition] = scalar_record(gains[condition], n)
    excess = None if any(value is None for value in gains.values()) else gains["Actual"] - max(gains["Static0"], gains["StaticHalf"])
    metrics[prefix + ".excess"] = scalar_record(excess, n)


def measure_frame(gray32, streams, disparity, identity):
    """Pure array measurement; masks are constructed before any error values."""
    gray = {v: gray32[v].astype(np.float64) for v in VIEWS}
    require(gray["L"].shape == gray["R"].shape == disparity.shape, "native array mismatch")
    masks = common_masks(streams)
    geometry = stereo_geometry(disparity, masks["L"], masks["R"])
    metrics, support = {}, {}
    total = disparity.size
    for view in VIEWS:
        common = masks[view]
        n = int(common.sum())
        support[view + ".all_pixels"] = scalar_record(total, total)
        support[view + ".common_count"] = scalar_record(n, total)
        support[view + ".common_fraction"] = scalar_record(n / total, total)
        support[view + ".complement_count"] = scalar_record(total - n, total)
        add_description(metrics, f"target.{view}.gray_common", gray[view][common])
        add_description(metrics, f"target.{view}.gray_complement", gray[view][~common])
        target_errors = {}
        for condition in CONDITIONS:
            data = streams[condition][view]
            bracket = data["B"]
            bn = int(bracket.sum())
            prefix = f"{view}.{condition}"
            for key, value in {"bracket_count": bn, "missing_left": int((data["a"] < 0).sum()),
                               "missing_right": int((data["b"] < 0).sum()),
                               "both_missing": int(((data["a"] < 0) & (data["b"] < 0)).sum()),
                               "zero_spikes": int((data["count"] == 0).sum()),
                               "one_spike": int((data["count"] == 1).sum())}.items():
                support[prefix + "." + key] = scalar_record(value, total)
            intervals = {"gap": data["b"][bracket] - data["a"][bracket],
                         "target_minus_a": 24 - data["a"][bracket], "b_minus_target": data["b"][bracket] - 24}
            for name, values in intervals.items():
                add_description(support, prefix + "." + name, values)
                support[prefix + "." + name + ".min"] = scalar_record(int(values.min()) if values.size else None, bn)
                support[prefix + "." + name + ".max"] = scalar_record(int(values.max()) if values.size else None, bn)
            gap = intervals["gap"]
            support[prefix + ".gap_below_5"] = scalar_record(int((gap < 5).sum()), bn)
            primary = []
            for estimator in ("C", "T"):
                values = data[estimator]
                primary.append(add_error(metrics, f"target.{view}.primary.{condition}.{estimator}",
                                         np.abs(values[common] - gray[view][common]), tails=True))
                add_error(metrics, f"target.{view}.own_bracket.{condition}.{estimator}",
                          np.abs(values[bracket] - gray[view][bracket]))
            target_errors[condition] = primary
        actual_c = streams["Actual"][view]["C"]
        add_error(metrics, f"target.{view}.actual_count_all", np.abs(actual_c - gray[view]))
        add_error(metrics, f"target.{view}.actual_count_complement", np.abs(actual_c[~common] - gray[view][~common]))
        add_gains(metrics, "target." + view, target_errors, n)
    base, joint = geometry["F"], geometry["J"]
    fn, jn = int(base.sum()), int(joint.sum())
    support["stereo.F_count"] = scalar_record(fn, total)
    support["stereo.J_count"] = scalar_record(jn, total)
    support["stereo.J_over_F"] = scalar_record(jn / fn if fn else None, fn)
    support["stereo.J_over_all"] = scalar_record(jn / total, total)
    left_g = gray["L"][joint]
    right_g = warp_selected(gray["R"], geometry)
    reference = left_g - right_g
    add_error(metrics, "stereo.reference_plain", np.abs(reference))
    fidelity_errors = {}
    for condition in CONDITIONS:
        errors = []
        for estimator in ("C", "T"):
            left = streams[condition]["L"][estimator][joint]
            right = warp_selected(streams[condition]["R"][estimator], geometry)
            difference = left - right
            prefix = f"stereo.{condition}.{estimator}"
            errors.append(add_error(metrics, prefix + ".fidelity", np.abs(difference - reference), tails=True))
            add_error(metrics, prefix + ".plain", np.abs(difference))
            add_error(metrics, prefix + ".individual_L", np.abs(left - left_g))
            add_error(metrics, prefix + ".individual_R", np.abs(right - right_g))
        fidelity_errors[condition] = errors
    add_gains(metrics, "stereo", fidelity_errors, jn)
    return {**identity, "metrics": metrics, "support": support}


def reduce_records(records):
    """Equal frames within sequence, then equal available sequences per split."""
    groups = defaultdict(list)
    for row in records:
        groups[(row["split"], row["source_id"])].append(row)
    scenes = []
    for (split, source_id), frames in groups.items():
        scene = {"split": split, "source_id": source_id, "total_frames": len(frames), "metrics": {}, "support": {}}
        for category in ("metrics", "support"):
            keys = list(frames[0][category])
            require(all(list(frame[category]) == keys for frame in frames), "inconsistent frame schema")
            for key in keys:
                values = [frame[category][key]["value"] for frame in frames if frame[category][key]["value"] is not None]
                scene[category][key] = {"value": math.fsum(values) / len(values) if values else None,
                                        "nonempty_frames": len(values), "total_frames": len(frames),
                                        "support_count_sum": sum(frame[category][key]["support_count"] for frame in frames)}
        scenes.append(scene)
    splits = {}
    for split in ("train", "dev"):
        selected = [scene for scene in scenes if scene["split"] == split]
        summary = {"total_scenes": len(selected), "total_frames": sum(s["total_frames"] for s in selected),
                   "metrics": {}, "support": {}, "signed_scene_counts": {}}
        if selected:
            for category in ("metrics", "support"):
                for key in selected[0][category]:
                    available = [scene[category][key] for scene in selected if scene[category][key]["value"] is not None]
                    values = [item["value"] for item in available]
                    summary[category][key] = {"value": math.fsum(values) / len(values) if values else None,
                                              "nonempty_scenes": len(values),
                                              "nonempty_frames": sum(item["nonempty_frames"] for item in available),
                                              "total_scenes": len(selected), "total_frames": summary["total_frames"]}
                    if category == "metrics" and (".gain." in key or key.endswith(".excess")):
                        summary["signed_scene_counts"][key] = {"positive": sum(v > 0 for v in values),
                                                                 "zero": sum(v == 0 for v in values),
                                                                 "negative": sum(v < 0 for v in values),
                                                                 "null": len(selected) - len(values)}
        splits[split] = summary
    return scenes, splits


def decision_rules(records, splits, complete):
    dev = splits["dev"]
    def val(key, category="metrics"):
        return dev[category].get(key, {}).get("value")
    def at_least(key, threshold, category="metrics"):
        value = val(key, category)
        return value is not None and value >= threshold
    def positive(key):
        value = val(key)
        return value is not None and value > 0
    def wins(key):
        return dev["signed_scene_counts"].get(key, {}).get("positive", 0) >= 20
    frames = [r for r in records if r["split"] == "dev"]
    per_frame = bool(frames) and all(r["support"][key]["value"] > 0 for r in frames
                                    for key in ("L.common_count", "R.common_count", "stereo.F_count", "stereo.J_count"))
    rule1 = complete and per_frame and all(at_least(k, .5, "support") for k in
                                          ("L.common_fraction", "R.common_fraction", "stereo.J_over_F"))
    individual, excess = {}, {}
    for view in VIEWS:
        mae = val(f"target.{view}.primary.Actual.C.mae")
        threshold = max(1/255, .1*mae) if mae is not None else None
        key = f"target.{view}.gain.Actual"
        individual[view] = {"threshold": threshold, "passed": threshold is not None and at_least(key, threshold) and wins(key)}
        key = f"target.{view}.excess"
        excess[view] = {"threshold": 1/255, "passed": at_least(key, 1/255) and wins(key)}
    rule2 = all(item["passed"] for item in individual.values())
    rule3 = all(item["passed"] for item in excess.values())
    rule4 = all(positive(k) and wins(k) for k in ("stereo.gain.Actual", "stereo.excess"))
    return {"rule1_complete_applicable": bool(rule1), "rule2_individual": rule2, "rule2_details": individual,
            "rule3_beyond_two_static_controls": rule3, "rule3_details": excess, "rule4_stereo": rule4,
            "all_four_provisional": bool(rule1 and rule2 and rule3 and rule4),
            "independent_saved_artifact_audit_required": True, "no_automatic_compute_release": True}


def safe_path(root, relative):
    path = Path(relative)
    require(not path.is_absolute() and ".." not in path.parts and "TEST" not in path.parts,
            "unsafe/non-TRAIN dataset member")
    resolved = (root / path).resolve()
    require(resolved.is_relative_to(root.resolve()), "dataset path escaped configured root")
    return resolved


def validate_manifest(manifest, index, source_index_sha):
    sequences = source_io.validate_index(index)
    require(manifest["source_index_sha256"] == source_index_sha, "manifest/source-index identity disagreement")
    seen, split_ids = set(), {}
    for split, expected_rows, expected_scenes in (("train", 960, 120), ("dev", 60, 30)):
        rows = manifest[split]
        require(len(rows) == expected_rows, "incomplete fixed cohort")
        expected = [(seq, center) for seq in sequences if seq["split"] == split
                    for center in (range(1, 9) if split == "train" else (3, 6))]
        for row, (sequence, center) in zip(rows, expected):
            frame = sequence["frames"][center]
            require(row["source_id"] == sequence["source_id"] and row["split"] == split
                    and row["center_index"] == row["source_time"] == center, "cohort/order/center mismatch")
            require(row["source_frame"] == frame["source_frame"]
                    and row["sample_id"] == sequence["source_id"] + "/" + frame["source_frame"], "target frame mismatch")
            require(row["sample_id"] not in seen, "duplicate sample")
            seen.add(row["sample_id"])
            require(row["spike_shape"] == [50, 540, 960] and row["vertical_flip"] is False
                    and row["half_frame_future_lookahead"] is True, "shape/time convention mismatch")
            require(row["global_indices"] == codec.window_indices(center)
                    and row["global_indices"][24] == 50 * center, "bin24 target mismatch")
            for key in ("rgb_l", "rgb_r", "disparity_l"):
                require(row[key] == frame[key], "actual source target mismatch: " + key)
            for view in ("l", "r"):
                require(row["spike_" + view] == f'{sequence["source_id"]}/spike_{view}/{frame["source_frame"]}.dat',
                        "DAT identity mismatch")
        split_ids[split] = {row["source_id"] for row in rows}
        require(len(split_ids[split]) == expected_scenes, "sequence count mismatch")
    require(not split_ids["train"] & split_ids["dev"], "split leakage")


def critical_checks():
    started = time.monotonic()
    output = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(CAMPAIGN / "tests"), pattern="test_target_time_observation.py")
    result = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)
    require(result.testsRun > 0 and result.wasSuccessful() and not result.skipped,
            "mandatory synthetic checks failed:\n" + output.getvalue())
    return {"status": "PASS_SYNTHETIC_SOFTWARE_CHECKS_ONLY", "tests": result.testsRun,
            "seconds": time.monotonic() - started, "output": output.getvalue(), "payload_reads": 0}


def write_tables(output, records, scenes):
    for name, rows, scene in (("frame", records, False), ("scene", scenes, True)):
        for category in ("metrics", "support"):
            path = output / (name + "-" + category + ".csv")
            fields = ["split", "source_id"] + ([] if scene else ["sample_id", "source_frame", "manifest_ordinal"])
            fields += ["key", "value", "nonempty_frames", "total_frames", "support_count_sum"] if scene else ["key", "value", "support_count"]
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for row in rows:
                    identity = {key: row[key] for key in fields if key in row and key not in ("metrics", "support")}
                    for key, values in row[category].items():
                        cells = {**identity, "key": key, **values}
                        writer.writerow({key: "null" if value is None else value for key, value in cells.items()})


def verify_execution(args):
    require(os.environ.get("SLURM_JOB_ID") and socket.gethostname() == "cccl3", "payload run requires cccl3 Slurm")
    require(os.environ.get("USER") == "rongliang", "unexpected cluster user")
    require(int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) == 4, "exact4CPU allocation required")
    require(os.environ.get("SLURM_MEM_PER_NODE") == "16384", "exact16GiB allocation required")
    require(os.environ.get("SLURM_GPUS", "0") in ("0", "")
            and os.environ.get("SLURM_GPUS_ON_NODE", "0") in ("0", "")
            and os.environ.get("CUDA_VISIBLE_DEVICES", "") == "", "S5 requires zeroGPU")
    intent = json.loads(args.intent.read_text())
    registration = json.loads(args.registration.read_text())
    registry = json.loads(args.source_registry.read_text())
    source_root = args.source_registry.resolve().parent
    require(registration["status"] == "registered" and registration["mode"] == "stage5-target-time-observation",
            "external root registration is required")
    require(registration["intent_sha256"] == sha(args.intent), "intent binding mismatch")
    require(registration["source_identity_sha256"] == registry["source_identity_sha256"], "source registration mismatch")
    require(registration["scientific_contract_sha256"] == intent["scientific_contract_sha256"] == CONTRACT_SHA,
            "scientific contract mismatch")
    require(registration["resources"] == intent["resources"] == {"cpus": 4, "memory_gib": 16, "wall_seconds": 7200, "gpus": 0},
            "resource contract mismatch")
    required = intent["source_bindings"]
    for relative, expected in required.items():
        require(registry["source_files"].get(relative) == expected and sha(source_root / relative) == expected,
                "frozen dependency mismatch: " + relative)
    for relative, expected in registry["source_files"].items():
        require(sha(source_root / relative) == expected, "frozen source bytes mismatch: " + relative)
    for path in (Path(__file__).resolve(), args.intent.resolve(), (CAMPAIGN / "analysis/stage5-target-time-observation-contract-astra.md").resolve()):
        relative = str(path.relative_to(source_root))
        require(registry["source_files"].get(relative) == sha(path), "required source is outside bound capsule")
    require(sha(CAMPAIGN / "analysis/stage5-target-time-observation-contract-astra.md") == CONTRACT_SHA, "contract file changed")
    require(sha(args.dataset_storage) == intent["dataset_storage_sha256"] == registration["dataset_storage_sha256"], "storage metadata mismatch")
    require(sha(args.source_index) == intent["source_index_sha256"] == registration["source_index_sha256"], "source-index mismatch")
    return intent, registration, registry


def run(args):
    started = time.monotonic()
    intent, registration, registry = verify_execution(args)
    output = args.output.resolve()
    storage = json.loads(args.dataset_storage.read_text())
    source_root, derived_root = Path(storage["source_root"]).resolve(), Path(storage["derived_root"]).resolve()
    require(source_root == Path(storage["data_root"]).resolve() / "source-data"
            and derived_root == Path(storage["data_root"]).resolve() / "derived", "noncanonical dataset roots")
    require(not output.is_relative_to(source_root) and not output.is_relative_to(derived_root), "output must not write source datasets")
    output.mkdir(parents=True, exist_ok=False)
    try:
        checks = critical_checks()
        save(output / "software-checks.json", checks)
        manifest_path = derived_root / "manifest.json"
        require(sha(manifest_path) == intent["manifest_sha256"] == registration["manifest_sha256"] == MANIFEST_SHA,
                "paired manifest identity mismatch")
        manifest = json.loads(manifest_path.read_text())
        index = json.loads(args.source_index.read_text())
        validate_manifest(manifest, index, sha(args.source_index))
        identities = {"intent_sha256": sha(args.intent), "registration_sha256": sha(args.registration),
                      "scientific_contract_sha256": CONTRACT_SHA, "manifest_sha256": MANIFEST_SHA,
                      "source_index_sha256": sha(args.source_index), "dataset_storage_sha256": sha(args.dataset_storage),
                      "source_registry_sha256": sha(args.source_registry), "source_identity_sha256": registry["source_identity_sha256"],
                      "git_commit": registry["git_commit"], "dirty_manifest_sha256": registry["dirty_manifest_sha256"],
                      "source_files": registry["source_files"], "source_root": str(source_root), "derived_root": str(derived_root),
                      "source_script_sha256": sha(__file__), "manifest_order_preserved": True}
        save(output / "identities.json", identities)
        records = []
        with (output / "frame-records.jsonl").open("w") as stream:
            for split in ("train", "dev"):
                for split_ordinal, row in enumerate(manifest[split]):
                    require(time.monotonic() - started < 7200 - 30, "S5 time cap: evidence incomplete, no cohort shrink")
                    gray32, actual, data_hashes = {}, {}, {}
                    for view in VIEWS:
                        suffix = view.lower()
                        rgb_path = safe_path(source_root, row["rgb_" + suffix])
                        gray32[view] = target_gray(source_io.read_rgb(rgb_path))
                        data_hashes["rgb_" + suffix] = {"relative_path": row["rgb_" + suffix], "sha256": sha(rgb_path)}
                        dat_path = safe_path(derived_root, row["spike_" + suffix])
                        raw = dat_path.read_bytes()  # Exactly one DAT payload read per view/frame.
                        require(len(raw) == 3240000, "native DAT byte count mismatch")
                        data_hashes["spike_" + suffix] = {"relative_path": row["spike_" + suffix], "sha256": hashlib.sha256(raw).hexdigest()}
                        spikes = codec.unpack_spikes(raw, 540, 960)
                        actual[view] = summarize_spikes(spikes)
                        del raw, spikes
                    pfm_path = safe_path(source_root, row["disparity_l"])
                    disparity, pfm_metadata = source_io.read_disparity_pfm(pfm_path)
                    data_hashes["disparity_l"] = {"relative_path": row["disparity_l"], "sha256": sha(pfm_path), "decode": pfm_metadata}
                    streams = {"Actual": actual,
                               "Static0": {v: simulate_static(gray32[v], 0.) for v in VIEWS},
                               "StaticHalf": {v: simulate_static(gray32[v], 2.5) for v in VIEWS}}
                    identity = {"split": split, "source_id": row["source_id"], "sample_id": row["sample_id"],
                                "source_frame": row["source_frame"], "center_index": row["center_index"],
                                "manifest_ordinal": len(records), "split_ordinal": split_ordinal, "input_sha256": data_hashes}
                    record = measure_frame(gray32, streams, disparity, identity)
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                    stream.flush()
                    records.append(record)
                    del streams, actual, gray32, disparity
                    if len(records) % 10 == 0:
                        save(output / "status.json", {"status": "RUNNING", "rows": len(records), "seconds": time.monotonic() - started,
                                                     "no_GPU_or_model": True})
        require(len(records) == 1020 and sum(r["split"] == "train" for r in records) == 960, "incomplete records")
        scenes, splits = reduce_records(records)
        require(len(scenes) == 150 and splits["train"]["total_scenes"] == 120 and splits["dev"]["total_scenes"] == 30,
                "incomplete scene reduction")
        write_tables(output, records, scenes)
        save(output / "scene-records.json", scenes)
        save(output / "split-summary.json", {"splits": splits, "decisions": decision_rules(records, splits, True),
                                             "reduction": "equal available frame values within scene, equal available scenes within split; gains/envelopes computed per frame",
                                             "quantiles": "linear within-frame quantiles averaged with same frame/scene reduction; not pooled percentiles",
                                             "independent_audit_required": True})
        artifacts = [p for p in output.iterdir() if p.name not in ("result.json", "status.json") and p.is_file()]
        usage = resource.getrusage(resource.RUSAGE_SELF)
        result = {"status": "COMPLETED_PENDING_INDEPENDENT_SAVED_ARTIFACT_AUDIT", "rows": 1020, "train": 960, "dev": 60,
                  "scientific_contract_sha256": CONTRACT_SHA, "source_identity_sha256": registry["source_identity_sha256"],
                  "runtime_seconds": time.monotonic() - started, "peak_RSS_bytes": int(usage.ru_maxrss * 1024),
                  "CPU_user_seconds": usage.ru_utime, "CPU_system_seconds": usage.ru_stime, "process_exit_code": 0,
                  "Slurm_final_accounting_pending": True, "slurm_job_id": os.environ["SLURM_JOB_ID"], "host": socket.gethostname(),
                  "cpus": 4, "gpus": 0, "numpy": np.__version__, "python": sys.version, "platform": platform.platform(),
                  "read_only_datasets": True, "neural_model_imports": False, "full_native_cohort": True,
                  "artifacts_sha256": {p.name: sha(p) for p in sorted(artifacts)}}
        save(output / "result.json", result)
        save(output / "status.json", {"status": result["status"], "rows": 1020})
    except BaseException as error:
        save(output / "failure.json", {"status": "INCOMPLETE", "error": repr(error), "runtime_seconds": time.monotonic() - started,
                                       "automatic_retry_or_extension": False, "process_exit_code": 1})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("registration", "source-registry", "intent", "dataset-storage", "source-index", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
