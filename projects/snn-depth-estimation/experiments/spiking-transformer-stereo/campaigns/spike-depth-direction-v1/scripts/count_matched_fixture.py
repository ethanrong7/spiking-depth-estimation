"""S7 fixed exact sample-and-hold fixture, not a physical camera or learned method.

Full rendering/oracle execution is restricted to the registered CPU Slurm entry.
Only the named tiny synthetic examples may execute on the control-plane Mac.
The oracle API sees observed spikes and candidate irradiance only; charge fields,
observed phase IDs, target masks and the relationship between episodes are absent.
"""
import argparse
import csv
from collections import Counter
from fractions import Fraction as F
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import socket
import sys
import time
import unittest

import numpy as np

N, Q, TARGET, H, W, D = 50, 5, 24, 32, 48, 5_000_000
BG, DISPNUM = 12_500_000, 28_800_000
OBSERVED = (0, 13)
VIEWS = ("L", "R")
CONTRACT_SHA = "f6c0897fc41013309b17700b7d33ca4dc44775ff688286134894403d88197498"
S = (0,125333,248690,368125,481754,587785,684547,770513,844328,904827,951057,982287,998027,998027,982287,951057,904827,844328,770513,684547,587785,481754,368125,248690,125333,0,-125333,-248690,-368125,-481754,-587785,-684547,-770513,-844328,-904827,-951057,-982287,-998027,-998027,-982287,-951057,-904827,-844328,-770513,-684547,-587785,-481754,-368125,-248690,-125333)
C = (1000000,992115,968583,929776,876307,809017,728969,637424,535827,425779,309017,187381,62791,-62791,-187381,-309017,-425779,-535827,-637424,-728969,-809017,-876307,-929776,-968583,-992115,-1000000,-992115,-968583,-929776,-876307,-809017,-728969,-637424,-535827,-425779,-309017,-187381,-62791,62791,187381,309017,425779,535827,637424,728969,809017,876307,929776,968583,992115)
SUMMARY_FIELDS = ("count50", "first", "last", "a2", "a1", "b1", "b2", "count20_28", "count16_32", "count12_36", "count0_48")
CAMPAIGN = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rational(value):
    value = F(value)
    return {"numerator": value.numerator, "denominator": value.denominator, "display": float(value)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temp = path.with_suffix(path.suffix + ".part")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def base_index(local_bin, phase):
    return (local_bin + phase) % N


def project_world(x, y, z, view):
    """Exact known-world-point projection, also used by named calibration tests."""
    cx = F(-3 if view == "L" else 3, 50)
    return F(47, 2) + 48 * (F(x) - cx) / F(z), F(31, 2) + 48 * F(y) / F(z)


def classify_local(xnum, ynum, denominator):
    """Integer half-open geometry and checker; black foreground remains a hit."""
    hit = ((4 * xnum >= -denominator) & (4 * xnum < denominator)
           & (40 * ynum >= -7 * denominator) & (40 * ynum < 7 * denominator))
    a = (20 * xnum + 5 * denominator) // denominator
    b = (40 * ynum + 7 * denominator) // (2 * denominator)
    intensity = np.where(hit, (a + b) % 2, 0).astype(np.uint8)
    return hit, intensity


def render_pixels(k, view, u, v):
    """Render supplied integer pixel coordinates only; decisions remain int64."""
    require(view in VIEWS and 0 <= k < N, "invalid pose/view")
    u, v = np.asarray(u, dtype=np.int64), np.asarray(v, dtype=np.int64)
    camera_num = -300_000 if view == "L" else 300_000
    zn = D + C[k]
    den = 96 * D
    xi = 96 * camera_num + (2 * u - 47) * zn - 96 * S[k]
    eta = (2 * v - 31) * zn
    hit, intensity = classify_local(xi, eta, den)
    return intensity, np.where(hit, zn, BG).astype(np.int64), hit


def target_visibility(k, depth_num, u, v):
    """Exact continuous right-ray visibility of a left target point.

    Xi at rectangle depth has denominator96*D*z_point. Maximum intermediate
    magnitude is below int64 for the fixed32x48 fixture, including40*eta.
    """
    zp = np.asarray(depth_num, dtype=np.int64)
    u, v = np.asarray(u, dtype=np.int64), np.asarray(v, dtype=np.int64)
    zr = D + C[k]
    den = 96 * D * zp
    xi = 96 * 300_000 * zp + (-600_000 * 96 + (2 * u - 47) * zp) * zr - 96 * S[k] * zp
    eta = (2 * v - 31) * zr * zp
    hit, _ = classify_local(xi, eta, den)
    nearest = np.where(hit, zr, BG)
    qnum = u * zp - DISPNUM
    return (qnum >= 0) & (qnum <= 47 * zp) & (nearest == zp)


def charge_field(view, height=H, width=W):
    require(view in VIEWS, "invalid charge-field view")
    return np.array([[int.from_bytes(hashlib.sha256(
        f"S7-v1|view={view}|v={v}|u={u}".encode("utf-8")).digest()[:8], "big") % Q
        for u in range(width)] for v in range(height)], dtype=np.int16)


def encode(irradiance, initial_charge):
    """Fresh integer IF episode; return spikes and final residual, never rotate spikes."""
    intensity = np.asarray(irradiance)
    require(intensity.shape[0] == N and np.all((intensity == 0) | (intensity == 1)), "need50binary charge bins")
    initial = np.asarray(initial_charge)
    require(np.issubdtype(initial.dtype, np.integer) and np.all((initial >= 0) & (initial < Q)), "integer generating charge in[0,5) required")
    voltage = np.broadcast_to(initial, intensity.shape[1:]).astype(np.int16).copy()
    require(np.all((voltage >= 0) & (voltage < Q)), "initial charge outside[0,5)")
    spikes = np.empty(intensity.shape, dtype=np.uint8)
    for t in range(N):
        total = voltage + intensity[t]
        spikes[t] = total >= Q
        voltage = total - Q * spikes[t].astype(np.int16)
        require(np.all((voltage >= 0) & (voltage < Q)), "integer reset invariant failed")
    require(np.array_equal(Q * spikes.sum(axis=0, dtype=np.int64), initial.astype(np.int64) + intensity.sum(axis=0, dtype=np.int64) - voltage),
            "charge conservation failed")
    return spikes, voltage


def summary(spikes):
    """Exact11-field censored summary, inclusive windows, sentinel−1 independently."""
    spikes = np.asarray(spikes)
    require(spikes.shape[0] == N and np.all((spikes == 0) | (spikes == 1)), "need50bit record")
    t = np.arange(N).reshape((N,) + (1,) * (spikes.ndim - 1))
    present = spikes.astype(bool)
    first = np.min(np.where(present, t, N), axis=0)
    first = np.where(first == N, -1, first)
    last = np.max(np.where(present, t, -1), axis=0)
    before = np.sort(np.where(present & (t < TARGET), t, -1), axis=0)
    after = np.sort(np.where(present & (t >= TARGET), t, N), axis=0)
    b1, b2 = (np.where(after[k] == N, -1, after[k]) for k in (0, 1))
    return np.stack((spikes.sum(axis=0), first, last, before[-2], before[-1], b1, b2,
                     *(spikes[a:b+1].sum(axis=0) for a, b in ((20, 28), (16, 32), (12, 36), (0, 48))))).astype(np.int16)


def baseline_components(fields):
    """Observed timing estimates as exact numerator/denominator arrays;den0=null.

    Order: one target interval, three target intervals, first-to-last.
    Units: charge per bin. These diagnostics do not feed either oracle.
    """
    count, first, last, a2, a1, b1, b2 = fields[:7]
    valid = np.stack(((a1 >= 0) & (b1 >= 0), (a2 >= 0) & (b2 >= 0), (first >= 0) & (last > first)))
    numerator = np.stack((np.full_like(count, Q), np.full_like(count, 3*Q), Q*(count-1)))
    denominator = np.stack((b1-a1, b2-a2, last-first))
    return np.where(valid, numerator, 0).astype(np.int16), np.where(valid, denominator, 0).astype(np.int16)


def first_incompatible(match_bits):
    locations = np.argwhere(match_bits == 0)
    return locations[0].astype(int).tolist() if len(locations) else None


def representative_cells(match_bits):
    witness = np.full(match_bits.shape, -1, dtype=np.int8)
    for r in reversed(range(Q)):
        witness[(match_bits & (1 << r)) != 0] = r
    return witness


def full_bounds(observed_spikes, candidate_irradiance):
    """Oracle API: only observation and candidate charge; continuous unknown phi."""
    observed_spikes, candidate_irradiance = np.asarray(observed_spikes), np.asarray(candidate_irradiance)
    require(observed_spikes.shape == candidate_irradiance.shape and observed_spikes.shape[0] == N,
            "oracle record/irradiance shape mismatch")
    require(np.all((observed_spikes == 0) | (observed_spikes == 1))
            and np.all((candidate_irradiance == 0) | (candidate_irradiance == 1)), "oracle needs binary arrays")
    zero = np.zeros_like(observed_spikes[:1], dtype=np.int16)
    n = np.concatenate((zero, np.cumsum(observed_spikes, axis=0, dtype=np.int16)))
    a = np.concatenate((zero, np.cumsum(candidate_irradiance, axis=0, dtype=np.int16)))
    difference = Q * n - a
    return difference.max(axis=0), (difference + Q).min(axis=0)


def oracle_pixel_evidence(observed_spikes, candidate_irradiance, representatives=None):
    """No phase, charge-field, GT, masks, or paired-episode inputs.

    Cell r stands for every real initial charge in[r,r+1). Bitr records a match.
    Analytic bounds must agree with exact representative RAW-record matching.
    """
    if representatives is None:
        representatives = np.stack([encode(candidate_irradiance, r)[0] for r in range(Q)])
    obs_summary = summary(observed_spikes)
    lo, hi = full_bounds(observed_spikes, candidate_irradiance)
    raw_bits = np.zeros(lo.shape, dtype=np.uint8)
    summary_bits = np.zeros(lo.shape, dtype=np.uint8)
    count_bits = np.zeros(lo.shape, dtype=np.uint8)
    for r in range(Q):
        raw_bits |= (np.all(representatives[r] == observed_spikes, axis=0).astype(np.uint8) << r)
        summary_bits |= (np.all(summary(representatives[r]) == obs_summary, axis=0).astype(np.uint8) << r)
        count_bits |= ((representatives[r].sum(axis=0) == observed_spikes.sum(axis=0)).astype(np.uint8) << r)
    require(np.array_equal(lo < hi, raw_bits != 0), "analytic/full-five-cell disagreement")
    require(np.all((raw_bits == 0) | (summary_bits != 0)), "full not contained in summary at pixel")
    return lo, hi, raw_bits, summary_bits, count_bits


def target_partition(depth_fields, mask):
    """Exact all-candidate target equality on one fixed common mask."""
    keys, groups, class_ids = {}, [], []
    for h, field in enumerate(depth_fields):
        key = np.asarray(field, dtype=np.int64)[mask].tobytes()
        if key not in keys:
            keys[key] = len(groups)
            groups.append([])
        group = keys[key]
        groups[group].append(h)
        class_ids.append(group)
    return groups, class_ids


def describe_set(candidates, class_ids, correct_class):
    classes = sorted({class_ids[h] for h in candidates})
    return {"candidates": candidates, "target_classes": classes, "target_class_count": len(classes),
            "all_have_correct_target_field": bool(candidates) and classes == [correct_class]}


def plane_multiset(spikes):
    """Exact plane bytes and multiplicity; synchronized pairs concatenate spatially."""
    counts = Counter(np.packbits(plane.reshape(-1), bitorder="big").tobytes() for plane in spikes)
    return counts


def multiset_payload(first, second):
    a, b = plane_multiset(first), plane_multiset(second)
    return {"equal": a == b, "episodes": [{"unique_planes": len(c), "total_planes": sum(c.values()),
             "members": [{"packed_plane_hex": p.hex(), "multiplicity": m} for p, m in sorted(c.items())]}
            for c in (a, b)]}


def mae_bound(depth0, depth1, mask):
    population = int(np.count_nonzero(mask))
    if not population:
        return None
    return sum((abs(F(DISPNUM, int(a)) - F(DISPNUM, int(b)))
                for a, b in zip(depth0[mask], depth1[mask])), F(0)) / (2 * population)


def classify_outcome(bound, oracle_sets):
    if bound is None:
        return "UNSUITABLE_SUPPORT"
    if bound == 0:
        return "NO_COUNT_TARGET_AMBIGUITY"
    if not all(e["full"]["stereo"]["all_have_correct_target_field"] for e in oracle_sets):
        return "COUNT_AMBIGUOUS_RAW_UNRESOLVED"
    if all(e["summary"]["stereo"]["all_have_correct_target_field"] for e in oracle_sets):
        return "COUNT_GAP_SUMMARIES_SUFFICE"
    return "RAW_ADDS_TARGET_DISCRIMINATION"


class StopFixture(Exception):
    pass


def fixed_fixture(output, guard):
    """Complete fixed cohort, called only after Slurm/source/registration guards."""
    counters = {"rendered_view_frames": 0, "verified_phase_totals": 0, "completed_candidate_views": 0,
                "completed_episode_candidate_views": 0, "full_five_cell_pixel_comparisons": 0}
    arrays, rows, stage = {}, [], "render"
    try:
        v, u = np.indices((H, W), dtype=np.int64)
        irradiance = np.empty((2, N, H, W), dtype=np.uint8)
        depth = np.empty((2, N, H, W), dtype=np.int64)
        foreground = np.empty((2, N, H, W), dtype=bool)
        for vi, view in enumerate(VIEWS):
            for k in range(N):
                guard()
                irradiance[vi, k], depth[vi, k], foreground[vi, k] = render_pixels(k, view, u, v)
                counters["rendered_view_frames"] += 1
        arrays.update(base_irradiance=irradiance, base_depth_num=depth, base_foreground=foreground,
                      pose_s=np.array(S, dtype=np.int64), pose_c=np.array(C, dtype=np.int64))
        stage = "phase_totals_and_observations"
        for vi in range(2):
            total = irradiance[vi].sum(axis=0)
            for h in range(N):
                guard()
                require(np.array_equal(irradiance[vi, (np.arange(N)+h) % N].sum(axis=0), total), "phase total-charge inequality")
                counters["verified_phase_totals"] += 1
        phi = np.stack([charge_field(view) for view in VIEWS])
        observed = np.empty((2, 2, N, H, W), dtype=np.uint8)
        residual = np.empty((2, 2, H, W), dtype=np.int16)
        observed_summaries = np.empty((2, 2, 11, H, W), dtype=np.int16)
        masks = np.empty((2, H, W), dtype=bool)
        for ei, h in enumerate(OBSERVED):
            for vi in range(2):
                observed[ei, vi], residual[ei, vi] = encode(irradiance[vi, (np.arange(N)+h) % N], phi[vi])
                observed_summaries[ei, vi] = summary(observed[ei, vi])
            k = base_index(TARGET, h)
            masks[ei] = target_visibility(k, depth[0, k], u, v)
        require(np.array_equal(observed[0].sum(axis=1), observed[1].sum(axis=1)), "observed full-image count mismatch")
        common = masks[0] & masks[1]
        target_depths = depth[0, (TARGET + np.arange(N)) % N]
        groups, class_ids = target_partition(target_depths, common)
        bound = mae_bound(target_depths[0], target_depths[13], common)
        arrays.update(audit_only_initial_charge=phi, observed_spikes=observed,
                      audit_only_final_residual=residual, observed_summaries=observed_summaries,
                      observed_timing_baseline_numerators=np.stack([np.stack([baseline_components(observed_summaries[e, vi])[0] for vi in range(2)]) for e in range(2)]),
                      observed_timing_baseline_denominators=np.stack([np.stack([baseline_components(observed_summaries[e, vi])[1] for vi in range(2)]) for e in range(2)]),
                      observed_target_valid=masks, common_mask=common, candidate_target_depth_num=target_depths)
        shape = (2, 2, N, H, W)  # episode,view,candidate,v,u
        lower = np.zeros(shape, dtype=np.int16)
        upper = np.zeros(shape, dtype=np.int16)
        raw_bits, summary_bits, count_bits = (np.zeros(shape, dtype=np.uint8) for _ in range(3))
        summary_witness = np.full(shape, -1, dtype=np.int8)
        arrays.update(summary_representative_charge_cell=summary_witness, full_lower_closed=lower, full_upper_open=upper, raw_cell_match_bits=raw_bits,
                      summary_cell_match_bits=summary_bits, count_cell_match_bits=count_bits)
        stage = "independent_episode_oracles"
        for h in range(N):
            for vi in range(2):
                guard()
                candidate = irradiance[vi, (np.arange(N)+h) % N]
                representatives = np.stack([encode(candidate, r)[0] for r in range(Q)])
                for ei in range(2):
                    guard()
                    evidence = oracle_pixel_evidence(observed[ei, vi], candidate, representatives)
                    lower[ei, vi, h], upper[ei, vi, h], raw_bits[ei, vi, h], summary_bits[ei, vi, h], count_bits[ei, vi, h] = evidence
                    summary_witness[ei, vi, h] = representative_cells(evidence[3])
                    require(np.all(count_bits[ei, vi, h] != 0), "count-only oracle lost a phase")
                    rows.append({"episode_index": ei, "candidate_phase": h, "view": VIEWS[vi],
                                 "full_incompatible_pixels": int(np.count_nonzero(evidence[0] >= evidence[1])),
                                 "summary_incompatible_pixels": int(np.count_nonzero(evidence[3] == 0)),
                                 "count_incompatible_pixels": int(np.count_nonzero(evidence[4] == 0)),
                                 "full_first_incompatible_vu": first_incompatible(evidence[2]),
                                 "summary_first_incompatible_vu": first_incompatible(evidence[3]),
                                 "count_first_incompatible_vu": first_incompatible(evidence[4]),
                                 "candidate_target_class": class_ids[h]})
                    counters["completed_episode_candidate_views"] += 1
                    counters["full_five_cell_pixel_comparisons"] += H * W
                counters["completed_candidate_views"] += 1
        stage = "external_generating_audit_and_reduction"
        sets = []
        for ei, phase in enumerate(OBSERVED):
            guard()
            # The generating phase/field is used ONLY outside the oracle, after ALL decisions.
            for vi in range(2):
                require(np.all((lower[ei, vi, phase] <= phi[vi]) & (phi[vi] < upper[ei, vi, phase])),
                        "generating charge not in continuous full-record bounds")
            episode = {"observed_phase_for_audit_only": phase, "correct_target_class": class_ids[phase]}
            for name, match in (("full", raw_bits), ("summary", summary_bits), ("count", count_bits)):
                by_view = [np.flatnonzero(np.all(match[ei, vi] != 0, axis=(1, 2))).tolist() for vi in range(2)]
                require(all(phase in candidates for candidates in by_view), "correct candidate missing")
                intersection = sorted(set(by_view[0]) & set(by_view[1]))
                episode[name] = {key: describe_set(candidates, class_ids, class_ids[phase])
                                 for key, candidates in zip(("L", "R", "stereo"), (*by_view, intersection))}
            for key in ("L", "R", "stereo"):
                require(set(episode["full"][key]["candidates"]) <= set(episode["summary"][key]["candidates"]), "view/stereo containment failed")
                require(episode["count"][key]["candidates"] == list(range(N)), "count view/stereo phase loss")
            sets.append(episode)
        multiset = {view: multiset_payload(observed[0, vi], observed[1, vi]) for vi, view in enumerate(VIEWS)}
        multiset["synchronized_LR"] = multiset_payload(np.concatenate((observed[0, 0], observed[0, 1]), axis=2),
                                                       np.concatenate((observed[1, 0], observed[1, 1]), axis=2))
        f0, f1 = foreground[0, TARGET], foreground[0, base_index(TARGET, 13)]
        support = {"individual_valid_counts": masks.sum(axis=(1, 2)).astype(int).tolist(), "common_count": int(common.sum()),
                   "common_fraction": rational(F(int(common.sum()), H*W)),
                   "changed_disparity_pixels": int(np.count_nonzero(common & (target_depths[0] != target_depths[13]))),
                   "foreground_background_membership": {f"{a}{b}": int(np.count_nonzero(common & (f0 == a) & (f1 == b)))
                                                         for a in (0, 1) for b in (0, 1)}}
        guard()
        result = {"outcome": classify_outcome(bound, sets), "support": support,
                  "count_only_paired_MAE_lower_bound_native_pixels": None if bound is None else rational(bound),
                  "target_partition": groups, "target_class_by_phase": class_ids, "oracle_sets": sets,
                  "unordered_plane_multisets": multiset, "identities": {"all_phase_irradiance_totals_equal": True,
                  "observed_count_fields_equal": True, "integer_reset_and_conservation": True,
                  "analytic_full_equals_five_cell_raw_every_pixel": True, "full_subset_summary": True,
                  "count_retains_all50": True, "generating_charge_and_candidate_retained": True},
                  "stop_reason": "complete_fixed_fixture"}
    except StopFixture as error:
        result = {"outcome": "INCOMPLETE", "stop_reason": str(error)}
    except ValueError as error:
        result = {"outcome": "INVALID", "stop_reason": str(error)}
    # Raw arrays and plane bytes remain outside Git. Partial arrays are explicitly not certificates.
    guard_before_save_seconds = time.monotonic()
    np.savez(output / "fixture-arrays.npz", **arrays)
    with (output / "oracle-rows.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    with (output / "candidate-compatibility.csv").open("w", newline="") as stream:
        columns = ("episode_index", "candidate_phase", "view", "candidate_target_class", "full_incompatible_pixels", "summary_incompatible_pixels", "count_incompatible_pixels")
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    result.update(stage=stage, counters=counters, arrays_axis_contract={"base": "view,time,v,u",
                  "observed_spikes": "episode,view,time,v,u", "oracle_arrays": "episode,view,candidate,v,u",
                  "views": list(VIEWS), "observed_phases_for_audit_only": list(OBSERVED),
                  "cell_bits": "bit r: exact representative charge r; r=0..4", "summary_representative_charge_cell": "smallest compatible r; -1 if none",
                  "timing_baselines": "episode,view,estimate,v,u; estimate=one_interval,three_intervals,first_to_last;denominator0=null;charge/bin", "depth_denominator": D},
                  artifact_write_seconds=time.monotonic()-guard_before_save_seconds)
    return result


def critical_checks():
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(CAMPAIGN / "tests"), pattern="test_count_matched_fixture.py")
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    require(result.testsRun == 11 and result.wasSuccessful() and not result.skipped, "named tiny checks failed:\n" + stream.getvalue())
    return {"status": "PASS_TINY_NAMED_EXAMPLES_ONLY", "tests": result.testsRun, "output": stream.getvalue(),
            "full_fixture_oracle_executed": False, "dataset_model_access": False}


def execute(args):
    started = time.monotonic()
    require(os.environ.get("SLURM_JOB_ID") and socket.gethostname() == "cccl3"
            and os.environ.get("USER") == "rongliang", "full fixture requires registered cccl3 Slurm")
    require(os.environ.get("SLURM_CPUS_PER_TASK") == "1" and os.environ.get("SLURM_MEM_PER_NODE") == "2048", "need1CPU2GiB")
    require(os.environ.get("SLURM_GPUS", "0") in ("", "0") and os.environ.get("SLURM_GPUS_ON_NODE", "0") in ("", "0")
            and os.environ.get("CUDA_VISIBLE_DEVICES", "") == "", "zero GPU required")
    intent, reg, registry = (json.loads(path.read_text()) for path in (args.intent, args.registration, args.source_registry))
    require(reg["status"] == "registered" and reg["mode"] == "stage7-count-matched-fixture", "external root registration required")
    require(reg["intent_sha256"] == sha(args.intent) and reg["source_identity_sha256"] == registry["source_identity_sha256"], "source/intent binding mismatch")
    require(reg["scientific_contract_sha256"] == intent["scientific_contract_sha256"] == CONTRACT_SHA, "contract changed")
    require(reg["resources"] == intent["resources"] == {"cpus": 1, "memory_gib": 2, "wall_seconds": 900, "gpus": 0}, "resources changed")
    require(intent["internal_wall_stop_seconds"] == 895 and intent["observed_phases"] == [0, 13]
            and intent["candidate_phases"] == list(range(50)) and intent["shape"] == [50, 32, 48]
            and intent["threshold"] == 5 and intent["target_bin"] == 24, "fixed scope changed")
    source = args.source_registry.resolve().parent
    for relative, expected in registry["source_files"].items():
        require(sha(source / relative) == expected, "frozen source changed: " + relative)
    contract = CAMPAIGN / "analysis/stage7-count-matched-fixture-contract-astra.md"
    require(sha(contract) == CONTRACT_SHA, "contract file changed")
    for path in (Path(__file__).resolve(), args.intent.resolve(), contract.resolve()):
        require(registry["source_files"].get(str(path.relative_to(source))) == sha(path), "required source outside capsule")
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        save(args.output / "software-checks.json", critical_checks())
        def guard():
            if time.monotonic() - started >= 895:
                raise StopFixture("fixed895second_internal_stop_no_retry")
        result = fixed_fixture(args.output, guard)
        usage = resource.getrusage(resource.RUSAGE_SELF)
        result.update(status="FIXTURE_RECEIPT_PENDING_INDEPENDENT_AUDIT", scientific_contract_sha256=CONTRACT_SHA,
                      source_identity_sha256=registry["source_identity_sha256"], source_files=registry["source_files"],
                      registration_sha256=sha(args.registration), intent_sha256=sha(args.intent), git_commit=registry["git_commit"],
                      dirty_manifest_sha256=registry["dirty_manifest_sha256"], runtime_seconds=time.monotonic()-started,
                      peak_RSS_bytes=int(usage.ru_maxrss*1024), CPU_user_seconds=usage.ru_utime, CPU_system_seconds=usage.ru_stime,
                      cpus=1, gpus=0, Slurm_job_id=os.environ["SLURM_JOB_ID"], python=sys.version, numpy=np.__version__,
                      dataset_access=False, model_imports=False, optimizer_updates=0, automatic_continuation=False,
                      scientific_constants={"N": N, "Q": Q, "target_bin": TARGET, "height": H, "width": W,
                      "focal": 48, "principal": ["47/2", "31/2"], "camera_x": ["-3/50", "3/50"],
                      "rectangle_width_height": ["1/2", "7/20"], "checker_cell": "1/20", "background_depth": "5/2",
                      "pose_s": S, "pose_c": C, "coordinate_denominator": D, "disparity_numerator": DISPNUM,
                      "initial_charge": "first8SHA256bigendian(S7-v1|view={L/R}|v={v}|u={u})mod5",
                      "summary_fields": SUMMARY_FIELDS, "candidate_phases": list(range(N)), "observed_phases": OBSERVED},
                      artifacts={p.name: sha(p) for p in args.output.iterdir() if p.is_file()},
                      independent_saved_artifact_audit_required=True)
        save(args.output / "result.json", result)
        if result["outcome"] == "INVALID":
            raise RuntimeError("fixture invalid; preserved arrays and counters")
    except BaseException as error:
        save(args.output / "failure.json", {"outcome": "INVALID_OR_INCOMPLETE", "stop_reason": repr(error),
                                           "runtime_seconds": time.monotonic()-started, "no_retry_or_relaxation": True})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("registration", "source-registry", "intent", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    execute(parser.parse_args())


if __name__ == "__main__":
    main()
