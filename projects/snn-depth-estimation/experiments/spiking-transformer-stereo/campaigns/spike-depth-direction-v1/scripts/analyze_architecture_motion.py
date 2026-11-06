"""Frozen saved-score/S0 join. No model imports, inference, or source-data reads.

Build a portable final-checkpoint catalog from explicitly named saved histories,
then join that catalog to an independently frozen S0 result. Comparisons stay
within seed and registered budget. Noise-censored motion is unknown, never
static. All associations are descriptive on the reused development cohort.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


EXPERIMENT = Path("projects/snn-depth-estimation/experiments/spiking-transformer-stereo")
RAW = Path("storage/experiment-runs/snn-depth-estimation/spiking-transformer-stereo")
MANIFEST_HASH = "bad4369b6fd8b0cdf871a20b1727c2d9b59733a16bd33f75904227bdfef4510a"
DEV_IDS = {(f"scene_{scene:04d}", frame) for scene in range(120, 150)
           for frame in ("0001", "0255")}
COVARIATES = ("mean_gt_disparity_pixels", "count_mean", "texture_gradient_mean")


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def specifications():
    runs, contrasts = [], []
    definitions = [
        ("original", "paired-loss-v1", "original", "updater-mechanism-v1"),
        ("full_lif", "paired-loss-v1", "spiking-normalized", "updater-mechanism-v1"),
        ("no_attention", "updater-mechanism-v1", "no-attention", "updater-mechanism-v1"),
        ("minimal_lif", "minimal-lif-v1", "minimal-lif", "minimal-lif-v1"),
        ("analog_attention", "analog-attention-v1", "analog-attention", "analog-attention-v1"),
    ]
    for seed in (1234, 2345, 3456):
        for model, campaign, prefix, audit in definitions:
            runs.append(dict(id=f"{model}:{seed}:10000", model=model, seed=seed,
                             updates=10000, campaign=campaign,
                             run=f"{prefix}-seed{seed}-10k-v1", audit_campaign=audit))
        for candidate, baseline in (("full_lif", "original"),
                                    ("no_attention", "full_lif"),
                                    ("minimal_lif", "no_attention"),
                                    ("analog_attention", "full_lif")):
            contrasts.append(dict(id=f"{candidate}_vs_{baseline}:seed{seed}:10k",
                                  candidate=f"{candidate}:{seed}:10000",
                                  baseline=f"{baseline}:{seed}:10000"))
    for seed, campaign in ((1234, "alif-mechanism-screen-v1"),
                           (2345, "alif-mechanism-repeat-v1")):
        for arm in ("O", "I"):
            runs.append(dict(id=f"current_{arm}:{seed}:5000", model=f"current_{arm}",
                             seed=seed, updates=5000, campaign=campaign,
                             run=f"{arm.lower()}-seed{seed}-5k-v1", audit_campaign=campaign))
        contrasts.append(dict(id=f"current_I_vs_current_O:seed{seed}:5k",
                              candidate=f"current_I:{seed}:5000",
                              baseline=f"current_O:{seed}:5000"))
    return runs, contrasts


def build_catalog(root):
    specs, contrasts = specifications()
    runs = []
    for spec in specs:
        directory = RAW / spec["campaign"] / spec["run"]
        paths = {name: directory / f"{name}.json" for name in ("history", "metadata", "result")}
        history, metadata, result = (read(root / paths[k]) for k in ("history", "metadata", "result"))
        audit_path = EXPERIMENT / "campaigns" / spec["audit_campaign"] / "results/audited-summary.json"
        audit = read(root / audit_path)
        require(audit["status"] in ("AUDIT_PASS", "AUDIT_PASSED", "COMPLETED"), "prior audit incomplete")
        require(history[-1]["update"] == spec["updates"] == result["updates_completed"], "final budget mismatch")
        require(result["updates_completed_this_run"] == spec["updates"], "not a complete fresh run")
        require(result["samples_seen_this_run"] == 2 * spec["updates"], "sample budget mismatch")
        require(metadata["restore_checkpoint"] is None and metadata["parent_lineage"] is None,
                "restore lineage present")
        require(metadata["confirmation_selection"] is None, "confirmation access present")
        require(metadata["manifest_sha256"] == MANIFEST_HASH, "prior manifest mismatch")
        require(metadata["config"]["eval_split"] == "dev" and metadata["config"]["train_split"] == "train",
                "unexpected data split")
        endpoint = history[-1]
        frames = [{key: row[key] for key in ("scene", "frame", "valid_pixels", "epe", "bad3")}
                  for row in endpoint["per_frame"]]
        ids = [(row["scene"], row["frame"]) for row in frames]
        require(len(ids) == 60 and len(set(ids)) == 60 and set(ids) == DEV_IDS, "development frame IDs mismatch")
        require(all(finite(row["epe"]) and finite(row["bad3"]) and row["valid_pixels"] > 0
                    for row in frames), "invalid saved frame metric")
        scene_scores = {scene: statistics.mean(row["epe"] for row in frames if row["scene"] == scene)
                        for scene in sorted({row["scene"] for row in frames})}
        require(all(abs(value - endpoint["scene_epe"][scene]) < 1e-8 for scene, value in scene_scores.items()),
                "saved frame/scene mismatch")
        require(abs(statistics.mean(scene_scores.values()) - endpoint["mean_scene_epe"]) < 1e-8,
                "saved mean-scene mismatch")
        semantics = metadata["training_semantics"]
        require(semantics["scheduler"]["total_steps"] == 10100, "schedule mismatch")
        common = {key: value for key, value in semantics.items()
                  if key not in ("model", "seed", "normalize_residual_current", "screen_identity", "data_root")}
        source = metadata["source"]
        core = {key: value for key, value in source["source_files"].items() if "/core/" in key}
        source_binding = dict(git_commit=source["git_commit"],
                              source_identity_sha256=source["source_identity_sha256"],
                              archive_sha256=source["archive_sha256"],
                              core_files_sha256=canonical_hash(core),
                              core_source_files=core,
                              dirty_patch_sha256=source["dirty_patch_sha256"])
        # Existing audits bind last/best binaries. This catalog rechecks saved
        # final evaluation records; it never loads a binary or substitutes best.
        require(any(abs(row.get("final_epe", row.get("analog_final_epe", -1)) - endpoint["mean_scene_epe"]) < 1e-8
                    and row["seed"] == spec["seed"] for row in audit.get("rows", audit.get("runs", []))),
                "final mean not found in cited prior audit")
        runs.append({**spec, "endpoint": "last_scheduled_final_not_best", "manifest_sha256": MANIFEST_HASH,
                     "final_mean_scene_epe": endpoint["mean_scene_epe"], "final_bad3_percent": endpoint["bad3"],
                     "scene_epe": scene_scores, "per_frame": frames, "source_binding": source_binding,
                     "training_contract": common, "training_contract_sha256": canonical_hash(common),
                     "parameter_count": metadata["parameter_count"],
                     "evidence": {key: {"path": str(path), "sha256": digest(root / path)} for key, path in paths.items()},
                     "prior_audit": {"path": str(audit_path), "sha256": digest(root / audit_path)},
                     "checkpoint_verification_scope": "prior saved-artifact audits; binary not loaded by this analysis"})
    catalog = dict(schema_version=1, prepared_date="2026-10-26", frozen_before_S0_outcomes=True,
                   scope="19 unique final Dev checkpoints; no inference, Test access, source-data reads or selected minima",
                   manifest_sha256=MANIFEST_HASH, runs=runs, contrasts=contrasts,
                   exclusions=["F has one seed only and is not an O/I replication", "cross-budget comparisons",
                               "best-selected checkpoints", "unpaired cross-campaign pooling"],
                   gain_definition="baseline_scene_EPE minus candidate_scene_EPE; positive favors candidate",
                   covariates_present_in_saved_prior_scores=False)
    validate_catalog(catalog)
    return catalog


def validate_catalog(catalog):
    require(catalog["manifest_sha256"] == MANIFEST_HASH, "catalog manifest mismatch")
    runs = {run["id"]: run for run in catalog["runs"]}
    require(len(runs) == len(catalog["runs"]), "duplicate run ID")
    for contrast in catalog["contrasts"]:
        candidate, baseline = (runs[contrast[key]] for key in ("candidate", "baseline"))
        require(candidate["seed"] == baseline["seed"] and candidate["updates"] == baseline["updates"],
                "unpaired seed or budget")
        require(candidate["training_contract_sha256"] == baseline["training_contract_sha256"],
                "training contracts differ")
        candidate_core = candidate["source_binding"]["core_source_files"]
        baseline_core = baseline["source_binding"]["core_source_files"]
        require(set(candidate_core) == set(baseline_core), "core file inventory differs")
        changed = {key for key in candidate_core if candidate_core[key] != baseline_core[key]}
        # Original runs predate optional normalized-attention implementation.
        # Those two optional modules are not executed by original ALIF. Existing
        # updater audits verify common encoders/correlation/heads and protocol.
        allowed = ({str(Path("projects/snn-depth-estimation/core") / name)
                    for name in ("spiking_transformer_stereo.py", "spiking_attention_update.py")}
                   if candidate["model"] == "full_lif" and baseline["model"] == "original" else set())
        require(changed <= allowed, "unexpected core source difference")
        require([(r["scene"], r["frame"], r["valid_pixels"]) for r in candidate["per_frame"]] ==
                [(r["scene"], r["frame"], r["valid_pixels"]) for r in baseline["per_frame"]],
                "frame order/validity mismatch")
    return runs


def load_covariates(path):
    if path is None:
        return {}, {key: "missing: requires saved CPU-Slurm covariate records" for key in COVARIATES}
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    by_id = {}
    for row in rows:
        key = (row["scene"], str(row["frame"]).zfill(4))
        if key not in DEV_IDS:
            continue
        require(row.get("model_split") == "dev", "exact Dev covariate frame has wrong split label")
        require(key not in by_id, "duplicate exact-frame covariate record")
        by_id[key] = row
    result = {}
    for scene in sorted({scene for scene, _ in DEV_IDS}):
        result[scene] = {}
        for covariate in COVARIATES:
            values = [by_id.get((scene, frame), {}).get(covariate) for frame in ("0001", "0255")]
            result[scene][covariate] = statistics.mean(values) if all(finite(v) for v in values) else None
    missing = {key: sum(not finite(row[key]) for row in result.values()) for key in COVARIATES}
    return result, missing


def ranks(values):
    return [1 + sum(x < value for x in values) + (sum(x == value for x in values) - 1) / 2
            for value in values]


def correlation(a, b):
    if len(a) < 3 or len(set(a)) < 2 or len(set(b)) < 2:
        return None
    ar, br = ranks(a), ranks(b)
    am, bm = statistics.mean(ar), statistics.mean(br)
    numerator = sum((x - am) * (y - bm) for x, y in zip(ar, br))
    denominator = math.sqrt(sum((x - am) ** 2 for x in ar) * sum((y - bm) ** 2 for y in br))
    return numerator / denominator


def join(catalog, motion, covariates=None):
    runs = validate_catalog(catalog)
    # The S0 Train/Dev-only manifest is intentionally a different file digest
    # from the historical full manifest; frame identities below bind the join.
    eligible = motion.get("eligible_for_motion_stratification") is True
    require(motion.get("Dev_used_for_threshold_or_control_calibration") is False,
            "motion thresholds not verified Train-only")
    require(motion.get("no_Test_or_confirmation") is True, "S0 no-Test contract missing")
    primary = {}
    for row in motion["scene_summaries"]:
        if row["model_split"] == "dev" and row["lag_steps"] == 1 and row["blur_sigma"] == 1.0:
            require(row["scene"] not in primary, "duplicate primary scene row")
            primary[row["scene"]] = row
    require(set(primary) == {scene for scene, _ in DEV_IDS}, "S0 development scene IDs mismatch")
    covariates = covariates or {}
    rows, summaries = [], []
    for contrast in catalog["contrasts"]:
        candidate, baseline = (runs[contrast[key]] for key in ("candidate", "baseline"))
        current = []
        for scene in sorted(candidate["scene_epe"]):
            proxy = primary[scene]
            observed = proxy["median_displacement_proxy"]
            resolved = eligible and proxy.get("above_floor_both_blurs") is True and finite(observed)
            label = proxy.get("primary_ordinal_bin_from_frozen_Train_cuts") if resolved else "unknown"
            require(not resolved or label in ("low", "middle", "high"), "resolved motion label invalid")
            base, cand = baseline["scene_epe"][scene], candidate["scene_epe"][scene]
            row = dict(contrast=contrast["id"], seed=candidate["seed"], updates=candidate["updates"],
                       scene=scene, baseline_model=baseline["model"], candidate_model=candidate["model"],
                       baseline_epe=base, candidate_epe=cand, gain_pixels=base - cand,
                       relative_gain=(base - cand) / base if base > 0 else None,
                       motion_proxy_native_pixels=observed if resolved else None,
                       motion_group=label,
                       motion_missing_reason=None if resolved else "S0_QC_failed" if not eligible else
                       "missing_proxy" if not finite(observed) else "below_proxy_resolution_unknown_not_static")
            row.update({key: covariates.get(scene, {}).get(key) for key in COVARIATES})
            current.append(row)
        counts = {label: sum(row["motion_group"] == label for row in current)
                  for label in ("low", "middle", "high", "unknown")}
        group_ok = (eligible and motion.get("Dev_categorical_high_low_contrast_eligible") is True
                    and counts["low"] >= 10 and counts["high"] >= 10)
        entry = dict(contrast=contrast["id"], seed=candidate["seed"], updates=candidate["updates"],
                     scenes=30, mean_gain_pixels=statistics.mean(row["gain_pixels"] for row in current),
                     motion_group_counts=counts, categorical_contrast_eligible=group_ok,
                     motion_analysis="high_low_descriptive_and_individual_scatter" if group_ok else
                     "individual_continuous_scatter_only" if eligible else "withheld_S0_QC_failed",
                     motion_associations=None, covariate_associations=None)
        if group_ok:
            low, high = ([row for row in current if row["motion_group"] == label] for label in ("low", "high"))
            low_gain, high_gain = (statistics.mean(row["gain_pixels"] for row in group) for group in (low, high))
            resolved_rows = [row for row in current if finite(row["motion_proxy_native_pixels"])]
            entry["motion_associations"] = dict(low_mean_gain_pixels=low_gain, high_mean_gain_pixels=high_gain,
                                               high_minus_low_gain_pixels=high_gain - low_gain,
                                               spearman_gain_vs_motion=correlation(
                                                   [row["motion_proxy_native_pixels"] for row in resolved_rows],
                                                   [row["gain_pixels"] for row in resolved_rows]),
                                               resolved_scenes=len(resolved_rows))
            entry["covariate_associations"] = {}
            for key in COVARIATES:
                complete = [row for row in resolved_rows if finite(row[key])]
                entry["covariate_associations"][key] = dict(
                    complete_resolved_scenes=len(complete),
                    spearman_gain_vs_covariate=correlation([row[key] for row in complete],
                                                         [row["gain_pixels"] for row in complete]),
                    spearman_motion_vs_covariate=correlation([row[key] for row in complete],
                                                           [row["motion_proxy_native_pixels"] for row in complete]))
        summaries.append(entry)
        rows.extend(current)
    return rows, summaries


def plot_contrasts(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for contrast in sorted({row["contrast"] for row in rows}):
        records = [row for row in rows if row["contrast"] == contrast]
        figure, axes = plt.subplots(1, 4, figsize=(16, 3.5), constrained_layout=True)
        for axis, key in zip(axes, ("motion_proxy_native_pixels",) + COVARIATES):
            observed = [row for row in records if finite(row[key])]
            axis.scatter([row[key] for row in observed], [row["gain_pixels"] for row in observed], s=15)
            for row in observed:
                axis.annotate(row["scene"].removeprefix("scene_"), (row[key], row["gain_pixels"]), fontsize=5)
            axis.axhline(0, color="gray", linewidth=.6)
            axis.set_xlabel(key)
            axis.set_ylabel("Baseline EPE - candidate EPE (pixels)")
            axis.set_title(f"{len(observed)} observed scenes; no trend fit")
        figure.suptitle(contrast + " | descriptive reused Dev; unknown motion omitted")
        figure.savefig(output / (contrast.replace(":", "-") + ".pdf"))
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--build-catalog", action="store_true")
    parser.add_argument("--motion-result", type=Path)
    parser.add_argument("--covariates", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    if args.build_catalog:
        require(not args.catalog.exists(), "refuse replacing a frozen score catalog")
        catalog = build_catalog(args.repo_root)
        args.catalog.parent.mkdir(parents=True, exist_ok=True)
        args.catalog.write_text(json.dumps(catalog, indent=2, allow_nan=False) + "\n")
    else:
        catalog = read(args.catalog)
        validate_catalog(catalog)
    if args.motion_result is None:
        print(json.dumps(dict(status="CATALOG_VALID_MOTION_NOT_SUPPLIED", runs=len(catalog["runs"]),
                              contrasts=len(catalog["contrasts"]))))
        return
    require(args.output is not None, "--output required for motion join")
    covariates, missing = load_covariates(args.covariates)
    motion = read(args.motion_result)
    rows, summaries = join(catalog, motion, covariates)
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "paired-scene-join.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = dict(status="DESCRIPTIVE_SAVED_EVIDENCE_JOIN", catalog_sha256=digest(args.catalog),
                  motion_result_sha256=digest(args.motion_result), source_script_sha256=digest(__file__),
                  covariates_sha256=digest(args.covariates) if args.covariates else None,
                  missing_covariates=missing, contrasts=summaries, seeds_pooled=False,
                  rows=len(rows), no_model_inference_or_source_dataset_reads=True,
                  causal_interpretation="none; inter-file count-reconstruction displacement, no physical speed or within-bin mechanism",
                  limits="reused Dev; model/operator packages differ; no population significance or independent scene-seed replicates; pairwise covariate associations do not adjust confounding")
    (args.output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if args.plots:
        plot_contrasts(rows, args.output)
    print(json.dumps(dict(status=report["status"], rows=len(rows), contrasts=len(summaries))))


if __name__ == "__main__":
    main()
