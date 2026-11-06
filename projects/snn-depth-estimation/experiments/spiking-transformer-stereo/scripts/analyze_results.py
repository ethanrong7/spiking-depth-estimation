"""Analyze saved GPU predictions; this utility never runs a model or reads datasets.

Example: --run original=/raw/original --run spiking=/raw/spiking --priors /raw/priors.npz
Input directories contain predictions/<scene>-<frame>.npz produced by run_experiment.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def gradient_magnitude(image, valid):
    """Forward differences only where both endpoints have valid ground truth."""
    dx, dy = np.zeros_like(image, dtype=float), np.zeros_like(image, dtype=float)
    dx[:, :-1] = np.where(valid[:, :-1] & valid[:, 1:], np.diff(image, axis=1), 0)
    dy[:-1] = np.where(valid[:-1] & valid[1:], np.diff(image, axis=0), 0)
    return np.hypot(dx, dy)


def padded_head_disparity_bound(width, padding_multiple=32, feature_stride=4):
    """Supremum from coarse d<=x, bilinear align_corners=False, symmetric unpad."""
    padded_width = int(np.ceil(width / padding_multiple)) * padding_multiple
    left_padding = (padded_width - width) // 2
    coarse_width = padded_width // feature_stride
    padded_x = np.arange(width) + left_padding
    coarse_x = (padded_x + .5) / feature_stride - .5
    return feature_stride * np.interp(coarse_x, np.arange(coarse_width), np.arange(coarse_width))


def image_metrics(prediction, truth, valid, input_rate):
    if any(value.shape != truth.shape for value in (prediction, valid, input_rate)):
        raise ValueError("prediction, truth, mask and input shapes must match")
    valid = valid.astype(bool) & np.isfinite(truth)
    if not valid.any() or not np.isfinite(prediction[valid]).all():
        raise ValueError("empty validity mask or nonfinite prediction")
    error = np.abs(prediction - truth)
    gt_gradient = gradient_magnitude(truth, valid)
    texture = gradient_magnitude(input_rate, valid)
    x = np.broadcast_to(np.arange(truth.shape[1]), truth.shape)
    right_x = x + truth  # model horizontal flow is negative positive-disparity magnitude
    out_of_view = (right_x < 0) | (right_x > truth.shape[1] - 1)
    regimes = {"all": valid, "edge": valid & (gt_gradient >= 1.0),
               "flat": valid & (gt_gradient < 1.0),
               "low_texture": valid & (texture <= 0.01),
               "in_view": valid & ~out_of_view,
               "out_of_view": valid & out_of_view}
    result = {}
    for name, mask in regimes.items():
        result[name + "_pixels"] = int(mask.sum())
        for metric, values in (("epe", error), ("bad1", 100 * (error > 1)),
                               ("bad3", 100 * (error > 3)), ("bad5", 100 * (error > 5))):
            result[name + "_" + metric] = float(values[mask].mean()) if mask.any() else None
    pred, gt = prediction[valid].astype(float), truth[valid].astype(float)
    pred_std, gt_std = pred.std(), gt.std()
    # Physical correspondence support differs from the padded inspired head support.
    constrained_oracle = -np.clip(-truth, 0, x)
    oracle_error = np.abs(constrained_oracle - truth)
    padded_bound = padded_head_disparity_bound(truth.shape[1])[None]
    padded_oracle_error = np.abs(-np.clip(-truth, 0, padded_bound) - truth)
    result.update(prediction_std=float(pred_std), truth_std=float(gt_std),
        variance_ratio=float(pred_std ** 2 / gt_std ** 2) if gt_std else None,
        spatial_correlation=float(np.corrcoef(pred, gt)[0, 1]) if pred_std and gt_std else None,
        centered_spatial_mae=float(np.abs((pred - pred.mean()) - (gt - gt.mean())).mean()),
        mean_bias=float((pred - gt).mean()), valid_fraction=float(valid.mean()),
        out_of_view_valid_fraction=float(out_of_view[valid].mean()),
        physical_in_view_oracle_floor_epe=float(oracle_error[valid].mean()),
        inspired_padded_head_oracle_floor_epe=float(padded_oracle_error[valid].mean()))
    return result


def scene_summary(rows):
    scenes = sorted(set(row["scene"] for row in rows))
    metrics = [key for key in rows[0] if key not in ("scene", "frame", "method")]
    result = {}
    for scene in scenes:
        members = [row for row in rows if row["scene"] == scene]
        result[scene] = {key: float(np.mean([row[key] for row in members if row[key] is not None]))
                         if any(row[key] is not None for row in members) else None for key in metrics}
    return result


def paired_bootstrap(reference, candidate, samples=10000, seed=1234):
    """Scene-resampled descriptive interval; frames within scene are not IID units."""
    scenes = sorted(reference)
    if scenes != sorted(candidate):
        raise ValueError("paired cohorts differ")
    differences = np.array([candidate[s]["all_epe"] - reference[s]["all_epe"] for s in scenes])
    rng = np.random.default_rng(seed)
    means = differences[rng.integers(0, len(scenes), (samples, len(scenes)))].mean(1)
    return {"candidate_minus_reference_scene_epe": float(differences.mean()),
            "descriptive_scene_bootstrap_95_interval": np.quantile(means, [.025, .975]).tolist() if len(scenes) >= 2 else None,
            "scenes": len(scenes), "resamples": samples, "seed": seed,
            "scope": "conditional on this trained checkpoint and selected frames; no training-seed uncertainty",
            "limitation": "one scene cannot estimate between-scene uncertainty" if len(scenes) < 2 else None}


def scene_epe_distribution(by_scene):
    reference = by_scene.get("pixel_prior")
    result = {}
    for name, scenes in by_scene.items():
        values = np.array([row["all_epe"] for row in scenes.values()])
        differences = np.array([row["all_epe"] - reference[scene]["all_epe"]
                                for scene, row in scenes.items()]) if reference else None
        ranked = sorted(scenes, key=lambda scene: scenes[scene]["all_epe"], reverse=True)
        result[name] = dict(scenes=len(scenes), median_scene_epe=float(np.median(values)),
            percentile90_scene_epe=float(np.quantile(values, .9)),
            scenes_better_than_pixel_prior=int((differences < 0).sum()) if reference else None,
            scenes_worse_than_pixel_prior=int((differences > 0).sum()) if reference else None,
            scenes_tied_with_pixel_prior=int((differences == 0).sum()) if reference else None,
            highest_error_scenes=[dict(scene=scene, mean_scene_epe=scenes[scene]["all_epe"])
                                  for scene in ranked[:5]],
            scope="descriptive scene EPE distribution; percentile uses linear interpolation; highest-error cases are post-hoc diagnostics and do not replace prespecified panels")
    return result


def save_panel(key, maps, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(maps)
    first = maps[names[0]]
    valid = first["valid"].astype(bool)
    truth = -first["truth"]
    vmax = max(float(np.quantile(truth[valid], .99)), 1e-3)
    error_max = max(float(np.quantile(np.concatenate([
        np.abs(value["prediction"] - value["truth"])[valid] for value in maps.values()]), .99)), 1e-3)
    fig, axes = plt.subplots(2, len(names) + 2, figsize=(4 * (len(names) + 2), 5.8), squeeze=False)
    cmap = plt.get_cmap("viridis").copy(); cmap.set_bad("#cccccc")
    ecmap = plt.get_cmap("magma").copy(); ecmap.set_bad("#cccccc")
    axes[0, 0].imshow(first["input_rate"], cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("Input firing rate (all 50 bins)")
    gt_artist = axes[0, 1].imshow(np.ma.masked_where(~valid, truth), cmap=cmap, vmin=0, vmax=vmax)
    axes[0, 1].set_title("Ground truth disparity (px)")
    axes[1, 0].imshow(np.ma.masked_where(~valid, gradient_magnitude(first["input_rate"], valid) <= .01), cmap="gray", vmin=0, vmax=1)
    axes[1, 0].set_title("Low input-gradient mask\n(rate gradient <=0.01)")
    axes[1, 1].imshow(np.ma.masked_where(~valid, gradient_magnitude(truth, valid) >= 1), cmap="gray", vmin=0, vmax=1)
    axes[1, 1].set_title("GT edge mask\n(gradient >=1 px/pixel)")
    for column, name in enumerate(names, 2):
        value = maps[name]
        axes[0, column].imshow(np.ma.masked_where(~valid, -value["prediction"]), cmap=cmap, vmin=0, vmax=vmax)
        axes[0, column].set_title(name + " disparity (px)")
        error = np.abs(value["prediction"] - value["truth"])
        error_artist = axes[1, column].imshow(np.ma.masked_where(~valid, error), cmap=ecmap, vmin=0, vmax=error_max)
        axes[1, column].set_title(f"{name} absolute error\nEPE {error[valid].mean():.2f} px")
    for axis in axes.flat:
        axis.set_axis_off()
        axis.title.set_fontsize(10)
    fig.colorbar(gt_artist, ax=axes[0].tolist(), shrink=.65, label="Disparity px (shared scale)")
    fig.colorbar(error_artist, ax=axes[1].tolist(), shrink=.65, label="Absolute error px (shared scale)")
    fig.suptitle(f"{key}; gray = invalid GT; color limits clip at common 99th percentiles")
    fig.savefig(output, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="METHOD=DIRECTORY")
    parser.add_argument("--priors", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", default="original")
    parser.add_argument("--cohort", required=True, choices=("development", "confirmation", "overfit"))
    parser.add_argument("--selected", nargs="*", help="Preselected scene-frame stems")
    parser.add_argument("--plot-methods", nargs="+",
                        help="Subset shown in panels; all loaded methods/priors remain in metrics")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    runs = dict(item.split("=", 1) for item in args.run)
    files = {name: {p.stem: p for p in (Path(path) / "predictions").glob("*.npz")}
             for name, path in runs.items()}
    keys = sorted(next(iter(files.values())))
    if not keys or any(sorted(value) != keys for value in files.values()):
        raise ValueError("prediction cohorts must be nonempty and identical across methods")
    args.output.mkdir(parents=True, exist_ok=True)
    priors = dict(np.load(args.priors)) if args.priors else None
    rows = []
    def load_case(key):
        maps = {name: dict(np.load(values[key])) for name, values in files.items()}
        first = next(iter(maps.values()))
        for value in maps.values():
            if any(not np.array_equal(value[field], first[field], equal_nan=True)
                   for field in ("truth", "valid", "input_rate")):
                raise ValueError("truth/mask/input mismatch across models: " + key)
        if priors:
            for name, prediction in (("constant_prior", np.full_like(first["truth"], float(priors["scalar"]))),
                                     ("pixel_prior", priors["pixel"])):
                maps[name] = dict(first, prediction=prediction)
        return maps
    for key in keys:
        maps = load_case(key)
        scene, frame = key.rsplit("-", 1)
        for name, value in maps.items():
            rows.append(dict(scene=scene, frame=frame, method=name,
                **image_metrics(value["prediction"], value["truth"], value["valid"], value["input_rate"])))
    methods = sorted(set(row["method"] for row in rows))
    if args.plot_methods and any(name not in methods for name in args.plot_methods):
        raise ValueError("plot methods must be among loaded methods/priors")
    by_scene = {name: scene_summary([row for row in rows if row["method"] == name]) for name in methods}
    summary = {name: {metric: float(np.mean([s[metric] for s in scenes.values() if s[metric] is not None]))
                     if any(s[metric] is not None for s in scenes.values()) else None
                     for metric in next(iter(scenes.values()))} for name, scenes in by_scene.items()}
    comparisons = {name: paired_bootstrap(by_scene[args.reference], value)
                   for name, value in by_scene.items() if name != args.reference}
    pooled = {}
    for name in methods:
        members = [row for row in rows if row["method"] == name]
        pooled[name] = {}
        for regime in ("all", "edge", "flat", "low_texture", "in_view", "out_of_view"):
            count = sum(row[regime + "_pixels"] for row in members)
            pooled[name][regime + "_pixels"] = count
            for metric in ("epe", "bad1", "bad3", "bad5"):
                field = regime + "_" + metric
                pooled[name][field] = sum(row[field] * row[regime + "_pixels"]
                    for row in members if row[field] is not None) / count if count else None
    if args.selected is not None:
        selected = args.selected
    else:
        # One first frame in first/middle/last scene, independent of prediction quality.
        scenes = sorted({key.rsplit("-", 1)[0] for key in keys})
        selected_scenes = list(dict.fromkeys((scenes[0], scenes[len(scenes) // 2], scenes[-1])))
        selected = [next(key for key in keys if key.rsplit("-", 1)[0] == scene) for scene in selected_scenes]
    selection = {"preselected": selected, "development_error_ranked": []}
    if "spiking" in methods and args.selected is None and args.cohort == "development":
        ranked = sorted((row for row in rows if row["method"] == "spiking"), key=lambda r: r["all_epe"])
        selection["development_error_ranked"] = [r["scene"] + "-" + r["frame"]
            for r in (ranked[0], ranked[len(ranked) // 2], ranked[-1])]
    if not args.no_plots:
        for key in dict.fromkeys(selected + selection["development_error_ranked"]):
            if key not in keys:
                raise ValueError("selected case absent: " + key)
            case = load_case(key)
            if args.plot_methods:
                case = {name: case[name] for name in args.plot_methods}
            save_panel(key, case, args.output / (key + ".png"))
    with (args.output / "per-frame.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    result = dict(cohort=args.cohort, mean_scene_metrics=summary, pooled_pixel_metrics=pooled,
                  per_scene=by_scene, paired_comparisons=comparisons,
                  scene_epe_distribution=scene_epe_distribution(by_scene),
                  figure_selection=selection, plotted_methods=args.plot_methods or methods,
                  panel_scaling="GT display common across models per case; error limits common across shown methods only (99th percentile); excluded methods remain in metric tables",
                  regimes={"edge": "GT forward gradient >=1 px/pixel",
                  "flat": "GT forward gradient <1 px/pixel", "low_texture": "50-bin input-rate forward gradient <=0.01",
                  "out_of_view": "valid GT maps x+signed_horizontal_flow outside [0,width-1]; geometric bound, not full occlusion labels",
                  "in_view": "valid complement of geometric out-of-view"},
                  oracle_floors={"physical_in_view": "GT positive disparity projected onto [0,x]; physical correspondence support diagnostic, not exact padded-head architecture bound",
                  "inspired_padded_head": "GT positive disparity projected onto [0,4*interpolated coarse x], accounting for symmetric padding to multiple32, quarter-grid align_corners=False upsampling and unpadding; support lower bound, not achieved baseline performance"},
                  conventions="model flow kept for metrics; displayed disparity is negative flow; badK means absolute error >K px")
    (args.output / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
