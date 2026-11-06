"""Plot saved development histories only; no model, dataset or cluster access.

Matplotlib adaptation of the CCFA small_multiple_lines recipe: aligned seed
panels, exact connected evaluation points, stable accessible encodings.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


EXPERIMENT = Path(__file__).resolve().parents[1]
STYLES = {
    "original": ("Original ALIF", "#666666", "o", "-"),
    "spiking": ("Normalized spiking", "#0072B2", "D", "-"),
    "analog": ("Normalized analog", "#D55E00", "s", "--"),
    "center": ("Normalized center-only", "#009E73", "^", "-."),
    "original_regularized": ("Original + neuron regularization", "#222222", "v", "--"),
    "ssde_inspired": ("SSDEFormer-inspired ANN", "#CC79A7", "P", "-."),
}
MAIN = ("original", "spiking", "analog", "center")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_curves(queue, runs_root):
    curves, sources, rows = {}, [], []
    for item in queue["runs"]:
        key = (item["seed"], item["model"])
        final = runs_root / item["output"]
        source = final
        status = "final history available"
        if not (source / "history.json").exists():
            parent = item.get("parent_run")
            if parent and (runs_root / parent / "history.json").exists():
                source = runs_root / parent
                status = "pilot history only; final pending"
            else:
                sources.append(dict(seed=key[0], method=key[1], requested_run=item["output"],
                                    status="pending; no saved history"))
                continue
        metadata = json.loads((source / "metadata.json").read_text())
        config = metadata["config"]
        if (config["model"], config.get("seed", 1234), config.get("normalize_residual_current", False)) != (
                item["model"], item["seed"], item["normalize_residual_current"]):
            raise ValueError("run identity does not match queue: " + source.name)
        if config.get("eval_split", "dev") != "dev":
            raise ValueError("learning curves must use development evaluations")
        history_path = source / "history.json"
        history = json.loads(history_path.read_text())
        points = {}
        for record in history:
            update, value = int(record["update"]), float(record["mean_scene_epe"])
            if update in points and points[update] != value:
                raise ValueError("conflicting evaluations at one update: " + source.name)
            points[update] = value
        selected = None
        result_path = final / "result.json"
        if source == final and result_path.exists():
            selected = int(json.loads(result_path.read_text())["best_update"])
            if selected not in points:
                raise ValueError("selected checkpoint missing from recorded history: " + final.name)
            status = "completed final run"
        elif source == final:
            status = "partial final history; run pending"
        digest = sha256(history_path)
        curves[key] = dict(points=sorted(points.items()), selected=selected, status=status)
        sources.append(dict(seed=key[0], method=key[1], requested_run=item["output"],
                            source_run=source.name, status=status, history_sha256=digest,
                            result_sha256=sha256(result_path) if result_path.exists() else None,
                            selected_update=selected, last_observed_update=max(points),
                            starting_updates=item["starting_updates"],
                            source_identity_sha256=metadata["source"]["source_identity_sha256"]))
        for update, value in sorted(points.items()):
            rows.append(dict(method=item["model"], seed=item["seed"], optimizer_update=update,
                development_mean_scene_epe=value, selected_checkpoint=update == selected,
                source_run=source.name, history_sha256=digest, source_status=status))
    return curves, sources, rows


def draw_curve(axis, key, curve):
    label, color, marker, linestyle = STYLES[key[1]]
    x, y = zip(*curve["points"])
    axis.plot(x, y, color=color, marker=marker, markersize=4.2, linewidth=1.7,
              linestyle=linestyle, label=label, zorder=3)
    if curve["selected"] is not None:
        value = dict(curve["points"])[curve["selected"]]
        axis.scatter([curve["selected"]], [value], marker="*", s=145,
                     facecolors="white", edgecolors=color, linewidths=1.4, zorder=5)


def configure_axis(axis, title, limit, note):
    axis.set(title=title, xlabel="Actual optimizer updates", xlim=(0, 3100), ylim=(0, limit))
    axis.set_xticks((0, 1000, 2000, 3000))
    axis.grid(axis="y", color="#DDDDDD", linewidth=.65, zorder=0)
    axis.spines[["top", "right"]].set_visible(False)
    axis.axvline(1000, color="#888888", linestyle=":", linewidth=1.2, zorder=1)
    axis.text(1000, limit * .97, note, ha="center", va="top", fontsize=8,
              bbox=dict(facecolor="white", edgecolor="none", alpha=.85, pad=2))


def legend_handle(method):
    label, color, marker, linestyle = STYLES[method]
    return Line2D([], [], label=label, color=color, marker=marker, markersize=4.5,
                  linewidth=1.7, linestyle=linestyle)


def render_main(curves, output):
    values = [value for (seed, method), curve in curves.items() if method in MAIN
              for _, value in curve["points"]]
    limit = max(values, default=10) * 1.17
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8), sharey=True)
    for i, seed in enumerate((1234, 2345)):
        axis = axes[i]
        note = "Pilot / continuation" if seed == 1234 else "Pilot-budget reference"
        configure_axis(axis, f"({chr(97+i)}) Seed {seed}", limit, note)
        pending = []
        for method in MAIN:
            curve = curves.get((seed, method))
            if curve:
                draw_curve(axis, (seed, method), curve)
            if not curve or curve["status"] != "completed final run":
                pending.append(STYLES[method][0])
        if pending:
            message = textwrap.fill("Final results pending: " + ", ".join(pending), 43)
            axis.text(.98, .83, message, transform=axis.transAxes, ha="right", va="top", fontsize=9,
                      color="#555555", bbox=dict(facecolor="white", edgecolor="#DDDDDD", pad=5))
    axes[0].set_ylabel("Development mean-scene EPE (px) ↓")
    handles = [legend_handle(method) for method in MAIN]
    handles.append(Line2D([], [], label="Selected checkpoint", marker="*", markersize=11,
                           markerfacecolor="white", markeredgecolor="#222222", linestyle="none"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, 1.0), ncol=3,
               frameon=False, fontsize=9)
    fig.text(.5, .045, "Scheduled development evaluations; initial random-model evaluations omitted. No smoothing or extrapolation.",
             ha="center", fontsize=8.5, color="#444444")
    fig.text(.5, .006, "Seed 1234 continues after 1,000. Seed 2345 normalized runs start fresh; its original continues the pilot.",
             ha="center", fontsize=8, color="#444444")
    fig.subplots_adjust(top=.77, bottom=.20, left=.08, right=.985, wspace=.14)
    for extension in ("png", "svg"):
        fig.savefig(output / f"learning-curves-main.{extension}", dpi=180)
    plt.close(fig)


def render_supplement(curves, output):
    methods = ("original", "original_regularized", "ssde_inspired")
    values = [value for method in methods if (1234, method) in curves
              for _, value in curves[(1234, method)]["points"]]
    fig, axis = plt.subplots(figsize=(8.0, 4.5))
    configure_axis(axis, "Seed 1234: supplementary objective/control curves", max(values, default=10) * 1.16,
                   "Pilot / continuation")
    pending = []
    for method in methods:
        curve = curves.get((1234, method))
        if curve:
            draw_curve(axis, (1234, method), curve)
        if not curve or curve["status"] != "completed final run":
            pending.append(STYLES[method][0])
    if pending:
        axis.text(.98, .92, textwrap.fill("Final results pending: " + ", ".join(pending), 45),
                  transform=axis.transAxes, ha="right", va="top", fontsize=8.5,
                  bbox=dict(facecolor="white", edgecolor="#DDDDDD", pad=4))
    axis.set_ylabel("Development mean-scene EPE (px) ↓")
    axis.legend(handles=[legend_handle(method) for method in methods], loc="upper right", fontsize=8.2,
                bbox_to_anchor=(1, .72), frameon=True, framealpha=.95)
    fig.text(.5, .035, "Supplementary scale; inspired ANN uses final smooth L1 and is not a faithful SSDEFormer reproduction.",
             ha="center", fontsize=7.8)
    fig.subplots_adjust(top=.87, bottom=.20, left=.10, right=.98)
    for extension in ("png", "svg"):
        fig.savefig(output / f"learning-curves-supplement.{extension}", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--queue", type=Path, default=EXPERIMENT / "configs/final-run-queue.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    queue = json.loads(args.queue.read_text())
    curves, sources, rows = load_curves(queue, args.runs_root)
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "none", "axes.titleweight": "semibold"})
    render_main(curves, args.output)
    render_supplement(curves, args.output)
    with (args.output / "learning-curve-source.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    manifest = dict(queue_sha256=sha256(args.queue), expected_runs=len(queue["runs"]),
        completed_runs=sum(source["status"] == "completed final run" for source in sources),
        source_histories=sources, metric="recorded development mean-scene EPE in pixels",
        selection="completed final run historical best update, from result.json; no selected mark for pending runs",
        connecting_lines="visual guides between actual evaluation points; no smoothing/extrapolation",
        excluded="initial.json evaluations are outside the requested history.json source; no confirmation data",
        recipe="Matplotlib adaptation of CCFA small_multiple_lines; stable Okabe-Ito roles and independent markers/line styles")
    (args.output / "learning-curve-sources.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
