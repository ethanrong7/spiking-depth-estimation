"""Render saved S3 logs only; no model/checkpoint/data imports or monitoring.

Standard-library editable SVG follows the CCFA deterministic plot recipe's
canvas/traceability approach. No runtime dependency on a local skill package.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
from html import escape
import json
import math
from pathlib import Path
import statistics
import tempfile
import xml.etree.ElementTree as ET

GRID = (0, 1000, 2500, 5000)
CYCLE = 480
SERIES = ("S3 hard", "S3 continuous", "I engineering comparator")
COLORS = ("#0072B2", "#D55E00", "#666666")
REPO = next(p for p in Path(__file__).resolve().parents if (p/"projects/snn-depth-estimation").is_dir())


def path_label(path):
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return "local:" + path.name


def read_saved(path, snapshots, jsonl=False):
    if not path.is_file():
        return [] if jsonl else None
    raw = path.read_bytes()
    entry = {"path": path_label(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    snapshots.append(entry)
    if not jsonl:
        return json.loads(raw)
    lines = raw.splitlines(keepends=True)
    records = []
    for index, line in enumerate(lines):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if index == len(lines)-1 and not line.endswith(b"\n"):
                entry["ignored_incomplete_final_line"] = True
            else:
                raise ValueError(f"Malformed completed JSONL record: {path}:{index+1}")
    return records


def finite(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("Metric must be finite and nonnegative")
    return value


def validation(run, label, snapshots):
    points = {}
    initial = read_saved(run / "initial.json", snapshots)
    records = [(0, initial, run / "initial.json")] if initial else []
    history = read_saved(run / "history.json", snapshots) or []
    records += [(row["update"], row, run / "history.json") for row in history]
    evaluations = read_saved(run / "evaluations.jsonl", snapshots, jsonl=True)
    records += [(row["update"], row, run / "evaluations.jsonl") for row in evaluations]
    for update, row, source in records:
        if update not in GRID:
            raise ValueError(f"Unexpected validation update {update}: {source}")
        value = finite(row["mean_scene_epe"])
        if update in points and points[update]["value"] != value:
            raise ValueError(f"Conflicting saved validation values: {label} update{update}")
        points.setdefault(update, {"series": label, "update": update, "value": value,
                                  "unit": "native pixels", "status": "final primary" if update == 5000 else "observed intermediate" if update else "observed initial",
                                  "source": path_label(source)})
    return list(sorted(points.values(), key=lambda p: p["update"]))


def training(run, label, snapshots):
    metadata = read_saved(run / "metadata.json", snapshots)
    rows = read_saved(run / "train.jsonl", snapshots, jsonl=True)
    if (metadata or rows) and (metadata or {}).get("config", {}).get("loss") != "final_masked_smooth_l1_beta1":
        raise ValueError(f"Not the S3 final SmoothL1 objective: {run}")
    updates = [row["update"] for row in rows]
    if updates != list(range(1, len(rows)+1)) or len(rows) > 5000:
        raise ValueError(f"Training logs must contain contiguous unique updates1..N<=5000: {run}")
    summaries = []
    for start in range(0, len(rows), CYCLE):
        block = rows[start:start+CYCLE]
        end = start + len(block)
        summaries.append({"series": label, "cycle": start//CYCLE+1, "start_update": start+1,
                          "end_update": end, "n_updates": len(block), "value": statistics.mean(finite(row["loss"]) for row in block),
                          "unit": "final SmoothL1 beta1", "status": "complete" if len(block) == CYCLE else "partial",
                          "source": path_label(run / "train.jsonl")})
    return summaries, len(rows)


def svg_chart(series, title, subtitle, ylabel, logarithmic=False):
    width, height = 1000, 490
    x0, x1, y0, y1 = 90, 955, 112, 387
    values = [p[1] for points in series for p in points]
    use_log = bool(values) and logarithmic and min(values) > 0
    if use_log:
        lo = 10**math.floor(math.log10(min(values)))
        hi = 10**math.ceil(math.log10(max(values)))
        if lo == hi:
            lo, hi = lo/10, hi*10
        ticks = [m*10**e for e in range(math.floor(math.log10(lo)), math.ceil(math.log10(hi))+1)
                 for m in (1, 2, 5) if lo <= m*10**e <= hi]
        convert = math.log10
    else:
        lo, hi = 0, max(values, default=1)*1.12 or 1
        ticks = [hi*i/4 for i in range(5)]
        convert = lambda v: v
    x = lambda update: x0 + update/5000*(x1-x0)
    y = lambda value: y1-(convert(value)-convert(lo))/(convert(hi)-convert(lo))*(y1-y0)
    items = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
             f'<title>{escape(title)}</title>', '<rect width="100%" height="100%" fill="white"/>',
             '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#222} .muted{fill:#555}</style>']
    def text(tx, ty, value, size=13, anchor="start", extra=""):
        items.append(f'<text x="{tx:.2f}" y="{ty:.2f}" font-size="{size}" text-anchor="{anchor}" {extra}>{escape(str(value))}</text>')
    text(30, 30, title, 20, extra='font-weight="bold"')
    text(30, 54, subtitle, 13)
    text(x0, 94, ylabel + (" · logarithmic axis" if use_log else " · linear axis"), 13)
    for tick in ticks:
        yy = y(tick)
        items.append(f'<line x1="{x0}" y1="{yy}" x2="{x1}" y2="{yy}" stroke="#e1e1e1"/>')
        text(x0-12, yy+4, f"{tick:.3g}", anchor="end")
    for tick in GRID:
        xx = x(tick)
        items.append(f'<line x1="{xx}" y1="{y0}" x2="{xx}" y2="{y1}" stroke="#eeeeee"/>')
        text(xx, y1+24, f"{tick:,}", anchor="middle")
    items.append(f'<path d="M{x0},{y0} V{y1} H{x1}" fill="none" stroke="#333"/>')
    text((x0+x1)/2, y1+48, "Optimizer updates", anchor="middle")
    for index, points in enumerate(series):
        color = COLORS[index]
        dashed = 'stroke-dasharray="7 5"' if index == 2 else ""
        coords = " ".join(f"{x(u):.3f},{y(v):.3f}" for u, v, _ in points)
        if len(points) >= 2:
            items.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2" {dashed}/>')
        for update, value, status in points:
            xx, yy = x(update), y(value)
            fill = "white" if status == "partial" else color
            if index == 1:
                shape = f'<rect x="{xx-4}" y="{yy-4}" width="8" height="8" fill="{fill}" stroke="{color}" stroke-width="2"/>'
            else:
                shape = f'<circle cx="{xx}" cy="{yy}" r="{6 if index == 0 else 4.5}" fill="{fill}" stroke="{color}" stroke-width="2"/>'
            items.append(f'<g><title>{escape(SERIES[index])}: update{update}, {value:.8g}, {status}</title>{shape}</g>')
            if status == "partial":
                # Opposite sides keep the two nearby latest-cycle labels apart.
                text(xx-9 if index == 0 else xx+9, yy-10, "partial", 12, anchor="end" if index == 0 else "start")
        lx = (90, 300, 560)[index]
        items.append(f'<line x1="{lx}" y1="{height-25}" x2="{lx+22}" y2="{height-25}" stroke="{color}" stroke-width="2" {dashed}/>')
        text(lx+29, height-21, SERIES[index], 13)
    if not values:
        text((x0+x1)/2, (y0+y1)/2, "No saved observations available", 16, anchor="middle")
    return "\n".join(items + ["</svg>"])


def write_csv(path, rows, columns):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def render(runs, output):
    output.mkdir(parents=True, exist_ok=True)
    snapshots, val, loss, progress = [], [], [], []
    for index, run in enumerate(runs):
        val.extend(validation(run, SERIES[index], snapshots))
        if index < 2:
            blocks, latest = training(run, SERIES[index], snapshots)
            loss.extend(blocks)
            progress.append({"series": SERIES[index], "latest_saved_train_update": latest,
                             "observed_validation_updates": [p["update"] for p in val if p["series"] == SERIES[index]]})
    chart_validation = svg_chart([[(p["update"], p["value"], p["status"]) for p in val if p["series"] == label] for label in SERIES],
                                 "S3 validation learning curves", "Observed Dev points only · I has a different graph and loss · final update5000 is primary",
                                 "Mean-scene EPE (native pixels; lower is better)", logarithmic=True)
    chart_loss = svg_chart([[(p["end_update"], p["value"], p["status"]) for p in loss if p["series"] == label] for label in SERIES[:2]],
                          "S3 final prediction training loss", "480-update cycle means · hollow markers label the latest incomplete cycle · no I training loss",
                          "Masked final SmoothL1 (beta1; mean objective value)")
    (output / "validation.svg").write_text(chart_validation)
    (output / "training-loss.svg").write_text(chart_loss)
    write_csv(output / "validation.csv", val, ["series", "update", "value", "unit", "status", "source"])
    write_csv(output / "training-cycle-loss.csv", loss, ["series", "cycle", "start_update", "end_update", "n_updates", "value", "unit", "status", "source"])
    provenance = {"generated_utc": datetime.now(timezone.utc).isoformat(), "source_snapshots": snapshots,
                  "run_directories": [path_label(p) for p in runs], "progress": progress,
                  "validation_grid": GRID, "training_cycle_updates": CYCLE,
                  "primary": "final5000-update validation; best-selected metrics are not read",
                  "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "engineering_comparator": "audited S1 raw original_I; different graph and sequence training loss",
                  "interpretation": "descriptive saved-log progress; no motion or SNN energy claim"}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    headers = "<tr><th>Series</th>" + "".join(f"<th>{u:,}</th>" for u in GRID) + "</tr>"
    table = ""
    for label in SERIES:
        by_update = {p["update"]: p["value"] for p in val if p["series"] == label}
        table += "<tr><td>"+escape(label)+"</td>"+"".join(f"<td>{by_update[u]:.4f}</td>" if u in by_update else "<td>Not observed</td>" for u in GRID)+"</tr>"
    cycles = "".join(f'<tr><td>{escape(p["series"])}</td><td>{p["start_update"]}–{p["end_update"]}</td><td>{p["n_updates"]}/480</td><td>{p["value"]:.6f}</td><td>{p["status"]}</td></tr>' for p in loss)
    status = " · ".join(f'{p["series"]}: saved train update{p["latest_saved_train_update"]:,}' for p in progress)
    html = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>S3 saved-log learning curves</title><style>body{{font:15px/1.5 Arial,sans-serif;color:#222;max-width:1040px;margin:24px auto;padding:0 18px}}svg{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%;margin:15px 0}}td,th{{border-bottom:1px solid #ddd;padding:8px;text-align:right}}td:first-child,th:first-child{{text-align:left}}code{{overflow-wrap:anywhere}}details{{margin:24px 0}}.note{{color:#555;font-size:13px}}a{{color:#00629b}}</style>
<h1>S3 saved-log learning curves</h1><p>{escape(status)}</p><p class="note">Snapshot {escape(provenance["generated_utc"])}. Final5000-update validation is primary once observed. No best-checkpoint selection or future data filling.</p>
{chart_validation}<table>{headers}{table}</table><p class="note">EPE: native pixels, equal scene means on the fixed Dev set. Dashed I is the audited raw original_I engineering comparator; its graph and sequence objective differ from S3. Lines connect observations only.</p>
{chart_loss}<details><summary>Training cycle values and partial-window sizes</summary><table><tr><th>Series</th><th>Updates</th><th>Count</th><th>SmoothL1</th><th>Status</th></tr>{cycles}</table></details>
<p class="note">Loss averages logged final masked SmoothL1(beta1) equally over updates. A complete cycle covers480 updates/960 samples at batch2. The final5000-update run naturally ends with a partial cycle. I training loss is excluded because it uses a different sequence objective.</p>
<p><a href="validation.csv">Validation CSV</a> · <a href="training-cycle-loss.csv">Training CSV</a> · <a href="provenance.json">Source paths and snapshot hashes</a> · <a href="validation.svg">Validation SVG</a> · <a href="training-loss.svg">Loss SVG</a></p>
<details><summary>Saved-log sources</summary><ul>{''.join('<li><code>'+escape(p['path'])+'</code> · '+p['sha256'][:16]+'…</li>' for p in snapshots)}</ul></details>
<p class="note">Descriptive learning progress only. These curves establish no motion mechanism, sensor advantage or SNN energy savings. Renderer reads saved local logs only and does not access datasets, checkpoints or models.</p></html>'''
    (output / "index.html").write_text(html)
    return provenance


def self_test():
    # Fabricated logs live only in this temporary software-check directory.
    with tempfile.TemporaryDirectory(prefix="stage3-curves-SOFTWARE-CHECK-") as directory:
        base = Path(directory)
        runs = [base/str(i) for i in range(3)]
        for run in runs:
            run.mkdir()
            (run/"initial.json").write_text(json.dumps({"mean_scene_epe": 80}))
            (run/"history.json").write_text(json.dumps([{"update": 1000, "mean_scene_epe": 7}]))
        for run in runs[:2]:
            (run/"metadata.json").write_text(json.dumps({"config": {"loss": "final_masked_smooth_l1_beta1"}}))
            (run/"train.jsonl").write_text("".join(json.dumps({"update": i, "loss": float(i)})+"\n" for i in range(1, 482))+'{"update":')
        result = render(runs, base/"artifacts")
        for name in ("validation.svg", "training-loss.svg"):
            ET.parse(base/"artifacts"/name)
        with (base/"artifacts/training-cycle-loss.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        assert [r["status"] for r in rows] == ["complete", "partial"]*2
        assert float(rows[0]["value"]) == 240.5 and float(rows[1]["value"]) == 481
        assert all(p["latest_saved_train_update"] == 481 for p in result["progress"])
        assert len(list(csv.DictReader((base/"artifacts/validation.csv").open()))) == 6
        # Fail on missing updates, rather than call a sparse cycle complete.
        (runs[0]/"train.jsonl").write_text('{"update":2,"loss":1}\n')
        try:
            training(runs[0], SERIES[0], [])
        except ValueError:
            pass
        else:
            raise AssertionError("Noncontiguous updates accepted")
    print("PASS: temporary fabricated logs verified parser/means/partial labels/SVG/CSV; no example data retained")


def main():
    campaign = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=REPO/"storage/experiment-runs/snn-depth-estimation/spiking-transformer-stereo/spike-depth-direction-v1")
    parser.add_argument("--hard-run", type=Path)
    parser.add_argument("--continuous-run", type=Path)
    parser.add_argument("--i-run", type=Path)
    parser.add_argument("--output", type=Path, default=campaign/"results/stage3-curves")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    runs = [args.hard_run or args.raw_root/"s3-hard-raw-seed1234-5k-v1",
            args.continuous_run or args.raw_root/"s3-continuous-raw-seed1234-5k-v1",
            args.i_run or args.raw_root/"s1-raw-seed1234-5k-v1"]
    result = render(runs, args.output)
    print(json.dumps({"output": str(args.output), "progress": result["progress"]}))


if __name__ == "__main__":
    main()
