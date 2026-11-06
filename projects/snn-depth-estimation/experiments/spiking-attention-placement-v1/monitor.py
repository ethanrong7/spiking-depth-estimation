"""Synchronize compact placement records and evaluate completed registered endpoints."""
import argparse
import json
import subprocess
from pathlib import Path

from evaluate_gate import compare

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[3] / "storage/experiment-runs/snn-depth-estimation/spiking-transformer-stereo"
CLUSTER = "cccl-5090b"
REMOTE = "/data/rongliang/experiment-runs/snn-depth-estimation/spiking-transformer-stereo/" + HERE.name


def sync():
    raw = BASE / HERE.name
    raw.mkdir(parents=True, exist_ok=True)
    args = ["rsync", "-az", "--prune-empty-dirs", "--include=*/",
            "--include=*.json", "--include=*.jsonl", "--exclude=*",
            f"{CLUSTER}:{REMOTE}/", str(raw) + "/"]
    subprocess.run(args, check=True)
    return raw


def refresh(raw):
    execution = json.loads((HERE / 'results/execution.json').read_text())
    guard_file = execution.get('queue_guard_file')
    if guard_file == 'cohost-queue-guard.json' and execution.get('active_job'):
        path = raw / guard_file
        if path.exists():
            guard = json.loads(path.read_text())
            if str(guard.get('job')) == str(execution['active_job']):
                guard['view_source'] = guard_file
                guard['historical_yield_preserved_in'] = 'bc-yield.request'
                (raw / 'bc-queue-guard.json').write_text(json.dumps(guard, indent=2) + '\n')
    baseline = BASE / "shared-gpu-30k-v1/train"
    for arm in ("A", "B", "C"):
        records = raw / "train" / arm / "evaluations.jsonl"
        if not records.exists():
            continue
        with records.open() as stream:
            steps = {json.loads(s)["update"] for s in stream if s.strip() and s.endswith("\n")}
        for update in sorted(steps & {7500, 30000}):
            receipt = compare(records, baseline / "original/evaluations.jsonl",
                              baseline / "s2m2_v2b/evaluations.jsonl", update)
            target = HERE / "results" / f"{arm}-gate-{update}.json"
            target.parent.mkdir(exist_ok=True)
            pending = target.with_suffix(".tmp")
            pending.write_text(json.dumps(receipt, allow_nan=False, indent=2) + "\n")
            pending.replace(target)
            print(arm, update, receipt.get("decision", "endpoint"), receipt["mean_scene_epe"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-only", action="store_true")
    args = parser.parse_args()
    refresh(BASE / HERE.name if args.local_only else sync())
