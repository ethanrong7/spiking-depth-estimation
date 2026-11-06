"""Slurm-only selected-checkpoint refinement and attention-intervention probe."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import run_experiment as runner


def wrong_right_permutation(rows):
    """Fixed next-scene permutation for the declared two-frames-per-scene dev set."""
    if len(rows) != 60 or any(sum(row["scene"] == scene for row in rows) != 2
                              for scene in {row["scene"] for row in rows}):
        raise ValueError("wrong-right probe requires the frozen 60-row, two-frame-per-scene development cohort")
    donors = [(i + 2) % len(rows) for i in range(len(rows))]
    if sorted(donors) != list(range(len(rows))) or any(rows[i]["scene"] == rows[j]["scene"]
                                                     for i, j in enumerate(donors)):
        raise ValueError("donor mapping must preserve right marginal and use another scene")
    return donors


class WrongRightDataset(Dataset):
    """Counterfactual stereo pairs; left, GT, validity and target IDs stay intact."""
    def __init__(self, aligned):
        self.aligned, self.rows = aligned, aligned.rows
        self.donors = wrong_right_permutation(self.rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        left, _, target, valid, target_index = self.aligned[index]
        donor = self.rows[self.donors[index]]
        wrong_right = runner.get_block_spikes(str(self.aligned.root / donor["spike_r"])).astype(np.float32)
        return left, torch.from_numpy(wrong_right), target, valid, target_index


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def combine_diagnostics(records):
    if not records:
        return {}
    return {scale: {key: float(np.mean([record[scale][key] for record in records]))
                    for key in records[0][scale]}
            for scale in records[0]}


def register_branch_hooks(cells):
    """Read-only hooks distinguish the raw projection from the actual residual."""
    branches, handles = {}, []
    for i, cell in enumerate(cells):
        record = branches[f"scale_{i}"] = {name: [] for name in (
            "raw_projection_rms", "normalized_projection_rms", "residual_current_rms",
            "gated_attention_rms", "gated_ffn_rms")}
        def branch_hook(name, record, cell):
            def capture(module, inputs, value):
                scale = 1.0
                if name == "gated_attention_rms":
                    scale = cell.attention_gate.detach().sigmoid() if cell.attention_enabled else 0.0
                elif name == "gated_ffn_rms":
                    scale = cell.ffn_gate.detach().sigmoid()
                rms = float((value.detach() * scale).square().mean().sqrt())
                record[name].append(rms)
                if name == "normalized_projection_rms":
                    actual = rms if cell.normalize_residual_current else record["raw_projection_rms"][-1]
                    record["residual_current_rms"].append(actual)
            return capture
        for name, module in (("raw_projection_rms", cell.input_proj),
                             ("normalized_projection_rms", cell.norm),
                             ("gated_attention_rms", cell.output_proj),
                             ("gated_ffn_rms", cell.ffn)):
            handles.append(module.register_forward_hook(branch_hook(name, record, cell)))
    return branches, handles


@torch.no_grad()
def evaluate_condition(model, loader, iterations, output, branch_diagnostics=False):
    model.eval()
    output.mkdir(parents=True)
    predictions = output / "predictions"
    predictions.mkdir()
    diagnostics, traces, per_frame = [], [], []
    weighted_error, count = 0., 0
    bad_counts = {k: 0 for k in (1, 3, 5)}
    trace = []
    spike_counts, occupancy = {}, []
    cells = getattr(model.update_block, "cells", [])
    collect_occupancy = branch_diagnostics and (not cells or all(cell.mode == "spiking" for cell in cells))
    branches, branch_handles = {}, []
    if branch_diagnostics:
        branches, branch_handles = register_branch_hooks(cells)
    def hook(module, inputs, result):
        if not isinstance(result, tuple) or len(result) != 4:
            return
        net, readout, _, _ = result
        if collect_occupancy:
            for i, value in enumerate(net):
                spikes = value.detach().to(torch.int16)
                spike_counts[i] = spike_counts.get(i, 0) + spikes
        trace.append({**{f"scale_{i}_output_mean": float(value.detach().mean())
                         for i, value in enumerate(net)},
                      "fine_readout_rms": float(readout.detach().square().mean().sqrt())})
    handle = model.update_block.register_forward_hook(hook)
    try:
        for left, right, target, valid, indices in loader:
            left, right, target, valid = [value.cuda() for value in (left, right, target, valid)]
            trace.clear()
            spike_counts.clear()
            pred = runner.forward(model, left, right, iterations)
            if not torch.isfinite(pred).all():
                raise FloatingPointError("nonfinite probe prediction")
            traces.append(list(trace))
            if collect_occupancy:
                occupancy.append({f"scale_{i}": {
                    "always_off_within_horizon_fraction": float((counts == 0).float().mean()),
                    "always_on_within_horizon_fraction": float((counts == iterations).float().mean())}
                    for i, counts in spike_counts.items()})
            if hasattr(model.update_block, "diagnostics"):
                diagnostics.append(model.update_block.diagnostics())
            for b, index in enumerate(indices.tolist()):
                error = (pred[b, 0] - target[b, 0]).abs()[valid[b]]
                if not error.numel():
                    raise ValueError("probe frame has no valid ground truth")
                row = loader.dataset.rows[index]
                per_frame.append(dict(scene=row["scene"], frame=row["frame"],
                    valid_pixels=error.numel(), **runner.metric_values(error)))
                weighted_error += float(error.sum()); count += error.numel()
                for k in bad_counts:
                    bad_counts[k] += int((error > k).sum())
                np.savez_compressed(predictions / f"{row['scene']}-{row['frame']}.npz",
                    prediction=pred[b, 0].cpu().numpy(), truth=target[b, 0].cpu().numpy(),
                    valid=valid[b].cpu().numpy(), input_rate=left[b].mean(0).cpu().numpy(),
                    right_input_rate=right[b].mean(0).cpu().numpy())
    finally:
        handle.remove()
        for branch_handle in branch_handles:
            branch_handle.remove()
        runner.reset_model(model)
    scenes = sorted(set(row["scene"] for row in per_frame))
    scene_epe = {scene: float(np.mean([r["epe"] for r in per_frame if r["scene"] == scene]))
                 for scene in scenes}
    trace_mean = [{key: float(np.mean([batch[t][key] for batch in traces]))
                   for key in traces[0][t]} for t in range(len(traces[0]))]
    result = dict(iterations=iterations, mean_scene_epe=float(np.mean(list(scene_epe.values()))),
        pooled_epe=weighted_error / count, valid_pixels=count,
        **{f"bad{k}": value / count * 100 for k, value in bad_counts.items()},
        scene_epe=scene_epe, per_frame=per_frame,
        diagnostics=combine_diagnostics(diagnostics), mean_refinement_trace=trace_mean)
    result["output_spike_occupancy"] = combine_diagnostics(occupancy)
    if occupancy:
        result["output_spike_occupancy_scope"] = "fraction of sample/channel/pixel units silent or active at every refinement within this evaluation horizon, not permanent neuron status"
    result["branch_rms_diagnostics"] = {}
    for scale, record in branches.items():
        values = {name: np.array(value).reshape(-1, iterations) for name, value in record.items()}
        raw_rms = np.maximum(values["raw_projection_rms"], 1e-12)
        residual_rms = np.maximum(values["residual_current_rms"], 1e-12)
        result["branch_rms_diagnostics"][scale] = dict(
            **{f"mean_{name}": float(value.mean()) for name, value in values.items()},
            mean_attention_to_residual_current_rms_ratio=float((values["gated_attention_rms"] / residual_rms).mean()),
            mean_ffn_to_residual_current_rms_ratio=float((values["gated_ffn_rms"] / residual_rms).mean()),
            mean_attention_to_raw_projection_rms_ratio=float((values["gated_attention_rms"] / raw_rms).mean()),
            mean_ffn_to_raw_projection_rms_ratio=float((values["gated_ffn_rms"] / raw_rms).mean()),
            mean_per_refinement={name: value.mean(0).tolist() for name, value in values.items()},
            scope="read-only hooks; principal ratios use actual residual current (normalized iff enabled); raw-projection ratios are separate; not additive explained variance or causal attribution")
    (output / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    return result


@torch.no_grad()
def latency(model, loader, iterations):
    """Batch latency on actual GPU, separate from accuracy and no speedup claim."""
    model.eval()
    left, right = [value.cuda() for value in next(iter(loader))[:2]]
    for _ in range(2):
        runner.forward(model, left, right, iterations)
    torch.cuda.synchronize()
    seconds = []
    for _ in range(5):
        torch.cuda.synchronize(); start = time.perf_counter()
        runner.forward(model, left, right, iterations)
        torch.cuda.synchronize(); seconds.append(time.perf_counter() - start)
    runner.reset_model(model)
    return dict(batch_size=left.shape[0], input_shape=list(left.shape), iterations=iterations,
        warmup_forwards=2, measured_forwards=5, synchronized_batch_seconds=seconds,
        median_batch_seconds=float(np.median(seconds)),
        scope="small fixed-batch checkpoint diagnostic; not a hardware-energy or robust throughput benchmark")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--source-registry", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--iterations", nargs="+", type=int)
    parser.add_argument("--split", choices=("overfit", "dev"), default="overfit")
    parser.add_argument("--measure-latency", action="store_true")
    parser.add_argument("--branch-diagnostics", action="store_true",
                        help="Read-only per-cell residual-branch RMS hooks; no core source changes")
    parser.add_argument("--wrong-right", action="store_true",
                        help="Add distorted stereo pairs using right donor row (i+2)%%60 from another dev scene")
    parser.add_argument("--correspondence-only", action="store_true",
                        help="Requires --wrong-right; run aligned versus wrong-right only, default32 refinements")
    args = parser.parse_args()
    if args.correspondence_only and not args.wrong_right:
        raise ValueError("--correspondence-only requires --wrong-right")
    if args.wrong_right and args.split != "dev":
        raise ValueError("--wrong-right is defined only on frozen development cohort")
    if args.iterations is None:
        args.iterations = [32] if args.correspondence_only else [8, 16, 24, 32]
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("probe requires coordinated Slurm CUDA allocation")
    if torch.cuda.device_count() != 1 or not any(name in torch.cuda.get_device_name(0) for name in ("4090", "5090")):
        raise RuntimeError("probe requires one authorized RTX4090/5090")
    if any(value < 1 for value in args.iterations):
        raise ValueError("refinement counts must be positive")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    manifest = json.loads(args.manifest.read_text())
    rows = manifest[args.split]
    if args.split == "overfit" and len(rows) != 4:
        raise ValueError("this diagnostic is bounded to the fixed four-example overfit cohort")
    dataset = runner.ManifestDataset(args.data_root, rows)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    wrong_dataset = WrongRightDataset(dataset) if args.wrong_right else None
    wrong_loader = DataLoader(wrong_dataset, batch_size=2, shuffle=False, num_workers=0) if wrong_dataset else None
    source = json.loads(args.source_registry.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = dict(command=sys.argv, source=source, manifest_sha256=file_sha256(args.manifest),
        probe_script_sha256=file_sha256(__file__), runner_script_sha256=file_sha256(runner.__file__),
        torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
        python=sys.version, hostname=socket.gethostname(),
        slurm={k: v for k, v in os.environ.items() if k.startswith("SLURM_")},
        split=args.split, cohort=rows, checkpoints={}, conditions={}, latency={}, correspondence_reliance={})
    if wrong_dataset:
        metadata["counterfactual"] = dict(
            scope="read-only distorted stereo pairs, not ordinary validation or retraining; measures sensitivity to wrong-right input, not proof of geometric matching",
            permutation="donor right row=(left row+2)%60; donor scene differs; same right-input marginal distribution",
            unchanged=["left input", "ground truth", "validity mask", "target row ordering", "checkpoint weights"],
            donor_pairs=[dict(left_index=i, donor_index=j, left_scene=rows[i]["scene"], left_frame=rows[i]["frame"],
                              donor_scene=rows[j]["scene"], donor_frame=rows[j]["frame"], donor_spike_r=rows[j]["spike_r"])
                         for i, j in enumerate(wrong_dataset.donors)])
    (args.output / "probe.json").write_text(json.dumps(metadata, indent=2, allow_nan=False))
    for entry in args.checkpoint:
        label, path = entry.split("=", 1)
        if Path(label).name != label or not label:
            raise ValueError("checkpoint label must be a simple directory name")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        if config["model"] == "ssde_inspired":
            raise ValueError("refinement probe targets original and recurrent attention models")
        old_files, current_files = checkpoint["source"]["source_files"], source["source_files"]
        differing_core = [key for key, value in old_files.items()
            if "/core/" in key and current_files.get(key) != value]
        if differing_core:
            raise ValueError("checkpoint/current core source differs: " + str(differing_core))
        source_root = runner.PROJECT.parents[1]
        stale_core = [key for key, value in old_files.items() if "/core/" in key
                      and file_sha256(source_root / key) != value]
        if stale_core:
            raise ValueError("actual core source does not match registry: " + str(stale_core))
        model = runner.build_model(config["model"], config.get("seed", 1234),
                                   config.get("normalize_residual_current", False)).cuda().eval()
        model.load_state_dict(checkpoint["model"], strict=True)
        metadata["checkpoints"][label] = dict(path=str(Path(path).resolve()), sha256=file_sha256(path),
            selected_update=checkpoint["updates"], config=config, training_source=checkpoint["source"])
        conditions = [(f"{label}-iters{count}", count, {}, False) for count in args.iterations]
        if config["model"] == "spiking" and not args.correspondence_only:
            conditions += [(label + "-center-iters32", 32, {"center_only": True}, False),
                           (label + "-disabled-iters32", 32, {"disable_attention": True}, False)]
        if args.wrong_right:
            conditions += [(f"{label}-wrong-right-iters{count}", count, {}, True) for count in args.iterations]
        for name, iterations, intervention, use_wrong_right in conditions:
            original_flags = [(cell.spatial_mixing, cell.attention_enabled)
                              for cell in getattr(model.update_block, "cells", [])]
            context = model.update_block.attention_intervention(**intervention) if intervention else nullcontext()
            with context:
                result = evaluate_condition(model, wrong_loader if use_wrong_right else loader, iterations,
                                            args.output / name, args.branch_diagnostics)
            restored_flags = [(cell.spatial_mixing, cell.attention_enabled)
                              for cell in getattr(model.update_block, "cells", [])]
            if original_flags != restored_flags:
                raise RuntimeError("intervention flags were not restored")
            metadata["conditions"][name] = dict(intervention=intervention, wrong_right=use_wrong_right,
                                                flags_restored=True, **result)
            if use_wrong_right:
                aligned = metadata["conditions"][f"{label}-iters{iterations}"]
                metadata["correspondence_reliance"][name] = dict(
                    **{metric + "_wrong_minus_aligned": result[metric] - aligned[metric]
                       for metric in ("mean_scene_epe", "pooled_epe", "bad1", "bad3", "bad5")},
                    relative_scene_epe_change_percent=100 * (result["mean_scene_epe"] / aligned["mean_scene_epe"] - 1),
                    scope="distorted-stereo sensitivity; bad-threshold changes are percentage points; not ordinary validation performance")
            print(name, result["mean_scene_epe"], flush=True)
            (args.output / "probe.json").write_text(json.dumps(metadata, indent=2, allow_nan=False))
        if args.measure_latency:
            metadata["latency"][label] = {str(count): latency(model, loader, count) for count in (16, 32)}
        del model, checkpoint
        torch.cuda.empty_cache()
    (args.output / "probe.json").write_text(json.dumps(metadata, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
