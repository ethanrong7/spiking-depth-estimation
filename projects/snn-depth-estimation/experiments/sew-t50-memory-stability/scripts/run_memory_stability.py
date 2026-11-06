#!/usr/bin/env python3
"""Run the bounded two-process A3-T50 memory-stability experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import random
import socket
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import numpy as np
import torch
from torch.utils.data import DataLoader, default_collate


PROJECT = Path(__file__).resolve().parents[3]
FEASIBILITY_SCRIPTS = PROJECT / "experiments" / "sew-t50-feasibility" / "scripts"
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "core"))
sys.path.insert(0, str(FEASIBILITY_SCRIPTS))

from exact_batch_loss import ExactBatchLossAccumulator, replay_loss
from run_feasibility import SyntheticTrainDataset, model_args, parameter_report
from core.spikestereonet import SpikeStereoNet
from training_lifecycle import (
    DeterministicRepeatingBatchSampler,
    capture_rng_state,
    cuda_memory_report,
    exact_training_update,
    optimizer_step_report,
    position_logical_counters,
    reset_report,
    restore_rng_state,
    rng_digest,
    state_digest,
)


CSV_FIELDS = (
    "timestamp",
    "event",
    "phase",
    "process_id",
    "actual_update",
    "logical_update",
    "late_counter_synthetic",
    "data_consumed_samples",
    "data_cycle",
    "data_position_in_cycle",
    "loss",
    "learning_rate",
    "seconds",
    "gradients_finite",
    "reset_clean",
    "replay_exact",
    "minimum_probe_delta",
    "cuda_allocated_mib",
    "cuda_reserved_mib",
    "cuda_peak_allocated_mib",
    "cuda_peak_reserved_mib",
    "gpu_capacity_mib",
    "main_rss_mib",
    "main_peak_rss_mib",
    "dataloader_child_rss_sum_mib",
)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(data: Any, path: str | Path) -> None:
    destination = Path(path)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)


def tensor_digest(tensor: torch.Tensor) -> str:
    return state_digest(tensor)


class TelemetryWriter:
    def __init__(self, run_dir: Path) -> None:
        self.jsonl = run_dir / "telemetry.jsonl"
        self.csv = run_dir / "telemetry.csv"

    def write(self, record: Mapping[str, Any]) -> None:
        with self.jsonl.open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        memory = record.get("memory", {})
        host = memory.get("host_memory", {})
        main = host.get("main", {})
        probes = record.get("parameter_probe_deltas", {})
        deltas = [value["max_abs_change"] for value in probes.values()]
        row = {
            "timestamp": record.get("timestamp"),
            "event": record.get("event"),
            "phase": record.get("phase"),
            "process_id": record.get("process_id"),
            "actual_update": record.get("actual_update"),
            "logical_update": record.get("logical_update"),
            "late_counter_synthetic": record.get("late_counter_synthetic"),
            "data_consumed_samples": record.get("data_cursor", {}).get(
                "consumed_samples"
            ),
            "data_cycle": record.get("data_cursor", {}).get("cycle"),
            "data_position_in_cycle": record.get("data_cursor", {}).get(
                "position_in_cycle"
            ),
            "loss": record.get("loss"),
            "learning_rate": record.get("learning_rate"),
            "seconds": record.get("seconds"),
            "gradients_finite": record.get("gradients_finite"),
            "reset_clean": record.get("reset_clean"),
            "replay_exact": record.get("replay_statistics", {}).get("all_exact"),
            "minimum_probe_delta": min(deltas) if deltas else None,
            "cuda_allocated_mib": memory.get("cuda_allocated_mib"),
            "cuda_reserved_mib": memory.get("cuda_reserved_mib"),
            "cuda_peak_allocated_mib": memory.get("cuda_peak_allocated_mib"),
            "cuda_peak_reserved_mib": memory.get("cuda_peak_reserved_mib"),
            "gpu_capacity_mib": memory.get("gpu_capacity_mib"),
            "main_rss_mib": main.get("rss_mib"),
            "main_peak_rss_mib": main.get("peak_rss_mib"),
            "dataloader_child_rss_sum_mib": host.get(
                "dataloader_child_rss_sum_mib"
            ),
        }
        exists = self.csv.exists()
        with self.csv.open("a", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
            if not exists:
                writer.writeheader()
            writer.writerow(row)


class BoundedSteadySummary:
    """Retain bounded scalar windows for updates 6--25 only."""

    def __init__(self) -> None:
        self.count = 0
        self.allocated_sum = 0.0
        self.reserved_sum = 0.0
        self.allocated_min = float("inf")
        self.allocated_max = float("-inf")
        self.reserved_min = float("inf")
        self.reserved_max = float("-inf")
        self.first: list[dict[str, float | int]] = []
        self.last: deque[dict[str, float | int]] = deque(maxlen=5)

    def add(self, record: Mapping[str, Any]) -> None:
        actual = int(record["actual_update"])
        if not 6 <= actual <= 25:
            return
        memory = record["memory"]
        item = {
            "actual_update": actual,
            "cuda_allocated_mib": float(memory["cuda_allocated_mib"]),
            "cuda_reserved_mib": float(memory["cuda_reserved_mib"]),
        }
        self.count += 1
        self.allocated_sum += item["cuda_allocated_mib"]
        self.reserved_sum += item["cuda_reserved_mib"]
        self.allocated_min = min(self.allocated_min, item["cuda_allocated_mib"])
        self.allocated_max = max(self.allocated_max, item["cuda_allocated_mib"])
        self.reserved_min = min(self.reserved_min, item["cuda_reserved_mib"])
        self.reserved_max = max(self.reserved_max, item["cuda_reserved_mib"])
        if len(self.first) < 5:
            self.first.append(item)
        self.last.append(item)

    def report(self) -> dict[str, Any]:
        if self.count != 20:
            raise RuntimeError(f"steady summary expected 20 updates, received {self.count}")
        return {
            "window": {"first_actual_update": 6, "last_actual_update": 25},
            "count": self.count,
            "cuda_allocated_mib": {
                "minimum": self.allocated_min,
                "maximum": self.allocated_max,
                "range": self.allocated_max - self.allocated_min,
                "mean": self.allocated_sum / self.count,
            },
            "cuda_reserved_mib": {
                "minimum": self.reserved_min,
                "maximum": self.reserved_max,
                "range": self.reserved_max - self.reserved_min,
                "mean": self.reserved_sum / self.count,
            },
            "first_five": self.first,
            "last_five": list(self.last),
            "interpretation": "raw bounded telemetry; no post-hoc pass threshold applied",
        }


def validate_config(config: Mapping[str, Any]) -> None:
    fixed = {
        "variant": "A3",
        "batch_size": 2,
        "effective_batch_size": 8,
        "exact_batch_replay": True,
        "actual_updates": 30,
        "initial_process_updates": 25,
        "resumed_regular_updates": 3,
        "late_counter_updates": 2,
        "schedule_num_steps": 320000,
        "input_bins": 50,
        "height": 250,
        "width": 400,
        "precision": "fp32",
        "train_iters": 16,
        "validation_iters": 32,
        "corr_implementation": "reg",
        "context_norm": "instance",
        "sew_checkpoint_chunk_size": 5,
        "sew_spatial_checkpoint": True,
        "seed": 1234,
        "manifest_samples": 16,
        "expected_optimizer_state_entries": 163,
    }
    differences = {
        key: {"required": value, "received": config.get(key)}
        for key, value in fixed.items()
        if config.get(key) != value
    }
    if differences:
        raise ValueError(f"memory-stability fixed controls changed: {differences}")
    if int(config["workers"]) < 1:
        raise ValueError("use DataLoader workers so child RSS and restart are exercised")


def source_identity(config_path: Path, registry_path: Path) -> dict[str, Any]:
    registry = json.loads(registry_path.read_text())
    identity = {
        "base_commit": registry["base_commit"],
        "upstream_commit": registry["upstream_commit"],
        "source_registry_path": str(registry_path),
        "source_registry_sha256": sha256(registry_path),
        "config_path": str(config_path),
        "config_sha256": sha256(config_path),
    }
    if identity["base_commit"] != "6ef353c5fc58878ead9f6fc5f58539b2a40abe28":
        raise ValueError("unexpected monorepo base commit")
    if identity["upstream_commit"] != "8c5e87750f25a035bca41bd37ae44b042a405f9a":
        raise ValueError("unexpected pinned upstream commit")
    return identity


def configure_runtime(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("highest")


def build_optimizer_scheduler(
    model: torch.nn.Module, config: Mapping[str, Any]
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.OneCycleLR]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=2e-4, weight_decay=1e-5, eps=1e-8
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        2e-4,
        int(config["schedule_num_steps"]) + 100,
        pct_start=0.01,
        cycle_momentum=False,
        anneal_strategy="linear",
    )
    return optimizer, scheduler


def make_manifest(
    dataset: SyntheticTrainDataset, config: Mapping[str, Any]
) -> dict[str, Any]:
    generator = np.random.default_rng(int(config["seed"]))
    indices = generator.permutation(len(dataset))[: int(config["manifest_samples"])].tolist()
    return {
        "dataset_root": str(dataset.root),
        "dataset_length": len(dataset),
        "indices": indices,
        "left_files": [
            str(dataset.samples[index][0].relative_to(dataset.root)) for index in indices
        ],
        "repeat_policy": "fixed manifest order repeated without itertools.cycle caching",
        "physical_batch_size": int(config["batch_size"]),
        "effective_batch_size": int(config["effective_batch_size"]),
        "updates_per_manifest_cycle": len(indices) // int(config["effective_batch_size"]),
        "probe_dataset_index": indices[0],
    }


def loader_child_pids(iterator: Any) -> list[int]:
    workers = getattr(iterator, "_workers", ())
    return [worker.pid for worker in workers if worker.pid is not None and worker.is_alive()]


def shutdown_loader(iterator: Any) -> None:
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if shutdown is not None:
        shutdown()


def run_validation_probe(
    model: torch.nn.Module,
    dataset: SyntheticTrainDataset,
    dataset_index: int,
    validation_iters: int,
) -> tuple[dict[str, Any], torch.Tensor]:
    model.eval()
    model.freeze_bn()
    batch = default_collate([dataset[dataset_index]])
    sample_ids, left, right, _, _ = batch
    left = left.cuda(non_blocking=True)
    right = right.cuda(non_blocking=True)
    reset_report(model)
    with torch.no_grad():
        output = model(left, right, iters=validation_iters, test_mode=True)[1]
    finite = bool(torch.isfinite(output).all())
    output_cpu = output.detach().cpu()
    report = {
        "scope": "deterministic Train-only engineering probe; not an accuracy split or benchmark",
        "sample": sample_ids[0],
        "dataset_index": dataset_index,
        "input_shape": list(left.shape),
        "refinements": validation_iters,
        "output_shape": list(output.shape),
        "output_finite": finite,
        "output_sha256": tensor_digest(output_cpu),
    }
    del output, left, right, batch
    reset = reset_report(model)
    model.train()
    model.freeze_bn()
    torch.cuda.synchronize()
    if not finite or not reset["clean"]:
        raise RuntimeError("validation probe was nonfinite or left state dirty")
    report["reset_clean"] = reset["clean"]
    return report, output_cpu


def lifecycle_event(
    telemetry: TelemetryWriter,
    *,
    phase: str,
    event: str,
    child_pids: Sequence[int] = (),
    **fields: Any,
) -> dict[str, Any]:
    record = {
        "timestamp": time.time(),
        "event": event,
        "phase": phase,
        "process_id": os.getpid(),
        **fields,
        "memory": cuda_memory_report(child_pids),
    }
    telemetry.write(record)
    return record


def checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.OneCycleLR,
    sampler: DeterministicRepeatingBatchSampler,
    config: Mapping[str, Any],
    identity: Mapping[str, Any],
    actual_update: int,
    logical_update: int,
    probe_report: Mapping[str, Any],
    probe_output: torch.Tensor,
) -> dict[str, Any]:
    rng_state = capture_rng_state()
    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict()
    scheduler_state = scheduler.state_dict()
    sampler_state = sampler.state_dict()
    return {
        "format_version": 1,
        "model": model_state,
        "optimizer": optimizer_state,
        "scheduler": scheduler_state,
        "actual_update": actual_update,
        "logical_update": logical_update,
        "config": dict(config),
        "source_identity": dict(identity),
        "rng_state": rng_state,
        "sampler_state": sampler_state,
        "probe_report": dict(probe_report),
        "probe_output": probe_output,
        "state_digests": {
            "model": state_digest(model_state),
            "optimizer": state_digest(optimizer_state),
            "scheduler": state_digest(scheduler_state),
            "rng": rng_digest(rng_state),
            "sampler": state_digest(sampler_state),
        },
    }


def save_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def restore_checkpoint(
    *,
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.OneCycleLR,
    sampler: DeterministicRepeatingBatchSampler,
    config: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> tuple[dict[str, Any], torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint["config"] != dict(config):
        raise RuntimeError("checkpoint configuration differs from resolved configuration")
    if checkpoint["source_identity"] != dict(identity):
        raise RuntimeError("checkpoint source identity differs from current frozen source")
    if checkpoint["actual_update"] != 25 or checkpoint["logical_update"] != 25:
        raise RuntimeError("resume checkpoint is not at actual/logical update 25")
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    sampler.load_state_dict(checkpoint["sampler_state"])
    restore_rng_state(checkpoint["rng_state"])
    observed = {
        "model": state_digest(model.state_dict()),
        "optimizer": state_digest(optimizer.state_dict()),
        "scheduler": state_digest(scheduler.state_dict()),
        "rng": rng_digest(),
        "sampler": state_digest(sampler.state_dict()),
    }
    expected = checkpoint["state_digests"]
    exact = {key: observed[key] == expected[key] for key in expected}
    if not all(exact.values()):
        raise RuntimeError(f"checkpoint restoration digest mismatch: {exact}")
    verification = {
        "checkpoint": str(path),
        "checkpoint_sha256": sha256(path),
        "actual_update": checkpoint["actual_update"],
        "logical_update": checkpoint["logical_update"],
        "state_digest_exact": exact,
        "expected_state_digests": expected,
        "observed_state_digests": observed,
        "sampler_state": sampler.state_dict(),
        "optimizer_steps": optimizer_step_report(optimizer),
        "scheduler_last_epoch": scheduler.last_epoch,
        "scheduler_step_count": scheduler._step_count,
        "rng_restored_before_continuation": True,
        "source_and_config_exact": True,
        "fresh_process_pid": os.getpid(),
        "saved_process_pid": checkpoint.get("saved_process_pid"),
    }
    expected_probe = checkpoint["probe_output"].clone()
    verification["expected_probe_report"] = checkpoint["probe_report"]
    del checkpoint
    return verification, expected_probe


def write_metadata(
    run_dir: Path,
    config: Mapping[str, Any],
    identity: Mapping[str, Any],
    model: torch.nn.Module,
) -> None:
    metadata = {
        "scope": "bounded A3-T50 memory stability and training lifecycle diagnostic",
        "config": dict(config),
        "source_identity": dict(identity),
        "environment": {
            "hostname": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu": torch.cuda.get_device_name(0),
            "gpu_total_memory_mib": torch.cuda.get_device_properties(0).total_memory / 2**20,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "python": sys.version,
            "python_executable": sys.executable,
            "spikingjelly": importlib.metadata.version("spikingjelly"),
            "parameter_dtype": str(next(model.parameters()).dtype),
            "autocast_enabled": False,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        },
        "parameters": parameter_report(model),
        "schedule": {
            "type": "OneCycleLR",
            "total_steps": int(config["schedule_num_steps"]) + 100,
            "max_lr": 2e-4,
            "initial_lr": 8e-6,
            "pct_start": 0.01,
            "anneal_strategy": "linear",
            "cycle_momentum": False,
        },
        "batch_semantics": {
            "physical_batch_size": 2,
            "effective_batch_size": 8,
            "microbatches_per_update": 4,
            "passes_per_microbatch": 2,
        },
    }
    atomic_json(metadata, run_dir / "run-metadata.json")


def create_loader(
    dataset: SyntheticTrainDataset,
    sampler: DeterministicRepeatingBatchSampler,
    config: Mapping[str, Any],
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(config["seed"]) + 991)
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=int(config["workers"]),
        pin_memory=True,
        persistent_workers=True,
        generator=generator,
    )


def run_updates(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.OneCycleLR,
    sampler: DeterministicRepeatingBatchSampler,
    iterator: Any,
    config: Mapping[str, Any],
    telemetry: TelemetryWriter,
    phase: str,
    actual_updates: Sequence[int],
    logical_updates: Sequence[int],
    steady_summary: BoundedSteadySummary | None = None,
) -> None:
    if len(actual_updates) != len(logical_updates):
        raise ValueError("actual and logical update lists differ in length")
    for actual_update, logical_update in zip(actual_updates, logical_updates):
        torch.cuda.reset_peak_memory_stats()
        data_started = time.monotonic()
        physical_batches = []
        cursor_before = sampler.state_dict()
        for _ in range(4):
            physical_batches.append(next(iterator))
            sampler.mark_consumed(int(config["batch_size"]))
        data_seconds = time.monotonic() - data_started
        child_pids = loader_child_pids(iterator)
        record = exact_training_update(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            physical_batches=physical_batches,
            train_iters=int(config["train_iters"]),
            actual_update=actual_update,
            logical_update=logical_update,
            late_counter_synthetic=logical_update != actual_update,
            accumulator_type=ExactBatchLossAccumulator,
            replay_loss_function=replay_loss,
        )
        if (
            record["optimizer_steps"]["state_entries"]
            != int(config["expected_optimizer_state_entries"])
        ):
            raise RuntimeError("Adam state-entry count changed from the verified model")
        del physical_batches
        record.update(
            timestamp=time.time(),
            event="update",
            phase=phase,
            process_id=os.getpid(),
            data_load_seconds=data_seconds,
            data_cursor_before=cursor_before,
            data_cursor=sampler.state_dict(),
            memory=cuda_memory_report(child_pids),
        )
        telemetry.write(record)
        if steady_summary is not None:
            steady_summary.add(record)
        print(
            json.dumps(
                {
                    "event": "update",
                    "actual": actual_update,
                    "logical": logical_update,
                    "allocated_mib": record["memory"]["cuda_allocated_mib"],
                    "reserved_mib": record["memory"]["cuda_reserved_mib"],
                    "peak_allocated_mib": record["memory"]["cuda_peak_allocated_mib"],
                    "seconds": record["seconds"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        del record


def run_initial(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    config: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=False)
    atomic_json(config, run_dir / "config.resolved.json")
    telemetry = TelemetryWriter(run_dir)
    dataset = SyntheticTrainDataset(args.data_root)
    manifest = make_manifest(dataset, config)
    atomic_json(manifest, run_dir / "selected-samples.json")
    sampler = DeterministicRepeatingBatchSampler(
        manifest["indices"], int(config["batch_size"])
    )
    model = SpikeStereoNet(model_args(config)).cuda().train()
    model.freeze_bn()
    optimizer, scheduler = build_optimizer_scheduler(model, config)
    write_metadata(run_dir, config, identity, model)
    loader = create_loader(dataset, sampler, config)
    iterator = iter(loader)
    lifecycle_event(telemetry, phase="initial", event="process_start")
    steady = BoundedSteadySummary()
    run_updates(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        iterator=iterator,
        config=config,
        telemetry=telemetry,
        phase="initial",
        actual_updates=range(1, 26),
        logical_updates=range(1, 26),
        steady_summary=steady,
    )
    atomic_json(steady.report(), run_dir / "steady-summary.json")

    child_pids = loader_child_pids(iterator)
    torch.cuda.reset_peak_memory_stats()
    lifecycle_event(
        telemetry,
        phase="initial",
        event="validation_before",
        child_pids=child_pids,
        actual_update=25,
        logical_update=25,
    )
    probe_report, probe_output = run_validation_probe(
        model,
        dataset,
        int(manifest["probe_dataset_index"]),
        int(config["validation_iters"]),
    )
    lifecycle_event(
        telemetry,
        phase="initial",
        event="validation_after",
        child_pids=child_pids,
        actual_update=25,
        logical_update=25,
        validation=probe_report,
        operation_peak_fields_capture_during=True,
    )

    torch.cuda.reset_peak_memory_stats()
    lifecycle_event(
        telemetry,
        phase="initial",
        event="checkpoint_save_before",
        child_pids=child_pids,
        actual_update=25,
        logical_update=25,
    )
    payload = checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        config=config,
        identity=identity,
        actual_update=25,
        logical_update=25,
        probe_report=probe_report,
        probe_output=probe_output,
    )
    payload["saved_process_pid"] = os.getpid()
    checkpoint = run_dir / "checkpoint-update25.pt"
    save_checkpoint(checkpoint, payload)
    checkpoint_record = {
        "path": str(checkpoint),
        "sha256": sha256(checkpoint),
        "bytes": checkpoint.stat().st_size,
        "actual_update": 25,
        "logical_update": 25,
        "sampler_state": sampler.state_dict(),
        "state_digests": payload["state_digests"],
    }
    del payload, probe_output
    atomic_json(checkpoint_record, run_dir / "checkpoint-update25.json")
    lifecycle_event(
        telemetry,
        phase="initial",
        event="checkpoint_save_after",
        child_pids=child_pids,
        actual_update=25,
        logical_update=25,
        checkpoint=checkpoint_record,
        operation_peak_fields_capture_during=True,
    )
    lifecycle_event(
        telemetry,
        phase="initial",
        event="process_complete",
        child_pids=child_pids,
        actual_update=25,
        logical_update=25,
        data_cursor=sampler.state_dict(),
    )
    shutdown_loader(iterator)


def run_resume(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    config: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> None:
    if not run_dir.is_dir():
        raise FileNotFoundError("initial process output is absent")
    telemetry = TelemetryWriter(run_dir)
    manifest = json.loads((run_dir / "selected-samples.json").read_text())
    dataset = SyntheticTrainDataset(args.data_root)
    if manifest != make_manifest(dataset, config):
        raise RuntimeError("dataset manifest changed between processes")
    sampler = DeterministicRepeatingBatchSampler(
        manifest["indices"], int(config["batch_size"])
    )
    model = SpikeStereoNet(model_args(config)).cuda().train()
    model.freeze_bn()
    optimizer, scheduler = build_optimizer_scheduler(model, config)
    lifecycle_event(telemetry, phase="resume", event="fresh_process_start")

    torch.cuda.reset_peak_memory_stats()
    lifecycle_event(telemetry, phase="resume", event="checkpoint_load_before")
    verification, expected_probe = restore_checkpoint(
        path=run_dir / "checkpoint-update25.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        config=config,
        identity=identity,
    )
    lifecycle_event(
        telemetry,
        phase="resume",
        event="checkpoint_load_after",
        actual_update=25,
        logical_update=25,
        verification=verification,
        data_cursor=sampler.state_dict(),
        operation_peak_fields_capture_during=True,
    )

    torch.cuda.reset_peak_memory_stats()
    lifecycle_event(
        telemetry,
        phase="resume",
        event="resume_probe_before",
        actual_update=25,
        logical_update=25,
    )
    probe_report, resumed_probe = run_validation_probe(
        model,
        dataset,
        int(manifest["probe_dataset_index"]),
        int(config["validation_iters"]),
    )
    probe_exact = bool(torch.equal(expected_probe, resumed_probe))
    probe_digest_exact = (
        probe_report["output_sha256"]
        == verification["expected_probe_report"]["output_sha256"]
    )
    verification["probe_after_resume"] = probe_report
    verification["probe_tensor_exact_equal"] = probe_exact
    verification["probe_digest_exact_equal"] = probe_digest_exact
    verification["fresh_process_distinct"] = (
        verification["saved_process_pid"] != verification["fresh_process_pid"]
    )
    if not probe_exact or not probe_digest_exact or not verification["fresh_process_distinct"]:
        raise RuntimeError("fresh-process validation probe or process identity check failed")
    del expected_probe, resumed_probe
    atomic_json(verification, run_dir / "resume-verification.json")
    lifecycle_event(
        telemetry,
        phase="resume",
        event="resume_probe_verified",
        actual_update=25,
        logical_update=25,
        validation=probe_report,
        probe_tensor_exact_equal=probe_exact,
        operation_peak_fields_capture_during=True,
    )

    loader = create_loader(dataset, sampler, config)
    iterator = iter(loader)
    run_updates(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        iterator=iterator,
        config=config,
        telemetry=telemetry,
        phase="resume_regular",
        actual_updates=range(26, 29),
        logical_updates=range(26, 29),
    )

    counter_transition = position_logical_counters(optimizer, scheduler, 319998)
    counter_transition.update(
        {
            "actual_updates_completed": 28,
            "positioned_completed_logical_updates": 319998,
            "meaning": (
                "synthetic counter position only; weights and moments have 28 actual "
                "updates, not 319,998 updates of training history"
            ),
            "data_cursor_unchanged_by_counter_positioning": sampler.state_dict(),
        }
    )
    atomic_json(counter_transition, run_dir / "late-counter-transition.json")
    lifecycle_event(
        telemetry,
        phase="late_counter",
        event="logical_counter_positioned",
        actual_update=28,
        logical_update=319998,
        counter_transition=counter_transition,
        data_cursor=sampler.state_dict(),
        child_pids=loader_child_pids(iterator),
    )
    run_updates(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        iterator=iterator,
        config=config,
        telemetry=telemetry,
        phase="late_counter",
        actual_updates=(29, 30),
        logical_updates=(319999, 320000),
    )

    child_pids = loader_child_pids(iterator)
    torch.cuda.reset_peak_memory_stats()
    lifecycle_event(
        telemetry,
        phase="final_boundary",
        event="validation_before",
        child_pids=child_pids,
        actual_update=30,
        logical_update=320000,
    )
    final_probe_report, final_probe_output = run_validation_probe(
        model,
        dataset,
        int(manifest["probe_dataset_index"]),
        int(config["validation_iters"]),
    )
    lifecycle_event(
        telemetry,
        phase="final_boundary",
        event="validation_after",
        child_pids=child_pids,
        actual_update=30,
        logical_update=320000,
        validation=final_probe_report,
        operation_peak_fields_capture_during=True,
    )
    torch.cuda.reset_peak_memory_stats()
    lifecycle_event(
        telemetry,
        phase="final_boundary",
        event="checkpoint_save_before",
        child_pids=child_pids,
        actual_update=30,
        logical_update=320000,
    )
    payload = checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        config=config,
        identity=identity,
        actual_update=30,
        logical_update=320000,
        probe_report=final_probe_report,
        probe_output=final_probe_output,
    )
    payload["saved_process_pid"] = os.getpid()
    payload["late_counter_synthetic"] = True
    payload["actual_training_updates"] = 30
    payload["weights_and_moments_are_not_320k_trained"] = True
    final_checkpoint = run_dir / "checkpoint-final.pt"
    save_checkpoint(final_checkpoint, payload)
    final_checkpoint_record = {
        "path": str(final_checkpoint),
        "sha256": sha256(final_checkpoint),
        "bytes": final_checkpoint.stat().st_size,
        "actual_update": 30,
        "logical_update": 320000,
        "late_counter_synthetic": True,
        "sampler_state": sampler.state_dict(),
        "optimizer_steps": optimizer_step_report(optimizer),
        "scheduler_last_epoch": scheduler.last_epoch,
        "state_digests": payload["state_digests"],
    }
    del payload, final_probe_output
    atomic_json(final_checkpoint_record, run_dir / "checkpoint-final.json")
    final_memory = lifecycle_event(
        telemetry,
        phase="final_boundary",
        event="checkpoint_save_after",
        child_pids=child_pids,
        actual_update=30,
        logical_update=320000,
        checkpoint=final_checkpoint_record,
        data_cursor=sampler.state_dict(),
        operation_peak_fields_capture_during=True,
    )
    summary = {
        "status": "completed",
        "scope": "short training-memory and lifecycle diagnostic only",
        "actual_training_updates": 30,
        "logical_final_update": 320000,
        "late_counter_synthetic": True,
        "weights_and_moments_are_not_320k_trained": True,
        "does_not_test_320k_accumulated_leaks": True,
        "steady_summary": json.loads((run_dir / "steady-summary.json").read_text()),
        "resume_verification": verification,
        "late_counter_transition": counter_transition,
        "final_validation": final_probe_report,
        "final_checkpoint": final_checkpoint_record,
        "final_memory": final_memory["memory"],
        "data_cursor": sampler.state_dict(),
        "source_identity": dict(identity),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    atomic_json(summary, run_dir / "summary.json")
    lifecycle_event(
        telemetry,
        phase="final_boundary",
        event="run_complete",
        child_pids=child_pids,
        actual_update=30,
        logical_update=320000,
        data_cursor=sampler.state_dict(),
    )
    shutdown_loader(iterator)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("initial", "resume"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-registry", required=True)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    registry_path = Path(args.source_registry).resolve()
    config = json.loads(config_path.read_text())
    validate_config(config)
    identity = source_identity(config_path, registry_path)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("run inside one authorized Slurm GPU allocation")
    if "5090" not in torch.cuda.get_device_name(0):
        raise RuntimeError(f"expected RTX 5090, received {torch.cuda.get_device_name(0)}")
    configure_runtime(int(config["seed"]))
    run_dir = Path(args.output).resolve()

    try:
        if args.phase == "initial":
            run_initial(
                args=args,
                run_dir=run_dir,
                config=config,
                identity=identity,
            )
        else:
            run_resume(
                args=args,
                run_dir=run_dir,
                config=config,
                identity=identity,
            )
    except Exception as error:
        if run_dir.is_dir():
            failure = {
                "status": "failed",
                "phase_process": args.phase,
                "error": repr(error),
                "timestamp": time.time(),
                "process_id": os.getpid(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "source_identity": identity,
            }
            if torch.cuda.is_available():
                failure["memory"] = cuda_memory_report()
            atomic_json(failure, run_dir / f"failure-{args.phase}.json")
        raise


if __name__ == "__main__":
    main()
