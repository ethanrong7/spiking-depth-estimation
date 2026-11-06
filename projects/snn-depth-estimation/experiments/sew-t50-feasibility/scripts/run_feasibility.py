#!/usr/bin/env python3
"""Run a bounded, real-data SpikeStereoNet T50 feasibility check."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import random
import resource
import socket
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "core"))

from core.loss import sequence_loss
from core.sew_encoder import FunctionalLIF
from core.spikestereonet import SpikeStereoNet
from core.utils.reader import depth2disp, get_block_spikes, read_depth
from exact_batch_loss import ExactBatchLossAccumulator, replay_loss
from spikingjelly.activation_based import functional


VARIANTS = {
    "A0": ("cnn", "cnn"),
    "A1": ("sew", "cnn"),
    "A2": ("cnn", "sew"),
    "A3": ("sew", "sew"),
}


def atomic_json(data, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def emit(run_dir, event, **fields):
    record = {"event": event, "time": time.time(), **fields}
    with (run_dir / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
    return record


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def host_memory_mib():
    current = None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    try:
        status = Path("/proc/self/status").read_text().splitlines()
        values = {
            line.split(":", 1)[0]: line.split(":", 1)[1].strip()
            for line in status
            if ":" in line
        }
        current = float(values["VmRSS"].split()[0]) / 1024.0
        peak = float(values["VmHWM"].split()[0]) / 1024.0
    except (FileNotFoundError, KeyError, ValueError):
        pass
    return {"rss_mib": current, "peak_rss_mib": peak}


class SyntheticTrainDataset(Dataset):
    def __init__(self, root):
        self.root = Path(root).resolve()
        train = self.root / "Train"
        if not train.is_dir():
            raise FileNotFoundError(f"Synthetic Train directory is absent: {train}")
        self.samples = []
        for left in sorted(train.glob("scene_*/spike_l/*_spike_l.dat")):
            frame = left.name[: -len("_spike_l.dat")]
            scene = left.parents[1]
            right = scene / "spike_r" / f"{frame}_spike_r.dat"
            depth = scene / "depth" / f"{frame}_depth.exr"
            if not right.is_file() or not depth.is_file():
                raise FileNotFoundError(f"Unpaired sample: {left}")
            self.samples.append((left, right, depth))
        if not self.samples:
            raise RuntimeError(f"No paired synthetic samples found under {train}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        left_path, right_path, depth_path = self.samples[index]
        left = torch.from_numpy(get_block_spikes(str(left_path)).astype(np.float32))
        right = torch.from_numpy(get_block_spikes(str(right_path)).astype(np.float32))
        depth = read_depth(str(depth_path))
        disparity = depth2disp(depth, f=289.740625, B=0.08)
        valid = torch.from_numpy((disparity < 512).astype(np.float32))
        disparity = torch.from_numpy((-disparity).astype(np.float32)).unsqueeze(0)
        relative = str(left_path.relative_to(self.root))
        return relative, left, right, disparity, valid


class SpikeActivity:
    def __init__(self, model):
        self.values = defaultdict(list)
        self.handles = []
        for name, module in model.named_modules():
            if isinstance(module, FunctionalLIF):
                self.handles.append(module.register_forward_hook(self._hook(name)))

    def _hook(self, name):
        def record(_module, _inputs, output):
            spike = output[0]
            self.values[name].append(spike.detach().float().mean())

        return record

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def report(self):
        result = {}
        for name, values in sorted(self.values.items()):
            stacked = torch.stack(values)
            result[name] = {
                "calls": len(values),
                "mean": float(stacked.mean()),
                "min": float(stacked.min()),
                "max": float(stacked.max()),
            }
        return result


def model_args(config):
    feature, context = VARIANTS[config["variant"]]
    return SimpleNamespace(
        hidden_dims=[128, 128, 128],
        context_norm=config["context_norm"],
        n_downsample=2,
        n_snn_layers=3,
        shared_backbone=False,
        feature_encoder=feature,
        context_encoder=context,
        sew_timesteps=config.get("sew_timesteps", 50),
        sew_input_mode=config.get("sew_input_mode", "sensor_time"),
        sew_input_bins=config.get("sew_input_bins", 50),
        sew_checkpoint_chunk_size=config["sew_checkpoint_chunk_size"],
        sew_spatial_checkpoint=config.get("sew_spatial_checkpoint", False),
        matched_initialization_seed=config["seed"],
        corr_levels=4,
        corr_radius=4,
        mixed_precision=False,
        corr_implementation=config["corr_implementation"],
    )


def parameter_report(model):
    groups = defaultdict(int)
    for name, parameter in model.named_parameters():
        groups[name.split(".", 1)[0]] += parameter.numel()
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "components": dict(sorted(groups.items())),
    }


def gradient_report(model):
    groups = defaultdict(lambda: {"squared_norm": 0.0, "tensors": 0, "nonzero": 0})
    finite = True
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach()
        finite = finite and bool(torch.isfinite(gradient).all())
        key = name.split(".", 1)[0]
        norm = float(gradient.float().norm())
        groups[key]["squared_norm"] += norm * norm
        groups[key]["tensors"] += 1
        groups[key]["nonzero"] += int(bool(torch.count_nonzero(gradient)))
    report = {}
    for key, value in sorted(groups.items()):
        report[key] = {
            "norm": math.sqrt(value.pop("squared_norm")),
            **value,
        }
    return finite, report


def batchnorm_buffer_hashes(model):
    result = {}
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.BatchNorm2d):
            continue
        digest = hashlib.sha256()
        for field in ("running_mean", "running_var", "num_batches_tracked"):
            value = getattr(module, field)
            digest.update(field.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        result[name] = digest.hexdigest()
    return result


def probe_parameters(model):
    probes = {}
    for component in ("fnet", "cnet", "update_block", "context_zqr_convs"):
        module = getattr(model, component)
        name, parameter = next(
            (name, parameter)
            for name, parameter in module.named_parameters()
            if parameter.requires_grad
        )
        probes[component] = {
            "name": name,
            "parameter": parameter,
            "before": parameter.detach().flatten()[:1024].clone(),
        }
    return probes


def probe_deltas(probes):
    return {
        component: {
            "name": probe["name"],
            "max_abs_change": float(
                (probe["parameter"].detach().flatten()[:1024] - probe["before"])
                .abs()
                .max()
            ),
        }
        for component, probe in probes.items()
    }


def probe_gradients(probes):
    result = {}
    for component, probe in probes.items():
        gradient = probe["parameter"].grad
        if gradient is None:
            result[component] = {
                "name": probe["name"],
                "present": False,
                "finite": False,
                "nonzero": False,
                "norm": None,
            }
        else:
            result[component] = {
                "name": probe["name"],
                "present": True,
                "finite": bool(torch.isfinite(gradient).all()),
                "nonzero": bool(torch.count_nonzero(gradient)),
                "norm": float(gradient.detach().float().norm()),
            }
    return result


def reset_report(model):
    functional.reset_net(model)
    states = {}
    clean = True
    for name, module in model.named_modules():
        for field in ("v", "spike"):
            if not hasattr(module, field):
                continue
            value = getattr(module, field)
            if torch.is_tensor(value):
                maximum = float(value.detach().abs().max())
            else:
                maximum = abs(float(value))
            states[f"{name}.{field}"] = maximum
            clean = clean and maximum == 0.0
    return {"clean": clean, "states": states}


def exact_batch_stats_report(stats):
    rate = stats.global_rate.detach().float()
    return {
        "true_loss": float(stats.true_loss),
        "true_disparity_loss": float(stats.true_disparity_loss),
        "true_rate_loss": float(stats.true_rate_loss),
        "true_voltage_loss": float(stats.true_voltage_loss),
        "valid_pixel_count": stats.valid_pixel_count,
        "spike_batch_refinement_count": stats.spike_batch_refinement_count,
        "voltage_element_count": stats.voltage_element_count,
        "global_spike_rate": {
            "mean": float(rate.mean()),
            "min": float(rate.min()),
            "max": float(rate.max()),
            "nonzero_elements": int(torch.count_nonzero(rate)),
            "elements": rate.numel(),
        },
    }


def compare_exact_batch_stats(first, replayed, rtol=1e-5, atol=1e-6):
    tensor_fields = (
        "disparity_numerator",
        "spike_sum",
        "voltage_square_sum",
    )
    count_fields = (
        "valid_pixel_count",
        "spike_batch_refinement_count",
        "voltage_element_count",
    )
    maximum_differences = {}
    tensors_match = True
    for field in tensor_fields:
        expected = getattr(first, field)
        actual = getattr(replayed, field)
        maximum_differences[field] = float((expected - actual).abs().max())
        tensors_match = tensors_match and bool(
            torch.allclose(expected, actual, rtol=rtol, atol=atol)
        )
    counts_match = all(
        getattr(first, field) == getattr(replayed, field) for field in count_fields
    )
    return {
        "rtol": rtol,
        "atol": atol,
        "tensor_max_abs_differences": maximum_differences,
        "counts_match": counts_match,
        "all_statistics_match": tensors_match and counts_match,
    }


def repeatability_check(model, left, right):
    model.eval()
    model.freeze_bn()
    functional.reset_net(model)
    with torch.no_grad():
        first = model(left[:1], right[:1], iters=1, test_mode=True)[1]
    first_reset = reset_report(model)
    with torch.no_grad():
        second = model(left[:1], right[:1], iters=1, test_mode=True)[1]
    maximum = float((first - second).abs().max())
    second_reset = reset_report(model)
    return {
        "max_abs_output_difference": maximum,
        "exact": bool(torch.equal(first, second)),
        "first_reset": first_reset,
        "second_reset": second_reset,
    }


def validate_config(config):
    required = {
        "variant",
        "batch_size",
        "successful_updates",
        "schedule_num_steps",
        "input_bins",
        "height",
        "width",
        "precision",
        "train_iters",
        "corr_implementation",
        "context_norm",
        "sew_checkpoint_chunk_size",
        "seed",
        "workers",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Missing configuration fields: {missing}")
    if config["variant"] not in VARIANTS:
        raise ValueError(f"Unknown variant: {config['variant']}")
    if config["precision"] != "fp32":
        raise ValueError("Initial feasibility candidates are FP32 only")
    if (config["input_bins"], config["height"], config["width"]) != (50, 250, 400):
        raise ValueError("Initial feasibility candidates require uncropped 50x250x400 inputs")
    if config["train_iters"] != 16:
        raise ValueError("Initial feasibility candidates require 16 refinements")
    if config["batch_size"] not in (2, 4):
        raise ValueError("Physical batch must be 2 or 4")
    if config["successful_updates"] < 3:
        raise ValueError("A successful feasibility run requires at least three updates")
    effective_batch_size = config.get("effective_batch_size", config["batch_size"])
    if effective_batch_size < config["batch_size"]:
        raise ValueError("Effective batch cannot be smaller than physical batch")
    if effective_batch_size % config["batch_size"]:
        raise ValueError("Effective batch must be divisible by physical batch")
    if effective_batch_size != config["batch_size"] and not config.get(
        "exact_batch_replay", False
    ):
        raise ValueError("Larger effective batches require exact_batch_replay=true")


def run(args):
    run_dir = Path(args.output).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    validate_config(config)
    atomic_json(config, run_dir / "config.resolved.json")
    source_registry_path = Path(args.source_registry).resolve()
    source_registry = json.loads(source_registry_path.read_text())

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Run inside one authorized Slurm GPU allocation")
    if "5090" not in torch.cuda.get_device_name(0):
        raise RuntimeError(f"Expected RTX 5090, received {torch.cuda.get_device_name(0)}")

    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    dataset = SyntheticTrainDataset(args.data_root)
    generator = np.random.default_rng(config["seed"])
    effective_batch_size = config.get("effective_batch_size", config["batch_size"])
    microbatches_per_update = effective_batch_size // config["batch_size"]
    required_samples = effective_batch_size * config["successful_updates"]
    indices = generator.permutation(len(dataset))[:required_samples].tolist()
    selected = [str(dataset.samples[index][0].relative_to(dataset.root)) for index in indices]
    atomic_json(
        {
            "dataset_root": str(dataset.root),
            "dataset_length": len(dataset),
            "indices": indices,
            "left_files": selected,
        },
        run_dir / "selected-samples.json",
    )
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["workers"],
        pin_memory=True,
        drop_last=True,
    )

    model = SpikeStereoNet(model_args(config)).cuda().train()
    model.freeze_bn()
    batchnorm_before = batchnorm_buffer_hashes(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=2e-4, weight_decay=1e-5, eps=1e-8
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        2e-4,
        config["schedule_num_steps"] + 100,
        pct_start=0.01,
        cycle_momentum=False,
        anneal_strategy="linear",
    )
    probes = probe_parameters(model)
    report = {
        "status": "running",
        "config": config,
        "config_sha256": sha256(config_path),
        "source": {
            "registry_path": str(source_registry_path),
            "registry_sha256": sha256(source_registry_path),
            "base_commit": source_registry["base_commit"],
            "upstream_commit": source_registry["upstream_commit"],
            "dirty_patch": source_registry["source_diff"],
        },
        "environment": {
            "hostname": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu": torch.cuda.get_device_name(0),
            "gpu_total_memory_mib": (
                torch.cuda.get_device_properties(0).total_memory / 2**20
            ),
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
        "optimizer": {
            "type": "AdamW",
            "max_lr": 2e-4,
            "initial_lr_after_scheduler_construction": optimizer.param_groups[0]["lr"],
            "weight_decay": 1e-5,
            "eps": 1e-8,
        },
        "scheduler": {
            "type": "OneCycleLR",
            "total_steps": config["schedule_num_steps"] + 100,
            "pct_start": 0.01,
            "cycle_momentum": False,
            "anneal_strategy": "linear",
        },
        "objective": (
            "upstream sequence_loss unchanged via exact two-pass sufficient-statistic "
            "gradient replay"
            if microbatches_per_update > 1
            else "upstream sequence_loss unchanged"
        ),
        "batch_semantics": {
            "physical_batch_size": config["batch_size"],
            "effective_batch_size": effective_batch_size,
            "microbatches_per_update": microbatches_per_update,
            "model_passes_per_microbatch": 2 if microbatches_per_update > 1 else 1,
            "exact_batch_replay": microbatches_per_update > 1,
        },
        "matched_initialization": {
            "base_seed": config["seed"],
            "component_seed_offsets": {
                "cnet": 101,
                "update_block": 202,
                "context_zqr_convs": 303,
                "fnet": 404,
            },
        },
        "batchnorm_buffer_hashes_before": batchnorm_before,
        "updates": [],
    }
    atomic_json(report, run_dir / "result.json")
    emit(run_dir, "start", variant=config["variant"], parameters=report["parameters"])

    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    last_batch = None
    loader_iterator = iter(loader)
    for update_index in range(1, config["successful_updates"] + 1):
        physical_batches = [next(loader_iterator) for _ in range(microbatches_per_update)]
        step_started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)

        if microbatches_per_update == 1:
            _, left, right, truth, valid = physical_batches[0]
            left, right, truth, valid = [
                value.cuda(non_blocking=True)
                for value in (left, right, truth, valid)
            ]
            if tuple(left.shape[1:]) != (50, 250, 400):
                raise RuntimeError(f"Unexpected input shape: {tuple(left.shape)}")
            last_batch = (left, right)
            functional.reset_net(model)
            activity = SpikeActivity(model) if update_index == 1 else None
            predictions, updater_spikes, potentials = model(
                left, right, iters=config["train_iters"]
            )
            if activity is not None:
                activity.close()
                encoder_activity = activity.report()
            else:
                encoder_activity = None
            loss, metrics = sequence_loss(
                updater_spikes,
                potentials,
                predictions,
                truth,
                valid,
                [250, 400],
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at update {update_index}")
            loss.backward()
            updater_spike_rates = [
                float(value.detach().float().mean()) for value in updater_spikes
            ]
            pass_evidence = None
        else:
            report["active_stage"] = {
                "update": update_index,
                "stage": "exact_batch_first_no_grad_pass",
            }
            atomic_json(report, run_dir / "result.json")
            first_accumulator = ExactBatchLossAccumulator()
            first_updater_rates = [0.0] * config["train_iters"]
            first_resets = []
            parameter_versions_before = {
                name: parameter._version
                for name, parameter in model.named_parameters()
            }
            first_probe_values = {
                component: probe["parameter"]
                .detach()
                .flatten()[:1024]
                .clone()
                for component, probe in probes.items()
            }
            cpu_rng_before = torch.random.get_rng_state().clone()
            cuda_rng_before = torch.cuda.get_rng_state().clone()
            encoder_activity = None

            with torch.no_grad():
                for microbatch_index, batch in enumerate(physical_batches):
                    _, left, right, truth, valid = batch
                    left, right, truth, valid = [
                        value.cuda(non_blocking=True)
                        for value in (left, right, truth, valid)
                    ]
                    if tuple(left.shape[1:]) != (50, 250, 400):
                        raise RuntimeError(
                            f"Unexpected input shape: {tuple(left.shape)}"
                        )
                    functional.reset_net(model)
                    activity = (
                        SpikeActivity(model)
                        if update_index == 1 and microbatch_index == 0
                        else None
                    )
                    predictions, updater_spikes, potentials = model(
                        left, right, iters=config["train_iters"]
                    )
                    if activity is not None:
                        activity.close()
                        encoder_activity = activity.report()
                    first_accumulator.accumulate(
                        updater_spikes,
                        potentials,
                        predictions,
                        truth,
                        valid,
                        [250, 400],
                    )
                    for refinement, value in enumerate(updater_spikes):
                        first_updater_rates[refinement] += float(
                            value.detach().float().mean()
                        ) / microbatches_per_update
                    first_resets.append(reset_report(model))
                    del predictions, updater_spikes, potentials
                    del left, right, truth, valid

            first_stats = first_accumulator.finalize()
            if not torch.isfinite(first_stats.true_loss):
                raise FloatingPointError(
                    f"Nonfinite exact global loss at update {update_index}"
                )
            versions_unchanged = parameter_versions_before == {
                name: parameter._version
                for name, parameter in model.named_parameters()
            }
            probes_unchanged = all(
                torch.equal(
                    first_probe_values[component],
                    probe["parameter"].detach().flatten()[:1024],
                )
                for component, probe in probes.items()
            )
            cpu_rng_unchanged = torch.equal(cpu_rng_before, torch.random.get_rng_state())
            cuda_rng_unchanged = torch.equal(cuda_rng_before, torch.cuda.get_rng_state())
            if not (
                versions_unchanged
                and probes_unchanged
                and cpu_rng_unchanged
                and cuda_rng_unchanged
                and all(item["clean"] for item in first_resets)
            ):
                raise RuntimeError(
                    "Exact-batch first pass changed parameters/RNG or left state dirty"
                )

            report["active_stage"] = {
                "update": update_index,
                "stage": "exact_batch_gradient_replay",
            }
            atomic_json(report, run_dir / "result.json")
            replay_accumulator = ExactBatchLossAccumulator()
            replay_resets = []
            replay_surrogate_scalars = []
            for microbatch_index, batch in enumerate(physical_batches):
                _, left, right, truth, valid = batch
                left, right, truth, valid = [
                    value.cuda(non_blocking=True)
                    for value in (left, right, truth, valid)
                ]
                functional.reset_net(model)
                predictions, updater_spikes, potentials = model(
                    left, right, iters=config["train_iters"]
                )
                replay_accumulator.accumulate(
                    updater_spikes,
                    potentials,
                    predictions,
                    truth,
                    valid,
                    [250, 400],
                )
                local_replay_loss = replay_loss(
                    first_stats,
                    updater_spikes,
                    potentials,
                    predictions,
                    truth,
                    valid,
                    [250, 400],
                )
                if not torch.isfinite(local_replay_loss):
                    raise FloatingPointError(
                        f"Nonfinite replay scalar at update {update_index}, "
                        f"microbatch {microbatch_index + 1}"
                    )
                replay_surrogate_scalars.append(float(local_replay_loss.detach()))
                local_replay_loss.backward()
                replay_resets.append(reset_report(model))
                if microbatch_index == microbatches_per_update - 1:
                    last_batch = (left, right)
                del predictions, updater_spikes, potentials, truth, valid

            replay_stats = replay_accumulator.finalize()
            stats_comparison = compare_exact_batch_stats(first_stats, replay_stats)
            report["last_exact_batch_statistics_comparison"] = {
                "update": update_index,
                **stats_comparison,
            }
            atomic_json(report, run_dir / "result.json")
            if not stats_comparison["all_statistics_match"]:
                raise RuntimeError(
                    "Exact-batch replay outputs do not reproduce first-pass statistics"
                )
            if not all(item["clean"] for item in replay_resets):
                raise RuntimeError("Exact-batch replay left stateful modules dirty")
            loss = first_stats.true_loss
            metrics = {
                "scope": "unchanged objective over the complete effective batch",
                **exact_batch_stats_report(first_stats),
            }
            updater_spike_rates = first_updater_rates
            pass_evidence = {
                "first_pass_under_no_grad": True,
                "parameter_versions_unchanged_before_replay": versions_unchanged,
                "parameter_probes_unchanged_before_replay": probes_unchanged,
                "cpu_rng_unchanged_before_replay": cpu_rng_unchanged,
                "cuda_rng_unchanged_before_replay": cuda_rng_unchanged,
                "first_pass_resets_clean": all(
                    item["clean"] for item in first_resets
                ),
                "replay_resets_clean": all(item["clean"] for item in replay_resets),
                "statistics_comparison": stats_comparison,
                "replay_surrogate_scalars_diagnostic": replay_surrogate_scalars,
                "replay_surrogate_scalars_excluded_from_reported_objective": True,
            }

        gradients_finite, gradients = gradient_report(model)
        raw_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not gradients_finite or not torch.isfinite(raw_grad_norm):
            raise FloatingPointError(f"Nonfinite gradients at update {update_index}")
        post_clip_finite, post_clip_gradients = gradient_report(model)
        post_clip_norm = math.sqrt(
            sum(value["norm"] ** 2 for value in post_clip_gradients.values())
        )
        if not post_clip_finite or post_clip_norm == 0.0:
            raise FloatingPointError(
                f"Nonfinite or zero loss gradients after clipping at update {update_index}"
            )
        probes_after_clipping = probe_gradients(probes)
        optimizer.step()
        scheduler.step()
        reset = reset_report(model)
        torch.cuda.synchronize()
        update = {
            "update": update_index,
            "loss": loss.detach().item(),
            "metrics": metrics,
            "raw_gradient_norm": float(raw_grad_norm),
            "gradient_groups": gradients,
            "gradients_finite": gradients_finite,
            "post_clip_gradient_norm": post_clip_norm,
            "post_clip_gradient_groups": post_clip_gradients,
            "post_clip_gradients_finite": post_clip_finite,
            "post_clip_probe_gradients": probes_after_clipping,
            "learning_rate": scheduler.get_last_lr()[0],
            "seconds": time.monotonic() - step_started,
            "updater_spike_rates": updater_spike_rates,
            "encoder_spike_activity": encoder_activity,
            "exact_batch_replay_evidence": pass_evidence,
            "reset": reset,
            "cumulative_cuda_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "cumulative_cuda_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
            "host_memory": host_memory_mib(),
        }
        report["updates"].append(update)
        report["active_stage"] = {"update": update_index, "stage": "completed"}
        atomic_json(report, run_dir / "result.json")
        emit(
            run_dir,
            "update",
            update=update_index,
            loss=update["loss"],
            raw_gradient_norm=update["raw_gradient_norm"],
            peak_allocated_mib=update["cumulative_cuda_peak_allocated_mib"],
        )

    if len(report["updates"]) != config["successful_updates"]:
        raise RuntimeError("Data loader ended before the requested successful updates")
    report["optimizer_state_entries"] = len(optimizer.state)
    report["expected_optimizer_state_entries"] = sum(
        parameter.grad is not None for parameter in model.parameters()
    )
    report["parameter_probe_deltas"] = probe_deltas(probes)
    report["repeatability"] = repeatability_check(model, *last_batch)
    report["runtime_seconds"] = time.monotonic() - started
    report["cuda_peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
    report["cuda_peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
    report["host_memory"] = host_memory_mib()
    report["batchnorm_buffer_hashes_after"] = batchnorm_buffer_hashes(model)
    report["batchnorm_buffers_unchanged"] = (
        report["batchnorm_buffer_hashes_before"]
        == report["batchnorm_buffer_hashes_after"]
    )

    first_activity = report["updates"][0]["encoder_spike_activity"] or {}
    silent = [name for name, value in first_activity.items() if value["mean"] == 0.0]
    saturated = [name for name, value in first_activity.items() if value["mean"] == 1.0]
    report["spike_health"] = {
        "silent_encoder_layers": silent,
        "saturated_encoder_layers": saturated,
        "all_sites_nonzero_and_nonsaturated": not silent and not saturated,
        "criterion": "Every recorded site has exact mean rate > 0 and < 1; raw rates retained",
    }
    changed = {
        name: value["max_abs_change"] > 0.0
        for name, value in report["parameter_probe_deltas"].items()
    }
    report["success_checks"] = {
        "three_or_more_updates": len(report["updates"]) >= 3,
        "finite_nonzero_total_gradient": all(
            update["gradients_finite"]
            and update["raw_gradient_norm"] > 0.0
            and update["post_clip_gradients_finite"]
            and update["post_clip_gradient_norm"] > 0.0
            for update in report["updates"]
        ),
        "nonzero_loss_gradient_each_component": all(
            all(
                update["post_clip_gradient_groups"].get(component, {}).get("norm", 0.0)
                > 0.0
                for component in ("fnet", "cnet", "update_block", "context_zqr_convs")
            )
            for update in report["updates"]
        ),
        "all_component_probes_changed": all(changed.values()),
        "all_component_probes_have_nonzero_loss_gradient": all(
            all(
                probe["present"] and probe["finite"] and probe["nonzero"]
                for probe in update["post_clip_probe_gradients"].values()
            )
            for update in report["updates"]
        ),
        "optimizer_state_initialized": (
            report["optimizer_state_entries"] > 0
            and report["optimizer_state_entries"]
            == report["expected_optimizer_state_entries"]
        ),
        "batchnorm_buffers_unchanged": report["batchnorm_buffers_unchanged"],
        "all_resets_clean": all(update["reset"]["clean"] for update in report["updates"]),
        "reset_repeatable": report["repeatability"]["exact"],
        "encoder_spike_sites_nonzero_nonsaturated": report["spike_health"][
            "all_sites_nonzero_and_nonsaturated"
        ],
        "exact_batch_replay_valid": (
            microbatches_per_update == 1
            or all(
                update["exact_batch_replay_evidence"] is not None
                and update["exact_batch_replay_evidence"][
                    "parameter_versions_unchanged_before_replay"
                ]
                and update["exact_batch_replay_evidence"][
                    "parameter_probes_unchanged_before_replay"
                ]
                and update["exact_batch_replay_evidence"][
                    "cpu_rng_unchanged_before_replay"
                ]
                and update["exact_batch_replay_evidence"][
                    "cuda_rng_unchanged_before_replay"
                ]
                and update["exact_batch_replay_evidence"][
                    "first_pass_resets_clean"
                ]
                and update["exact_batch_replay_evidence"]["replay_resets_clean"]
                and update["exact_batch_replay_evidence"][
                    "statistics_comparison"
                ]["all_statistics_match"]
                for update in report["updates"]
            )
        ),
    }
    report["status"] = (
        "successful_updates_with_nonzero_nonsaturated_spikes"
        if all(report["success_checks"].values())
        else "completed_with_failed_health_criteria"
    )
    atomic_json(report, run_dir / "result.json")
    emit(run_dir, report["status"], checks=report["success_checks"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-registry", required=True)
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        output = Path(args.output).resolve()
        if output.is_dir():
            failure = {
                "status": "failed",
                "error": repr(error),
                "time": time.time(),
                "host_memory": host_memory_mib(),
            }
            partial_path = output / "result.json"
            if partial_path.is_file():
                partial = json.loads(partial_path.read_text())
                failure.update(
                    slurm_job_id=partial.get("environment", {}).get("slurm_job_id"),
                    config=partial.get("config"),
                    config_sha256=partial.get("config_sha256"),
                    source=partial.get("source"),
                    environment=partial.get("environment"),
                    failure_stage=partial.get("active_stage"),
                    completed_updates=len(partial.get("updates", [])),
                )
            if torch.cuda.is_available():
                failure.update(
                    cuda_peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                    cuda_peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                )
            atomic_json(failure, output / "failure.json")
            emit(output, "failure", error=repr(error))
        raise


if __name__ == "__main__":
    main()
