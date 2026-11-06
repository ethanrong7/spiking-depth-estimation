"""Bounded training and lifecycle helpers for the T50 memory-stability run."""

from __future__ import annotations

import hashlib
import math
import os
import pickle
import random
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Sampler


def _hash_value(digest: "hashlib._Hash", value: Any) -> None:
    """Hash nested PyTorch state without retaining tensor copies."""

    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor\0")
        digest.update(str(tensor.dtype).encode())
        digest.update(repr(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
        return
    if isinstance(value, Mapping):
        digest.update(b"mapping\0")
        for key in sorted(value, key=lambda item: repr(item)):
            _hash_value(digest, key)
            _hash_value(digest, value[key])
        return
    if isinstance(value, (list, tuple)):
        digest.update(type(value).__name__.encode() + b"\0")
        for item in value:
            _hash_value(digest, item)
        return
    digest.update(b"scalar\0")
    digest.update(pickle.dumps(value, protocol=5))


def state_digest(value: Any) -> str:
    digest = hashlib.sha256()
    _hash_value(digest, value)
    return digest.hexdigest()


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def rng_digest(state: Mapping[str, Any] | None = None) -> str:
    return state_digest(capture_rng_state() if state is None else state)


class DeterministicRepeatingBatchSampler(Sampler[list[int]]):
    """Repeat a fixed manifest while tracking consumed, not prefetched, samples.

    A DataLoader may ask the iterator for future batches.  ``state_dict`` uses
    only ``consumed_samples`` advanced explicitly by the training loop, so a
    fresh process resumes from the last batch actually received by that loop.
    """

    def __init__(
        self,
        manifest_indices: Sequence[int],
        batch_size: int,
        consumed_samples: int = 0,
    ) -> None:
        self.manifest_indices = tuple(int(index) for index in manifest_indices)
        self.batch_size = int(batch_size)
        self.consumed_samples = int(consumed_samples)
        if not self.manifest_indices:
            raise ValueError("repeating sampler requires a nonempty manifest")
        if self.batch_size <= 0:
            raise ValueError("batch size must be positive")
        if len(self.manifest_indices) % self.batch_size:
            raise ValueError("manifest length must be divisible by physical batch size")
        if self.consumed_samples < 0 or self.consumed_samples % self.batch_size:
            raise ValueError("consumed position must end on a physical-batch boundary")

    def __iter__(self) -> Iterable[list[int]]:
        position = self.consumed_samples
        manifest_size = len(self.manifest_indices)
        while True:
            yield [
                self.manifest_indices[(position + offset) % manifest_size]
                for offset in range(self.batch_size)
            ]
            position += self.batch_size

    def __len__(self) -> int:
        # The stream is intentionally unbounded.  This large finite value keeps
        # DataLoader length arithmetic valid without caching repeated batches.
        return (2**63 - 1) // self.batch_size

    def mark_consumed(self, count: int) -> None:
        if int(count) != self.batch_size:
            raise ValueError("consume exactly one physical batch at a time")
        self.consumed_samples += self.batch_size

    def state_dict(self) -> dict[str, Any]:
        size = len(self.manifest_indices)
        return {
            "manifest_indices": list(self.manifest_indices),
            "manifest_size": size,
            "batch_size": self.batch_size,
            "consumed_samples": self.consumed_samples,
            "cycle": self.consumed_samples // size,
            "position_in_cycle": self.consumed_samples % size,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = {
            "manifest_indices": list(self.manifest_indices),
            "manifest_size": len(self.manifest_indices),
            "batch_size": self.batch_size,
        }
        for key, value in expected.items():
            if state[key] != value:
                raise ValueError(f"sampler checkpoint mismatch for {key}")
        consumed = int(state["consumed_samples"])
        if state["cycle"] != consumed // len(self.manifest_indices):
            raise ValueError("sampler cycle is inconsistent with consumed position")
        if state["position_in_cycle"] != consumed % len(self.manifest_indices):
            raise ValueError("sampler in-cycle position is inconsistent")
        if consumed % self.batch_size:
            raise ValueError("sampler checkpoint is not on a batch boundary")
        self.consumed_samples = consumed


def _proc_memory_mib(pid: int) -> dict[str, float | int | None]:
    result: dict[str, float | int | None] = {"pid": pid, "rss_mib": None, "peak_rss_mib": None}
    try:
        lines = Path(f"/proc/{pid}/status").read_text().splitlines()
        fields = {
            line.split(":", 1)[0]: line.split(":", 1)[1].strip()
            for line in lines
            if ":" in line
        }
        result["rss_mib"] = float(fields["VmRSS"].split()[0]) / 1024.0
        result["peak_rss_mib"] = float(fields["VmHWM"].split()[0]) / 1024.0
    except (FileNotFoundError, KeyError, PermissionError, ProcessLookupError, ValueError):
        pass
    return result


def process_memory_report(child_pids: Sequence[int] = ()) -> dict[str, Any]:
    live_children = []
    for pid in sorted(set(int(value) for value in child_pids)):
        report = _proc_memory_mib(pid)
        if report["rss_mib"] is not None:
            live_children.append(report)
    child_sum = sum(float(item["rss_mib"]) for item in live_children)
    return {
        "main": _proc_memory_mib(os.getpid()),
        "dataloader_children": live_children,
        "dataloader_child_rss_sum_mib": child_sum,
        "child_rss_sum_caveat": "RSS sums can double-count shared pages",
    }


def cuda_memory_report(child_pids: Sequence[int] = ()) -> dict[str, Any]:
    torch.cuda.synchronize()
    return {
        "cuda_allocated_mib": torch.cuda.memory_allocated() / 2**20,
        "cuda_reserved_mib": torch.cuda.memory_reserved() / 2**20,
        "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "cuda_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "gpu_capacity_mib": torch.cuda.get_device_properties(0).total_memory / 2**20,
        "host_memory": process_memory_report(child_pids),
    }


def reset_report(model: torch.nn.Module) -> dict[str, Any]:
    from spikingjelly.activation_based import functional

    functional.reset_net(model)
    dirty = {}
    for name, module in model.named_modules():
        for field in ("v", "spike"):
            if not hasattr(module, field):
                continue
            value = getattr(module, field)
            maximum = (
                float(value.detach().abs().max())
                if torch.is_tensor(value)
                else abs(float(value))
            )
            if maximum != 0.0:
                dirty[f"{name}.{field}"] = maximum
    return {"clean": not dirty, "dirty_states": dirty}


def gradient_report(model: torch.nn.Module) -> tuple[bool, float, dict[str, float]]:
    finite = True
    squared = 0.0
    groups: dict[str, float] = defaultdict(float)
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach()
        finite = finite and bool(torch.isfinite(gradient).all())
        norm = float(gradient.float().norm())
        squared += norm * norm
        groups[name.split(".", 1)[0]] += norm * norm
    return finite, math.sqrt(squared), {
        key: math.sqrt(value) for key, value in sorted(groups.items())
    }


def _parameter_probes(model: torch.nn.Module) -> dict[str, tuple[str, torch.nn.Parameter, torch.Tensor]]:
    probes = {}
    for component in ("fnet", "cnet", "update_block", "context_zqr_convs"):
        module = getattr(model, component)
        name, parameter = next(
            (name, parameter)
            for name, parameter in module.named_parameters()
            if parameter.requires_grad
        )
        probes[component] = (
            name,
            parameter,
            parameter.detach().flatten()[:1024].clone(),
        )
    return probes


def _probe_deltas(
    probes: Mapping[str, tuple[str, torch.nn.Parameter, torch.Tensor]],
) -> dict[str, dict[str, Any]]:
    return {
        component: {
            "name": name,
            "max_abs_change": float(
                (parameter.detach().flatten()[:1024] - before).abs().max()
            ),
        }
        for component, (name, parameter, before) in probes.items()
    }


def optimizer_step_report(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    steps = []
    for state in optimizer.state.values():
        if "step" in state:
            value = state["step"]
            steps.append(int(value.item()) if torch.is_tensor(value) else int(value))
    return {
        "state_entries": len(optimizer.state),
        "step_entries": len(steps),
        "step_min": min(steps) if steps else None,
        "step_max": max(steps) if steps else None,
        "all_steps_equal": len(set(steps)) <= 1,
    }


def position_logical_counters(
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.OneCycleLR,
    completed_logical_updates: int,
) -> dict[str, Any]:
    """Place Adam and OneCycleLR counters before synthetic late updates."""

    if not optimizer.state:
        raise RuntimeError("Adam state must be initialized before counter positioning")
    before = {
        "optimizer": optimizer_step_report(optimizer),
        "scheduler_last_epoch": scheduler.last_epoch,
        "scheduler_step_count": scheduler._step_count,
        "learning_rates": list(scheduler.get_last_lr()),
    }
    for state in optimizer.state.values():
        if "step" not in state:
            continue
        if torch.is_tensor(state["step"]):
            state["step"].fill_(completed_logical_updates)
        else:
            state["step"] = completed_logical_updates
    scheduler.last_epoch = completed_logical_updates
    scheduler._step_count = completed_logical_updates + 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        positioned_lrs = list(scheduler.get_lr())
    for group, learning_rate in zip(optimizer.param_groups, positioned_lrs):
        group["lr"] = learning_rate
    scheduler._last_lr = positioned_lrs
    after = {
        "optimizer": optimizer_step_report(optimizer),
        "scheduler_last_epoch": scheduler.last_epoch,
        "scheduler_step_count": scheduler._step_count,
        "learning_rates": list(scheduler.get_last_lr()),
    }
    if after["optimizer"]["step_min"] != completed_logical_updates:
        raise RuntimeError("Adam steps did not reach the requested logical position")
    if scheduler.last_epoch != completed_logical_updates:
        raise RuntimeError("scheduler did not reach the requested logical position")
    return {"before": before, "after": after}


def _exact_statistics_equal(first: Any, replayed: Any) -> dict[str, Any]:
    tensor_fields = ("disparity_numerator", "spike_sum", "voltage_square_sum")
    count_fields = (
        "valid_pixel_count",
        "spike_batch_refinement_count",
        "voltage_element_count",
    )
    tensor_equal = {
        field: bool(torch.equal(getattr(first, field), getattr(replayed, field)))
        for field in tensor_fields
    }
    count_equal = {
        field: getattr(first, field) == getattr(replayed, field)
        for field in count_fields
    }
    maximum_differences = {
        field: float((getattr(first, field) - getattr(replayed, field)).abs().max())
        for field in tensor_fields
    }
    return {
        "tensor_exact_equal": tensor_equal,
        "count_exact_equal": count_equal,
        "tensor_max_abs_difference": maximum_differences,
        "all_exact": all(tensor_equal.values()) and all(count_equal.values()),
    }


def exact_training_update(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.OneCycleLR,
    physical_batches: Sequence[Sequence[Any]],
    train_iters: int,
    actual_update: int,
    logical_update: int,
    late_counter_synthetic: bool,
    accumulator_type: type,
    replay_loss_function: Any,
) -> dict[str, Any]:
    """Run one effective-B8 exact update and leave a clean measurement boundary."""

    from spikingjelly.activation_based import functional

    if len(physical_batches) != 4:
        raise ValueError("the fixed effective-B8 run requires four physical-B2 batches")
    started = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    probes = _parameter_probes(model)

    first_accumulator = accumulator_type()
    with torch.no_grad():
        for batch in physical_batches:
            _, left, right, truth, valid = batch
            left, right, truth, valid = [
                value.cuda(non_blocking=True) for value in (left, right, truth, valid)
            ]
            if tuple(left.shape[1:]) != (50, 250, 400):
                raise RuntimeError(f"unexpected input shape: {tuple(left.shape)}")
            functional.reset_net(model)
            predictions, updater_spikes, potentials = model(
                left, right, iters=train_iters
            )
            first_accumulator.accumulate(
                updater_spikes,
                potentials,
                predictions,
                truth,
                valid,
                [250, 400],
            )
            reset = reset_report(model)
            if not reset["clean"]:
                raise RuntimeError("no-grad exact-loss pass left stateful modules dirty")
            del predictions, updater_spikes, potentials, left, right, truth, valid
    first_stats = first_accumulator.finalize()
    if not bool(torch.isfinite(first_stats.true_loss)):
        raise FloatingPointError(f"nonfinite loss at actual update {actual_update}")

    replay_accumulator = accumulator_type()
    for batch in physical_batches:
        _, left, right, truth, valid = batch
        left, right, truth, valid = [
            value.cuda(non_blocking=True) for value in (left, right, truth, valid)
        ]
        functional.reset_net(model)
        predictions, updater_spikes, potentials = model(left, right, iters=train_iters)
        replay_accumulator.accumulate(
            updater_spikes,
            potentials,
            predictions,
            truth,
            valid,
            [250, 400],
        )
        replay_scalar = replay_loss_function(
            first_stats,
            updater_spikes,
            potentials,
            predictions,
            truth,
            valid,
            [250, 400],
        )
        if not bool(torch.isfinite(replay_scalar)):
            raise FloatingPointError(f"nonfinite replay scalar at update {actual_update}")
        replay_scalar.backward()
        reset = reset_report(model)
        if not reset["clean"]:
            raise RuntimeError("gradient replay left stateful modules dirty")
        del (
            predictions,
            updater_spikes,
            potentials,
            replay_scalar,
            left,
            right,
            truth,
            valid,
        )
    replay_stats = replay_accumulator.finalize()
    replay_equality = _exact_statistics_equal(first_stats, replay_stats)
    if not replay_equality["all_exact"]:
        raise RuntimeError("exact-loss replay statistics differ from no-grad pass")

    gradients_finite, gradient_norm, gradient_groups = gradient_report(model)
    raw_clip_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    post_finite, post_clip_norm, post_clip_groups = gradient_report(model)
    required_components = ("fnet", "cnet", "update_block", "context_zqr_convs")
    if not (
        gradients_finite
        and post_finite
        and math.isfinite(float(raw_clip_norm))
        and gradient_norm > 0.0
        and post_clip_norm > 0.0
        and all(post_clip_groups.get(component, 0.0) > 0.0 for component in required_components)
    ):
        raise FloatingPointError(f"invalid gradients at actual update {actual_update}")

    optimizer.step()
    scheduler.step()
    optimizer_steps = optimizer_step_report(optimizer)
    if optimizer_steps["step_min"] != logical_update or optimizer_steps["step_max"] != logical_update:
        raise RuntimeError("Adam logical step does not match requested update counter")
    if scheduler.last_epoch != logical_update:
        raise RuntimeError("scheduler logical step does not match requested update counter")
    del first_accumulator, replay_accumulator, replay_stats
    loss = float(first_stats.true_loss)
    del first_stats
    probe_deltas = _probe_deltas(probes)
    if not all(value["max_abs_change"] > 0.0 for value in probe_deltas.values()):
        raise RuntimeError(f"parameter probe did not change at update {actual_update}")
    del probes

    final_reset = reset_report(model)
    optimizer.zero_grad(set_to_none=True)
    if not final_reset["clean"]:
        raise RuntimeError("post-update reset was not clean")
    torch.cuda.synchronize()
    return {
        "actual_update": actual_update,
        "logical_update": logical_update,
        "late_counter_synthetic": late_counter_synthetic,
        "loss": loss,
        "seconds": time.monotonic() - started,
        "gradients_finite": gradients_finite and post_finite,
        "raw_gradient_norm": float(raw_clip_norm),
        "preclip_gradient_norm_reconstructed": gradient_norm,
        "post_clip_gradient_norm": post_clip_norm,
        "gradient_groups": gradient_groups,
        "post_clip_gradient_groups": post_clip_groups,
        "parameter_probe_deltas": probe_deltas,
        "replay_statistics": replay_equality,
        "reset_clean": final_reset["clean"],
        "optimizer_steps": optimizer_steps,
        "scheduler_last_epoch": scheduler.last_epoch,
        "scheduler_step_count": scheduler._step_count,
        "learning_rate": scheduler.get_last_lr()[0],
    }
