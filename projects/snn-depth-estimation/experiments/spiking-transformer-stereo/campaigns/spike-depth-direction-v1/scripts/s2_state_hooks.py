"""Temporary native hooks and exact neuronal-state caches; no weight edits."""
from contextlib import contextmanager
from copy import deepcopy
from unittest.mock import patch
import torch


def clone(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: clone(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(clone(v) for v in value)
    return deepcopy(value)


def snapshot(model):
    result = {}
    for name, module in model.named_modules():
        values = {}
        for key in ("_memories", "membrane", "_diagnostics", "v_threshold"):
            if hasattr(module, key):
                values[key] = clone(getattr(module, key))
        if values:
            result[name] = values
    return result


def restore(model, saved):
    modules = dict(model.named_modules())
    for name, values in saved.items():
        for key, value in values.items():
            setattr(modules[name], key, clone(value))


def equivalent(a, b, binary=False):
    """Return maximum float difference; fail on shape, masks or binary changes."""
    if isinstance(a, torch.Tensor):
        if not isinstance(b, torch.Tensor) or a.shape != b.shape or a.dtype != b.dtype:
            raise RuntimeError("replay tensor shape/dtype mismatch")
        if not torch.equal(torch.isfinite(a), torch.isfinite(b)):
            raise RuntimeError("replay finite pattern mismatch")
        finite = torch.isfinite(a)
        if binary or not a.is_floating_point() or (bool(((a == 0) | (a == 1)).all()) and bool(((b == 0) | (b == 1)).all())):
            if not torch.equal(a, b):
                raise RuntimeError("hard spike/discrete state mismatch")
        elif not torch.allclose(a, b, atol=1e-6, rtol=1e-5, equal_nan=True):
            raise RuntimeError("floating replay mismatch")
        return float((a[finite].float() - b[finite].float()).abs().max()) if finite.any() else 0.
    if isinstance(a, dict):
        if set(a) != set(b):
            raise RuntimeError("state keys mismatch")
        return max((equivalent(a[k], b[k]) for k in a), default=0.)
    if isinstance(a, (tuple, list)):
        if type(a) != type(b) or len(a) != len(b):
            raise RuntimeError("state sequence mismatch")
        return max((equivalent(x, y) for x, y in zip(a, b)), default=0.)
    if a != b:
        raise RuntimeError("scalar state mismatch")
    return 0.


@contextmanager
def capture_native(model, corr_class, steps=(4, 8, 16, 32)):
    """Original correlation init/call and updater execute exactly once per step.

    Temporary class-call wrappers only observe return values; model.forward and
    updater operators remain native. Handles/wrappers are restored on exceptions.
    """
    record = dict(calls=[], steps={}, feature_maps=None, correlation=None, updater_calls=0)
    original_init, original_call = corr_class.__init__, corr_class.__call__
    def init(obj, left, right, *args, **kwargs):
        original_init(obj, left, right, *args, **kwargs)
        if record["correlation"] is not None:
            raise RuntimeError("unexpected second correlation object")
        record["correlation"] = obj
        record["feature_maps"] = (clone(left), clone(right))
    def call(obj, coordinates):
        result = original_call(obj, coordinates)
        record["calls"].append(dict(coordinates=clone(coordinates), vector=clone(result)))
        return result
    def before(module, args, kwargs):
        record["updater_calls"] += 1
        step = len(record["calls"])
        if step != record["updater_calls"] or any(not kwargs.get(flag, True) for flag in
                                                   ("iter08", "iter16", "iter32", "update")):
            raise RuntimeError("requires one full three-scale update per native lookup")
        if step in steps:
            record["steps"][step] = dict(arguments=clone(args), kwargs=clone(kwargs), before=snapshot(model))
    def after(module, args, kwargs, output):
        step = len(record["calls"])
        if step in steps:
            record["steps"][step].update(output=clone(output), after=snapshot(model))
    pre = model.update_block.register_forward_pre_hook(before, with_kwargs=True)
    post = model.update_block.register_forward_hook(after, with_kwargs=True)
    try:
        with patch.object(corr_class, "__init__", init), patch.object(corr_class, "__call__", call):
            yield record
        if len(record["calls"]) != 32 or record["updater_calls"] != 32 or set(record["steps"]) != set(steps):
            raise RuntimeError("native32 rollout structure mismatch")
    finally:
        pre.remove()
        post.remove()


def replay_primary(model, record):
    """One factual, ±1-query and preceding-vector replay at cached pre16 state."""
    saved = record["steps"][16]
    native_after_rollout = snapshot(model)
    corr_object = record["correlation"]
    coordinates = record["calls"][15]["coordinates"]
    results = {}
    try:
        for mode in ("factual", "plus", "minus", "stale"):
            restore(model, saved["before"])
            args = clone(saved["arguments"])
            if mode in ("plus", "minus"):
                shifted = clone(coordinates)
                shifted[:, 0] += 1 if mode == "plus" else -1
                args = (*args[:2], corr_object(shifted), *args[3:])
            elif mode == "stale":
                args = (*args[:2], clone(record["calls"][14]["vector"]), *args[3:])
            output = model.update_block(*args, **clone(saved.get("kwargs", {})))
            state = snapshot(model)
            if mode == "factual":
                error = max(equivalent(output, saved["output"]), equivalent(state, saved["after"]))
            else:
                error = None
            results[mode] = dict(output=clone(output), state=state, corr=clone(args[2]),
                                 factual_equivalence_max_abs=error)
    finally:
        restore(model, native_after_rollout)
        equivalent(snapshot(model), native_after_rollout)
    results["query_coordinate_difference"] = clone(coordinates - record["calls"][14]["coordinates"])
    return results
