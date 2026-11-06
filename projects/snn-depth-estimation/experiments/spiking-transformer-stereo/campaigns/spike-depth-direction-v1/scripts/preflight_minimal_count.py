"""S8 separate zero-update native GPU preflight; never scientific training.

Only the first fixed Train B2 is decoded. Numerical warmups here do not enter
scientific training: that later job starts a new process with the untouched
historical initial-full-Dev behavior. No optimizer or scheduler step is made.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import time

import numpy as np
import torch
from stage8_controls import (CAMPAIGN, CONTRACT_SHA, MANIFEST_SHA, canonical_sha,
                             require, save, sha, validate_registration)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rng_digest():
    return {"python": hashlib.sha256(pickle.dumps(random.getstate())).hexdigest(),
            "numpy": hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest(),
            "torchCPU": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
            "torchCUDA": [hashlib.sha256(value.cpu().numpy().tobytes()).hexdigest() for value in torch.cuda.get_rng_state_all()]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "data-root", "source-registry", "registration", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get("SLURM_JOB_ID") and torch.cuda.is_available() and torch.cuda.device_count() == 1
            and "5090" in torch.cuda.get_device_name(0), "one5090SlurmGPU required")
    require(os.environ.get("SLURM_CPUS_PER_TASK") == "8" and os.environ.get("SLURM_MEM_PER_NODE") == "32768", "8CPU32GiB preflight required")
    require(not args.output.exists(), "preflight output exists; inspect without retry")
    started = time.monotonic()
    source, registration = (json.loads(path.read_text()) for path in (args.source_registry, args.registration))
    validate_registration(registration, source, "S8-preflight")
    require(sha(args.manifest) == MANIFEST_SHA and registration["optimizer_updates"] == 0, "preflight scope changed")
    import run_minimal_count as s
    require(s.CONTEXT is None, "preflight cannot write scientific initialization receipts")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.reset_peak_memory_stats()
    initialization = {}
    for seed in (1234, 2345):
        seed_all(seed)
        parent = s.BASE_BUILD("minimal_lif", seed, True)
        parent_rng = rng_digest()
        seed_all(seed)
        wrapper = s.build_model("minimal_lif", seed, True)
        wrapper_rng = rng_digest()
        first, second = s.state_receipt(parent), s.state_receipt(wrapper)
        require(first == second and parent_rng == wrapper_rng, "fresh parent/wrapper initialization or RNG mismatch")
        require(parent.state_dict().keys() == wrapper.state_dict().keys()
                and all(torch.equal(v, wrapper.state_dict()[name]) for name, v in parent.state_dict().items()),
                "parent/wrapper state tensors differ")
        initialization[str(seed)] = {**second, "parent_wrapper_all_tensors_exact": True,
                                     "parent_wrapper_postbuild_RNG_exact": True, "postbuild_RNG_sha256": wrapper_rng,
                                     "canonical_order_sha256": canonical_sha(torch.randperm(960, generator=torch.Generator().manual_seed(seed)).tolist())}
        if seed == 1234:
            first_parent, first_wrapper = parent, wrapper
        del parent, wrapper
    parent, wrapper = first_parent.cuda(), first_wrapper.cuda()
    del first_parent, first_wrapper
    seed_all(1234)
    manifest = json.loads(args.manifest.read_text())
    order = torch.randperm(960, generator=torch.Generator().manual_seed(1234)).tolist()
    indices = order[:2]
    parent_data = s.BASE_DATASET(args.data_root, manifest["train"])
    s.set_mode("raw")
    wrapped_data = s.InputDataset(args.data_root, manifest["train"])
    parent_items = [parent_data[i] for i in indices]
    raw_items = [wrapped_data[i] for i in indices]
    require(all(torch.equal(parent_items[j][k], raw_items[j][k]) for j in range(2) for k in range(4))
            and all(parent_items[j][4] == raw_items[j][4] for j in range(2)), "raw input/target/valid/index boundary changed")
    def batch(items):
        return [torch.stack([item[k] for item in items]).cuda() for k in range(4)]
    parent_left, parent_right, _, _ = batch(parent_items)
    left, right, _, _ = batch(raw_items)
    parent.eval()
    wrapper.eval()
    before = s.state_receipt(wrapper)
    require(all(initialization["1234"][key] == value for key, value in before.items()), "CPU-to-GPU initialization changed")
    # Both computations use one identical zero-update32-step warmup then reset.
    # This is engineering only, not a new warmup in the historical training path.
    with torch.no_grad():
        s.driver.forward(parent, parent_left, parent_right, 32)
        s.driver.reset_model(parent)
        reference = s.driver.forward(parent, parent_left, parent_right, 32)
        s.driver.reset_model(parent)
        s.driver.forward(wrapper, left, right, 32)
        s.driver.reset_model(wrapper)
        adapted = s.driver.forward(wrapper, left, right, 32)
        s.driver.reset_model(wrapper)
    require(torch.equal(reference, adapted), "raw wrapper prediction differs under matched numerical protocol")
    require(s.state_receipt(wrapper) == before and s.state_receipt(parent) == before, "raw preflight changed persistent tensors")
    raw_max_difference = float((reference-adapted).abs().max())
    del parent, parent_left, parent_right, left, right, reference, adapted
    s.set_mode("count_only")
    count_items = [wrapped_data[i] for i in indices]
    require(all(torch.equal(count_items[j][view].sum(0), parent_items[j][view].sum(0))
                and torch.all((count_items[j][view] == 0) | (count_items[j][view] == 1))
                for j in range(2) for view in (0, 1)), "count input count/range changed")
    require(all(torch.equal(count_items[j][k], parent_items[j][k]) for j in range(2) for k in (2, 3))
            and all(count_items[j][4] == parent_items[j][4] for j in range(2)), "count target/valid/index changed")
    left, right, target, valid = batch(count_items)
    wrapper.train()
    wrapper.freeze_bn()
    require(all(not module.training for module in wrapper.modules() if isinstance(module, torch.nn.BatchNorm2d)), "BN not frozen")
    optimizer = torch.optim.AdamW(wrapper.parameters(), lr=.0002, weight_decay=1e-5, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=.0002, total_steps=10100,
                      pct_start=.01, cycle_momentum=False, anneal_strategy="linear")
    wrapper.zero_grad(set_to_none=True)
    predictions, spikes, volts = s.driver.forward(wrapper, left, right, 16, True)
    loss = s.driver.disparity_loss(predictions, target, valid)
    require(torch.isfinite(loss), "nonfinite count preflight loss")
    loss.backward()
    gradient_norm = float(torch.nn.utils.clip_grad_norm_(wrapper.parameters(), 1.))
    require(np.isfinite(gradient_norm) and gradient_norm > 0
            and all(torch.isfinite(p.grad).all() for p in wrapper.parameters() if p.grad is not None), "invalid count backward")
    require(all(p.grad is None for p in wrapper.parameters() if not p.requires_grad), "dormant gradient present")
    require(s.state_receipt(wrapper) == before, "zero-update count backward changed parameter/buffer tensors")
    require(not optimizer.state and scheduler.last_epoch == 0, "optimizer/scheduler advanced")
    s.driver.reset_model(wrapper)
    require(all(lif.membrane is None for cell in wrapper.update_block.cells
                for lif in (cell.q_lif, cell.k_lif, cell.v_lif, cell.output_lif)), "state survived reset")
    torch.cuda.synchronize()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save(args.output, {"status": "PASS", "scientific_contract_sha256": CONTRACT_SHA,
         "source_identity_sha256": source["source_identity_sha256"], "manifest_sha256": MANIFEST_SHA,
         "registration_sha256": sha(args.registration), "historical_bindings_sha256": sha(CAMPAIGN / "configs/stage8-historical-bindings.json"),
         "initialization_checks": initialization, "native_checked_seed": 1234,
         "geometry": [2, 50, 250, 400], "indices": indices, "canonical_order_sha256": canonical_sha(order),
         "raw_input_target_valid_exact": True, "raw_parent_prediction_exact": True,
         "raw_max_abs_difference": raw_max_difference, "preflight_numerical_protocol": "one32stepwarmup+reset then32stepcomparison permodel;separateprocess",
         "count_loss_finite": float(loss.detach()), "count_gradient_norm_finite": gradient_norm,
         "count_preserves_binary_range_counts_targets_validity": True, "dormant_gradients_absent": True,
         "all_learned_and_buffer_tensors_unchanged": True, "postforward_states_reset": True,
         "optimizer_updates": 0, "scheduler_updates": 0, "scheduler_last_epoch": scheduler.last_epoch,
         "fresh_training_process_required": True, "no_training_warmup_added": True,
         "runtime_seconds": time.monotonic()-started, "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
         "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(), "optimizer_state_allocation_not_profiled": True,
         "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda,
         "Slurm_job_id": os.environ["SLURM_JOB_ID"], "no_training_release": True})


if __name__ == "__main__":
    main()
