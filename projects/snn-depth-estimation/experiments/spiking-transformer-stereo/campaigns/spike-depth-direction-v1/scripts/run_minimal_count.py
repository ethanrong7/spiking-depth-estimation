"""S8 input-only adapter around the exact historical minimal-LIF training driver.

No generated driver source, S1 original-I wrapper, added training warmup, model
rewrite or checkpoint initialization. Only binary CPU input coding and compact
provenance receipts differ. Historical initial full Dev evaluation, every500
Dev evaluations, RNG, frozen BN, loss/optimizer/schedule remain in the parent.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

from stage8_controls import (CAMPAIGN, CONTRACT_SHA, HISTORICAL_ID, MANIFEST_SHA,
                             bindings, canonical_sha, require, save, sha,
                             validate_training_release)

HISTORY = bindings()  # guard all20 dependencies including15 core files before importing them
EXPERIMENT = CAMPAIGN.parents[1]
PARENT_PATH = EXPERIMENT / "campaigns/minimal-lif-v1/scripts/run_minimal.py"
sys.path.insert(0, str(PARENT_PATH.parent))
spec = importlib.util.spec_from_file_location("s8_historical_minimal", PARENT_PATH)
historical = importlib.util.module_from_spec(spec)
spec.loader.exec_module(historical)
driver = historical.driver
BASE_BUILD = historical.build_model
BASE_DATASET = driver.ManifestDataset
BASE_CONTRACT = driver.training_contract
BASE_BUDGET = driver.validate_execution_budget
from input_codes import transform_stereo
import torch

MODE = None
CONTEXT = None
EXPECTED_INITIAL = None


def set_mode(mode):
    global MODE
    require(mode in ("raw", "count_only"), "unknown input code")
    MODE = mode


class InputDataset(BASE_DATASET):
    def __getitem__(self, index):
        require(MODE is not None, "input code must be bound")
        left, right, target, valid, idx = super().__getitem__(index)
        left, right = transform_stereo(left[None], right[None], MODE)
        return left[0], right[0], target, valid, idx


def digest_named(items):
    import hashlib
    digest = hashlib.sha256()
    for name, value in sorted(items):
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def state_receipt(model):
    parameters = list(model.named_parameters())
    active = [(n, p) for n, p in parameters if p.requires_grad]
    dormant = [(n, p) for n, p in parameters if not p.requires_grad]
    counts = (sum(p.numel() for _, p in parameters), sum(p.numel() for _, p in active), sum(p.numel() for _, p in dormant))
    require(counts == (8398902, 8150544, 248358), "historical active/stored/dormant count changed")
    return {"all_state_sha256": digest_named(model.state_dict().items()),
            "active_parameters_sha256": digest_named(active), "buffers_sha256": digest_named(model.named_buffers()),
            "dormant_parameters_sha256": digest_named(dormant),
            "total_parameters": counts[0], "active_parameters": counts[1], "dormant_parameters": counts[2],
            "active_parameter_names": sorted(n for n, _ in active), "dormant_parameter_names": sorted(n for n, _ in dormant)}


def build_model(name, seed, normalize_residual_current=False):
    require(name == "minimal_lif" and normalize_residual_current, "exact historical minimal builder required")
    model = BASE_BUILD(name, seed, True)  # preserves reference seed,seed+777 and common-state loading
    if CONTEXT is not None:
        receipt = state_receipt(model)
        require(EXPECTED_INITIAL is not None and all(EXPECTED_INITIAL[key] == value for key, value in receipt.items()),
                "scientific initialization differs from separate bound preflight")
        receipt.update(seed=seed, historical_source_identity_sha256=HISTORICAL_ID,
                       input_code=MODE, trained_checkpoint_restored=False,
                       scientific_warmup="none; exact parent initial full60Dev evaluation follows",
                       historical_active_initial_binary_available=False)
        save(CONTEXT / "initialization-control.json", receipt)
        order = torch.randperm(960, generator=torch.Generator().manual_seed(seed)).tolist()
        require(canonical_sha(order) == EXPECTED_INITIAL["canonical_order_sha256"], "seeded order differs from preflight")
        save(CONTEXT / "sample-order-control.json", {"seed": seed, "order": order,
             "canonical_order_sha256": canonical_sha(order), "generator": "independentCPUtorch.Generator.manual_seed(seed)",
             "parent_actual_order": "same untouched parent expression; verify saved checkpoints against this receipt",
             "cyclic_batch_indices": "order[(2*step+i)%960], i=0,1", "steps_per_full_cycle": 480,
             "steps": 10000, "presentations": 20000, "Dev_loader_seed": seed+991,
             "TrainDev_manifest_sha256": MANIFEST_SHA})
    return model


def training_contract(config, *args, **kwargs):
    contract = BASE_CONTRACT(config, *args, **kwargs)
    contract["S8_information_screen"] = {"scientific_contract_sha256": CONTRACT_SHA,
        "historical_parent_source_identity_sha256": HISTORICAL_ID,
        "historical_bindings_sha256": sha(CAMPAIGN / "configs/stage8-historical-bindings.json"),
        "input_code": config["input_code"], "transform_location": "datasetCPU_beforecuda_padding_and_native2Sminus1",
        "count_formula": "((t+1)*C)//50-(t*C)//50; C=sum_tS, all50bins",
        "all_training_and_evaluation_inputs_transformed": True,
        "initial_evaluation": "unchangedparent full60Dev32; no added warmup",
        "evaluation_updates": list(range(500, 10001, 500)), "primary": "final10000/history_and_last.pt_not_selected_best",
        "historical_initial_state_limit": HISTORY["initialization_limit"],
        "interpretation": "information/encodingpackage; artificial brightness timing/synchrony/conditioning; no motion/SNN/energy causal claim"}
    return contract


def validate_budget(config, registration, manifest_sha, source, has_checkpoint=False):
    global EXPECTED_INITIAL
    preflight = validate_training_release(config, registration, source)
    EXPECTED_INITIAL = preflight["initialization_checks"][str(config["seed"])]
    BASE_BUDGET(config, registration, manifest_sha, source, has_checkpoint)
    require(not has_checkpoint and MODE == "count_only" and manifest_sha == MANIFEST_SHA, "fresh count-only training required")


driver.ManifestDataset = InputDataset
driver.build_model = build_model
driver.training_contract = training_contract
driver.validate_execution_budget = validate_budget


def main():
    global CONTEXT
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args, _ = parser.parse_known_args()
    config = json.loads(args.config.read_text())
    set_mode(config["input_code"])
    CONTEXT = args.output
    # The unmodified historical main owns RNG initialization, model/device/order,
    # initial evaluation, optimization, reset, schedule and checkpoint semantics.
    driver.main()


if __name__ == "__main__":
    main()
