"""Small source/registration guards for S8; no model or dataset imports."""
import hashlib
import json
from pathlib import Path

CAMPAIGN = Path(__file__).resolve().parents[1]
REPO = CAMPAIGN.parents[5]
CONTRACT_SHA = "b1fcc7024a246de78be36b96ff69274f03d0c6e598b47f12e5f85c347551dfbc"
HISTORICAL_ID = "969e210b3d23e9cb0b33c3cb553e3c0a51d8e95af7be1a4fecc9ab5748f41cc9"
MANIFEST_SHA = "d8f07b9dc3a17fcc640777e71d55884f17fb116d29fc5e8c9a23c1892f181d71"
HELPER_SHA = "d20ecdd3d5c309f113fae1dab96493977de6cd2b22d09c37fa613825808a7ca0"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def bindings():
    path = CAMPAIGN / "configs/stage8-historical-bindings.json"
    value = json.loads(path.read_text())
    require(value["scientific_contract_sha256"] == CONTRACT_SHA and value["historical_minimal_source_identity_sha256"] == HISTORICAL_ID,
            "S8 historical identity changed")
    require(sha(CAMPAIGN / "analysis/stage8-minimal-count-contract-astra.md") == CONTRACT_SHA, "S8 contract changed")
    require(value["S1_input_helper_sha256"] == sha(CAMPAIGN / "scripts/input_codes.py") == HELPER_SHA, "S1 helper changed")
    core = {k: v for k, v in value["historical_source_files"].items() if "/core/" in k}
    require(len(core) == 15 and len(value["historical_source_files"]) == 20, "historical dependency census changed")
    for relative, expected in value["historical_source_files"].items():
        require(sha(REPO / relative) == expected, "historical source changed: " + relative)
    manifest = CAMPAIGN / "configs/train-dev-manifest.json"
    require(sha(manifest) == MANIFEST_SHA == value["manifest_projection"]["safe_projection_sha256"], "safe manifest changed")
    rows = json.loads(manifest.read_text())
    for split, count in (("train", 960), ("dev", 60)):
        receipt = value["manifest_projection"]["row_equality"][split]
        require(len(rows[split]) == receipt["count"] == count and receipt["equal_in_order"]
                and canonical_sha(rows[split]) == receipt["canonical_rows_sha256"], "Train/Dev row equality changed")
    return value


def validate_config(config):
    fixed = {"campaign_id": "spike-depth-direction-v1-S8", "stage": "paired_training", "model": "minimal_lif",
             "normalize_residual_current": True, "updates": 10000, "schedule_updates": 10000, "eval_every": 500,
             "train_split": "train", "eval_split": "dev", "physical_batch": 2, "effective_batch": 2,
             "train_refinements": 16, "eval_refinements": 32, "lr": .0002, "final_extras": False, "input_code": "count_only"}
    require(all(config.get(k) == v for k, v in fixed.items()) and config.get("seed") in (1234, 2345), "unauthorized S8 recipe")


def validate_registration(registration, source, mode):
    require(registration["status"] == "registered" and registration["mode"] == mode, "S8 separate registration required")
    require(registration["scientific_contract_sha256"] == CONTRACT_SHA
            and registration["source_identity_sha256"] == source["source_identity_sha256"], "S8 contract/source registration mismatch")
    require(registration["historical_bindings_sha256"] == sha(CAMPAIGN / "configs/stage8-historical-bindings.json")
            and registration["manifest_sha256"] == MANIFEST_SHA, "historical/data registration mismatch")
    resources = {"cpus": 8, "memory_gib": 32, "gpus": 1, "wall_seconds": 3600 if mode == "S8-preflight" else 14400}
    require(registration["resources"] == resources, "S8 bounded resources changed")
    for relative, expected in source["source_files"].items():
        require(sha(REPO / relative) == expected, "frozen S8 source changed: " + relative)
    history = bindings()
    require(all(r["final_checkpoint"].get("sha256") for r in history["historical_references"]), "historical checkpoint identity unresolved")
    require(source["source_override"]["parent_source_identity_sha256"] == HISTORICAL_ID, "historical minimal parent mismatch")
    actual_core = {k: v for k, v in source["source_files"].items() if "/core/" in k}
    expected_core = {k: v for k, v in history["historical_source_files"].items() if "/core/" in k}
    require(actual_core == expected_core, "15 core bytes changed")
    return history


def validate_training_release(config, registration, source):
    validate_config(config)
    validate_registration(registration, source, "S8-training")
    require(registration["root_released_seed"] == config["seed"] and len(registration["runs"]) == 1,
            "only one explicitly released seed may execute")
    if config["seed"] == 2345:
        require(registration["seed1234_audit_passed"] and registration["seed1234_P_or_T_passed"]
                and sha(Path(registration["seed1234_audit_receipt"])) == registration["seed1234_audit_sha256"], "second seed remains conditional")
    path = Path(registration["preflight_receipt"])
    require(sha(path) == registration["preflight_receipt_sha256"], "preflight receipt changed")
    preflight = json.loads(path.read_text())
    require(preflight["status"] == "PASS" and preflight["optimizer_updates"] == 0
            and preflight["source_identity_sha256"] == source["source_identity_sha256"]
            and preflight["scientific_contract_sha256"] == CONTRACT_SHA
            and preflight["manifest_sha256"] == MANIFEST_SHA
            and str(config["seed"]) in preflight["initialization_checks"], "bound zero-update preflight required")

    return preflight
