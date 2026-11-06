"""S9 standard-library guards, ordered evidence and numeric-environment binding."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import sys

CAMP = Path(__file__).resolve().parents[1]
REPO = CAMP.parents[5]
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
CONTRACT = "217c7f7d1e0947fab351d2068577421a67c93f1936ba8612e250d58391975d2d"
PARITY = "1fb91653cce57e8c8f4264e2bbe4444cc2f3d02f856cdff837b4aa3e1c36b826"


def require(ok, message):
    if not ok:raise RuntimeError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def save(path, value):
    path = Path(path);temporary = path.with_suffix(path.suffix+".part")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def numerical_environment():
    import numpy as np
    import scipy
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):np.show_config()
    config = stream.getvalue()
    return {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
            "numpy_config_sha256": hashlib.sha256(config.encode()).hexdigest(),
            "threads": {name: os.environ.get(name) for name in THREAD_VARS}}


def ordered_calibration_binding(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    require(len(rows) == 160 and all(r["split"] == "calibration" and r["arm"] == "V" for r in rows), "ordered calibration160 required")
    expected = json.loads((CAMP/"configs/stage9-split-rows.json").read_text())["rows"]
    ids = [r["sample_id"] for r in rows]
    require(ids == [r["sample_id"] for r in expected if r["group"] == "policy_calibration"], "calibration row order changed")
    return {"calibration_jsonl_sha256": sha(path), "calibration_ordered_sample_ids_sha256": hashlib.sha256(canonical(ids).encode()).hexdigest(),
            "calibration_rows": 160, "calibration_jsonl_path": str(Path(path).resolve())}


def validate_execution(args, mode):
    # Called before importing torch/native modules; local scientific execution is forbidden.
    require(os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_JOB_PARTITION") == "debug", "registered Slurm debug allocation required")
    require(all(os.environ.get(k) == "1" for k in THREAD_VARS), "single-thread numeric environment must precede imports")
    require(os.environ.get("SLURM_CPUS_PER_TASK") == "8", "8CPU allocation required")
    config = json.loads(args.config.read_text());registration = json.loads(args.registration.read_text())
    source = json.loads(args.source_registry.read_text())
    require(config["scientific_contract_sha256"] == CONTRACT and config["parity_addendum_sha256"] == PARITY, "contract binding changed")
    require(registration["status"] == "registered" and registration["mode"] == mode, "separate root-reviewed registration required")
    require(registration["scientific_contract_sha256"] == CONTRACT and registration["parity_addendum_sha256"] == PARITY, "registered contract changed")
    require(registration["config_sha256"] == sha(args.config) and registration["source_identity_sha256"] == source["source_identity_sha256"], "configuration/source registration changed")
    resources = {"cpus": 8, "memory_gib": 32, "gpus": 1, "wall_seconds": 3600 if mode == "S9-preflight" else 28800}
    require(registration["resources"] == resources, "bounded resources changed")
    require(registration["split_sha256"] == sha(CAMP/"configs/stage9-split-rows.json") == config["split_sha256"], "split binding changed")
    require(registration["source_bindings_sha256"] == sha(CAMP/"configs/stage9-source-bindings.json"), "historical binding changed")
    require(registration["storage_config_sha256"] == sha(args.storage_config), "canonical storage metadata changed")
    binding=json.loads((CAMP/"configs/stage9-source-bindings.json").read_text())
    mandatory={str((CAMP/"scripts"/name).relative_to(REPO)) for name in
        ("stage9_native.py","stage9_actions.py","stage9_data.py","stage9_controls.py","stage9_recipe.py","preflight_stage9.py","stage9_analyze.py")}
    require(mandatory <= set(source["source_files"]), "coupled native/analyzer source absent from capsule")
    require(all(source["source_files"].get(k)==v for k,v in binding["source_files"].items()), "historical/native dependencies absent or changed")
    for relative, expected in source["source_files"].items():require(sha(REPO/relative) == expected, "frozen source changed: "+relative)
    require(not args.output.exists(), "fresh exclusive output directory required")
    args.output.mkdir(parents=True)
    return config, registration, source
