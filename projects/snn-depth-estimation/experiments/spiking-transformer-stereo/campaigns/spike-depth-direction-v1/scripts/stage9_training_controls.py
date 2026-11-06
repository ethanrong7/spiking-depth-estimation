"""Separate S9 training release; exact preflight component reuse, no fallback."""
import json
from pathlib import Path
from stage9_controls import CAMP, REPO, require, sha, validate_execution


def validate_training(args):
    config,registration,source=validate_execution(args,"S9-training")
    expected={"updates":5000,"presentations":10000,"seed":1234,"physical_batch":1,"effective_batch":2,
        "schedule_steps":5100,"lr":.0002,"weight_decay":.00001,"epsilon":1e-8,"clip_norm":1,
        "train_refinements":16,"loss_endpoints":[4,8,16],"neural_eval_updates":[0,2500,5000],"primary_update":5000,
        "final_extraction":True,"restore_checkpoint":None,"extra_warmup":False}
    require(all(config.get(k)==v for k,v in expected.items()) and config["arm"] in ("V","F"),"unauthorized S9 recipe")
    require(registration["root_released_arm"]==config["arm"] and registration["scientific_optimizer_updates"]==5000,
            "one explicitly released arm required")
    for path,expected_sha in registration["preflight_input_sha256"].items():require(sha(path)==expected_sha,"preflight/independent acceptance changed")
    preflight=json.loads(Path(registration["preflight_receipt"]).read_text())
    audit=json.loads(Path(registration["preflight_audit_receipt"]).read_text())
    acceptance=json.loads(Path(registration["root_actual_preflight_acceptance"]).read_text())
    preflight_source=json.loads(Path(registration["preflight_source_registry"]).read_text())
    require(preflight["status"]=="PASS" and preflight["scientific_optimizer_updates"]==0
            and preflight["scientific_scheduler_updates"]==0 and preflight["scientific_tensors_unchanged"],"passing zero-update native preflight required")
    require(audit["status"].startswith("PASS") and acceptance["training_packet_preparation_authorized"],"independent audit/root actual preflight acceptance required")
    require(preflight_source["source_identity_sha256"]==preflight["source_identity_sha256"]==registration["preflight_source_identity_sha256"],"preflight source identity mismatch")
    # All50 previously checked files retain exact bytes, not just selected models.
    require(all(source["source_files"].get(k)==v and sha(REPO/k)==v for k,v in preflight_source["source_files"].items()),"preflight-tested member changed")
    require(preflight["split_sha256"]==config["split_sha256"] and preflight["T_gpu_ms"]==registration["T_gpu_ms"]>0,
            "fixed split/T changed")
    require(preflight["forecast"]["fits_per_job_and_global_caps"],"fixed native forecast failed")
    return config,registration,source,preflight
