"""PREPARE-ONLY S2 runtime. Execution requires a separately registered packet.

One process/checkpoint in its exact source checkout. No optimization, trained
weight edits, long counterfactual rollout, GT-edited correlation, or Test access.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from s2_geometry import nearest_native_lattice, score_evidence, lookup_support
from s2_state_hooks import capture_native, clone, equivalent, replay_primary, snapshot


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def endpoint_gate(actual, reference):
    errors = {}
    def check(name, new, old):
        require(np.isfinite(new) and np.isfinite(old), "nonfinite endpoint")
        error = abs(new - old)
        errors[name] = error
        require(error <= 1e-6 + 1e-5 * abs(old), "native endpoint mismatch: " + name)
    require(isinstance(actual["valid_pixels"], int) and isinstance(reference["valid_pixels"], int) and
            actual["valid_pixels"] == reference["valid_pixels"], "exact endpoint valid-pixel count differs")
    errors["valid_pixels_exact_difference"] = 0
    for key in ("mean_scene_epe", "pooled_epe", "valid_fraction", "bad1", "bad3", "bad5"):
        check(key, actual[key], reference[key])
    require(actual["scene_epe"].keys() == reference["scene_epe"].keys(), "endpoint scenes differ")
    for scene, old in reference["scene_epe"].items():
        check("scene:" + scene, actual["scene_epe"][scene], old)
    require(len(actual["per_frame"]) == len(reference["per_frame"]) == 60, "endpoint frame count differs")
    for new, old in zip(actual["per_frame"], reference["per_frame"]):
        require((new["scene"], new["frame"], new["valid_pixels"]) ==
                (old["scene"], old["frame"], old["valid_pixels"]), "endpoint frame/validity differs")
        for key in ("epe", "bad1", "bad3", "bad5"):
            check(new["scene"] + ":" + new["frame"] + ":" + key, new[key], old[key])
    return errors


def frozen_builder(root, entry):
    require("core" not in sys.modules, "one checkpoint/source per fresh process")
    project = root / "projects/snn-depth-estimation"
    experiment = project / "experiments/spiking-transformer-stereo"
    sys.path[:0] = [str(project), str(project / "core"), str(experiment / "scripts")]
    path = root / entry["builder_script"]
    # importlib does not add the selected campaign directory; O/I imports its
    # exact sibling mechanism.py by name. One fresh process prevents collisions.
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("s2_exact_frozen_builder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, module.driver if hasattr(module, "driver") else module


def fine_native(prediction, geometry):
    return prediction[:, 0][:, geometry["native_y"][:, None], geometry["native_x"][None]]


def decoded(model, coordinates, output, padder):
    from core.utils.utils import coords_grid
    _, _, mask, delta = output
    delta = clone(delta)
    delta[:, 1] = 0
    base = coords_grid(coordinates.shape[0], coordinates.shape[2], coordinates.shape[3]).to(coordinates.device)
    prediction = model.upsample_disp(coordinates - base + delta, mask)
    return padder.unpad(prediction[:, :1])


def frame_fields(model, record, replays, target, valid, prediction):
    from core.utils.utils import InputPadder
    primary = record["steps"][16]
    coordinates = record["calls"][15]["coordinates"]
    b, _, h, w = coordinates.shape
    padder = InputPadder(target.shape, divis_by=32)
    require(tuple(padder._pad) == (8, 8, 3, 3), "padding changed")
    require(model.args.n_downsample == 2 and model.args.n_snn_layers == 3, "unregistered scales")
    # Verify actual main-branch strides, not merely the downsample argument.
    strides = [model.fnet.conv1.stride, model.fnet.layer1[0].conv1.stride,
               model.fnet.layer2[0].conv1.stride, model.fnet.layer3[0].conv1.stride]
    require(strides == [(1, 1), (1, 1), (2, 2), (2, 2)], "feature convolution strides changed")
    require(record["feature_maps"][0].shape[-2:] == (h, w), "feature/context lattice differs")
    geometry = nearest_native_lattice(target, valid, (h, w))
    corr = record["correlation"]
    require(corr.num_levels == 4 and corr.radius == 4, "correlation interface changed")
    require(all(level.device == coordinates.device and level.dtype == coordinates.dtype == torch.float32
                for level in corr.corr_pyramid), "support basis/native correlation device or dtype differs")
    rows = corr.corr_pyramid[0].reshape(b, h, w, w)
    evidence = score_evidence(rows, geometry["qgt"], geometry["key_valid"])
    support, taps = lookup_support(coordinates[:, 0], evidence["group"],
                                   [level.shape[-1] for level in corr.corr_pyramid[:4]])
    qgt, qbefore = geometry["qgt"], coordinates[:, 0]
    outputs = {mode: value["output"] for mode, value in replays.items()
               if mode in ("factual", "plus", "minus", "stale")}
    factual = outputs["factual"]
    delta = factual[3][:, 0]
    before_error = (qbefore - qgt).abs()
    after_error = (qbefore + delta - qgt).abs()
    reduction = before_error - after_error
    response = (outputs["plus"][3][:, 0] - outputs["minus"][3][:, 0]) / 2
    fields = {key: value for key, value in geometry.items()
              if key in ("valid", "labeled", "center_inside", "nonboundary", "qgt")}
    fields.update({key: value for key, value in evidence.items() if key != "group"})
    fields.update(support)
    fields.update(q_before=qbefore, signed_delta_q=delta, coordinate_error_before=before_error,
                  coordinate_error_after=after_error, natural_reduction=reduction, response_R=response,
                  coarse_failure=before_error > 1, weak_response=response.abs() <= .05,
                  reversed_response=response > .05, tracking_response=response < -.05,
                  stalled=reduction <= .05, materially_correcting=reduction >= .25,
                  intermediate=(reduction > .05) & (reduction < .25),
                  overshoot=(qbefore - qgt) * (qbefore + delta - qgt) < 0,
                  prior_query_difference=replays["query_coordinate_difference"][:, 0],
                  prior_corr_difference_rms=(replays["stale"]["corr"] - replays["factual"]["corr"]).square().mean(1).sqrt(),
                  query_plus_corr_difference_rms=(replays["plus"]["corr"] - replays["factual"]["corr"]).square().mean(1).sqrt(),
                  query_minus_corr_difference_rms=(replays["minus"]["corr"] - replays["factual"]["corr"]).square().mean(1).sqrt())
    # Native target is negative flow: predicted flow - target, at exact centers.
    native_target = target[:, 0][:, geometry["native_y"][:, None], geometry["native_x"][None]]
    fields["native_final_error_pixels"] = (fine_native(prediction, geometry) - native_target).abs()
    factual_prediction = decoded(model, coordinates, factual, padder)
    fields["native_step16_error_pixels"] = (fine_native(factual_prediction, geometry) - native_target).abs()
    for mode in ("plus", "minus", "stale"):
        output = outputs[mode]
        fields[mode + "_delta_q_difference"] = output[3][:, 0] - delta
        fields[mode + "_fine_spike_flip_fraction"] = (output[0][0] != factual[0][0]).float().mean(1)
        fields[mode + "_fine_membrane_difference_rms"] = (output[1] - factual[1]).square().mean(1).sqrt()
        fields[mode + "_mask_difference_rms"] = (output[2] - factual[2]).square().mean(1).sqrt()
        fields[mode + "_native_output_difference_pixels"] = fine_native(
            decoded(model, coordinates, output, padder) - factual_prediction, geometry)
    for step in (4, 8, 16, 32):
        state = record["steps"][step]
        coord = record["calls"][step - 1]["coordinates"]
        out = state["output"]
        fields[f"trajectory_{step}_coordinate_error_before"] = (coord[:, 0] - qgt).abs()
        fields[f"trajectory_{step}_signed_delta_q"] = out[3][:, 0]
        fields[f"trajectory_{step}_coordinate_error_after"] = (coord[:, 0] + out[3][:, 0] - qgt).abs()
        fields[f"trajectory_{step}_native_error_pixels"] = (fine_native(decoded(model, coord, out, padder), geometry) - native_target).abs()
    for level, tap in enumerate(taps):
        for key, value in tap.items():
            fields[f"lookup_level_{level}_{key}"] = value
    return fields


def summarize_scene(frames, subset):
    data = {key: np.concatenate([value[key].reshape(-1) for value in frames]) for key in
            ("valid", "labeled", "center_inside", "nonboundary", "score_defined", "evidence_available",
             "coarse_failure", "fine_full", "pooled_only_full", "outside_all_support", "stalled",
             "materially_correcting", "response_R", "coordinate_error_before", "coordinate_error_after", "margin",
             "row_std", "peak_unique", "peak_correct", "unrestricted_peak_padded", "natural_reduction",
             "native_step16_error_pixels", "native_final_error_pixels")}
    keep = data["valid"] & (data["nonboundary"] if subset == "nonboundary" else True)
    failure = keep & data["coarse_failure"]
    access = failure & data["evidence_available"] & data["pooled_only_full"]
    base = failure & data["evidence_available"] & data["fine_full"]
    weak_or_reversed = data["response_R"] >= -.05
    correction = base & data["stalled"] & weak_or_reversed
    cells = []
    for low, high in ((1, 2), (2, 4), (4, np.inf)):
        mask = base & (data["coordinate_error_before"] > low) & (data["coordinate_error_before"] <= high)
        stalled, successful = mask & data["stalled"], mask & data["materially_correcting"]
        enough = int(stalled.sum()) >= 32 and int(successful.sum()) >= 32
        cells.append(dict(low_exclusive=low, high_inclusive=high if np.isfinite(high) else None,
                          stalled_sites=int(stalled.sum()), correcting_sites=int(successful.sum()), eligible=enough,
                          stalled_weak_or_reversed_fraction=float(weak_or_reversed[stalled].mean()) if stalled.any() else None,
                          correcting_weak_or_reversed_fraction=float(weak_or_reversed[successful].mean()) if successful.any() else None,
                          excess=float(weak_or_reversed[stalled].mean() - weak_or_reversed[successful].mean()) if enough else None))
    eligible_cells = [cell["excess"] for cell in cells if cell["eligible"]]
    defined = keep & data["score_defined"]
    nonflat = defined & (data["row_std"] > 1e-6)
    continuous = {}
    for key in ("coordinate_error_before", "coordinate_error_after", "natural_reduction", "response_R", "margin",
                "native_step16_error_pixels", "native_final_error_pixels"):
        values = data[key][keep & np.isfinite(data[key])]
        continuous[key] = dict(mean=float(values.mean()), median=float(np.median(values))) if len(values) else None
    return dict(subset=subset, valid_sites=int(keep.sum()), padded_left_center_exclusions=int((~data["center_inside"]).sum()),
                invalid_labeled_center_exclusions=int((data["center_inside"] & ~data["labeled"]).sum()),
                right_out_of_bounds_exclusions=int((data["labeled"] & ~data["valid"]).sum()),
                undefined_score_sites=int((keep & ~data["score_defined"]).sum()), coarse_failure_sites=int(failure.sum()),
                flat_score_rows=int((defined & ~nonflat).sum()),
                wrong_or_tied_margin_rows=int((nonflat & (data["margin"] <= 0)).sum()),
                weakly_separated_rows=int((nonflat & (data["margin"] > 0) & (data["margin"] < .25)).sum()),
                separated_rows=int((nonflat & (data["margin"] >= .25)).sum()),
                primary_evidence_available_sites=int((keep & data["evidence_available"]).sum()),
                nonunique_valid_argmax_rows=int((defined & ~data["peak_unique"]).sum()),
                unrestricted_argmax_padded_rows=int((keep & data["unrestricted_peak_padded"]).sum()),
                continuous=continuous,
                access_regime_sites=int(access.sum()), correction_regime_sites=int(correction.sum()),
                access_fraction_of_coarse_failures=float(access.sum() / failure.sum()) if failure.any() else None,
                correction_bin_cells=cells,
                correction_scene_equal_bin_excess=float(np.mean(eligible_cells)) if eligible_cells else None,
                outside_all_support_failures=int((failure & data["outside_all_support"]).sum()))


def checkpoint_summary(scene_summaries):
    answer = {}
    for subset in ("all", "nonboundary"):
        records = [row for row in scene_summaries if row["subset"] == subset]
        supported_access = [r for r in records if r["access_regime_sites"] >= 128]
        supported_correction = [r for r in records if r["correction_regime_sites"] >= 128]
        comparable = [r["correction_scene_equal_bin_excess"] for r in records
                      if r["correction_scene_equal_bin_excess"] is not None]
        access_mean = float(np.mean([r["access_fraction_of_coarse_failures"] for r in supported_access])) if supported_access else None
        excess = float(np.mean(comparable)) if comparable else None
        answer[subset] = dict(access_supported_scenes=len(supported_access),
                              access_equal_supported_scene_prevalence=access_mean,
                              correction_supported_scenes=len(supported_correction),
                              correction_comparable_scenes=len(comparable), correction_equal_scene_bin_excess=excess,
                              access_checkpoint_selector=len(supported_access) >= 20 and access_mean >= .2,
                              correction_checkpoint_selector=len(supported_correction) >= 20 and len(comparable) >= 20 and excess >= .1,
                              pair_selection="requires corresponding selector in BOTH matched checkpoints; no automatic launch")
    return answer


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("intent", "checkpoint-label", "run-root", "frozen-root", "source-registry", "manifest", "data-root", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    cfg = read(args.intent)
    require(cfg["status"] == "registered", "PREPARED packet is not authorized for inference")
    for name, expected_hash in cfg["diagnostic_source_files_sha256"].items():
        require(sha(Path(__file__).parent / name) == expected_hash, "diagnostic source changed: " + name)
    require(os.environ.get("SLURM_JOB_ID") and torch.cuda.is_available(), "Slurm CUDA only")
    require(torch.cuda.device_count() == 1 and "5090" in torch.cuda.get_device_name(0), "one allocated RTX5090 required")
    entry = cfg["checkpoints"][args.checkpoint_label]
    source = read(args.source_registry)
    require(source["source_identity_sha256"] == entry["source_identity_sha256"], "wrong frozen source")
    require(hashlib.sha256(json.dumps(source["source_files"], sort_keys=True).encode()).hexdigest() ==
            source["source_identity_sha256"] and source["archive_sha256"] == entry["source_archive_sha256"],
            "source registry identity differs")
    require(source["source_files"].get(entry["builder_script"]) == entry["builder_sha256"], "builder identity differs")
    root = Path(args.frozen_root)
    for relative, expected in source["source_files"].items():
        require(sha(root / relative) == expected, "frozen source mismatch: " + relative)
    require(sha(args.manifest) == cfg["manifest_sha256"], "projected manifest mismatch")
    manifest = read(args.manifest)
    require(set(manifest) == {"schema_version", "source_manifest_sha256", "train", "dev"}, "unexpected manifest keys")
    require(manifest["source_manifest_sha256"] == cfg["historical_manifest_sha256"], "historical manifest projection differs")
    require(len(manifest["dev"]) == 60 and {(r["scene"], r["frame"]) for r in manifest["dev"]} ==
            {(f"scene_{i:04d}", f) for i in range(120, 150) for f in ("0001", "0255")}, "Dev cohort differs")
    require(all(str(r[k]).startswith("Train/") for r in manifest["dev"] for k in ("depth", "spike_l", "spike_r")), "non-Train file path")
    checkpoint_path = Path(args.run_root) / entry["checkpoint_relative"]
    require(checkpoint_path.name == "last.pt" and sha(checkpoint_path) == entry["checkpoint_sha256"], "final binary identity mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    require(checkpoint["source"]["source_identity_sha256"] == entry["source_identity_sha256"] and
            checkpoint["manifest_sha256"] == cfg["historical_manifest_sha256"] and
            checkpoint["updates"] == entry["updates"] and checkpoint["samples_seen"] == 2 * entry["updates"] and
            checkpoint["config"]["model"] == entry["model_name"] and checkpoint["config"]["seed"] == 1234 and
            checkpoint["config"]["normalize_residual_current"] == entry["normalize_residual_current"],
            "checkpoint lineage/configuration mismatch")
    expected = entry["expected_endpoint"]
    require(expected["update"] == checkpoint["updates"], "endpoint update differs")
    torch.manual_seed(1234); torch.cuda.manual_seed_all(1234); np.random.seed(1234)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = True
    builder, driver = frozen_builder(root, entry)
    model = builder.build_model(entry["model_name"], 1234, entry["normalize_residual_current"]).cuda().eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    require(model.args.n_snn_layers == 3 and model.args.n_downsample == 2 and
            model.args.corr_implementation == "reg" and model.args.corr_levels == 4 and
            model.args.corr_radius == 4 and model.args.slow_fast_gru is False and
            model.args.mixed_precision is False,
            "frozen one-correlation/one-three-scale-update-per-refinement interface differs")
    del checkpoint
    initial_weights = clone(model.state_dict())
    dataset = driver.ManifestDataset(args.data_root, manifest["dev"])
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0,
                        generator=torch.Generator().manual_seed(2225))
    output = Path(args.output); output.mkdir(parents=True, exist_ok=False)
    predictions = output / "native-endpoint-predictions"; predictions.mkdir()
    # Historical P0v2 numerical warmup: fixed first Dev batch, never a retry.
    warmup_batch = next(iter(loader))
    with torch.no_grad():
        driver.forward(model, warmup_batch[0].cuda(), warmup_batch[1].cuda(), 32)
    driver.reset_model(model)
    require(all(torch.equal(initial_weights[k], value) for k, value in model.state_dict().items()), "warmup changed weights/buffers")
    reset_memories = []
    for name, module in model.named_modules():
        if hasattr(module, "membrane"):
            require(module.membrane is None, "warmup LIF membrane not reset: " + name)
        if hasattr(module, "v"):
            require(isinstance(module.v, (float, int)) and module.v == 0,
                    "warmup ALIF membrane not reset: " + name)
            reset_memories.append(name)
    warmup = dict(frames=2, refinements=32, training=False, gradients=False,
                  learned_buffer_state_exactly_unchanged=True, reset_memories=reset_memories,
                  indices=warmup_batch[4].tolist(), excluded_from_metrics=True,
                  protocol="historical alif-root-cause-p0-v2 firstDevbatch2 numericalwarmup")
    (output / "warmup.json").write_text(json.dumps(warmup, indent=2) + "\n")
    del warmup_batch
    endpoint = driver.evaluate(model, loader, 32, predictions)
    endpoint_errors = endpoint_gate(endpoint, expected)
    require(all(torch.equal(initial_weights[k], value) for k, value in model.state_dict().items()), "native evaluation changed weights/buffers")
    (output / "endpoint-gate.json").write_text(json.dumps(dict(status="PASS", differences=endpoint_errors, endpoint=endpoint), indent=2) + "\n")
    from core.corr import CorrBlock1D
    scene_frames, replay_checks = {}, []
    site_root = output / "sites"; site_root.mkdir()
    for left, right, target, valid, indices in loader:
        require(time.monotonic() - started <= cfg["max_process_hours_per_checkpoint"] * 3600,
                "checkpoint diagnostic walltime cap reached; no partial interpretation")
        left, right, target, valid = (v.cuda() for v in (left, right, target, valid))
        with torch.no_grad(), capture_native(model, CorrBlock1D) as record:
            prediction = driver.forward(model, left, right, 32)
        instrumented_errors = []
        for batch, index in enumerate(indices.tolist()):
            row = dataset.rows[index]
            with np.load(predictions / (row["scene"] + "-" + row["frame"] + ".npz")) as native:
                require(np.array_equal(native["valid"], valid[batch].cpu().numpy()), "native validity changed")
                instrumented_errors.append(equivalent(prediction[batch, 0], torch.from_numpy(native["prediction"]).to(prediction.device)))
        with torch.no_grad():
            replays = replay_primary(model, record)
            fields = frame_fields(model, record, replays, target, valid, prediction)
        batch_checks = dict(indices=indices.tolist(), factual_max_abs=replays["factual"]["factual_equivalence_max_abs"],
                            feature_shapes=[list(x.shape) for x in record["feature_maps"]],
                            pyramid_shapes=[list(x.shape) for x in record["correlation"].corr_pyramid],
                            support_basis_native_device=str(prediction.device), support_basis_native_dtype="torch.float32",
                            native_vs_instrumented="PASS", instrumented_max_abs=max(instrumented_errors), cache_reset_restore="PASS")
        batch_checks["multiscale_probe_responses"] = {mode: [dict(
            spike_flip_fraction=float((replays[mode]["output"][0][scale] != replays["factual"]["output"][0][scale]).float().mean()),
            hidden_rms_difference=float((replays[mode]["output"][0][scale] - replays["factual"]["output"][0][scale]).square().mean().sqrt()))
            for scale in range(3)] for mode in ("plus", "minus", "stale")}
        batch_checks["neuronal_memory_responses"] = {}
        for mode in ("plus", "minus", "stale"):
            changes = {}
            for name, values in replays["factual"]["state"].items():
                for field in ("_memories", "membrane", "v_threshold"):
                    if field not in values:
                        continue
                    before = values[field]
                    after = replays[mode]["state"][name][field]
                    pairs = before.items() if isinstance(before, dict) else [(field, before)]
                    for memory_name, value in pairs:
                        other = after[memory_name] if isinstance(before, dict) else after
                        if isinstance(value, torch.Tensor) and isinstance(other, torch.Tensor):
                            changes[name + ":" + memory_name] = float((other - value).square().mean().sqrt())
            batch_checks["neuronal_memory_responses"][mode] = changes
        replay_checks.append(batch_checks)
        for batch, index in enumerate(indices.tolist()):
            row = dataset.rows[index]
            arrays = {key: value[batch].detach().cpu().numpy() for key, value in fields.items()}
            np.savez_compressed(site_root / (row["scene"] + "-" + row["frame"] + ".npz"), **arrays)
            scene_frames.setdefault(row["scene"], []).append(arrays)
        driver.reset_model(model)
        require(all(torch.equal(initial_weights[k], value) for k, value in model.state_dict().items()), "instrumentation changed weights/buffers")
        del record, replays, fields, prediction, left, right, target, valid
    scenes = [dict(scene=scene, **summarize_scene(frames, subset)) for scene, frames in sorted(scene_frames.items())
              for subset in ("all", "nonboundary")]
    result = dict(status="PREPARED_PACKET_EXECUTION_COMPLETE_NO_CAUSAL_OR_MOTION_CLAIM", checkpoint_label=args.checkpoint_label,
                  checkpoint_sha256=entry["checkpoint_sha256"], checkpoint_source_identity_sha256=source["source_identity_sha256"],
                  intent_sha256=sha(args.intent), manifest_sha256=sha(args.manifest),
                  endpoint_gate="PASS", endpoint_errors=endpoint_errors, replay_checks=replay_checks,
                  warmup=warmup,
                  scene_summaries=scenes, checkpoint_selectors=checkpoint_summary(scenes),
                  optimizer_updates=0, weights_and_buffers_unchanged=True, no_Test_access=True,
                  runtime_seconds=time.monotonic() - started, GPU_hours=(time.monotonic() - started) / 3600,
                  job_id=os.environ["SLURM_JOB_ID"], GPU=torch.cuda.get_device_name(0),
                  no_counterfactual_accuracy_evaluation=True, primary_pre_refinement=16, secondary_refinements=[4, 8, 32],
                  limits="labelled in-bounds not certified nonoccluded; local inconsistent-query probes; distinct trained encoders; neither significance nor causation")
    (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
