#!/usr/bin/env python3
import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))

from core.spikestereonet import SpikeStereoNet
import torch.nn as nn


VARIANTS = {
    "A0": ("cnn", "cnn"),
    "A1": ("sew", "cnn"),
    "A2": ("cnn", "sew"),
    "A3": ("sew", "sew"),
}


def model_args(feature, context, context_norm):
    return SimpleNamespace(
        hidden_dims=[128, 128, 128],
        context_norm=context_norm,
        n_downsample=2,
        n_snn_layers=3,
        shared_backbone=False,
        feature_encoder=feature,
        context_encoder=context,
        sew_checkpoint_chunk_size=0,
        matched_initialization_seed=1234,
        corr_levels=4,
        corr_radius=4,
        mixed_precision=False,
        corr_implementation="alt",
    )


def describe(variant, context_norm):
    feature, context = VARIANTS[variant]
    model = SpikeStereoNet(model_args(feature, context, context_norm))
    groups = defaultdict(int)
    tensors = defaultdict(int)
    for name, parameter in model.named_parameters():
        group = name.split(".", 1)[0]
        groups[group] += parameter.numel()
        tensors[group] += 1
    total = sum(groups.values())
    normalization_affine = sum(
        parameter.numel()
        for module in model.modules()
        if isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm))
        for parameter in module.parameters(recurse=False)
    )
    component_hashes = {}
    for component in ("fnet", "cnet", "update_block", "context_zqr_convs"):
        digest = hashlib.sha256()
        for name, value in sorted(getattr(model, component).state_dict().items()):
            digest.update(name.encode())
            digest.update(str(value.dtype).encode())
            digest.update(str(tuple(value.shape)).encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        component_hashes[component] = digest.hexdigest()
    return {
        "variant": variant,
        "feature_encoder": feature,
        "context_encoder": context,
        "context_norm": context_norm,
        "total_parameters": total,
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "used_parameters": total,
        "unused_parameters": 0,
        "normalization_affine_parameters": normalization_affine,
        "component_parameters": dict(sorted(groups.items())),
        "component_parameter_tensors": dict(sorted(tensors.items())),
        "feature_stem_shape": list(model.fnet.conv1.weight.shape),
        "context_stem_shape": list(model.cnet.conv1.weight.shape),
        "component_state_hashes": component_hashes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    normalization_sets = {}
    for context_norm in ("batch", "instance"):
        variants = [describe(variant, context_norm) for variant in VARIANTS]
        by_name = {variant["variant"]: variant for variant in variants}
        comparisons = {
            "cnn_feature_A0_A2": by_name["A0"]["component_state_hashes"]["fnet"]
            == by_name["A2"]["component_state_hashes"]["fnet"],
            "sew_feature_A1_A3": by_name["A1"]["component_state_hashes"]["fnet"]
            == by_name["A3"]["component_state_hashes"]["fnet"],
            "cnn_context_A0_A1": by_name["A0"]["component_state_hashes"]["cnet"]
            == by_name["A1"]["component_state_hashes"]["cnet"],
            "sew_context_A2_A3": by_name["A2"]["component_state_hashes"]["cnet"]
            == by_name["A3"]["component_state_hashes"]["cnet"],
            "update_block_all_variants": len(
                {
                    variant["component_state_hashes"]["update_block"]
                    for variant in variants
                }
            )
            == 1,
            "context_zqr_all_variants": len(
                {
                    variant["component_state_hashes"]["context_zqr_convs"]
                    for variant in variants
                }
            )
            == 1,
        }
        normalization_sets[context_norm] = {
            "matched_state_checks": comparisons,
            "variants": variants,
        }
    report = {
        "input_bins": 50,
        "sew": {
            "residual": "SEW-ADD",
            "tau": 2.0,
            "decay_input": True,
            "threshold": 1.0,
            "reset": "hard to 0; detach_reset=False",
            "surrogate": "ATan alpha=2",
            "temporal_reduction": "mean of final dense projection over 50 ordered steps",
        },
        "matched_initialization_seed_offsets": {
            "cnet": 101,
            "update_block": 202,
            "context_zqr_convs": 303,
            "fnet": 404,
        },
        "normalization_change": {
            "batch": "BatchNorm2d affine=True with running mean/variance; frozen for training",
            "instance": "InstanceNorm2d affine=False with no running statistics",
        },
        "normalization_sets": normalization_sets,
    }
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
