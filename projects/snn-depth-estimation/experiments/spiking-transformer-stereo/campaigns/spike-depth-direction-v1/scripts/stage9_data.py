"""Certified FT split and unchanged audited native decoder; no RGB/model access."""
import importlib.util
import json
from pathlib import Path
import sys
import time
import torch
from stage9_native import CAMP, EXPERIMENT, BINDINGS, guard_sources, require, sha, visible_input

path = EXPERIMENT / "campaigns/native-input-pilot-v1/scripts/native_data.py"
spec = importlib.util.spec_from_file_location("s9_audited_native_data", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def split_rows(group):
    data = json.loads((CAMP / "configs/stage9-split-rows.json").read_text())
    rows = [r for r in data["rows"] if r["group"] == group]
    require(len(rows) == {"model_fit": 800, "policy_calibration": 160, "dev": 60}[group], "certified cohort changed")
    return rows


class NativeData:
    def __init__(self, storage_path, group):
        storage = json.loads(Path(storage_path).read_text())
        self.source, self.derived = Path(storage["source_root"]), Path(storage["derived_root"])
        legacy = Path(storage["legacy_source_symlink"])
        require(not legacy.exists() and not legacy.is_symlink(), "removed source alias must remain absent")
        require(self.source.name == "source-data" and self.derived.name == "derived"
                and self.source.parent == self.derived.parent, "canonical FT storage required")
        self.rows = split_rows(group)
        native_rows = [{"sample_id": r["sample_id"], "source_id": r["source_id"],
                        **{k: r["files"][k]["path"] for k in ("spike_l", "spike_r", "disparity_l")}} for r in self.rows]
        self.base = module.NativeDataset(self.source, self.derived, native_rows, "original")
        self.verified = set()

    def __len__(self):return len(self.rows)

    def __getitem__(self, index):
        begun = time.perf_counter();row = self.rows[index]
        if index not in self.verified:
            # Check only the3 inputs actually consumed, once per job/sample.
            # No RGB/extra label payloads are read and no files are modified.
            for key in ("spike_l", "spike_r", "disparity_l"):
                root = self.derived if key.startswith("spike") else self.source
                require(sha(root / row["files"][key]["path"]) == row["files"][key]["sha256"], "certified input hash changed")
            self.verified.add(index)
        item = self.base[index]
        require(item["left"].shape == item["right"].shape == (50, 540, 960)
                and item["target"].shape == (1, 540, 960), "native geometry changed")
        item.update(row=row, manifest_ordinal=row["manifest_ordinal"], I_O_decode_ms=1000*(time.perf_counter()-begun))
        return item


def device_pairs(item, prefixes=(24, 32, 49)):
    begun = time.perf_counter();result = {};cpu25 = None
    for p in prefixes:
        pair = tuple(visible_input(item[side], p) for side in ("left", "right"))
        if p == 24:cpu25 = pair
        result[p] = tuple(x[None].cuda() for x in pair)
    torch.cuda.synchronize()
    item["host_device_transfer_ms"] = 1000*(time.perf_counter()-begun)
    item["host_device_transfer_scope"] = "CPU masking and all requested masked supports preloaded; separate from device budget and not live acquisition latency"
    return result, cpu25
