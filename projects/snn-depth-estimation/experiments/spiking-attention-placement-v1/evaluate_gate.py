"""Compare an official placement evaluation with the two matched reference runs."""
import argparse
import json
import math
from pathlib import Path
from statistics import mean


def endpoint(path, update):
    with path.open() as records:
        rows = [row for line in records if line.endswith("\n")
                and line.strip() and (row := json.loads(line))["update"] == update]
    if len(rows) != 1:
        raise ValueError(f"{path}: expected one complete record at {update}")
    row = rows[0]
    frames = row["per_frame"]
    support = {(f["scene"], f["frame"]): f["valid_pixels"] for f in frames}
    cohort = {s for s, _ in support}
    if (row["samples_seen"] != 8 * update or len(support) != 60 or len(frames) != 60
            or len(cohort) != 30 or cohort != set(row["scene_epe"])
            or any(n <= 0 for n in support.values())):
        raise ValueError(f"{path}: incomplete cohort or incorrect training exposure")
    scenes = {}
    for scene in sorted(cohort):
        errors = [f["epe"] for f in frames if f["scene"] == scene]
        if len(errors) != 2 or not math.isclose(mean(errors), row["scene_epe"][scene], abs_tol=1e-6):
            raise ValueError(f"{path}: inconsistent scene {scene}")
        scenes[scene] = mean(errors)
    if (not math.isclose(mean(scenes.values()), row["mean_scene_epe"], abs_tol=1e-6)
            or not all(math.isfinite(v) and v >= 0 for v in scenes.values())
            or not 0 <= row["bad3"] <= 100):
        raise ValueError(f"{path}: inconsistent evaluation")
    return row, support, scenes


def compare(candidate, original, v2b, update):
    result = {"update": update, "comparisons": {}}
    row, cohort, scenes = endpoint(candidate, update)
    result.update(mean_scene_epe=mean(scenes.values()), bad3=row["bad3"])
    for label, path in (("original", original), ("s2m2_v2b", v2b)):
        ref, ref_cohort, ref_scenes = endpoint(path, update)
        if cohort != ref_cohort:
            raise ValueError(f"{path}: candidate frame IDs or valid-pixel counts differ")
        deltas = {s: scenes[s] - ref_scenes[s] for s in sorted(scenes)}
        result["comparisons"][label] = {
            "mean_epe_delta": mean(deltas.values()), "scene_wins": sum(d < 0 for d in deltas.values()),
            "scene_deltas": deltas, "bad3_delta_pp": row["bad3"] - ref["bad3"]}
    if update == 7500:
        promising = result["comparisons"]["s2m2_v2b"]["mean_epe_delta"] <= 0
        result["decision"] = "continue_same_state_to_30000" if promising else "stop"
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in ("candidate", "original", "v2b"):
        parser.add_argument("--" + arm, type=Path, required=True)
    parser.add_argument("--update", type=int, choices=(7500, 30000), required=True)
    args = parser.parse_args()
    print(json.dumps(compare(**vars(args)), indent=2, allow_nan=False))
