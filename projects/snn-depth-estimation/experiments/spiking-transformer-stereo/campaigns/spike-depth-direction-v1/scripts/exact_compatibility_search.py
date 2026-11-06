"""S6: one fixed exact analytic search; no dataset or learned dependencies.

Meaningful enumeration is permitted only by the registered CPU Slurm entry.
Tiny tests exercise named examples and rational boundaries, never the lattice.
"""
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction as F
from pathlib import Path
import argparse
import datetime
import hashlib
import io
import json
import os
import resource
import socket
import sys
import time
import unittest

Q, N, TARGET = 5, 50, 24
CONTRACT_SHA = "afe1ffc17d40c6bd1c98b34b5957f134cf1b4da8d562501b58d95dc4fdbf1851"
LD_LIMIT = R_LIMIT = 2_000_000
SIGNATURE_FIELDS = ("count50", "first", "last", "a2", "a1", "b1", "b2",
                    "count20_28", "count16_32", "count12_36", "count0_48")
CAMPAIGN = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rational(value):
    value = F(value)
    return {"numerator": value.numerator, "denominator": value.denominator}


@dataclass(frozen=True)
class Interval:
    lower: F
    upper: F
    lower_closed: bool
    upper_closed: bool

    def nonempty(self):
        return self.lower < self.upper or (self.lower == self.upper and self.lower_closed and self.upper_closed)

    def contains(self, value):
        return ((value > self.lower or (value == self.lower and self.lower_closed))
                and (value < self.upper or (value == self.upper and self.upper_closed)))

    def payload(self):
        return {"lower": {**rational(self.lower), "closed": self.lower_closed},
                "upper": {**rational(self.upper), "closed": self.upper_closed}, "nonempty": self.nonempty()}


def encode(intensity_numerator, phase_numerator):
    """Exact hundredth-unit generator, threshold500, retaining overshoot."""
    require(isinstance(intensity_numerator, int) and 0 <= intensity_numerator <= 100, "intensity numerator outside fixed lattice")
    require(isinstance(phase_numerator, int) and 0 <= phase_numerator < 500, "phase numerator outside fixed lattice")
    voltage = phase_numerator
    events = []
    for k in range(N):
        voltage += intensity_numerator
        if voltage >= 500:
            events.append(k)
            voltage -= 500
        require(0 <= voltage < 500, "integer reset invariant failed")
    return tuple(events)


def cumulative(events):
    require(tuple(sorted(set(events))) == tuple(events) and all(isinstance(k, int) and 0 <= k < N for k in events), "invalid event record")
    counts, indices = [0], set(events)
    for k in range(N):
        counts.append(counts[-1] + (k in indices))
    return counts


def signature(events):
    cumulative(events)
    before = [k for k in events if k < TARGET]
    after = [k for k in events if k >= TARGET]
    missing = tuple(name for name, absent in (("a1", not before), ("a2", len(before) < 2),
                                             ("b1", not after), ("b2", len(after) < 2)) if absent)
    if missing:
        return None, missing
    result = (len(events), events[0], events[-1], before[-2], before[-1], after[0], after[1])
    result += tuple(sum(start <= k <= end for k in events) for start, end in ((20, 28), (16, 32), (12, 36), (0, 48)))
    return result, ()


def baseline_values(events):
    summary, missing = signature(events)
    require(not missing, "censored baseline endpoints")
    count, first, last, a2, a1, b1, b2 = summary[:7]
    return {"count50": F(Q * count, N), "target_one_interval": F(Q, b1 - a1),
            "target_three_intervals": F(3 * Q, b2 - a2), "first_to_last": F(Q * (count - 1), last - first)}


def project(events):
    """Eliminate unknown phase using all1275 strict u<t bounds, then [0,1]."""
    counts = cumulative(events)
    ln, ld, un, ud, lc, uc = 0, 1, 1, 1, True, True
    pairs = 0
    for t in range(1, N + 1):
        for u in range(t):
            denominator = t - u
            difference = counts[t] - counts[u]
            low, high = Q * (difference - 1), Q * (difference + 1)
            left = low * ld - ln * denominator
            if left > 0:
                ln, ld, lc = low, denominator, False
            elif left == 0:
                lc = False
            right = high * ud - un * denominator
            if right < 0:
                un, ud, uc = high, denominator, False
            elif right == 0:
                uc = False
            pairs += 1
    require(pairs == 1275, "projection constraint census failed")
    return Interval(F(ln, ld), F(un, ud), lc, uc)


def phase_interval(events, intensity):
    intensity = F(intensity)
    require(0 <= intensity <= 1, "intensity outside continuous domain")
    counts = cumulative(events)
    lower = max(Q * n - t * intensity for t, n in enumerate(counts))
    upper = min(Q * (n + 1) - t * intensity for t, n in enumerate(counts))
    return Interval(lower, upper, True, False)


def feasible_pair(events, intensity, phase):
    intensity, phase = F(intensity), F(phase)
    return 0 <= intensity <= 1 and all(0 <= phase + t * intensity - Q * n < Q
                                       for t, n in enumerate(cumulative(events)))


def strict_gap(left, distractor):
    require(left.nonempty() and distractor.nonempty(), "empty generated-record projection")
    if left.upper < distractor.lower:
        return distractor.lower - left.upper, "L_below_D"
    if distractor.upper < left.lower:
        return left.lower - distractor.upper, "D_below_L"
    return None


def shared_pair(left_generators, right_generators):
    left, right = defaultdict(list), defaultdict(list)
    for j, phase in left_generators:
        left[j].append(phase)
    for j, phase in right_generators:
        right[j].append(phase)
    for j in sorted(set(left) & set(right)):
        for phase_l in sorted(left[j]):
            for phase_r in sorted(right[j]):
                if phase_l != phase_r:
                    return j, phase_l, phase_r
    return None


def pair_payload(j, phase):
    return {"lattice_intensity_numerator": j, "lattice_phase_numerator": phase, "lattice_denominator": 100,
            "intensity": rational(F(j, 100)), "phase": rational(F(phase, 100))}


def certificate(left, right, distractor, generators, projections, positions):
    require(len({left, right, distractor}) == 3, "witness records must be distinct")
    summaries = [signature(record)[0] for record in (left, right, distractor)]
    require(summaries[0] is not None and summaries[0] == summaries[1] == summaries[2], "strong signature mismatch")
    baselines = [baseline_values(record) for record in (left, right, distractor)]
    require(baselines[0] == baselines[1] == baselines[2], "derived baseline mismatch")
    pair = shared_pair(generators[left], generators[right])
    require(pair is not None, "no correct-pair common intensity/different phases")
    j, phase_l, phase_r = pair
    intensity = F(j, 100)
    gap = strict_gap(projections[left], projections[distractor])
    require(gap is not None and gap[0] > 0, "no strict positive continuous projection gap")
    require(projections[left].contains(intensity) and projections[right].contains(intensity), "correct-pair projection does not contain common intensity")
    require(feasible_pair(left, intensity, F(phase_l, 100)) and feasible_pair(right, intensity, F(phase_r, 100)), "correct-pair original inequalities failed")
    j_d, phase_d = min(generators[distractor])
    require(j_d != j and all(j_other != j for j_other, _ in generators[distractor]), "distractor shares correct intensity")
    require(feasible_pair(distractor, F(j_d, 100), F(phase_d, 100)), "distractor generator inequalities failed")
    records = {}
    for name, events, selected in (("L", left, (j, phase_l)), ("R", right, (j, phase_r)), ("D", distractor, (j_d, phase_d))):
        point = pair_payload(*selected)
        records[name] = {"events_zero_based": list(events), "selected_generator": point,
                         "all_generating_integer_pairs": [list(p) for p in sorted(generators[events])],
                         "projection": projections[events].payload(),
                         "phase_interval_at_selected_intensity": phase_interval(events, F(selected[0], 100)).payload(),
                         "selected_phase_satisfies_all_original_inequalities": True}
    return {"records": records, "signature_fields": list(SIGNATURE_FIELDS), "identical_signature": list(summaries[0]),
            "identical_baseline_values": {key: rational(value) for key, value in baselines[0].items()},
            "common_correct_intensity": rational(intensity), "different_correct_phases": True,
            "strict_gap": {**rational(gap[0]), "ordering": gap[1]}, "positions_and_counters": positions,
            "claim": "information loss in these specified summaries under the fixed static law only"}


class StopSearch(Exception):
    pass


def find_witness(groups, generators, projections, counters, guard, ld_limit=LD_LIMIT, r_limit=R_LIMIT):
    """Counters checked before the next operation; never exceed either cap."""
    for group_ordinal, summary in enumerate(sorted(groups)):
        records = sorted(groups[summary])
        for l_ordinal, left in enumerate(records):
            for d_ordinal, distractor in enumerate(records):
                if left == distractor:
                    continue
                guard()
                if counters["ordered_L_D_comparisons"] >= ld_limit:
                    raise StopSearch("ordered_L_D_comparison_limit")
                counters["ordered_L_D_comparisons"] += 1
                if strict_gap(projections[left], projections[distractor]) is None:
                    continue
                for r_ordinal, right in enumerate(records):
                    if right in (left, distractor):
                        continue
                    guard()
                    if counters["R_record_candidate_checks"] >= r_limit:
                        raise StopSearch("R_record_candidate_limit")
                    counters["R_record_candidate_checks"] += 1
                    if shared_pair(generators[left], generators[right]) is not None:
                        positions = {"signature_group_ordinal_zero_based": group_ordinal,
                                     "L_record_ordinal_zero_based": l_ordinal, "D_record_ordinal_zero_based": d_ordinal,
                                     "R_record_ordinal_zero_based": r_ordinal, **counters}
                        return certificate(left, right, distractor, generators, projections, positions)
    return None


def fixed_search(guard):
    """Only the registered Slurm CLI may call this full50500-pair search."""
    generators, projections, groups = {}, {}, defaultdict(list)
    counters = {"generated_parameter_pairs": 0, "projected_unique_records": 0, "verified_generating_pairs": 0,
                "ordered_L_D_comparisons": 0, "R_record_candidate_checks": 0}
    census = {"projection_pairs_per_record": 1275, "generation_complete": False, "validation_complete": False}
    stage = "generation"
    try:
        for j in range(101):
            for phase in range(500):
                guard()
                events = encode(j, phase)
                generators.setdefault(events, []).append((j, phase))
                counters["generated_parameter_pairs"] += 1
        require(counters["generated_parameter_pairs"] == 50500, "generation census mismatch")
        census.update(generation_complete=True, unique_records=len(generators),
                      duplicate_parameter_pairs=50500-len(generators))
        exclusions = {name: 0 for name in ("a1", "a2", "b1", "b2")}
        census["exclusion_counts_overlapping"] = exclusions
        stage = "projection_and_all_generator_validation"
        eligible_pairs = 0
        for events in sorted(generators):
            guard()
            interval = project(events)
            require(interval.nonempty(), "generated record has empty continuous projection")
            projections[events] = interval
            counters["projected_unique_records"] += 1
            for j, phase in generators[events]:
                guard()
                require(interval.contains(F(j, 100)) and feasible_pair(events, F(j, 100), F(phase, 100)),
                        "retained generator not in exact projection/original inequalities")
                counters["verified_generating_pairs"] += 1
            summary, missing = signature(events)
            for name in missing:
                exclusions[name] += 1
            if summary is not None:
                groups[summary].append(events)
                eligible_pairs += len(generators[events])
        require(counters["verified_generating_pairs"] == 50500, "retained generator census mismatch")
        unique_eligible = sum(map(len, groups.values()))
        population_hash = hashlib.sha256()
        for events in sorted(generators):
            guard()
            population_hash.update((json.dumps([events, generators[events]], separators=(",", ":")) + "\n").encode())
        census.update(validation_complete=True, eligible_unique_records=unique_eligible,
                      excluded_unique_records=len(generators)-unique_eligible, exclusion_counts_overlapping=exclusions,
                      eligible_generating_pairs=eligible_pairs, signature_groups=len(groups),
                      groups_with_at_least_two_records=sum(len(v) >= 2 for v in groups.values()),
                      groups_with_at_least_three_records=sum(len(v) >= 3 for v in groups.values()),
                      all_retained_generators_population_sha256=population_hash.hexdigest())
        stage = "deterministic_witness_scan"
        witness = find_witness(groups, generators, projections, counters, guard)
        return {"outcome": "POSITIVE" if witness else "EXHAUSTIVE_NEGATIVE",
                "stop_reason": "first_qualifying_witness" if witness else "all_fixed_signature_pairs_exhausted",
                "stage": stage, "counters": counters, "census": census, "witness": witness}
    except StopSearch as error:
        census.update(unique_records_observed=len(generators), eligible_unique_records_observed=sum(map(len, groups.values())))
        return {"outcome": "INCOMPLETE", "stop_reason": str(error), "stage": stage,
                "counters": counters, "census": census, "witness": None}
    except ValueError as error:
        census.update(unique_records_observed=len(generators), eligible_unique_records_observed=sum(map(len, groups.values())))
        return {"outcome": "INVALID", "stop_reason": str(error), "stage": stage,
                "counters": counters, "census": census, "witness": None}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def critical_checks():
    output = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(CAMPAIGN / "tests"), pattern="test_exact_compatibility.py")
    result = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)
    require(result.testsRun > 0 and result.wasSuccessful() and not result.skipped, "critical exact checks failed:\n" + output.getvalue())
    return {"status": "PASS_TINY_NAMED_EXAMPLES_ONLY", "tests": result.testsRun, "output": output.getvalue(),
            "full_lattice_enumerated": False, "dataset_or_model_imports": False}


def execute(args):
    started = time.monotonic()
    require(os.environ.get("SLURM_JOB_ID") and socket.gethostname() == "cccl3"
            and os.environ.get("USER") == "rongliang", "full search requires registered cccl3 Slurm")
    require(os.environ.get("SLURM_CPUS_PER_TASK") == "1" and os.environ.get("SLURM_MEM_PER_NODE") == "2048", "exact1CPU2GiB allocation required")
    require(os.environ.get("SLURM_GPUS", "0") in ("", "0") and os.environ.get("SLURM_GPUS_ON_NODE", "0") in ("", "0")
            and os.environ.get("CUDA_VISIBLE_DEVICES", "") == "", "S6 requires zero GPU")
    intent = json.loads(args.intent.read_text())
    registration = json.loads(args.registration.read_text())
    registry = json.loads(args.source_registry.read_text())
    require(registration["status"] == "registered" and registration["mode"] == "stage6-exact-compatibility", "external root registration required")
    require(registration["intent_sha256"] == sha(args.intent) and registration["source_identity_sha256"] == registry["source_identity_sha256"], "source/intent binding mismatch")
    require(registration["scientific_contract_sha256"] == intent["scientific_contract_sha256"] == CONTRACT_SHA, "contract binding mismatch")
    require(registration["resources"] == intent["resources"] == {"cpus": 1, "memory_gib": 2, "wall_seconds": 900, "gpus": 0}, "resource contract mismatch")
    require(intent["population"] == {"Q": 5, "N": 50, "target_bin": 24, "intensity_numerators": [0, 100],
                                    "phase_numerators": [0, 499], "denominator": 100, "parameter_pairs": 50500}, "population changed")
    require(intent["ordered_L_D_comparison_cap"] == LD_LIMIT and intent["R_record_candidate_cap"] == R_LIMIT,
            "comparison limits changed")
    require(intent["internal_wall_stop_seconds"] == 895, "fixed receipt-write margin changed")
    source = args.source_registry.resolve().parent
    for relative, expected in registry["source_files"].items():
        require(sha(source / relative) == expected, "frozen source changed: " + relative)
    contract = CAMPAIGN / "analysis/stage6-exact-compatibility-contract-astra.md"
    require(sha(contract) == CONTRACT_SHA, "contract file changed")
    for path in (Path(__file__).resolve(), args.intent.resolve(), contract.resolve()):
        require(registry["source_files"].get(str(path.relative_to(source))) == sha(path), "required source outside capsule")
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        checks = critical_checks()
        save(args.output / "software-checks.json", checks)
        def guard():
            if time.monotonic() - started >= 895:
                raise StopSearch("wall_time_limit_with_fixed5second_receipt_margin")
        result = fixed_search(guard)
        usage = resource.getrusage(resource.RUSAGE_SELF)
        result.update(status="ANALYTIC_RECEIPT_PENDING_INDEPENDENT_AUDIT", scientific_contract_sha256=CONTRACT_SHA,
                      source_identity_sha256=registry["source_identity_sha256"], source_files=registry["source_files"],
                      registration_sha256=sha(args.registration), intent_sha256=sha(args.intent), git_commit=registry["git_commit"],
                      dirty_manifest_sha256=registry["dirty_manifest_sha256"], runtime_seconds=time.monotonic()-started,
                      peak_RSS_bytes=int(usage.ru_maxrss*1024), CPU_user_seconds=usage.ru_utime, CPU_system_seconds=usage.ru_stime,
                      cpus=1, gpus=0, Slurm_job_id=os.environ["SLURM_JOB_ID"], process_exit_code=1 if result["outcome"] == "INVALID" else 0,
                      python=sys.version, dataset_access=False, model_imports=False, optimizer_updates=0,
                      comparison_caps={"ordered_L_D": LD_LIMIT, "R_record": R_LIMIT},
                      ordering="complete fixed lattice; signature tuple; event tuples; ordered L,D then remaining R; min common intensity then phases",
                      cap_semantics="check before next operation; at most cap operations; no unfinished next operation counted",
                      independent_certificate_or_accounting_audit_required=True, automatic_continuation=False)
        if result["witness"]:
            save(args.output / "witness.json", result["witness"])
        save(args.output / "result.json", result)
        if result["outcome"] == "INVALID":
            raise RuntimeError("software consistency invalidated search; preserved result/counters")
    except BaseException as error:
        save(args.output / "failure.json", {"outcome": "INVALID_OR_INCOMPLETE", "stop_reason": repr(error),
                                             "runtime_seconds": time.monotonic()-started, "process_exit_code": 1,
                                             "no_retry_or_relaxation": True})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("registration", "source-registry", "intent", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    execute(parser.parse_args())


if __name__ == "__main__":
    main()
