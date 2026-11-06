"""CPU-only audit/normalization of an existing Original diagnostic replay.

Preserves the historical per-frame measurements and evaluator provenance. No
checkpoint loading, source capsule edits, GPU access or new inference occurs.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize(diagnostic, historical_evaluator, current_evaluator, current_registry,
              reuse_receipt, manifest, raw_root, receipt):
    old = json.loads(diagnostic.read_text())
    reuse = json.loads(reuse_receipt.read_text())
    current = json.loads(current_registry.read_text())
    assert old['arm'] == 'Original' and old['update'] == 30000
    assert reuse['status'] == 'VERIFIED_REUSE' and reuse['arm'] == 'Original' and reuse['seed'] == 1234
    assert current['source_identity_sha256'] == reuse['current_source_identity']
    assert reuse['stream_audit']['actual_indices_hv_lr_verified']
    assert reuse['native_initial_state_sha256'] == reuse['common_initial_state_sha256']
    endpoints = [r for r in reuse['endpoints'] if r['update'] == old['update']]
    assert len(endpoints) == 1
    endpoint = endpoints[0]
    assert endpoint['checkpoint_verified'] and endpoint['checkpoint_sha256']
    assert old['checkpoint_sha256'] == endpoint['checkpoint_sha256']
    assert endpoint['archived_source_identity'] == reuse['archived_source_identity']
    assert old['evaluator_sha256'] == digest(historical_evaluator)
    old_ast = ast.dump(ast.parse(historical_evaluator.read_text()), include_attributes=False)
    new_ast = ast.dump(ast.parse(current_evaluator.read_text()), include_attributes=False)
    assert old_ast == new_ast, 'diagnostic evaluators have different executable syntax'
    current_name = 'projects/snn-depth-estimation/experiments/spiking-attention-placement-v1/probes.py'
    assert digest(current_evaluator) == current['source_files'][current_name]
    dependency_names = [name for name in old['source_registry']['source_files']
                        if '/core/' in name or name.endswith('/scripts/run_experiment.py')]
    assert dependency_names
    for name in dependency_names:
        assert old['source_registry']['source_files'][name] == current['source_files'][name], name
    assert digest(manifest) == old['manifest_sha256']
    expected = {(r['scene'], r['frame']) for r in json.loads(manifest.read_text())['dev']}
    saved = endpoint['evaluation']
    assert saved['samples_seen'] == 240000 and saved['update'] == 30000
    official = {(r['scene'], r['frame']): r for r in saved['per_frame']}
    frames = old['per_frame']
    actual = {(r['scene'], r['frame']): r for r in frames}
    assert len(actual) == len(official) == len(frames) == len(expected) == 60
    assert set(actual) == set(official) == expected
    assert len({s for s, _ in actual}) == 30
    assert old['parity']['passed']
    for key, row in actual.items():
        for field in ('target_support_sha256', 'fine_cohort_sha256'):
            assert len(row[field]) == 64 and int(row[field], 16) >= 0
        assert set(row['iterations']) == {'1', '2', '4', '8', '16', '32'}
        for iteration in row['iterations'].values():
            assert set(iteration) == {'full', 'inview', 'oov'}
            assert iteration['full']['valid_pixels'] == iteration['inview']['valid_pixels'] + iteration['oov']['valid_pixels']
            for values in iteration.values():
                if values['valid_pixels']:
                    assert all(math.isfinite(values[k]) for k in ('epe', 'bad1', 'bad3', 'bad5', 'error_sum'))
        full = row['iterations']['32']['full']
        for metric in ('valid_pixels', 'epe', 'bad1', 'bad3', 'bad5'):
            assert full[metric] == official[key][metric], (key, metric)
        rank = row['rank']['native']
        assert rank['pixel_count'] + rank['excluded_oov_count'] == rank['valid_fine_count']
    # Recompute summary with the semantically identical current CPU reducer.
    spec = importlib.util.spec_from_file_location('normalized_original_probes', current_evaluator)
    probes = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probes)
    parity = probes.assert_parity(frames, saved['per_frame'])
    assert parity['max_frame_epe_difference'] == 0
    assert probes.summarize(frames) == old['summary']
    provenance = dict(status='VERIFIED_ARCHIVED_DIAGNOSTIC_REUSE', arm='Original', seed=1234, update=30000,
        diagnostic_sha256=digest(diagnostic), checkpoint_sha256=old['checkpoint_sha256'],
        historical_evaluator_sha256=digest(historical_evaluator),
        current_semantically_identical_evaluator_sha256=digest(current_evaluator),
        evaluator_ast_identical=True, semantic_diff='comments only; executable AST identical',
        dependency_files_matched=len(dependency_names),
        historical_diagnostic_source_identity=old['source_registry']['source_identity_sha256'],
        checkpoint_source_identity=reuse['archived_source_identity'],
        admitted_phase2_source_identity=current['source_identity_sha256'],
        reference_reuse_receipt_sha256=digest(reuse_receipt),
        official_frames=60, scenes=30, official_metric_max_differences={k: 0 for k in ('valid_pixels', 'epe', 'bad1', 'bad3', 'bad5')},
        target_and_fine_cohort_hashes='preserved unchanged from historical replay; downstream pairwise equality still required',
        gpu_replay_performed=False, normalizer_sha256=digest(__file__))
    normalized = dict(old)
    normalized.update(kind='NORMALIZED_ARCHIVED_REPLAY', seed=1234, branch_seed=2017, samples_seen=240000,
        source_identity=current['source_identity_sha256'], checkpoint_source_identity=reuse['archived_source_identity'],
        probes_sha256=old['evaluator_sha256'], reuse_receipt_sha256=digest(reuse_receipt),
        normalization=provenance, weight_variation=[])
    normalized['parity'] = dict(parity, max_frame_bad_percentage_point_difference=0., all_official_frame_metrics_exact=True)
    output = raw_root/'seed-1234/Original/replay-30000.json'
    if output.exists():
        assert json.loads(output.read_text()) == normalized, 'refusing to overwrite different existing diagnostic'
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        temp = output.with_suffix('.tmp')
        temp.write_text(json.dumps(normalized, indent=2, allow_nan=False)+'\n')
        temp.replace(output)
    provenance['normalized_diagnostic_sha256'] = digest(output)
    provenance['normalized_relative_path'] = 'seed-1234/Original/replay-30000.json'
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(provenance, indent=2, allow_nan=False)+'\n')
    return provenance


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('diagnostic', 'historical-evaluator', 'current-evaluator', 'current-registry', 'reuse-receipt', 'manifest', 'raw-root', 'receipt'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    result = normalize(**vars(args))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
