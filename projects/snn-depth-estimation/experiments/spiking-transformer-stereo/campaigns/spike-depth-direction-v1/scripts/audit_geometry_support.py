"""S4 Slurm-only CPU audit. Streaming, source-bound, no learned components."""
import os
for _key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_key] = '2'
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import argparse
import csv
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import socket
import signal
import sys
import time
import cv2
import numpy as np
from geometry_support import OFFSETS, measure, overlap_report, aggregate

MANIFEST_SHA = 'd8f07b9dc3a17fcc640777e71d55884f17fb116d29fc5e8c9a23c1892f181d71'
READER_SHA = 'e5edc76c5ff13a26975154c16b7cb5a4a763902960b8d10eccc2a9877978b9af'
CONTRACT_SHA = '816270906a3d38fab63ab250dfa08f018f0cfd5f0107f2cd1def4a83964dc2d7'
REFERENCES = ['scene_0000', 'scene_0029', 'scene_0059', 'scene_0089', 'scene_0119']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def shifted(left, shift):
    h, w = left.shape
    right = np.zeros_like(left)
    rv = np.zeros(left.shape, bool)
    right[:, :w-shift] = left[:, shift:]
    rv[:, :w-shift] = True
    return right, rv, np.full((h, w), shift, np.float64)


def control_checks(result, affine=False):
    assessable = result['support']['common_patch'] > 0 and result['support']['common_N'] > 0
    a = result['costs']['rate_MAE']['candidates']['0']['maximum']
    n = result['costs']['normalized_disagreement']['candidates']['0']['maximum']
    separate = any(s['minimum'] is not None and (s['minimum'] < 0 or s['maximum'] > 0)
                   for c in result['costs'].values() for s in c['paired_margins_vs_zero'].values())
    passed = assessable and n <= 1e-8 and (affine or a <= 1e-12) and separate
    return {'assessable': assessable, 'numerical_construction_pass': bool(passed),
            'nonzero_candidate_separation': bool(separate)}


def control_suite(reference_rates, emit):
    checks = []
    noisy_means = {}
    for ri, (scene, left) in enumerate(reference_rates):
        assert REFERENCES[ri] == scene  # seed index is ordinal, never the scene number
        for s in [0, 4, 12]:
            right, rv, g = shifted(left, s)
            for affine in [False, True]:
                result = measure(left, .5*right+.25 if affine else right, g, np.ones(left.shape, bool), rv)
                check = control_checks(result, affine)
                record = {'kind': 'affine_known_shift' if affine else 'clean_known_shift',
                          'reference_scene': scene, 'reference_index': ri, 'known_shift': s,
                          'engineering_control': True, 'measurement': result, 'checks': check}
                emit(record)
                checks.append({k: v for k, v in record.items() if k != 'measurement'})
        right, rv, g = shifted(left, 4)
        for repetition in range(8):
            seed = 41000+100*ri+repetition
            rng = np.random.default_rng(seed)
            nl = rng.binomial(50, left)/50.
            nr = rng.binomial(50, right)/50.
            noisy = measure(nl, nr, g, np.ones(left.shape, bool), rv)
            emit({'kind': 'noisy_known_shift', 'reference_scene': scene, 'reference_index': ri,
                  'known_shift': 4, 'repetition': repetition, 'seed': seed,
                  'engineering_control': True,
                  'measurement': noisy})
            for cost, c in noisy['costs'].items():
                for offset, stats in c['paired_margins_vs_zero'].items():
                    noisy_means.setdefault((scene, cost, offset), []).append(stats['mean'])
    null_means = {}
    shape = left.shape
    for li, probability in enumerate([.1, .5]):
        for repetition in range(20):
            seed = 42000+100*li+repetition
            rng = np.random.default_rng(seed)
            left = rng.binomial(50, probability, size=shape)/50.
            right = rng.binomial(50, probability, size=shape)/50.
            result = measure(left, right, np.full(shape, 4.), np.ones(shape, bool))
            emit({'kind': 'independent_no_correspondence', 'level_index': li, 'probability': probability,
                  'repetition': repetition, 'seed': seed, 'engineering_control': True, 'measurement': result})
            for cost, c in result['costs'].items():
                for offset, s in c['paired_margins_vs_zero'].items():
                    null_means.setdefault((probability, cost, offset), []).append(s['mean'])
    null_summary = []
    for (probability, cost, offset), values in sorted(null_means.items()):
        vv = [v for v in values if v is not None]
        null_summary.append({'probability': probability, 'cost': cost, 'offset': int(offset),
                             'repetitions': len(values), 'contributing_repetitions': len(vv),
                             'per_repetition_mean_margins': values,
                             'signed_mean': float(np.mean(vv)) if vv else None,
                             'empirical_2_5_97_5_percentiles': np.quantile(vv, [.025, .975]).tolist() if vv else None,
                             'not_real_data_confidence_intervals': True})
    status = 'INCOMPLETE' if not all(c['checks']['assessable'] for c in checks) else 'PASS' if all(c['checks']['numerical_construction_pass'] for c in checks) else 'QCFAIL'
    noisy_summary = [{'reference_scene': scene, 'cost': cost, 'offset': int(offset),
                      'per_repetition_mean_margins': vv,
                      'signed_mean': float(np.mean(vv)) if all(v is not None for v in vv) else None,
                      'spread_std': float(np.std(vv, ddof=1)) if all(v is not None for v in vv) else None}
                     for (scene, cost, offset), vv in sorted(noisy_means.items())]
    return {'status': status, 'mandatory_checks': checks, 'noisy_positive_control_spread': noisy_summary, 'no_correspondence_distributions': null_summary,
            'no_noisy_or_real_data_accuracy_gate': True}


def validate_manifest(manifest):
    assert set(manifest) >= {'train', 'dev'}
    expected = {'train': (120, 8), 'dev': (30, 2)}
    ids = {}
    for split, (ns, nf) in expected.items():
        rows = manifest[split]
        scenes = sorted({r['scene'] for r in rows})
        assert len(rows) == ns*nf and len(scenes) == ns
        assert len({(r['scene'], r['frame']) for r in rows}) == len(rows)
        assert all(sum(r['scene'] == scene for r in rows) == nf for scene in scenes)
        for r in rows:
            for key in ['spike_l', 'spike_r', 'depth']:
                assert r[key].startswith('Train/') and '..' not in Path(r[key]).parts
        ids[split] = set(scenes)
    assert not ids['train'] & ids['dev']
    assert all(any(r['scene'] == s and r['frame'] == '0001' for r in manifest['train']) for s in REFERENCES)


def main():
    parser = argparse.ArgumentParser()
    for key in ['registration', 'source-registry', 'manifest', 'reader', 'data-root', 'output']:
        parser.add_argument('--'+key, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname() != 'cccl3':
        raise RuntimeError('registered cccl3 CPU Slurm only')
    if os.environ.get('SLURM_GPUS') not in [None, '', '0'] or os.environ.get('CUDA_VISIBLE_DEVICES') not in [None, '', '-1', 'NoDevFiles']:
        raise RuntimeError('S4 must allocate zero GPUs')
    cv2.setNumThreads(2)
    reg = json.loads(Path(args.registration).read_text())
    source = json.loads(Path(args.source_registry).read_text())
    repo = Path(args.source_registry).resolve().parent
    assert reg['stage'] == 'S4' and reg['status'] == 'registered'
    assert reg['source_identity_sha256'] == source['source_identity_sha256']
    for name, h in source['source_files'].items():
        assert sha(repo/name) == h, name
    assert sha(args.manifest) == reg['manifest_sha256'] == MANIFEST_SHA
    assert sha(args.reader) == READER_SHA
    assert reg['scientific_contract_sha256'] == CONTRACT_SHA
    intent = json.loads((Path(__file__).resolve().parents[1]/'configs/stage4-geometry-support-intent.json').read_text())
    assert reg['config'] == intent['config'], 'external config must equal frozen reviewed intent'
    assert reg['config']['offsets'] == list(OFFSETS) and reg['config']['references'] == REFERENCES
    manifest = json.loads(Path(args.manifest).read_text())
    validate_manifest(manifest)
    spec = importlib.util.spec_from_file_location('S4_pinned_reader', args.reader)
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    root = Path(args.data_root).resolve()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)  # no blind overwrite, resume or retry
    started = time.monotonic()
    identity = {'source_identity_sha256': source['source_identity_sha256'],
                'registration_sha256': sha(args.registration), 'manifest_sha256': sha(args.manifest),
                'reader_sha256': sha(args.reader), 'scientific_contract_sha256': CONTRACT_SHA,
                'git_commit': reg['git_commit'], 'dirty_manifest_sha256': reg['dirty_manifest_sha256']}
    metadata = {**identity, 'status': 'RUNNING', 'data_root': str(root), 'job_id': os.environ['SLURM_JOB_ID'],
                'host': socket.gethostname(), 'python': sys.version, 'numpy': np.__version__,
                'opencv': cv2.__version__, 'platform': platform.platform(), 'GPU_count': 0,
                'CPU_threads': 2, 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'scientific_updates': 0, 'no_test': True, 'no_checkpoint_or_model': True,
                'memory_policy': 'one frame/candidate; two-pass common-N; summary-only cohort storage'}
    save(output/'status.json', metadata)
    def time_limit(signum, frame):
        raise TimeoutError('Slurm termination: retain partial audit; no fallback or retry')
    signal.signal(signal.SIGTERM, time_limit)

    def rate(relative):
        path = root/relative
        assert path.stat().st_size == 625000
        s = reader.get_block_spikes(str(path), begin_idx=0, spike_height=250, spike_width=400, block_len=50, flipud=False, with_head=False)
        assert s.shape == (50, 250, 400) and np.isin(s, [0, 1]).all()
        return s.sum(axis=0, dtype=np.float64)/50.

    try:
        # Five predeclared references only, controls are never used to tune settings.
        references = ((scene, rate(next(r['spike_l'] for r in manifest['train'] if r['scene'] == scene and r['frame'] == '0001'))) for scene in REFERENCES)
        with (output/'controls.jsonl').open('w') as handle:
            def emit(record):
                handle.write(json.dumps(record, allow_nan=False)+'\n')
                handle.flush()
            controls = control_suite(references, emit)
        save(output/'control-summary.json', controls)
        del references
        overlaps = []
        for scene in REFERENCES:
            for first, last in [('0001', '0002'), ('0254', '0255')]:
                for view in ['l', 'r']:
                    a = f'Train/{scene}/spike_{view}/{first}_spike_{view}.dat'
                    b = f'Train/{scene}/spike_{view}/{last}_spike_{view}.dat'
                    overlaps.append({'scene': scene, 'view': view, 'first_file': a, 'last_file': b,
                                     **overlap_report((root/a).read_bytes(), (root/b).read_bytes())})
        save(output/'overlap.json', {'fixed_files': 40, 'comparisons': overlaps,
                                    'no_physical_cadence_or_synchronization_claim': True})
        all_rows = []
        with (output/'frames.jsonl').open('w') as handle, (output/'frame-costs.csv').open('w', newline='') as cf:
            writer = csv.DictWriter(cf, fieldnames=['split', 'scene', 'frame', 'cost', 'offset', 'kind', 'count', 'mean', 'median', 'q25', 'q75', 'minimum', 'maximum', 'fraction_strictly_positive'])
            writer.writeheader()
            for split in ['train', 'dev']:
                for r in manifest[split]:
                    left, right = rate(r['spike_l']), rate(r['spike_r'])
                    depth = reader.read_depth(str(root/r['depth'])).astype(np.float32)
                    positive = np.isfinite(depth) & (depth > 0)
                    # Preserve inherited FLOAT32 values before diagnostic float64.
                    g = np.zeros(depth.shape, np.float32)
                    g[positive] = (289.740625*.08)/depth[positive]
                    valid = positive & (g < 512)
                    measurement = measure(left, right, g.astype(np.float64), valid)
                    record = {'split': split, 'scene': r['scene'], 'frame': r['frame'],
                              'source_files': {k: r[k] for k in ['spike_l', 'spike_r', 'depth']},
                              'measurement': measurement}
                    handle.write(json.dumps(record, allow_nan=False)+'\n')
                    handle.flush()
                    for cost, values in measurement['costs'].items():
                        for kind in ['candidates', 'paired_margins_vs_zero']:
                            for offset, stats in values[kind].items():
                                writer.writerow({k: record[k] for k in ['split', 'scene', 'frame']} |
                                                {'cost': cost, 'offset': offset, 'kind': kind} | stats)
                    cf.flush()
                    all_rows.append(record)  # summaries only; no image cohort arrays
                    if len(all_rows) % 10 == 0:
                        save(output/'status.json', {**metadata, 'completed_frames': len(all_rows), 'elapsed_seconds': time.monotonic()-started, 'control_status': controls['status']})
        scenes = []
        for split in ['train', 'dev']:
            for scene in sorted({r['scene'] for r in all_rows if r['split'] == split}):
                group = [r for r in all_rows if r['split'] == split and r['scene'] == scene]
                scenes.append({'split': split, 'scene': scene, 'frames': [r['frame'] for r in group],
                               'equal_frame_summary': aggregate([r['measurement'] for r in group])})
        # Split means are means of scene means, including explicit nulls/counts.
        splits = {}
        for split in ['train', 'dev']:
            group = [r for r in scenes if r['split'] == split]
            keys = sorted({k for r in group for k in r['equal_frame_summary']})
            splits[split] = {'scene_count': len(group), 'frame_count': sum(len(r['frames']) for r in group), 'equal_scene_summary': {}}
            for key in keys:
                vals = [r['equal_frame_summary'][key]['mean_of_record_summaries'] for r in group if r['equal_frame_summary'][key]['mean_of_record_summaries'] is not None]
                splits[split]['equal_scene_summary'][key] = {'mean_of_scene_means': float(np.mean(vals)) if vals else None,
                                                           'contributing_scenes': len(vals), 'total_scenes': len(group)}
        assert len(all_rows) == 1020 and len(scenes) == 150
        for split in ['train', 'dev']:
            assert {(r['scene'], r['frame']) for r in all_rows if r['split'] == split} == {(r['scene'], r['frame']) for r in manifest[split]}
        overlap_valid = all(r['status'] == 'COMPLETE' for r in overlaps)
        status = 'COMPLETE' if controls['status'] == 'PASS' and overlap_valid else 'INCOMPLETE' if controls['status'] == 'INCOMPLETE' or not overlap_valid else 'QCFAIL'
        result = {**identity, 'status': status, 'controls': controls, 'scene_summaries': scenes,
                  'split_summaries': splits, 'frame_count': len(all_rows), 'scene_count': len(scenes),
                  'elapsed_seconds': time.monotonic()-started, 'no_training_release': True,
                  'aggregation': 'equal frames per scene then equal scenes; averaged frame quantiles are NOT pooled quantiles'}
        save(output/'result.json', result)
        save(output/'status.json', {**metadata, 'status': status, 'completed_frames': len(all_rows), 'elapsed_seconds': result['elapsed_seconds']})
    except BaseException as error:
        save(output/'status.json', {**metadata, 'status': 'INCOMPLETE_ERROR_NO_AUTOMATIC_RETRY',
                                   'error': repr(error), 'elapsed_seconds': time.monotonic()-started})
        raise


if __name__ == '__main__':
    main()
