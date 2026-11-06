"""User-authorized bounded native-Dev CPU matching experiment; never uses torch."""
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import cv2
import numpy as np
from patch_match import chunk_rates, zncc, score_metrics

HERE = Path(__file__).resolve().parent.parent
METRICS = ('epe', 'bad1', 'bad3', 'bad5', 'mean_gt_rank', 'top1_fraction', 'mean_gt_near1_rank',
           'gt_nearest_vs_far3_margin', 'top1_top2_margin', 'argmax_tie_fraction', 'degenerate_candidate_chunk_fraction')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def decode(path, shape):
    raw = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    assert raw.size*8 == np.prod(shape)
    return np.unpackbits(raw, bitorder='little').reshape(shape).astype(bool)


def reduce_frames(rows):
    methods = sorted({r['method'] for r in rows})
    result = {}
    for method in methods:
        selected = [r for r in rows if r['method'] == method]
        scenes = sorted({r['scene'] for r in selected})
        by_scene = {scene: {metric: float(np.mean([r[metric] for r in selected if r['scene']==scene and r[metric] is not None]))
                                if any(r['scene']==scene and r[metric] is not None for r in selected) else None
                            for metric in METRICS} for scene in scenes}
        result[method] = dict(per_scene=by_scene,
            equal_scene={metric: float(np.mean([v[metric] for v in by_scene.values() if v[metric] is not None]))
                         if any(v[metric] is not None for v in by_scene.values()) else None for metric in METRICS},
            scenes=len(scenes), frames=len(selected), queries=sum(r['count'] for r in selected))
    return result


def contrasts(summary, lengths):
    output = {}
    for length in lengths:
        if length == 50:
            continue
        aligned, shifted = f'aligned_L{length}', f'right_shift_L{length}'
        for first, second in ((aligned, 'aggregate_first'), (aligned, shifted)):
            name = first+'-minus-'+second
            scenes = sorted(summary[first]['per_scene'])
            values = {s: summary[first]['per_scene'][s]['epe']-summary[second]['per_scene'][s]['epe'] for s in scenes}
            output[name] = dict(mean_scene_epe_delta=float(np.mean(list(values.values()))),
                median_scene_epe_delta=float(np.median(list(values.values()))),
                scene_wins=sum(v < 0 for v in values.values()), scene_ties=sum(v == 0 for v in values.values()),
                scene_losses=sum(v > 0 for v in values.values()), per_scene_epe_delta=values,
                bad3_percentage_point_delta=summary[first]['equal_scene']['bad3']-summary[second]['equal_scene']['bad3'],
                gt_rank_delta=summary[first]['equal_scene']['mean_gt_rank']-summary[second]['equal_scene']['mean_gt_rank'])
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=HERE/'configs/default.json')
    parser.add_argument('--repo-root', type=Path, default=HERE.parents[3])
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    repo = args.repo_root.resolve()
    manifest_path = repo/config['manifest']
    assert digest(manifest_path) == config['manifest_sha256']
    cohort = json.loads(manifest_path.read_text())['dev']
    assert len(cohort)==config['expected_frames'] and len({r['scene'] for r in cohort})==config['expected_scenes']
    assert all(r['depth'].startswith('Train/') for r in cohort)
    data = repo/config['dataset_root']
    raw = repo/'storage/experiment-runs/snn-depth-estimation/temporal-aggregation-cpu-v1'/args.run_id
    compact = HERE/'results'/args.run_id
    raw.mkdir(parents=True, exist_ok=False)
    compact.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    provenance = dict(started_utc=datetime.now(timezone.utc).isoformat(), config=config,
        config_sha256=digest(args.config), git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        dirty_status=subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).splitlines(),
        source_sha256={p.name: digest(p) for p in [Path(__file__), HERE/'scripts/patch_match.py', repo/'projects/snn-depth-estimation/core/utils/reader.py']},
        python=platform.python_version(), numpy=np.__version__, opencv=cv2.__version__, machine=platform.machine(),
        processor=platform.processor(), execution='local CPU only, no scheduler/GPU/model',
        cohort=cohort, raw_location='storage/experiment-runs/snn-depth-estimation/temporal-aggregation-cpu-v1/'+args.run_id)
    write_json(raw/'provenance.json', provenance)
    r = config['patch_radius']
    d = np.arange(config['min_disparity'], config['max_disparity']+1)
    yy, xx = np.meshgrid(np.arange(r, config['shape'][1]-r, config['query_stride']),
                         np.arange(config['max_disparity']+r, config['shape'][2]-r, config['query_stride']), indexing='ij')
    grid = np.stack([yy.ravel(), xx.ravel()], -1)
    frames, support, inputs, checks = [], [], [], []
    started = time.monotonic()
    for frame_index, item in enumerate(cohort):
        if time.monotonic()-started > config['cpu_wall_limit_seconds']:
            write_json(raw/'incomplete.json', dict(completed_frames=frame_index, reason='prospective600secondwalllimit'))
            raise RuntimeError('bounded CPU run incomplete; no cohort or resource extension')
        paths = {key: data/item[key] for key in ('spike_l', 'spike_r', 'depth')}
        inputs.append(dict(scene=item['scene'], frame=item['frame'], files={k:dict(path=item[k],sha256=digest(p),bytes=p.stat().st_size) for k,p in paths.items()}))
        left, right = [decode(paths[k], tuple(config['shape'])) for k in ('spike_l', 'spike_r')]
        depth = cv2.imread(str(paths['depth']), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        assert depth is not None and depth.shape[:2] == tuple(config['shape'][1:])
        depth = depth[:, :, 0] if depth.ndim==3 else depth
        depth_at_grid = depth[grid[:, 0], grid[:, 1]]
        positive = np.isfinite(depth_at_grid) & (depth_at_grid > 0)
        truth = np.divide(config['focal_length']*config['baseline'], depth_at_grid,
                          out=np.full(len(grid), np.nan), where=positive)
        native = positive & (truth < 512)
        eligible = native & (truth >= d[0]) & (truth <= d[-1])
        queries, truth = grid[eligible], truth[eligible]
        assert len(queries) > 0
        support.append(dict(scene=item['scene'], frame=item['frame'], geometry_grid_count=len(grid),
            native_valid_grid_count=int(native.sum()), excluded_invalid_depth=int((~positive).sum()),
            excluded_native512=int((positive & ~native).sum()), excluded_gt_out_of_candidate_range=int((native & ~eligible).sum()),
            matched_count=len(queries), query_support_sha256=hashlib.sha256(queries.tobytes()+truth.tobytes()).hexdigest()))
        saved = dict(queries=queries, truth=truth, disparities=d)
        aggregate_left, aggregate_right = left.mean(0), right.mean(0)
        aggregate, aggregate_zero = zncc(aggregate_left, aggregate_right, queries, d, r, config['zncc_denominator_epsilon'])
        conditions = [('aggregate_first', aggregate, aggregate_zero.astype(np.int64), 1)]
        check = dict(scene=item['scene'], frame=item['frame'], joint_permutation_max_score_difference={},
                     aggregate_preservation_max_difference={}, k1_score_max_difference=None)
        for length in config['chunk_lengths']:
            l, rr = chunk_rates(left, length), chunk_rates(right, length)
            permutation = np.roll(np.arange(len(l)), config['right_chunk_cyclic_shift'])
            per_chunk, zero_count = [], np.zeros_like(aggregate_zero, dtype=np.int64)
            for i in range(len(l)):
                scores, zero = zncc(l[i], rr[i], queries, d, r, config['zncc_denominator_epsilon'])
                per_chunk.append(scores)
                zero_count += zero
            per_chunk = np.stack(per_chunk)
            aligned = per_chunk.mean(0)
            if length == 50:
                check['k1_score_max_difference'] = float(np.abs(aligned-aggregate).max())
                assert check['k1_score_max_difference'] == 0
                continue
            shifted, shifted_zeros = np.zeros_like(aggregate), np.zeros_like(zero_count)
            for i in range(len(l)):
                scores, zero = zncc(l[i], rr[permutation[i]], queries, d, r, config['zncc_denominator_epsilon'])
                shifted += scores
                shifted_zeros += zero
            shifted /= len(l)
            joint = per_chunk[permutation].mean(0)
            check['joint_permutation_max_score_difference'][str(length)] = float(np.abs(joint-aligned).max())
            assert np.allclose(joint, aligned, rtol=0, atol=1e-14)
            preservation = max(float(np.abs(l.mean(0)-aggregate_left).max()),
                               float(np.abs(rr[permutation].mean(0)-aggregate_right).max()))
            check['aggregate_preservation_max_difference'][str(length)] = preservation
            assert preservation < 1e-14
            conditions.extend([(f'aligned_L{length}', aligned, zero_count, len(l)),
                               (f'right_shift_L{length}', shifted, shifted_zeros, len(l))])
        for name, scores, zeros, chunks in conditions:
            metrics = score_metrics(scores, truth, d)
            arrays = {k[1:]: metrics.pop(k) for k in list(metrics) if k.startswith('_')}
            saved.update({name+'_'+k: v for k,v in arrays.items()})
            saved[name+'_scores'] = scores
            frames.append(dict(scene=item['scene'], frame=item['frame'], method=name, **metrics,
                degenerate_candidate_chunk_fraction=float(zeros.sum()/(zeros.size*chunks))))
        np.savez_compressed(raw/(item['scene']+'-'+item['frame']+'.npz'), **saved)
        checks.append(check)
        print(json.dumps(dict(completed=frame_index+1,total=len(cohort),scene=item['scene'],frame=item['frame'],queries=len(queries),elapsed_seconds=round(time.monotonic()-started,2))),flush=True)
    summary = reduce_frames(frames)
    result = dict(status='COMPLETED_CPU_DIAGNOSTIC', started_utc=provenance['started_utc'],
        finished_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-started,
        config=config, config_sha256=provenance['config_sha256'], git_commit=provenance['git_commit'],
        source_sha256=provenance['source_sha256'], input_files=inputs,
        method_summary=summary, contrasts=contrasts(summary, config['chunk_lengths']), support=support,
        checks=checks, per_frame=frames, raw_location=provenance['raw_location'],
        interpretation='descriptive classical patch matching; repeated native Dev, limitedrange+geometrysupport, unknown exact depth timestamp; no trained-model result')
    write_json(raw/'result.json', result)
    write_json(compact/'summary.json', result)
    with (compact/'per-frame.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(frames[0]));writer.writeheader();writer.writerows(frames)
    print(json.dumps(dict(status=result['status'],elapsed_seconds=result['elapsed_seconds'],summary=summary,contrasts=result['contrasts'])),flush=True)


if __name__ == '__main__':
    main()
