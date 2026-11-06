"""Summarize complete, parity-verified paired phase-2 endpoint replay JSONs."""
import argparse
import json
import math
from pathlib import Path
import numpy as np

SEEDS = (1234, 2345, 3456)
ARMS = ('Original', 'SC', 'SP', 'CC', 'CP')
CONTRASTS = {'SP-SC': {'SP': 1, 'SC': -1}, 'CC-SC': {'CC': 1, 'SC': -1},
             'CP-CC': {'CP': 1, 'CC': -1}, 'CP-SP': {'CP': 1, 'SP': -1},
             'interaction': {'SP': 1, 'SC': -1, 'CP': -1, 'CC': 1},
             **{f'{a}-Original': {a: 1, 'Original': -1} for a in ('SC', 'SP', 'CC', 'CP')}}


def validate(replays, update):
    expected = {(seed, arm) for seed in SEEDS for arm in ARMS}
    if set(replays) != expected:
        raise ValueError(f'incomplete seed/arm set; missing {sorted(expected-set(replays))}')
    reference = None
    manifest = None
    for (seed, arm), replay in replays.items():
        assert (replay['seed'], replay['arm'], replay['update']) == (seed, arm, update)
        assert replay['samples_seen'] == update*8 and replay['parity']['passed']
        if manifest is None:
            manifest = replay['manifest_sha256']
        assert replay['manifest_sha256'] == manifest
        rows = replay['per_frame']
        keyed = {(r['scene'], r['frame']): r for r in rows}
        assert len(keyed) == len(rows) == 60
        scenes = sorted({s for s, _ in keyed})
        assert len(scenes) == 30 and all(sum(s == scene for s, _ in keyed) == 2 for scene in scenes)
        support = {key: (r['target_support_sha256'], r['fine_cohort_sha256']) for key, r in keyed.items()}
        if reference is None:
            reference = support
        assert support == reference, 'target support / fine rank cohort differ'
        for row in rows:
            for iteration in ('1', '2', '4', '8', '16', '32'):
                for stratum in ('full', 'inview', 'oov'):
                    metric = row['iterations'][iteration][stratum]
                    assert metric['valid_pixels'] >= 0
                    if metric['valid_pixels']:
                        assert all(math.isfinite(metric[key]) for key in ('epe', 'bad1', 'bad3', 'bad5'))
                    if stratum == 'full':
                        assert metric['valid_pixels'] > 0
        replays[(seed, arm)]['_keyed'] = keyed
    # Counts match at every refinement/stratum, not only full support.
    ref = replays[(SEEDS[0], 'Original')]['_keyed']
    for replay in replays.values():
        for key, row in replay['_keyed'].items():
            for iteration, strata in row['iterations'].items():
                for name, metric in strata.items():
                    assert metric['valid_pixels'] == ref[key]['iterations'][iteration][name]['valid_pixels']
    return sorted({s for s, _ in reference})


def scene_metric(replay, scene, metric='epe', iteration='32', stratum='full'):
    values = [r['iterations'][iteration][stratum][metric] for r in replay['per_frame']
              if r['scene'] == scene and r['iterations'][iteration][stratum]['valid_pixels']]
    return float(np.mean(values)) if values else None


def contrast_matrix(replays, scenes, coefficients, metric='epe', iteration='32', stratum='full'):
    values = []
    for seed in SEEDS:
        row = []
        for scene in scenes:
            inputs = [(weight, scene_metric(replays[(seed, arm)], scene, metric, iteration, stratum))
                      for arm, weight in coefficients.items()]
            row.append(sum(w*v for w, v in inputs) if all(v is not None for _, v in inputs) else np.nan)
        values.append(row)
    return np.asarray(values)


def classify(interval, seeds, margin=.05):
    low, high = interval
    if low > margin and all(v > 0 for v in seeds):
        return 'reference_advantage'
    if high < -margin and all(v < 0 for v in seeds):
        return 'replacement_advantage'
    if low > -margin and high < margin and all(-margin < v < margin for v in seeds):
        return 'two_sided_practical_equivalence'
    if high < margin and all(v < margin for v in seeds):
        return 'replacement_substitutability'
    return 'unresolved'


def describe(matrix, scenes):
    # All seeds must define a scene's stratum before it contributes.
    valid = np.isfinite(matrix).all(0)
    scene_ids = [s for s, keep in zip(scenes, valid) if keep]
    matrix = matrix[:, valid]
    if not scene_ids:
        return dict(defined_scenes=0, mean_epe_delta=None)
    effects = matrix.mean(0)
    total = float(effects.sum())
    top = np.argsort(-np.abs(effects))[:3]
    leave_out = (total-effects)/(len(effects)-1) if len(effects) > 1 else effects
    return dict(defined_scenes=len(scene_ids), mean_epe_delta=float(effects.mean()),
                seed_means={str(s): float(v) for s, v in zip(SEEDS, matrix.mean(1))},
                scene_deltas=dict(zip(scene_ids, effects.tolist())), median_scene_delta=float(np.median(effects)),
                scene_wins=int((effects < 0).sum()), scene_ties=int((effects == 0).sum()), scene_losses=int((effects > 0).sum()),
                practically_small_scenes=int((np.abs(effects) <= .05).sum()),
                leave_one_scene_out_range=[float(leave_out.min()), float(leave_out.max())],
                three_largest_absolute_contributions=[dict(scene=scene_ids[i], delta=float(effects[i])) for i in top],
                three_largest_signed_share_of_aggregate=float(effects[top].sum()/total) if abs(total) > 1e-12 else None,
                three_largest_share_of_absolute_mass=float(np.abs(effects[top]).sum()/np.abs(effects).sum()) if np.abs(effects).sum() else None)


def rank_contrasts(replays, scenes):
    result = {}
    for seed in SEEDS:
        result[str(seed)] = {}
        for arm in ARMS:
            replay = replays[(seed, arm)]
            per_scene = {}
            for scene in scenes:
                rows = [r for r in replay['per_frame'] if r['scene'] == scene]
                names = ('native',) if arm == 'Original' else ('pre', 'post')
                metrics = {}
                for name in names:
                    ranks = [r['rank'][name] for r in rows]
                    count = sum(r['pixel_count'] for r in ranks)
                    metrics[name] = dict(pixel_count=count,
                        mean_rank=sum(r['rank_sum'] for r in ranks)/count if count else None,
                        top1_fraction=sum(r['top1_count'] for r in ranks)/count if count else None,
                        argmax_error_cells=sum(r['argmax_error_cells_sum'] for r in ranks)/count if count else None)
                if arm != 'Original':
                    metrics['post_minus_pre'] = {k: metrics['post'][k]-metrics['pre'][k] if metrics['post'][k] is not None else None
                                                 for k in ('mean_rank', 'top1_fraction', 'argmax_error_cells')}
                per_scene[scene] = metrics
            result[str(seed)][arm] = per_scene
    return result


def summarize_variation(rows):
    result = dict(batches=len(rows), sampled_rows=sum(len(r['sampled_rows']) for r in rows))
    for key in ('cost_vs_zero_mean_abs', 'fixed_position_row_sd_mean', 'interior_position_sd_mean'):
        values = [r[key] for r in rows if r[key] is not None]
        result[key] = dict(mean=float(np.mean(values)), maximum=max(values)) if values else None
    return result


def summarize(replays, update=30000):
    scenes = validate(replays, update)
    # Whole scenes resampled once per draw, keeping all seeds and both frames.
    indices = np.random.default_rng(20261104).integers(0, len(scenes), size=(20000, len(scenes)))
    result = dict(update=update, seeds=list(SEEDS), scenes=len(scenes), frames_per_scene=2,
                  delta_sign='first arm minus second; positive EPE means reference is better',
                  inference='scene variation conditional on three trained seeds',
                  bootstrap=dict(draws=20000, seed=20261104, unit='scene with all seeds and both frames retained',
                                 primary_interval_percent=97.5, percentile_bounds=[1.25, 98.75]),
                  contrasts={}, endpoint_by_arm={}, rank_per_scene=rank_contrasts(replays, scenes),
                  weight_variation={f'{seed}/{arm}': summarize_variation(r.get('weight_variation', [])) for (seed, arm), r in replays.items()})
    for arm in ARMS:
        result['endpoint_by_arm'][arm] = {metric: float(np.mean([scene_metric(replays[(seed, arm)], scene, metric)
                                                   for seed in SEEDS for scene in scenes])) for metric in ('epe', 'bad1', 'bad3', 'bad5')}
    for name, coefficients in CONTRASTS.items():
        matrix = contrast_matrix(replays, scenes, coefficients)
        entry = describe(matrix, scenes)
        entry['bad_delta_pp'] = {f'bad{k}': float(contrast_matrix(replays, scenes, coefficients, f'bad{k}').mean()) for k in (1, 3, 5)}
        entry['bad3_tradeoff'] = entry['mean_epe_delta'] * entry['bad_delta_pp']['bad3'] < 0
        entry['refinement'] = {iteration: {stratum: describe(contrast_matrix(replays, scenes, coefficients, iteration=iteration, stratum=stratum), scenes)
                                         for stratum in ('full', 'inview', 'oov')} for iteration in ('1', '2', '4', '8', '16', '32')}
        if name in ('SP-SC', 'CC-SC'):
            interval = np.percentile(matrix.mean(0)[indices].mean(1), [1.25, 98.75]).tolist()
            entry['paired_97_5_percent_interval'] = interval
            entry['decision'] = classify(interval, matrix.mean(1).tolist()) if update == 30000 else 'screen_descriptive_only'
        else:
            entry['decision'] = 'descriptive_only'
        result['contrasts'][name] = entry
    result['provenance'] = {f'{seed}/{arm}': {k: r[k] for k in ('checkpoint_sha256', 'source_identity', 'manifest_sha256', 'parity')}
                            for (seed, arm), r in replays.items()}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--update', type=int, choices=(7500, 30000), default=30000)
    a = p.parse_args()
    replays = {}
    for seed in SEEDS:
        for arm in ARMS:
            path = a.raw_root/f'seed-{seed}'/arm/f'replay-{a.update}.json'
            if path.exists():
                replays[(seed, arm)] = json.loads(path.read_text())
    result = summarize(replays, a.update)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(a.output)


if __name__ == '__main__':
    main()
