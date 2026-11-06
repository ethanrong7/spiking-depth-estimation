"""Refresh saved confirmation evidence after explicitly authorized frozen evaluations.

Reads exported predictions only; never runs inference, selects checkpoints, or opens data.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import analyze_results as analysis

FROZEN_SELECTION = '020b74466174848a51720aa35a841bd6f4d8f927ddacc344a267df71010579fe'
CASES = ['scene_0150-0001', 'scene_0170-0001', 'scene_0189-0001']
NAMES = {'original': 'Original ALIF', 'spiking': 'Normalized spiking',
         'analog': 'Normalized analog', 'center': 'Normalized center-only',
         'original_regularized': 'Original\nregularized', 'ssde_inspired': 'Inspired ANN\n(non-faithful)'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    experiment = Path(__file__).resolve().parents[1]
    queue = json.loads((experiment / 'configs/final-run-queue.json').read_text())['runs']
    runs, records, pending = {}, {}, []
    for entry in queue:
        directory = args.runs_root / ('confirmation-' + entry['output'])
        if not (directory / 'result.json').is_file():
            pending.append(directory.name)
            continue
        meta = json.loads((directory / 'metadata.json').read_text())
        result = json.loads((directory / 'result.json').read_text())
        selection = meta['confirmation_selection']
        if selection['selection_sha256'] != FROZEN_SELECTION:
            raise ValueError('confirmation selection differs from authorized frozen record')
        if result['checkpoint_update'] != selection['entry']['selected_update']:
            raise ValueError('exported checkpoint differs from frozen selected update')
        files = sorted((directory / 'predictions').glob('*.npz'))
        if len(files) != 80:
            raise ValueError('confirmation export must contain all 80 frames: ' + directory.name)
        name = f"{entry['model']}_{entry['seed']}"
        runs[name] = directory
        records[name] = dict(run=directory.name, config=meta['config'],
            selected_update=result['checkpoint_update'], predictions=len(files),
            checkpoint_sha256=meta['restore_checkpoint_sha256'],
            evaluation_source_identity_sha256=meta['source']['source_identity_sha256'],
            checkpoint_source_identity_sha256=selection['entry']['checkpoint_source_identity_sha256'],
            manifest_sha256=meta['manifest_sha256'])
    if not runs:
        raise ValueError('no completed confirmation exports')
    args.output.mkdir(parents=True, exist_ok=True)
    reference = 'original_1234' if 'original_1234' in runs else next(iter(runs))
    argv = ['analyze', '--cohort', 'confirmation', '--output', str(args.output),
            '--reference', reference, '--priors', str(next(iter(runs.values())) / 'priors.npz'),
            '--no-plots', '--selected', *CASES]
    for name, directory in runs.items():
        argv += ['--run', name + '=' + str(directory)]
    previous = sys.argv
    try:
        sys.argv = argv
        analysis.main()
    finally:
        sys.argv = previous
    result = json.loads((args.output / 'analysis.json').read_text())
    result['frozen_selection_sha256'] = FROZEN_SELECTION
    result['pending_runs'] = pending
    result['selected_pairwise_comparisons'] = {}
    for seed in (1234, 2345):
        spike = f'spiking_{seed}'
        if spike in runs:
            for control in ('original', 'analog', 'center'):
                ref = f'{control}_{seed}'
                if ref in runs:
                    result['selected_pairwise_comparisons'][spike + '_minus_' + ref] = analysis.paired_bootstrap(result['per_scene'][ref], result['per_scene'][spike])
        main = [f'{model}_{seed}' for model in ('original', 'spiking', 'analog', 'center') if f'{model}_{seed}' in runs]
        if main:
            folder = args.output / f'seed-{seed}'
            folder.mkdir(exist_ok=True)
            for case in CASES:
                maps = {NAMES[name.rsplit('_', 1)[0]] + f'\nseed {seed}': dict(np.load(runs[name] / 'predictions' / (case + '.npz'))) for name in main}
                analysis.save_panel(case, maps, folder / (case + '.png'))
    supplementary = [name for name in ('original_1234', 'original_regularized_1234', 'ssde_inspired_1234') if name in runs]
    if len(supplementary) > 1:
        folder = args.output / 'supplementary-controls'
        folder.mkdir(exist_ok=True)
        for case in CASES:
            maps = {NAMES[name.rsplit('_', 1)[0]]: dict(np.load(runs[name] / 'predictions' / (case + '.npz'))) for name in supplementary}
            analysis.save_panel(case, maps, folder / (case + '.png'))
    table = ['# Saved confirmation metrics', '', '| Method | EPE px | Bad >3 px % | Edge EPE | Correlation | Variance ratio | Centered MAE |', '|---|---:|---:|---:|---:|---:|---:|']
    for name, metrics in result['mean_scene_metrics'].items():
        values = [metrics[key] for key in ('all_epe', 'all_bad3', 'edge_epe', 'spatial_correlation', 'variance_ratio', 'centered_spatial_mae')]
        table.append('| ' + name + ' | ' + ' | '.join(f'{value:.5f}' if value is not None else 'undefined' for value in values) + ' |')
    table += ['', 'EPE is mean scene EPE (two frames per scene); bad-3 is pixel pooled (equal valid counts per frame here). Edge EPE and shape metrics are mean frame/scene statistics, not pixel-pooled regime errors. The analysis JSON separately records pooled regime metrics.', '', f'Completed {len(runs)}/10 registered runs; pending outputs: ' + (', '.join(pending) or 'none') + '.']
    (args.output / 'metrics.md').write_text('\n'.join(table) + '\n')
    summary = dict(scope='Descriptive two-observed-seed means/ranges only; scene-bootstrap intervals condition on frozen checkpoints and frames.', models={}, pending=pending)
    for model in ('original', 'spiking', 'analog', 'center'):
        present = [f'{model}_{seed}' for seed in (1234, 2345) if f'{model}_{seed}' in runs]
        summary['models'][model] = {}
        for metric in ('all_epe', 'all_bad3', 'edge_epe', 'spatial_correlation', 'variance_ratio', 'centered_spatial_mae'):
            values = {name.rsplit('_', 1)[1]: result['mean_scene_metrics'][name][metric] for name in present}
            summary['models'][model][metric] = dict(per_seed=values,
                mean=float(np.mean(list(values.values()))) if len(values) == 2 else None,
                range=[min(values.values()), max(values.values())] if len(values) == 2 else None)
    (args.output / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    (args.output / 'inputs.json').write_text(json.dumps(dict(frozen_selection_sha256=FROZEN_SELECTION, cases=CASES, runs=records, pending=pending), indent=2) + '\n')
    (args.output / 'repeat-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (args.output / 'scene-distribution.json').write_text(json.dumps(result['scene_epe_distribution'], indent=2) + '\n')


if __name__ == '__main__':
    main()
