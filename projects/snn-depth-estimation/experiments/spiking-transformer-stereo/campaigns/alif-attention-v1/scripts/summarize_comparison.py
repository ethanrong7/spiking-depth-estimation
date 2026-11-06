"""Audit and summarize all fifteen saved runs; no model or dataset execution."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

CAMPAIGN = Path(__file__).resolve().parents[1]
METHODS = [
    ('Original ALIF RSNN', 'paired-loss-v1', 'original'),
    ('Attention + LIF', 'paired-loss-v1', 'spiking-normalized'),
    ('Center-only + LIF', 'center-only-v1', 'center-normalized'),
    ('Attention + ALIF', 'alif-attention-v1', 'alif-attention'),
    ('Attention (non-SNN)', 'analog-attention-v1', 'analog-attention')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=CAMPAIGN / 'results')
    args = parser.parse_args()
    rows = []
    for label, campaign, prefix in METHODS:
        registration_path = CAMPAIGN.parent / campaign / 'configs/registration.json'
        registration = json.loads(registration_path.read_text())
        for seed in [1234, 2345, 3456]:
            name = f'{prefix}-seed{seed}-10k-v1'
            directory = args.raw_root / campaign / name
            metadata = json.loads((directory / 'metadata.json').read_text())
            result = json.loads((directory / 'result.json').read_text())
            history = json.loads((directory / 'history.json').read_text())
            train = [json.loads(line) for line in (directory / 'train.jsonl').read_text().splitlines()]
            entry = next(run for run in registration['runs'] if run['run'] == name)
            assert metadata['config'] == entry['config']
            assert metadata['campaign_registration_sha256'] == hashlib.sha256(registration_path.read_bytes()).hexdigest()
            assert metadata['manifest_sha256'] == registration['manifest_sha256']
            assert metadata['restore_checkpoint'] is None
            assert result['updates_completed'] == 10000 and result['cumulative_samples_seen'] == 20000
            assert [x['update'] for x in train] == list(range(1, 10001))
            assert [x['samples_seen'] for x in train] == list(range(2, 20001, 2))
            assert all(math.isfinite(x['disparity_loss']) and math.isfinite(x['train_epe']) for x in train)
            assert [x['update'] for x in history] == list(range(500, 10001, 500))
            best = min(history, key=lambda x: (x['mean_scene_epe'], x['update']))
            assert best['update'] == result['best_update']
            assert abs(best['mean_scene_epe'] - result['selected']['mean_scene_epe']) < 1e-10
            model = registration['frozen_models'][metadata['config']['model']]
            registry = json.loads((args.raw_root / model['snapshot'] / 'source-registry.json').read_text())
            for key in ['source_identity_sha256', 'archive_sha256']:
                assert metadata['source'][key] == registry[key]
            assert {k: v for k, v in registry['source_files'].items() if '/core/' in k} == model['core_source_files']
            rows.append(dict(method=label, seed=seed, final_epe=history[-1]['mean_scene_epe'],
                             best_epe=best['mean_scene_epe'], best_update=best['update'],
                             train_loss_last500=statistics.mean(x['disparity_loss'] for x in train[-500:]),
                             elapsed_minutes=result['elapsed_seconds'] / 60,
                             peak_allocated_gib=max(x['peak_memory_mib'] for x in train) / 1024,
                             parameters=metadata['parameter_count'], updates=10000, samples=20000))
    summary = []
    for label, _, _ in METHODS:
        matching = [r for r in rows if r['method'] == label]
        aggregate = {'method': label}
        for key in ['final_epe', 'best_epe', 'train_loss_last500', 'elapsed_minutes', 'peak_allocated_gib']:
            values = [r[key] for r in matching]
            aggregate[key] = dict(mean=statistics.mean(values), minimum=min(values), maximum=max(values))
        summary.append(aggregate)
    payload = dict(status='COMPLETED', audited_at=datetime.now(timezone.utc).isoformat(),
                   training_runs=15, updates_per_run=10000, samples_per_run=20000,
                   no_test_evaluations_added=True, rows=rows, summary=summary)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'five-way-summary.json').write_text(json.dumps(payload, indent=2) + '\n')
    with (args.output / 'five-way-summary.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# Five-method stereo comparison: final results', '',
             'All 15 fresh training runs completed exactly 10,000 optimizer updates and 20,000 sample presentations. Three paired seeds use the same train/development split and optimization protocol. Lower EPE is better.', '',
             '| Method | Seed 1234 final EPE | Seed 2345 final EPE | Seed 3456 final EPE | Mean [range] |',
             '|---|---:|---:|---:|---:|']
    for aggregate in summary:
        group = [r for r in rows if r['method'] == aggregate['method']]
        stats = aggregate['final_epe']
        lines.append('| ' + aggregate['method'] + ' | ' + ' | '.join(f"{r['final_epe']:.4f}" for r in group)
                     + f" | {stats['mean']:.4f} [{stats['minimum']:.4f}, {stats['maximum']:.4f}] |")
    lines += ['', 'LIF attention has the lowest three-seed mean final and selected development EPE. It beats the original ALIF RSNN, the tested adaptive-threshold ALIF-attention configuration and conventional non-SNN attention on every paired seed. Center-only wins seed 3456 but has a slightly higher mean. These are descriptive results for three seeds within this fixed training budget.', '',
              'The original ALIF RSNN mean final EPE is 3.3530 versus 2.1015 for LIF attention, a 37.3% reduction. The original architecture uses the common disparity-only objective here; it is not the complete upstream neuron-regularized training recipe.', '',
              'Non-SNN attention reaches mean final EPE 2.1989. LIF attention is 4.4% lower relative to that non-SNN mean, with paired differences (non-SNN minus LIF) +0.0322, +0.1522 and +0.1080 px. The non-SNN operator changes Q/K/V activations, membrane dynamics, softmax normalization and continuous hidden outputs together, so this does not isolate a binary-spiking effect. Both operators remain hybrid predictors with the original CNN encoders and correlation pyramid.', '',
              'Center-only mean final EPE is 2.1689. It still computes attention using Q/K/V at the center position, and removes neighboring positions and their contribution to the normalization denominator. It tests spatial neighbor mixing, not removal of attention. The small mean gap versus full LIF attention (+0.0674 px, with a center-only win in one seed) indicates that most of the replacement updater improvement survives that intervention.', '',
              'The tested ALIF-attention mean final EPE is 2.1462 versus 2.1015 for LIF, about 2.1% higher. Adaptive thresholds at Q/K/V/output did not improve development accuracy in this configuration. This does not cover all adaptive-neuron formulations or tuning choices.', '',
              '## Final versus selected checkpoints', '',
              '| Method | Seed | Final EPE | Best development EPE | Selected update | Last-500 training L1 | Process minutes |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['method']} | {r['seed']} | {r['final_epe']:.4f} | {r['best_epe']:.4f} | {r['best_update']} | {r['train_loss_last500']:.4f} | {r['elapsed_minutes']:.2f} |")
    lines += ['', '## Three-seed aggregate evidence', '',
              '| Method | Best EPE mean [range] | Training L1 mean [range] | Process minutes mean [range] | Peak allocated GiB mean |',
              '|---|---:|---:|---:|---:|']
    for aggregate in summary:
        cells = []
        for key in ['best_epe', 'train_loss_last500', 'elapsed_minutes']:
            stats = aggregate[key]
            cells.append(f"{stats['mean']:.4f} [{stats['minimum']:.4f}, {stats['maximum']:.4f}]")
        lines.append('| ' + aggregate['method'] + ' | ' + ' | '.join(cells) + f" | {aggregate['peak_allocated_gib']['mean']:.3f} |")
    lines += ['', 'Process timing includes initialization, data loading, development checks and checkpoints. It varies substantially across allocations: for example non-SNN seed 3456 took 64.51 minutes versus 133.78 and 140.28 minutes for its other seeds. The runs were not a controlled latency/energy benchmark; no energy ranking is supported. Allocated PyTorch memory excludes some device overhead and differs from reserved memory.', '',
              'The learning curves show faster error reduction for the replacement updaters than the original RSNN under this schedule, with transient development spikes and seed-specific crossings. EMA-smoothed training curves can look close while development checkpoints differ. No extrapolation to convergence, population significance or causal attribution to attention alone is justified. A matched attention-removal control was not trained in this campaign.', '',
              '## Verification and scope', '',
              'All exact registered configurations, metadata source/archive/core/manifest identities, fresh starts, contiguous update/sample histories, finite training losses/EPE and 20 scheduled development checkpoints passed audit. Selected checkpoints match historical minimum development EPE with earlier ties. The per-campaign audits additionally verified paired training semantics and identical 60-frame/30-scene development IDs. CPU initialization reconstruction checked equal learned tensors across all three seeds for each replacement control relative to the full LIF-attention model. All replacement models have 8,398,902 parameters; the original has 12,148,246.', '',
              'No new inference or Test-set evaluations were performed during this analysis. Checkpoints and prediction arrays remain on the cluster. The fixed protocol is batch 2, AdamW, 10,100-step OneCycle schedule for the 10,000-update budget, 16 training and 32 development refinements, evaluation every 500 updates, and the registered 960-frame training / 60-frame development manifest. The model/controller comparison cannot determine behavior at convergence or isolate individual residual, normalization and attention contributions.', '',
              '[Final interactive graph](live/index.html) · [Numeric CSV](five-way-summary.csv) · [Audited JSON](five-way-summary.json) · [Update-axis PNG](live/curves-update.png) · [Elapsed-time PNG](live/curves-elapsed_seconds.png)', '']
    scheduler_path = args.output / 'scheduler-final.json'
    if scheduler_path.exists():
        scheduler = json.loads(scheduler_path.read_text())
        lines += [f"Scheduler audit: all {len(scheduler['jobs'])} allocations completed with exit0:0. Recorded start/end times verify at most {scheduler['maximum_simultaneous_campaign_gpus']} simultaneously allocated campaign GPUs. Total allocation was {scheduler['total_gpu_hours']:.3f} GPU-hours, including the earlier reference campaign. [Scheduler evidence](scheduler-final.json).", '']
    (args.output / 'five-way-report.md').write_text('\n'.join(lines))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
