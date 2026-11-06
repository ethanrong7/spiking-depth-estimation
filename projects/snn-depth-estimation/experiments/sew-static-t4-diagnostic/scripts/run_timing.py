#!/usr/bin/env python3
"""Matched full-update GPU timing, preserving every one of the 50 input bins."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import statistics
import sys
import time

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT / 'experiments/sew-t50-training/scripts'))
import train as common
import torch
from torch.utils.data import DataLoader, Subset
from spikingjelly.activation_based import functional


def run_case(args, settings):
    if socket.gethostname() != 'cccl3' or torch.cuda.device_count() != 1:
        raise RuntimeError('requires one allocated GPU on the documented cluster')
    if '5090' not in torch.cuda.get_device_name(0):
        raise RuntimeError('requires RTX 5090')
    provenance = json.loads(args.provenance.read_text())
    for name, expected in provenance['source_files'].items():
        if hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'source changed: {name}')
    case = next(x for x in settings['cases'] if x['name'] == args.case)
    output = args.output / args.case
    output.mkdir(parents=True, exist_ok=False)
    config = {**settings['base_training_config'], **case}
    common.configure_runtime(config['seed'])
    torch.backends.cudnn.allow_tf32 = config['runtime']['cudnn_tf32']
    dataset = common.SyntheticTrainDataset(args.data_root)
    manifest, training, _ = common.freeze_manifest(dataset, config)
    common.atomic_json(manifest, output / 'manifest.json')
    common.atomic_json(config, output / 'config.json')
    model = common.SpikeStereoNet(common.model_args(config)).cuda()
    model.train()
    model.freeze_bn()
    parameters = common.parameter_report(model)
    if parameters['total'] != case['expected_parameters']:
        raise RuntimeError('unexpected architecture size')
    optimizer, scheduler = common.build_optimizer_scheduler(model, config)
    sampler = common.ShuffledBatchSampler(len(training), config['seed'])
    loader = DataLoader(Subset(dataset, training), batch_sampler=sampler,
        num_workers=config['workers'], pin_memory=True, persistent_workers=True,
        generator=torch.Generator().manual_seed(config['seed']))
    iterator = iter(loader)
    records = []
    counts = {}
    total = settings['warmup_updates'] + settings['measured_updates']
    for step in range(1, total + 1):
        batches = [next(iterator) for _ in range(4)]
        ids = [name for b in batches for name in b[0]]
        if step == 1:
            handles = []
            for key in ('fnet', 'cnet'):
                counts[key] = 0
                def count(_module, _inputs, _output, key=key):
                    counts[key] += 1
                if hasattr(getattr(model, key), 'lif1'):
                    handles.append(getattr(model, key).lif1.register_forward_hook(count))
            with torch.no_grad():
                result = model(batches[0][1].cuda(), batches[0][2].cuda(), iters=16)
            del result
            functional.reset_net(model)
            for handle in handles:
                handle.remove()
            if counts != {'fnet': 0 if config['variant'] == 'A2' else case['sew_timesteps'], 'cnet': case['sew_timesteps']}:
                raise RuntimeError(f'wrong actual neuron step counts: {counts}')
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        row = common.training_update(model, optimizer, scheduler, batches)
        torch.cuda.synchronize()
        row['synchronized_seconds'] = time.perf_counter() - started
        row.update(step=step, measured=step > settings['warmup_updates'], sample_ids=ids)
        if not all(row['gradient_groups'].get(k, 0) > 0
                   for k in ('fnet', 'cnet', 'update_block', 'context_zqr_convs')):
            raise RuntimeError('an intended component has zero gradient')
        sampler.commit_update()
        records.append(row)
        with (output / 'updates.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        print(json.dumps(dict(case=args.case, step=step, seconds=row['synchronized_seconds'],
                              loss=row['loss'], replay_exact=row['replay_exact'])), flush=True)
        del batches
    measured = [x['synchronized_seconds'] for x in records if x['measured']]
    result = dict(case=case, actual_forward_steps=counts, parameters=parameters,
        updates=total, measured_updates=len(measured), times_seconds=measured,
        median_seconds=statistics.median(measured), mean_seconds=statistics.mean(measured),
        min_seconds=min(measured), max_seconds=max(measured),
        peak_allocated_mib=max(x['peak_allocated_mib'] for x in records),
        all_replay_exact=all(x['replay_exact'] for x in records),
        sample_sequence_sha256=common.identity([x['sample_ids'] for x in records]),
        manifest_sha256=common.identity(manifest), config_sha256=common.identity(config),
        source_sha256=common.identity(provenance), git_commit=provenance['git_commit'],
        job_id=os.environ['SLURM_JOB_ID'], gpu=torch.cuda.get_device_name(0),
        torch_version=torch.__version__, cuda=torch.version.cuda,
        runtime=dict(matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                     cudnn_tf32=torch.backends.cudnn.allow_tf32,
                     autocast=False, tensors='FP32'),
        checkpoint_scope='engineering diagnostic; separate fresh initialization, not a T50 resume')
    common.atomic_json(result, output / 'result.json')
    common.save_state(output / 'diagnostic-state.pth', model=model, optimizer=optimizer,
        scheduler=scheduler, sampler=sampler, step=total, config=config,
        binding=dict(source=common.identity(provenance), manifest=common.identity(manifest)),
        rng=common.capture_rng_state(), best=None, resume_verified=False)
    iterator._shutdown_workers()


def summarize(args, settings):
    results = [json.loads((args.output / c['name'] / 'result.json').read_text())
               for c in settings['cases']]
    baseline, candidate = settings['reference_baseline'], results[0]
    if any(r['sample_sequence_sha256'] != baseline['sample_sequence_sha256'] for r in results):
        raise RuntimeError('timing cases did not consume the same samples')
    speedup = baseline['median_seconds'] / candidate['median_seconds']
    result = dict(results=results, historical_median_speedup=speedup,
        historical_mean_speedup=settings['baseline_telemetry_seconds'] / candidate['mean_seconds'],
        reduction_percent=100 * (1 - candidate['median_seconds'] / baseline['median_seconds']),
        same_samples=True, comparison_note='Historical T50 used cuDNN TF32; current T4 uses TF32 disabled for replay reproducibility.', passed=bool(speedup >= settings['minimum_speedup']
            and all(r['all_replay_exact'] for r in results)),
        decision='Report timing result before any longer T4 training; no accuracy conclusion.')
    common.atomic_json(result, args.output / 'summary.json')
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--provenance', type=Path, required=True)
    parser.add_argument('--case')
    args = parser.parse_args()
    settings = json.loads(args.config.read_text())
    if args.case:
        run_case(args, settings)
    else:
        summarize(args, settings)


if __name__ == '__main__':
    main()
