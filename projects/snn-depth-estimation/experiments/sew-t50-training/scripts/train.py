#!/usr/bin/env python3
"""Time-budgeted one-GPU A0/A3-T50 training on a fixed 300k schedule."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import time

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

PROJECT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(PROJECT), str(PROJECT / 'core'),
    str(PROJECT / 'experiments/sew-t50-feasibility/scripts'),
    str(PROJECT / 'experiments/sew-t50-memory-stability/scripts')]
from run_feasibility import SyntheticTrainDataset, model_args, parameter_report
from exact_batch_loss import ExactBatchLossAccumulator, replay_loss
from run_memory_stability import configure_runtime, build_optimizer_scheduler
from training_lifecycle import (capture_rng_state, restore_rng_state, reset_report,
    state_digest, _exact_statistics_equal)
from core.spikestereonet import SpikeStereoNet
from core.utils.utils import InputPadder
from spikingjelly.activation_based import functional
from training_state import ShuffledBatchSampler, identity, save_state, restore_state, segment_deadline


def atomic_json(value, path):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def emit(output, event, **fields):
    item = dict(time=time.time(), event=event, job_id=os.environ.get('SLURM_JOB_ID'), **fields)
    line = json.dumps(item, allow_nan=False)
    with (output / 'events.jsonl').open('a') as stream:
        stream.write(line + '\n')
    print(line, flush=True)
    return item


def freeze_manifest(dataset, config):
    scenes = sorted({sample[0].parents[1].name for sample in dataset.samples})
    if len(scenes) != 150:
        raise ValueError(f'expected 150 scenes, got {len(scenes)}')
    ranked = sorted(scenes, key=lambda s: hashlib.sha256(f"1234:{s}".encode()).hexdigest())
    validation_scenes = sorted(ranked[:config['validation_scene_count']])
    training_scenes = sorted(set(scenes) - set(validation_scenes))
    train_indices, validation_indices, records = [], [], []
    for index, sample in enumerate(dataset.samples):
        scene = sample[0].parents[1].name
        validation = scene in validation_scenes
        (validation_indices if validation else train_indices).append(index)
        records.append(dict(split='validation' if validation else 'train',
            files=[dict(path=str(p.relative_to(dataset.root)), bytes=p.stat().st_size)
                   for p in sample]))
    if len(training_scenes) != config['expected_train_scenes']:
        raise ValueError('training scene count changed')
    if len(validation_indices) != config['expected_validation_frames']:
        raise ValueError('validation frame count changed')
    if len(train_indices) != 34425:
        raise ValueError('training frame count changed')
    return dict(training_scenes=training_scenes, validation_scenes=validation_scenes,
                training_frames=len(train_indices), validation_frames=len(validation_indices),
                records=records, input_bins=list(range(50)), augmentation='none'), train_indices, validation_indices


def training_update(model, optimizer, scheduler, batches):
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    first = ExactBatchLossAccumulator()
    train_error_sum, train_pixels = 0.0, 0
    with torch.no_grad():
        for _, left, right, truth, valid in batches:
            left, right, truth, valid = [x.cuda(non_blocking=True) for x in (left, right, truth, valid)]
            if tuple(left.shape) != (2, 50, 250, 400):
                raise ValueError(f'unexpected training input: {tuple(left.shape)}')
            functional.reset_net(model)
            predictions, spikes, voltages = model(left, right, iters=16)
            first.accumulate(spikes, voltages, predictions, truth, valid, [250, 400])
            final_prediction = InputPadder(left.shape, divis_by=4).unpad(predictions[-1])
            mask = (valid >= .5) & (truth[:, 0].abs() < 700)
            errors = (final_prediction[:, 0] - truth[:, 0]).abs()[mask]
            train_error_sum += float(errors.double().sum())
            train_pixels += errors.numel()
            del final_prediction, errors, mask
            functional.reset_net(model)
            del predictions, spikes, voltages, left, right, truth, valid
    stats = first.finalize()
    loss = float(stats.true_loss)
    if not math.isfinite(loss):
        raise FloatingPointError('nonfinite true batch loss')
    second = ExactBatchLossAccumulator()
    for _, left, right, truth, valid in batches:
        left, right, truth, valid = [x.cuda(non_blocking=True) for x in (left, right, truth, valid)]
        functional.reset_net(model)
        predictions, spikes, voltages = model(left, right, iters=16)
        second.accumulate(spikes, voltages, predictions, truth, valid, [250, 400])
        scalar = replay_loss(stats, spikes, voltages, predictions, truth, valid, [250, 400])
        if not bool(torch.isfinite(scalar)):
            raise FloatingPointError('nonfinite replay loss')
        scalar.backward()
        functional.reset_net(model)
        del predictions, spikes, voltages, scalar, left, right, truth, valid
    replay_check = _exact_statistics_equal(stats, second.finalize())
    if not replay_check['all_exact']:
        raise RuntimeError('first-pass/replay statistics mismatch: ' + json.dumps(replay_check))
    # FP64 norm avoids overflow of finite large gradients; clipping threshold is unchanged.
    groups = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach()
        if not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError(f'nonfinite gradient: {name}')
        key = name.split('.')[0]
        groups[key] = groups.get(key, 0.0) + float(gradient.double().square().sum())
    norm = math.sqrt(sum(groups.values()))
    if not math.isfinite(norm) or norm <= 0:
        raise FloatingPointError('invalid global gradient norm')
    scale = min(1.0, 1.0 / (norm + 1e-6))
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.mul_(scale)
    lr_used = optimizer.param_groups[0]['lr']
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    if not reset_report(model)['clean']:
        raise RuntimeError('unclean neuron reset')
    return dict(loss=loss, lr_used=lr_used, lr_next=optimizer.param_groups[0]['lr'],
                train_epe=train_error_sum / train_pixels,
                disparity_loss=float(stats.true_disparity_loss),
                rate_loss=float(stats.true_rate_loss), voltage_loss=float(stats.true_voltage_loss),
                preclip_norm=norm, gradient_groups={k: math.sqrt(v) for k, v in groups.items()},
                seconds=time.monotonic() - started, replay_exact=True,
                peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20)


@torch.no_grad()
def evaluate(model, dataset, indices, output, step, stop, deadline=None, kind='full'):
    """Full held-out validation, streamed per frame, with explicit aggregation."""
    preserved_rng = capture_rng_state()
    model.eval()
    pixel_error = pixels = 0
    bad = {k: 0 for k in (1, 2, 3)}
    frame_sum = frames = 0
    destination = output / f'validation-{kind}-{step:06d}.jsonl'
    temporary = destination.with_suffix('.tmp')
    try:
        with temporary.open('w') as stream:
            loader = DataLoader(Subset(dataset, indices), batch_size=1, shuffle=False,
                                num_workers=2, generator=torch.Generator().manual_seed(1234))
            for names, left, right, truth, valid in loader:
                if stop[0] or (output / 'STOP').exists() or (deadline and time.time() >= deadline):
                    return None
                functional.reset_net(model)
                prediction = model(left.cuda(), right.cuda(), iters=32, test_mode=True)[1]
                prediction = InputPadder(left.shape, divis_by=4).unpad(prediction).cpu()
                if not bool(torch.isfinite(prediction).all()):
                    raise FloatingPointError('nonfinite validation prediction')
                mask = (valid >= .5) & torch.isfinite(truth[:, 0]) & (truth[:, 0].abs() < 192)
                error = (prediction[:, 0] - truth[:, 0]).abs()[mask]
                count = error.numel()
                if not count:
                    raise ValueError('empty validation mask')
                total = float(error.double().sum())
                row_bad = {k: int((error > k).sum()) for k in bad}
                stream.write(json.dumps(dict(sample=names[0], valid_pixels=count,
                    error_sum=total, bad=row_bad)) + '\n')
                pixel_error += total
                pixels += count
                frames += 1
                frame_sum += total / count
                for k in bad:
                    bad[k] += row_bad[k]
                functional.reset_net(model)
                if frames % 100 == 0:
                    emit(output, 'validation_progress', step=step, frames=frames,
                         total_frames=len(indices))
                del prediction, error
        temporary.replace(destination)
        return dict(step=step, kind=kind, frames=frames, valid_pixels=pixels,
                    epe_pixel_mean=pixel_error / pixels, epe_frame_mean=frame_sum / frames,
                    **{f'bad{k}_percent': 100 * bad[k] / pixels for k in bad})
    finally:
        functional.reset_net(model)
        model.train()
        model.freeze_bn()
        restore_rng_state(preserved_rng)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--provenance', type=Path, required=True)
    parser.add_argument('--resume-verified', action='store_true')
    parser.add_argument('--segment-seconds', type=int, required=True)
    args = parser.parse_args()
    if not 600 <= args.segment_seconds <= 18000:
        parser.error('each GPU allocation must have a finite budget of at most five hours')
    # os.execv for the resume check must preserve this allocation's original deadline.
    deadline = segment_deadline(args.segment_seconds, time.time(),
        os.environ.get('SPIKESTEREO_SEGMENT_DEADLINE'), os.environ.get('SLURM_JOB_END_TIME'))
    os.environ['SPIKESTEREO_SEGMENT_DEADLINE'] = str(deadline)
    training_deadline = deadline - 300
    config = json.loads(args.config.read_text())
    if config['gpu_count'] != 1 or torch.cuda.device_count() != 1:
        raise RuntimeError('exactly one visible GPU is required')
    if '5090' not in torch.cuda.get_device_name(0):
        raise RuntimeError('requires RTX 5090')
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / '.training.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    stop = [False]
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(signum, lambda *_: stop.__setitem__(0, True))
    configure_runtime(config['seed'])
    torch.backends.cudnn.allow_tf32 = config.get('runtime', {}).get('cudnn_tf32', True)
    provenance = json.loads(args.provenance.read_text())
    for name, digest in provenance['source_files'].items():
        if hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'source changed: {name}')
    dataset = SyntheticTrainDataset(args.data_root)
    manifest, training, validation = freeze_manifest(dataset, config)
    by_scene = {}
    for index in validation:
        by_scene.setdefault(dataset.samples[index][0].parents[1].name, []).append(index)
    probe = [indices[int(position)] for indices in by_scene.values()
             for position in np.linspace(0, len(indices) - 1, config['probe_frames_per_scene'], dtype=int)]
    manifest['probe_samples'] = [str(dataset.samples[i][0].relative_to(dataset.root)) for i in probe]
    binding = dict(source=identity(provenance), config=identity(config), manifest=identity(manifest))
    manifest_path = output / 'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise RuntimeError('dataset manifest changed')
    atomic_json(manifest, manifest_path)
    atomic_json(config, output / 'config.json')
    shutil.copyfile(args.provenance, output / 'provenance.json')
    model = SpikeStereoNet(model_args(config)).cuda()
    if parameter_report(model)['total'] != config['expected_parameters']:
        raise RuntimeError('architecture parameter count changed')
    model.train()
    model.freeze_bn()
    optimizer, scheduler = build_optimizer_scheduler(model, config)
    sampler = ShuffledBatchSampler(len(training), config['seed'])
    latest = output / 'latest.pth'
    best = None
    step = 0
    verified = False
    restored_rng = None
    if latest.exists():
        saved = restore_state(latest, model=model, optimizer=optimizer, scheduler=scheduler,
                              sampler=sampler, config=config, binding=binding)
        step, best = saved['step'], saved['best']
        restored_rng = saved['rng']
        verified = bool(saved['resume_verified']) or args.resume_verified
        # Exact tensor/state comparisons at each process restart, before training resumes.
        for name, actual in [('model', model.state_dict()), ('optimizer', optimizer.state_dict()),
                             ('scheduler', scheduler.state_dict())]:
            if state_digest(actual) != state_digest(saved[name]):
                raise RuntimeError(f'{name} restoration differs')
        emit(output, 'restored', step=step, next_lr=saved['next_lr'],
             state_equal=True, sampler=sampler.state_dict())
        del saved
    elif args.resume_verified:
        raise RuntimeError('resume test checkpoint is absent')
    loader = DataLoader(Subset(dataset, training), batch_sampler=sampler,
                        num_workers=config['workers'], pin_memory=True,
                        generator=torch.Generator().manual_seed(config['seed']), persistent_workers=True)
    iterator = iter(loader)
    if restored_rng is not None:
        restore_rng_state(restored_rng)
    emit(output, 'start', step=step, binding=binding, parameters=parameter_report(model),
         gpu=torch.cuda.get_device_name(0), torch_version=torch.__version__,
         python=sys.version, resume_verified=verified,
         segment_deadline_epoch=deadline, training_deadline_epoch=training_deadline,
         schedule_horizon=scheduler.total_steps, next_lr=optimizer.param_groups[0]['lr'])

    def save(milestone=False):
        functional.reset_net(model)
        if latest.exists():
            previous = output / 'previous.pth'
            temporary_previous = output / 'previous.tmp'
            if temporary_previous.exists():
                temporary_previous.unlink()
            os.link(latest, temporary_previous)
            temporary_previous.replace(previous)
        save_state(latest, model=model, optimizer=optimizer, scheduler=scheduler,
                   sampler=sampler, step=step, config=config, binding=binding,
                   rng=capture_rng_state(), best=best, resume_verified=verified)
        if milestone:
            destination = output / f'step-{step:06d}.pth'
            if not destination.exists():
                os.link(latest, destination)
        emit(output, 'checkpoint', step=step, next_lr=optimizer.param_groups[0]['lr'],
             milestone=milestone)

    if step == 0:
        save()

    def validation_boundary(kind='full'):
        nonlocal best
        save()
        atomic_json(dict(status='VALIDATING', step=step, kind=kind,
            next_lr=optimizer.param_groups[0]['lr']), output / 'status.json')
        result = evaluate(model, dataset, validation if kind == 'full' else probe,
                          output, step, stop, deadline - 45, kind)
        if result is None:
            save()
            atomic_json(dict(status='PAUSED', step=step,
                next_lr=optimizer.param_groups[0]['lr']), output / 'status.json')
            return False
        atomic_json(result, output / f'validation-{kind}-{step:06d}.json')
        emit(output, 'validation', **result)
        if kind == 'full' and step > 0 and (best is None or result['epe_pixel_mean'] < best['epe_pixel_mean']):
            best = result
            save()
            shutil.copyfile(latest, output / 'best.tmp')
            (output / 'best.tmp').replace(output / 'best.pth')
        save()
        return True

    def pause(reason):
        save()
        if not stop[0] and not (output / 'STOP').exists() and time.time() < deadline - 90:
            validation_boundary('probe')
        save()
        status = dict(status='PAUSED_BUDGET' if reason == 'time_budget' else 'PAUSED',
            reason=reason, step=step, next_lr=optimizer.param_groups[0]['lr'],
            checkpoint=str(latest), resume_verified=verified)
        atomic_json(status, output / 'status.json')
        emit(output, 'paused', **status)

    if step == 0 and not validation_boundary('probe'):
        pause('signal_or_budget')
        return
    # A stop/crash during validation must not silently skip that selection point.
    if step > 0 and step % config['validation_every'] == 0:
        if not validation_boundary():
            return
    while step < config['schedule_num_steps']:
        if stop[0] or (output / 'STOP').exists() or time.time() >= training_deadline:
            pause('time_budget' if time.time() >= training_deadline else 'signal_or_STOP')
            return
        torch.cuda.reset_peak_memory_stats()
        batches = [next(iterator) for _ in range(4)]
        sample_ids = [name for batch in batches for name in batch[0]]
        record = training_update(model, optimizer, scheduler, batches)
        del batches
        step += 1
        sampler.commit_update()
        if scheduler.last_epoch != step:
            raise RuntimeError('scheduler and optimizer update counts differ')
        emit(output, 'update', step=step, sample_ids=sample_ids, **record)
        atomic_json(dict(status='TRAINING', step=step,
            next_lr=record['lr_next'], loss=record['loss'], resume_verified=verified), output / 'status.json')
        if step == config['resume_check_update'] and not verified:
            save()
            emit(output, 'fresh_process_resume_check', step=step)
            iterator._shutdown_workers()
            # Same job and one GPU; this replaces the process, never starts a second worker.
            os.execv(sys.executable, [sys.executable, '-u', str(Path(__file__).resolve()),
                *sys.argv[1:], '--resume-verified'])
        if step % config['recovery_every'] == 0 or step in config['milestones']:
            save(milestone=step in config['milestones'])
        if step % config['validation_every'] == 0:
            if not validation_boundary():
                pause('signal_or_budget')
                return
        elif step % config['probe_every'] == 0:
            if not validation_boundary('probe'):
                pause('signal_or_budget')
                return
    save(milestone=True)
    atomic_json(dict(status='COMPLETED', step=step, best=best), output / 'status.json')


if __name__ == '__main__':
    main()
