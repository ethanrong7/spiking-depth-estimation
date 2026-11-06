"""Three fresh recipe-aligned 10k runs; historical models, counted effective batch8."""
import argparse
from datetime import datetime, timezone
import getpass
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
import time

from recipe import augment, batch_indices, build_model, digest, load_frozen, paired_flags, validate_config

STARTED = time.monotonic()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def append(path, value):
    with path.open('a') as stream:
        stream.write(json.dumps(value, allow_nan=False) + '\n')


def tensor_hash(state, prefixes=None):
    result = hashlib.sha256()
    for name, value in sorted(state.items()):
        if prefixes is not None and not any(name.startswith(prefix + '.') for prefix in prefixes):
            continue
        result.update(name.encode()); result.update(str(value.dtype).encode())
        result.update(str(tuple(value.shape)).encode()); result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def initialization_proof(driver, mechanism, model, arm, seed):
    """Validate copied tensors, and every baseline tensor for the I-only adapter."""
    import torch
    reference = driver.build_model('original', seed, False)
    shared = ['cnet', 'context_zqr_convs', 'update_block.encoder', 'update_block.disp_head', 'update_block.mask', 'fnet']
    actual = model.state_dict(); original = reference.state_dict()
    names = [name for name in original if arm != 'spiking_normalized' or any(name.startswith(prefix + '.') for prefix in shared)]
    if not all(name in actual and torch.equal(actual[name].cpu(), original[name].cpu()) for name in names):
        raise RuntimeError('fresh common initialization mismatch')
    proof = {'seed': seed, 'common_prefixes': shared, 'common_tensor_count': sum(any(n.startswith(p + '.') for p in shared) for n in original),
             'common_state_sha256': tensor_hash(actual, shared), 'reference_common_state_sha256': tensor_hash(original, shared),
             'matched_reference_tensors': len(names), 'all_original_tensors_equal': arm != 'spiking_normalized',
             'complete_initial_state_sha256': tensor_hash(actual), 'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
             'stored_parameters': sum(p.numel() for p in model.parameters()), 'I_extra_initialization': 'isolated CPU fork seed+19001' if arm == 'original_I' else None}
    del reference
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('base-source', 'config', 'manifest', 'data-root', 'output', 'source-registry', 'registration'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text()); validate_config(config)
    registration = json.loads(args.registration.read_text())
    driver, mechanism, base_registry = load_frozen(args.base_source)
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available(), 'Slurm CUDA required'
    assert torch.cuda.device_count() == 1 and any(gpu in torch.cuda.get_device_name(0) for gpu in ('4090', '5090'))
    assert args.output.name == config['run_id']
    source = json.loads(args.source_registry.read_text())
    overlay_root = args.source_registry.parent
    for name, expected in source['source_files'].items():
        if digest(overlay_root / name) != expected: raise RuntimeError('overlay mismatch: ' + name)
    if source['base_source_identity_sha256'] != base_registry['source_identity_sha256']:
        raise RuntimeError('base capsule identity mismatch')
    if digest(args.manifest) != registration['manifest_sha256'] or digest(args.config) != registration['config_sha256'][config['run_id']][str(config['physical_batch'])]:
        raise RuntimeError('registered config/manifest mismatch')
    manifest = json.loads(args.manifest.read_text())
    assert len(manifest['train']) == 960 and len(manifest['dev']) == 60
    args.output.mkdir(parents=True, exist_ok=False)
    seed = config['seed']; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(8); torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = True
    model = build_model(driver, mechanism, config['model'], seed)
    proof = initialization_proof(driver, mechanism, model, config['model'], seed)
    write(args.output / 'initialization.json', proof)
    model = model.cuda()
    train = driver.ManifestDataset(args.data_root, manifest['train'])
    dev = driver.ManifestDataset(args.data_root, manifest['dev'])
    loader = DataLoader(dev, batch_size=2, num_workers=0, generator=torch.Generator().manual_seed(seed + 991))
    order = torch.randperm(960, generator=torch.Generator().manual_seed(seed)).tolist()
    write(args.output / 'sample-order.json', {'order': order, 'batch': 8, 'updates_per_cycle': 120, 'full_cycles': 83, 'partial_cycle_updates': 40})
    opt = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-5, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=config['lr'], total_steps=10100, pct_start=.01,
                                                  cycle_momentum=False, anneal_strategy='linear', div_factor=25., final_div_factor=1e4)
    metadata = {'config': config, 'command': sys.argv, 'registration': registration, 'source': source,
                'base_source': base_registry, 'manifest_sha256': digest(args.manifest), 'initialization': proof,
                'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'cuda': torch.version.cuda,
                'python': sys.version, 'hostname': socket.gethostname(), 'user': getpass.getuser(),
                'slurm': {k: v for k, v in os.environ.items() if k.startswith('SLURM_')},
                'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                'pip_freeze': subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True),
                'environment': os.environ.get('STEREO_ENV'), 'started_utc': datetime.now(timezone.utc).isoformat(),
                'loss': 'historical masked mean disparity-only sequence L1; same pooled valid-pixel objective for effective8',
                'BN': 'frozen running statistics during training', 'precision': 'FP32 historical cudnn TF32 enabled/matmul disabled',
                'selection': 'final10k primary; best trained Dev over [1000,2500,5000,7500,10000] secondary; initialization0 separate'}
    write(args.output / 'metadata.json', metadata)
    def timing(): return {'elapsed_seconds': time.monotonic() - STARTED, 'timestamp_utc': datetime.now(timezone.utc).isoformat()}
    def checkpoint(step, best):
        state = {'model': model.state_dict(), 'optimizer': opt.state_dict(), 'scheduler': scheduler.state_dict(),
                 'updates': step, 'samples_seen': step * 8, 'config': config, 'order': order, 'best': best,
                 'source': source, 'base_source_identity_sha256': base_registry['source_identity_sha256'],
                 'manifest_sha256': digest(args.manifest), 'initialization': proof,
                 'rng': {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}}
        torch.save(state, args.output / 'last.tmp.pt'); (args.output / 'last.tmp.pt').replace(args.output / 'last.pt')
    # Historical warmed Dev convention; state equality excludes any scientific change.
    warm_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    warm = next(iter(loader)); model.eval()
    with torch.no_grad(): driver.forward(model, warm[0].cuda(), warm[1].cuda(), 32)
    driver.reset_model(model)
    assert all(torch.equal(warm_state[k], v.detach().cpu()) for k, v in model.state_dict().items())
    del warm_state, warm
    evaluation_started = time.monotonic()
    initial = {'update': 0, **driver.evaluate(model, loader, 32), **timing()}
    initial['evaluation_seconds'] = time.monotonic() - evaluation_started
    append(args.output / 'evaluations.jsonl', initial); write(args.output / 'initial.json', initial)
    history = [initial]; best = float('inf'); best_update = None
    import shutil
    for step in range(10000):
        model.train(); model.freeze_bn(); opt.zero_grad(set_to_none=True)
        indices = batch_indices(order, step)
        flags = [paired_flags(seed, step * 8 + i) for i in range(8)]
        begun = time.monotonic()
        items = [augment(train[index], *flag) for index, flag in zip(indices, flags)]
        decode_seconds = time.monotonic() - begun
        valid_count = sum(int(item[3].sum()) for item in items)
        if not valid_count: raise RuntimeError('no valid effective-batch pixels')
        disparity = 0.; train_error = 0.; diagnostics = []
        physical = config['physical_batch']
        for offset in range(0, 8, physical):
            micro = items[offset:offset + physical]
            left, right, target, valid = [torch.stack([item[j] for item in micro]).cuda() for j in range(4)]
            preds, spikes, volts = driver.forward(model, left, right, 16, True)
            micro_count = int(valid.sum()); weight = micro_count / valid_count
            loss = driver.disparity_loss(preds, target, valid) * weight
            if not torch.isfinite(loss): raise FloatingPointError('nonfinite disparity loss')
            loss.backward(); disparity += float(loss.detach())
            train_error += float((preds[-1] - target).abs()[:, 0][valid].sum()) / valid_count
            if hasattr(model.update_block, 'diagnostics'): diagnostics.append(model.update_block.diagnostics())
            driver.reset_model(model)
            del preds, spikes, volts, loss, left, right, target, valid, micro
        grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.))
        if not np.isfinite(grad) or grad <= 0: raise FloatingPointError('invalid gradient')
        lr_used = opt.param_groups[0]['lr']
        opt.step(); scheduler.step(); torch.cuda.synchronize()
        if scheduler.last_epoch != step + 1: raise RuntimeError('scheduler count mismatch')
        row = {'update': step + 1, 'samples_seen': (step + 1) * 8, 'indices': indices, 'augmentation_hv': flags,
               'loss': disparity, 'disparity_loss': disparity, 'neuron_loss': 0., 'train_epe': train_error,
               'valid_pixels': valid_count, 'gradient_norm': grad, 'lr_used': lr_used, 'lr_next': scheduler.get_last_lr()[0], 'lr': scheduler.get_last_lr()[0],
               'seconds': time.monotonic() - begun, 'decode_seconds': decode_seconds, 'peak_memory_mib': torch.cuda.max_memory_allocated() / 2**20,
               'diagnostics': diagnostics, **timing()}
        append(args.output / 'train.jsonl', row)
        append(args.output / 'loss-components.jsonl', {key: row[key] for key in ('update', 'disparity_loss', 'neuron_loss', 'loss', 'valid_pixels')})
        del items
        if step + 1 in config['eval_updates']:
            evaluation_started = time.monotonic()
            result = {'update': step + 1, **driver.evaluate(model, loader, 32), **timing()}
            result['evaluation_seconds'] = time.monotonic() - evaluation_started
            history.append(result); append(args.output / 'evaluations.jsonl', result); write(args.output / 'history.json', history)
            improved = result['mean_scene_epe'] < best
            if improved: best = result['mean_scene_epe']; best_update = step + 1
            checkpoint(step + 1, best)
            if improved: shutil.copy2(args.output / 'last.pt', args.output / 'best.pt')
            print('EVAL', step + 1, result['mean_scene_epe'], flush=True)
    steps = [int(state['step'].item()) for state in opt.state.values() if 'step' in state]
    if not steps or min(steps) != 10000 or max(steps) != 10000 or scheduler.last_epoch != 10000:
        raise RuntimeError('final optimizer/scheduler counters mismatch')
    rows = [json.loads(line) for line in (args.output / 'train.jsonl').read_text().splitlines()]
    if [row['update'] for row in rows] != list(range(1, 10001)) or rows[-1]['samples_seen'] != 80000:
        raise RuntimeError('noncontiguous optimizer update/presentation log')
    finite_model = all(bool(torch.isfinite(value).all()) for value in model.state_dict().values())
    finite_optimizer = all(bool(torch.isfinite(value).all()) for state in opt.state.values() for value in state.values() if torch.is_tensor(value))
    if not finite_model or not finite_optimizer: raise FloatingPointError('nonfinite final model or optimizer tensors')
    write(args.output / 'result.json', {'status': 'COMPLETED', 'updates_completed': 10000, 'samples_seen': 80000,
          'physical_batch': config['physical_batch'], 'effective_batch': 8, 'final_primary': history[-1],
          'best_dev_secondary': min(history[1:], key=lambda row: row['mean_scene_epe']), 'best_update': best_update,
          'initial': initial, 'optimizer_step_min_max': [min(steps), max(steps)], 'scheduler_last_epoch': scheduler.last_epoch,
          'all_training_records_finite': True, 'all_checkpoint_tensors_finite': finite_model and finite_optimizer, 'contiguous_updates': True,
          'training_compute_seconds': sum(row['seconds'] for row in rows),
          'dev_seconds': sum(row['evaluation_seconds'] for row in history), **timing()})


if __name__ == '__main__': main()
