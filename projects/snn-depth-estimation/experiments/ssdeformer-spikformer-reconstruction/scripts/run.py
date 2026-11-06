"""Shared Synthetic Train/Dev runner; --check is metadata-only, no job submission.

Training requires a prospectively registered source/config/budget and Slurm RTX.
Historical confirmation is never opened by this experiment. No automatic resume.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import socket
import sys
import time

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))
STUDY = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_config(config, manifest_path, executing=False):
    if (config.get('schema') != 'ssdeformer-spikformer-reconstruction-v1'
            or config.get('dataset') != 'SpikeStereoNet_Synthetic'
            or config.get('source_shape') != [50, 250, 400]
            or config.get('test_access') != 'sealed'
            or config.get('training_split') != 'train' or config.get('validation_split') != 'dev'
            or config.get('manifest_sha256') != sha(manifest_path)):
        raise ValueError('Exact Synthetic Train/Dev registration required')
    model = config['model']
    if (model['variant'] not in ('softmax', 'spikformer') or model['width'] != 1024
            or model['heads'] != 8 or model['layers'] != 6 or model['mlp_ratio'] != 4
            or model['neural_steps'] != 50 or model['pyramid'] != [64, 128, 256, 512]):
        raise ValueError('Full registered reconstruction geometry required; tiny fixtures are not experiments')
    for name in ('microbatch', 'effective_batch', 'eval_every'):
        if type(config.get(name)) is not int or config[name] <= 0:
            raise ValueError('Positive integer required: ' + name)
    if (config['microbatch'] > config['effective_batch']
            or config['learning_rate'] <= 0 or config['weight_decay'] < 0
            or config['gradient_clip'] <= 0):
        raise ValueError('Explicit valid batch/optimizer values required')
    manifest = json.loads(Path(manifest_path).read_text())
    scenes, samples = {}, set()
    for split, count in [('train', 960), ('dev', 60)]:
        rows = manifest[split]
        if len(rows) != count:
            raise ValueError('Exact existing Train/Dev cohort required')
        for row in rows:
            scene = row['scene']
            if not scene.startswith('scene_') or scene in scenes and scenes[scene] != split:
                raise ValueError('Scene overlap/identity changed')
            scenes[scene] = split
            identity = (scene, row['frame'])
            if identity in samples:
                raise ValueError('Duplicate sample')
            samples.add(identity)
            for key, subfolder, suffix in [('spike_l', 'spike_l', '_spike_l.dat'),
                    ('spike_r', 'spike_r', '_spike_r.dat'), ('depth', 'depth', '_depth.exr')]:
                if row[key] != f"Train/{scene}/{subfolder}/{row['frame']}{suffix}":
                    raise ValueError('Source role/path changed or Test requested')
    if executing:
        if (config.get('registered') is not True or config.get('gpu_budget_seconds') is None
                or config.get('updates') is None or config.get('max_seconds') is None
                or config['gpu_budget_seconds'] < config['max_seconds']
                or not 0 < config['max_seconds'] <= 3600
                or config['updates'] < 1):
            raise ValueError('Prospective horizon, stage budget and registered=true required before training')
    return manifest


def validate_execution(args, config):
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname() != 'cccl3':
        raise RuntimeError('Substantive execution requires registered cccl3 Slurm allocation')
    import torch
    if (not torch.cuda.is_available() or torch.cuda.device_count() != 1
            or not any(s in torch.cuda.get_device_name(0) for s in ('RTX 4090', 'RTX 5090'))):
        raise RuntimeError('Exactly one authorized RTX4090/5090 required')
    if args.source_registry is None or args.output is None or args.data_root is None:
        raise ValueError('Immutable executing source, separate fresh output and configured source root required')
    registry = json.loads(args.source_registry.read_text())
    source_root = args.source_registry.parent.resolve()
    config_name = str(args.config.resolve().relative_to(source_root))
    if registry['source_files'].get(config_name) != sha(args.config):
        raise ValueError('Config absent from executing source registry')
    for name, digest in registry['source_files'].items():
        if sha(source_root / name) != digest:
            raise ValueError('Executing source changed: ' + name)
    if not registry.get('git_commit') or not registry.get('source_identity_sha256'):
        raise ValueError('Exact base commit and source identity required')
    output, data = args.output.resolve(), args.data_root.resolve()
    if any(output.is_relative_to(p) or p.is_relative_to(output) for p in (source_root, data)):
        raise ValueError('Output must be separate from source and data')
    return registry


def execute(args, config, manifest):
    started = time.monotonic()
    registry = validate_execution(args, config)
    import numpy as np
    import torch
    from torch.nn import functional as F
    from core.ssdeformer_reconstruction import build_from_config
    # Preserve existing Synthetic EXR decoding; never source RGB or other data.
    os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
    from core.utils.reader import read_depth

    def asset(name):
        root = args.data_root.resolve()
        path = root / name
        current = root
        for part in Path(name).parts:
            current = current / part
            if current.is_symlink():
                raise ValueError('Symlinked source not allowed')
        if not path.is_file() or not path.resolve().is_relative_to(root):
            raise ValueError('Missing exact source asset')
        return path

    def sample(row):
        views = []
        for name in ('spike_l', 'spike_r'):
            raw = asset(row[name]).read_bytes()
            if len(raw) != 50 * 250 * 400 // 8:
                raise ValueError('Exactly50 complete packed LSB-first spike planes required')
            array = np.unpackbits(np.frombuffer(raw, np.uint8), bitorder='little').reshape(50, 250, 400)
            views.append(torch.from_numpy(array.astype(np.float32)))
        depth = read_depth(str(asset(row['depth']))).astype(np.float32)
        if depth.shape != (250, 400):
            raise ValueError('Native target geometry changed')
        valid = np.isfinite(depth) & (depth > 0)
        target = np.zeros_like(depth)
        target[valid] = (289.740625 * .08) / depth[valid]
        valid &= target < 512
        if not valid.any():
            raise ValueError('No valid target pixels')
        return (*views, torch.from_numpy(target[None]), torch.from_numpy(valid[None]))

    def batch(rows):
        values = [sample(row) for row in rows]
        return [torch.stack([value[i] for value in values]).cuda() for i in range(4)]

    seed = config['initialization_seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    model = build_from_config(config).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
                                 weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['updates'],
                                                          eta_min=config['learning_rate'] * .1)
    args.output.mkdir(parents=True, exist_ok=False)
    def save_json(name, value):
        with (args.output / name).open('x') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
    metadata = dict(config=config, source_registry=registry, manifest_sha256=sha(args.manifest),
        gpu=torch.cuda.get_device_name(0), job_id=os.environ['SLURM_JOB_ID'],
        torch=torch.__version__, cuda=torch.version.cuda, python=platform.python_version(),
        parameters=sum(p.numel() for p in model.parameters()), physical_time_claim=False,
        command=sys.argv, data_root=str(args.data_root.resolve()),
        validation_scope='historical_Train_derived_Dev; no_confirmation_or_Test',
        source_gpu_readiness='first_execution_requires_registered_resource_preflight',
        elapsed_boundary='entry_to_execute_including_source_verification_and_model_setup')
    save_json('metadata.json', metadata)
    best, completed = float('inf'), 0
    generator = torch.Generator().manual_seed(seed + 991)
    order, cursor = [], 0

    def deadline():
        if time.monotonic() - started > config['max_seconds'] - 30:
            raise TimeoutError('Prospective stage deadline; partial state retained, no automatic retry')

    @torch.no_grad()
    def evaluate(update):
        model.eval()
        scenes = {}
        for row in manifest['dev']:
            deadline()
            left, right, target, valid = batch([row])
            pred = model(left, right)['disparity']
            if not torch.isfinite(pred).all():
                raise FloatingPointError('Nonfinite Val prediction')
            error = (pred - target).abs()[valid]
            entry = scenes.setdefault(row['scene'], [0., 0, 0, 0])
            entry[0] += error.double().sum().item()
            entry[1] += error.numel()
            entry[2] += int((error > 3).sum())
            entry[3] += int(((target > 399) & valid).sum())
        return dict(update=update, mean_scene_epe=sum(s[0] / s[1] for s in scenes.values()) / len(scenes),
            scenes={key: dict(epe=s[0] / s[1], bad3=s[2] / s[1], valid=s[1],
                target_above_matching_support=s[3]) for key, s in scenes.items()})

    def checkpoint(name):
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), config=config, source_identity=registry['source_identity_sha256'],
            completed_updates=completed, best_trained_val=best, order=order, cursor=cursor,
            rng=dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all(),
                     generator=generator.get_state(), numpy=np.random.get_state(), python=random.getstate())),
            args.output / name)
    try:
        for update in range(1, config['updates'] + 1):
            deadline()
            model.train()
            rows = []
            for _ in range(config['effective_batch']):
                if cursor == len(order):
                    order = torch.randperm(len(manifest['train']), generator=generator).tolist()
                    cursor = 0
                rows.append(manifest['train'][order[cursor]])
                cursor += 1
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.
            for first in range(0, len(rows), config['microbatch']):
                subset = rows[first:first + config['microbatch']]
                left, right, target, valid = batch(subset)
                pred = model(left, right)['disparity']
                if not torch.isfinite(pred).all():
                    raise FloatingPointError('Nonfinite Train prediction')
                loss = torch.stack([F.smooth_l1_loss(p[v], t[v])
                    for p, t, v in zip(pred, target, valid)]).mean() * len(subset) / len(rows)
                loss.backward()
                total_loss += loss.item()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            completed = update
            with (args.output / 'training.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(update=update, presentations=update * len(rows),
                    loss=total_loss, gradient_norm=float(norm), elapsed_seconds=time.monotonic() - started)) + '\n')
            if update % config['eval_every'] == 0 or update == config['updates']:
                metric = evaluate(update)
                with (args.output / 'validation.jsonl').open('a') as stream:
                    stream.write(json.dumps(metric) + '\n')
                if metric['mean_scene_epe'] < best:
                    best = metric['mean_scene_epe']
                    checkpoint('best.pt')
                checkpoint('last.pt')
        save_json('result.json', dict(status='COMPLETED_REGISTERED_HORIZON', updates=completed,
            best_trained_val=best, elapsed_seconds=time.monotonic() - started,
            peak_cuda_allocated=torch.cuda.max_memory_allocated(), Test_opened=False))
    except BaseException as error:
        checkpoint('interrupted.pt')
        save_json('failure.json', dict(status='STOPPED_PRESERVE_PARTIAL', updates=completed,
            type=type(error).__name__, error=str(error)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--source-registry', type=Path)
    parser.add_argument('--output', type=Path)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--check', action='store_true')
    modes.add_argument('--train', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    manifest = validate_config(config, args.manifest, args.train)
    if args.check:
        print(json.dumps(dict(status='METADATA_CHECK_PASS', registered=config['registered'],
            variant=config['model']['variant'], train=len(manifest['train']), dev=len(manifest['dev']),
            source_pixels_opened=False, GPU_used=False, training_ready=False)))
    else:
        execute(args, config, manifest)


if __name__ == '__main__':
    main()
