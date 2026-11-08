"""Matched fine-tuning of native 50-bin NMRF from a common checkpoint.

Each arm uses a fresh AdamW and OneCycle schedule. The selected architecture
is installed before loading weights. Checkpoints store the architecture even
when its weight keys are identical to the dense parent.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tarfile
import time

import torch
from torch.utils.data import DataLoader

from nmrf_selective_updates import (ALTERNATING_SELECTOR, COMPACT_SUPPORT, LEGACY_SELECTOR,
                                    install_selective_updates)

ROOT = Path(__file__).resolve().parent
MODEL = 'nmrf-native-50'
FORMAT = 'nmrf-matched-finetune-v1'


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['preflight', 'train'])
    p.add_argument('--source-root', type=Path, default=ROOT,
                   help='Checkout or portable source containing native NMRF dependencies')
    p.add_argument('--init-checkpoint', type=Path, required=True)
    p.add_argument('--parent-config', type=Path, help='Defaults to init checkpoint config.json')
    p.add_argument('--manifest', type=Path, help='Defaults to init checkpoint split.json')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--variant', choices=['original', 'continuous', 'ternary', 'tanh'], required=True)
    p.add_argument('--keep-fraction', type=float, default=.5)
    p.add_argument('--layers', type=int, nargs='+', default=[1, 2, 3])
    p.add_argument('--threshold', type=float, default=.5)
    p.add_argument('--selector', choices=[LEGACY_SELECTOR, ALTERNATING_SELECTOR],
                   default=LEGACY_SELECTOR)
    p.add_argument('--steps', type=int, required=True)
    p.add_argument('--learning-rate', type=float, required=True)
    p.add_argument('--batch-size', type=int, required=True)
    p.add_argument('--crop', type=int, nargs=2, required=True, metavar=('HEIGHT', 'WIDTH'))
    p.add_argument('--accumulation', type=int, default=1)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--eval-every', type=int, default=0, help='Optional diagnostic interval')
    p.add_argument('--save-every', type=int, default=100)
    p.add_argument('--log-every', type=int, default=20)
    p.add_argument('--memory-fraction', type=float, default=1.)
    p.add_argument('--resume', action='store_true',
                   help='Restore this output directory’s last checkpoint with identical run settings')
    a = p.parse_args()
    for name in ('source_root', 'init_checkpoint', 'output', 'data_root'):
        setattr(a, name, getattr(a, name).resolve())
    a.parent_config = (a.parent_config or a.init_checkpoint.parent/'config.json').resolve()
    a.manifest = (a.manifest or a.init_checkpoint.parent/'split.json').resolve()
    if any(a.output.is_relative_to(root) for root in (ROOT, a.source_root)):
        p.error('Keep run outputs outside the source trees')
    if a.device == 'cpu' and (a.command != 'preflight' or max(a.crop) > 64):
        p.error('CPU is limited to tiny preflights with crops up to 64 pixels')
    if a.device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA unavailable')
    if a.resume and a.command != 'train':
        p.error('Only training runs can resume')
    if a.variant == 'original' and a.selector != LEGACY_SELECTOR:
        p.error('The alternate selector requires a selective variant')
    if min(a.steps, a.batch_size, a.accumulation, a.save_every, a.log_every) < 1:
        p.error('Step, batch and interval counts must be positive')
    if min(a.workers, a.eval_every, a.seed) < 0 or a.seed >= 2**32:
        p.error('Workers, eval interval and seed must be nonnegative, with a 32-bit seed')
    if min(a.crop) < 8 or any(dim % 8 for dim in a.crop):
        p.error('Crop dimensions must be positive multiples of eight')
    if not math.isfinite(a.learning_rate) or a.learning_rate <= 0:
        p.error('Learning rate must be finite and positive')
    if a.steps == 20:
        p.error('OneCycleLR with a 5% warmup requires a horizon other than 20 steps')
    if not 0 < a.memory_fraction <= 1:
        p.error('Memory fraction must be in (0, 1]')
    return a


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def file_digest(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def split_rows(manifest):
    train, dev = manifest['train'], manifest['dev']
    for rows in (train, dev):
        ids = {(r['scene'], r['frame']) for r in rows}
        if not ids or len(ids) != len(rows):
            raise ValueError('Train/Dev must be nonempty with unique scene/frame IDs')
    splits = [train, dev, manifest.get('confirmation', [])]
    for i, rows in enumerate(splits):
        for other in splits[:i]:
            for key in ('scene', 'depth', 'spike_l', 'spike_r'):
                if {r[key] for r in rows} & {r[key] for r in other}:
                    raise ValueError(f'Train/Dev/confirmation split overlap in {key}')
    return train, dev


def matched_batches(size, batch_size, seed, stop):
    """Fixed complete batches, reshuffled per epoch with a dedicated generator."""
    full = size // batch_size
    if full == 0:
        raise ValueError('Training split must fill a batch')
    generator = torch.Generator().manual_seed(seed)
    batches = []
    while len(batches) < stop:
        order = torch.randperm(size, generator=generator).tolist()
        batches.extend(order[i:i+batch_size] for i in range(0, full*batch_size, batch_size))
    return batches[:stop]


def crop_batch(batch, crop, seed, micro_index):
    """One shared stereo crop per microbatch, independent of model/Dev RNG."""
    left, right, flow, valid, indices = batch
    h, w = left.shape[-2:]
    ch, cw = crop
    if h < ch or w < cw:
        raise ValueError(f'Crop {crop} exceeds native dimensions {(h, w)}')
    # CPU generators use the low 32 bits of the seed. Mix both inputs there.
    generator = torch.Generator().manual_seed((seed * 0x9E3779B9 + micro_index) % 2**32)
    y = int(torch.randint(h-ch+1, (), generator=generator))
    x = int(torch.randint(w-cw+1, (), generator=generator))
    region = (..., slice(y, y+ch), slice(x, x+cw))
    return (left[region], right[region], flow[region], valid[region], indices), [y, x]


def install_architecture(model, variant, keep_fraction=.5, layers=(1, 2, 3), threshold=.5,
                         selector=LEGACY_SELECTOR, attention_support=COMPACT_SUPPORT):
    architecture = dict(model_name=MODEL, sensor_bins=50, variant=variant, selective=None)
    if variant != 'original':
        architecture['selective'] = install_selective_updates(
            model, keep_fraction=keep_fraction, coding=variant, layers=layers, threshold=threshold,
            selector=selector, attention_support=attention_support)
    return architecture


def restore_architecture(model, state):
    if state.get('format') != FORMAT:
        raise ValueError('Expected a matched fine-tuning checkpoint')
    architecture = state['architecture']
    selective = architecture['selective'] or {}
    restored = install_architecture(model, architecture['variant'], **{
        key: selective[key] for key in ('keep_fraction', 'layers', 'threshold', 'selector', 'attention_support')
        if key in selective})
    if restored != architecture:
        raise ValueError('Incompatible fine-tuning architecture')
    model.load_state_dict(state['model'], strict=True)


def load_initial_weights(model, state, architecture, parent, upstream):
    if state['model_name'] != MODEL or state['upstream_revision'] != upstream:
        raise ValueError('Initial checkpoint must use the native 50-bin NMRF parent')
    if state.get('format') == FORMAT:
        if (state['architecture'] != architecture or state['run_config'] != parent.get('identity')
                or state['step'] != parent.get('steps')):
            raise ValueError('Fine-tuning initializer must match the architecture and finished parent run')
    elif (state.get('architecture', {}).get('sensor_bins', 50) != 50
          or parent.get('model') not in (MODEL, 'continuous NMRF with 50-channel CNN input')
          or parent.get('steps') != state['step']):
        raise ValueError('Initial checkpoint and parent config are incompatible')
    model.load_state_dict(state['model'], strict=True)


def capture_source(dependency_root, output, vendor, module_files):
    files = {Path(__file__).resolve(): 'run_nmrf_finetune.py',
             ROOT/'nmrf_selective_updates.py': 'nmrf_selective_updates.py'}
    for path in module_files:
        path = Path(path).resolve()
        files[path] = path.relative_to(dependency_root).as_posix()
    for folder in (vendor, dependency_root/'projects/snn-depth-estimation/core/utils'):
        for path in folder.rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:
                files[path] = path.relative_to(dependency_root).as_posix()
    for root in dict.fromkeys((ROOT, dependency_root)):
        receipt = root/'deployment-source.json'
        if receipt.is_file():
            files[receipt] = ('deployment-source.json' if root == ROOT
                              else 'dependency-deployment-source.json')
    if (ROOT/'.git').exists():
        revision = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
        status = subprocess.check_output(['git', '-C', str(ROOT), 'status', '--short'], text=True)
        diff = subprocess.check_output(['git', '-C', str(ROOT), 'diff', 'HEAD'], text=True)
        if output is not None:
            (output/'working-tree.diff').write_text(diff, encoding='utf-8')
        provenance = dict(revision=revision, status=status)
    else:
        provenance = read_json(ROOT/'deployment-source.json')
        if not provenance.get('revision'):
            raise ValueError('Deployment source receipt must identify its source revision')
    if output is not None:
        with tarfile.open(output/'source.tar.gz', 'w:gz') as archive:
            for path, name in files.items():
                archive.add(path, arcname=name)
    # Resume compares the executed files. Unrelated workspace changes can coexist.
    return dict(revision=provenance['revision'],
                files={name: file_digest(path) for path, name in files.items()}), provenance


def save_checkpoint(output, model, optimizer, scheduler, step, config):
    destination = output/'checkpoint-last.pt'
    temporary = destination.with_suffix('.tmp')
    torch.save(dict(format=FORMAT, model_name=MODEL, step=step,
                    architecture=config['architecture'],
                    upstream_revision=config['upstream_revision'],
                    run_config=config['identity'],
                    model=model.state_dict(), optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(), rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state() if config['device'] == 'cuda' else None),
               temporary)
    temporary.replace(destination)


def emit(output, record):
    line = json.dumps(record, allow_nan=False)
    print(line, flush=True)
    with (output/'metrics.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(line+'\n')


@torch.inference_mode()
def evaluate(model, dataset, device, max_disp, output, step):
    from evaluate_nmrf import input_tensors, metric_counts, summarize

    started = time.monotonic()
    frames, scenes = [], defaultdict(list)
    was_training = model.training
    cuda_devices = [torch.cuda.current_device()] if device == 'cuda' else []
    with torch.random.fork_rng(devices=cuda_devices):
        model.eval()
        try:
            for i, row in enumerate(dataset.rows):
                left, right, target, valid = input_tensors(dataset[i], device, None)
                prediction = model(left, right)['disp'][0]
                counts = metric_counts(prediction, target, valid & (target < max_disp))
                if not counts['pixels']:
                    raise ValueError(f'No valid training-range disparity in Dev row {row}')
                frames.append(dict(index=i, scene=row['scene'], frame=row['frame'],
                                   **summarize([counts])))
                scenes[row['scene']].append(counts)
        finally:
            model.train(was_training)
    per_scene = {scene: summarize(counts) for scene, counts in scenes.items()}
    summary = dict(**summarize(frames), step=step, pairs=len(frames), scenes=len(scenes),
                   scene_macro_epe=sum(s['epe'] for s in per_scene.values())/len(scenes),
                   scene_macro_bad3_percent=sum(s['bad3_percent'] for s in per_scene.values())/len(scenes),
                   max_disparity=max_disp, resolution='native',
                   seconds=time.monotonic()-started)
    write_json(output/f'dev-{step:06d}.json', dict(summary=summary, per_scene=per_scene, frames=frames))
    return summary


def prune_logs(output, step):
    """Drop any records written after the last durable optimizer checkpoint."""
    for name in ('train-batches.jsonl', 'metrics.jsonl'):
        path = output/name
        if path.exists():
            lines = []
            for line in path.read_text(encoding='utf-8').splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    break
                if record['step'] <= step:
                    lines.append(line+'\n')
            temporary = path.with_suffix('.tmp')
            temporary.write_text(''.join(lines), encoding='utf-8')
            temporary.replace(path)


def main():
    a = arguments()
    sys.path.insert(1, str(a.source_root))
    import evaluate_nmrf
    import nmrf_runtime
    import runtime

    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if a.device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(a.memory_fraction)
    manifest = read_json(a.manifest)
    train_rows, dev_rows = split_rows(manifest)
    if a.init_checkpoint.parent.joinpath('split.json').is_file():
        parent_split = read_json(a.init_checkpoint.parent/'split.json')
        if manifest != parent_split:
            raise ValueError('Use the initializer’s saved split, including its Train/Dev row order')
    parent = read_json(a.parent_config)
    model, criterion, cfg = nmrf_runtime.build_native_nmrf(a.seed)
    arch = install_architecture(model, a.variant, a.keep_fraction, a.layers, a.threshold, a.selector)
    args = {key: str(value) if isinstance(value, Path) else value
            for key, value in vars(a).items() if key != 'resume'}
    checkpoint_digest = file_digest(a.init_checkpoint)
    manifest_digest = file_digest(a.manifest)
    parent_digest = file_digest(a.parent_config)
    if not a.resume:
        a.output.mkdir(parents=True, exist_ok=False)
    source, provenance = capture_source(
        a.source_root, None if a.resume else a.output, nmrf_runtime.VENDOR,
        [module.__file__ for module in (nmrf_runtime, runtime, evaluate_nmrf)])
    identity = dict(arguments=args, architecture=arch, source=source,
                    init_checkpoint_sha256=checkpoint_digest, split_sha256=manifest_digest,
                    parent_config_sha256=parent_digest, torch=str(torch.__version__), cuda=torch.version.cuda)
    config = dict(**args, architecture=arch, identity=identity, source_provenance=provenance,
                  source_revision=source['revision'], upstream_revision=nmrf_runtime.UPSTREAM_REVISION,
                  model=MODEL, precision='FP32', matmul_tf32=False, cudnn_tf32=False,
                  train_pairs=len(train_rows), dev_pairs=len(dev_rows), final_dev_rows=dev_rows,
                  device_name=torch.cuda.get_device_name() if a.device == 'cuda' else 'CPU',
                  optimizer=dict(type='AdamW', weight_decay=1e-5, betas=[.9, .999], eps=1e-8,
                                 gradient_clip_norm=1.),
                  scheduler=dict(type='OneCycleLR', total_steps=a.steps, peak_lr=a.learning_rate,
                                 pct_start=.05, cycle_momentum=False, div_factor=25.,
                                 final_div_factor=1e4, anneal_strategy='cos'),
                  initialization='weights only, fresh optimizer and schedule',
                  selection='fixed endpoint, full native-resolution Dev at step zero and final step',
                  batch_plan='seeded epoch permutations, drop incomplete batches; one crop per microbatch')
    start = 0
    if a.resume:
        if (a.output/'completed.json').exists():
            raise ValueError('Run already completed')
        saved = read_json(a.output/'config.json')
        state = torch.load(a.output/'checkpoint-last.pt', map_location='cpu', weights_only=True)
        if (saved['identity'] != identity or state['run_config'] != identity
                or state.get('format') != FORMAT or state['architecture'] != arch
                or state['upstream_revision'] != config['upstream_revision']):
            raise ValueError('Resume settings or source differ from the original run')
        model.load_state_dict(state['model'], strict=True)
        start = state['step']
    else:
        state = torch.load(a.init_checkpoint, map_location='cpu', weights_only=True)
        load_initial_weights(model, state, arch, parent, config['upstream_revision'])
        config['init_step'] = state['step']
        cfg.SEED = a.seed
        cfg.SOLVER.MAX_ITER = a.steps
        cfg.SOLVER.BASE_LR = a.learning_rate
        cfg.SOLVER.IMS_PER_BATCH = a.batch_size*a.accumulation
        cfg.DATASETS.CROP_SIZE = a.crop
        cfg.DATASETS.TRAIN = ['native-synthetic-manifest']
        cfg.DATASETS.TEST = []
        (a.output/'nmrf-config.yaml').write_text(cfg.dump(), encoding='utf-8')
        write_json(a.output/'parent-config.json', parent)
        write_json(a.output/'split.json', manifest)
        write_json(a.output/'config.json', config)

    model.to(a.device)
    criterion.to(a.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=a.learning_rate, total_steps=a.steps, pct_start=.05, cycle_momentum=False)
    if a.resume:
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        torch.set_rng_state(state['rng'])
        if a.device == 'cuda':
            torch.cuda.set_rng_state(state['cuda_rng'])
        prune_logs(a.output, start)
    else:
        torch.manual_seed(a.seed)
    del state

    stop = 1 if a.command == 'preflight' else a.steps
    batches = matched_batches(len(train_rows), a.batch_size, a.seed, stop*a.accumulation)
    loader = DataLoader(runtime.NativeDataset(a.data_root, train_rows),
                        batch_sampler=batches[start*a.accumulation:],
                        num_workers=a.workers, pin_memory=a.device == 'cuda',
                        generator=torch.Generator().manual_seed(a.seed))
    dev = runtime.NativeDataset(a.data_root, dev_rows[:1] if a.command == 'preflight' else dev_rows)
    began = time.monotonic()
    if a.device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    if a.command == 'train' and not a.resume:
        summary = evaluate(model, dev, a.device, criterion.max_disp, a.output, 0)
        emit(a.output, dict(event='validation', **summary))
        save_checkpoint(a.output, model, optimizer, scheduler, 0, config)
    model.train()
    if start < stop:
        iterator = iter(loader)
    with (a.output/'train-batches.jsonl').open('a', encoding='utf-8') as trace:
        for step in range(start+1, stop+1):
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.
            used_lr = optimizer.param_groups[0]['lr']
            for micro in range(a.accumulation):
                index = (step-1)*a.accumulation + micro
                batch, origin = crop_batch(next(iterator), a.crop, a.seed, index)
                left, right, target, valid, indices = nmrf_runtime.prepare_batch(batch, a.device)
                outputs = model(left, right)
                loss, _ = nmrf_runtime.weighted_loss(criterion, outputs, target, valid)
                (loss/a.accumulation).backward()
                loss_sum += float(loss.detach())/a.accumulation
                trace.write(json.dumps(dict(step=step, micro=micro, rows=indices.tolist(),
                                            crop_origin=origin))+'\n')
                del batch, left, right, target, valid, outputs, loss, _
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            if step == start+1 or step % a.log_every == 0 or step == stop:
                emit(a.output, dict(event='train', step=step, loss=loss_sum, grad_norm=float(grad),
                                    learning_rate=used_lr, seconds=time.monotonic()-began,
                                    peak_cuda_bytes=torch.cuda.max_memory_allocated() if a.device == 'cuda' else None,
                                    peak_reserved_cuda_bytes=torch.cuda.max_memory_reserved() if a.device == 'cuda' else None))
            if a.command == 'train' and a.eval_every and step < stop and step % a.eval_every == 0:
                summary = evaluate(model, dev, a.device, criterion.max_disp, a.output, step)
                emit(a.output, dict(event='validation', **summary))
            if a.command == 'train' and (step % a.save_every == 0 or step == stop):
                trace.flush()
                save_checkpoint(a.output, model, optimizer, scheduler, step, config)
    optimizer.zero_grad(set_to_none=True)
    final_eval = a.output/f'dev-{stop:06d}.json'
    if a.resume and start == stop and final_eval.is_file():
        summary = read_json(final_eval)['summary']
    else:
        summary = evaluate(model, dev, a.device, criterion.max_disp, a.output, stop)
        emit(a.output, dict(event='validation', **summary))
    completed = dict(status='completed', step=stop, variant=a.variant,
                     command=a.command, seconds=time.monotonic()-began,
                     peak_cuda_bytes=torch.cuda.max_memory_allocated() if a.device == 'cuda' else None,
                     peak_reserved_cuda_bytes=torch.cuda.max_memory_reserved() if a.device == 'cuda' else None)
    write_json(a.output/'completed.json', completed)
    emit(a.output, completed)


if __name__ == '__main__':
    main()
