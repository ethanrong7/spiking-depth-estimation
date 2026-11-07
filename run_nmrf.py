"""NMRF native-spike preflight and fresh 10k-step pilot (CUDA training only)."""
import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import tarfile
import time

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
import numpy as np
import torch
from torch.utils.data import DataLoader

from nmrf_runtime import (VENDOR, UPSTREAM_REVISION, build_native_nmrf,
                          prepare_batch, weighted_loss)
from runtime import NativeDataset

ROOT = Path(__file__).resolve().parent


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['preflight', 'train', 'benchmark'])
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--manifest', type=Path, default=ROOT/'configs/synthetic-split.json')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--steps', type=int, default=10000)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--accumulation', type=int, default=1)
    p.add_argument('--crop', type=int, nargs=2, metavar=('HEIGHT', 'WIDTH'))
    p.add_argument('--learning-rate', type=float, default=0.0002)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--save-every', type=int, default=500)
    p.add_argument('--log-every', type=int, default=20)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--resume', type=Path)
    p.add_argument('--benchmark-warmup', type=int, default=3)
    p.add_argument('--benchmark-steps', type=int, default=10)
    a = p.parse_args()
    if min(a.steps, a.batch_size, a.accumulation, a.save_every, a.log_every) < 1:
        p.error('Steps, batch size, accumulation and intervals must be positive')
    if a.workers < 0 or a.benchmark_warmup < 1 or a.benchmark_steps < 1:
        p.error('Workers must be nonnegative; benchmark intervals must be positive')
    if a.command in ('train', 'benchmark') and a.device != 'cuda':
        p.error('Substantial training requires the explicitly authorized CUDA machine')
    if a.command == 'preflight' and a.device == 'cpu' and not a.crop:
        a.crop = [32, 64]
    if a.device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA is unavailable')
    if a.output.resolve().is_relative_to(ROOT):
        p.error('Keep experiment outputs outside the source repository')
    return a


def record_run(a, cfg, manifest):
    a.output.mkdir(parents=True, exist_ok=False)
    def git(*args):
        if not (ROOT/'.git').exists():
            source = json.loads((ROOT/'deployment-source.json').read_text(encoding='utf-8'))
            keys = {('rev-parse', 'HEAD'): 'revision', ('status', '--short'): 'status',
                    ('diff', 'HEAD'): 'diff'}
            return source[keys[args]]
        return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    config.update(source_revision=git('rev-parse', 'HEAD'),
                  source_status=git('status', '--short'), upstream_revision=UPSTREAM_REVISION,
                  torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
                  device_name=torch.cuda.get_device_name() if a.device == 'cuda' else 'CPU',
                  model='continuous NMRF with 50-channel CNN input',
                  loss_patch='proposal_disp weight renamed to loss_prop',
                  protocol='pilot; not a reproduction of published or historical results',
                  resume_semantics='model/optimizer/scheduler restored; data order restarts from seed')
    if a.resume:
        config['parent_config'] = json.loads((a.resume.parent/'config.json').read_text(encoding='utf-8'))
    if a.device == 'cuda':
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        config['gpu'] = dict(compute_capability=list(torch.cuda.get_device_capability()),
                             compiled_architectures=torch.cuda.get_arch_list(),
                             total_memory_bytes=properties.total_memory,
                             cudnn_version=torch.backends.cudnn.version(),
                             matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                             cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)
        try:
            config['nvidia_smi'] = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
                 '--format=csv,noheader'], text=True, timeout=10).strip()
        except (OSError, subprocess.SubprocessError):
            config['nvidia_smi'] = 'unavailable'
    (a.output/'config.json').write_text(json.dumps(config, indent=2)+'\n', encoding='utf-8')
    (a.output/'nmrf-config.yaml').write_text(cfg.dump(), encoding='utf-8')
    (a.output/'split.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    (a.output/'working-tree.diff').write_text(git('diff', 'HEAD'), encoding='utf-8')
    # Include new/untracked code too: HEAD alone does not identify this runner.
    with tarfile.open(a.output/'source.tar.gz', 'w:gz') as archive:
        for name in ['run_nmrf.py', 'nmrf_runtime.py', 'runtime.py', 'requirements-nmrf.txt',
                     'projects/snn-depth-estimation/core/utils/reader.py',
                     'projects/snn-depth-estimation/core/utils/utils.py']:
            archive.add(ROOT/name, arcname=name)
        archive.add(VENDOR, arcname=str(VENDOR.relative_to(ROOT)),
                    filter=lambda info: None if '__pycache__' in info.name else info)


def emit(a, record):
    line = json.dumps(record)
    print(line, flush=True)
    with (a.output/'metrics.jsonl').open('a', encoding='utf-8') as f:
        f.write(line+'\n')


def save(a, model, optimizer, scheduler, step):
    temporary = a.output/'checkpoint.tmp'
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 'step': step,
                'batch_size': a.batch_size, 'accumulation': a.accumulation,
                'upstream_revision': UPSTREAM_REVISION,
                'model_name': 'nmrf-native-50'}, temporary)
    temporary.replace(a.output/'checkpoint-last.pt')


def main():
    a = arguments()
    torch.set_num_threads(2)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    manifest = json.loads(a.manifest.read_text(encoding='utf-8'))
    if set(r['scene'] for r in manifest['train']) & set(r['scene'] for r in manifest['dev']):
        raise ValueError('Train/Dev scene overlap')
    dataset = NativeDataset(a.data_root, manifest['train'])
    # Fail before recording/allocating a run if its first triplet is unavailable.
    sample = dataset[0]
    if not all(torch.all((s == 0) | (s == 1)) for s in sample[:2]):
        raise ValueError('Native input must contain binary sensor spikes')
    model, criterion, cfg = build_native_nmrf(a.seed)
    cfg.SEED = a.seed
    cfg.SOLVER.MAX_ITER = 1 if a.command == 'preflight' else a.steps
    cfg.SOLVER.BASE_LR = a.learning_rate
    cfg.SOLVER.IMS_PER_BATCH = a.batch_size*a.accumulation
    cfg.DATASETS.TRAIN = ['native-synthetic-manifest']
    cfg.DATASETS.TEST = []
    if a.crop:
        cfg.DATASETS.CROP_SIZE = a.crop
    else:
        cfg.DATASETS.CROP_SIZE = [int(sample[0].shape[-2]), int(sample[0].shape[-1])]
    model.to(a.device); criterion.to(a.device)
    record_run(a, cfg, manifest)
    loader = DataLoader(dataset, batch_size=a.batch_size, shuffle=a.command != 'preflight',
                        num_workers=a.workers, persistent_workers=a.workers > 0,
                        drop_last=a.command != 'preflight')
    if a.command == 'preflight':
        model.train()
        batch = prepare_batch(next(iter(loader)), a.device, a.crop)
        left, right, target, valid, _ = batch
        started = time.monotonic()
        if a.device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        optimizer = torch.optim.AdamW(model.parameters(), lr=a.learning_rate)
        before = model.parent.backbone.conv1.weight.detach().clone()
        outputs = model(left, right)
        if outputs['disp'].shape != target.shape or not torch.isfinite(outputs['disp']).all():
            raise RuntimeError('Invalid training prediction')
        loss, parts = weighted_loss(criterion, outputs, target, valid)
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise RuntimeError('Missing/nonfinite gradients')
        stem_grad = model.parent.backbone.conv1.weight.grad
        if stem_grad is None or not torch.all(stem_grad.abs().sum((0, 2, 3)) > 0):
            raise RuntimeError('Not all 50 sensor channels receive gradient')
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if torch.equal(before, model.parent.backbone.conv1.weight):
            raise RuntimeError('Optimizer did not update input stem')
        del outputs, loss, gradients, stem_grad
        model.eval()
        with torch.no_grad():
            # Exercise the upstream evaluation padding/unpadding on odd sizes.
            prediction = model(left[..., :-1, :-1], right[..., :-1, :-1])['disp']
        if prediction.shape != target[..., :-1, :-1].shape or not torch.isfinite(prediction).all():
            raise RuntimeError('Invalid evaluation prediction/unpadding')
        result = dict(status='passed', input_shape=list(left.shape),
                      eval_shape=list(prediction.shape), train_row=manifest['train'][0],
                      losses={k: float(v.detach()) for k,v in parts.items()},
                      all_50_channels_have_gradient=True, optimizer_steps=1,
                      parameter_count=sum(p.numel() for p in model.parameters()),
                      seconds=time.monotonic()-started,
                      peak_cuda_bytes=(torch.cuda.max_memory_allocated()
                                       if a.device == 'cuda' else None),
                      scope='native-data crop forward/backward/optimizer/eval; no accuracy claim')
        emit(a, result)
        (a.output/'preflight.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=a.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=a.learning_rate,
                  total_steps=a.steps, pct_start=0.05, cycle_momentum=False)
    start_step = 0
    if a.resume:
        checkpoint = torch.load(a.resume, map_location='cpu', weights_only=False)
        if checkpoint['model_name'] != 'nmrf-native-50' or checkpoint['upstream_revision'] != UPSTREAM_REVISION:
            raise ValueError('Incompatible NMRF checkpoint')
        if checkpoint['scheduler']['total_steps'] != a.steps:
            raise ValueError('Resume must retain the original scheduler total steps')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_step = checkpoint['step']
        del checkpoint
        if start_step >= a.steps:
            raise ValueError('Checkpoint has already reached requested steps')
        (a.output/'resume.json').write_text(json.dumps(dict(step=start_step,
            checkpoint=str(a.resume), batch_size=a.batch_size,
            data_order_restarted=True))+'\n', encoding='utf-8')
    model.train()
    iterator = iter(loader)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    last_step = (start_step+a.benchmark_warmup+a.benchmark_steps
                 if a.command == 'benchmark' else a.steps)
    for step in range(start_step+1, last_step+1):
        if a.command == 'benchmark' and step == start_step+a.benchmark_warmup+1:
            torch.cuda.synchronize()
            measured_started = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
        optimizer.zero_grad(set_to_none=True)
        loss_total = 0.
        for _ in range(a.accumulation):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader); batch = next(iterator)
            left, right, target, valid, _ = prepare_batch(batch, a.device, a.crop, True)
            outputs = model(left, right)
            loss, _ = weighted_loss(criterion, outputs, target, valid)
            (loss/a.accumulation).backward()
            loss_total += float(loss.detach())/a.accumulation
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step(); scheduler.step()
        if step == start_step+1 or step % a.log_every == 0 or step == last_step:
            emit(a, dict(step=step, loss=loss_total, grad_norm=float(grad),
                         seconds=time.monotonic()-started,
                         peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                         reserved_cuda_bytes=torch.cuda.max_memory_reserved(),
                         batch_size=a.batch_size,
                         learning_rate=scheduler.get_last_lr()[0]))
        if a.command != 'benchmark' and (step % a.save_every == 0 or step == a.steps):
            save(a, model, optimizer, scheduler, step)
    if a.command == 'benchmark':
        torch.cuda.synchronize()
        elapsed = time.monotonic()-measured_started
        result = dict(batch_size=a.batch_size, workers=a.workers,
                      measured_steps=a.benchmark_steps, seconds=elapsed,
                      steps_per_second=a.benchmark_steps/elapsed,
                      examples_per_second=a.benchmark_steps*a.batch_size*a.accumulation/elapsed,
                      peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                      reserved_cuda_bytes=torch.cuda.max_memory_reserved(),
                      nvidia_smi=subprocess.check_output(['nvidia-smi',
                          '--query-gpu=memory.used,memory.total,utilization.gpu',
                          '--format=csv,noheader'], text=True).strip())
        (a.output/'benchmark.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
        print(json.dumps(result), flush=True)
        return
    (a.output/'completed.json').write_text(json.dumps({'steps': a.steps,
                           'seconds': time.monotonic()-started})+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
