"""Slurm-only complete-model engineering feasibility; no final benchmark access."""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
import traceback


def serial(value):
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    if hasattr(value, 'detach'):
        array = value.detach().float().cpu()
        return serial(array.item() if array.numel() == 1 else array.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--source-registry', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('GPU execution requires an authorized Slurm allocation')
    registry = json.loads(a.source_registry.read_text())
    source = a.source_registry.parent
    for name, expected in registry['source_files'].items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Frozen source changed: ' + name)
    cfg = json.loads(a.config.read_text())
    if cfg['stage'] != 'feasibility' or cfg['data']['purpose'] != 'engineering':
        raise RuntimeError('This driver permits registered engineering feasibility only')
    if cfg['model']['time_steps'] != 50 or cfg['data']['split'] != 'train':
        raise RuntimeError('All 50 sensor steps and engineering Train required')
    a.output.mkdir(parents=True, exist_ok=False)
    result = {'status': 'starting', 'scope': 'engineering; no accuracy or energy verdict',
              'git_commit': registry['git_commit'], 'source_identity': registry['source_identity_sha256'],
              'config': cfg, 'config_sha256': hashlib.sha256(a.config.read_bytes()).hexdigest(),
              'slurm_job_id': os.environ['SLURM_JOB_ID'],
              'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    def save():
        temp = a.output / 'result.tmp'
        temp.write_text(json.dumps(serial(result), indent=2, allow_nan=False) + '\n')
        temp.replace(a.output / 'result.json')
    save()
    try:
        import numpy as np
        import torch
        import torch.nn.functional as F
        from core.study_data import LegacyNativeEngineeringDataset
        from core.study_models import build_model
        from core.study_losses import disparity_loss
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable in registered GPU job')
        device = torch.device('cuda:0')
        props = torch.cuda.get_device_properties(device)
        if not any(name in props.name for name in ('4090', '5090')):
            raise RuntimeError('Unauthorized GPU hardware: ' + props.name)
        random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
        torch.cuda.manual_seed_all(cfg['seed'])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.autograd.set_detect_anomaly(cfg.get('gradient_anomaly_audit', False))
        result['environment'] = {'python': sys.version, 'torch': torch.__version__,
                                 'cuda': torch.version.cuda, 'gpu': props.name,
                                 'gpu_total_bytes': props.total_memory,
                                 'tf32': False, 'precision': cfg['precision']}
        result['nvidia_smi'] = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,uuid,driver_version,power.limit',
                                                       '--format=csv,noheader'], text=True)
        source_root, derived_root = os.environ['STUDY_NATIVE_SOURCE'], os.environ['STUDY_NATIVE_DERIVED']
        dataset = LegacyNativeEngineeringDataset(source_root, derived_root, purpose='engineering',
                                                 split='train', crop=cfg['data'].get('crop'))
        sample = dataset[cfg['data']['sample_index']]
        left, right, target, valid = [sample[k].unsqueeze(0).to(device)
                                      for k in ('left', 'right', 'target', 'valid')]
        left.requires_grad_(True)
        right.requires_grad_(True)
        result['input'] = {'sample_id': sample['sample_id'], 'scene_id': sample['scene_id'],
                           'shape': list(left.shape), 'evidence_scope': sample['metadata']['evidence_scope'],
                           'time_unit': sample['metadata']['time_unit'], 'physical_fps': None,
                           'valid_pixels': int(valid.sum()), 'valid_fraction': float(valid.float().mean()),
                           'disparity_min': float(target[valid].min()), 'disparity_max': float(target[valid].max())}
        model = build_model(cfg['model']).to(device)
        result['parameters'] = sum(v.numel() for v in model.parameters())
        decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
        no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
        opt = torch.optim.AdamW([{'params': decay, 'weight_decay': cfg['optimizer']['weight_decay']},
                                {'params': no_decay, 'weight_decay': 0.}], lr=cfg['optimizer']['lr'])
        if cfg['precision'] != 'float32':
            raise ValueError('Initial feasibility uses explicit FP32; register mixed precision separately')
        model.train()
        rows = []
        result['steps'] = rows
        save()
        for step in range(cfg['warmup_steps'] + cfg['measured_steps']):
            opt.zero_grad(set_to_none=True)
            left.grad = None; right.grad = None
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            out = model(left, right)
            prediction = out['disparity']
            if prediction.shape != target.shape or not torch.isfinite(prediction).all():
                raise RuntimeError('Invalid disparity output shape/values')
            if 'aux_disparities_native' not in out:
                raise RuntimeError('Native-scale auxiliary disparities required by registered full objective')
            loss, loss_terms = disparity_loss(out, target, valid, cfg['model']['max_disparity'])
            torch.cuda.synchronize(device)
            forward_seconds = time.perf_counter() - start
            result['current_forward'] = {'step': step, 'loss': serial(loss), 'loss_terms': serial(loss_terms),
                                         'diagnostics': serial(out.get('diagnostics', {})),
                                         'peak_allocated_bytes': torch.cuda.max_memory_allocated(device)}
            save()
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite engineering loss')
            backward_start = time.perf_counter()
            loss.backward()
            torch.cuda.synchronize(device)
            backward_seconds = time.perf_counter() - backward_start
            result['current_gradient_audit'] = {
                'left_finite_by_time': serial(torch.isfinite(left.grad).all(dim=(0, 2, 3, 4))) if left.grad is not None else None,
                'right_finite_by_time': serial(torch.isfinite(right.grad).all(dim=(0, 2, 3, 4))) if right.grad is not None else None,
                'left_abs_mean_by_time': serial(left.grad.abs().mean(dim=(0, 2, 3, 4), dtype=torch.float64)) if left.grad is not None else None,
                'right_abs_mean_by_time': serial(right.grad.abs().mean(dim=(0, 2, 3, 4), dtype=torch.float64)) if right.grad is not None else None,
                'parameters': {name: {'finite': bool(torch.isfinite(param.grad).all()),
                                      'abs_max': serial(param.grad.abs().max())}
                               for name, param in model.named_parameters() if param.grad is not None}}
            save()
            if left.grad is None or right.grad is None:
                raise RuntimeError('Missing left/right input gradient')
            grad_left = left.grad.abs().mean(dim=(0, 2, 3, 4), dtype=torch.float64)
            grad_right = right.grad.abs().mean(dim=(0, 2, 3, 4), dtype=torch.float64)
            if not torch.isfinite(left.grad).all() or not torch.isfinite(right.grad).all():
                raise RuntimeError('Nonfinite sensor-time gradient')
            optimizer_start = time.perf_counter()
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            if not all(bool(torch.isfinite(g).all()) for g in grads):
                raise RuntimeError('Nonfinite parameter gradient')
            norm = torch.stack([g.double().square().sum() for g in grads]).sum().sqrt()
            if not torch.isfinite(norm):
                raise RuntimeError('Nonfinite FP64 global gradient norm')
            coefficient = (cfg['optimizer']['clip_norm'] / (norm + 1e-6)).clamp(max=1.)
            for gradient in grads:
                gradient.mul_(coefficient.to(gradient.dtype))
            opt.step()
            torch.cuda.synchronize(device)
            optimizer_seconds = time.perf_counter() - optimizer_start
            row = {'step': step, 'warmup': step < cfg['warmup_steps'],
                   'seconds': forward_seconds + backward_seconds + optimizer_seconds,
                   'forward_seconds': forward_seconds, 'backward_seconds': backward_seconds,
                   'optimizer_seconds': optimizer_seconds, 'gradient_clip_coefficient': serial(coefficient),
                   'loss': float(loss.detach()), 'gradient_norm_before_clip': float(norm),
                   'loss_terms': serial(loss_terms), 'occlusion_supervision': 'unavailable; no trusted labels',
                   'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
                   'peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
                   'left_gradient_by_sensor_time': serial(grad_left),
                   'right_gradient_by_sensor_time': serial(grad_right),
                   'diagnostics': serial(out.get('diagnostics', {}))}
            rows.append(row)
            save()
            print(json.dumps({k: v for k, v in row.items() if k not in ('diagnostics', 'left_gradient_by_sensor_time', 'right_gradient_by_sensor_time')}), flush=True)
            del out, prediction, loss
        measured = [row['seconds'] for row in rows if not row['warmup']]
        result['training_microbatch_seconds_median'] = statistics.median(measured)
        result['status'] = 'passed'
        result['limitations'] = ['Repeated single training example; optimizer exposure is engineering only',
                                 'Centered native spike input; no causal paired sensor inference',
                                 'Microbatch one FP32; effective-batch forecast needs accumulation and validation timing',
                                 'No measured inference joules in this stage']
    except Exception as error:
        result['status'] = 'failed_oom' if 'out of memory' in str(error).lower() else 'failed'
        result['error'] = repr(error)
        result['traceback'] = traceback.format_exc()
        save()
        raise
    finally:
        result['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save()


if __name__ == '__main__':
    main()
