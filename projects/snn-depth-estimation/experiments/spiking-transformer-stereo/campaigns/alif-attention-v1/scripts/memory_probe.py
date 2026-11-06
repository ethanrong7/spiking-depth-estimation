"""Slurm-only synthetic full-resolution forward/backward memory gate."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import run_alif
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-registry', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    assert torch.cuda.device_count() == 1 and '5090' in torch.cuda.get_device_name(0)
    registry = json.loads(args.source_registry.read_text())
    source_root = args.source_registry.parent
    for relative, expected in registry['source_files'].items():
        assert hashlib.sha256((source_root / relative).read_bytes()).hexdigest() == expected, relative
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    model = run_alif.build_model('alif_attention', 1234, True).cuda()
    assert sum(p.numel() for p in model.parameters()) == 8398902
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002, weight_decay=1e-5, eps=1e-8)
    left = torch.randint(0, 2, (2, 50, 250, 400), device='cuda').float()
    right = torch.randint(0, 2, left.shape, device='cuda').float()
    target = torch.full((2, 1, 250, 400), -1.0, device='cuda')
    valid = torch.ones((2, 250, 400), dtype=torch.bool, device='cuda')
    torch.cuda.reset_peak_memory_stats()
    model.train()
    model.freeze_bn()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        predictions, _, _ = run_alif.driver.forward(model, left, right, 16, training=True)
        loss = run_alif.driver.disparity_loss(predictions, target, valid)
        assert torch.isfinite(loss)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        assert torch.isfinite(norm)
        optimizer.step()
        del predictions, loss
    model.eval()
    with torch.no_grad():
        prediction = run_alif.driver.forward(model, left, right, 32)
        assert prediction.shape == target.shape and torch.isfinite(prediction).all()
    torch.cuda.synchronize()
    total = torch.cuda.get_device_properties(0).total_memory
    reserved = torch.cuda.max_memory_reserved()
    passed = reserved < total * 0.9
    result = {'passed': passed, 'synthetic_only': True, 'dataset_accessed': False,
              'seed': 1234, 'training_steps': 2, 'batch': 2, 'input': [50, 250, 400],
              'train_refinements': 16, 'eval_refinements': 32,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': reserved,
              'device_total_bytes': total, 'gpu': torch.cuda.get_device_name(0),
              'slurm_job_id': os.environ['SLURM_JOB_ID'], 'source_identity_sha256': registry['source_identity_sha256'],
              'timestamp_utc': datetime.now(timezone.utc).isoformat()}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    if not passed:
        raise RuntimeError('memory gate failed: less than 10 percent reserved-memory headroom')


if __name__ == '__main__':
    main()
