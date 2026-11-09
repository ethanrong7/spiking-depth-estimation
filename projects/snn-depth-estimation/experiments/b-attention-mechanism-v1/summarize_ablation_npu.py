"""Regenerate graph-ready tables from saved NPU ablations; never select best Dev."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

SEEDS = (1234, 2345)
VARIANTS = ('linear-identity', 'signed-surrogate', 'linear-surrogate', 'signed-identity')
EVALS = [0, 250, 500, 1000, 1500, 2500, *range(5000, 30001, 2500)]
RULES = dict(zip(VARIANTS, [('linear', 'identity'), ('signed', 'surrogate'),
                           ('linear', 'surrogate'), ('signed', 'identity')]))


def rows(path):
    if not path.exists():
        return []
    # Writers append one complete line per result. Ignore only an in-flight last line.
    text = path.read_text()
    lines = text.splitlines()
    if lines and not text.endswith('\n'):
        lines.pop()
    return [json.loads(line) for line in lines if line.strip()]


def table(path, fields, values):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)
    temporary.replace(path)


def checked_evaluation(row):
    frames = row['per_frame']
    assert len(frames) == 60 and len({(f['scene'], f['frame']) for f in frames}) == 60
    scenes = {}
    for frame in frames:
        assert frame['valid_pixels'] > 0 and math.isfinite(frame['epe']) and math.isfinite(frame['bad3'])
        scenes.setdefault(frame['scene'], []).append(frame['epe'])
    assert len(scenes) == 30 and all(len(v) == 2 for v in scenes.values())
    scene_means = {s: statistics.mean(v) for s, v in scenes.items()}
    epe = statistics.mean(scene_means.values())
    count = sum(f['valid_pixels'] for f in frames)
    bad3 = sum(f['bad3'] * f['valid_pixels'] for f in frames) / count
    assert math.isclose(epe, row['mean_scene_epe'], rel_tol=1e-10, abs_tol=1e-10)
    assert math.isclose(bad3, row['bad3'], rel_tol=1e-10, abs_tol=1e-10)
    assert scene_means.keys() == row['scene_epe'].keys()
    assert all(math.isclose(v, row['scene_epe'][s], rel_tol=1e-10, abs_tol=1e-10)
               for s, v in scene_means.items())
    return epe, bad3, scene_means


def contrasts(c, h, g, s):
    return {'hybrid_package': c-h, 'forward_at_identity': c-s,
            'forward_at_surrogate': g-h, 'surrogate_at_linear': c-g,
            'surrogate_at_signed': s-h, 'interaction': s+g-h-c}


def audit_training(path, seed):
    """Check original absolute sample/flip addressing and scheduler per saved row."""
    train = rows(path/'train.jsonl')
    metadata_path = path/'metadata.json'
    if not metadata_path.exists():
        assert not train
        return dict(updates=0, stream_identity=None)
    metadata = json.loads(metadata_path.read_text())
    order = metadata['order']
    assert len(order) == 960 and sorted(order) == list(range(960))
    digest = hashlib.sha256()
    for update, row in enumerate(train, 1):
        assert row['update'] == update and row['samples_seen'] == update*8
        assert math.isfinite(row['loss']) and math.isfinite(row['gradient_norm'])
        positions = list(range((update-1)*8, update*8))
        assert row['indices'] == [order[p % 960] for p in positions]
        expected_flags = []
        for p in positions:
            value = hashlib.sha256(f'recipe-hv-v1:{seed}:{p}'.encode()).digest()
            expected_flags.append([value[0] < 128, value[1] < 128])
        assert row['augmentation_hv'] == expected_flags
        assert (row['physical_batch'], row['accumulation'], row['world_size']) == (4, 2, 1)
        # Preserve PyTorch OneCycle's phase endpoints: 299 and 29999.
        def lr(step):
            if step <= 299:
                return 8e-6+(2e-4-8e-6)*step/299
            return 2e-4+(8e-10-2e-4)*(step-299)/29700
        assert math.isclose(row['lr_used'], lr(update-1), rel_tol=1e-7, abs_tol=1e-12)
        assert math.isclose(row['lr_next'], lr(update), rel_tol=1e-7, abs_tol=1e-12)
        digest.update(json.dumps({k: row[k] for k in ('update', 'indices', 'augmentation_hv',
                                                     'lr_used', 'lr_next')}, sort_keys=True).encode())
    return dict(updates=len(train), stream_identity=digest.hexdigest())


def summarize(root):
    root = Path(root)
    destination = root/'summaries'
    destination.mkdir(parents=True, exist_ok=True)
    endpoints, curves, deltas, effects = [], [], [], []
    audit, values = [], {}
    for seed in SEEDS:
        for variant in VARIANTS:
            path = root/f'seed-{seed}'/variant
            record = dict(seed=seed, variant=variant, status='pending')
            try:
                record.update(audit_training(path, seed))
                if (path/'metadata.json').exists():
                    metadata = json.loads((path/'metadata.json').read_text())
                    config = metadata['config']
                    assert config['seed'] == seed and config['variant'] == variant
                    assert (config['forward_rule'], config['backward_rule']) == RULES[variant]
                    record['initial_state_sha256'] = metadata['initial_state_sha256']
                    record['source_identity'] = metadata['source_registry']['source_identity_sha256']
                evaluations = rows(path/'evaluations.jsonl')
                assert len({r['update'] for r in evaluations}) == len(evaluations)
                for row in evaluations:
                    assert row['update'] in EVALS and row['samples_seen'] == row['update']*8
                    epe, bad3, scenes = checked_evaluation(row)
                    values[seed, variant, row['update']] = (epe, bad3, scenes)
                    base = dict(study=root.name, backend='npu', seed=seed, variant=variant,
                                update=row['update'], samples_seen=row['samples_seen'])
                    for metric, value, unit in [('mean_scene_epe', epe, 'px'), ('bad3', bad3, 'percent')]:
                        curves.append(dict(**base, metric=metric, value=value, unit=unit))
                    if row['update'] == 30000:
                        metadata = json.loads((path/'metadata.json').read_text())
                        forward, backward = RULES[variant]
                        endpoints.append(dict(study=root.name, backend='npu', seed=seed, variant=variant,
                            forward_rule=forward, backward_rule=backward, update=30000,
                            mean_scene_epe=epe, bad3_percent=bad3, reuse=False,
                            source_identity=metadata['source_registry']['source_identity_sha256']))
                if (path/'result.json').exists():
                    result = json.loads((path/'result.json').read_text())
                    assert result['status'] == 'COMPLETED' and result['updates'] == record['updates'] == 30000
                    assert result['finite_model_optimizer'] and result['scheduler_last_epoch'] == 30000
                    assert sorted(r['update'] for r in evaluations) == EVALS
                    record['status'] = 'completed'
                elif evaluations or record['updates']:
                    record['status'] = 'partial'
            except (AssertionError, KeyError, ValueError) as error:
                record.update(status='audit_failed', error=repr(error))
            audit.append(record)
        for update in EVALS:
            entries = [values.get((seed, variant, update)) for variant in VARIANTS]
            if all(entry is not None for entry in entries):
                for name, value in contrasts(*(entry[0] for entry in entries)).items():
                    effects.append(dict(seed=seed, update=update, contrast=name, epe_improvement=value))
            for i, left in enumerate(VARIANTS):
                for right in VARIANTS[i+1:]:
                    a, b = values.get((seed, left, update)), values.get((seed, right, update))
                    if a is not None and b is not None:
                        for scene in sorted(a[2]):
                            deltas.append(dict(seed=seed, update=update, scene=scene, variant_a=left,
                                               variant_b=right, epe_a=a[2][scene], epe_b=b[2][scene],
                                               epe_a_minus_b=a[2][scene]-b[2][scene]))
    for seed in SEEDS:
        initialized = [r for r in audit if r['seed'] == seed and 'initial_state_sha256' in r]
        if len({r['initial_state_sha256'] for r in initialized}) > 1 or len({r['source_identity'] for r in initialized}) > 1:
            for record in initialized:
                record.update(status='audit_failed', error='Paired source/initial-state mismatch')
        complete = [r for r in audit if r['seed'] == seed and r['status'] == 'completed']
        if len({r['stream_identity'] for r in complete}) > 1:
            for record in complete:
                record.update(status='audit_failed', error='Paired presentation/LR stream mismatch')
    table(destination/'endpoints.csv', ['study', 'backend', 'seed', 'variant', 'forward_rule',
          'backward_rule', 'update', 'mean_scene_epe', 'bad3_percent', 'reuse', 'source_identity'], endpoints)
    table(destination/'curves.csv', ['study', 'backend', 'seed', 'variant', 'update', 'samples_seen',
          'metric', 'value', 'unit'], curves)
    table(destination/'scene-deltas.csv', ['seed', 'update', 'scene', 'variant_a', 'variant_b',
          'epe_a', 'epe_b', 'epe_a_minus_b'], deltas)
    table(destination/'factorial-contrasts.csv', ['seed', 'update', 'contrast', 'epe_improvement'], effects)
    status = 'failed' if any(r['status'] == 'audit_failed' for r in audit) else (
        'completed' if all(r['status'] == 'completed' for r in audit) else 'partial')
    (destination/'completion-audit.json').write_text(json.dumps(dict(status=status, cells=audit), indent=2)+'\n')
    lines = ['# NPU B factorial results', '', f'Audit status: **{status}**.', '',
             'Positive contrasts mean lower EPE. Endpoint is fixed at 30,000 updates.', '',
             '| Seed | Variant | EPE | bad3 (%) |', '| --- | --- | ---: | ---: |']
    for row in endpoints:
        lines.append(f"| {row['seed']} | {row['variant']} | {row['mean_scene_epe']:.6f} | {row['bad3_percent']:.4f} |")
    lines += ['', '| Contrast | Seed 1234 | Seed 2345 | Screen |', '| --- | ---: | ---: | --- |']
    for name in contrasts(0, 0, 0, 0):
        pair = [next((r['epe_improvement'] for r in effects if r['seed'] == seed and
                      r['update'] == 30000 and r['contrast'] == name), None) for seed in SEEDS]
        if all(value is not None for value in pair):
            screen = 'positive in both (≥0.05 px)' if min(pair) >= .05 else 'inconclusive / no consistent practical gain'
            lines.append(f'| {name} | {pair[0]:.6f} | {pair[1]:.6f} | {screen} |')
    lines += ['', 'These are two-seed training-intervention effects, not proof of uniquely discrete-spike benefit.',
              'Inspect per-scene deltas and bad3 before drawing a conclusion. Audit failures invalidate interpretation.']
    (destination/'interpretation.md').write_text('\n'.join(lines)+'\n')
    if status == 'failed':
        raise RuntimeError(f'Result audit failed; inspect {destination}/completion-audit.json')
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    print(summarize(parser.parse_args().output_root))
