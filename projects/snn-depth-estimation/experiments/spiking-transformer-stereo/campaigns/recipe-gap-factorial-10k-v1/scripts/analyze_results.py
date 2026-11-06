"""Audit saved factorial logs and compute preregistered contrasts without inference."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from audit_existing import lr

MODELS = ('original', 'original_I', 'spiking_normalized')
CELLS = ((2, 'none'), (8, 'none'), (2, 'hv'), (8, 'hv'))
EXPOSURE_PAIRS = ((480, 120), (1000, 250), (2500, 625), (5000, 1250), (7500, 1875), (10000, 2500))


def contrasts(values):
    """values maps (batch, augmentation) to EPE or method-minus-Original gap."""
    base, batch, aug, both = (values[k] for k in CELLS)
    b0, b1 = batch-base, both-aug
    a0, a1 = aug-base, both-batch
    interaction = a1-a0
    joint = both-base
    assert math.isclose(joint, b0+a0+interaction, abs_tol=1e-10)
    return dict(batch_without_hv=b0, batch_with_hv=b1, hv_at_b2=a0, hv_at_b8=a1,
                interaction=interaction, joint=joint, average_batch=(b0+b1)/2,
                average_hv=(a0+a1)/2)


def quantile(values, p):
    values = sorted(values)
    x = (len(values)-1)*p
    lo, hi = math.floor(x), math.ceil(x)
    return values[lo] + (values[hi]-values[lo])*(x-lo)


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def write_csv(path, rows):
    if rows:
        with path.open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw', type=Path, required=True, help='Campaign root containing cases/')
    p.add_argument('--registration', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    reg = json.loads(a.registration.read_text())
    a.output.mkdir(parents=True, exist_ok=True)
    missing = [c['id'] for c in reg['cases'] if not (a.raw/'cases'/c['id']/'result.json').exists()]
    if missing:
        (a.output/'analysis-status.json').write_text(json.dumps(dict(status='INCOMPLETE', missing=missing,
            causal_report_available=False), indent=2)+'\n')
        print(f'Incomplete: {len(missing)}/12 cases lack result.json; no factorial report generated.')
        return
    data, trajectories, frames, scenes, gradients, audits = {}, [], [], [], [], []
    common_order = None
    common_source = None
    dev_ids = None
    train_ids = None
    for c in reg['cases']:
        root = a.raw/'cases'/c['id']
        meta = json.loads((root/'metadata.json').read_text())
        result = json.loads((root/'result.json').read_text())
        assert result['status'] == 'COMPLETED' and meta['config'] == result['config'] == c
        assert meta['registration'] == reg
        assert meta['manifest_sha256'] == reg['manifest_sha256']
        assert meta['initial_state_sha256'] == result['initial_state_sha256'] == reg['initial_hashes'][c['model']]
        if common_source is None: common_source = meta['source']
        assert meta['source'] == common_source
        order = meta['order']
        assert sorted(order) == list(range(960))
        if common_order is None: common_order = order
        assert common_order == order
        rows = jsonl(root/'train.jsonl')
        assert [r['update'] for r in rows] == list(range(1, 10001))
        for i, row in enumerate(rows):
            b = c['batch']
            assert row['samples_seen'] == (i+1)*b
            assert row['indices'] == [order[(i*b+j)%960] for j in range(b)]
            flags = []
            for j in range(b):
                h = hashlib.sha256(f'recipe-hv-v1:1234:{i*b+j}'.encode()).digest()
                flags.append([h[0]<128, h[1]<128] if c['augmentation']=='hv' else [False, False])
            assert row['augmentation_hv'] == flags
            assert all(math.isfinite(row[k]) for k in ('loss', 'train_epe', 'gradient_norm', 'lr_used', 'lr_next'))
            assert row['gradient_norm'] > 0 and row['clipped'] == (row['gradient_norm'] > 1)
            assert math.isclose(row['lr_used'], lr(i), rel_tol=1e-12, abs_tol=1e-16)
            assert math.isclose(row['lr_next'], lr(i+1), rel_tol=1e-12, abs_tol=1e-16)
        assert result['updates'] == result['scheduler_last_epoch'] == 10000
        assert result['optimizer_steps'] == [10000, 10000]
        assert result['samples_seen'] == 10000*c['batch'] and result['finite_model_optimizer']
        dev, train = jsonl(root/'evaluations.jsonl'), jsonl(root/'train-evaluations.jsonl')
        assert [r['update'] for r in dev] == [r['update'] for r in train] == reg['eval_updates']
        assert result['history'] == dev
        for split, history, nframes, nscenes in [('dev', dev, 60, 30), ('train_panel', train, 120, 120)]:
            for row in history:
                assert row['samples_seen'] == row['update']*c['batch']
                assert len(row['per_frame']) == nframes and len(row['scene_epe']) == nscenes
                ids = {(f['scene'], f['frame']) for f in row['per_frame']}
                assert len(ids) == nframes
                if split == 'dev':
                    if dev_ids is None: dev_ids = ids
                    assert ids == dev_ids
                else:
                    if train_ids is None: train_ids = ids
                    assert ids == train_ids
                assert math.isclose(statistics.mean(row['scene_epe'].values()), row['mean_scene_epe'], abs_tol=1e-10)
                assert all(math.isfinite(v) and v >= 0 for v in row['scene_epe'].values())
                identity = dict(case=c['id'], model=c['model'], batch=c['batch'], augmentation=c['augmentation'],
                                split=split, update=row['update'], presentations=row['samples_seen'])
                for f in row['per_frame']: frames.append(dict(**identity, **f))
                for scene, epe in row['scene_epe'].items(): scenes.append(dict(**identity, scene=scene, epe=epe))
                trajectories.append(dict(**identity, epe=row['mean_scene_epe'], pooled_epe=row['pooled_epe'],
                    bad1=row['bad1'], bad3=row['bad3'], bad5=row['bad5'],
                    scene_median=statistics.median(row['scene_epe'].values()),
                    scene_p95=quantile(list(row['scene_epe'].values()), .95),
                    frame_p95=quantile([f['epe'] for f in row['per_frame']], .95),
                    worst5=statistics.mean(sorted(row['scene_epe'].values(), reverse=True)[:5])))
        assert result['dev_evaluation_presentations'] == len(dev)*60
        assert result['train_panel_evaluation_presentations'] == len(train)*120
        assert result['warmup_dev_presentations'] == 2
        for start in range(0, 10000, 500):
            window = rows[start:start+500]
            norms = [r['gradient_norm'] for r in window]
            gradients.append(dict(case=c['id'], first_update=start+1, last_update=start+500,
                loss=statistics.mean(r['loss'] for r in window), gradient_median=statistics.median(norms),
                gradient_p95=quantile(norms, .95), gradient_max=max(norms),
                clip_fraction=statistics.mean(r['clipped'] for r in window)))
        data[c['model'], c['batch'], c['augmentation']] = {r['update']:r for r in dev}
        audits.append(dict(case=c['id'], status='LOG_AUDIT_PASS',
            result_sha256=hashlib.sha256((root/'result.json').read_bytes()).hexdigest()))
    assert not {x[0] for x in dev_ids} & {x[0] for x in train_ids}
    gaps, effects, exposure = [], [], []
    def e(model, batch, aug, t): return data[model, batch, aug][t]['mean_scene_epe']
    for t in reg['eval_updates']:
        for model in MODELS:
            values = {(b,g):e(model,b,g,t) for b,g in CELLS}
            effects.append(dict(model=model, update=t, estimand='model_epe', **contrasts(values)))
            if model == 'original': continue
            delta = {(b,g):values[b,g]-e('original',b,g,t) for b,g in CELLS}
            effects.append(dict(model=model, update=t, estimand='method_minus_original', **contrasts(delta)))
            for b,g in CELLS:
                gaps.append(dict(model=model, batch=b, augmentation=g, update=t, presentations=t*b,
                    original_epe=e('original',b,g,t), method_epe=values[b,g], gap=delta[b,g],
                    improvement_percent=-100*delta[b,g]/e('original',b,g,t)))
    for t2,t8 in EXPOSURE_PAIRS:
        for model in MODELS:
            for g in ('none','hv'):
                exposure.append(dict(model=model, augmentation=g, presentations=t2*2, b2_update=t2, b8_update=t8,
                    b2_epe=e(model,2,g,t2), b8_epe=e(model,8,g,t8),
                    model_epe_difference=e(model,8,g,t8)-e(model,2,g,t2),
                    gap_difference=(e(model,8,g,t8)-e('original',8,g,t8))-(e(model,2,g,t2)-e('original',2,g,t2)),
                    b2_last_used_lr=lr(t2-1), b8_last_used_lr=lr(t8-1)))
    for filename, rows in [('trajectories', trajectories), ('frames', frames), ('scenes', scenes),
                           ('gradient-windows', gradients), ('method-gaps', gaps), ('factorial-effects', effects),
                           ('equal-presentations', exposure)]: write_csv(a.output/(filename+'.csv'), rows)
    (a.output/'analysis-status.json').write_text(json.dumps(dict(status='COMPLETE_LOG_AUDIT', cases=audits,
        checkpoint_binary_audit='Separate inspection required before final causal report',
        inference='One paired seed; conditional controlled contrasts only; no Test or seed-population claim'), indent=2)+'\n')
    print('Twelve complete log sets audited; contrasts written. Binary checkpoint audit remains separate.')


if __name__ == '__main__': main()
