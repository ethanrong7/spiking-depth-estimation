"""Read historical records and source archives only; no model or dataset loading."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import tarfile


def read(path):
    return json.loads(path.read_text())


def lr(index, total=10100):
    peak = .01 * total - 1
    if index <= peak:
        return 8e-6 + (2e-4 - 8e-6) * index / peak
    return 2e-4 + (8e-10 - 2e-4) * (index - peak) / (total - 1 - peak)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    sources = []
    for name in ('source-alif-mechanism-screen-v1', 'source-recipe-aligned-three-way-v1',
                 'source-recipe-gap-diagnostic-v1', 'source-paired-loss-original-v1',
                 'source-paired-loss-normalized-v1', 'source-alif-mechanism-repeat-v1'):
        root = a.raw_root / name
        registry = read(root / 'source-registry.json')
        with tarfile.open(root / 'source.tar.gz') as archive:
            members = {m.name.removeprefix('./'): m for m in archive.getmembers() if m.isfile()}
            for rel, expected in registry['source_files'].items():
                assert hashlib.sha256(archive.extractfile(members[rel]).read()).hexdigest() == expected, (name, rel)
        sources.append(dict(name=name, identity=registry['source_identity_sha256'],
                            git_commit=registry['git_commit'], verified_files=len(registry['source_files'])))
    runs = []
    histories = {}
    definitions = [
        ('paired-loss-v1', 'original-seed1234-10k-v1', 10000, 2),
        ('paired-loss-v1', 'spiking-normalized-seed1234-10k-v1', 10000, 2),
        ('alif-mechanism-screen-v1', 'o-seed1234-5k-v1', 5000, 2),
        ('alif-mechanism-screen-v1', 'i-seed1234-5k-v1', 5000, 2),
        ('alif-mechanism-repeat-v1', 'o-seed2345-5k-v1', 5000, 2),
        ('alif-mechanism-repeat-v1', 'i-seed2345-5k-v1', 5000, 2),
    ] + [('recipe-aligned-three-way-v1', m + '-seed1234-10k-v1', 10000, 8)
         for m in ('original', 'original_i', 'spiking_normalized')]
    manifest = 'bad4369b6fd8b0cdf871a20b1727c2d9b59733a16bd33f75904227bdfef4510a'
    for campaign, name, count, batch in definitions:
        root = a.raw_root / campaign / name
        metadata = read(root / 'metadata.json')
        assert metadata['manifest_sha256'] == manifest
        assert metadata['config']['physical_batch'] == min(batch, 4)
        assert metadata['config']['effective_batch'] == batch
        history = read(root / 'history.json')
        histories[name] = {row['update']: row for row in history}
        assert history[-1]['update'] == count
        for row in history:
            assert len(row['per_frame']) == 60 and len(row['scene_epe']) == 30
            assert math.isclose(sum(row['scene_epe'].values()) / 30, row['mean_scene_epe'], abs_tol=1e-12)
        rows = [json.loads(line) for line in (root / 'train.jsonl').read_text().splitlines()]
        assert [row['update'] for row in rows] == list(range(1, count + 1))
        for row in rows:
            assert row['samples_seen'] == row['update'] * batch
            assert all(math.isfinite(row[k]) for k in ('loss', 'gradient_norm', 'train_epe'))
            assert row['gradient_norm'] > 0 and row.get('neuron_loss', 0) == 0
            assert math.isclose(row['lr'], lr(row['update']), rel_tol=1e-12, abs_tol=1e-16)
        runs.append(dict(campaign=campaign, run=name, updates=count, presentations=count*batch,
                         config=metadata['config'], final_epe=history[-1]['mean_scene_epe'],
                         matched_unaugmented_train_history=False,
                         train_log_sha256=hashlib.sha256((root/'train.jsonl').read_bytes()).hexdigest()))
    # Names collide across historical and revised Original runs; use explicit paths here.
    def e(campaign, name, at):
        return next(x['mean_scene_epe'] for x in read(a.raw_root/campaign/name/'history.json') if x['update'] == at)
    comparisons = []
    for method, at, old_campaign, old_o, old_m, new_m in [
        ('Transformer', 10000, 'paired-loss-v1', 'original-seed1234-10k-v1',
         'spiking-normalized-seed1234-10k-v1', 'spiking_normalized-seed1234-10k-v1'),
        ('I', 5000, 'alif-mechanism-screen-v1', 'o-seed1234-5k-v1',
         'i-seed1234-5k-v1', 'original_i-seed1234-10k-v1')]:
        o0, m0 = e(old_campaign, old_o, at), e(old_campaign, old_m, at)
        o1 = e('recipe-aligned-three-way-v1', 'original-seed1234-10k-v1', at)
        m1 = e('recipe-aligned-three-way-v1', new_m, at)
        reduction = (o0-m0)-(o1-m1)
        comparisons.append(dict(method=method, update=at, historical_original=o0, historical_method=m0,
                                revised_original=o1, revised_method=m1,
                                advantage_before_percent=100*(o0-m0)/o0,
                                advantage_after_percent=100*(o1-m1)/o1,
                                gap_reduction_px=reduction, baseline_arithmetic_share=(o0-o1)/reduction))
    result = dict(status='PASS', audit_scope='Local saved logs and archived source bytes; no inference or fresh binary checkpoint inspection',
                  sources=sources, runs=runs, comparisons=comparisons,
                  historical_reuse='Context only; 12 fresh common-runner cells registered',
                  checkpoint_evidence='Prior checkpoint audits retained; resume equivalence not established')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(status='PASS', sources=len(sources), runs=len(runs), comparisons=comparisons), indent=2))


if __name__ == '__main__':
    main()
