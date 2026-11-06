"""Parity-gated endpoint diagnostics: run each arm/seed in a separate GPU process."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import torch

HERE = Path(__file__).resolve().parent
SEEDS = {1234: 2017, 2345: 3128, 3456: 4239}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@torch.no_grad()
def weight_variation(module, cost, max_rows=128):
    """Fixed evenly spaced native cost rows; valid-offset position/cost variation.

    Position SD uses interior positions to avoid confusing the boundary mask with
    position dependence. Cost sensitivity compares actual versus zero costs on
    the same positions. Row SD holds position fixed across sampled native rows.
    """
    rows = cost.reshape(-1, cost.shape[-1])
    indices = torch.linspace(0, rows.shape[0] - 1, min(max_rows, rows.shape[0]), device=cost.device).long()
    sample = rows[indices]
    actual = module.routing_weights(sample)
    zero = module.routing_weights(torch.zeros_like(sample))
    width = sample.shape[-1]
    interior = actual[:, 2:-2] if width > 4 else actual[:, :0]
    return dict(sampled_rows=indices.cpu().tolist(), total_rows=rows.shape[0], right_width=width,
                cost_vs_zero_mean_abs=float((actual-zero).abs().mean()),
                fixed_position_row_sd_mean=float(actual.std(0, unbiased=False).mean()),
                interior_position_sd_mean=float(interior.mean(0).std(0, unbiased=False).mean()) if width > 4 else None,
                note='native padded cost rows, deterministic evenly spaced sample; five-offset probabilities')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source-root', 'data-root', 'manifest', 'checkpoint', 'evaluations', 'raw-root'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--reuse-receipt', type=Path, help='Explicit verified historical SC/Original admission receipt')
    p.add_argument('--arm', choices=('Original', 'SC', 'SP', 'CC', 'CP'), required=True)
    p.add_argument('--seed', type=int, choices=tuple(SEEDS), required=True)
    a = p.parse_args()
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert any(x in torch.cuda.get_device_name() for x in ('4090', '5090'))
    sys.path[:0] = [str(a.source_root), str(a.source_root/'core'), str(a.source_root/'experiments/spiking-transformer-stereo/scripts')]
    probes = load_module('phase2_placement_probes', a.source_root/'experiments/spiking-attention-placement-v1/probes.py')
    driver = load_module('phase2_replay_driver', a.source_root/'experiments/spiking-transformer-stereo/scripts/run_experiment.py')
    assert probes.digest(a.manifest) == probes.MANIFEST_SHA
    registry_path = a.source_root.parents[1]/'source-registry.json'
    registry = json.loads(registry_path.read_text())
    for rel, expected in registry['source_files'].items():
        assert probes.digest(a.source_root.parents[1]/rel) == expected, rel
    import cv2
    torch.set_num_threads(6)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    update = ck['updates']
    assert update in (7500, 30000)
    assert ck['samples_seen'] == update*8
    config = ck['config']
    assert config['seed'] == a.seed
    reuse = None
    if a.reuse_receipt:
        reuse = json.loads(a.reuse_receipt.read_text())
        assert a.arm in ('SC', 'Original') and reuse['status'] == 'VERIFIED_REUSE'
        assert (reuse['seed'], reuse['arm'], reuse['branch_seed']) == (a.seed, a.arm, SEEDS[a.seed])
        assert reuse['current_source_identity'] == registry['source_identity_sha256']
        assert reuse['stream_audit']['actual_indices_hv_lr_verified']
        assert reuse['common_initial_state_sha256'] == reuse['native_initial_state_sha256']
        assert config['arm'] == reuse['historical_arm'] == ('A' if a.arm == 'SC' else 'original')
        endpoints = [r for r in reuse['endpoints'] if r['update'] == update]
        assert len(endpoints) == 1
        admitted = endpoints[0]
        assert admitted['checkpoint_sha256'] == probes.digest(a.checkpoint)
        assert admitted['archived_source_identity'] == ck['source_identity'] == reuse['archived_source_identity']
    else:
        assert config['arm'] == a.arm
        if a.arm != 'Original':
            assert config['branch_seed'] == SEEDS[a.seed]
        assert ck['source_identity'] == registry['source_identity_sha256']
    saved = [json.loads(line) for line in a.evaluations.read_text().splitlines() if line.strip()]
    saved = [r for r in saved if r['update'] == update]
    assert len(saved) == 1 and saved[0]['samples_seen'] == update*8
    if reuse:
        assert saved[0] == admitted['evaluation'], 'official evaluation differs from reuse receipt'
    model = driver.build_model('original', a.seed)
    if a.arm != 'Original':
        from core.spiking_attention_phase2 import install
        install(model, a.arm, seed=SEEDS[a.seed])
    model.load_state_dict(ck['model'], strict=True)
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor.cpu(), ck['model'][name]), name
    model = model.cuda().eval()
    manifest = json.loads(a.manifest.read_text())
    assert len(manifest['dev']) == 60
    dev = driver.ManifestDataset(a.data_root, manifest['dev'])
    loader = torch.utils.data.DataLoader(dev, batch_size=2, num_workers=0, generator=torch.Generator().manual_seed(2225))
    frames, variation = [], []
    for left, right, target, valid, indices in loader:
        captured = []
        handle = None
        if a.arm != 'Original':
            def capture(module, args):
                if not captured:
                    captured.append(weight_variation(module, args[0].detach()))
            handle = model.placement.register_forward_pre_hook(capture)
        try:
            batch = probes.replay_batch(model, driver, *[v.cuda() for v in (left, right, target, valid)], 'Original' if a.arm == 'Original' else 'A')
        finally:
            if handle is not None:
                handle.remove()
        if captured:
            variation.append(dict(frame_indices=indices.tolist(), **captured[0]))
        for index, row in zip(indices.tolist(), batch):
            item = dev.rows[index]
            row.update(scene=item['scene'], frame=item['frame'])
            frames.append(row)
    parity = probes.assert_parity(frames, saved[0]['per_frame'])
    official = {(r['scene'], r['frame']): r for r in saved[0]['per_frame']}
    bad_differences = []
    for row in frames:
        ref = official[(row['scene'], row['frame'])]
        full = row['iterations']['32']['full']
        for key in ('bad1', 'bad3', 'bad5'):
            delta = abs(full[key]-ref[key])
            assert delta <= 1e-5, (row['scene'], row['frame'], key, delta)
            bad_differences.append(delta)
    parity['max_frame_bad_percentage_point_difference'] = max(bad_differences)
    parity['all_official_frame_metrics_exact'] = parity['max_frame_epe_difference'] == 0 and max(bad_differences) == 0
    result = dict(arm=a.arm, seed=a.seed, branch_seed=SEEDS[a.seed], update=update, samples_seen=update*8,
                  checkpoint=str(a.checkpoint), checkpoint_sha256=probes.digest(a.checkpoint),
                  source_identity=registry['source_identity_sha256'], checkpoint_source_identity=ck['source_identity'],
                  reuse_receipt_sha256=probes.digest(a.reuse_receipt) if reuse else None, evaluator_sha256=probes.digest(__file__),
                  probes_sha256=probes.digest(probes.__file__), manifest_sha256=probes.digest(a.manifest),
                  parity=parity, per_frame=frames, summary=probes.summarize(frames), weight_variation=variation,
                  slurm_job_id=os.environ['SLURM_JOB_ID'], gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                  units=dict(epe='native full-resolution pixels', bad='percentage points', rank_error='feature cells'))
    output = a.raw_root/f'seed-{a.seed}'/a.arm/f'replay-{update}.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix('.tmp')
    temp.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    temp.replace(output)
    print(output)


if __name__ == '__main__':
    main()
