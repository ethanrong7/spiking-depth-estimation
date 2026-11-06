"""Frozen A/B checkpoint interventions: one arm per process, no training.

Baseline executes the archived model unchanged. Interventions temporarily change
only placement computation, restore methods afterwards, and never edit tensors.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import MethodType
import torch
from torch.nn import functional as F

CONDITIONS = ('baseline', 'branch_off', 'uniform_weights', 'code_tanh')
VERIFIED_PROBE_SHAS = {'da0fa6ab64de0649af1e7afc9c586e79fc0ccc83740824431713e1a8eb766098', '45876d358b3420e328ad4f5937764b2fa637109438e27a7a67d12d46b7c4e979'}
MANIFEST_SHA = 'bad4369b6fd8b0cdf871a20b1727c2d9b59733a16bd33f75904227bdfef4510a'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def state_digest(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def continuous_codes(module, x):
    z = module.norm(x)
    return tuple(torch.tanh(layer(z)) for layer in (module.q, module.k, module.v))


def uniform_cost(module, cost):
    shape = cost.shape
    width = shape[-1]
    x = cost.reshape(-1, width, 1)
    position = torch.linspace(-1, 1, width, device=x.device).view(1, width, 1).expand(x.shape[0], -1, -1)
    q, k, v = module.codes(module.input(torch.cat([x, position], -1)))
    k = F.pad(k.transpose(1, 2), (2, 2)).unfold(2, 5, 1).permute(0, 2, 3, 1)
    v = F.pad(v.transpose(1, 2), (2, 2)).unfold(2, 5, 1).permute(0, 2, 3, 1)
    index = torch.arange(width, device=x.device)[:, None] + torch.arange(-2, 3, device=x.device)[None, :]
    valid = (index >= 0) & (index < width)
    logits = (q[:, :, None, :] * k).sum(-1)/4
    weights = torch.zeros_like(logits).masked_fill(~valid[None], -torch.inf).softmax(-1)
    return (x + module.output((weights[..., None]*v).sum(-2))).reshape(shape)


def uniform_descriptors(module, left, right):
    l, r = left.permute(0, 2, 3, 1), right.permute(0, 2, 3, 1)
    ql, kl, vl = module.codes(module.input(l))
    qr, kr, vr = module.codes(module.input(r))
    weights_l = torch.zeros_like(ql @ kr.transpose(-1, -2)).softmax(-1)
    weights_r = torch.zeros_like(qr @ kl.transpose(-1, -2)).softmax(-1)
    a = l + module.output(weights_l @ vr)
    b = r + module.output(weights_r @ vl)
    return a.permute(0, 3, 1, 2).contiguous(), b.permute(0, 3, 1, 2).contiguous()


@contextmanager
def intervention(model, arm, condition):
    if arm not in ('A', 'B') or condition not in CONDITIONS:
        raise ValueError((arm, condition))
    module = model.placement
    if condition == 'baseline':
        yield
        return
    target, name = (module.codes, 'forward') if condition == 'code_tanh' else (module, 'compute')
    # Restore the actual instance dictionary, not a new bound-method shadow.
    existed = name in target.__dict__
    old = target.__dict__.get(name)
    if condition == 'code_tanh':
        method = continuous_codes
    elif condition == 'branch_off':
        method = (lambda self, cost: cost) if arm == 'A' else (lambda self, left, right: (left, right))
    else:
        method = uniform_cost if arm == 'A' else uniform_descriptors
    setattr(target, name, MethodType(method, target))
    try:
        yield
    finally:
        if existed:
            setattr(target, name, old)
        else:
            delattr(target, name)


def summarize(frames):
    scenes = sorted({r['scene'] for r in frames})
    assert len(frames) == 60 and len(scenes) == 30
    result = dict(per_scene={})
    for scene in scenes:
        rows = [r for r in frames if r['scene'] == scene]
        assert len(rows) == 2
        result['per_scene'][scene] = {key: sum(r[key] for r in rows)/2 for key in ('epe', 'bad1', 'bad3', 'bad5')}
    result['mean_scene'] = {key: sum(v[key] for v in result['per_scene'].values())/len(scenes) for key in ('epe', 'bad1', 'bad3', 'bad5')}
    count = sum(r['valid_pixels'] for r in frames)
    result['valid_pixels'] = count
    result['pooled'] = {key: sum(r[key]*r['valid_pixels'] for r in frames)/count for key in ('epe', 'bad1', 'bad3', 'bad5')}
    return result


def pair(candidate, baseline):
    base = {(r['scene'], r['frame']): r for r in baseline}
    actual = {(r['scene'], r['frame']): r for r in candidate}
    assert len(base) == len(actual) == len(candidate) == len(baseline) == 60
    assert set(actual) == set(base)
    deltas = []
    for key, row in actual.items():
        reference = base[key]
        assert row['valid_pixels'] == reference['valid_pixels']
        assert row['target_support_sha256'] == reference['target_support_sha256']
        deltas.append(dict(scene=key[0], frame=key[1], valid_pixels=row['valid_pixels'],
                           **{metric: row[metric]-reference[metric] for metric in ('epe', 'bad1', 'bad3', 'bad5')}))
    result = summarize(deltas)
    epe = [r['epe'] for r in result['per_scene'].values()]
    result.update(scene_wins=sum(v < 0 for v in epe), scene_ties=sum(v == 0 for v in epe), scene_losses=sum(v > 0 for v in epe),
                  interpretation='condition minus frozen baseline; positive EPE means baseline better; bad deltas are percentage points')
    return result


class ParityError(AssertionError):
    def __init__(self, report):
        self.report = report
        super().__init__(f"official parity failed: {report['max_frame_differences']}")


def official_parity(frames, reference, tolerance=1e-5):
    if reference is None:
        return dict(status='BASELINE_TIED_TO_CHECKPOINT', official_same_update_record_present=False,
                    explanation='No same-update official evaluation record; this baseline is paired only within this immutable checkpoint.')
    saved = {(r['scene'], r['frame']): r for r in reference['per_frame']}
    actual = {(r['scene'], r['frame']): r for r in frames}
    assert len(saved) == len(actual) == len(frames) == 60 and set(saved) == set(actual)
    maxima = {metric: 0. for metric in ('epe', 'bad1', 'bad3', 'bad5')}
    drift = []
    counts_exact = True
    for key, row in actual.items():
        differences = {metric: row[metric]-saved[key][metric] for metric in maxima}
        count_delta = row['valid_pixels']-saved[key]['valid_pixels']
        counts_exact = counts_exact and count_delta == 0
        drift.append(dict(scene=key[0], frame=key[1], valid_pixel_difference=count_delta, **differences))
        for metric in maxima:
            maxima[metric] = max(maxima[metric], abs(differences[metric]))
    passed = counts_exact and all(v <= tolerance for v in maxima.values())
    report = dict(status='OFFICIAL_PARITY_PASSED' if passed else 'FAILED', official_same_update_record_present=True,
                  valid_counts_exact=counts_exact, absolute_tolerance=tolerance, max_frame_differences=maxima,
                  frame_differences=drift, all_frame_metrics_exact=counts_exact and all(v == 0 for v in maxima.values()))
    if not passed:
        raise ParityError(report)
    return report


def backend_flags():
    return dict(autocast=torch.is_autocast_enabled(), matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_tf32=torch.backends.cudnn.allow_tf32, cudnn_benchmark=torch.backends.cudnn.benchmark,
                cudnn_deterministic=torch.backends.cudnn.deterministic)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


@torch.no_grad()
def evaluate(model, driver, loader, probes, arm, first_forward):
    """Use the already parity-verified placement replay execution path exactly.

    Its prefix forwards and retained placement hooks can affect CUDA workspace
    selection. Only the final official 32-refinement prediction is scored here.
    """
    frames = []
    model.eval()
    driver.reset_model(model)
    original_forward = driver.forward
    def observed_forward(*args, **kwargs):
        if not first_forward:
            first_forward.update(backend_flags())
            assert first_forward == dict(autocast=False, matmul_tf32=False, cudnn_tf32=True,
                cudnn_benchmark=False, cudnn_deterministic=True), first_forward
        return original_forward(*args, **kwargs)
    driver.forward = observed_forward
    try:
        for left, right, target, valid, indices in loader:
            left, right, target, valid = [x.cuda() for x in (left, right, target, valid)]
            replay = probes.replay_batch(model, driver, left, right, target, valid, arm)
            for index, row in zip(indices.tolist(), replay):
                source = loader.dataset.rows[index]
                frames.append(dict(scene=source['scene'], frame=source['frame'],
                    target_support_sha256=row['target_support_sha256'], **row['iterations']['32']['full']))
            del left, right, target, valid, replay
    finally:
        driver.forward = original_forward
        driver.reset_model(model)
    assert len(frames) == 60 and len({(r['scene'], r['frame']) for r in frames}) == 60
    return frames


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source-root', 'checkpoint', 'evaluations', 'data-root', 'manifest', 'output-root'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--probes', type=Path, required=True, help='Pinned external parity-verified placement probes.py')
    p.add_argument('--arm', choices=('A', 'B'), required=True)
    p.add_argument('--seed', type=int, default=1234)
    a = p.parse_args()
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert any(name in torch.cuda.get_device_name() for name in ('4090', '5090'))
    assert digest(a.manifest) == MANIFEST_SHA
    sys.path[:0] = [str(a.source_root), str(a.source_root/'core'), str(a.source_root/'experiments/spiking-transformer-stereo/scripts')]
    registry_path = a.source_root.parents[1]/'source-registry.json'
    registry = json.loads(registry_path.read_text())
    for rel, expected in registry['source_files'].items():
        assert digest(a.source_root.parents[1]/rel) == expected, rel
    driver = load_module('frozen_ablation_driver', a.source_root/'experiments/spiking-transformer-stereo/scripts/run_experiment.py')
    probes_path = a.probes
    assert digest(probes_path) in VERIFIED_PROBE_SHAS
    probes = load_module('frozen_ablation_probes', probes_path)
    from core.spiking_attention_placement import install
    import cv2
    torch.set_num_threads(6)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    checkpoint_hash = digest(a.checkpoint)
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    assert ck['source_identity'] == registry['source_identity_sha256']
    assert ck['config']['arm'] == a.arm and ck['config']['seed'] == a.seed
    update = ck['updates']
    assert ck['samples_seen'] == update*8
    records = [json.loads(line) for line in a.evaluations.read_text().splitlines() if line.strip()]
    reference = [r for r in records if r['update'] == update]
    assert len(reference) <= 1
    reference = reference[0] if reference else None
    if reference:
        assert reference['samples_seen'] == update*8
    model = install(driver.build_model('original', a.seed), a.arm)
    model.load_state_dict(ck['model'], strict=True)
    for name, value in model.state_dict().items():
        assert torch.equal(value.cpu(), ck['model'][name]), name
    receipt = dict(arm=a.arm, seed=a.seed, updates=update, samples_seen=ck['samples_seen'],
        checkpoint=str(a.checkpoint), checkpoint_sha256=checkpoint_hash, checkpoint_config=ck['config'],
        source_identity=registry['source_identity_sha256'], source_registry_sha256=digest(registry_path),
        source_git_commit=registry['git_commit'], source_dirty_status=registry['dirty_status'],
        evaluator_sha256=digest(__file__), probes_sha256=digest(probes_path),
        replay_route='archived probes.replay_batch with native placement hooks and prefixes1,2,4,8,16,32; score32only',
        manifest_sha256=digest(a.manifest), evaluations_sha256=digest(a.evaluations),
        slurm_job_id=os.environ['SLURM_JOB_ID'], gpu=torch.cuda.get_device_name(),
        gpu_uuid=str(getattr(torch.cuda.get_device_properties(0), 'uuid', 'unavailable')),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), slurm_job_gpus=os.environ.get('SLURM_JOB_GPUS'),
        torch=torch.__version__, cuda=torch.version.cuda, iterations=32, dev_frames=60, batch_size=2,
        precision=dict(model_dtype=str(next(model.parameters()).dtype), autocast=False,
                       matmul_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32,
                       cudnn_benchmark=torch.backends.cudnn.benchmark, cudnn_deterministic=torch.backends.cudnn.deterministic),
        training_steps=0, interpretation='fixed-checkpoint reliance diagnostic; no retraining or causal necessity claim')
    del ck
    model = model.cuda().eval()
    initial_hash = state_digest(model)
    manifest = json.loads(a.manifest.read_text())
    assert len(manifest['dev']) == 60
    dev = driver.ManifestDataset(a.data_root, manifest['dev'])
    loader = torch.utils.data.DataLoader(dev, batch_size=2, num_workers=0, generator=torch.Generator().manual_seed(2225))
    output_dir = a.output_root/a.arm/f'update-{update}'
    if (output_dir/'baseline.json').exists():
        raise FileExistsError(f'Use a fresh output root; existing baseline at {output_dir}')
    results = {}
    baseline_frames = None
    for condition in CONDITIONS:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        resident_before = dict(allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved())
        started = time.perf_counter()
        first_forward = {}
        with intervention(model, a.arm, condition):
            frames = evaluate(model, driver, loader, probes, a.arm, first_forward)
        torch.cuda.synchronize()
        measurements = dict(first_forward_backend_flags=first_forward, elapsed_seconds=time.perf_counter()-started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            resident_before=resident_before,
            resident_after=dict(allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved()),
            scope='condition forward plus data decoding/metrics; excludes model load and parameter hash checks')
        assert state_digest(model) == initial_hash, f'{condition}: model tensors changed'
        result = dict(condition=condition, receipt=receipt, state_sha256=initial_hash, parameters_unchanged=True,
                      per_frame=frames, summary=summarize(frames), measurements=measurements)
        if condition == 'baseline':
            baseline_frames = frames
            result['official_parity'] = dict(status='PENDING')
            write_json(output_dir/f'{condition}.json', result)
            try:
                result['official_parity'] = official_parity(frames, reference)
            except AssertionError as error:
                result['official_parity'] = getattr(error, 'report', dict(status='FAILED', error=str(error), absolute_tolerance=1e-5))
                write_json(output_dir/f'{condition}.json', result)
                write_json(output_dir/'failure.json', dict(condition=condition, receipt=receipt,
                    error=str(error), interventions_run=False, baseline_frames_saved=True))
                raise
        else:
            result['paired_vs_baseline'] = pair(frames, baseline_frames)
        write_json(output_dir/f'{condition}.json', result)
        results[condition] = result
        print(json.dumps(dict(arm=a.arm, update=update, condition=condition, metrics=result['summary']['mean_scene']), allow_nan=False), flush=True)
    # Checkpoint is read-only and all temporary method overrides have ended.
    assert digest(a.checkpoint) == checkpoint_hash
    write_json(output_dir/'ablation.json', dict(receipt=receipt, conditions=results, checkpoint_unchanged=True))


if __name__ == '__main__':
    main()
