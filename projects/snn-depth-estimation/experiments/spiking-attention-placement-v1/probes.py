"""Frozen endpoint replay, separate process per checkpoint; never trains or submits.

--source-root selects the exact frozen project tree, rather than the evaluator's
tree. Run Original separately from A: A installs a process-local native matcher.
Official iteration-32 per-frame parity is mandatory before publishing diagnostics.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import torch

ITERATIONS = (1, 2, 4, 8, 16, 32)
MANIFEST_SHA = 'bad4369b6fd8b0cdf871a20b1727c2d9b59733a16bd33f75904227bdfef4510a'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def fine_geometry(target, valid, padding, grid_shape, stride=4):
    """Native grid origin is padded (0,0); retain only real image grid sites.

    Sample signed GT at those exact integer full-image coordinates. Candidate
    right sites are real, unpadded grid centers, including neither replicated
    padding nor interpolated/rounded labels. OOV uses the actual image boundary.
    """
    b, _, height, width = target.shape
    gh, gw = grid_shape
    px, _, py, _ = padding
    ys = torch.arange(gh, device=target.device) * stride - py
    xs = torch.arange(gw, device=target.device) * stride - px
    real = (ys[:, None] >= 0) & (ys[:, None] < height) & (xs[None] >= 0) & (xs[None] < width)
    gt = target[:, 0][:, ys.clamp(0, height-1)[:, None], xs.clamp(0, width-1)[None, :]]
    support = valid[:, ys.clamp(0, height-1)[:, None], xs.clamp(0, width-1)[None, :]].bool() & real
    right_coordinate = xs[None, None, :] + gt
    inside = (right_coordinate >= 0) & (right_coordinate <= width-1)
    candidates = (xs >= 0) & (xs <= width-1)
    return dict(support=support, inview=support & inside,
                right_cells=(right_coordinate + px) / stride,
                candidate_mask=candidates)


def rank_scores(scores, geometry):
    """Competition rank; all score ties at the maximum count as top-1.

    Coordinate ties when choosing nearest GT grid point and score argmax are
    resolved toward the smaller right x. Report both forms of ambiguity.
    """
    if scores.ndim == 5:
        scores = scores.squeeze(-2)
    mask = geometry['inview']
    candidate = geometry['candidate_mask']
    assert candidate.any() and torch.isfinite(scores).all()
    x = torch.arange(scores.shape[-1], device=scores.device)
    distance = (x - geometry['right_cells'][..., None]).abs().masked_fill(~candidate, torch.inf)
    nearest = distance.argmin(-1, keepdim=True)
    truth_score = scores.gather(-1, nearest)
    eligible = candidate.expand_as(scores)
    rank = 1 + ((scores > truth_score) & eligible).sum(-1)
    truth_ties = ((scores == truth_score) & eligible).sum(-1)
    best = scores.masked_fill(~candidate, -torch.inf)
    selected = best.argmax(-1)
    top_ties = (best == best.max(-1, keepdim=True).values).sum(-1)
    coordinate_ties = (distance == distance.min(-1, keepdim=True).values).sum(-1)
    error = (selected - geometry['right_cells']).abs()
    count = int(mask.sum())
    return dict(pixel_count=count, valid_fine_count=int(geometry['support'].sum()),
                excluded_oov_count=int((geometry['support'] & ~mask).sum()),
                candidate_count=int(candidate.sum()), rank_sum=int(rank[mask].sum()),
                top1_count=int(((rank == 1) & mask).sum()),
                argmax_error_cells_sum=float(error[mask].double().sum()),
                truth_score_tied_count=int(((truth_ties > 1) & mask).sum()),
                max_score_tied_count=int(((top_ties > 1) & mask).sum()),
                nearest_coordinate_tied_count=int(((coordinate_ties > 1) & mask).sum()),
                mean_rank=float(rank[mask].double().mean()) if count else None,
                top1_fraction=float((rank[mask] == 1).double().mean()) if count else None,
                mean_argmax_error_cells=float(error[mask].double().mean()) if count else None)


def frame_metrics(prediction, target, valid):
    assert torch.isfinite(prediction).all()
    error = (prediction[:, 0] - target[:, 0]).abs()
    x = torch.arange(target.shape[-1], device=target.device)[None, None, :]
    inside = (x + target[:, 0] >= 0) & (x + target[:, 0] <= target.shape[-1]-1)
    frames = []
    for i in range(target.shape[0]):
        strata = {}
        for name, mask in [('full', valid[i]), ('inview', valid[i] & inside[i]), ('oov', valid[i] & ~inside[i])]:
            values = error[i][mask]
            count = values.numel()
            strata[name] = dict(valid_pixels=count, error_sum=float(values.double().sum()),
                                epe=float(values.mean()) if count else None,
                                **{f'bad{k}': float((values > k).float().mean()*100) if count else None for k in (1, 3, 5)})
        frames.append(strata)
    return frames


def assert_parity(frames, saved, tolerance=1e-5):
    keyed = {(r['scene'], r['frame']): r for r in saved}
    assert len(keyed) == len(saved) == len(frames), 'cohort size/duplicate mismatch'
    errors = []
    seen = set()
    for row in frames:
        key = (row['scene'], row['frame'])
        assert key not in seen and key in keyed, 'cohort identity mismatch'
        seen.add(key)
        reference = keyed[key]
        actual = row['iterations']['32']['full']
        assert actual['valid_pixels'] == reference['valid_pixels'], (key, 'valid-count mismatch')
        difference = abs(actual['epe'] - reference['epe'])
        assert difference <= tolerance, (key, 'endpoint replay mismatch', difference)
        errors.append(difference)
    return dict(passed=True, absolute_epe_tolerance=tolerance, max_frame_epe_difference=max(errors))


def summarize(frames):
    """Match official scene aggregation: mean of frame means, then scenes.

    Undefined empty stratum frames/scenes are excluded with counts reported.
    All full-mask frames must be defined; fine ranks are pixel-pooled.
    """
    result = {}
    for iteration in ITERATIONS:
        result[str(iteration)] = {}
        for stratum in ('full', 'inview', 'oov'):
            rows = [(r['scene'], r['iterations'][str(iteration)][stratum]) for r in frames]
            scenes = sorted({s for s, _ in rows})
            per_scene = {}
            for scene in scenes:
                defined = [v['epe'] for s, v in rows if s == scene and v['valid_pixels']]
                if defined:
                    per_scene[scene] = sum(defined) / len(defined)
            count = sum(v['valid_pixels'] for _, v in rows)
            result[str(iteration)][stratum] = dict(scene_epe=per_scene,
                mean_scene_epe=sum(per_scene.values())/len(per_scene) if per_scene else None,
                valid_pixels=count, defined_frames=sum(bool(v['valid_pixels']) for _, v in rows),
                defined_scenes=len(per_scene), pooled_epe=sum(v['error_sum'] for _, v in rows)/count if count else None)
    ranks = {}
    for name in sorted({k for r in frames for k in r.get('rank', {})}):
        values = [r['rank'][name] for r in frames]
        fields = ('pixel_count', 'valid_fine_count', 'excluded_oov_count', 'rank_sum', 'top1_count',
                  'argmax_error_cells_sum', 'truth_score_tied_count', 'max_score_tied_count', 'nearest_coordinate_tied_count')
        totals = {key: sum(v[key] for v in values) for key in fields}
        count = totals['pixel_count']
        totals.update(mean_rank=totals['rank_sum']/count if count else None,
                      top1_fraction=totals['top1_count']/count if count else None,
                      mean_argmax_error_cells=totals['argmax_error_cells_sum']/count if count else None)
        ranks[name] = totals
    return dict(refinement=result, rank=ranks)


def compare_replays(candidate, original):
    """Pair only matched exposures, cohorts, target support and grid geometry."""
    assert candidate['arm'] in ('A', 'B', 'C') and original['arm'] == 'Original'
    assert candidate['update'] == original['update']
    assert candidate['manifest_sha256'] == original['manifest_sha256'] == MANIFEST_SHA
    assert candidate['parity']['passed'] and original['parity']['passed']
    control = {(r['scene'], r['frame']): r for r in original['per_frame']}
    rows = candidate['per_frame']
    assert len(control) == len(original['per_frame']) == len(rows) == 60
    assert len({(r['scene'], r['frame']) for r in rows}) == 60
    for row in rows:
        baseline = control[(row['scene'], row['frame'])]
        assert row['target_support_sha256'] == baseline['target_support_sha256']
        if candidate['arm'] in ('A', 'B'):
            assert row['fine_cohort_sha256'] == baseline['fine_cohort_sha256']
    curves = {}
    for iteration in ITERATIONS:
        curves[str(iteration)] = {}
        for stratum in ('full', 'inview', 'oov'):
            values = []
            for row in rows:
                baseline = control[(row['scene'], row['frame'])]
                actual = row['iterations'][str(iteration)][stratum]
                reference = baseline['iterations'][str(iteration)][stratum]
                assert actual['valid_pixels'] == reference['valid_pixels']
                if actual['valid_pixels']:
                    values.append((row['scene'], actual['epe'] - reference['epe']))
            scene_delta = {s: sum(d for scene, d in values if scene == s)/sum(scene == s for scene, _ in values)
                           for s in sorted({s for s, _ in values})}
            curves[str(iteration)][stratum] = dict(scene_epe_delta=scene_delta,
                mean_scene_epe_delta=sum(scene_delta.values())/len(scene_delta) if scene_delta else None,
                scene_wins=sum(v < 0 for v in scene_delta.values()), defined_frames=len(values), defined_scenes=len(scene_delta))
    return dict(arm=candidate['arm'], update=candidate['update'], parity_passed=True,
                refinement_vs_original=curves, candidate_summary=candidate['summary'], original_summary=original['summary'])


@torch.no_grad()
def replay_batch(model, driver, left, right, target, valid, arm):
    from core.corr import CorrBlock1D
    from core.utils.utils import InputPadder
    padder = InputPadder(left.shape, divis_by=32)
    captured = {}
    handles = []
    def features(module, args, output):
        captured['native'] = CorrBlock1D.corr(output[0].float(), output[1].float())
    def pre_filter(module, args):
        captured['pre'] = args[0].detach()
    def post_filter(module, args, output):
        captured['post'] = output.detach()
    def pre_exchange(module, args):
        captured['pre'] = CorrBlock1D.corr(args[0].float(), args[1].float())
    def post_exchange(module, args, output):
        captured['post'] = CorrBlock1D.corr(output[0].float(), output[1].float())
    if arm == 'Original':
        handles.append(model.fnet.register_forward_hook(features))
    elif arm in ('A', 'B'):
        handles.append(model.placement.register_forward_pre_hook(pre_filter if arm == 'A' else pre_exchange))
        handles.append(model.placement.register_forward_hook(post_filter if arm == 'A' else post_exchange))
    driver.reset_model(model)
    try:
        # Use precisely the official test-mode forward at every prefix rather
        # than assuming the all-intermediate-output path replays equivalently.
        metrics = {str(i): frame_metrics(driver.forward(model, left, right, i), target, valid) for i in ITERATIONS}
        result = [dict(iterations={str(i): metrics[str(i)][b] for i in ITERATIONS}) for b in range(left.shape[0])]
        for b, row in enumerate(result):
            h = hashlib.sha256()
            h.update(target[b].cpu().contiguous().numpy().tobytes())
            h.update(valid[b].cpu().contiguous().numpy().tobytes())
            row['target_support_sha256'] = h.hexdigest()
        if captured:
            shape = next(iter(captured.values())).shape[1:3]
            geometry = fine_geometry(target, valid, padder._pad, shape)
            for b, row in enumerate(result):
                g = {k: (v[b:b+1] if v.ndim == 3 else v) for k, v in geometry.items()}
                row['rank'] = {name: rank_scores(scores[b:b+1], g) for name, scores in captured.items()}
                h = hashlib.sha256()
                for key in ('support', 'inview', 'right_cells', 'candidate_mask'):
                    h.update(g[key].cpu().contiguous().numpy().tobytes())
                row['fine_cohort_sha256'] = h.hexdigest()
        return result
    finally:
        for handle in handles:
            handle.remove()
        driver.reset_model(model)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'compare':
        p = argparse.ArgumentParser(description='Pair saved parity-verified diagnostic results (CPU JSON only).')
        for name in ('candidate', 'original', 'output'):
            p.add_argument('--'+name, type=Path, required=True)
        a = p.parse_args(sys.argv[2:])
        result = compare_replays(json.loads(a.candidate.read_text()), json.loads(a.original.read_text()))
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        return
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source-root', 'data-root', 'manifest', 'checkpoint', 'evaluations', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--arm', choices=('Original', 'A', 'B', 'C'), required=True)
    a = p.parse_args()
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert torch.cuda.get_device_name().find('4090') >= 0 or torch.cuda.get_device_name().find('5090') >= 0
    assert digest(a.manifest) == MANIFEST_SHA
    sys.path[:0] = [str(a.source_root), str(a.source_root/'core'), str(a.source_root/'experiments/spiking-transformer-stereo/scripts')]
    spec = importlib.util.spec_from_file_location('placement_replay_driver', a.source_root/'experiments/spiking-transformer-stereo/scripts/run_experiment.py')
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
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
    saved = [json.loads(line) for line in a.evaluations.read_text().splitlines() if line.strip()]
    saved = [row for row in saved if row['update'] == update]
    assert len(saved) == 1
    model = driver.build_model('original', 1234)
    if a.arm != 'Original':
        assert ck['config']['arm'] == a.arm
        from core.spiking_attention_placement import install
        model = install(model, a.arm)
    model.load_state_dict(ck['model'], strict=True)
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor.cpu(), ck['model'][name]), name
    registry_path = a.source_root.parents[1]/'source-registry.json'
    registry = json.loads(registry_path.read_text())
    for rel, expected in registry['source_files'].items():
        assert digest(a.source_root.parents[1]/rel) == expected, rel
    if a.arm != 'Original':
        assert ck['source_identity'] == registry['source_identity_sha256']
    model = model.cuda().eval()
    manifest = json.loads(a.manifest.read_text())
    assert len(manifest['dev']) == 60
    dev = driver.ManifestDataset(a.data_root, manifest['dev'])
    loader = torch.utils.data.DataLoader(dev, batch_size=2, num_workers=0, generator=torch.Generator().manual_seed(2225))
    frames = []
    for left, right, target, valid, indices in loader:
        batch = replay_batch(model, driver, *[v.cuda() for v in (left, right, target, valid)], a.arm)
        for index, row in zip(indices.tolist(), batch):
            item = dev.rows[index]
            row.update(scene=item['scene'], frame=item['frame'])
            frames.append(row)
    parity = assert_parity(frames, saved[0]['per_frame'])
    result = dict(arm=a.arm, update=update, checkpoint=str(a.checkpoint), checkpoint_sha256=digest(a.checkpoint),
                  evaluator_sha256=digest(__file__), source_registry=registry, manifest_sha256=digest(a.manifest),
                  parity=parity, per_frame=frames, summary=summarize(frames),
                  slurm_job_id=os.environ['SLURM_JOB_ID'], gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                  support='official all valid targets; rank: exact native grid sites, unpadded right candidates, in-view signed GT',
                  units=dict(rank_argmax_error='feature cells', refinement_epe='full-resolution pixels'),
                  tie_policy='strictly larger competition rank; ties count top-1; nearest-coordinate and argmax ties choose smaller x')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    temp = a.output.with_suffix('.tmp')
    temp.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    temp.replace(a.output)


if __name__ == '__main__':
    main()
