"""Evaluate a native 50-bin NMRF checkpoint without optimizer steps.

Outputs belong outside the repository. Uses the checkpoint's saved split, a
scene-balanced training diagnostic, and all development pairs by default.
"""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent


def metric_counts(prediction, target, valid):
    """Sufficient statistics; never silently discard nonfinite predictions."""
    if prediction.shape != target.shape or valid.shape != target.shape:
        raise ValueError('Prediction, target and mask must have equal shapes')
    if not torch.isfinite(prediction).all():
        raise FloatingPointError('Nonfinite prediction')
    mask = valid.bool() & torch.isfinite(target) & (target > 0)
    errors = (prediction - target).abs()[mask].double()
    return dict(pixels=errors.numel(), absolute_error=float(errors.sum()),
                bad1_pixels=int((errors > 1).sum()), bad3_pixels=int((errors > 3).sum()))


def summarize(counts):
    totals = {key: sum(c[key] for c in counts)
              for key in ('pixels', 'absolute_error', 'bad1_pixels', 'bad3_pixels')}
    n = totals['pixels']
    return dict(**totals, epe=totals['absolute_error']/n if n else None,
                bad1_percent=100*totals['bad1_pixels']/n if n else None,
                bad3_percent=100*totals['bad3_pixels']/n if n else None)


def select_train(rows):
    """One first-listed pair per training scene, fixed before looking at errors."""
    selected = {}
    for row in rows:
        selected.setdefault(row['scene'], row)
    return list(selected.values())


def input_tensors(sample, device, crop):
    left, right, flow, valid, _ = sample
    target = -flow[0]
    if crop:
        h, w = target.shape
        ch, cw = crop
        if ch > h or cw > w or min(ch, cw) < 8 or ch % 8 or cw % 8:
            raise ValueError('Crop must fit input and have dimensions divisible by eight')
        y, x = (h-ch)//2, (w-cw)//2
        left, right = left[:, y:y+ch, x:x+cw], right[:, y:y+ch, x:x+cw]
        target, valid = target[y:y+ch, x:x+cw], valid[y:y+ch, x:x+cw]
    # Full images use NMRF's own evaluation padding/unpadding, with no resizing.
    return (left[None].to(device), right[None].to(device),
            target.to(device), valid.to(device))


@torch.inference_mode()
def profile(model, left, right, repeats, warmup):
    """CUDA event timings. Sibling-module intervals are disjoint, summed per call.

    Whole-forward timing without hooks is primary. Instrumented stage timings
    include launch gaps and are diagnostic, not operation counts or energy.
    """
    for _ in range(warmup):
        model(left, right)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times, wall = [], []
    for _ in range(repeats):
        start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        t = time.perf_counter()
        start.record()
        model(left, right)
        end.record(); end.synchronize()
        times.append(start.elapsed_time(end))
        wall.append(1000*(time.perf_counter()-t))
    allocated = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.max_memory_reserved()
    names = ('backbone', 'dpn', 'concatconv', 'gw', 'inference', 'infer_head',
             'infer_score_head', 'refinement', 'refine_head')
    intervals = defaultdict(list)
    pending = {}
    handles = []

    def before(name):
        def hook(module, inputs):
            event = torch.cuda.Event(enable_timing=True)
            event.record(); pending[name] = event
        return hook

    def after(name):
        def hook(module, inputs, output):
            event = torch.cuda.Event(enable_timing=True)
            event.record(); intervals[name].append((pending.pop(name), event))
        return hook

    for name in names:
        module = getattr(model.parent, name, None)
        if module is not None:
            handles.extend([module.register_forward_pre_hook(before(name)),
                            module.register_forward_hook(after(name))])
    instrumented, stage_times = [], defaultdict(list)
    try:
        for _ in range(repeats):
            intervals.clear()
            start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            start.record(); model(left, right); end.record(); end.synchronize()
            instrumented.append(start.elapsed_time(end))
            for name, events in intervals.items():
                stage_times[name].append(sum(s.elapsed_time(e) for s, e in events))
    finally:
        for handle in handles:
            handle.remove()
    mean = float(np.mean(instrumented))
    stages = {name: dict(mean_ms=float(np.mean(values)),
                         percent_instrumented_forward=100*float(np.mean(values))/mean)
              for name, values in stage_times.items()}
    return dict(batch_size=1, dtype='float32', input_shape=list(left.shape),
                repeats=repeats, warmup=warmup, mean_cuda_ms=float(np.mean(times)),
                median_cuda_ms=float(np.median(times)), p90_cuda_ms=float(np.percentile(times, 90)),
                mean_synchronized_wall_ms=float(np.mean(wall)),
                peak_allocated_bytes=allocated, peak_reserved_bytes=reserved,
                instrumented_mean_ms=mean, stages=stages,
                unassigned_mean_ms=mean-sum(s['mean_ms'] for s in stages.values()),
                scope='Forward only, both views; excludes decoding, host-to-device copies and metrics. '
                      'Stage intervals include GPU launch gaps. No FLOPs or energy claim.')


def save_example(path, left, target, prediction, valid, row):
    np.savez_compressed(path, spike_count=left[0].sum(0).cpu().numpy(),
                        target=target.cpu().numpy(), prediction=prediction.cpu().numpy(),
                        valid=valid.cpu().numpy(), scene=row['scene'], frame=row['frame'])


@torch.inference_mode()
def evaluate(model, dataset, rows, device, crop, output, max_disp, center_reference=None):
    results, worst = [], (-1, None)
    started = time.monotonic()
    output.mkdir()
    with (output/'frames.jsonl').open('w', encoding='utf-8') as log:
        for i, row in enumerate(rows):
            left, right, target, valid = input_tensors(dataset[i], device, crop)
            prediction = model(left, right)['disp'][0]
            in_range = valid & (target > 0) & (target < max_disp)
            # This is geometric availability, NOT an occlusion ground-truth mask.
            x = torch.arange(target.shape[-1], device=device)[None, :]
            in_view = in_range & (x-target >= 0)
            counts = {name: metric_counts(prediction, target, mask) for name, mask in
                      [('all_dataset_valid', valid), ('training_range', in_range),
                       ('training_range_right_in_view', in_view)]}
            if center_reference:
                ch, cw = center_reference
                h, w = target.shape
                if ch > h or cw > w:
                    raise ValueError('Center reference must fit native image')
                y, x0 = (h-ch)//2, (w-cw)//2
                region = (slice(y, y+ch), slice(x0, x0+cw))
                counts['training_range_center'] = metric_counts(
                    prediction[region], target[region], in_range[region])
            record = dict(index=i, scene=row['scene'], frame=row['frame'],
                          shape=list(target.shape), metrics=counts)
            results.append(record)
            log.write(json.dumps(record)+'\n'); log.flush()
            epe = summarize([counts['training_range']])['epe']
            if epe is None:
                raise ValueError(f'No in-range valid pixels in {row}')
            if i < 2:
                save_example(output/f'fixed-{i}.npz', left, target, prediction, in_range, row)
            if epe > worst[0]:
                worst = (epe, row)
                save_example(output/'worst.npz', left, target, prediction, in_range, row)
            if (i+1) % 20 == 0 or i+1 == len(rows):
                print(json.dumps(dict(group=output.name, completed=i+1, pairs=len(rows),
                                      epe=summarize([r['metrics']['training_range'] for r in results])['epe'])),
                      flush=True)
    summaries = {}
    for name in results[0]['metrics']:
        scenes = defaultdict(list)
        for row in results:
            scenes[row['scene']].append(row['metrics'][name])
        per_scene = {scene: summarize(counts) for scene, counts in scenes.items()}
        values = [s['epe'] for s in per_scene.values() if s['epe'] is not None]
        summaries[name] = dict(**summarize([r['metrics'][name] for r in results]),
                               scene_macro_epe=float(np.mean(values)) if values else None,
                               scene_p90_epe=float(np.percentile(values, 90)) if values else None,
                               per_scene=per_scene)
    result = dict(pairs=len(rows), scenes=len(set(r['scene'] for r in rows)),
                  crop=crop, seconds=time.monotonic()-started, metrics=summaries,
                  worst_frame=worst[1], worst_frame_epe=worst[0])
    (output/'summary.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    return result


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--manifest', type=Path, help='Defaults to checkpoint directory split.json')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--crop', type=int, nargs=2, default=[128, 256])
    p.add_argument('--splits', nargs='+', choices=['train', 'dev'], default=['train', 'dev'])
    p.add_argument('--geometries', nargs='+', choices=['crop', 'native'], default=['crop', 'native'])
    p.add_argument('--limit', type=int, help='Smoke check only; cap pairs per split')
    p.add_argument('--profile-repeats', type=int, default=20)
    p.add_argument('--profile-warmup', type=int, default=5)
    a = p.parse_args()
    if a.output.resolve().is_relative_to(ROOT):
        p.error('Keep evaluation outputs outside the repository')
    if a.profile_repeats < 1 or a.profile_warmup < 1 or (a.limit is not None and a.limit < 1):
        p.error('Counts must be positive')
    if not torch.cuda.is_available():
        p.error('Run evaluation on the authorized CUDA resource')
    return a


def main():
    from nmrf_runtime import build_native_nmrf, UPSTREAM_REVISION
    from runtime import NativeDataset
    a = arguments()
    torch.set_num_threads(2)
    manifest = json.loads((a.manifest or a.checkpoint.parent/'split.json').read_text(encoding='utf-8'))
    if {r['scene'] for r in manifest['train']} & {r['scene'] for r in manifest['dev']}:
        raise ValueError('Training and development scenes overlap')
    training = json.loads((a.checkpoint.parent/'config.json').read_text(encoding='utf-8'))
    model, criterion, cfg = build_native_nmrf(training['seed'])
    checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    if checkpoint['model_name'] != 'nmrf-native-50' or checkpoint['upstream_revision'] != UPSTREAM_REVISION:
        raise ValueError('Checkpoint architecture or upstream revision mismatch')
    model.load_state_dict(checkpoint['model'], strict=True)
    step = checkpoint['step']
    del checkpoint
    model.eval().cuda()
    selected = {'train': select_train(manifest['train']), 'dev': manifest['dev']}
    selected = {k: v for k, v in selected.items() if k in a.splits}
    if a.limit:
        selected = {k: v[:a.limit] for k, v in selected.items()}
    a.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, a.output/'evaluate_nmrf.py')
    for name in ('config.json', 'nmrf-config.yaml', 'split.json'):
        shutil.copy2(a.checkpoint.parent/name, a.output/f'training-{name}')
    (a.output/'selected-split.json').write_text(json.dumps(selected, indent=2)+'\n', encoding='utf-8')
    source = (ROOT/'evaluation-source.json')
    if source.exists():
        provenance = json.loads(source.read_text(encoding='utf-8'))
    else:
        provenance = {name: subprocess.check_output(['git', '-C', str(ROOT), *args], text=True)
                      for name, args in [('revision', ['rev-parse', 'HEAD']),
                                         ('diff', ['diff', 'HEAD'])]}
    config = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
                  checkpoint_step=step, training_config=training, evaluation_source=provenance,
                  torch=str(torch.__version__), cuda=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(), inference_dtype='float32',
                  matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                  cudnn_tf32=torch.backends.cudnn.allow_tf32,
                  model='Continuous ANN NMRF, 50-bin sensor input',
                  training_range=f'0 < disparity < {criterion.max_disp}',
                  all_dataset_valid='Reader mask: finite positive depth, disparity < 512',
                  bad3='Absolute disparity error strictly greater than 3 native pixels',
                  crop_policy='Identical centered crop in both views; no resizing. Full uses native H/W.',
                  visibility_policy='Primary metric includes out-of-view correspondences. '
                                    'Secondary mask checks x_left-disparity >= 0; not an occlusion mask.',
                  train_selection='First-listed pair per scene; diagnostic only',
                  dev_selection='All manifest dev pairs unless --limit; confirmation split untouched')
    (a.output/'config.json').write_text(json.dumps(config, indent=2)+'\n', encoding='utf-8')
    result = dict(checkpoint_step=step, groups={}, profiles={})
    for geometry, crop in [('crop', a.crop), ('native', None)]:
        if geometry not in a.geometries:
            continue
        for split, rows in selected.items():
            dataset = NativeDataset(a.data_root, rows)
            key = f'{split}-{geometry}'
            result['groups'][key] = evaluate(model, dataset, rows, 'cuda', crop,
                                              a.output/key, criterion.max_disp,
                                              center_reference=a.crop if geometry == 'native' else None)
        sample = NativeDataset(a.data_root, selected.get('dev', next(iter(selected.values()))))[0]
        left, right, _, _ = input_tensors(sample, 'cuda', crop)
        result['profiles'][geometry] = profile(model, left, right, a.profile_repeats, a.profile_warmup)
        (a.output/'results.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(completed=True, output=str(a.output))), flush=True)


if __name__ == '__main__':
    main()
