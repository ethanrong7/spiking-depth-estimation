"""One-variant GPU benchmark on a saved NMRF split; no optimizer steps.

Run each variant/execution combination in its own process for isolated VRAM
accounting. Frozen-checkpoint EPE is a disruption diagnostic, not the accuracy
of a trained selective model. Output and source receipts belong outside Git.
"""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
import numpy as np
import torch

from nmrf_selective_updates import (ALTERNATING_SELECTOR, COMPACT_SUPPORT, FULL_KV_SUPPORT,
                                    LEGACY_SELECTOR, install_selective_updates)


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant', choices=['original', 'full', 'continuous', 'ternary', 'tanh'], required=True)
    p.add_argument('--execution', choices=['eager', 'graph'], required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--keep-fraction', type=float,
                   help='Frozen transplant retention (default 0.5); trained checkpoints use saved retention')
    p.add_argument('--selector', choices=[LEGACY_SELECTOR, ALTERNATING_SELECTOR])
    p.add_argument('--attention-support', choices=[COMPACT_SUPPORT, FULL_KV_SUPPORT])
    p.add_argument('--pairs', type=int, default=0, help='0 = entire saved Dev split; otherwise evenly spaced rows')
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--warmup', type=int, default=10)
    a = p.parse_args()
    if (a.pairs < 0 or min(a.repeats, a.warmup) < 1
            or (a.keep_fraction is not None and not 0 < a.keep_fraction <= 1)):
        p.error('Invalid pair/repeat/warmup/retention count')
    for source in (a.source_root, Path(__file__).resolve().parent):
        if a.output.resolve().is_relative_to(source.resolve()):
            p.error('Keep outputs outside source directories')
    if not torch.cuda.is_available():
        p.error('This benchmark requires CUDA; use the separate tiny CPU probe')
    return a


def load_benchmark_weights(model, state, config, variant, keep_fraction=None,
                           selector=None, attention_support=None):
    """Restore trained architecture identity or run a frozen ANN transplant."""
    if 'format' in state or 'identity' in config:
        from run_nmrf_finetune import FORMAT, restore_architecture
        architecture = state.get('architecture')
        if (state.get('format') != FORMAT or architecture is None or 'identity' not in config
                or architecture != config.get('architecture') or state.get('run_config') != config['identity']
                or state['step'] != config.get('steps')):
            raise ValueError('Expected the completed fine-tuning checkpoint for the supplied config')
        if variant != architecture['variant']:
            raise ValueError('Benchmark variant must match the trained architecture')
        selective = architecture['selective']
        if keep_fraction is not None and (selective is None or keep_fraction != selective['keep_fraction']):
            raise ValueError('Benchmark retention must match the trained architecture')
        if selector is not None and (selective is None or selector != selective['selector']):
            raise ValueError('Benchmark selector must match the trained architecture')
        if attention_support is not None and (
                selective is None or attention_support != selective.get('attention_support', COMPACT_SUPPORT)):
            raise ValueError('Benchmark attention support must match the trained architecture')
        restore_architecture(model, state)
        return architecture

    architecture = dict(architecture='original')
    if variant != 'original':
        fraction = .5 if keep_fraction is None else keep_fraction
        architecture = install_selective_updates(
            model, keep_fraction=1. if variant == 'full' else fraction,
            coding='continuous' if variant == 'full' else variant,
            selector=LEGACY_SELECTOR if selector is None else selector,
            attention_support=COMPACT_SUPPORT if attention_support is None else attention_support)
    elif selector is not None or attention_support is not None:
        raise ValueError('A selector or attention support override requires a selective variant')
    model.load_state_dict(state['model'], strict=True)
    return architecture


@torch.no_grad()
def timed(model, left, right, repeats):
    times, wall = [], []
    # Event construction and data loading are outside the measured interval.
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    torch.cuda.synchronize()
    for _ in range(repeats):
        before = time.perf_counter()
        start.record()
        prediction = model(left, right)['disp']
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
        wall.append((time.perf_counter()-before)*1000)
    return prediction, times, wall


@torch.no_grad()
def main():
    a = arguments()
    from nmrf_inference_runtime import CUDAGraphInference
    sys.path.insert(0, str(a.source_root.resolve()))
    from nmrf_runtime import build_native_nmrf, UPSTREAM_REVISION
    from evaluate_nmrf import input_tensors, metric_counts, summarize
    from runtime import NativeDataset
    from torch.utils.flop_counter import FlopCounterMode

    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    cfg = json.loads(a.config.read_text())
    all_rows = cfg['final_dev_rows']
    if not all_rows:
        raise ValueError('Empty saved Dev split')
    indices = np.linspace(0, len(all_rows)-1, min(a.pairs, len(all_rows)), dtype=int) if a.pairs else range(len(all_rows))
    rows = [all_rows[i] for i in indices]
    dataset = NativeDataset(a.data_root, rows)
    state = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    if state['model_name'] != 'nmrf-native-50' or state['upstream_revision'] != UPSTREAM_REVISION:
        raise ValueError('Expected the pinned ANN NMRF checkpoint')
    model, _, _ = build_native_nmrf(cfg['seed'])
    architecture = load_benchmark_weights(model, state, cfg, a.variant, a.keep_fraction,
                                          a.selector, a.attention_support)
    fine_tuned = 'format' in state
    step = state['step']
    del state
    model = model.cuda().eval()
    a.output.mkdir(parents=True, exist_ok=False)
    sources = ['benchmark_nmrf_selective_updates.py', 'nmrf_selective_updates.py', 'nmrf_inference_runtime.py']
    if fine_tuned:
        sources.append('run_nmrf_finetune.py')
    for name in sources:
        shutil.copyfile(Path(__file__).resolve().parent / name, a.output / name)
    shutil.copyfile(a.config, a.output / 'parent-training-config.json')
    receipt = Path(__file__).resolve().parent / 'source-receipt.json'
    if receipt.exists():
        shutil.copyfile(receipt, a.output / receipt.name)
    config = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
                  architecture=architecture, rows=rows, checkpoint_step=step,
                  upstream_revision=UPSTREAM_REVISION, torch=str(torch.__version__),
                  cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                  compute_capability=list(torch.cuda.get_device_capability()),
                  compiled_architectures=torch.cuda.get_arch_list(), cudnn=torch.backends.cudnn.version(),
                  dtype='FP32', tf32=False, batch_size=1,
                  scope='Forward only, excludes data decoding/transfers/metrics; graph includes input copy/output clone',
                  accuracy_scope=('Fixed endpoint of the recorded fine-tuning run' if fine_tuned else
                                  'Frozen ANN checkpoint transplant, no fine-tuning'))
    try:
        config['nvidia_smi_start'] = subprocess.check_output(['nvidia-smi',
            '--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu',
            '--format=csv,noheader'], text=True, timeout=10).strip()
    except (OSError, subprocess.SubprocessError) as error:
        config['nvidia_smi_error'] = str(error)
    (a.output / 'config.json').write_text(json.dumps(config, indent=2))
    left, right, _, _ = input_tensors(dataset[0], 'cuda', None)
    config['input_shape'] = list(left.shape)
    with FlopCounterMode(display=False) as counter:
        model(left, right)
    flops = counter.get_total_flops()
    for _ in range(a.warmup):
        model(left, right)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    setup_start = time.perf_counter()
    function = CUDAGraphInference(model, (left, right)) if a.execution == 'graph' else model
    torch.cuda.synchronize()
    setup_seconds = time.perf_counter()-setup_start
    setup_peak = torch.cuda.max_memory_allocated()
    for _ in range(3):
        function(left, right)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    records = []
    for index, sample in enumerate(dataset):
        left, right, target, valid = input_tensors(sample, 'cuda', None)
        if list(left.shape) != config['input_shape']:
            raise ValueError('Fixed-shape benchmark received a different native size')
        torch.cuda.reset_peak_memory_stats()
        prediction, times, wall = timed(function, left, right, a.repeats)
        frame = dict(row=rows[index], cuda_ms=times, synchronized_wall_ms=wall,
                     peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                     peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                     metrics=metric_counts(prediction[0], target, valid & (target > 0) & (target < 192)))
        if a.execution == 'graph':
            eager = model(left, right)['disp']
            frame['graph_max_abs_difference'] = float((eager-prediction).abs().max())
            torch.testing.assert_close(prediction, eager, atol=1e-5, rtol=1e-5)
            del eager
        records.append(frame)
        del prediction
        with (a.output / 'frames.jsonl').open('a') as f:
            f.write(json.dumps(frame)+'\n')
        if (index+1) % 10 == 0:
            print(json.dumps(dict(variant=a.variant, execution=a.execution, pairs=index+1)), flush=True)
    scene_counts = defaultdict(list)
    for frame in records:
        scene_counts[frame['row']['scene']].append(frame['metrics'])
    times = [v for r in records for v in r['cuda_ms']]
    wall = [v for r in records for v in r['synchronized_wall_ms']]
    result = dict(**config, counted_contraction_flops=flops,
                  flop_exclusions='coding, normalization, softmax, indexing and memory traffic',
                  mean_cuda_ms=float(np.mean(times)), median_cuda_ms=float(np.median(times)),
                  p90_cuda_ms=float(np.percentile(times, 90)), mean_wall_ms=float(np.mean(wall)),
                  setup_seconds=setup_seconds, setup_peak_allocated_bytes=setup_peak,
                  forward_peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in records),
                  peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in records),
                  validation=summarize([r['metrics'] for r in records]),
                  mean_scene_epe=float(np.mean([summarize(v)['epe'] for v in scene_counts.values()])),
                  max_graph_difference=max(r.get('graph_max_abs_difference', 0) for r in records))
    (a.output / 'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('mean_cuda_ms', 'p90_cuda_ms', 'forward_peak_allocated_bytes',
                                           'counted_contraction_flops', 'validation')}), flush=True)


if __name__ == '__main__':
    main()
