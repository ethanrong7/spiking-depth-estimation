"""Paired GPU latency/output checks of inference-only NMRF optimizations.

Loads saved model and split. No training. Experimental BN folding is screened
separately and never changes checkpoint files or the standard model builder.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')

import torch
from torch import nn

from nmrf_inference_runtime import CUDAGraphInference


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=['ann', 'spiking'], required=True)
    p.add_argument('--variant', choices=['graph', 'backbone-graph', 'bn-fold'], required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--pairs', type=int, default=0, help='0 = full saved dev split')
    p.add_argument('--repeats', type=int, default=3)
    a = p.parse_args()
    if a.repeats < 1 or a.pairs < 0:
        p.error('Invalid repeat/pair count')
    if a.output.resolve().is_relative_to(a.source_root.resolve()):
        p.error('Write results outside source')
    return a


@torch.no_grad()
def timed(model, left, right, repeats):
    samples, wall = [], []
    torch.cuda.synchronize()
    for _ in range(repeats):
        begin, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        started = time.perf_counter()
        begin.record()
        output = model(left, right)
        end.record(); end.synchronize()
        samples.append(begin.elapsed_time(end))
        wall.append((time.perf_counter()-started)*1000)
    return output['disp'], samples, wall


@torch.no_grad()
def main():
    a = arguments()
    sys.path.insert(0, str(a.source_root.resolve()))
    from nmrf_runtime import build_native_nmrf, UPSTREAM_REVISION
    from spiking_nmrf import build_spiking_nmrf, TemporalConv, MODEL_NAME
    from evaluate_nmrf import input_tensors, metric_counts, summarize
    from runtime import NativeDataset
    import numpy as np
    import shutil

    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.70)
    config = json.loads(a.config.read_text())
    rows = config['final_dev_rows'][:a.pairs or None]
    dataset = NativeDataset(a.data_root, rows)
    state = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    expected = MODEL_NAME if a.model == 'spiking' else 'nmrf-native-50'
    if state['model_name'] != expected or state['upstream_revision'] != UPSTREAM_REVISION:
        raise ValueError('Checkpoint model/source mismatch')
    model, _, _ = (build_spiking_nmrf if a.model == 'spiking' else build_native_nmrf)(config['seed'])
    model.load_state_dict(state['model'], strict=True)
    step = state['step']
    del state
    model = model.cuda().eval()
    a.output.mkdir(parents=True, exist_ok=False)
    for filename in ['benchmark_nmrf_inference_runtime.py', 'nmrf_inference_runtime.py']:
        shutil.copyfile(Path(__file__).parent/filename, a.output/filename)
    shutil.copyfile(a.config, a.output/'training-config.json')
    left, right, _, _ = input_tensors(dataset[0], 'cuda', None)
    for _ in range(5):
        model(left, right)
    torch.cuda.synchronize()
    started = time.monotonic()
    if a.variant == 'graph':
        optimized = CUDAGraphInference(model, (left, right))
    elif a.variant == 'backbone-graph':
        optimized = copy.deepcopy(model)
        captured = []
        handle = model.parent.backbone.register_forward_pre_hook(lambda m, args: captured.append(args[0]))
        model(left, right)
        handle.remove()
        optimized.parent.backbone = CUDAGraphInference(optimized.parent.backbone, (captured[0],))
    else:
        if a.model != 'spiking':
            raise ValueError('BN folding experiment is for the spiking backbone')
        optimized = copy.deepcopy(model)
        for module in optimized.parent.backbone.modules():
            if isinstance(module, TemporalConv) and isinstance(module.norm, nn.BatchNorm2d):
                module.conv = torch.nn.utils.fuse_conv_bn_eval(module.conv, module.norm)
                module.norm = nn.Identity()
    torch.cuda.synchronize()
    setup_seconds = time.monotonic()-started
    for _ in range(3):
        optimized(left, right)
    records = []
    for index, sample in enumerate(dataset):
        left, right, target, valid = input_tensors(sample, 'cuda', None)
        functions = [('baseline', model), ('optimized', optimized)]
        if index % 2:
            functions.reverse()
        predictions = {}
        frame = dict(row=rows[index])
        for name, function in functions:
            prediction, times, wall = timed(function, left, right, a.repeats)
            predictions[name] = prediction
            frame[name] = dict(cuda_ms=times, synchronized_wall_ms=wall,
                              metrics=metric_counts(prediction[0], target, valid & (target>0) & (target<192)))
        delta = (predictions['baseline']-predictions['optimized']).abs()
        frame['identical'] = torch.equal(predictions['baseline'], predictions['optimized'])
        frame['max_abs_delta_px'] = float(delta.max())
        frame['changed_pixels'] = int(torch.count_nonzero(delta))
        frame['mean_abs_delta_px'] = float(delta.double().mean())
        records.append(frame)
        with (a.output/'frames.jsonl').open('a') as log:
            log.write(json.dumps(frame)+'\n')
        if (index+1) % 10 == 0:
            print(json.dumps(dict(model=a.model, variant=a.variant, pairs=index+1,
                                 all_identical=all(r['identical'] for r in records))), flush=True)
    result = dict(model=a.model, variant=a.variant, checkpoint=str(a.checkpoint), checkpoint_step=step,
                  upstream_revision=UPSTREAM_REVISION, rows=rows, setup_seconds=setup_seconds,
                  torch_version=torch.__version__, cuda_version=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(), batch_size=1, dtype='FP32',
                  repetitions_per_pair=a.repeats, paired_order='Alternated baseline/optimized per frame',
                  scope='Forward only, includes graph input copy and output clones; excludes IO/transfers/metrics and one-time graph capture.',
                  all_predictions_bitwise_identical=all(r['identical'] for r in records),
                  max_abs_delta_px=max(r['max_abs_delta_px'] for r in records),
                  changed_pixels=sum(r['changed_pixels'] for r in records),
                  allocated_bytes_with_both_models=torch.cuda.memory_allocated(),
                  reserved_bytes_with_both_models=torch.cuda.memory_reserved())
    for name in ['baseline', 'optimized']:
        times = [t for r in records for t in r[name]['cuda_ms']]
        wall = [t for r in records for t in r[name]['synchronized_wall_ms']]
        result[name] = dict(mean_cuda_ms=float(np.mean(times)), median_cuda_ms=float(np.median(times)),
                            p90_cuda_ms=float(np.percentile(times, 90)), mean_wall_ms=float(np.mean(wall)),
                            validation=summarize([r[name]['metrics'] for r in records]))
    result['speedup'] = result['baseline']['mean_cuda_ms']/result['optimized']['mean_cuda_ms']
    (a.output/'summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
