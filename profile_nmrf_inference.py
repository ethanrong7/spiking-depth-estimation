"""Inference arithmetic/activity audit of saved ANN or SDSA NMRF pilots.

No model changes or optimizer steps. Dense contraction FLOPs use PyTorch's
registry (multiply + add = 2). Activity-derived ACs are a hardware opportunity,
not measured GPU operations or energy. Outputs must be outside the checkout.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.flop_counter import FlopCounterMode, flop_registry
from torch.utils._python_dispatch import TorchDispatchMode


def active_connections(module, x):
    """Exact nonzero input-to-output synapses, excluding zero-padding edges.

    Summing input activity over batch/channels before a one-channel convolution
    counts spatial reuse, including stride, dilation and groups. All groups
    have the same output fanout. Float64 avoids large-count rounding.
    """
    if module.padding_mode != 'zeros':
        raise ValueError('Only zero-padding convolution is supported')
    activity = (x != 0).sum((0, 1), keepdim=True).double()
    kernel = torch.ones((1, 1, *module.kernel_size), device=x.device, dtype=torch.float64)
    uses = F.conv2d(activity, kernel, stride=module.stride,
                    padding=module.padding, dilation=module.dilation).sum()
    return int(uses.item()) * (module.out_channels // module.groups)


class ActivityAudit:
    """Hooks run separately from FLOP counting and timing."""
    def __init__(self, model, kind):
        from spiking_nmrf import LIF, TemporalConv
        self.handles = []
        self.rows = {}
        self.lif = {}
        backbone = model.parent.backbone
        eligible = {id(m.conv) for m in backbone.modules() if isinstance(m, TemporalConv)}
        if kind == 'ann':
            eligible = {id(backbone.conv1)}
        for name, module in backbone.named_modules():
            if isinstance(module, nn.Conv2d):
                def conv_hook(m, args, out, name=name):
                    x = args[0]
                    discrete = id(m) in eligible
                    signed = kind == 'ann' and discrete
                    legal = ((x == -1) | (x == 1)) if signed else ((x == 0) | (x == 1))
                    if discrete and not bool(legal.all()):
                        raise ValueError(f'Non-discrete input at {name}')
                    dense = out.numel() * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
                    row = self.rows.setdefault(name, dict(dense_macs=0, active_acs=0, active_macs=0, nonzero=0, elements=0,
                                                         encoding=('signed' if signed else 'binary') if discrete else 'continuous'))
                    row['dense_macs'] += dense
                    row['active_acs' if discrete else 'active_macs'] += active_connections(m, x)
                    row['nonzero'] += int(torch.count_nonzero(x))
                    row['elements'] += x.numel()
                self.handles.append(module.register_forward_hook(conv_hook))
            if isinstance(module, LIF):
                def lif_hook(m, args, out, name=name):
                    row = self.lif.setdefault(name, dict(spikes=0, updates=0))
                    row['spikes'] += int(torch.count_nonzero(out))
                    row['updates'] += out.numel()
                self.handles.append(module.register_forward_hook(lif_hook))

    def close(self):
        for handle in self.handles:
            handle.remove()


class OperatorInventory(TorchDispatchMode):
    """Coverage inventory; element counts are NOT claimed to be FLOPs."""
    def __init__(self):
        super().__init__()
        self.calls = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.calls[str(func._overloadpacket)] += 1
        return func(*args, **(kwargs or {}))


@torch.no_grad()
def contraction_counts(model, left, right):
    inventory = OperatorInventory()
    with inventory, FlopCounterMode(display=False) as counter:
        prediction = model(left, right)['disp']
    counts = counter.get_flop_counts()
    totals = {name: sum(ops.values()) for name, ops in counts.items()}
    backbone_keys = [name for name in totals if name.endswith('.backbone')]
    if len(backbone_keys) != 1:
        raise RuntimeError(f'Ambiguous backbone scope: {backbone_keys}')
    total = counter.get_total_flops()
    backbone = totals[backbone_keys[0]]
    return prediction, dict(total_flops=total, backbone_flops=backbone,
                            continuous_rest_flops=total-backbone,
                            by_module=totals,
                            global_operations={str(k): v for k, v in counts['Global'].items()},
                            operator_calls=dict(inventory.calls),
                            registered_operators=sorted(str(k) for k in flop_registry))


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=['ann', 'spiking'], required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--warmup', type=int, default=5)
    a = p.parse_args()
    if a.repeats < 1 or a.warmup < 1:
        p.error('Positive repeats/warmup required')
    if a.output.resolve().is_relative_to(a.source_root.resolve()):
        p.error('Outputs must be outside source')
    return a


@torch.no_grad()
def main():
    a = arguments()
    sys.path.insert(0, str(a.source_root.resolve()))
    from evaluate_nmrf import input_tensors, metric_counts, summarize, profile
    from nmrf_runtime import build_native_nmrf, UPSTREAM_REVISION
    from spiking_nmrf import build_spiking_nmrf, MODEL_NAME
    from runtime import NativeDataset
    import numpy as np
    import shutil

    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.70)
    config = json.loads(a.config.read_text())
    rows = config['final_dev_rows']
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
    shutil.copyfile(__file__, a.output/'profile_nmrf_inference.py')
    shutil.copyfile(a.config, a.output/'training-config.json')
    dataset = NativeDataset(a.data_root, rows)
    result = dict(model=a.model, checkpoint=str(a.checkpoint), checkpoint_step=step,
                  upstream_revision=UPSTREAM_REVISION, torch_version=torch.__version__,
                  cuda_version=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                  dtype='FP32', batch_size=1, sensor_bins=50, rows=rows,
                  parameters=sum(p.numel() for p in model.parameters()),
                  definitions=dict(dense_flops='PyTorch registered contractions; one MAC = two FLOPs',
                    excluded='Bias, normalization, activation, LIF, gates, cost-volume elementwise arithmetic, '
                             'sampling, reductions, indexing and memory traffic are not covered by contraction FLOPs. '
                             'Operator inventory and LIF updates are supplied separately.',
                    ac_estimate='Exact nonzero discrete synaptic connections with spatial fanout; zero padding excluded. '
                                'SNN binary temporal convolutions; ANN signed-input stem also eligible. '
                                'Readouts and matcher remain continuous. Continuous backbone convolutions also '
                                'report nonzero MAC opportunities. No zero-skipping or energy savings measured.',
                    timing=f'Uninstrumented forward with inputs already on GPU; no decoding/transfers/metrics. '
                           f'{a.repeats} synchronized repetitions per native pair, {a.warmup} warmups on first pair.'))
    counts_by_shape = {}
    frame_rows = []
    started = time.monotonic()
    for index, sample in enumerate(dataset):
        left, right, target, valid = input_tensors(sample, 'cuda', None)
        shape = str(list(left.shape))
        if index == 0:
            result['stage_timing_first_pair'] = profile(model, left, right, repeats=10, warmup=a.warmup)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        milliseconds = []
        for _ in range(a.repeats):
            begin, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            begin.record()
            prediction = model(left, right)['disp']
            end.record(); end.synchronize()
            milliseconds.append(begin.elapsed_time(end))
        peak = torch.cuda.max_memory_allocated()
        metrics = metric_counts(prediction[0], target, valid & (target > 0) & (target < 192))
        if shape not in counts_by_shape:
            profiled, counts = contraction_counts(model, left, right)
            if not torch.allclose(prediction, profiled, rtol=1e-5, atol=1e-5):
                raise RuntimeError('FLOP instrumentation changed prediction')
            counts_by_shape[shape] = counts
            del profiled
        audit = ActivityAudit(model, a.model)
        try:
            activity_prediction = model(left, right)['disp']
        finally:
            audit.close()
        if not torch.equal(prediction, activity_prediction):
            raise RuntimeError('Activity instrumentation changed prediction')
        frame = dict(row=rows[index], shape=shape, cuda_ms=milliseconds, peak_allocated_bytes=peak,
                     metrics=metrics, synapses=audit.rows, lif=audit.lif)
        frame_rows.append(frame)
        with (a.output/'frames.jsonl').open('a') as log:
            log.write(json.dumps(frame)+'\n')
        del prediction, activity_prediction, left, right, target, valid
        if (index+1) % 10 == 0:
            print(json.dumps(dict(model=a.model, pairs=index+1, seconds=time.monotonic()-started)), flush=True)
    result['contractions_by_shape'] = counts_by_shape
    result['validation'] = summarize([f['metrics'] for f in frame_rows])
    timings = [t for f in frame_rows for t in f['cuda_ms']]
    result['timing'] = dict(mean_ms=float(np.mean(timings)), median_ms=float(np.median(timings)),
                            p90_ms=float(np.percentile(timings, 90)), forwards=len(timings),
                            peak_allocated_bytes=max(f['peak_allocated_bytes'] for f in frame_rows))
    result['mean_per_pair'] = dict(
        dense_flops=float(np.mean([counts_by_shape[f['shape']]['total_flops'] for f in frame_rows])),
        backbone_flops=float(np.mean([counts_by_shape[f['shape']]['backbone_flops'] for f in frame_rows])),
        continuous_rest_flops=float(np.mean([counts_by_shape[f['shape']]['continuous_rest_flops'] for f in frame_rows])),
        eligible_dense_macs=float(np.mean([sum(r['dense_macs'] for r in f['synapses'].values() if r['encoding'] != 'continuous') for f in frame_rows])),
        active_acs=float(np.mean([sum(r['active_acs'] for r in f['synapses'].values()) for f in frame_rows])),
        continuous_backbone_dense_macs=float(np.mean([sum(r['dense_macs'] for r in f['synapses'].values() if r['encoding'] == 'continuous') for f in frame_rows])),
        continuous_backbone_active_macs=float(np.mean([sum(r['active_macs'] for r in f['synapses'].values()) for f in frame_rows])),
        lif_updates=float(np.mean([sum(r['updates'] for r in f['lif'].values()) for f in frame_rows])))
    result['lif_rates'] = {name: sum(f['lif'][name]['spikes'] for f in frame_rows)/sum(f['lif'][name]['updates'] for f in frame_rows)
                           for name in frame_rows[0]['lif']}
    result['seconds'] = time.monotonic()-started
    (a.output/'summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ['model', 'validation', 'timing', 'mean_per_pair', 'seconds']}), flush=True)


if __name__ == '__main__':
    main()
