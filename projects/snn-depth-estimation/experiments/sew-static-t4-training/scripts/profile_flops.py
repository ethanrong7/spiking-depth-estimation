"""Architecture FLOPs per stereo pair; run on an allocated cluster GPU."""
import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import socket
import sys
from types import SimpleNamespace

import torch
from torch.utils.flop_counter import FlopCounterMode
from torch.utils._python_dispatch import TorchDispatchMode


class Inventory(TorchDispatchMode):
    """List executed operators, including ones absent from the FLOP counter."""
    def __init__(self):
        super().__init__()
        self.calls = Counter()
        self.matmul_flops = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.calls[str(func._overloadpacket)] += 1
        result = func(*args, **(kwargs or {}))
        if func in (torch.ops.aten.mm.default, torch.ops.aten.bmm.default):
            self.matmul_flops += 2 * result.numel() * args[0].shape[-1]
        elif func == torch.ops.aten.addmm.default:
            self.matmul_flops += 2 * result.numel() * args[1].shape[-1]
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--variant', choices=['A0-upstream', 'A2', 'A3'], required=True)
    parser.add_argument('--iters', type=int, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('GPU allocation required; no local performance profiling')
    sys.path[:0] = [str(args.source), str(args.source / 'core')]
    from core.spikestereonet import SpikeStereoNet
    from spikingjelly.activation_based import functional
    config = json.loads(args.config.read_text())
    torch.manual_seed(config['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    original = args.variant == 'A0-upstream'
    model_args = SimpleNamespace(
        hidden_dims=config['hidden_dims'], context_norm='batch' if original else config['context_norm'],
        n_downsample=config['n_downsample'], n_snn_layers=config['n_snn_layers'],
        shared_backbone=False, feature_encoder='sew' if args.variant == 'A3' else 'cnn',
        context_encoder='cnn' if original else 'sew', sew_timesteps=4,
        sew_input_mode='static_channels', sew_input_bins=50,
        sew_checkpoint_chunk_size=0, sew_spatial_checkpoint=False,
        corr_levels=config['corr_levels'], corr_radius=config['corr_radius'],
        mixed_precision=False, corr_implementation=config['corr_implementation'])
    model = SpikeStereoNet(model_args).cuda().eval()
    inputs = [torch.randint(0, 2, (1, 50, config['height'], config['width']),
                           device='cuda').float() for _ in range(2)]
    conv_flops = defaultdict(int)
    conv_calls = Counter()
    handles = []
    def hook(name):
        def count(module, inp, out):
            if isinstance(module, torch.nn.ConvTranspose2d):
                value = 2 * inp[0].numel() * (module.out_channels // module.groups) * math.prod(module.kernel_size)
            else:
                value = 2 * out.numel() * (module.in_channels // module.groups) * math.prod(module.kernel_size)
            conv_flops[name] += value
            conv_calls[name] += 1
        return count
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            handles.append(module.register_forward_hook(hook(name)))
    inventory = Inventory()
    counter = FlopCounterMode(display=False)
    functional.reset_net(model)
    with torch.no_grad(), inventory, counter:
        result = model(*inputs, iters=args.iters, test_mode=True)
    torch.cuda.synchronize()
    if not bool(torch.isfinite(result[1]).all()):
        raise RuntimeError('Nonfinite prediction')
    for handle in handles:
        handle.remove()
    totals = counter.get_flop_counts()['Global']
    counted = {str(op): int(n) for op, n in totals.items()}
    independent = sum(conv_flops.values()) + inventory.matmul_flops
    total = counter.get_total_flops()
    if independent != total:
        raise RuntimeError(f'Independent conv/matmul count {independent} != torch count {total}; operators={counted}; conv={sum(conv_flops.values())}; matmul={inventory.matmul_flops}')
    if not original:
        if conv_calls.get('cnet.conv1') != 4:
            raise RuntimeError('Context stem did not execute exactly four steps')
        expected_feature_calls = 4 if args.variant == 'A3' else 1
        if conv_calls.get('fnet.conv1') != expected_feature_calls:
            raise RuntimeError('Unexpected feature stem execution count')
    components = defaultdict(int)
    for name, value in conv_flops.items():
        components[name.split('.')[0]] += value
    report = {
        'variant': args.variant, 'refinements': args.iters,
        'input_shape_per_view': list(inputs[0].shape), 'batch_size': 1,
        'encoder_steps': None if original else 4, 'sensor_bins': 50,
        'source': str(args.source), 'configuration': config,
        'effective_model_args': vars(model_args), 'weights': 'seeded random; dense counts are weight-independent',
        'precision': 'FP32, TF32 disabled', 'test_mode': True,
        'parameters': sum(p.numel() for p in model.parameters()),
        'dense_conv_matmul_flops': total, 'gflops_mac_equals_2': total / 1e9,
        'gmac_mac_equals_1': total / 2e9,
        'convolution_flops': sum(conv_flops.values()), 'matrix_flops': inventory.matmul_flops,
        'independent_count_matches': True,
        'component_convolution_flops': dict(components), 'convolution_calls': dict(conv_calls),
        'counted_operators': counted,
        'executed_operators_without_formula': {k: v for k, v in inventory.calls.items() if k not in counted},
        'scope': 'Dense convolutions and matrix products only; bias, normalization, neuron arithmetic, nonlinearities, interpolation, pooling and elementwise arithmetic excluded. No spike-sparsity discount.',
        'paper_reference_giga_operations': 473,
        'paper_counting_convention': 'Not established; report both conventions, do not assert exact comparability.',
        'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(),
        'hostname': socket.gethostname(), 'slurm_job_id': os.getenv('SLURM_JOB_ID'),
        'dataset': 'synthetic binary input; no dataset split read; architecture-only inference count',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('variant', 'refinements', 'gflops_mac_equals_2', 'gmac_mac_equals_1', 'independent_count_matches')}), flush=True)


if __name__ == '__main__':
    main()
