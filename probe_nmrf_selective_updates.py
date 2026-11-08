"""Tiny CPU mechanics/contraction check, not a disparity-accuracy experiment.

Uses random weights and generated binary inputs; no dataset or GPU is accessed.
Store output outside Git. All reported FLOPs are PyTorch-counted contractions
(MAC=2 FLOPs), excluding coding, softmax, indexing and memory movement.
"""
import argparse
import copy
import json
from pathlib import Path
import subprocess

import torch
from torch.utils.flop_counter import FlopCounterMode

from nmrf_runtime import build_native_nmrf, UPSTREAM_REVISION
from nmrf.models.NMP import SwinNMP, WindowAttention
from nmrf_selective_updates import SelectiveSwinNMP, install_selective_updates

ROOT = Path(__file__).resolve().parent


def counted_forward(model, *inputs):
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        result = model(*inputs)
    if isinstance(result, dict):
        result = result['disp']
    return result, counter.get_total_flops()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(ROOT):
        parser.error('Keep reports outside the repository')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    report = dict(seed=args.seed, device='cpu', dtype='float32', torch=str(torch.__version__),
                  upstream_revision=UPSTREAM_REVISION,
                  source_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  source_status=subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
                  data_split='none; generated random binary inputs, no dataset or labels',
                  weights='random initialization; not a trained checkpoint',
                  metric='registered contraction FLOPs; one MAC = two FLOPs',
                  excludes='coding, softmax, normalization, gather/scatter and memory traffic',
                  limitations='No EPE, GPU latency, measured energy or trained quality evidence',
                  block=[], pipeline=[])
    # Retain exact experimental source, including untracked prototype files.
    source_dir = args.output / 'source'
    source_dir.mkdir()
    for name in ('nmrf_selective_updates.py', 'probe_nmrf_selective_updates.py',
                 'tests/test_nmrf_selective_updates.py', 'nmrf_runtime.py'):
        target = source_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    (args.output / 'source.diff').write_text(subprocess.check_output(
        ['git', 'diff', 'HEAD'], cwd=ROOT, text=True))

    base = SwinNMP(128, 159, 4, window_size=6, shift_size=3).eval()
    base.H = base.W = 12
    with torch.no_grad():
        base.attn.relative_position_enc_table.normal_(0, .02)
    x = torch.randn(144, 4, 128)
    g = torch.randn(144, 4, 31)
    mask = WindowAttention.gen_shift_window_attn_mask((12, 12), (6, 6, 4), 3, 'cpu')
    reference, original_flops = counted_forward(base, x, g, mask)
    report['block'].append(dict(name='upstream', flops=original_flops))
    for fraction, coding in ((1., 'continuous'), (.5, 'continuous'), (.5, 'tanh'), (.5, 'ternary')):
        block = SelectiveSwinNMP(copy.deepcopy(base), fraction, coding, phase=1)
        result, flops = counted_forward(block, x, g, mask)
        assert torch.isfinite(result).all()
        inactive = torch.ones(36, dtype=torch.bool)
        inactive[block.selected_pixels('cpu')] = False
        unchanged = torch.equal(block.partition(result)[:, inactive], block.partition(x)[:, inactive])
        assert unchanged
        if fraction == 1:
            torch.testing.assert_close(result, reference, atol=2e-6, rtol=2e-5)
        report['block'].append(dict(name=f'{coding}_{fraction}', flops=flops,
                                    reduction=1-flops/original_flops,
                                    output_max_abs_change=float((result-reference).abs().max()),
                                    inactive_bitwise_unchanged=unchanged,
                                    selected_pixels=int((~inactive).sum()), candidates_per_pixel=4))

    parent, _, _ = build_native_nmrf(args.seed)
    parent.eval()
    left = torch.randint(0, 2, (1, 50, 32, 64)).float()
    right = torch.randint(0, 2, left.shape).float()
    reference, full_flops = counted_forward(parent, left, right)
    report['pipeline'].append(dict(name='upstream', flops=full_flops))
    for fraction, coding in ((1., 'continuous'), (.5, 'continuous'), (.5, 'tanh'), (.5, 'ternary')):
        model = copy.deepcopy(parent)
        config = install_selective_updates(model, fraction, coding)
        result, flops = counted_forward(model, left, right)
        assert torch.isfinite(result).all()
        if fraction == 1:
            torch.testing.assert_close(result, reference, atol=5e-4, rtol=1e-4)
        report['pipeline'].append(dict(name=f'{coding}_{fraction}', configuration=config,
                                       flops=flops, reduction=1-flops/full_flops,
                                       output_max_abs_change=float((result-reference).abs().max()),
                                       output_shape=list(result.shape)))
    report['pipeline_input_shape'] = list(left.shape)
    (args.output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    for scope in ('block', 'pipeline'):
        print(scope)
        for row in report[scope]:
            print(row['name'], f"{row['flops']/1e9:.6f} counted GFLOPs",
                  f"{row.get('reduction',0):.2%} reduction")
    print('No trained accuracy or GPU speed conclusion follows from this CPU probe.')


if __name__ == '__main__':
    main()
