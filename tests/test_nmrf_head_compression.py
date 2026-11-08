import copy
import io
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F
from torch.utils.flop_counter import FlopCounterMode

from nmrf_head_compression import (FORMAT, ReducedHeadSwinNMP, install_head_compression,
                                   restore_head_compression)
from nmrf_runtime import build_native_nmrf
from nmrf.models.NMP import SwinNMP, WindowAttention
from nmrf_selective_updates import SelectiveSwinNMP


def parity_case(shift, coding, heads):
    torch.manual_seed(81)
    dim, head_dim, window, candidates = 16, 4, 6, 4
    original = SwinNMP(dim, 23, 4, window_size=window, shift_size=shift,
                       normalize_before=True).double().eval()
    original.H = original.W = 12
    with torch.no_grad():
        original.attn.relative_position_enc_table.normal_(0, .2)
    projection = None
    columns = torch.cat([torch.arange(h*head_dim, (h+1)*head_dim) for h in heads])
    if len(heads) < 4:
        projection = dict(weight=torch.randn(dim, len(columns))*.2, bias=torch.randn(dim)*.2)
    rng = torch.get_rng_state()
    narrow = ReducedHeadSwinNMP(copy.deepcopy(original), heads, coding, projection)
    assert torch.equal(torch.get_rng_state(), rng)
    if projection is not None:
        with torch.no_grad():
            original.proj.weight.zero_()
            original.proj.weight.index_copy_(1, columns, projection['weight'].double())
            original.proj.bias.copy_(projection['bias'].double())
    reference = (original if coding == 'continuous'
                 else SelectiveSwinNMP(original, keep_fraction=1., coding='ternary'))
    batch = 2 if shift else 1

    def padded(ch):
        source = torch.randn(batch, 10, 10, candidates, ch, dtype=torch.float64)
        return F.pad(source, (0, 0, 0, 0, 1, 1, 1, 1)).reshape(-1, candidates, ch).requires_grad_()

    label, geometry = padded(dim), padded(7)
    size = (window, window, candidates)
    mask = (WindowAttention.gen_shift_window_attn_mask((12, 12), size, shift, 'cpu') if shift else
            WindowAttention.gen_window_attn_mask(size, 'cpu')[None]).double()
    expected, actual = reference(label, geometry, mask), narrow(label, geometry, mask)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-11)
    assert actual.shape == label.shape and not torch.equal(actual, label)
    probe = torch.randn_like(actual)
    names = ['input.label', 'input.geometry', *dict(narrow.named_parameters())]

    def gradients(module, result):
        params = dict(module.named_parameters())
        variables = (label, geometry, *(params[name] for name in names[2:]))
        return dict(zip(names, torch.autograd.grad((result*probe).sum(), variables)))

    grad_actual, grad_expected = gradients(narrow, actual), gradients(reference, expected)
    qkv_rows = (torch.arange(3)[:, None]*dim+columns).flatten()
    dropped = [head for head in range(4) if head not in heads]
    dropped_cols = [col for col in range(3*dim) if col not in qkv_rows.tolist()]
    max_error = 0.
    for name, got in grad_actual.items():
        ref = grad_expected[name]
        if name in ('qkv.weight', 'qkv.bias'):
            if dropped_cols:
                assert torch.count_nonzero(ref[dropped_cols]) == 0
            ref = ref.index_select(0, qkv_rows)
        elif name == 'attn.relative_position_enc_table':
            ref = ref.reshape(-1, 4, 3*head_dim)
            if dropped:
                assert torch.count_nonzero(ref[:, dropped]) == 0
            ref = ref[:, heads].flatten(1)
        elif name == 'proj.weight':
            ref = ref.index_select(1, columns)
        torch.testing.assert_close(got, ref, rtol=1e-9, atol=1e-10)
        max_error = max(max_error, float((got-ref).abs().max()))
    return dict(max_output_abs=float((actual-expected).detach().abs().max()),
                max_gradient_abs=max_error)


class ReducedHeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_full_width_and_narrow_forward_backward(self):
        for shift in (0, 3):
            for coding in ('continuous', 'ternary'):
                with self.subTest(shift=shift, coding=coding):
                    parity_case(shift, coding, (0, 1, 2, 3))
                    parity_case(shift, coding, (1, 3))

    def test_native_width_shapes_full_attention_support_and_contraction_count(self):
        base = SwinNMP(128, 159, 4, window_size=6, normalize_before=True).eval()
        base.H = base.W = 6
        label, geometry = torch.randn(36, 4, 128), torch.randn(36, 4, 31)
        mask = WindowAttention.gen_window_attn_mask((6, 6, 4), 'cpu')[None]
        counts = []
        for coding in ('continuous', 'ternary'):
            reduced = ReducedHeadSwinNMP(copy.deepcopy(base), (1, 3), coding)
            scores, ffn_inputs, projection_inputs = [], [], []
            hooks = [
                reduced.attn.attn_drop.register_forward_hook(lambda _, args, __: scores.append(args[0])),
                reduced.mlp.register_forward_pre_hook(lambda _, args: ffn_inputs.append(args[0].shape)),
                reduced.proj.register_forward_pre_hook(lambda _, args: projection_inputs.append(args[0].shape)),
            ]
            try:
                with torch.inference_mode(), FlopCounterMode(display=False) as counter:
                    result = reduced(label, geometry, mask)
                counts.append(counter.get_total_flops())
                with torch.inference_mode(), patch('nmrf_head_compression.F.linear', wraps=F.linear) as linear:
                    reduced(label, geometry, mask)
                projections = [call.args[1].shape for call in linear.call_args_list]
                if coding == 'continuous':
                    self.assertIn(torch.Size([192, 159]), projections)
                else:
                    self.assertIn(torch.Size([192, 128]), projections)
                    self.assertIn(torch.Size([192, 31]), projections)
            finally:
                for hook in hooks:
                    hook.remove()
            self.assertEqual(tuple(result.shape), (36, 4, 128))
            self.assertEqual(tuple(reduced.qkv.weight.shape), (192, 159))
            self.assertEqual(tuple(reduced.attn.relative_position_enc_table.shape), (121, 192))
            self.assertEqual(reduced.head_dim, 32)
            self.assertEqual(reduced.attn.scale, 32**-.5)
            self.assertEqual(scores[0].shape, (1, 2, 144, 144))
            self.assertTrue(torch.equal(scores[0] > 0, mask.isfinite()[:, None].expand_as(scores[0])))
            self.assertEqual(ffn_inputs[0], torch.Size([36, 4, 128]))
            self.assertEqual(projection_inputs[0][-1], 64)
        with torch.inference_mode(), FlopCounterMode(display=False) as counter:
            base(label, geometry, mask)
        dense = counter.get_total_flops()
        self.assertEqual(dense, 86_593_536)
        self.assertEqual(counts, [62_171_136, 62_171_136])
        self.assertEqual(138_242_580_480-5*54*(dense-counts[0]), 131_648_532_480)

    def test_checkpoint_roundtrip_and_wrong_format_rejected(self):
        pairs = [(1, 2), (1, 3), (1, 3), (1, 2), (0, 2)]
        model, _, _ = build_native_nmrf(81)
        projection = [dict(weight=torch.randn(128, 64)*.02, bias=torch.randn(128)*.02) for _ in pairs]
        metadata = install_head_compression(model, pairs, coding='ternary',
                                            projections=projection, projection_sha256='a'*64)
        state = dict(format=FORMAT, architecture=metadata, model=model.state_dict())
        file = io.BytesIO()
        torch.save(state, file)
        file.seek(0)
        checkpoint = torch.load(file, map_location='cpu', weights_only=True)
        restored, _, _ = build_native_nmrf(1234)
        with self.assertRaisesRegex(ValueError, 'head-compression checkpoint'):
            restore_head_compression(restored, dict(checkpoint, format='nmrf-matched-finetune-v1'))
        self.assertEqual(restore_head_compression(restored, checkpoint), metadata)
        self.assertEqual([layer.nmp.retained_heads for layer in restored.parent.inference.layers], pairs)
        label, geometry = torch.randn(36, 4, 128), torch.randn(36, 4, 31)
        mask = WindowAttention.gen_window_attn_mask((6, 6, 4), 'cpu')[None]
        outputs = []
        for arm in (model, restored):
            block = arm.eval().parent.inference.layers[0].nmp
            block.H = block.W = 6
            with torch.no_grad():
                outputs.append(block(label, geometry, mask))
        self.assertTrue(torch.equal(*outputs))


if __name__ == '__main__':
    unittest.main()
