"""Tiny CPU tests: custom derivatives, B integration and checkpoint recompute."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP.parents[1]))
from core.attention_encoder_controls import (
    VARIANTS, configure_codes, controlled_code, install, signed_value_and_derivative)
from core.conventional_attention_placement import LinearCodes
from core.epipolar_spiking_stereo import signed_rate
from core.spiking_attention_placement import DescriptorAttention, install as historical_install
from settings import load, run_folder


class ControlsTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(18)

    def test_scalar_forward_and_vjp(self):
        for dtype in (torch.float32, torch.float64):
            x = torch.cat((torch.linspace(-6, 6, 257, dtype=dtype),
                           torch.tensor([-1, 0, .25, .75, 1, 2, 4], dtype=dtype))).requires_grad_()
            incoming = torch.randn_like(x)
            original = signed_rate(x, 4)
            expected_grad, = torch.autograd.grad(original, x, incoming)
            values, derivative = signed_value_and_derivative(x)
            self.assertTrue(torch.equal(values, original))
            torch.testing.assert_close(derivative * incoming, expected_grad)
            for forward, backward in VARIANTS.values():
                y = controlled_code(x, forward, backward)
                self.assertTrue(torch.equal(y, original if forward == 'signed' else x))
                actual, = torch.autograd.grad(y, x, incoming)
                torch.testing.assert_close(actual, expected_grad if backward == 'surrogate' else incoming)
            with self.assertRaises(ValueError):
                controlled_code(x, 'tanh', 'identity')
        with self.assertRaises(TypeError):
            controlled_code(torch.ones(2, dtype=torch.float16), 'linear', 'identity')

    def test_b_outputs_and_checkpoint_gradients(self):
        for dtype in (torch.float32, torch.float64):
            base = DescriptorAttention().to(dtype)
            # Open the zero-initialized residual to exercise Q/K/V gradients.
            with torch.no_grad():
                base.output.weight.normal_(std=.02)
            inputs = [torch.randn(1, 256, 2, 7, dtype=dtype) for _ in range(2)]
            outputs, gradients = {}, {}
            for variant in VARIANTS:
                direct, checked = copy.deepcopy(base), copy.deepcopy(base)
                for model in (direct, checked):
                    configure_codes(model.codes, variant)
                pairs = []
                for model, checkpointed in ((direct, False), (checked, True)):
                    x = [v.clone().requires_grad_() for v in inputs]
                    out = model(*x) if checkpointed else model.compute(*x)
                    sum(v.square().sum() for v in out).backward()
                    pairs.append((out, [p.grad for p in model.parameters()], [v.grad for v in x]))
                for group in range(3):
                    for actual, expected in zip(pairs[0][group], pairs[1][group]):
                        torch.testing.assert_close(actual, expected)
                for name in ('q', 'k', 'v'):
                    gradient = getattr(direct.codes, name).weight.grad
                    self.assertTrue(torch.isfinite(gradient).all() and gradient.norm() > 0)
                outputs[variant] = pairs[0][0]
                gradients[variant] = direct.codes.q.weight.grad
                # Diagonals reproduce the actual native/linear module, not a proxy.
                if variant in ('linear-identity', 'signed-surrogate'):
                    reference = copy.deepcopy(base)
                    if variant == 'linear-identity':reference.codes.__class__ = LinearCodes
                    out = reference.compute(*inputs)
                    sum(v.square().sum() for v in out).backward()
                    for actual, expected in zip(pairs[0][0], out):
                        self.assertTrue(torch.equal(actual, expected))
                    for actual, expected in zip(direct.parameters(), reference.parameters()):
                        self.assertTrue(torch.equal(actual.grad, expected.grad))
            for forward in ('linear', 'signed'):
                for x, y in zip(outputs[forward + '-identity'], outputs[forward + '-surrogate']):
                    self.assertTrue(torch.equal(x, y))
                self.assertFalse(torch.allclose(gradients[forward + '-identity'], gradients[forward + '-surrogate']))

    def test_install_state_and_hook(self):
        class Pair(torch.nn.Module):
            def forward(self, left, right):return left, right
        class Shell(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fnet = Pair()
        for seed in (1234, 2345):
            states = []
            for variant in VARIANTS:
                before = torch.get_rng_state().clone()
                model = install(Shell(), 'B', variant=variant, seed=seed)
                self.assertTrue(torch.equal(before, torch.get_rng_state()))
                self.assertEqual(sum(p.numel() for p in model.parameters()), 9312)
                x = [torch.randn(1, 256, 1, 3) for _ in range(2)]
                for a, b in zip(model.fnet(*x), x):self.assertTrue(torch.equal(a, b))
                states.append(model.state_dict())
            for state in states[1:]:
                self.assertEqual(state.keys(), states[0].keys())
                for key in state:self.assertTrue(torch.equal(state[key], states[0][key]))
            if seed == 1234:
                reference = historical_install(Shell(), 'B').state_dict()
                for key in reference:self.assertTrue(torch.equal(reference[key], states[0][key]))
            else:
                self.assertFalse(torch.equal(first_seed['placement.input.weight'], states[0]['placement.input.weight']))
            first_seed = states[0]

    def test_seed_stream_and_resume_addressing(self):
        contract = load('b_test_contract', EXP.parent / 'spiking-attention-placement-phase2/protocol.py')
        import common
        class Dataset:
            def __getitem__(self, index):
                tensor = torch.arange(12).reshape(3, 4) + index
                return tensor, tensor, tensor, tensor.bool()
        dataset, order = Dataset(), list(range(16))
        old = common.decode(dataset, order, 5)
        prefetch = contract.Prefetch(dataset, order, 5, 6, 1234)
        try:new = prefetch.get(5)
        finally:prefetch.close()
        self.assertEqual(old['indices'], new['indices'])
        self.assertEqual(old['flags'], new['flags'])
        for xs, ys in zip(old['items'], new['items']):
            for x, y in zip(xs, ys):self.assertTrue(torch.equal(x, y))
        full = contract.Prefetch(dataset, order, 4, 6, 2345)
        resumed = contract.Prefetch(dataset, order, 5, 6, 2345)
        try:
            full.get(4)
            a, b = full.get(5), resumed.get(5)
        finally:
            full.close();resumed.close()
        self.assertEqual(a['flags'], b['flags'])
        self.assertNotEqual(a['flags'], new['flags'])
        for xs, ys in zip(a['items'], b['items']):
            for x, y in zip(xs, ys):self.assertTrue(torch.equal(x, y))

    def test_matrix_and_isolated_paths(self):
        matrix = json.loads((EXP / 'configs/matrix.json').read_text())
        paths = [run_folder('/runs', r['seed'], r['variant']) for r in matrix['runs']]
        self.assertEqual(len(paths), len(set(paths)))
        self.assertEqual(sum(r['stage'] == 'primary' for r in matrix['runs']), 4)

    def test_runner_forwards_identity_and_resume(self):
        runner = load('b_test_runner', EXP / 'run_on_gpu.py')
        captured = {}
        def legacy_main(**kwargs):
            import os
            captured.update(kwargs, argv=list(sys.argv), root=os.environ['STEREO_RUN_ROOT'])
        with patch.dict('os.environ', {'STEREO_RUN_ROOT': '/configured/runs'}), \
             patch.object(sys, 'argv', ['runner', '--seed', '2345', '--variant',
                                       'linear-surrogate', '--resume']), \
             patch.object(runner, 'load', return_value=SimpleNamespace(main=legacy_main)):
            runner.main()
        self.assertEqual(captured['root'], '/configured/runs/seed-2345/linear-surrogate')
        self.assertEqual(captured['argv'][1:], ['--arm', 'B', '--resume'])
        self.assertEqual(captured['worker_path'], EXP / 'worker.py')
        self.assertEqual(captured['worker_args'], ('--seed', '2345', '--variant', 'linear-surrogate',
            '--input-mode', 'fast', '--preload-gib', '12.0', '--input-workers', '2', '--prefetch-depth', '2'))

    def test_worker_forwards_seed_and_rules(self):
        worker = load('b_test_worker_adapter', EXP / 'worker.py')
        captured = {}
        def matched_main(**kwargs):
            captured.update(kwargs, argv=list(sys.argv))
            return 75
        fake_prefetch = lambda *args, **kwargs: None
        with patch.object(sys, 'argv', ['worker', '--seed', '2345', '--variant',
                                       'signed-identity', '--arm', 'B', '--preflight']), \
             patch.object(worker, 'load', side_effect=[SimpleNamespace(Prefetch=fake_prefetch),
                                                      SimpleNamespace(main=matched_main)]):
            self.assertEqual(worker.main(), 75)
        self.assertEqual(captured['seed'], 2345)
        self.assertEqual(captured['argv'][1:], ['--arm', 'B', '--preflight'])
        self.assertEqual(captured['config_extension']['forward_rule'], 'signed')
        self.assertEqual(captured['config_extension']['backward_rule'], 'identity')
        self.assertEqual(captured['config_extension']['placement_seed'], 3128)
        self.assertEqual(captured['prefetch_factory'].keywords['seed'], 2345)


if __name__ == '__main__':
    unittest.main()
