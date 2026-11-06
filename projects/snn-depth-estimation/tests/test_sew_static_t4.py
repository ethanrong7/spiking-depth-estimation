"""Tiny CPU-only encoder tests: 50 input channels, four neuron steps."""
import importlib.util
import os
from pathlib import Path
import sys
import unittest

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from core.sew_encoder import SpikingBasicEncoder, SpikingMultiBasicEncoder


def tensors(output):
    if torch.is_tensor(output):
        return [output]
    return [tensor for group in output for tensor in group]


class StaticT4Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(1234)

    def test_both_encoders_have_four_updates_and_gradients_to_all_input_channels(self):
        for kind in (SpikingBasicEncoder, SpikingMultiBasicEncoder):
            output_dim = 8 if kind is SpikingBasicEncoder else [[8, 8, 8], [8, 8, 8]]
            model = kind(output_dim=output_dim, norm_fn='none', downsample=2,
                         timesteps=4, input_mode='static_channels', input_bins=50)
            steps = []
            def capture(_module, _inputs, output):
                output[0].retain_grad()
                steps.append(output[0])
            handle = model.lif1.register_forward_hook(capture)
            x = torch.randn(1, 50, 16, 16, requires_grad=True)
            result = model(x)
            self.assertEqual(len(steps), 4)
            self.assertEqual(model.conv1.in_channels, 50)
            sum(t.square().mean() for t in tensors(result)).backward()
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertTrue((x.grad.abs().sum(dim=(0, 2, 3)) > 0).all())
            self.assertTrue(all(s.grad is not None and s.grad.abs().sum() > 0 for s in steps))
            handle.remove()

    def test_default_sensor_time_is_exactly_equal_to_frozen_reference(self):
        reference = os.environ.get('SEW_T50_REFERENCE')
        if not reference:
            self.skipTest('set SEW_T50_REFERENCE to the preserved T50 encoder file')
        spec = importlib.util.spec_from_file_location('frozen_sew_reference', reference)
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        for name in ('SpikingBasicEncoder', 'SpikingMultiBasicEncoder'):
            output_dim = 8 if name == 'SpikingBasicEncoder' else [[8, 8, 8], [8, 8, 8]]
            config = dict(output_dim=output_dim, norm_fn='none', downsample=2, timesteps=2)
            first = getattr(old, name)(**config)
            second = globals()[name](**config)
            second.load_state_dict(first.state_dict())
            x = torch.randn(1, 2, 16, 16, requires_grad=True)
            y = x.detach().clone().requires_grad_(True)
            a, b = tensors(first(x)), tensors(second(y))
            self.assertTrue(all(torch.equal(p, q) for p, q in zip(a, b)))
            sum(t.square().mean() for t in a).backward()
            sum(t.square().mean() for t in b).backward()
            self.assertTrue(torch.equal(x.grad, y.grad))
            for p, q in zip(first.parameters(), second.parameters()):
                if p.grad is not None:
                    self.assertTrue(torch.equal(p.grad, q.grad))


if __name__ == '__main__':
    unittest.main()
