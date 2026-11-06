"""Tiny CPU activation contracts; no training, checkpoints or data."""
import sys
from pathlib import Path
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.epipolar_spiking_stereo import signed_rate
from forward_backward_controls import controlled_code, signed_value_and_derivative


class ForwardBackwardTest(unittest.TestCase):
    def currents(self, dtype):
        boundaries = torch.tensor([-2., -1., 0., 1., 2.], dtype=dtype)
        neighbors = [torch.nextafter(boundaries, torch.full_like(boundaries, sign * float('inf')))
                     for sign in (-1, 1)]
        return torch.cat([torch.linspace(-3, 3, 121, dtype=dtype), boundaries, *neighbors])

    def test_all_cells_forward_and_first_order_gradient(self):
        for dtype, tolerance in [(torch.float64, 2e-14), (torch.float32, 2e-6)]:
            values = self.currents(dtype)
            upstream = torch.linspace(-2, 3, values.numel(), dtype=dtype)
            expected_values, expected_gradients = {}, {}
            for rule in ('signed', 'tanh'):
                x = values.clone().requires_grad_()
                y = signed_rate(x, 4) if rule == 'signed' else x.tanh()
                expected_values[rule] = y.detach()
                expected_gradients[rule] = torch.autograd.grad(y, x, upstream)[0]
            for forward in ('signed', 'tanh'):
                for backward in ('signed', 'tanh'):
                    with self.subTest(dtype=dtype, forward=forward, backward=backward):
                        x = values.clone().requires_grad_()
                        y = controlled_code(x, forward, backward)
                        gradient = torch.autograd.grad(y, x, upstream)[0]
                        self.assertTrue(torch.equal(y, expected_values[forward]))
                        torch.testing.assert_close(gradient, expected_gradients[backward], atol=tolerance, rtol=tolerance)

    def test_signed_threshold_polarity_and_soft_reset(self):
        values = torch.tensor([-1., 0., 1.], dtype=torch.float64)
        output, derivative = signed_value_and_derivative(values)
        self.assertTrue(torch.equal(output, torch.tensor([-1., 0., 1.], dtype=torch.float64)))
        self.assertTrue(torch.all(derivative > 0))
        self.assertEqual(derivative[0].item(), derivative[2].item())
        # At zero, h=.5 each step. Soft-reset derivative D=(1-h)A is
        # retained; a detached reset instead would give D=A and differs.
        d, total = 0., 0.
        for _ in range(4):
            a = .9 * d + 1
            total += .5 * a
            d = .5 * a
        self.assertAlmostEqual(derivative[1].item(), total / 4, places=14)
        self.assertAlmostEqual(derivative[1].item(), .730765625, places=14)
        self.assertNotAlmostEqual(derivative[1].item(), .5 * (1 + 1.9 + 2.71 + 3.439) / 4)

    def test_crossed_controls_are_distinct(self):
        x = torch.tensor([.2, .7, 1.4], dtype=torch.float64, requires_grad=True)
        sc = controlled_code(x, 'signed', 'tanh')
        cs = controlled_code(x, 'tanh', 'signed')
        self.assertFalse(torch.equal(sc, cs))
        gs = torch.autograd.grad(cs.sum(), x)[0]
        gc = torch.autograd.grad(sc.sum(), x)[0]
        self.assertFalse(torch.allclose(gs, gc))

    def test_higher_order_not_supported_and_invalid_rule_rejected(self):
        x = torch.tensor([.4], requires_grad=True)
        first = torch.autograd.grad(controlled_code(x).sum(), x, create_graph=True)[0]
        with self.assertRaises(RuntimeError):
            torch.autograd.grad(first.sum(), x)
        with self.assertRaises(ValueError):
            controlled_code(x, 'relu', 'signed')


if __name__ == '__main__':
    unittest.main()
