"""Small CPU checks for changed v2 learning paths, not experimental results."""
import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.s2m2_spiking_v2 import SpikingS2M2V2, global_correspondence, s2m2_v2_loss
from core.s2m2_spiking import SpikeProjection, SpikeHidden
torch.set_num_threads(2)

class V2Tests(unittest.TestCase):
    def test_global_probability_gradient(self):
        p = torch.full((1, 1, 100), .005, requires_grad=True)
        position, _ = global_correspondence(p)
        (position - 80).abs().sum().backward()
        self.assertLess(p.grad[0, 0, 80].item(), 0)

    def test_scope_backward_and_out_of_frame_mask(self):
        torch.manual_seed(1234)
        m = SpikingS2M2V2()
        for root in (m.feat_pyramid, m.global_refiner, m.refiner):
            self.assertFalse(any(isinstance(x, (SpikeProjection, SpikeHidden)) for x in root.modules()))
        self.assertTrue(any(isinstance(x, SpikeProjection) for x in m.transformer.modules()))
        left = torch.rand(1, 50, 32, 64)
        output = m(left, torch.rand_like(left))
        target = torch.full((1, 1, 32, 64), -4.)
        valid = torch.ones(1, 32, 64, dtype=torch.bool)
        loss, parts = s2m2_v2_loss(output, target, valid)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for weight in (m.disp_init.q.linear.weight, m.disp_init.k.linear.weight):
            self.assertTrue(torch.isfinite(weight.grad).all())
            self.assertGreater(weight.grad.norm().item(), 0)
        # Entirely unrepresentable GT contributes no initial matching penalty.
        _, outside = s2m2_v2_loss(output, torch.full_like(target, -100.), valid)
        self.assertEqual(outside['initial_l1'].item(), 0)
        # Zero-initialized residual refiner starts exactly on the initial field.
        self.assertTrue(torch.allclose(output['global_flow'], output['initial_flow']))

if __name__ == '__main__': unittest.main()
