"""Retain the tiny inspired-control smoke checks; not faithful SSDEFormer validation."""
from pathlib import Path
import sys
import unittest

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.ssdeformer_inspired import SSDEFormerInspired, local_disparity_readout


class InspiredControlTest(unittest.TestCase):
    def test_readout_sign_scale_and_border(self):
        logits = torch.zeros(1, 1, 6, 6)
        logits[..., 4, 2] = 8
        disparity = local_disparity_readout(logits)
        torch.testing.assert_close(disparity[..., 4], torch.tensor([[[2.]]]))
        torch.testing.assert_close(4 * disparity[..., 4], torch.tensor([[[8.]]]))
        self.assertEqual(disparity[..., 0].item(), 0)
        self.assertTrue(torch.all(disparity >= 0))

    def test_shape_matching_gradients_final_bin_and_inference_agreement(self):
        torch.set_num_threads(1)
        torch.manual_seed(23)
        model = SSDEFormerInspired()
        left = torch.randint(0, 2, (1, 50, 16, 32)).float().requires_grad_()
        right = torch.randint(0, 2, left.shape).float().requires_grad_()
        predictions, spikes, voltages = model(left, right)
        self.assertEqual(predictions[0].shape, (1, 1, 16, 32))
        self.assertEqual((spikes, voltages), ([], []))
        self.assertTrue(torch.isfinite(predictions[0]).all())
        self.assertTrue(torch.all(predictions[0] <= 0))
        F.smooth_l1_loss(predictions[0], torch.full_like(predictions[0], -3)).backward()
        for parameter in (model.match_left.weight, model.match_right.weight,
                          model.layers[0].attention.in_proj_weight):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.norm().item(), 0)
        self.assertGreater(left.grad[:, -1].abs().sum().item(), 0)
        self.assertGreater(right.grad[:, -1].abs().sum().item(), 0)
        model.eval()
        _, inference, _ = model(left, right, test_mode=True)
        torch.testing.assert_close(predictions[0], inference)


if __name__ == "__main__":
    unittest.main()
