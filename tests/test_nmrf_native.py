"""Checks for the intentional sensor/domain adapter, not NMRF internals."""
import unittest

import torch
from torch import nn

from nmrf_runtime import SpikeInputNMRF, prepare_batch, weighted_loss


class CaptureParent(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)

    def forward(self, sample):
        return {key: 2*(value/255)-1 for key, value in sample.items()}


class NativeNMRFTests(unittest.TestCase):
    def test_all_bins_preserved_and_normalized_once(self):
        model = SpikeInputNMRF(CaptureParent()).eval()
        left = torch.zeros(1, 50, 8, 8)
        left[:, 37] = 1
        result = model(left, 1-left)
        self.assertEqual(model.parent.backbone.conv1.in_channels, 50)
        torch.testing.assert_close(result['img1'], 2*left-1)
        torch.testing.assert_close(result['img2'], 1-2*left)
        with self.assertRaises(ValueError):
            model(left[:, :3], left[:, :3])

    def test_disparity_sign_and_padding_validity(self):
        left = torch.zeros(1, 50, 9, 10)
        flow = torch.full((1, 1, 9, 10), -12.5)
        valid = torch.ones(1, 9, 10, dtype=torch.bool)
        l, r, disparity, mask, _ = prepare_batch(
            (left, left+1, flow, valid, [0]), 'cpu')
        self.assertEqual(l.shape, (1, 50, 16, 16))
        self.assertTrue(torch.all(disparity[:, :9, :10] == 12.5))
        self.assertEqual(int(mask.sum()), 90)
        self.assertFalse(mask[:, 9:].any())
        self.assertFalse(mask[:, :, 10:].any())
        self.assertTrue(torch.all(r == 1))
        self.assertTrue(torch.all(flow == -12.5))

    def test_loss_cannot_silently_drop_proposal_supervision(self):
        class Criterion:
            max_disp = 192
            weight_dict = {'proposal_disp': 1}

            def __call__(self, outputs, targets, log=False):
                return {'loss_prop': torch.tensor(2., requires_grad=True)}

        criterion = Criterion()
        target = torch.ones(1, 8, 8)
        valid = torch.ones_like(target, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, 'Unweighted'):
            weighted_loss(criterion, {}, target, valid)
        criterion.weight_dict = {'loss_prop': 1}
        loss, _ = weighted_loss(criterion, {}, target, valid)
        self.assertEqual(float(loss.detach()), 2.)


if __name__ == '__main__':
    unittest.main()
