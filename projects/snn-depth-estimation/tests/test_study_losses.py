"""Tiny support/units checks for registered disparity objectives."""
import unittest
import torch
from core.study_losses import disparity_loss


class StudyLossTest(unittest.TestCase):
    def test_final_preserves_unmatchable_and_aux_native_pixel_units(self):
        gt = torch.full((1, 1, 16, 16), 4.)
        valid = torch.ones_like(gt, dtype=torch.bool)
        prediction = torch.full_like(gt, 4., requires_grad=True)
        aux = torch.full((1, 1, 4, 4), 4., requires_grad=True)
        output = {'disparity': prediction, 'aux_disparities_native': {4: aux}}
        loss, terms = disparity_loss(output, gt, valid, 16)
        self.assertEqual(float(loss.detach()), 0)
        self.assertEqual(set(terms), {'final_left', 'aux_left_4'})
        changed = gt.clone(); changed[..., 0] = 8.
        loss, _ = disparity_loss(output, changed, valid, 16)
        loss.backward()
        self.assertGreater(float(prediction.grad[..., 0].abs().sum()), 0)
        self.assertEqual(float(aux.grad[..., 0].abs().sum()), 0)

    def test_padding_invalid_and_absent_occlusion_not_supervised(self):
        gt = torch.ones(1, 1, 15, 15)
        pred = torch.ones_like(gt, requires_grad=True)
        occ = torch.zeros_like(gt, requires_grad=True)
        output = {'disparity': pred, 'occlusion_logits': occ,
                  'aux_disparities_native': {16: torch.zeros(1, 1, 1, 1, requires_grad=True)}}
        loss, terms = disparity_loss(output, gt, torch.ones_like(gt, dtype=torch.bool), 256)
        loss.backward()
        self.assertNotIn('occlusion', terms)
        self.assertIsNone(occ.grad)


if __name__ == '__main__':
    unittest.main()
