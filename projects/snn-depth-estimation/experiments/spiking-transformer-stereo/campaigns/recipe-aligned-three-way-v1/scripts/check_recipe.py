"""Tiny CPU changed-path checks only; no data, model training or benchmarks."""
import json
from pathlib import Path
import unittest
import torch
from recipe import augment, batch_indices, paired_flags, validate_config


class RecipeTests(unittest.TestCase):
    def test_joint_signed_geometry_and_bin_identity(self):
        left = torch.arange(2 * 3 * 5).reshape(2, 3, 5).float()
        right = left + 100
        target = -torch.arange(1, 16).reshape(1, 3, 5).float()
        valid = target[0].remainder(3) != 0
        item = (left, right, target, valid, 7)
        for horizontal, vertical in ((False, False), (True, False), (False, True), (True, True)):
            actual = augment(item, horizontal, vertical)
            dims = ([-1] if horizontal else []) + ([-2] if vertical else [])
            expected = [x.flip(dims) if dims else x for x in item[:4]]
            if horizontal: expected[2] = -expected[2]
            for a, e in zip(actual, expected): self.assertTrue(torch.equal(a, e))
            self.assertEqual(int(actual[3].sum()), int(valid.sum()))
            for original, restored in zip(item, augment(actual, horizontal, vertical)):
                self.assertTrue(torch.equal(original, restored) if torch.is_tensor(original) else original == restored)
        # For a valid correspondence xR=xL+u, horizontal mirror maps u'=-u.
        width = 5; x_left = 3; u = -2; x_right = x_left + u
        self.assertEqual((width-1-x_right)-(width-1-x_left), -u)

    def test_stateless_schedule_order_and_exact_budget(self):
        order = torch.randperm(960, generator=torch.Generator().manual_seed(1234)).tolist()
        self.assertEqual(sorted(order), list(range(960)))
        self.assertEqual(batch_indices(order, 0), batch_indices(order, 120))
        self.assertEqual(batch_indices(order, 9999), order[312:320])
        all_flags = [paired_flags(1234, p) for p in range(960)]
        torch.manual_seed(999); torch.rand(11)
        self.assertEqual(all_flags, [paired_flags(1234, p) for p in range(960)])
        self.assertEqual(10000 * 8, 80000); self.assertEqual(divmod(10000, 120), (83, 40))
        for path in (Path(__file__).parents[1] / 'configs').glob('*seed1234*.json'):
            validate_config(json.loads(path.read_text()))

    def test_unequal_valid_weighted_loss_and_gradient(self):
        # Shared scalar and multiple refinements; deliberately unequal valid counts 3/6.
        x = torch.arange(12, dtype=torch.float64).reshape(4, 1, 1, 3) / 5
        truth = torch.zeros_like(x); valid = torch.ones(4, 1, 3, dtype=torch.bool); valid[0] = False
        def loss(parameter, selection):
            mask = valid[selection, None]
            predictions = [parameter * x[selection] + shift for shift in (.2, .4, .8)]
            return sum((.9 ** (15 / 2)) ** (2-i) * prediction.abs()[mask].mean() for i, prediction in enumerate(predictions))
        full = torch.tensor(.6, dtype=torch.float64, requires_grad=True)
        reference = loss(full, slice(None)); reference.backward()
        micro = torch.tensor(.6, dtype=torch.float64, requires_grad=True); total = 0.
        for selection in (slice(0, 2), slice(2, 4)):
            weighted = loss(micro, selection) * (int(valid[selection].sum()) / int(valid.sum()))
            weighted.backward(); total += float(weighted.detach())
        self.assertAlmostEqual(total, float(reference.detach()), places=12)
        self.assertTrue(torch.allclose(full.grad, micro.grad, atol=1e-12, rtol=1e-12))


if __name__ == '__main__': unittest.main()
