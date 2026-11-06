"""Tiny scalar CPU check of the new50k schedule; no model/data experiment."""
import unittest
import warnings

import torch

from fresh import optimizer, optimizer_steps


class ScheduleChecks(unittest.TestCase):
    def test_fresh_start_milestones_and_saved_clock(self):
        model = torch.nn.Linear(1, 1, bias=False)
        opt, schedule = optimizer(model)
        self.assertEqual(optimizer_steps(opt), [0, 0])
        self.assertEqual(schedule.last_epoch, 0)
        self.assertEqual(opt.param_groups[0]['lr'], 8e-6)
        for local in (0, 1, 499, 500, 501, 49999, 50000):
            expected = (8e-6 + (2e-4 - 8e-6) * local / 500 if local <= 500
                        else 2e-4 + (8e-10 - 2e-4) * (local - 500) / 49599)
            schedule.last_epoch = local
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                actual = schedule.get_lr()[0]
            self.assertAlmostEqual(actual, expected, places=15)
        schedule.last_epoch = 0
        model(torch.ones(1, 1)).sum().backward()
        opt.step()
        schedule.step()
        self.assertEqual(optimizer_steps(opt), [1, 1])
        self.assertEqual(schedule.last_epoch, 1)
        self.assertEqual(schedule.state_dict()['total_steps'], 50100)


if __name__ == '__main__':
    unittest.main()
