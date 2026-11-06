"""Tiny CPU regression checks for the full trainer's interruption semantics."""
import importlib.util
from itertools import islice
from pathlib import Path
import tempfile
import unittest

import torch

HELPER = Path(__file__).resolve().parents[1] / 'experiments/sew-t50-training/scripts/training_state.py'
SPEC = importlib.util.spec_from_file_location('t50_training_state', HELPER)
STATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STATE)


class ResumeTests(unittest.TestCase):
    def test_segment_budget_does_not_restart_on_exec_and_has_a_hard_cap(self):
        first = STATE.segment_deadline(18000, now=100, slurm_end=18050)
        self.assertEqual(first, 18050)
        self.assertEqual(STATE.segment_deadline(18000, now=200, inherited=first), first)
        self.assertEqual(STATE.segment_deadline(18000, now=20000), 38000)
        for bad in (0, -1, 18001):
            with self.assertRaises(ValueError):
                STATE.segment_deadline(bad, now=100)

    def test_prefetch_does_not_advance_committed_sample_order(self):
        sampler = STATE.ShuffledBatchSampler(25, 1234)
        stream = iter(sampler)
        expected = list(islice(stream, 12))
        sampler.commit_update()
        restored = STATE.ShuffledBatchSampler(25, 1234)
        restored.load_state_dict(sampler.state_dict())
        self.assertEqual(list(islice(iter(restored), 8)), expected[4:12])
        # Cross an epoch boundary, dropping only the one incomplete batch-8 tail.
        restored.commit_update()
        restored.commit_update()
        self.assertEqual(restored.consumed, 24)
        self.assertEqual(len(list(islice(iter(restored), 4))), 4)

    @staticmethod
    def objects():
        torch.manual_seed(1234)
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, eps=1e-8, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-4,
            total_steps=300100, pct_start=.01, cycle_momentum=False, anneal_strategy='linear')
        return model, optimizer, scheduler

    @staticmethod
    def update(model, optimizer, scheduler):
        optimizer.zero_grad(set_to_none=True)
        loss = model(torch.tensor([[1., 2.], [-1., 3.]])).square().mean()
        loss.backward()
        lr = optimizer.param_groups[0]['lr']
        optimizer.step()
        scheduler.step()
        return loss.detach().clone(), lr

    def test_resumed_lr_loss_adam_and_parameters_equal_uninterrupted(self):
        model, optimizer, scheduler = self.objects()
        sampler = STATE.ShuffledBatchSampler(25, 1234)
        for _ in range(3):
            self.update(model, optimizer, scheduler)
            sampler.commit_update()
        config = dict(schedule_num_steps=300000, effective_batch_size=8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'latest.pth'
            STATE.save_state(path, model=model, optimizer=optimizer, scheduler=scheduler,
                sampler=sampler, step=3, config=config, binding={'source': 'same'},
                rng={}, best=None, resume_verified=False)
            expected = [self.update(model, optimizer, scheduler) for _ in range(5)]
            resumed, opt2, sch2 = self.objects()
            sam2 = STATE.ShuffledBatchSampler(25, 1234)
            STATE.restore_state(path, model=resumed, optimizer=opt2, scheduler=sch2,
                sampler=sam2, config=config, binding={'source': 'same'})
            actual = [self.update(resumed, opt2, sch2) for _ in range(5)]
            for (loss1, lr1), (loss2, lr2) in zip(expected, actual):
                self.assertEqual(lr1, lr2)
                self.assertTrue(torch.equal(loss1, loss2))
            for p, q in zip(model.parameters(), resumed.parameters()):
                self.assertTrue(torch.equal(p, q))
            self.assertEqual(scheduler.state_dict(), sch2.state_dict())
            for p, q in zip(model.parameters(), resumed.parameters()):
                for key in optimizer.state[p]:
                    self.assertTrue(torch.equal(optimizer.state[p][key], opt2.state[q][key]))
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                STATE.restore_state(path, model=resumed, optimizer=opt2, scheduler=sch2,
                    sampler=sam2, config={**config, 'schedule_num_steps': 10000},
                    binding={'source': 'same'})


if __name__ == '__main__':
    unittest.main()
