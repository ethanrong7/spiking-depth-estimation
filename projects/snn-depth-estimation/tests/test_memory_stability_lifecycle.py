import importlib.util
import tempfile
import unittest
from pathlib import Path

import torch


PROJECT = Path(__file__).resolve().parents[1]
HELPER = (
    PROJECT
    / "experiments"
    / "sew-t50-memory-stability"
    / "scripts"
    / "training_lifecycle.py"
)
SPEC = importlib.util.spec_from_file_location("memory_stability_lifecycle", HELPER)
LIFECYCLE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(LIFECYCLE)


class RepeatingSamplerTest(unittest.TestCase):
    def test_saved_cursor_ignores_prefetched_yields(self):
        sampler = LIFECYCLE.DeterministicRepeatingBatchSampler(
            [7, 3, 9, 1], batch_size=2
        )
        iterator = iter(sampler)
        self.assertEqual(next(iterator), [7, 3])
        sampler.mark_consumed(2)
        self.assertEqual(next(iterator), [9, 1])
        self.assertEqual(next(iterator), [7, 3])  # yielded ahead, not consumed

        state = sampler.state_dict()
        self.assertEqual(state["consumed_samples"], 2)
        self.assertEqual(state["cycle"], 0)
        self.assertEqual(state["position_in_cycle"], 2)

        resumed = LIFECYCLE.DeterministicRepeatingBatchSampler(
            [7, 3, 9, 1], batch_size=2
        )
        resumed.load_state_dict(state)
        self.assertEqual(next(iter(resumed)), [9, 1])

    def test_update25_cursor_for_fixed_manifest(self):
        sampler = LIFECYCLE.DeterministicRepeatingBatchSampler(
            list(range(16)), batch_size=2
        )
        for _ in range(25 * 4):
            sampler.mark_consumed(2)
        state = sampler.state_dict()
        self.assertEqual(state["consumed_samples"], 200)
        self.assertEqual(state["cycle"], 12)
        self.assertEqual(state["position_in_cycle"], 8)


class CounterAndCheckpointTest(unittest.TestCase):
    @staticmethod
    def make_training_objects():
        parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
        optimizer = torch.optim.AdamW([parameter], lr=2e-4, eps=1e-8, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=2e-4,
            total_steps=320100,
            pct_start=0.01,
            anneal_strategy="linear",
            cycle_momentum=False,
        )
        return parameter, optimizer, scheduler

    def test_late_counter_position_and_two_real_steps(self):
        torch.set_num_threads(1)
        parameter, optimizer, scheduler = self.make_training_objects()
        parameter.grad = torch.tensor([0.25, -0.5])
        optimizer.step()
        scheduler.step()
        transition = LIFECYCLE.position_logical_counters(
            optimizer, scheduler, completed_logical_updates=319998
        )
        self.assertEqual(
            transition["after"]["optimizer"]["step_min"], 319998
        )
        self.assertEqual(scheduler.last_epoch, 319998)

        for expected in (319999, 320000):
            optimizer.zero_grad(set_to_none=True)
            parameter.grad = torch.tensor([0.25, -0.5])
            optimizer.step()
            scheduler.step()
            report = LIFECYCLE.optimizer_step_report(optimizer)
            self.assertEqual(report["step_min"], expected)
            self.assertEqual(report["step_max"], expected)
            self.assertEqual(scheduler.last_epoch, expected)

    def test_tiny_checkpoint_state_digests_restore_exactly(self):
        torch.set_num_threads(1)
        parameter, optimizer, scheduler = self.make_training_objects()
        parameter.grad = torch.tensor([0.125, -0.75])
        optimizer.step()
        scheduler.step()
        sampler = LIFECYCLE.DeterministicRepeatingBatchSampler(
            list(range(16)), batch_size=2, consumed_samples=200
        )
        payload = {
            "parameter": parameter.detach().clone(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "sampler": sampler.state_dict(),
        }
        expected = LIFECYCLE.state_digest(payload)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.pt"
            torch.save(payload, path)
            loaded = torch.load(path, map_location="cpu", weights_only=False)
        self.assertEqual(LIFECYCLE.state_digest(loaded), expected)

        parameter2, optimizer2, scheduler2 = self.make_training_objects()
        parameter2.data.copy_(loaded["parameter"])
        optimizer2.load_state_dict(loaded["optimizer"])
        scheduler2.load_state_dict(loaded["scheduler"])
        sampler2 = LIFECYCLE.DeterministicRepeatingBatchSampler(
            list(range(16)), batch_size=2
        )
        sampler2.load_state_dict(loaded["sampler"])
        self.assertEqual(
            LIFECYCLE.state_digest(optimizer2.state_dict()),
            LIFECYCLE.state_digest(optimizer.state_dict()),
        )
        self.assertEqual(
            LIFECYCLE.state_digest(scheduler2.state_dict()),
            LIFECYCLE.state_digest(scheduler.state_dict()),
        )
        self.assertEqual(sampler2.state_dict(), sampler.state_dict())


if __name__ == "__main__":
    unittest.main()
