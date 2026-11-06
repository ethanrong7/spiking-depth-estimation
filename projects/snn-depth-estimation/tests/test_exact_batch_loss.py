import sys
import unittest
from pathlib import Path

import torch


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "core"))
sys.path.insert(
    0, str(PROJECT / "experiments" / "sew-t50-feasibility" / "scripts")
)

from exact_batch_loss import ExactBatchLossAccumulator, replay_loss
from loss import sequence_loss


class ExactBatchLossTest(unittest.TestCase):
    BATCH_SIZE = 8
    REFINEMENTS = 16
    SPIKE_SIZE = (3, 5)
    PADDED_SIZE = (4, 8)

    def _inputs(self, seed):
        generator = torch.Generator().manual_seed(seed)
        dtype = torch.double
        batch = self.BATCH_SIZE

        spike_offsets = torch.arange(batch, dtype=dtype).view(batch, 1, 1, 1)
        spikes = [
            (
                torch.randn(batch, 3, 2, 3, dtype=dtype, generator=generator)
                * 0.1
                + spike_offsets * 0.2
                + refinement * 0.01
            ).requires_grad_()
            for refinement in range(self.REFINEMENTS)
        ]
        voltages = [
            torch.randn(batch, 2, 2, 3, dtype=dtype, generator=generator)
            .add(refinement * 0.02)
            .requires_grad_()
            for refinement in range(self.REFINEMENTS)
        ]
        predictions = [
            torch.randn(
                batch,
                1,
                *self.PADDED_SIZE,
                dtype=dtype,
                generator=generator,
            )
            .add(refinement * 0.03)
            .requires_grad_()
            for refinement in range(self.REFINEMENTS)
        ]
        ground_truth = torch.randn(
            batch, 1, *self.SPIKE_SIZE, dtype=dtype, generator=generator
        )
        valid = torch.zeros(batch, *self.SPIKE_SIZE, dtype=dtype)
        for index in range(batch):
            valid[index].view(-1)[: index + 1] = 1.0

        # Exercise both mask stages. The loader-supplied mask has already
        # rejected the 600 disparity, while sequence_loss itself rejects 750.
        ground_truth[7, 0, -1, -1] = -600.0
        valid[7, -1, -1] = 0.0
        ground_truth[6, 0, -1, -1] = -750.0
        valid[6, -1, -1] = 1.0
        return spikes, voltages, predictions, ground_truth, valid

    @staticmethod
    def _leaf_copy(tensors):
        return [value.detach().clone().requires_grad_() for value in tensors]

    def _assert_gradients_match(self, reference, replayed, label):
        self.assertEqual(len(reference), len(replayed))
        for index, (expected, actual) in enumerate(zip(reference, replayed)):
            self.assertIsNotNone(expected.grad, f"{label}[{index}] reference")
            self.assertIsNotNone(actual.grad, f"{label}[{index}] replay")
            self.assertTrue(
                torch.allclose(
                    expected.grad, actual.grad, rtol=2e-12, atol=2e-13
                ),
                f"{label}[{index}] maximum gradient difference: "
                f"{float((expected.grad - actual.grad).abs().max())}",
            )

    def _check_physical_batch(self, physical_batch):
        source = self._inputs(seed=100 + physical_batch)
        (
            source_spikes,
            source_voltages,
            source_predictions,
            ground_truth,
            valid,
        ) = source
        replay_spikes = self._leaf_copy(source_spikes)
        replay_voltages = self._leaf_copy(source_voltages)
        replay_predictions = self._leaf_copy(source_predictions)

        reference_loss, _ = sequence_loss(
            source_spikes,
            source_voltages,
            list(source_predictions),
            ground_truth,
            valid,
            self.SPIKE_SIZE,
        )
        reference_loss.backward()

        accumulator = ExactBatchLossAccumulator()
        with torch.no_grad():
            for start in range(0, self.BATCH_SIZE, physical_batch):
                batch_slice = slice(start, start + physical_batch)
                accumulator.accumulate(
                    [value[batch_slice] for value in replay_spikes],
                    [value[batch_slice] for value in replay_voltages],
                    [value[batch_slice] for value in replay_predictions],
                    ground_truth[batch_slice],
                    valid[batch_slice],
                    self.SPIKE_SIZE,
                )
        stats = accumulator.finalize()

        self.assertFalse(stats.true_loss.requires_grad)
        self.assertTrue(
            torch.allclose(
                reference_loss.detach(), stats.true_loss, rtol=2e-14, atol=2e-15
            ),
            f"source={reference_loss.item()} replay_stats={stats.true_loss.item()}",
        )

        for start in range(0, self.BATCH_SIZE, physical_batch):
            batch_slice = slice(start, start + physical_batch)
            local_loss = replay_loss(
                stats,
                [value[batch_slice] for value in replay_spikes],
                [value[batch_slice] for value in replay_voltages],
                [value[batch_slice] for value in replay_predictions],
                ground_truth[batch_slice],
                valid[batch_slice],
                self.SPIKE_SIZE,
            )
            local_loss.backward()

        self._assert_gradients_match(
            source_predictions, replay_predictions, "prediction"
        )
        self._assert_gradients_match(source_spikes, replay_spikes, "spike")
        self._assert_gradients_match(source_voltages, replay_voltages, "voltage")

        global_rate = torch.stack(replay_spikes, dim=1).mean(dim=(0, 1))
        global_rate_penalty = 1e-4 * ((global_rate - 0.05) ** 2).sum()
        naive_microbatch_penalties = []
        for start in range(0, self.BATCH_SIZE, physical_batch):
            local_spikes = torch.stack(
                [
                    value[start : start + physical_batch]
                    for value in replay_spikes
                ],
                dim=1,
            )
            local_rate = local_spikes.mean(dim=(0, 1))
            naive_microbatch_penalties.append(
                1e-4 * ((local_rate - 0.05) ** 2).sum()
            )
        naive_rate_penalty = torch.stack(naive_microbatch_penalties).mean()
        self.assertGreater(
            float(
                (naive_rate_penalty - global_rate_penalty)
                .detach()
                .abs()
            ),
            1e-6,
        )

    def test_exact_loss_and_gradients_for_b2x4_and_b4x2(self):
        for physical_batch in (2, 4):
            with self.subTest(
                physical_batch=physical_batch,
                accumulation=self.BATCH_SIZE // physical_batch,
            ):
                self._check_physical_batch(physical_batch)


if __name__ == "__main__":
    unittest.main()
