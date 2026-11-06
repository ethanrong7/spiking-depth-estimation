"""Tiny counterfactual wiring tests; no real inputs or model execution."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "experiments/spiking-transformer-stereo/scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("correspondence_probe", SCRIPTS / "probe_refinements.py")
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


class CounterfactualTest(unittest.TestCase):
    def rows(self):
        return [dict(scene=f"scene_{i // 2:04d}", frame=f"{i % 2:04d}", spike_r=f"right-{i}.dat")
                for i in range(60)]

    def test_permutation_preserves_marginal_and_changes_scene(self):
        rows = self.rows()
        donors = PROBE.wrong_right_permutation(rows)
        self.assertEqual(sorted(donors), list(range(60)))
        self.assertTrue(all(rows[i]["scene"] != rows[j]["scene"] for i, j in enumerate(donors)))
        self.assertEqual(donors[0], 2)
        self.assertEqual(donors[-1], 1)
        with self.assertRaises(ValueError):
            PROBE.wrong_right_permutation(rows[:4])

    def test_only_right_input_changes(self):
        class Aligned:
            root = Path("/unused")
            def __init__(self, rows):
                self.rows = rows
            def __getitem__(self, index):
                return (torch.full((50, 2, 3), float(index)), torch.zeros(50, 2, 3),
                        torch.full((1, 2, 3), float(index + 100)), torch.ones(2, 3, dtype=torch.bool), index)
        aligned = Aligned(self.rows())
        distorted = PROBE.WrongRightDataset(aligned)
        def fake_decode(path):
            index = int(Path(path).stem.split("-")[1])
            return np.full((50, 2, 3), index, dtype=float)
        with patch.object(PROBE.runner, "get_block_spikes", side_effect=fake_decode):
            for index in (0, 59):
                before, after = aligned[index], distorted[index]
                for field in (0, 2, 3):
                    torch.testing.assert_close(before[field], after[field])
                self.assertEqual(before[4], after[4])
                self.assertTrue(torch.all(after[1] == (index + 2) % 60))

    def test_branch_hooks_select_actual_residual_current(self):
        from core.spiking_attention_update import LocalAttentionRefinement
        torch.set_num_threads(1)
        torch.manual_seed(19)
        hidden, context, motion = (torch.randn(1, 16, 2, 3),
            [torch.randn(1, 16, 2, 3) for _ in range(3)], torch.randn(1, 8, 2, 3))
        for normalize in (False, True):
            cell = LocalAttentionRefinement(16, 8, normalize_residual_current=normalize)
            with torch.no_grad():
                cell.input_proj.bias.fill_(20)
            records, handles = PROBE.register_branch_hooks([cell])
            cell(hidden, context, motion)
            for handle in handles:
                handle.remove()
            record = records["scale_0"]
            self.assertEqual(len(record["residual_current_rms"]), 1)
            expected = "normalized_projection_rms" if normalize else "raw_projection_rms"
            self.assertEqual(record["residual_current_rms"], record[expected])
            self.assertGreater(record["raw_projection_rms"][0], 10)
            self.assertLess(record["normalized_projection_rms"][0], 2)


if __name__ == "__main__":
    unittest.main()
