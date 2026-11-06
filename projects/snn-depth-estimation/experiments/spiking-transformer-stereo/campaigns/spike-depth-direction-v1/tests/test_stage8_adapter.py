"""Three local software checks; no parent model construction or dataset decoding."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import unittest

import torch

CAMP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CAMP / "scripts"))
import stage8_controls as controls
from input_codes import transform_stereo


class Stage8SoftwareChecks(unittest.TestCase):
    def test_frozen_dependency_and_manifest_binding(self):
        binding = controls.bindings()  # source/config bytes only, no model imports
        self.assertEqual(len(binding["historical_source_files"]), 20)
        self.assertEqual(sum("/core/" in p for p in binding["historical_source_files"]), 15)
        self.assertEqual([binding["manifest_projection"]["row_equality"][s]["count"] for s in ("train", "dev")], [960, 60])
        source = ast.parse((CAMP / "scripts/run_minimal_count.py").read_text())
        self.assertFalse(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ("exec", "compile") for n in ast.walk(source)))

    def test_adapter_cpu_boundary_and_unchanged_supervision(self):
        # Execute only the new adapter class with a tiny in-memory decoder stub.
        # The historical model/driver modules are never imported or instantiated.
        left = ((torch.arange(300).reshape(50, 2, 3) % 7) == 0).float()
        right = left.flip(0)
        original = [left.clone(), right.clone()]
        target = torch.full((1, 2, 3), -.5)
        valid = torch.tensor([[True, False, True], [True, True, True]])
        class StubDecodedDataset:
            def __getitem__(self, index):
                return left, right, target, valid, index
        node = next(n for n in ast.parse((CAMP / "scripts/run_minimal_count.py").read_text()).body
                    if isinstance(n, ast.ClassDef) and n.name == "InputDataset")
        ns = {"BASE_DATASET": StubDecodedDataset, "MODE": "raw", "require": controls.require, "transform_stereo": transform_stereo}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "adapter_class_only", "exec"), ns)
        dataset = ns["InputDataset"]()
        raw = dataset[7]
        self.assertTrue(torch.equal(raw[0], left) and torch.equal(raw[1], right))
        self.assertIs(raw[2], target)
        self.assertIs(raw[3], valid)
        self.assertEqual(raw[4], 7)
        ns["MODE"] = "count_only"
        count = dataset[7]
        for view in (0, 1):
            self.assertTrue(torch.equal(count[view].sum(0), original[view].sum(0)))
            self.assertTrue(torch.all((count[view] == 0) | (count[view] == 1)))
            self.assertEqual(count[view].shape, (50, 2, 3))
        self.assertIs(count[2], target)
        self.assertIs(count[3], valid)
        self.assertEqual(count[4], 7)
        self.assertTrue(torch.equal(left, original[0]) and torch.equal(right, original[1]))

    def test_only_registered_historical_recipe_plus_input_code(self):
        for seed in (1234, 2345):
            current = json.loads((CAMP / f"configs/s8-count-only-seed{seed}-10k-v1.json").read_text())
            parent = json.loads((CAMP.parent / f"minimal-lif-v1/configs/minimal-lif-seed{seed}-10k-v1.json").read_text())
            controls.validate_config(current)
            self.assertEqual({k for k in current if current[k] != parent.get(k)}, {"campaign_id", "input_code"})
            for key, wrong in (("input_code", "raw"), ("normalize_residual_current", False), ("updates", 5000), ("seed", 3456)):
                with self.assertRaises(RuntimeError):
                    controls.validate_config(dict(current, **{key: wrong}))


if __name__ == "__main__":
    unittest.main()
