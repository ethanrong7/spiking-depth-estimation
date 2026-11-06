"""Tiny CPU checks of the changed loader path; no native data/model experiment."""
import importlib.util
from pathlib import Path
import random
import threading
import unittest

import numpy as np
import torch

from prefetch import Prefetch, decode_batch


REPO = next(p for p in Path(__file__).resolve().parents if (p / 'projects/AGENTS.md').exists())
RECIPE_PATH = REPO / 'storage/experiment-runs/snn-depth-estimation/spiking-transformer-stereo/source-recipe-aligned-three-way-v1/scripts/recipe.py'
SPEC = importlib.util.spec_from_file_location('frozen_prefetch_unit_recipe', RECIPE_PATH)
RECIPE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RECIPE)


class TinyDataset:
    def __init__(self):
        self.calls = 0
        self.third_batch = threading.Event()

    def __getitem__(self, index):
        self.calls += 1
        if self.calls == 24:
            self.third_batch.set()
        left = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3) + index
        target = -torch.arange(1, 7, dtype=torch.float32).reshape(1, 2, 3)
        valid = torch.tensor([[True, False, True], [False, True, True]])
        return left, left + 1000, target, valid, index


def rng_state():
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


class Checks(unittest.TestCase):
    def setUp(self):
        self.order = torch.randperm(960, generator=torch.Generator().manual_seed(1234)).tolist()

    def test_exact_fifo_global_offsets_augmentation_and_rng(self):
        before = rng_state()
        seen_flags = set()
        for start in (10000, 15999, 49999):
            stop = min(start + 3, 50000)
            with Prefetch(TinyDataset(), RECIPE, self.order, 1234, start, stop) as stream:
                for step in range(start, stop):
                    expected = decode_batch(TinyDataset(), RECIPE, self.order, 1234, step)
                    actual = stream.get(step)
                    self.assertEqual(actual.global_index, step)
                    self.assertEqual(actual.indices, expected.indices)
                    self.assertEqual(actual.flags, expected.flags)
                    seen_flags.update(actual.flags)
                    for a, b in zip(actual.items, expected.items):
                        for x, y in zip(a[:4], b[:4]):
                            self.assertTrue(torch.equal(x, y))
                            self.assertEqual(x.dtype, y.dtype)
                            self.assertEqual(x.device.type, 'cpu')
                        self.assertEqual(a[4], b[4])
            self.assertFalse(stream.thread.is_alive())
        self.assertEqual(seen_flags, {(False, False), (False, True), (True, False), (True, True)})
        after = rng_state()
        self.assertEqual(before[0], after[0])
        self.assertEqual(before[1][0], after[1][0])
        self.assertTrue(np.array_equal(before[1][1], after[1][1]))
        self.assertEqual(before[1][2:], after[1][2:])
        self.assertTrue(torch.equal(before[2], after[2]))

    def test_bounded_queue_and_close_while_producer_blocked(self):
        dataset = TinyDataset()
        stream = Prefetch(dataset, RECIPE, self.order, 1234, 10000, 50000)
        try:
            self.assertTrue(dataset.third_batch.wait(timeout=5))
            self.assertEqual(stream.queue.maxsize, 2)
            self.assertEqual(stream.queue.qsize(), 2)
        finally:
            stream.close()
        self.assertFalse(stream.thread.is_alive())
        self.assertTrue(stream.queue.empty())

    def test_failure_propagation(self):
        class Broken:
            def __getitem__(self, index):
                raise ValueError('unit decoder failure')
        with Prefetch(Broken(), RECIPE, self.order, 1234, 10000, 10001) as stream:
            with self.assertRaisesRegex(RuntimeError, 'CPU prefetch failed') as caught:
                stream.get(10000)
            self.assertIsInstance(caught.exception.__cause__, ValueError)


if __name__ == '__main__':
    unittest.main()
