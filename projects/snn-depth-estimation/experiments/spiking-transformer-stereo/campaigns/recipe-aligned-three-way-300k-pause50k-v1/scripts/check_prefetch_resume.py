"""Tiny CPU queue reconstruction test; no native dataset/model experiment."""
from contextlib import ExitStack
import importlib.util
from pathlib import Path
import random
import unittest
import numpy as np
import torch
from prefetch import Prefetch

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'projects/AGENTS.md').exists())
PARENT=ROOT/'storage/experiment-runs/snn-depth-estimation/spiking-transformer-stereo/source-recipe-aligned-three-way-v1/scripts/recipe.py'
spec=importlib.util.spec_from_file_location('queue_resume_frozen_recipe',PARENT);recipe=importlib.util.module_from_spec(spec);spec.loader.exec_module(recipe)


class TinyDataset:
    def __getitem__(self,index):
        x=torch.arange(12,dtype=torch.float32).reshape(2,2,3)+index
        return x,x+10,x[:1],torch.ones((2,3),dtype=torch.bool),index


class Checks(unittest.TestCase):
    def test_reconstruct_at240_without_skip_or_rng_change(self):
        order=torch.randperm(960,generator=torch.Generator().manual_seed(1234)).tolist()
        before=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone());seen=[]
        with ExitStack() as stack:
            stream=stack.enter_context(Prefetch(TinyDataset(),recipe,order,1234,0,480))
            for step in range(240):seen.append(stream.get(step).global_index)
            oldnext=stream.get(240);stream.close();self.assertFalse(stream.thread.is_alive())
            stream=stack.enter_context(Prefetch(TinyDataset(),recipe,order,1234,240,480));pending=stream.get(240)
            self.assertEqual((pending.global_index,pending.indices,pending.flags),(oldnext.global_index,oldnext.indices,oldnext.flags))
            for a,b in zip(oldnext.items,pending.items):
                for x,y in zip(a[:4],b[:4]):self.assertTrue(torch.equal(x,y))
            seen.append(pending.global_index)
            for step in range(241,480):seen.append(stream.get(step).global_index)
        self.assertEqual(seen,list(range(480)))
        self.assertEqual(before[0],random.getstate());after=np.random.get_state();self.assertEqual(before[1][0],after[0]);self.assertTrue(np.array_equal(before[1][1],after[1]));self.assertEqual(before[1][2:],after[2:]);self.assertTrue(torch.equal(before[2],torch.get_rng_state()))


if __name__=='__main__':unittest.main()
