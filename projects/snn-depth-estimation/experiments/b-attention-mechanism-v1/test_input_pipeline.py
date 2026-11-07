"""Tiny CPU input-equivalence tests; no dataset preload or throughput benchmark."""
import copy
import importlib.util
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1])]
from input_pipeline import NativeBytesDataset, RamDataset, InputPipeline, PreparedPrefetch, prepare_batch, read_spike_bytes
from settings import load
from core.utils.reader import get_block_spikes


class TinyDataset:
    def __init__(self, count=16):
        self.rows = [dict(scene=str(i)) for i in range(count)]
        self.calls = 0
        self.values = []
        generator = torch.Generator().manual_seed(135)
        for i in range(count):
            left = torch.randint(0, 2, (5, 4, 8), generator=generator, dtype=torch.uint8)
            right = torch.randint(0, 2, left.shape, generator=generator, dtype=torch.uint8)
            target = torch.randn(1, 4, 8, generator=generator)
            valid = torch.rand(4, 8, generator=generator) > .2
            self.values.append((left, right, target, valid, i))
    def __len__(self):return len(self.rows)
    def __getitem__(self, i):
        self.calls += 1
        return self.values[i]


class PipelineTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_little_endian_decode_matches_original(self):
        shape = (5, 4, 16)
        values = np.random.default_rng(42).integers(0, 2, shape, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'spikes.dat'
            # Extra frames must not enter the returned block.
            path.write_bytes(np.packbits(values.flatten(), bitorder='little').tobytes() + b'\xff'*20)
            expected = get_block_spikes(str(path), block_len=5, spike_height=4, spike_width=16)
            actual = read_spike_bytes(path, shape)
            self.assertTrue(np.array_equal(actual.numpy(), expected))
            self.assertEqual(actual.dtype, torch.uint8)
            path.write_bytes(b'\x01')
            with self.assertRaises(ValueError):read_spike_bytes(path, shape)

    def test_target_mask_and_float_inputs(self):
        depth = np.array([[1., 0., -2., np.inf], [2., np.nan, .001, 3.]], np.float32)
        rows = [dict(spike_l='l', spike_r='r', depth='d')]
        spike = torch.ones(50, 2, 4, dtype=torch.uint8)
        with patch('core.utils.reader.read_depth', return_value=depth), \
             patch('input_pipeline.read_spike_bytes', return_value=spike):
            left, right, target, valid, index = NativeBytesDataset('/unused', rows)[0]
        mask = np.isfinite(depth) & (depth > 0)
        expected = np.zeros_like(depth)
        expected[mask] = -(289.740625*.08)/depth[mask]
        mask &= np.abs(expected) < 512
        self.assertTrue(np.array_equal(target.numpy()[0], expected))
        self.assertTrue(np.array_equal(valid.numpy(), mask))
        self.assertEqual(index, 0)
        self.assertEqual((2*left.float()-1).min(), 1)

    def test_budget_and_no_repeated_decoding_or_mutation(self):
        source = TinyDataset()
        with self.assertRaises(MemoryError):RamDataset(source, 1)
        self.assertEqual(source.calls, 1)
        source.calls = 0
        cached = RamDataset(source, 100000)
        self.assertEqual(source.calls, len(source))
        before = copy.deepcopy(cached.samples)
        for step in range(8):prepare_batch(cached, list(range(16)), step, 2345)
        self.assertEqual(source.calls, len(source))
        for old, new in zip(before, cached.samples):
            for a, b in zip(old[:4], new[:4]):self.assertTrue(torch.equal(a, b))
        self.assertEqual(cached.bytes, sum(v.numel()*v.element_size() for item in before for v in item[:4]))

    def test_all_batches_match_legacy_and_rng_is_unchanged(self):
        contract = load('pipeline_legacy_contract', HERE.parent/'spiking-attention-placement-phase2/protocol.py')
        dataset = TinyDataset()
        class FloatDataset:
            def __getitem__(self, i):
                l,r,t,v,index = dataset[i]
                return l.float(),r.float(),t,v,index
        order = torch.randperm(16, generator=torch.Generator().manual_seed(25)).tolist()
        for seed in (1234, 2345):
            before = torch.get_rng_state().clone(), random.getstate(), np.random.get_state()
            legacy = contract.Prefetch(FloatDataset(), order, 0, 5, seed)
            fast = PreparedPrefetch(dataset, order, 0, 5, seed, workers=2, depth=3)
            try:
                for step in range(5):
                    old,new = legacy.get(step),fast.get(step)
                    self.assertEqual(old['indices'],new['indices'])
                    self.assertEqual(old['flags'],new['flags'])
                    denominator = sum(int(item[3].sum()) for item in old['items'])
                    self.assertEqual(new['valid_count'], denominator)
                    for i, micro in enumerate(new['microbatches']):
                        originals=old['items'][i*4:i*4+4]
                        expected=[torch.stack([item[j] for item in originals]) for j in range(4)]
                        actual=[v.float() if j<2 else v for j,v in enumerate(micro['tensors'])]
                        for a,b in zip(actual,expected):self.assertTrue(torch.equal(a,b))
                        self.assertEqual(micro['valid_count'],int(expected[3].sum()))
                        # Preserve loss normalization across unequal valid counts.
                        truth=actual[2][:,0];mask=actual[3]
                        x=torch.zeros_like(truth,requires_grad=True)
                        loss=(x-truth).abs()[mask].mean()*micro['valid_count']/denominator
                        y=torch.zeros_like(truth,requires_grad=True)
                        ref=(y-expected[2][:,0]).abs()[expected[3]].mean()*int(expected[3].sum())/denominator
                        loss.backward();ref.backward();self.assertTrue(torch.equal(x.grad,y.grad))
            finally:legacy.close();fast.close()
            self.assertTrue(torch.equal(before[0],torch.get_rng_state()))
            self.assertEqual(before[1],random.getstate())
            self.assertTrue(np.array_equal(before[2][1],np.random.get_state()[1]))

    def test_resume_and_prefetch_errors(self):
        ds=TinyDataset();order=list(range(16))
        full=PreparedPrefetch(ds,order,4,7,2345,workers=2,depth=2)
        resumed=PreparedPrefetch(ds,order,5,7,2345,workers=1,depth=1)
        try:
            with self.assertRaises(RuntimeError):full.get(5)
            full.get(4)
            a,b=full.get(5),resumed.get(5)
            self.assertEqual(a['flags'],b['flags'])
            for x,y in zip(a['microbatches'],b['microbatches']):
                for u,v in zip(x['tensors'],y['tensors']):self.assertTrue(torch.equal(u,v))
        finally:full.close();resumed.close()
        with self.assertRaises(ValueError):PreparedPrefetch(ds,order,0,1,2345,depth=0)

    def test_worker_preload_selection(self):
        worker=load('preload_worker_adapter',HERE/'worker.py')
        captured={}
        def main(**kwargs):captured.update(kwargs);return 0
        with patch.object(sys,'argv',['worker','--seed','2345','--variant','linear-surrogate',
                '--input-mode','preload','--preload-gib','12','--input-workers','1','--prefetch-depth','3','--arm','B']), \
             patch.object(worker,'load',side_effect=[SimpleNamespace(Prefetch=lambda *a,**k:None),SimpleNamespace(main=main)]):
            self.assertEqual(worker.main(),0)
        pipeline=captured['input_pipeline']
        self.assertEqual((pipeline.mode,pipeline.seed,pipeline.workers,pipeline.depth),('preload',2345,1,3))

    def test_launcher_forwards_preload_selection(self):
        runner=load('preload_runner_adapter',HERE/'run_on_gpu.py')
        captured={}
        def main(**kwargs):captured.update(kwargs)
        with patch.dict('os.environ',{'STEREO_RUN_ROOT':'/configured/runs'}), \
             patch.object(sys,'argv',['runner','--seed','1234','--variant','signed-surrogate',
                '--input-mode','preload','--preload-gib','12','--input-workers','2','--prefetch-depth','3']), \
             patch.object(runner,'load',return_value=SimpleNamespace(main=main)):
            runner.main()
        self.assertEqual(captured['worker_args'],('--seed','1234','--variant','signed-surrogate',
            '--input-mode','preload','--preload-gib','12.0','--input-workers','2','--prefetch-depth','3'))

    def test_preflight_does_not_preload_corpus(self):
        pipeline=InputPipeline('preload',budget_gib=12)
        with patch('input_pipeline.RamDataset',side_effect=AssertionError('unexpected preload')):
            train,dev=pipeline.datasets('/unused',[{}]*960,[{}]*2,preflight=True)
        self.assertEqual(pipeline.metadata['cached_bytes'],0)
        self.assertEqual(len(train),960)
        self.assertEqual(len(dev),2)


if __name__=='__main__':unittest.main()
