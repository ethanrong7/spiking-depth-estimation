"""Tiny CPU integration checks for the new entry point."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tempfile
import unittest
import numpy as np
import torch
import runtime as rt


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_recurrent_forward_gradient_and_state_reset(self):
        model=rt.build_model('spikestereonet')
        model.train(); model.freeze_bn()
        left=torch.rand(1,50,64,128); right=torch.rand_like(left)
        prediction,aux=rt.forward(model,'spikestereonet',left,right,2,training=True)
        value=rt.loss('spikestereonet',aux,torch.full_like(prediction,-2.),torch.ones(1,64,128,dtype=torch.bool))
        value.backward()
        self.assertTrue(torch.isfinite(value))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters()))
        model.eval()
        with torch.no_grad():
            first,_=rt.forward(model,'spikestereonet',left,right,2)
            second,_=rt.forward(model,'spikestereonet',left,right,2)
        torch.testing.assert_close(first,second)

    def test_direct_forward_loss_and_gradient(self):
        model=rt.build_model('s2m2-v2')
        left=torch.rand(1,50,32,64);right=torch.rand_like(left)
        prediction,aux=rt.forward(model,'s2m2-v2',left,right,training=True)
        value=rt.loss('s2m2-v2',aux,torch.full_like(prediction,-2.),torch.ones(1,32,64,dtype=torch.bool))
        value.backward()
        self.assertTrue(torch.isfinite(value))
        self.assertTrue(torch.isfinite(model.disp_init.q.linear.weight.grad).all())
        self.assertGreater(float(model.disp_init.q.linear.weight.grad.abs().sum()),0)

    def test_numpy_and_packed_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp)
            values=np.random.default_rng(12).integers(0,2,(50,250,400),dtype=np.uint8)
            np.save(tmp/'spikes.npy',values)
            np.packbits(values.reshape(-1),bitorder='little').tofile(tmp/'spikes.dat')
            torch.testing.assert_close(rt.load_spikes(tmp/'spikes.npy'),rt.load_spikes(tmp/'spikes.dat'))
            np.save(tmp/'bad.npy',values[:2])
            with self.assertRaises(ValueError):rt.load_spikes(tmp/'bad.npy')


if __name__=='__main__':unittest.main()
