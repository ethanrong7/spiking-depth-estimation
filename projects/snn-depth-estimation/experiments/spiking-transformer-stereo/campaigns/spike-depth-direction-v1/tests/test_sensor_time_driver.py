"""Tiny synthetic driver-only checks; no model forward or dataset payload."""
import importlib.util,sys,unittest
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import run_sensor_time as s

class DriverTests(unittest.TestCase):
 def test_entire_cyclic_order_and_independent_RNG(self):
  torch.manual_seed(987);before=torch.get_rng_state().clone();order=s.sample_order();self.assertTrue(torch.equal(before,torch.get_rng_state()));self.assertEqual(order[:2],[495,243]);self.assertEqual(sorted(order),list(range(960)));self.assertEqual([i for step in range(480)for i in s.step_indices(order,step)],order);self.assertEqual(s.step_indices(order,480),[495,243]);self.assertEqual(s.step_indices(order,4999),s.step_indices(order,4999%480))
 def test_one_final_smoothL1_and_outofview_support(self):
  pred=torch.zeros(2,1,1,4,requires_grad=True);target=torch.tensor([[[[-40.,-2.,-600.,float('nan')]]],[[[-3.,-4.,-5.,-6.]]]]);valid=torch.tensor([[[True,True,True,True]],[[False,True,True,True]]]);loss=s.final_loss(pred,target,valid);expected=torch.tensor([39.5,1.5,3.5,4.5,5.5]).mean();self.assertTrue(torch.equal(loss,expected));loss.backward();self.assertGreater(float(pred.grad[0,0,0,0]),0);self.assertEqual(float(pred.grad[0,0,0,2]),0);self.assertEqual(float(pred.grad[1,0,0,0]),0)
 def test_binary_reader_no_norm_or_stereo_swap(self):
  depth=np.ones((250,400),np.float32);depth[0,0]=0;depth[0,1]=np.inf;left=np.zeros((50,250,400),np.uint8);right=left.copy();left[0,2,3]=1;right[49,4,5]=1;old_read,old_spike=s.reader.read_depth,s.reader.get_block_spikes
  try:
   s.reader.read_depth=lambda _:depth;s.reader.get_block_spikes=lambda name:left if 'left'in name else right;data=s.ManifestDataset('.', [{'depth':'d','spike_l':'left','spike_r':'right'}]);a,b,t,v,index=data[0];self.assertEqual(tuple(a.shape),(50,250,400));self.assertTrue(torch.equal(a,torch.from_numpy(left.astype(np.float32))));self.assertTrue(torch.equal(b,torch.from_numpy(right.astype(np.float32))));self.assertEqual(float(a[49,4,5]),0);self.assertEqual(float(b[49,4,5]),1);self.assertFalse(bool(v[0,0]));self.assertFalse(bool(v[0,1]));self.assertTrue(bool(v[2,0]));self.assertAlmostEqual(float(t[0,2,0]),-(289.740625*.08),places=5)
  finally:s.reader.read_depth,s.reader.get_block_spikes=old_read,old_spike
 def test_optimizer_scheduler_actual_fixed_law(self):
  tiny=torch.nn.Linear(2,1);opt=s.optimizer(tiny);schedule=s.scheduler(opt);self.assertEqual(opt.param_groups[0]['weight_decay'],1e-4);self.assertEqual(opt.param_groups[0]['eps'],1e-8);self.assertEqual(opt.param_groups[0]['betas'],(.9,.999));self.assertEqual(schedule.total_steps,10100);self.assertEqual(schedule.last_epoch,0);self.assertAlmostEqual(opt.param_groups[0]['lr'],8e-6);self.assertFalse(opt.state)
 def test_reject_graph_bearing_diagnostics(self):
  with self.assertRaises(RuntimeError):s.diagnostics({'x':torch.tensor(1.,requires_grad=True)})
  self.assertEqual(s.diagnostics({'rate':torch.tensor(.5),'sensor_steps':50}),{'rate':.5,'sensor_steps':50.})
if __name__=='__main__':unittest.main()
