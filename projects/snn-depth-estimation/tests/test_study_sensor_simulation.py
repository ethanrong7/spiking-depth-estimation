"""Tiny CPU sensor physics/representation checks, not dataset generation."""
from pathlib import Path
import sys
import unittest
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core.study_sensor_simulation import (PairedSensorState, SensorConfig, rgb_to_gray,
    registered_output_shape, resample_float_gray, resample_disparity_nearest)


class SensorTests(unittest.TestCase):
    def test_constant_intensity_vs_contrast(self):
        gray=torch.full((2,3),.5)
        state=PairedSensorState(gray)
        outputs=[state.step(gray) for _ in range(50)]
        self.assertEqual(sum(int(s.sum()) for s,e in outputs),30) # 5 spikes/pixel
        self.assertEqual(sum(int(e.sum()) for s,e in outputs),0)
        torch.testing.assert_close(state.voltage,torch.zeros_like(gray))

    def test_count_polarity_reference_and_reversal(self):
        values=[.05,.25,.25,.08,.5,.02,.02,.8]
        state=PairedSensorState(torch.tensor([[values[0]]],dtype=torch.float32))
        previous=np.log(np.float32(values[0])+np.float32(.001)).astype(np.float32)
        reference=previous
        for value in values[1:]:
            current=np.log(np.float32(value)+np.float32(.001)).astype(np.float32)
            up=current>=previous; delta=np.float32(.2 if up else -.2); count=0
            if abs(current-previous)>1e-6:
                cross=np.float32(reference+delta)
                while ((cross>previous and cross<=current) if up else (cross<previous and cross>=current)):
                    count+=1;reference=cross;cross=np.float32(cross+delta)
            _,events=state.step(torch.tensor([[value]],dtype=torch.float32))
            self.assertEqual(events[:,0,0].tolist(),[count,0] if up else [0,count])
            previous=current
        self.assertGreater(count,1)

    def test_no_window_reset_and_mass_balance(self):
        state=PairedSensorState(torch.full((1,1),.37)); total=0
        for _ in range(100):
            spikes,_=state.step(torch.full((1,1),.37)); total+=int(spikes.sum())
        self.assertEqual(state.steps,100)
        self.assertAlmostEqual(5*total+float(state.voltage.item()),37.,places=4)
        # Two 50-bin windows continue the same per-sequence charge.
        reset=PairedSensorState(torch.full((1,1),.37)); reset_total=0
        for _ in range(50): reset_total+=int(reset.step(torch.full((1,1),.37))[0].sum())
        self.assertNotEqual(total,2*reset_total)

    def test_registered_shapes_and_disparity_pixel_units(self):
        self.assertEqual(registered_output_shape((540,960),.5),(270,480))
        self.assertEqual(registered_output_shape((540,960),.25),(135,240))
        target=torch.full((8,16),8.)
        for scale,expected in ((.5,4.),(.25,2.)):
            resized=resample_disparity_nearest(target,scale)
            torch.testing.assert_close(resized,torch.full_like(resized,expected))
            # Translate the right image +1 native pixel: d=xL-xR becomes 7.
            shifted=resample_disparity_nearest(target-1,scale)
            torch.testing.assert_close(shifted,torch.full_like(shifted,7*scale))
            torch.testing.assert_close(resized-shifted,torch.full_like(resized,scale))
        invalid=target.clone();invalid[1,1]=float('nan');invalid[2,2]=0
        self.assertTrue(torch.isnan(resample_disparity_nearest(invalid,.5)[0,0]))
        self.assertEqual(resample_disparity_nearest(invalid,.25)[0,0].item(),0)
        with self.assertRaises(ValueError):registered_output_shape((540,960),.3)

    def test_area_gray_preserves_known_epipolar_shift(self):
        left=torch.arange(16,dtype=torch.float32)[None].repeat(8,1)/24
        right=torch.roll(left,-8,dims=1) # d=8 native pixels; ignore wraparound.
        for scale in (.5,.25):
            a=resample_float_gray(left,scale);b=resample_float_gray(right,scale)
            shift=int(8*scale)
            torch.testing.assert_close(a[:,shift:],b[:,:-shift])
            self.assertEqual(a.shape,registered_output_shape(left.shape,scale))

    def test_gray_is_resampled_before_binary_sensor(self):
        gray=torch.tensor([[0.,1.],[0.,1.]])
        reduced=resample_float_gray(gray,.5)
        self.assertEqual(reduced.item(),.5)
        native_state=PairedSensorState(gray);reduced_state=PairedSensorState(reduced)
        for _ in range(5):
            native,_=native_state.step(gray);small,_=reduced_state.step(reduced)
        self.assertEqual(small.item(),0)
        self.assertEqual(native.float().mean().item(),.5)
        # Post-hoc binary averaging would give .5 instead of the proper zero.
        self.assertNotEqual(small.item(),native.float().mean().item())

    def test_parameter_bounds_and_shared_radiometry(self):
        rgb=torch.tensor([[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]])
        torch.testing.assert_close(rgb_to_gray(rgb),torch.tensor([[.299,.587,.114]]))
        for settings in ({'T':49},{'spike_threshold':.5},{'noise':.1},{'refractory':.1},{'log_epsilon':0}):
            with self.assertRaises(ValueError): SensorConfig(**settings)
        with self.assertRaises(ValueError): PairedSensorState(torch.tensor([[float('nan')]]))

if __name__=='__main__':unittest.main()
