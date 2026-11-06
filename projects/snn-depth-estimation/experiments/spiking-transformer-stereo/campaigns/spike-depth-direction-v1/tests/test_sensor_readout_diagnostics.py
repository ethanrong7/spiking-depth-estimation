"""Synthetic scalar readout aggregation/geometry checks, no model/data/GPU."""
import sys,unittest,copy
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from sensor_readout_diagnostics import frame_summaries,aggregate,MEASUREMENTS

def fields():
 j=torch.arange(104);prob=(j[None,:]<=j[:,None]).float()/(j[:,None]+1);p=prob[None,None].expand(1,64,-1,-1).clone();d0=(j.float()/2)[None,None,None].expand(1,1,64,-1).clone();r=torch.full_like(d0,-.5);return {'d0':d0,'r':r,'d':d0+r,'p':p}

class ReadoutTests(unittest.TestCase):
 def test_nativeFOV_distinct_padded_support_entropy_GTmass(self):
  flow=torch.full((1,1,250,400),-4.);target=torch.full_like(flow,-4.);valid=torch.zeros((1,250,400),dtype=torch.bool);valid[0,1,0]=True;valid[0,1,4]=True;f=fields();row=frame_summaries(flow,target,valid,f,[{'scene':'s','frame':'0001'}])[0];a=row['populations'];out=a['native_right_out_of_FOV'];inside=a['native_right_inbounds'];self.assertEqual(a['all_valid_centers']['site_count'],2);self.assertEqual(out['site_count'],1);self.assertEqual(out['head_support_outside_count'],0);self.assertEqual(out['GT_neighborhood_mass_eligible_sites'],0);self.assertIsNone(out['metrics']['GT_neighborhood_probability_mass']);self.assertEqual(inside['site_count'],1);self.assertAlmostEqual(inside['metrics']['GT_neighborhood_probability_mass'],.5);self.assertAlmostEqual(inside['metrics']['padded_right_probability_mass'],.5);self.assertAlmostEqual(inside['metrics']['normalized_entropy'],1.);self.assertAlmostEqual(inside['metrics']['matching_error_pixels'],2.);self.assertAlmostEqual(inside['metrics']['post_residual_coarse_error_pixels'],0.);self.assertAlmostEqual(inside['metrics']['native_center_error_pixels'],0.);self.assertAlmostEqual(inside['metrics']['residual_correction_gain_pixels'],2.)
  # Larger left-edge GT is outside both native FOV and numerical head support.
  target[0,0,1,0]=-12.;row=frame_summaries(flow,target,valid,f,[{'scene':'s','frame':'0001'}])[0];self.assertEqual(row['populations']['native_right_out_of_FOV']['head_support_outside_count'],1)
 def test_equal_frame_then_equal_scene_and_empty_rows(self):
  flow=torch.zeros((1,1,250,400));valid=torch.zeros((1,250,400),dtype=torch.bool);valid[0,1,4]=True;base=frame_summaries(flow,torch.full_like(flow,-4.),valid,fields(),[{'scene':'A','frame':'1'}])[0];second=copy.deepcopy(base);second['frame']='2';third=copy.deepcopy(base);third['scene']='B'
  for r,value,count in [(base,1.,1),(second,3.,100),(third,10.,1)]:
   for pop in r['populations'].values():
    if pop['site_count']:
     pop['site_count']=count
     for key in MEASUREMENTS:
      if pop['metrics'][key]is not None:pop['metrics'][key]=value
  result=aggregate([base,second,third]);a=result['aggregate_equal_scene']['all_valid_centers'];self.assertEqual(a['site_count'],102);self.assertEqual(a['contributing_scenes'],2);self.assertEqual(a['metrics']['matching_error_pixels'],6.);self.assertEqual(result['per_scene'][0]['populations']['all_valid_centers']['metrics']['matching_error_pixels'],2.);empty=result['aggregate_equal_scene']['native_right_out_of_FOV'];self.assertEqual(empty['site_count'],0);self.assertEqual(empty['contributing_scenes'],0);self.assertTrue(all(x is None for x in empty['metrics'].values()))
if __name__=='__main__':unittest.main()
