import importlib.util
from pathlib import Path
import unittest
import numpy as np
P=Path(__file__).resolve().parents[1]/'scripts/stage9_analyze.py'
spec=importlib.util.spec_from_file_location('s9',P);s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
class AnalysisTests(unittest.TestCase):
 def test_ridge_normal_equations(self):
  x=np.array([[0.,1],[1,2],[2,5],[3,9],[4,7]])
  y=np.column_stack([2+x[:,0],x[:,1]**2])
  f=s.ridge_fit(x,y);z=np.clip((x-x.mean(0))/np.maximum(x.std(0),1e-6),-5,5);d=np.column_stack([np.ones(5),z,z*z]);p=np.diag([0,.01,.01,.01,.01])
  np.testing.assert_allclose(d.T@(d@f['beta']-y)/5+p@f['beta'],0,atol=1e-12)
  np.testing.assert_allclose(s.ridge_predict(f,x),d@f['beta'])
 def test_lp_exact_expected_mixture(self):
  e=np.full((2,9),9.);e[:,0]=3;e[:,3]=1
  c=np.ones((2,9));lp=s.lp_allocation(e,c,4,1,[0,3],True,True)
  self.assertTrue(lp['feasible']);np.testing.assert_allclose(lp['weights'][:,[0,3]],[[.5,.5]]*2,atol=1e-8);self.assertAlmostEqual(lp['objective'],2)
 def test_oracle_restricted_and_missing_costs(self):
  e=np.full((2,9),99.);e[0,0]=0;e[1,1]=0
  c=np.full((2,9),np.nan);c[:,:2]=1
  lp=s.lp_allocation(e,c,0,1,[0,1]);self.assertTrue(lp['feasible']);self.assertEqual(lp['objective'],0);self.assertTrue(np.isfinite(lp['gpu_ms']))
  self.assertFalse(s.lp_allocation(e,c,0,.5,[0,1])['feasible'])
 def test_calibration_margin_and_ties(self):
  pred=np.zeros((4,9));pred[:,1]=1;actual=np.ones((4,9));actual[:,1]=0;cost=np.ones((4,9))
  self.assertIsNone(s.choose_penalties(pred,actual,cost,0,1,1,1,[0,1]))
  p=s.choose_penalties(pred,actual,cost,0,2,1,1,[0,1]);self.assertEqual(p['lambda'],0);self.assertEqual(p['mu'],0)
 def test_uniform_cost_and_lookahead(self):
  c=np.ones((3,9));w=s.random_allocation(c,4,2);self.assertAlmostEqual(float(w[0]@s.LOOK),4);self.assertAlmostEqual(w.sum(1)[0],1)
 def test_expected_frame_scene_and_pooled(self):
  rows=[]
  for sid,e,vc in [('a',2,1),('a',4,3),('b',10,2)]:
   rows.append({'source_id':sid,'valid_pixels':vc,'actions':{a:{'epe':e+i,'bad3_percent':10*i,'bad3_count':vc*i/10} for i,a in enumerate(s.ACTIONS)}})
  w=np.zeros((3,9));w[:,0]=.5;w[:,1]=.5
  r=s.reduction(rows,w,np.ones((3,9)));self.assertAlmostEqual(r['epe'],7);self.assertAlmostEqual(r['bad3_percent'],5);self.assertAlmostEqual(r['pooled_bad3_percent'],5)
 def test_scalar_boundary(self):
  for scalar in ('density','entropy'):
   x=np.array([[0.],[1.],[2.]])
   yy=np.zeros((3,9));yy[:,1]=[0,1,2]
   fit=s.ridge_fit(x,yy)
   policy={'feasible':True,'scalar':scalar,'start_action':0,'ridge':fit,'selection':{'allowed':[0,1],'lambda':0,'mu':0,'calibration_mean_action_cost':[1,1]}}
   a=np.zeros(14);b=np.ones(14)*999
   if scalar=='density':a[:2]=b[:2]=1
   else:a[11]=b[11]=1
   self.assertEqual(s.predict_action(policy,a,1),s.predict_action(policy,b,1))
 def test_actual_action_must_match(self):
  row={'source_id':'x','valid_pixels':10,'deployed_policies':{'L8-C1':{'joint':{'action':'p24-r8'}}}}
  policy={'name':'joint','result':{'frame_weights':s.onehot([0]).tolist()}}
  with self.assertRaises(AssertionError):s.actual_policy_result([row],'L8-C1',policy)
 def test_packet_content_tampering(self):
  import copy
  packet={'T_gpu_ms':1.,'feature_names':['x'],'cells':[{'ridge':{'beta':[[1.]]},'selection':{'lambda':0.},'mixture':[.5,.5]}]}
  self.assertTrue(s.verify_frozen_packet(packet,copy.deepcopy(packet)))
  for key in ('T','beta','penalty','mixture'):
   changed=copy.deepcopy(packet)
   if key=='T':changed['T_gpu_ms']=2.
   elif key=='beta':changed['cells'][0]['ridge']['beta'][0][0]=1.01
   elif key=='penalty':changed['cells'][0]['selection']['lambda']=.01
   else:changed['cells'][0]['mixture']=[.6,.4]
   with self.assertRaises(AssertionError):s.verify_frozen_packet(changed,packet)
 def test_fixed_oracle_ties_compute_then_lex(self):
  e=np.ones((2,9));c=np.ones((2,9));c[:,0]=2
  lp=s.lp_allocation(e,c,0,3,[0,1],shared=True,tie=True)
  np.testing.assert_allclose(lp['weights'][:,1],1,atol=1e-10)
  c[:,0]=1;lp=s.lp_allocation(e,c,0,3,[0,1],shared=True,tie=True)
  np.testing.assert_allclose(lp['weights'][:,0],0,atol=1e-10)
 def test_actual_parity_and_metric_fail_closed(self):
  import copy
  direct={'epe':2.,'bad3_count':2,'bad3_percent':20.,'valid_pixels':10,'output_sha256':'a'*64}
  par={'status':'PASS','both_finite':True,'atol_pixels':1e-5,'rtol':0,'max_abs_difference':0.,'bitwise_equal':True,'differing_pixel_count':0,'actual_output_sha256':'a'*64,'direct_output_sha256':'a'*64,'bad3_flip_count':0,'sample_id':'s/1','action':'p24-r4'}
  rec={**direct,'action':'p24-r4','gpu_ms':1.,'CPU_selector_ms':.01,'wall_ms':2.,'parity_against_direct_max_abs':0.,'parity':par}
  row={'source_id':'s','sample_id':'s/1','valid_pixels':10,'actions':{'p24-r4':direct},'deployed_policies':{'L8-C1':{'joint':rec}}}
  policy={'name':'joint','result':{'frame_weights':s.onehot([0]).tolist()}}
  self.assertEqual(s.actual_policy_result([row],'L8-C1',policy)['epe'],2.)
  signed=copy.deepcopy(row);q=signed['deployed_policies']['L8-C1']['joint']['parity'];q.update(bitwise_equal=False,differing_pixel_count=1,actual_output_sha256='b'*64)
  self.assertEqual(s.actual_policy_result([signed],'L8-C1',policy)['epe'],2.)
  for problem in ('tolerance','maxdiff','status','identity','metric','count'):
   bad=copy.deepcopy(row);v=bad['deployed_policies']['L8-C1']['joint']
   if problem=='tolerance':v['parity']['atol_pixels']=1e-3
   elif problem=='maxdiff':v['parity']['max_abs_difference']=v['parity_against_direct_max_abs']=.01
   elif problem=='status':v['parity']['status']='FAIL'
   elif problem=='identity':v['parity']['direct_output_sha256']='b'*64
   elif problem=='metric':v['epe']=1.
   else:v['bad3_count']=3;v['bad3_percent']=30.
   with self.assertRaises(AssertionError):s.actual_policy_result([bad],'L8-C1',policy)
 def test_ordered_calibration_bytes_fail_closed(self):
  import tempfile,hashlib,json
  with tempfile.TemporaryDirectory() as td:
   path=Path(td)/'cal.jsonl';saved=[{'sample_id':'b'},{'sample_id':'a'}];path.write_text(''.join(json.dumps(r)+'\n' for r in saved))
   sha=hashlib.sha256(path.read_bytes()).hexdigest();meta={'calibration_jsonl_path':str(path),'calibration_jsonl_sha256':sha}
   with self.assertRaises(AssertionError):s.verify_calibration_environment(meta,list(reversed(saved)),{str(path):sha})
if __name__=='__main__':unittest.main()
