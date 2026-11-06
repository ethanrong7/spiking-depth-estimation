"""Retained B endpoint evaluation; fixed official batch2 and cross-scene donors."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path

def main():
 p=argparse.ArgumentParser()
 for name in ('source','data-root','manifest','run-root','reference-root','shared-lock-root','output'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args()
 import torch,cv2
 assert os.environ.get('SLURM_JOB_ID')=='2751' and torch.cuda.device_count()==1
 assert str(torch.cuda.get_device_properties(0).uuid)=='8d9598da-dfec-a680-9b2f-975c955e5130'
 torch.cuda.set_per_process_memory_fraction(.30);torch.set_num_threads(6);cv2.setNumThreads(1)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
 project=a.source/'projects/snn-depth-estimation'
 sys.path[:0]=[str(project),str(project/'experiments/shared-gpu-30k-v1'),str(project/'experiments/spiking-transformer-stereo/scripts')]
 import run_experiment as d
 from common import phase_lock
 from core.stereo_factor_feedforward import install_feedforward
 from core.utils.utils import InputPadder
 from b_endpoint_probes import probe_models
 import core.stereo_factor_feedforward as impl
 checkpoint=a.run_root/'train/B/step-2500.pt';assert checkpoint.exists()
 assert json.loads((a.run_root/'train/B/result.json').read_text())['updates']==2500
 ck=torch.load(checkpoint,map_location='cpu',weights_only=False);assert ck['updates']==2500
 model=install_feedforward(d.build_model('original',1234)).cuda().eval();model.load_state_dict(ck['model']);del ck
 original_checkpoint=a.reference_root/'original/last.pt';ck=torch.load(original_checkpoint,map_location='cpu',weights_only=False);assert ck['updates']==30000
 original=d.build_model('original',1234).cuda().eval();original.load_state_dict(ck['model']);del ck
 manifest=json.loads(a.manifest.read_text());ds=d.ManifestDataset(a.data_root,manifest['dev'])
 def loader():return torch.utils.data.DataLoader(ds,batch_size=2,num_workers=0,generator=torch.Generator().manual_seed(2225))
 official=next(json.loads(x) for x in (a.run_root/'train/B/evaluations.jsonl').read_text().splitlines() if json.loads(x)['update']==2500)
 expected={(r['scene'],r['frame']):r for r in official['per_frame']}
 bank={};parity=[];panels=[];native_class=impl.CorrBlock1D;captured={}
 class Capture(native_class):
  def __init__(self,left,right,*args,**kwargs):captured['right']=right.detach().cpu();super().__init__(left,right,*args,**kwargs)
 def guard():
  free,total=torch.cuda.mem_get_info()
  if free<4*1024**3:raise RuntimeError('Memory guard: below 4GiB free; retained checkpoint untouched')
 def save(status):
  outcomes={}
  for panel in panels:
   for intervention,values in panel['interventions'].items():
    arm=outcomes.setdefault(intervention,dict(per_frame=[],per_scene=[]))
    for frame in values['native_metrics']['per_frame']:
     identity=panel['frames'][frame['batch_index']];arm['per_frame'].append(dict(**identity,**frame,epe=frame['error_sum']/frame['count'],bad3_percent=100*frame['bad3_count']/frame['count']))
  for arm in outcomes.values():
   for scene in sorted({r['scene'] for r in arm['per_frame']}):
    rows=[r for r in arm['per_frame'] if r['scene']==scene];arm['per_scene'].append(dict(scene=scene,frames=len(rows),epe=sum(r['epe'] for r in rows)/len(rows),bad3_percent=sum(r['bad3_percent'] for r in rows)/len(rows)))
  result=dict(status=status,job=os.environ['SLURM_JOB_ID'],step=os.environ.get('SLURM_STEP_ID'),gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),original_checkpoint=str(original_checkpoint),original_updates=30000,original_exposure_matched=False,source_identity=json.loads((a.source/'source-registry.json').read_text())['source_identity_sha256'],diagnostic_files={str(Path(__file__).name):hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'b_endpoint_probes.py':hashlib.sha256(Path(__file__).with_name('b_endpoint_probes.py').read_bytes()).hexdigest()},native_outcomes=outcomes,physical_batch=2,donor='(official frame index+2)%60, distinct scene',parity=parity,panels=panels,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
  a.output.parent.mkdir(parents=True,exist_ok=True);tmp=a.output.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2,allow_nan=False));tmp.replace(a.output)
 with phase_lock(a.shared_lock_root,'stereo-factor-B-probe',True,torch.cuda.empty_cache),torch.no_grad():
  try:
   impl.CorrBlock1D=Capture
   for left,right,target,valid,indices in loader():
    guard();left,right,target,valid=[v.cuda() for v in (left,right,target,valid)];padder=InputPadder(left.shape,divis_by=32);model.set_native_geometry(left.shape,padder._pad);pred=d.forward(model,left,right,32)
    for j,index in enumerate(indices.tolist()):
     row=ds.rows[index];epe=float((pred[j,0]-target[j,0]).abs()[valid[j]].mean());diff=abs(epe-expected[(row['scene'],row['frame'])]['epe']);parity.append(dict(index=index,scene=row['scene'],frame=row['frame'],epe=epe,official_epe_difference=diff));assert diff<1e-5
     bank[index]=captured['right'][j]
    d.reset_model(model)
  finally:impl.CorrBlock1D=native_class
  save('PARITY_PASSED')
  for left,right,target,valid,indices in loader():
   guard();ids=indices.tolist();donors=[(i+2)%len(ds) for i in ids];assert all(ds.rows[i]['scene']!=ds.rows[j]['scene'] for i,j in zip(ids,donors))
   left,right,target,valid=[v.cuda() for v in (left,right,target,valid)]
   panel=probe_models(model,original,left,right,target,valid[:,None],permuted_right_features=torch.stack([bank[i] for i in donors]).cuda())
   for frame in panel['interventions']['native']['native_metrics']['per_frame']:
    row=ds.rows[ids[frame['batch_index']]];assert abs(frame['error_sum']/frame['count']-expected[(row['scene'],row['frame'])]['epe'])<1e-5,'probe native official parity failed'
   panel['frames']=[dict(index=i,scene=ds.rows[i]['scene'],frame=ds.rows[i]['frame'],donor_index=j,donor_scene=ds.rows[j]['scene']) for i,j in zip(ids,donors)];panels.append(panel);save('RUNNING');d.reset_model(model);torch.cuda.empty_cache()
  save('COMPLETED')
 print(str(a.output),flush=True)
if __name__=='__main__':main()
