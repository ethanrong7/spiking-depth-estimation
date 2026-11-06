"""Retained two-pass trained weights, official replay then one-pass reliance probe."""
import argparse,json,hashlib,os,sys
from pathlib import Path
import torch

def main():
 p=argparse.ArgumentParser()
 for k in ('source','root','data-root','manifest','shared-lock-root','output'):p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();assert os.environ['SLURM_JOB_ID']=='2751' and torch.cuda.device_count()==1
 assert str(torch.cuda.get_device_properties(0).uuid)=='8d9598da-dfec-a680-9b2f-975c955e5130';torch.cuda.set_per_process_memory_fraction(.30)
 project=a.source/'projects/snn-depth-estimation';sys.path[:0]=[str(project),str(project/'experiments/s2m2-mechanism-next-v1'),str(project/'experiments/shared-gpu-30k-v1'),str(project/'experiments/spiking-transformer-stereo/scripts')]
 from core.s2m2_spiking_v2 import SpikingS2M2V2
 from core.s2m2_tied_mrt import apply_tied_mrt
 from common import phase_lock
 from metrics import correspondence_metrics
 import run_experiment as driver
 import random,numpy as np,cv2
 torch.set_num_threads(6);cv2.setNumThreads(1);random.seed(1234);np.random.seed(1234);torch.manual_seed(1234);torch.cuda.manual_seed_all(1234)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
 directory=a.root/'train/s2m2_v2b';checkpoint=directory/'step-2500.pt';ck=torch.load(checkpoint,map_location='cpu',weights_only=False);assert ck['updates']==2500
 model=apply_tied_mrt(SpikingS2M2V2()).cuda().eval();model.load_state_dict(ck['model']);identity=ck['source_identity'];del ck
 ds=driver.ManifestDataset(a.data_root,json.loads(a.manifest.read_text())['dev']);official=next(json.loads(s) for s in (directory/'evaluations.jsonl').read_text().splitlines() if json.loads(s)['update']==2500)
 result=dict(status='RUNNING',source_identity=identity,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),batch_size=2,job=os.environ['SLURM_JOB_ID'],step=os.environ.get('SLURM_STEP_ID'),conditions={},scope='One-passprobeonTWO-PASStrainedweights;notseparatelytrainedone-passcontrol')
 a.output.parent.mkdir(parents=True,exist_ok=True)
 def save():a.output.write_text(json.dumps(result,indent=2)+'\n')
 baseline={}
 with phase_lock(a.shared_lock_root,'s2m2-frozen-pass-probe',True,torch.cuda.empty_cache),torch.no_grad():
  for name,passes in [('two_pass',2),('one_pass',1)]:
   model.transformer.tied_mrt_passes=passes;frames=[]
   loader=torch.utils.data.DataLoader(ds,batch_size=2,num_workers=0,generator=torch.Generator().manual_seed(2225))
   for left,right,target,valid,indices in loader:
    assert torch.cuda.mem_get_info()[0]>4*1024**3
    left,right,target,valid=[v.cuda() for v in (left,right,target,valid)];out=model(left,right)
    for j,index in enumerate(indices.tolist()):
     row=ds.rows[index];key=(row['scene'],row['frame']);err=(out['flow'][j,0]-target[j,0]).abs()[valid[j]];item=dict(scene=key[0],frame=key[1],valid_pixels=int(valid[j].sum()),**driver.metric_values(err))
     item['initial_epe']=float((out['initial_flow'][j,0]-target[j,0]).abs()[valid[j]].mean());item['global_epe']=float((out['global_flow'][j,0]-target[j,0]).abs()[valid[j]].mean())
     item['correspondence']=correspondence_metrics({k:(v[j:j+1] if torch.is_tensor(v) and v.ndim>0 and v.shape[0]==left.shape[0] else v) for k,v in out.items()},target[j:j+1],valid[j:j+1])
     if name=='two_pass':baseline[key]=(out['flow'][j,0].cpu(),item['correspondence']['support_digest'])
     else:
      assert baseline[key][1]==item['correspondence']['support_digest'];item['mean_abs_prediction_change']=float((out['flow'][j,0].cpu()-baseline[key][0]).abs().mean())
     frames.append(item)
    driver.reset_model(model)
   scenes=sorted({x['scene'] for x in frames});scene={s:float(np.mean([x['epe'] for x in frames if x['scene']==s])) for s in scenes};n=sum(x['valid_pixels'] for x in frames)
   metrics=dict(mean_scene_epe=float(np.mean(list(scene.values()))),scene_epe=scene,bad3=sum(x['bad3']*x['valid_pixels'] for x in frames)/n,per_frame=frames)
   result['conditions'][name]=metrics
   if name=='two_pass':
    expected={(x['scene'],x['frame']):x for x in official['per_frame']};difference=max(abs(x['epe']-expected[(x['scene'],x['frame'])]['epe']) for x in frames);result['official_max_frame_epe_difference']=difference;result['official_parity']=difference<1e-5;save();assert result['official_parity']
   else:
    full=result['conditions']['two_pass'];metrics['delta_epe']=metrics['mean_scene_epe']-full['mean_scene_epe'];metrics['delta_bad3']=metrics['bad3']-full['bad3'];metrics['paired_scene_delta']={s:v-full['scene_epe'][s] for s,v in scene.items()}
   save();print(name,metrics['mean_scene_epe'],metrics['bad3'],flush=True)
 result['status']='COMPLETED';save()
if __name__=='__main__':main()
