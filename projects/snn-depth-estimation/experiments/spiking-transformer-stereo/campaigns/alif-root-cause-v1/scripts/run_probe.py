"""P0: frozen50k checkpoints at16/32 on exact fullTrain and Dev, Slurm GPUs only."""
import argparse
from datetime import datetime,timezone
import hashlib,json,os,socket,sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

CAMPAIGN=Path(__file__).resolve().parents[1]
EXPERIMENT=CAMPAIGN.parents[1]
sys.path.insert(0,str(EXPERIMENT/'scripts'))
import run_experiment as runner


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as handle:
  for block in iter(lambda:handle.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()


def atomic(path,value):
 temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def main():
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True)
 for name in ['registration','manifest','data-root','source-registry','output']:p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args()
 if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available() or torch.cuda.device_count()!=1 or '5090' not in torch.cuda.get_device_name(0):raise RuntimeError('requires one coordinated RTX5090 Slurm GPU per worker')
 reg=json.loads(a.registration.read_text());manifest=json.loads(a.manifest.read_text());source=json.loads(a.source_registry.read_text())
 assert a.seed in reg['paired_seeds'] and sha(a.manifest)==reg['manifest_sha256']
 assert source['source_override']['parent_source_identity_sha256']==reg['parent_source_identity_sha256']
 assert {k:v for k,v in source['source_files'].items() if '/core/' in k}==reg['frozen_core']
 for relative,digest in source['source_files'].items():
  assert sha(runner.PROJECT.parents[1]/relative)==digest,relative
 for split,info in reg['splits'].items():assert len(manifest[split])==info['frames'] and len({row['scene'] for row in manifest[split]})==info['scenes']
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
 a.output.mkdir(parents=True,exist_ok=False)
 result=dict(status='RUNNING',phase='P0',seed=a.seed,source=source,registration_sha256=sha(a.registration),manifest_sha256=sha(a.manifest),checkpoints=[],conditions=[],reproduction={},slurm={k:v for k,v in os.environ.items() if k.startswith('SLURM_')},gpu=torch.cuda.get_device_name(0),torch=torch.__version__,hostname=socket.gethostname(),command=sys.argv)
 def save():
  result['updated_at']=datetime.now(timezone.utc).isoformat();atomic(a.output/'result.json',result)
 loaders={split:DataLoader(runner.ManifestDataset(a.data_root,manifest[split]),batch_size=reg['batch'],shuffle=False,num_workers=0,generator=torch.Generator().manual_seed(a.seed+991)) for split in reg['splits']}
 save()
 for target in [x for x in reg['checkpoints'] if x['seed']==a.seed]:
  checkpoint_path=Path(target['path']);assert sha(checkpoint_path)==target['sha256']
  ck=torch.load(checkpoint_path,map_location='cpu',weights_only=False);config=ck['config']
  assert ck['updates']==50000 and ck['samples_seen']==100000 and config==target['config']
  assert ck['manifest_sha256']==reg['manifest_sha256'] and ck['source']['source_identity_sha256']==target['source_identity_sha256']==reg['parent_source_identity_sha256']
  assert {k:v for k,v in ck['source']['source_files'].items() if '/core/' in k}==reg['frozen_core']
  for name in ['run_experiment.py','run_provenance.py']:
   relative='projects/snn-depth-estimation/experiments/spiking-transformer-stereo/scripts/'+name;assert source['source_files'][relative]==ck['source']['source_files'][relative]
  model=runner.build_model(config['model'],a.seed,config.get('normalize_residual_current',False)).cuda().eval();model.load_state_dict(ck['model'],strict=True)
  result['checkpoints'].append(target);save()
  # The first and only Dev32 evaluation is also the exact reproduction gate.
  order=[('dev',32),('dev',16),('train',16),('train',32)]
  for split,count in order:
   metrics=runner.evaluate(model,loaders[split],count)
   if split=='dev' and count==32:
    expected=next(x for x in json.loads((checkpoint_path.parent/'history.json').read_text()) if x['update']==50000)
    difference=abs(metrics['mean_scene_epe']-expected['mean_scene_epe']);scene_difference=max(abs(metrics['scene_epe'][s]-v) for s,v in expected['scene_epe'].items())
    result['reproduction'][config['model']]=dict(mean_epe_abs_difference=difference,max_scene_epe_abs_difference=scene_difference,tolerance_px=reg['reproduction']['absolute_tolerance_px'])
    save()
    if max(difference,scene_difference)>reg['reproduction']['absolute_tolerance_px']:raise RuntimeError('Dev32 reproduction failed; stop interpretation before other conditions')
   cell=dict(model=config['model'],seed=a.seed,split=split,iterations=count,expected_frames=len(manifest[split]),completed_frames=len(metrics['per_frame']),complete=True,metrics=metrics)
   assert cell['completed_frames']==cell['expected_frames'];result['conditions'].append(cell);save();print(a.seed,config['model'],split,count,metrics['mean_scene_epe'],flush=True)
  runner.reset_model(model);del model,ck;torch.cuda.empty_cache()
 assert sum(c['completed_frames'] for c in result['conditions'])==4080
 result['status']='COMPLETED';save()


if __name__=='__main__':main()
