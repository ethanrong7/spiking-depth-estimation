"""P0: frozen50k checkpoints at16/32 on exact fullTrain and Dev, Slurm GPUs only."""
import argparse
from datetime import datetime,timezone
import hashlib,json,os,socket,sys,subprocess
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
 result['runtime']=dict(cpu_initial_seed=torch.initial_seed(),cuda_initial_seed=torch.cuda.initial_seed(),CUBLAS_WORKSPACE_CONFIG=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),default_dtype=str(torch.get_default_dtype()),float32_matmul_precision=torch.get_float32_matmul_precision(),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),device_inventory=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,driver_version','--format=csv,noheader'],text=True),torch_config=torch.__config__.show())
 def save():
  result['updated_at']=datetime.now(timezone.utc).isoformat();atomic(a.output/'result.json',result)
 loaders={split:DataLoader(runner.ManifestDataset(a.data_root,manifest[split]),batch_size=reg['batch'],shuffle=False,num_workers=0,generator=torch.Generator().manual_seed(a.seed+991)) for split in reg['splits']}
 sample=next(iter(loaders['dev']))
 result['input_layout']=[dict(shape=list(x.shape),stride=list(x.stride()),dtype=str(x.dtype),contiguous=x.is_contiguous()) for x in sample[:4]];del sample
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
  expected=next(x for x in json.loads((checkpoint_path.parent/'history.json').read_text()) if x['update']==50000)
  assert expected['mean_scene_epe']==target['final_dev32_mean_scene_epe']
  runs=[]
  def differences(first, second):
   frames={(x['scene'],str(x['frame'])):x for x in second['per_frame']}
   return dict(metric_delta={key:first[key]-second[key] for key in ['mean_scene_epe','pooled_epe','bad1','bad3','bad5']},mean_scene_epe=first['mean_scene_epe']-second['mean_scene_epe'],scene_epe={s:v-second['scene_epe'][s] for s,v in first['scene_epe'].items()},per_frame=[dict(scene=x['scene'],frame=x['frame'],epe_delta=x['epe']-frames[(x['scene'],str(x['frame']))]['epe'],bad3_delta=x['bad3']-frames[(x['scene'],str(x['frame']))]['bad3']) for x in first['per_frame']])
  for repeat in [1,2]:
   metrics=runner.evaluate(model,loaders['dev'],32);runs.append(metrics)
   result['conditions'].append(dict(model=config['model'],seed=a.seed,split='dev',iterations=32,repeat=repeat,expected_frames=60,completed_frames=len(metrics['per_frame']),complete=True,metrics=metrics,difference_vs_parent=differences(metrics,expected)))
   save();print(a.seed,config['model'],'dev32',repeat,metrics['mean_scene_epe'],flush=True)
  result['reproduction'][config['model']]=dict(repeat2_minus_repeat1=differences(runs[1],runs[0]),state_layout={name:dict(shape=list(value.shape),stride=list(value.stride()),dtype=str(value.dtype),contiguous=value.is_contiguous()) for name,value in model.state_dict().items()})
  save()
  runner.reset_model(model);del model,ck;torch.cuda.empty_cache()
 assert sum(c['completed_frames'] for c in result['conditions'])==240
 result['status']='COMPLETED';save()


if __name__=='__main__':main()
