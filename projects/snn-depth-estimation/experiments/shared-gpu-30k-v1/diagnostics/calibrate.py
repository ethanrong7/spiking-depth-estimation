"""Frozen native checkpoint decoder interventions; execute only in Slurm GPU allocation."""
import argparse, hashlib, importlib.util, json, os, sys
from pathlib import Path
import torch
import torch.nn.functional as F
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
assert os.environ.get('SLURM_JOB_ID') and torch.cuda.device_count()==1
assert hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()=='b38cb385945017bcfb3d7b00d2cb260011ae87236efcf12263cb92d0088df9a9'
project=a.source/'projects/snn-depth-estimation';sys.path.insert(0,str(project))
f=project/'experiments/spiking-transformer-stereo/scripts/run_experiment.py';sys.path.insert(0,str(f.parent));s=importlib.util.spec_from_file_location('driver',f);d=importlib.util.module_from_spec(s);s.loader.exec_module(d)
from core.s2m2_spiking_v2 import SpikingS2M2V2
from core.s2m2_spiking import fixed_native_initial
from core.utils.utils import InputPadder
manifest=json.loads((project/'experiments/spiking-transformer-stereo/configs/manifest.json').read_text());dataset=d.ManifestDataset(a.data_root,manifest['dev'])
torch.set_num_threads(6);torch.manual_seed(1234);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
m=SpikingS2M2V2().cuda().eval();m.load_state_dict(torch.load(a.checkpoint,map_location='cpu',weights_only=False)['model']);native=m.disp_init.forward
rows=[]
with torch.no_grad():
 for left,right,target,valid,indices in torch.utils.data.DataLoader(dataset,batch_size=2,num_workers=0):
  left,right,target,valid=[v.cuda() for v in (left,right,target,valid)];out=m(left,right)
  for b,index in enumerate(indices.tolist()):
   error=(out['flow'][b,0]-target[b,0]).abs()[valid[b]];rows.append(dict(index=index,scene=dataset.rows[index]['scene'],epe=float(error.mean())))
  d.reset_model(m)
scenes={s:sum(r['epe'] for r in rows if r['scene']==s)/sum(r['scene']==s for r in rows) for s in {r['scene'] for r in rows}}
metric=sum(scenes.values())/len(scenes);assert abs(metric-4.70301753282547)<1e-5,metric
a.output.write_text(json.dumps(dict(mean_scene_epe=metric,frames=rows),indent=2));print(metric)
