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
 for i in range(len(dataset)):
  left,right,target,valid,_=dataset[i];left,right,target,valid=[v[None].cuda() for v in (left,right,target,valid)];pad=InputPadder(target.shape,divis_by=32)._pad
  v=valid[:,None].bool();x=torch.arange(target.shape[-1],device='cuda')[None,None,None];inside=v&(x+target>=0)&(x+target<=target.shape[-1]-1)
  sampled=F.pad(target,pad)[...,::4,::4][:,0];mask=F.pad(inside,pad)[...,::4,::4][:,0]
  for mode in ('native','local1','temperature2','temperature4','rowlocal1','oracle'):
   def intervention(features):
    disp,conf,occ,cv,down=native(features);prob=m.disp_init.last_probability;keys=torch.arange(prob.shape[-1],device=prob.device,dtype=prob.dtype)
    if mode=='native':return disp,conf,occ,cv,down
    q=prob/prob.sum(-1,keepdim=True).clamp_min(1e-12)
    if mode=='rowlocal1':q=m.disp_init.last_logits.softmax(-1)*m.disp_init.key_valid[...,None,:]
    if mode in ('local1','rowlocal1'):q=q*((keys-q.argmax(-1)[...,None]).abs()<=1)
    if mode.startswith('temperature'):q=q.pow(int(mode[-1]))
    position=(q*keys).sum(-1)/q.sum(-1).clamp_min(1e-12);new=(keys-position)[:,None]
    if mode=='oracle':new=torch.where(mask[:,None],-sampled[:,None]/4,disp)
    return new,conf,occ,cv,down
   m.disp_init.forward=intervention;out=m(left,right)
   row=dict(index=i,scene=dataset.rows[i]['scene'],frame=dataset.rows[i]['frame'],mode=mode)
   for scope,scope_mask in [('valid',v),('inframe',inside)]:
    for name in ('flow','initial_flow','global_flow'):row[scope+'_'+name+'_epe']=float((out[name]-target).abs()[scope_mask].mean())
   if mode=='native':
    gt=torch.arange(sampled.shape[-1],device='cuda')[None,None]+sampled/4
    for score_name,score in [('pre',out['logits']),('post',out['probabilities'])]:
     nearest=gt.round().long().clamp(0,score.shape[-1]-1);truth=score.gather(-1,nearest[...,None]).squeeze(-1);rank=(score>truth[...,None]).sum(-1)+1
     candidates=score.topk(5,-1).indices;err=(candidates-gt[...,None]).abs()*4
     estimates={'top1':err[...,0],'oracle_top5':err.min(-1).values,'rank':rank.float(),'top5_hit':(rank<=5).float()}
     for split,sm in [('all',mask),('low',mask&(sampled.abs()<32)),('high',mask&(sampled.abs()>=32))]:
      row[score_name+'_'+split+'_count']=int(sm.sum())
      for name,value in estimates.items():row[score_name+'_'+split+'_'+name]=float(value[sm].mean()) if sm.any() else None
   rows.append(row);d.reset_model(m)
  m.disp_init.forward=native
  a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(dict(checkpoint=str(a.checkpoint),frames=rows),indent=2));print(i,rows[-6:],flush=True)
