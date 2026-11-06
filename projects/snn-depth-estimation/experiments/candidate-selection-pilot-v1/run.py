"""Bounded frozen-matcher candidate-selection pilot; no automatic escalation."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,sys,time
import torch
import torch.nn.functional as F
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--steps',type=int,default=1000);a=p.parse_args();a.root.mkdir(parents=True,exist_ok=True)
assert os.environ.get('SLURM_JOB_ID') and torch.cuda.device_count()==1
project=a.source/'projects/snn-depth-estimation';sys.path.insert(0,str(project));sys.path.insert(0,str(project/'experiments/shared-gpu-30k-v1'));sys.path.insert(0,str(project/'experiments/spiking-transformer-stereo/scripts'))
from common import atomic_json,append,flags
s=importlib.util.spec_from_file_location('driver',project/'experiments/spiking-transformer-stereo/scripts/run_experiment.py');d=importlib.util.module_from_spec(s);s.loader.exec_module(d)
from core.s2m2_spiking_v2 import SpikingS2M2V2
from core.utils.utils import InputPadder
from selector import Selector,candidates
assert hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()=='b38cb385945017bcfb3d7b00d2cb260011ae87236efcf12263cb92d0088df9a9'
torch.set_num_threads(6);torch.manual_seed(1234);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
manifest=json.loads((project/'experiments/spiking-transformer-stereo/configs/manifest.json').read_text());train=d.ManifestDataset(a.data_root,manifest['train']);dev=d.ManifestDataset(a.data_root,manifest['dev']);model=SpikingS2M2V2().cuda().eval();model.load_state_dict(torch.load(a.checkpoint,map_location='cpu',weights_only=False)['model']);model.requires_grad_(False)
heads={name:Selector(name=='spatial').cuda() for name in ('pointwise','spatial')};heads['spatial'].load_state_dict(heads['pointwise'].state_dict());opts={n:torch.optim.AdamW(h.parameters(),lr=2e-4,weight_decay=1e-5) for n,h in heads.items()};order=torch.randperm(960,generator=torch.Generator().manual_seed(1234)).tolist();native=model.disp_init.forward
registry=json.loads((a.source/'source-registry.json').read_text())
for rel,digest in registry['source_files'].items():assert hashlib.sha256((a.source/rel).read_bytes()).hexdigest()==digest
atomic_json(a.root/'metadata.json',dict(job=os.environ['SLURM_JOB_ID'],source_identity=registry['source_identity_sha256'],seed=1234,updates=a.steps,batch=2,checkpoint_sha256='b38cb385945017bcfb3d7b00d2cb260011ae87236efcf12263cb92d0088df9a9',frozen='entire v2b matcher/refiners',trainable=list(heads),candidate_definition='literal raw preOT valid-key top5',offset='none'))
def geometry(target,valid,pad):
 x=torch.arange(target.shape[-1],device='cuda')[None,None,None];v=valid[:,None].bool();inside=v&(x+target>=0)&(x+target<=target.shape[-1]-1);sample=F.pad(target,pad)[...,::4,::4][:,0];mask=F.pad(inside,pad)[...,::4,::4][:,0];return sample,mask,inside
@torch.no_grad()
def evaluate(step):
 allrows=[]
 for left,right,target,valid,indices in torch.utils.data.DataLoader(dev,batch_size=2,num_workers=0):
  left,right,target,valid=[v.cuda() for v in (left,right,target,valid)];pad=InputPadder(target.shape,divis_by=32)._pad;sample,mask,inside=geometry(target,valid,pad);cache={}
  for mode in ('native','top1','pointwise','spatial','oracle'):
   def intervene(features):
    disp,conf,occ,cv,down=native(features);output=dict(logits=model.disp_init.last_logits,probabilities=model.disp_init.last_probability);feat,keys=candidates(output);x=torch.arange(keys.shape[-2],device='cuda')[None,None,:,None];candidate=(x-keys)*4;err=(-candidate-sample[...,None]).abs()
    if mode=='native':cache.update(features=feat,keys=keys,error=err);return disp,conf,occ,cv,down
    if mode=='top1':choice=torch.zeros_like(keys[...,0])
    elif mode=='oracle':choice=torch.where(mask,err.argmin(-1),torch.zeros_like(keys[...,0]))
    else:choice=heads[mode](feat,model.disp_init.query_valid).argmax(-1)
    cache['choice']=choice;seed=candidate.gather(-1,choice[...,None]).squeeze(-1)/4
    return seed[:,None],conf,occ,cv,down
   model.disp_init.forward=intervene;out=model(left,right)
   for b,index in enumerate(indices.tolist()):
    row=dict(scene=dev.rows[index]['scene'],frame=dev.rows[index]['frame'],mode=mode)
    for scope,vm in [('full',valid[b,None]),('inframe',inside[b]),('out_of_frame',valid[b,None]&~inside[b]),('near',inside[b]&(target[b].abs()<32)),('far',inside[b]&(target[b].abs()>=32))]:
     error=(out['flow'][b]-target[b]).abs()[vm];row[scope+'_epe']=float(error.mean()) if error.numel() else None
    if mode=='native':
     kk=cache['keys'][b];sm=mask[b];dist=(kk[..., :,None]-kk[...,None,:]).abs();eye=torch.eye(5,device='cuda',dtype=torch.bool)
     row['adjacent_candidate_fraction']=float(((dist==1)&~eye).float()[sm].mean());row['within_one_bin_fraction']=float(((dist<=1)&~eye).any(-1).float()[sm].mean())
     # Offline alternate candidate definition: greedy suppress ±1 coarse bin.
     scores=out['logits'][b].clone();peakkeys=[];axis=torch.arange(scores.shape[-1],device='cuda')
     for _ in range(5):
      pk=scores.argmax(-1);peakkeys.append(pk);scores=scores.masked_fill((axis-pk[...,None]).abs()<=1,-1e4)
     peaks=torch.stack(peakkeys,-1);xx=torch.arange(peaks.shape[-2],device='cuda')[None,:,None];pe=(4*(peaks-xx)-sample[b,...,None]).abs()
     row['suppressed_peaks_oracle_epe']=float(pe.min(-1).values[sm].mean());row['suppressed_peaks_coverage4']=float((pe.min(-1).values[sm]<=4).float().mean())
    if mode!='native':
     e=cache['error'][b];sm=mask[b];chosen=e.gather(-1,cache['choice'][b,...,None]).squeeze(-1);oracle=e.min(-1).values;top=e[...,0]
     row.update(candidate_epe=float(chosen[sm].mean()),oracle_regret=float((chosen-oracle)[sm].mean()),gap_captured=float((top-chosen)[sm].sum()/(top-oracle)[sm].sum().clamp_min(1e-12)))
    allrows.append(row)
   d.reset_model(model)
  model.disp_init.forward=native
 metrics={}
 for mode in ('native','top1','pointwise','spatial','oracle'):
  rr=[r for r in allrows if r['mode']==mode];metrics[mode]={}
  for key in ('full_epe','inframe_epe','out_of_frame_epe','near_epe','far_epe','candidate_epe','oracle_regret','gap_captured'):
   scenevalues=[]
   for scene in sorted({r['scene'] for r in rr}):
    values=[r[key] for r in rr if r['scene']==scene and r.get(key) is not None]
    if values:scenevalues.append(sum(values)/len(values))
   if scenevalues:metrics[mode][key]=sum(scenevalues)/len(scenevalues)
 append(a.root/'evaluations.jsonl',dict(update=step,metrics_by_mode=metrics,per_frame=allrows));print('DEV',step,metrics,flush=True)
 if step==0:
  assert abs(metrics['native']['full_epe']-4.70301753282547)<1e-5
  for name in heads:assert abs(metrics[name]['full_epe']-metrics['top1']['full_epe'])<1e-6

def checkpoint(step):
 tmp=a.root/'last.tmp.pt';torch.save(dict(update=step,heads={n:h.state_dict() for n,h in heads.items()},optimizers={n:o.state_dict() for n,o in opts.items()},rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),order=order,source_identity=registry['source_identity_sha256']),tmp);tmp.replace(a.root/'last.pt');atomic_json(a.root/'checkpoint.json',dict(update=step,complete=True))
start=0
if (a.root/'last.pt').exists():
 ck=torch.load(a.root/'last.pt',weights_only=False);assert ck['source_identity']==registry['source_identity_sha256'];start=ck['update']
 for n in heads:heads[n].load_state_dict(ck['heads'][n]);opts[n].load_state_dict(ck['optimizers'][n])
 torch.set_rng_state(ck['rng']);torch.cuda.set_rng_state_all(ck['cuda_rng'])
else:evaluate(0);checkpoint(0)
for step in range(start,a.steps):
 items=[]
 for j in range(2):
  pos=step*2+j;item=list(train[order[pos%960]]);hf,vf=flags(pos)
  if hf:item[:4]=[v.flip(-1) for v in item[:4]];item[2]=-item[2]
  if vf:item[:4]=[v.flip(-2) for v in item[:4]]
  items.append(item)
 left,right,target,valid=[torch.stack([r[j] for r in items]).cuda() for j in range(4)]
 with torch.no_grad():
  out=model(left,right);feat,keys=candidates(out);sample,mask,_=geometry(target,valid,out['padding']);x=torch.arange(keys.shape[-2],device='cuda')[None,None,:,None];error=((keys-x)*4-sample[...,None]).abs();minimum=error.min(-1,keepdim=True).values;labels=(error<=minimum+1e-6).float();labels/=labels.sum(-1,keepdim=True)
 losses={}
 for name,head in heads.items():
  opts[name].zero_grad(set_to_none=True);logits=head(feat.detach(),out['query_valid']);loss=-(labels*logits.log_softmax(-1)).sum(-1)[mask].mean();assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(head.parameters(),1.);assert torch.isfinite(norm) and norm>0;
  if step==0:assert head.out.weight.grad.abs().sum()>0
  if step==1:assert head.embed.weight.grad.abs().sum()>0
  opts[name].step();losses[name]=float(loss)
 d.reset_model(model);append(a.root/'train.jsonl',dict(update=step+1,loss=sum(losses.values()),loss_by_head=losses));atomic_json(a.root/'status.json',dict(phase='training',update=step+1,job=os.environ['SLURM_JOB_ID']))
 if (step+1)%50==0:checkpoint(step+1)
 if step+1 in (250,a.steps):evaluate(step+1);checkpoint(step+1)
atomic_json(a.root/'result.json',dict(status='COMPLETED',updates=a.steps));atomic_json(a.root/'status.json',dict(phase='completed',update=a.steps,job=os.environ['SLURM_JOB_ID']))
