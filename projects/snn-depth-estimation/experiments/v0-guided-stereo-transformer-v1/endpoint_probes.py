"""Eval-only registered C30k panel; immutable training source is never edited."""
import argparse,hashlib,importlib.util,json,os,sys,types,time
from pathlib import Path
import torch

def permutation(content,valid,seed):
 """Bijection on valid content slots; geometry/invalid slots stay fixed."""
 generator=torch.Generator(device=content.device).manual_seed(seed)
 n=content.shape[-1];slot=torch.arange(n,device=content.device)
 order=torch.argsort(torch.rand(content.shape,device=content.device,generator=generator).masked_fill(~valid,2)+slot*1e-8,dim=-1)
 rank=(valid.long().cumsum(-1)-1).clamp_min(0)
 donor=order.gather(-1,rank)
 return torch.where(valid,content.gather(-1,donor),content)

def attention_statistics(head,readout,corr,disp,width):
 b,h,w,_=readout.shape;n=len(head.offsets);scale=(2**head.level_ids)[None,None,None,:]
 center=torch.arange(w,device=disp.device)[None,None,:,None]+disp[:,0].unsqueeze(-1)
 valid=(center/scale+(head.offsets/(2**head.level_ids))[None,None,None,:]>=0)&(center/scale+(head.offsets/(2**head.level_ids))[None,None,None,:]<=torch.div(torch.tensor(width,device=disp.device),scale,rounding_mode='floor')-1)
 content=corr.permute(0,2,3,1);tokens=torch.stack((content,head.offsets.expand_as(content)/32,head.level_ids.expand_as(content)/3,valid.to(content)),-1)
 q=(head.query_norm(head.query(readout))>=.5).to(readout);k=(head.key_norm(head.key(tokens))>=.5).to(readout)
 scores=(q.reshape(b,h,w,4,4).unsqueeze(3)*k.reshape(b,h,w,n,4,4)).sum(-1).permute(0,1,2,4,3)/2
 weights=scores.masked_fill(~valid.unsqueeze(3),-1e9).softmax(-1)*valid.unsqueeze(3);weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-12)
 return valid,dict(entropy=-(weights*weights.clamp_min(1e-12).log()).sum(-1).mean(-1),q_rate=q.mean(-1),k_rate=(k.mean(-1)*valid).sum(-1)/valid.sum(-1).clamp_min(1))

class Recorder:
 def __init__(self,model,truth,groups,intervention,batch_identity):
  self.block=model.update_block;self.head=self.block.guided_attention;self.truth=truth;self.groups=groups;self.intervention=intervention;self.batch_identity=batch_identity;self.stage=0;self.rows={};self.handles=[]
 def before(self,module,args,kwargs):
  self.stage+=1;self.chunks=[];self.before_flow=kwargs.get('disp',args[3] if len(args)>3 else None)[:,0].detach()
 def chunk(self,head,readout,corr,disp,start,width):
  valid,_=attention_statistics(head,readout,corr,disp,width)
  if self.intervention=='permuted':
   # Only a private copy of the extra key's scalar changes. Native corr is intact.
   content=permutation(corr.permute(0,2,3,1),valid,9107+self.batch_identity*100000+self.stage*100+start)
   corr=content.permute(0,3,1,2).contiguous()
  _,stats=attention_statistics(head,readout,corr,disp,width);self.chunks.append(stats)
  return self.original_chunk(readout,corr,disp,start,width)
 def head_after(self,module,args,output):self.branch=output[:,0].detach()
 def native_after(self,module,args,output):self.native=output[:,0].detach().clone()
 def after(self,module,args,output):
  total=output[3][:,0].detach();native=self.native;remaining=self.truth-self.before_flow-native
  stats={k:torch.cat([x[k] for x in self.chunks],1) for k in self.chunks[0]}
  values=dict(branch_sq=self.branch.square(),native_sq=native.square(),branch_abs=self.branch.abs(),branch_signed=self.branch,remaining_signed=remaining,remaining_abs=remaining.abs(),branch_remaining_abs_error=(self.branch-remaining).abs(),post_total_abs_error=(self.truth-self.before_flow-total).abs(),sign_agreement=(self.branch*remaining>0).float(),**stats)
  for group,mask in self.groups.items():
   names=list(values);vector=torch.stack([mask.sum().float()]+[v[mask].sum() for v in values.values()]).cpu().tolist();row=self.rows.setdefault(str(self.stage),{}).setdefault(group,dict(count=0,**{k+'_sum':0. for k in names}));row['count']+=int(vector[0])
   for k,value in zip(names,vector[1:]):row[k+'_sum']+=value
 def __enter__(self):
  self.original_chunk=self.head._chunk;self.previous_override=self.head.override;self.head.override='off' if self.intervention=='off' else None
  self.head._chunk=types.MethodType(self.chunk,self.head)
  self.handles=[self.block.register_forward_pre_hook(self.before,with_kwargs=True),self.head.register_forward_hook(self.head_after),self.block.disp_head.register_forward_hook(self.native_after),self.block.register_forward_hook(self.after)];return self
 def __exit__(self,*unused):
  self.head._chunk=self.original_chunk;self.head.override=self.previous_override
  for handle in self.handles:handle.remove()

def merge_stats(target,source):
 for stage,groups in source.items():
  for group,row in groups.items():
   out=target.setdefault(stage,{}).setdefault(group,{k:0 for k in row})
   for key,value in row.items():out[key]+=value

def main():
 p=argparse.ArgumentParser()
 for key in ('source','run-root','data-root','output'):p.add_argument('--'+key,type=Path,required=True)
 p.add_argument('--shared-lock-root',type=Path);p.add_argument('--expected-gpu-uuid')
 a=p.parse_args();assert os.environ.get('SLURM_JOB_ID') and torch.cuda.device_count()==1
 if a.expected_gpu_uuid:assert str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-')==a.expected_gpu_uuid.removeprefix('GPU-')
 torch.set_num_threads(6);torch.cuda.set_per_process_memory_fraction(.20 if a.shared_lock_root else .80)
 import cv2;cv2.setNumThreads(1)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
 project=a.source/'projects/snn-depth-estimation';sys.path[:0]=[str(project/'experiments/v0-guided-stereo-transformer-v1'),str(project/'experiments/shared-gpu-30k-v1')]
 import recipe
 from common import phase_lock
 from torch.nn import functional as F
 driver,mechanism,_=recipe.load_frozen(a.source/'historical')
 from core.utils.utils import InputPadder
 registry=json.loads((a.source/'source-registry.json').read_text());assert registry['source_identity_sha256']=='fb2799c40cc1a5db9535eb05073b410195d3cf19b1bdfaaf7e7266fdccaab996'
 for name,digest in registry['source_files'].items():assert hashlib.sha256((a.source/name).read_bytes()).hexdigest()==digest
 folder=a.run_root/'train/C';assert json.loads((folder/'result.json').read_text())['updates']==30000
 checkpoint=folder/'step-30000.pt';ck=torch.load(checkpoint,map_location='cpu',weights_only=False);assert ck['updates']==ck['scheduler']['last_epoch']==30000 and ck['source_identity']==registry['source_identity_sha256']
 spec=importlib.util.spec_from_file_location('guided',project/'core/guided_stereo_attention.py');guided=importlib.util.module_from_spec(spec);spec.loader.exec_module(guided)
 official=[json.loads(x) for x in (folder/'evaluations.jsonl').read_text().splitlines() if json.loads(x)['update']==30000];assert len(official)==1 and len(official[0]['per_frame'])==60;expected={(r['scene'],r['frame']):r for r in official[0]['per_frame']}
 manifest=json.loads((project/'experiments/spiking-transformer-stereo/configs/manifest.json').read_text());dataset=driver.ManifestDataset(a.data_root,manifest['dev']);loader=torch.utils.data.DataLoader(dataset,batch_size=2,num_workers=0,generator=torch.Generator().manual_seed(2225))
 report=dict(status='RUNNING',checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),source_identity=registry['source_identity_sha256'],diagnostic_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),job=os.environ['SLURM_JOB_ID'],step=os.environ.get('SLURM_STEP_ID'),gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),physical_batch=2,iterations=32,shared_lock_root=str(a.shared_lock_root) if a.shared_lock_root else None,allocator_fraction=.20 if a.shared_lock_root else .80,precision='FP32;cuDNN TF32 yes;matmulTF32 no',gate=float(ck['model']['update_block.guided_attention.gate']),gate_tanh=float(ck['model']['update_block.guided_attention.gate'].tanh()),parity=[],conditions={},units='Native errors px; branch/native coarse deltas and errors featurecells; multiply by4 for px',permutation='privategenerator seed9107+batch_identity*100000+iteration*100+rowstart; bijection on valid scalar slots only',flip='Horizontal diagnostic mirrors both views/valid and negates target; canonical official support unchanged')
 def save():
  a.output.parent.mkdir(parents=True,exist_ok=True);temporary=a.output.with_suffix('.tmp');temporary.write_text(json.dumps(report,indent=2,allow_nan=False));temporary.replace(a.output)
 class Lease:
  def __enter__(self):
   self.context=phase_lock(a.shared_lock_root or a.run_root/'train','C-endpoint-probes',True,torch.cuda.empty_cache);self.context.__enter__();return self
  def __exit__(self,*args):
   context=getattr(self,'context',None);self.context=None
   if context:return context.__exit__(*args)
  def rotate(self):
   self.__exit__(None,None,None);time.sleep(.2);self.__enter__()
 with Lease() as lease,torch.no_grad():
  assert torch.cuda.mem_get_info()[0]>4*1024**3
  model=guided.install_guidance(recipe.build_model(driver,mechanism,'spiking_normalized',1234)).cuda().eval();model.load_state_dict(ck['model']);del ck
  for batch_index,(left,right,target,valid,indices) in enumerate(loader):
   if torch.cuda.mem_get_info()[0]<4*1024**3:raise RuntimeError('Probe memory guard: <4GiB free; checkpoint remains intact')
   left,right,target,valid=[x.cuda() for x in (left,right,target,valid)]
   reference=driver.forward(model,left,right,32);driver.reset_model(model)
   for j,index in enumerate(indices.tolist()):
    identity=dataset.rows[index];assert int(valid[j].sum())==expected[identity['scene'],identity['frame']]['valid_pixels'];epe=float((reference[j,0]-target[j,0]).abs()[valid[j]].mean());difference=abs(epe-expected[identity['scene'],identity['frame']]['epe']);assert difference<1e-5
    report['parity'].append(dict(scene=identity['scene'],frame=identity['frame'],official_epe_difference=difference))
   for flip in (False,True):
    l,r,t,v=(left,right,target,valid) if not flip else (left.flip(-1),right.flip(-1),-target.flip(-1),valid.flip(-1))
    padding=InputPadder(l.shape,divis_by=32)._pad;truth=F.pad(t,padding)[:,0,::4,::4]/4;coarse=F.pad(v[:,None],padding)[:,0,::4,::4]
    x=torch.arange(t.shape[-1],device=t.device)[None,None,:];native_truth=t[:,0];visible=(x+native_truth>=0)&(x+native_truth<=t.shape[-1]-1)
    border=(x<32)|(x>=t.shape[-1]-32)
    native_groups=dict(valid=v,inview=v&visible,oov=v&~visible,negative_flow=v&(native_truth<0),positive_flow=v&(native_truth>0),border=v&border,interior=v&~border,tail64=v&(native_truth.abs()>64),tail128=v&(native_truth.abs()>128))
    groups={k:F.pad(mask[:,None],padding)[:,0,::4,::4] for k,mask in native_groups.items()}
    for intervention in ('native','off','permuted'):
     key=('hflip_' if flip else 'canonical_')+intervention;condition=report['conditions'].setdefault(key,dict(per_frame=[],stages={}))
     with Recorder(model,truth,groups,intervention,batch_index*2+int(flip)) as recorder:prediction=driver.forward(model,l,r,32)
     driver.reset_model(model);assert torch.isfinite(prediction).all()
     if not flip and intervention=='native':assert float((prediction-reference).abs().max())<1e-5
     merge_stats(condition['stages'],recorder.rows)
     error=(prediction[:,0]-t[:,0]).abs()
     for j,index in enumerate(indices.tolist()):
      identity=dataset.rows[index];row=dict(scene=identity['scene'],frame=identity['frame'],groups={})
      for group,mask in native_groups.items():
       m=mask[j];row['groups'][group]=dict(count=int(m.sum()),error_sum=float(error[j][m].sum()),bad1_count=int(((error[j]>1)&m).sum()),bad3_count=int(((error[j]>3)&m).sum()),bad5_count=int(((error[j]>5)&m).sum()))
      condition['per_frame'].append(row)
   save();torch.cuda.empty_cache()
   if a.shared_lock_root and batch_index+1<len(loader):lease.rotate()
  for condition in report['conditions'].values():
   condition['per_scene']=[]
   for scene in sorted({r['scene'] for r in condition['per_frame']}):
    rows=[r['groups']['valid'] for r in condition['per_frame'] if r['scene']==scene];condition['per_scene'].append(dict(scene=scene,epe=sum(r['error_sum']/r['count'] for r in rows)/len(rows),bad3=sum(100*r['bad3_count']/r['count'] for r in rows)/len(rows)))
   condition['mean_scene_epe']=sum(r['epe'] for r in condition['per_scene'])/len(condition['per_scene']);condition['mean_scene_bad3']=sum(r['bad3'] for r in condition['per_scene'])/len(condition['per_scene'])
  for key,condition in report['conditions'].items():
   native=report['conditions']['hflip_native' if key.startswith('hflip_') else 'canonical_native']
   condition['epe_delta_vs_same_transform_native']=condition['mean_scene_epe']-native['mean_scene_epe']
   for stage,groups in condition['stages'].items():
    for group,row in groups.items():
     n=row['count'];row['branch_rms_feature_cells']=(row['branch_sq_sum']/n)**.5 if n else None;row['native_rms_feature_cells']=(row['native_sq_sum']/n)**.5 if n else None
     row['branch_to_native_rms_ratio']=(row['branch_sq_sum']/row['native_sq_sum'])**.5 if row['native_sq_sum'] else None
     for field in ('entropy','q_rate','k_rate','remaining_abs','branch_remaining_abs_error','post_total_abs_error'):row[field+'_mean']=row[field+'_sum']/n if n else None
  report.update(status='COMPLETED',peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20);save()
 print(str(a.output),flush=True)
if __name__=='__main__':main()
