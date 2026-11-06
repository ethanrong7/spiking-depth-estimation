"""Detached same-forward final-eval readout descriptions, never a GT model input."""
import math
import torch
MEASUREMENTS=('matching_error_pixels','post_residual_coarse_error_pixels','native_center_error_pixels','signed_residual_pixels','absolute_residual_pixels','residual_correction_gain_pixels','normalized_entropy','padded_right_probability_mass','GT_neighborhood_probability_mass')
POPULATIONS=('all_valid_centers','native_right_inbounds','native_right_out_of_FOV')

@torch.no_grad()
def frame_summaries(flow,target,valid,fields,source_rows):
 if flow.shape!=target.shape or tuple(flow.shape[1:])!=(1,250,400)or len(source_rows)!=flow.shape[0]:raise RuntimeError('native diagnostic geometry differs')
 if set(fields)!={'d0','r','d','p'}:raise RuntimeError('readout fields differ')
 if any(value.requires_grad or value.dtype!=torch.float32 for value in fields.values()):raise RuntimeError('readout fields must be detached FP32')
 b=flow.shape[0]
 if any(tuple(fields[k].shape)!=(b,1,64,104)for k in ['d0','r','d'])or tuple(fields['p'].shape)!=(b,64,104,104):raise RuntimeError('quarter-lattice readout geometry differs')
 d0,r,d=[fields[k][:,0]for k in ['d0','r','d']];p=fields['p']
 if not torch.equal(d,d0+r)or not all(torch.isfinite(x).all()for x in [d0,r,d,p])or not bool((p>=0).all()):raise RuntimeError('invalid actual head fields')
 y=torch.arange(64,device=flow.device)*4-3;x=torch.arange(104,device=flow.device)*4-8;j=torch.arange(104,device=flow.device);k=j
 if not bool((p[...,k[None,:]>j[:,None]]==0).all())or not torch.allclose(p.sum(-1),torch.ones_like(d),atol=1e-6,rtol=1e-5):raise RuntimeError('head support/probability differs')
 inside=(y[:,None]>=0)&(y[:,None]<250)&(x[None,:]>=0)&(x[None,:]<400);yy=y.clamp(0,249);xx=x.clamp(0,399)
 g=-target[:,0][:,yy[:,None],xx[None,:]];selected=valid[:,yy[:,None],xx[None,:]]&inside[None];right=x[None,None,:]-g;inbounds=selected&(right>=0)&(right<=399);out=selected&~inbounds;outside_head=(g/4<0)|(g/4>j[None,None,:]);native_error=(flow[:,0][:,yy[:,None],xx[None,:]]+g).abs();e0=(4*d0-g).abs();ec=(4*d-g).abs()
 denominator=(j+1).float().log();entropy=-torch.xlogy(p,p).sum(-1)/torch.where(denominator>0,denominator,torch.ones_like(denominator))[None,None];entropy[...,0]=0
 key_native=k*4-8;key_valid=(key_native>=0)&(key_native<=399);padded_mass=p[...,~key_valid].sum(-1);qgt=j[None,None,:]-g/4;near=(k[None,None,None,:]-qgt[...,None]).abs()<=1;GT_mass=(p*near*key_valid).sum(-1)
 values={'matching_error_pixels':e0,'post_residual_coarse_error_pixels':ec,'native_center_error_pixels':native_error,'signed_residual_pixels':4*r,'absolute_residual_pixels':4*r.abs(),'residual_correction_gain_pixels':e0-ec,'normalized_entropy':entropy,'padded_right_probability_mass':padded_mass,'GT_neighborhood_probability_mass':GT_mass}
 output=[]
 for batch,row in enumerate(source_rows):
  populations={}
  for name,mask in zip(POPULATIONS,[selected,inbounds,out]):
   chosen=mask[batch];count=int(chosen.sum());mass_chosen=chosen&inbounds[batch];mass_count=int(mass_chosen.sum());metrics={}
   for key,value in values.items():
    keep=mass_chosen if key=='GT_neighborhood_probability_mass'else chosen;data=value[batch][keep]
    if data.numel()and not torch.isfinite(data).all():raise RuntimeError('nonfinite readout diagnostic')
    metrics[key]=float(data.mean())if data.numel()else None
   populations[name]={'site_count':count,'head_support_outside_count':int((chosen&outside_head[batch]).sum()),'GT_neighborhood_mass_eligible_sites':mass_count,'metrics':metrics}
  output.append({'scene':row['scene'],'frame':row['frame'],'populations':populations})
 return output

def average_records(records,unit):
 # Counts pool; measurement means give each available input record equal weight.
 answer={}
 for population in POPULATIONS:
  rows=[record['populations'][population]for record in records];metrics={};contributors={}
  for key in MEASUREMENTS:
   values=[r['metrics'][key]for r in rows if r['metrics'][key]is not None];metrics[key]=sum(values)/len(values)if values else None;contributors[key]=len(values)
  answer[population]={'site_count':sum(r['site_count']for r in rows),'head_support_outside_count':sum(r['head_support_outside_count']for r in rows),'GT_neighborhood_mass_eligible_sites':sum(r['GT_neighborhood_mass_eligible_sites']for r in rows),'contributing_'+unit:sum(r['site_count']>0 for r in rows),'metric_contributing_'+unit:contributors,'metrics':metrics}
 return answer

def aggregate(frames):
 keys=[(r['scene'],r['frame'])for r in frames]
 if len(keys)!=len(set(keys)):raise RuntimeError('duplicate readout diagnostic frame')
 scenes=[]
 for scene in sorted({r['scene']for r in frames}):
  selected=[r for r in frames if r['scene']==scene];scenes.append({'scene':scene,'frames':len(selected),'populations':average_records(selected,'frames')})
 return {'scope':'same final5000 Dev forwards only; detached descriptions, no new gate/pass/mask; notbinocularcausality/confidence/motion','units':'allerrorsandresidualsnativepixels; probabilitiesunitless; exactquartercenters(4i-3,4j-8)','population_notes':'native right FOV and numerical padded head support are distinct; no visibilitycertificate','per_frame':frames,'per_scene':scenes,'aggregate_equal_scene':average_records(scenes,'scenes'),'frame_count':len(frames),'scene_count':len(scenes),'aggregation':'equal available frame means within each scene, then equal available scene means; counts separately pooled; empties retained/null'}
