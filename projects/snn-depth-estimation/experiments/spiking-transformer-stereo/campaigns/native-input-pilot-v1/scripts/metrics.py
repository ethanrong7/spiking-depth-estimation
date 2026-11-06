"""Full source-valid native disparity metrics; <512 restriction is secondary only."""
import statistics
import torch


def frame_metrics(prediction,target,valid):
 error=(prediction-target).abs()[valid]
 count=error.numel()
 return dict(valid_pixels=count,error_sum=float(error.sum()),epe=float(error.mean()) if count else None,**{f'bad{k}_count':int((error>k).sum()) for k in [1,3,5]},**{f'bad{k}':100*float((error>k).float().mean()) if count else None for k in [1,3,5]})


def aggregate(rows):
 count=sum(r['valid_pixels'] for r in rows);scenes={s:statistics.mean(r['epe'] for r in rows if r['scene']==s and r['valid_pixels']) for s in sorted({r['scene'] for r in rows if r['valid_pixels']})}
 return dict(mean_scene_epe=statistics.mean(scenes.values()) if scenes else None,pooled_epe=sum(r['error_sum'] for r in rows)/count if count else None,valid_pixels=count,scene_epe=scenes,per_frame=rows,**{f'bad{k}':100*sum(r[f'bad{k}_count'] for r in rows)/count if count else None for k in [1,3,5]},median_scene_epe=statistics.median(scenes.values()) if scenes else None,worst5_scene_epe=sorted(scenes.items(),key=lambda p:p[1],reverse=True)[:5])


@torch.no_grad()
def evaluate(model,name,dataset,forward_fn,reset_fn,iterations=32):
 model.eval();primary=[];secondary=[];coverage=[]
 for i in range(len(dataset)):
  item=dataset[i];left,right,target,valid=[item[k].unsqueeze(0).cuda() for k in ['left','right','target','valid']]
  prediction=forward_fn(model,name,left,right,iterations,False)
  if not torch.isfinite(prediction).all() or prediction.shape!=target.shape:raise FloatingPointError('invalid native evaluation prediction')
  row=item['row'];label=dict(scene=row['source_id'],frame=row['source_frame'],sample_id=row['sample_id']);mask=valid[:,None];restricted=mask&(target.abs()<512)
  primary.append({**label,**frame_metrics(prediction,target,mask)});secondary.append({**label,**frame_metrics(prediction,target,restricted)})
  if not primary[-1]['valid_pixels']:raise RuntimeError('frame has no source-valid GT')
  d=-target[0,0];x=torch.arange(d.shape[-1],device=d.device)[None];v=valid[0]
  values=d[v];coverage.append({**label,'source_valid_pixels':int(v.sum()),'source_invalid_pixels':int((~v).sum()),'secondary_0_lt_d_lt512_pixels':int(restricted.sum()),'right_target_out_of_native_bounds_pixels':int((v&((x-d<0)|(x-d>d.shape[-1]-1))).sum()),'positive_disparity_quantiles':torch.quantile(values,torch.tensor([0.,.25,.5,.75,.95,.99,1.],device=d.device)).tolist(),**{f'disparity_above_{k}_pixels':int((v&(d>k)).sum()) for k in [192,512,700]}})
  reset_fn(model);del left,right,target,valid,prediction
 result=aggregate(primary);result['secondary_disparity_lt512']=aggregate(secondary);result['coverage']=coverage;result['primary_validity']='all_source_valid_finite_positive_disparity_no_cutoff';result['secondary_coverage_fraction']=sum(x['valid_pixels'] for x in secondary)/sum(x['valid_pixels'] for x in primary)
 return result
