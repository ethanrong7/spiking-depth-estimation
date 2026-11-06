"""Slurm-CPU count-reconstruction displacement proxy. No neural/GPU inference."""
import argparse,csv,hashlib,json,os,time,sys,socket
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR","1")
import cv2
import numpy as np

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
 p=Path(p);t=p.with_suffix(p.suffix+'.part');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');t.replace(p)
def count_image(path,begin=0,end=50):
 raw=np.frombuffer(Path(path).read_bytes(),dtype=np.uint8)
 if raw.size!=625000:raise ValueError('expected original50x250x400DAT: '+str(path))
 return np.unpackbits(raw,bitorder='little').reshape(50,250,400)[begin:end].mean(0,dtype=np.float32)
def pair_indices(anchor,lag):return (anchor,anchor+lag) if anchor+lag<=255 else (anchor-lag,anchor)
def flow_pair(first,second,blur,cfg):
 a=cv2.GaussianBlur(first,(0,0),blur);b=cv2.GaussianBlur(second,(0,0),blur)
 low,high=np.percentile(np.concatenate([a.ravel(),b.ravel()]),cfg['joint_scaling_percentiles']);span=max(float(high-low),1e-6)
 aa=np.clip((a-low)/span,0,1).astype(np.float32)*255.;bb=np.clip((b-low)/span,0,1).astype(np.float32)*255.
 f=cfg['farneback'];args=[f[k]for k in ['pyr_scale','levels','winsize','iterations','poly_n','poly_sigma','flags']]
 fw=cv2.calcOpticalFlowFarneback(aa,bb,None,*args);bw=cv2.calcOpticalFlowFarneback(bb,aa,None,*args)
 yy,xx=np.mgrid[:a.shape[0],:a.shape[1]].astype(np.float32);mx=xx+fw[...,0];my=yy+fw[...,1]
 inside=(mx>=0)&(mx<=a.shape[1]-1)&(my>=0)&(my<=a.shape[0]-1)
 warped_b=cv2.remap(b,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT);back=cv2.remap(bw,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
 fb=np.linalg.norm(fw+back,axis=2);magnitude=np.linalg.norm(fw,axis=2)
 texture=np.hypot(cv2.Sobel(aa,cv2.CV_32F,1,0),cv2.Sobel(aa,cv2.CV_32F,0,1));texture_mask=texture>=max(float(np.quantile(texture,cfg['texture_quantile'])),cfg['texture_gradient_floor'])
 residual=np.abs(a-warped_b);valid=inside&texture_mask&(fb<=cfg['forward_backward_max_pixels'])&(residual<=cfg['photometric_max_count_fraction'])
 valid[:cfg['border_pixels']]=False;valid[-cfg['border_pixels']:]=False;valid[:,:cfg['border_pixels']]=False;valid[:,-cfg['border_pixels']:]=False
 n=int(valid.sum());fraction=float(valid.mean());enough=n>0 and fraction>=cfg['minimum_confident_fraction']
 vector=np.median(fw[valid],axis=0) if n else None;spatial_mad=float(np.median(np.linalg.norm(fw[valid]-vector,axis=1))) if n else None
 phase,response=cv2.phaseCorrelate(aa.copy(),bb.copy(),cv2.createHanningWindow((a.shape[1],a.shape[0]),cv2.CV_32F))
 metrics={'median_local_dx_pixels':float(vector[0]) if n else None,'median_local_dy_pixels':float(vector[1]) if n else None,'local_vector_spatial_MAD_pixels':spatial_mad,'phase_local_vector_difference_pixels':float(np.linalg.norm(np.array(phase)-vector)) if n else None,'phase_dx_pixels':float(phase[0]),'phase_dy_pixels':float(phase[1]),'phase_response':float(response),'all_pixel_magnitude_median':float(np.median(magnitude)),'confident_magnitude_median':float(np.median(magnitude[valid])) if n else None,'confident_magnitude_p90':float(np.quantile(magnitude[valid],.9)) if n else None,'confident_fraction':fraction,'confident_pixels':n,'usable':bool(enough),'forward_backward_error_median':float(np.median(fb[valid])) if n else None,'warped_count_MAE':float(residual[valid].mean()) if n else None,'unwarped_count_MAE':float(np.abs(a-b)[valid].mean()) if n else None,'mean_count_first':float(first.mean()),'mean_count_second':float(second.mean()),'scaling_low_count':float(low),'scaling_high_count':float(high)}
 return metrics,fw,valid

def scene_worker(task):
 scene,split,data_root,cfg,gallery_root,gallery_scene=task;cv2.setNumThreads(1);root=Path(data_root);cache={};rows=[]
 def image(i):
  if i not in cache:cache[i]=count_image(root/'Train'/scene/'spike_l'/f'{i:04d}_spike_l.dat')
  return cache[i]
 for anchor in cfg['anchors']:
  for lag in cfg['lag_steps']:
   first,last=pair_indices(anchor,lag);a,b=image(first),image(last)
   for blur in cfg['blur_sigma_native_pixels']:
    stats,fw,mask=flow_pair(a,b,blur,cfg);record={'scene':scene,'model_split':split,'anchor':anchor,'lag_steps':lag,'first_frame':first,'last_frame':last,'first_filename':f'Train/{scene}/spike_l/{first:04d}_spike_l.dat','last_filename':f'Train/{scene}/spike_l/{last:04d}_spike_l.dat','blur_sigma':blur,**stats};rows.append(record)
    if gallery_scene and anchor==cfg['anchors'][0] and lag==1 and blur==cfg['primary_blur_sigma']:
     lo,hi=stats['scaling_low_count'],stats['scaling_high_count'];display=lambda x:cv2.cvtColor(np.uint8(np.clip((x-lo)/max(hi-lo,1e-6),0,1)*255),cv2.COLOR_GRAY2BGR)
     mag,ang=cv2.cartToPolar(fw[...,0],fw[...,1]);hsv=np.zeros((*mag.shape,3),np.uint8);hsv[...,0]=np.uint8(ang*90/np.pi);hsv[...,1]=255;hsv[...,2]=np.uint8(np.clip(mag/cfg['gallery_flow_scale_pixels'],0,1)*255);color=cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR);color[~mask]=color[~mask]//4
     panels=[display(a),display(b),color]
     for panel,text in zip(panels,[f'count mean {first:04d}',f'count mean {last:04d}',f'proxy flow max {cfg["gallery_flow_scale_pixels"]}px']):cv2.putText(panel,text,(8,22),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,255),1)
     png=Path(gallery_root)/(scene+'.png');cv2.imwrite(str(png),np.concatenate(panels,axis=1));record['gallery_file']=png.name
 covariates=[];within=[]
 for anchor in cfg['GT_covariate_anchors'][split]:
  reference=image(anchor);depth_path=root/'Train'/scene/'depth'/f'{anchor:04d}_depth.exr';depth=cv2.imread(str(depth_path),cv2.IMREAD_ANYCOLOR|cv2.IMREAD_ANYDEPTH)
  if depth is None:raise RuntimeError('cannot decode sourceEXR '+str(depth_path))
  if depth.ndim==3:depth=depth[...,0]
  if depth.shape!=(250,400):raise RuntimeError('nativeGTgeometry changed')
  positive=np.isfinite(depth)&(depth>0);disparity=np.zeros_like(depth,dtype=np.float32);disparity[positive]=cfg['Synthetic_calibration']['f']*cfg['Synthetic_calibration']['B']/depth[positive];valid=positive&(disparity<512)
  gradient=np.hypot(cv2.Sobel(reference,cv2.CV_32F,1,0),cv2.Sobel(reference,cv2.CV_32F,0,1));d_gradient=np.hypot(cv2.Sobel(disparity,cv2.CV_32F,1,0,scale=1/8),cv2.Sobel(disparity,cv2.CV_32F,0,1,scale=1/8));neighbors=cv2.erode(valid.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
  covariates.append({'scene':scene,'model_split':split,'frame':f'{anchor:04d}','depth_filename':str(depth_path.relative_to(root)),'valid_pixels':int(valid.sum()),'GT_valid_fraction':float(valid.mean()),'median_gt_disparity_pixels':float(np.median(disparity[valid])) if valid.any() else None,'p90_gt_disparity_pixels':float(np.quantile(disparity[valid],.9)) if valid.any() else None,'finite_positive_depth_pixels':int(positive.sum()),'mean_gt_disparity_pixels':float(disparity[valid].mean()) if valid.any() else None,'gt_disparity_quantiles':np.quantile(disparity[valid],[0,.25,.5,.75,.95,1.]).tolist() if valid.any() else [],'disparity_edge_prevalence':float((d_gradient[neighbors]>cfg['disparity_edge_threshold_pixels_per_pixel']).mean()) if neighbors.any() else None,'count_mean':float(reference.mean()),'texture_gradient_mean':float(gradient.mean()),'low_count_texture_fraction':float((gradient<cfg['low_texture_count_gradient']).mean())})
 for anchor in cfg['within_window_anchors']:
  path=root/'Train'/scene/'spike_l'/f'{anchor:04d}_spike_l.dat';first=count_image(path,0,25);last=count_image(path,25,50)
  for blur in cfg['blur_sigma_native_pixels']:
   stats,_,_=flow_pair(first,last,blur,cfg);within.append({'scene':scene,'model_split':split,'frame':f'{anchor:04d}','source_filename':str(path.relative_to(root)),'comparison':'within_window_first25_vs_last25','first_bin_indices':[0,24],'last_bin_indices':[25,49],'normalization_bins_per_half':25,'blur_sigma':blur,**stats})
 return {'flow_rows':rows,'covariates':covariates,'within_rows':within}

def control_suite(reference,cfg,count_bins=50,reference_id=''):
 cv2.setNumThreads(1);latent=cv2.GaussianBlur(reference,(0,0),cfg['noise_null_latent_blur_sigma']);conditions=[('identity',0,reference,reference,None)]
 ref_index=cfg['control_reference_indices'].index(cfg['_reference_index']);base_seed=cfg['control_seed']+ref_index*1009+count_bins*101
 for trial in range(cfg['noise_null_trials']):
  rng=np.random.default_rng(np.random.SeedSequence([base_seed,0,trial]));a=rng.binomial(count_bins,np.clip(latent,0,1)).astype(np.float32)/count_bins;b=rng.binomial(count_bins,np.clip(latent,0,1)).astype(np.float32)/count_bins;conditions.append(('static_binomial_count_noise',trial,a,b,None))
 for gain_index,gain in enumerate(cfg['flicker_gains']):
  other=np.clip(gain*latent,0,1);conditions.append(('stationary_clean_flicker',0,latent,other,gain))
  for trial in range(cfg['noise_null_trials']):
   rng=np.random.default_rng(np.random.SeedSequence([base_seed,gain_index+1,trial]));a=rng.binomial(count_bins,np.clip(latent,0,1)).astype(np.float32)/count_bins;b=rng.binomial(count_bins,other).astype(np.float32)/count_bins;conditions.append(('stationary_flicker_count_noise',trial,a,b,gain))
 rows=[]
 for blur in cfg['blur_sigma_native_pixels']:
  for kind,trial,a,b,gain in conditions:
   stats,_,_=flow_pair(a,b,blur,cfg);rows.append({'kind':kind,'reference_scene':reference_id,'count_bins':count_bins,'flicker_gain':gain,'blur_sigma':blur,'trial':trial,**stats})
  for dx in cfg['known_shift_pixels']:
   shifted=cv2.warpAffine(reference,np.float32([[1,0,dx],[0,1,0]]),(400,250),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_REFLECT_101);stats,fw,mask=flow_pair(reference,shifted,blur,cfg)
   rows.append({'kind':'known_translation','reference_scene':reference_id,'count_bins':count_bins,'known_dx_pixels':dx,'known_dy_pixels':0,'blur_sigma':blur,'trial':0,'confident_vector_error_median':float(np.median(np.linalg.norm(fw-np.array([dx,0]),axis=2)[mask])) if mask.any() else None,'phase_vector_error_pixels':float(np.hypot(stats['phase_dx_pixels']-dx,stats['phase_dy_pixels'])),**stats})
 return rows

def summarize_controls(rows,cfg,count_bins):
 references=sorted({r['reference_scene']for r in rows});quality=[];floors={}
 for reference in references:
  tests=[]
  for blur in cfg['blur_sigma_native_pixels']:
   rr=[r for r in rows if r['reference_scene']==reference and r['blur_sigma']==blur]
   identity=next(r for r in rr if r['kind']=='identity');shifts=[r for r in rr if r['kind']=='known_translation'];static=[r['confident_magnitude_median']for r in rr if r['kind']=='static_binomial_count_noise' and r['usable']]
   identity_pass=identity['usable'] and identity['confident_magnitude_median']<=cfg['control_gates']['identity_max_pixels'];shift_pass=all(r['usable'] and r['confident_vector_error_median']<=cfg['control_gates']['known_shift_median_error_max_pixels']for r in shifts)
   q95=float(np.quantile(static,.95)) if static else None;noise_pass=len(static)>=cfg['minimum_usable_static_null_trials'] and q95<=cfg['control_gates']['noise_null_p95_max_pixels'];tests.append({'blur_sigma':blur,'identity_pass':bool(identity_pass),'all_known_shifts_pass':bool(shift_pass),'usable_static_nulls':len(static),'static_null_q95':q95,'static_null_pass':bool(noise_pass)})
  quality.append({'reference_scene':reference,'eligible':all(t['identity_pass'] and t['all_known_shifts_pass'] and t['static_null_pass']for t in tests),'blur_tests':tests})
 for blur in cfg['blur_sigma_native_pixels']:
  per_ref=[]
  for reference in references:
   vals=[r['confident_magnitude_median']for r in rows if r['reference_scene']==reference and r['blur_sigma']==blur and r['kind']in['static_binomial_count_noise','stationary_flicker_count_noise']and r['usable']]
   if vals:per_ref.append({'reference_scene':reference,'usable_nulls':len(vals),'q95':float(np.quantile(vals,.95))})
  clean=[r['confident_magnitude_median']for r in rows if r['blur_sigma']==blur and r['kind']=='stationary_clean_flicker'and r['usable']];n=max([r['q95']for r in per_ref],default=None);f=max(clean,default=0.)
  floors[str(blur)]={'floor_pixels':max(cfg['minimum_signal_floor_pixels'],n+cfg['signal_margin_pixels'],f+cfg['signal_margin_pixels']) if n is not None else None,'N_max_reference_q95':n,'F_max_usable_clean_flicker':f,'all_reference_null_calibration':per_ref,'references_failing_positive_controls_NOT_removed':True}
 shifts=[r for r in rows if r['kind']=='known_translation'];phase_ok=all(r['phase_vector_error_pixels']<=cfg['control_gates']['known_shift_median_error_max_pixels']for r in shifts)
 return {'count_bins':count_bins,'passed':sum(q['eligible']for q in quality)>=cfg['minimum_eligible_control_references'] and all(f['floor_pixels'] is not None for f in floors.values()),'eligible_references':sum(q['eligible']for q in quality),'reference_quality':quality,'resolution_floors_by_blur':floors,'null_unusable_is_abstention_not_zero':True,'phase_positive_control_crosscheck_reliable':phase_ok,'rows':rows,'null_is_simplified_count_noise_not_authenticated_sensor_noise':True}

def rank_values(values):
 order=np.argsort(values,kind='stable');rank=np.empty(len(values),float);i=0
 while i<len(order):
  j=i+1
  while j<len(order) and values[order[j]]==values[order[i]]:j+=1
  rank[order[i:j]]=(i+j-1)/2.;i=j
 return rank

def summarize(rows,scenes,cfg,controls):
 summaries=[];primary={};above={}
 for scene,split in scenes:
  tests=[]
  for blur in cfg['blur_sigma_native_pixels']:
   vals=[r['confident_magnitude_median']for r in rows if r['scene']==scene and r['lag_steps']==1 and r['blur_sigma']==blur and r['usable']];floor=controls['resolution_floors_by_blur'][str(blur)]['floor_pixels'];median=float(np.median(vals)) if len(vals)>=cfg['minimum_pairs_per_scene'] else None
   tests.append(median is not None and floor is not None and median>floor and sum(v>floor for v in vals)>=cfg['minimum_pairs_above_floor'])
   if blur==cfg['primary_blur_sigma']:primary[(scene,split)]=median
  above[(scene,split)]=all(tests)
  for lag in cfg['lag_steps']:
   for blur in cfg['blur_sigma_native_pixels']:
    valid=[r['confident_magnitude_median']for r in rows if r['scene']==scene and r['lag_steps']==lag and r['blur_sigma']==blur and r['usable']];summaries.append({'scene':scene,'model_split':split,'lag_steps':lag,'blur_sigma':blur,'usable_pairs':len(valid),'median_displacement_proxy':float(np.median(valid)) if len(valid)>=cfg['minimum_pairs_per_scene'] else None})
 train_usable={s:v for(s,split),v in primary.items()if split=='train'and v is not None};train_signal={s:v for(s,split),v in primary.items()if split=='train'and above[(s,split)]};cuts=np.quantile(list(train_signal.values()),cfg['train_bin_quantiles']).tolist() if train_signal else None;stability=[]
 for lag in cfg['lag_steps']:
  for blur in cfg['blur_sigma_native_pixels']:
   other={r['scene']:r['median_displacement_proxy']for r in summaries if r['model_split']=='train'and r['lag_steps']==lag and r['blur_sigma']==blur};matched=[s for s in train_signal if other.get(s)is not None];a=rank_values(np.array([train_signal[s]for s in matched]));b=rank_values(np.array([other[s]for s in matched]));rho=float(np.corrcoef(a,b)[0,1]) if len(matched)>2 and np.std(a)>0 and np.std(b)>0 else None;stability.append({'lag_steps':lag,'blur_sigma':blur,'above_floor_Train_scenes_matched':len(matched),'spearman':rho})
 stable=all(x['above_floor_Train_scenes_matched']>=cfg['minimum_signal_train_scenes_for_bins']and x['spearman']is not None and x['spearman']>=cfg['minimum_train_rank_spearman']for x in stability);labels={}
 for(scene,split),value in primary.items():
  labels[scene]='unusable' if value is None else 'below_proxy_resolution' if not above[(scene,split)]else 'unclassified_no_Train_reference' if cuts is None else 'low'if value<=cuts[0]else'middle'if value<=cuts[1]else'high'
 for row in summaries:row['above_floor_both_blurs']=above[(row['scene'],row['model_split'])];row['primary_ordinal_bin_from_frozen_Train_cuts']=labels[row['scene']]
 dev_groups={label:sum(split=='dev'and labels[s]==label for s,split in scenes)for label in ['low','middle','high','below_proxy_resolution','unusable','unclassified_no_Train_reference']}
 phase_groups=[]
 for split in ['train','dev']:
  for translationlike in [True,False]:
   for highresponse in [True,False]:
    rr=[r for r in rows if r['model_split']==split and r['usable']and(r['local_vector_spatial_MAD_pixels']<=cfg['phase_translationlike_max_MAD_pixels'])==translationlike and(r['phase_response']>=cfg['phase_high_response_threshold'])==highresponse]
    phase_groups.append({'model_split':split,'translationlike_local_flow':translationlike,'phase_response_at_least_threshold':highresponse,'pair_records':len(rr),'median_phase_local_vector_difference_pixels':float(np.median([r['phase_local_vector_difference_pixels']for r in rr])) if rr else None})
 return {'scene_summaries':summaries,'train_only_above_floor_bin_cuts':cuts,'Train_usable_primary_scenes':len(train_usable),'Train_above_floor_both_blurs_scenes':len(train_signal),'rank_stability_above_floor_Train_only':stability,'coverage_gate_pass':len(train_usable)>=cfg['minimum_train_scenes_for_bins'],'signal_support_gate_pass':len(train_signal)>=cfg['minimum_signal_train_scenes_for_bins'],'rank_stability_gate_pass':stable,'Dev_ordinal_group_counts':dev_groups,'Dev_categorical_high_low_contrast_eligible':dev_groups['low']>=cfg['minimum_Dev_scenes_per_contrast_group']and dev_groups['high']>=cfg['minimum_Dev_scenes_per_contrast_group'],'phase_diagnostic_summaries_NO_hard_agreement_gate':phase_groups,'Dev_used_for_threshold_or_control_calibration':False}

def main():
 p=argparse.ArgumentParser()
 for name in ['registration','manifest','data-root','source-registry','output']:p.add_argument('--'+name,required=True)
 a=p.parse_args();cfg=json.loads(Path(a.registration).read_text());manifest=json.loads(Path(a.manifest).read_text())
 if cfg['status']!='registered':raise RuntimeError('prospective root/Astra protocol registration required')
 if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU Slurm only; CUDA visibility must be empty')
 if digest(a.manifest)!=cfg['manifest_sha256']:raise RuntimeError('manifest identity mismatch')
 registry=json.loads(Path(a.source_registry).read_text());source_root=Path(a.source_registry).parent
 if registry['source_identity_sha256']!=cfg['source_identity_sha256']:raise RuntimeError('source identity binding mismatch')
 for relative,expected in registry['source_files'].items():
  if digest(source_root/relative)!=expected:raise RuntimeError('source bytes changed: '+relative)
 out=Path(a.output);out.mkdir(parents=True,exist_ok=False);cv2.setNumThreads(1);started=time.monotonic();splits={}
 for split,expected in [('train',120),('dev',30)]:
  ids={r['scene']for r in manifest[split]};assert len(ids)==expected
  for s in ids:
   if s in splits:raise RuntimeError('scene split overlap')
   if not s.startswith('scene_') or '/'in s:raise RuntimeError('unsafe scene')
   splits[s]=split
 scenes=sorted(splits.items());train_scenes=sorted(s for s,v in scenes if v=='train');control_rows=[]
 for i in cfg['control_reference_indices']:
  scene=train_scenes[i];local=dict(cfg,_reference_index=i);control_rows.extend(control_suite(count_image(Path(a.data_root)/'Train'/scene/'spike_l'/'0001_spike_l.dat'),local,50,scene))
 control=summarize_controls(control_rows,cfg,50);save(out/'controls.json',control)
 within_control_rows=[]
 for i in cfg['control_reference_indices']:
  scene=train_scenes[i];local=dict(cfg,_reference_index=i);within_control_rows.extend(control_suite(count_image(Path(a.data_root)/'Train'/scene/'spike_l'/'0001_spike_l.dat',0,25),local,25,scene))
 within_controls=summarize_controls(within_control_rows,cfg,25);save(out/'within-window-25bin-controls.json',within_controls)
 gallery=out/'gallery';gallery.mkdir();gallery_ids=set(sorted(s for s,v in scenes if v=='train')[:4]+sorted(s for s,v in scenes if v=='dev')[:2]);tasks=[(s,split,a.data_root,cfg,str(gallery),s in gallery_ids)for s,split in scenes];rows=[];covariates=[];within_rows=[]
 with ProcessPoolExecutor(max_workers=cfg['workers'])as executor:
  for completed,result in enumerate(executor.map(scene_worker,tasks),1):
   rows.extend(result['flow_rows']);covariates.extend(result['covariates']);within_rows.extend(result['within_rows']);save(out/'status.json',{'status':'RUNNING_CPU_MOTION_AUDIT','completed_scenes':completed,'total_scenes':150,'pairs_processed':len(rows),'GPU_used':False})
 summary=summarize(rows,scenes,cfg,control);eligible=control['passed']and summary['coverage_gate_pass']and summary['signal_support_gate_pass']and summary['rank_stability_gate_pass'];status='MOTION_PROXY_QC_PASS' if eligible else 'MOTION_PROXY_QC_FAILED_RETAIN_RAW_NO_TUNING'
 with(out/'raw-pairs.jsonl').open('w')as f:
  for row in rows:f.write(json.dumps(row,allow_nan=False)+'\n')
 for filename,records in [('covariates.jsonl',covariates),('within-window-halves.jsonl',within_rows)]:
  with(out/filename).open('w')as f:
   for row in records:f.write(json.dumps(row,allow_nan=False)+'\n')
 with(out/'scene-proxies.csv').open('w')as f:
  writer=csv.DictWriter(f,fieldnames=list(summary['scene_summaries'][0]));writer.writeheader();writer.writerows(summary['scene_summaries'])
 save(out/'result.json',{'status':status,'eligible_for_motion_stratification':eligible,'controls':{k:v for k,v in control.items()if k!='rows'},**summary,'rows':len(rows),'GT_count_texture_covariate_records':len(covariates),'within_window_diagnostic_records':len(within_rows),'within_window_diagnostic_ranked':False,'within_window_25bin_controls_pass':within_controls['passed'],'within_window_25bin_resolution_floors':within_controls['resolution_floors_by_blur'],'distinct_temporal_pairs':150*len(cfg['anchors'])*len(cfg['lag_steps']),'configuration':cfg,'source_identity_sha256':registry['source_identity_sha256'],'git_commit':registry['git_commit'],'dirty_manifest_sha256':registry['dirty_manifest_sha256'],'registration_sha256':digest(a.registration),'source_script_sha256':digest(__file__),'manifest_sha256':digest(a.manifest),'job_id':os.environ['SLURM_JOB_ID'],'CPU_workers':cfg['workers'],'numpy_version':np.__version__,'opencv_version':cv2.__version__,'python_version':sys.version,'hostname':socket.gethostname(),'runtime_seconds':time.monotonic()-started,'no_GPU_or_neural_inference':True,'no_Test_or_confirmation':True,'units':'nativepixels_per_stated_fileindexgap_NOTphysicalspeed','timestamp_window_overlap_unknown':True,'source_data_read_only':True});save(out/'status.json',{'status':status,'completed_scenes':150,'GPU_used':False})
if __name__=='__main__':main()
