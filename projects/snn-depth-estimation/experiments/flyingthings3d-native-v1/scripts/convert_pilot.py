"""One registered native-resolution sequence, EMA-VFI + continuous I&F; no training."""
import argparse,datetime,hashlib,json,os,platform,subprocess,time,traceback
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
from ema_adapter import EMAInterpolator,sha
from source_io import read_rgb,read_disparity_pfm,validate_index
from spike_codec import window_indices,integrate_torch,atomic_dat


def save(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix(path.suffix+'.part');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def resolve(root,name):
 p=Path(name)
 if p.is_absolute() or '..' in p.parts:raise ValueError('source reference must remain source-root-relative')
 p=(root/p).resolve()
 if not p.is_relative_to(root.resolve()):raise ValueError('source path escaped')
 return p


@torch.no_grad()
def warp_qc(left,right,disparity):
 """GT-centered identity check, not a stereo-model accuracy measurement."""
 import torch.nn.functional as F
 l=torch.from_numpy(left.copy()).to('cuda',dtype=torch.float32)/255
 r=torch.from_numpy(right.copy()).to('cuda',dtype=torch.float32).permute(2,0,1)[None]/255
 d=torch.from_numpy(disparity).to('cuda');h,w=d.shape
 y,x=torch.meshgrid(torch.arange(h,device='cuda'),torch.arange(w,device='cuda'),indexing='ij')
 result={}
 for direction,sign in [('declared_xR_equals_xL_minus_d',-1),('opposite_sign_diagnostic',1)]:
  xr=x+sign*d;valid=torch.isfinite(d)&(d>0)&(xr>=0)&(xr<=w-1)
  safe=torch.where(torch.isfinite(xr),xr,torch.zeros_like(xr))
  grid=torch.stack([2*safe/(w-1)-1,2*y/(h-1)-1],-1)[None]
  warped=F.grid_sample(r,grid,align_corners=True,padding_mode='zeros')[0].permute(1,2,0)
  result[direction]={'valid_pixels':int(valid.sum()),'mean_absolute_RGB_difference':float((warped-l).abs().mean(-1)[valid].mean()) if valid.any() else None}
 return result


def contact_sheet(output,images):
 canvas=Image.new('RGB',(960,180*len(images)),'white');draw=ImageDraw.Draw(canvas)
 for row,(name,rgb) in enumerate(images):
  image=Image.fromarray(np.asarray(rgb,dtype=np.uint8)).resize((320,180))
  canvas.paste(image,(0,180*row));draw.text((330,180*row+10),name,fill='black')
 canvas.save(output)


@torch.no_grad()
def run(args):
 started=time.monotonic();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
 registration=json.loads(Path(args.registration).read_text());protocol=json.loads(Path(args.protocol).read_text());index=json.loads(Path(args.source_index).read_text())
 if registration['status']!='registered' or registration['mode']!='one-sequence-pilot' or registration['max_allocated_gpus']!=1:raise RuntimeError('not an authorized one-GPU pilot registration')
 for key,path in [('protocol_sha256',args.protocol),('source_index_sha256',args.source_index)]:
  if registration[key]!=sha(path):raise RuntimeError(key+' mismatch')
 sequences=validate_index(index);seq=next(s for s in sequences if s['split']=='train')
 if registration['source_id']!=seq['source_id']:raise RuntimeError('pilot must use predetermined first TRAIN selection')
 for name,digest in registration['converter_source_sha256'].items():
  if sha(Path(__file__).parent/name)!=digest:raise RuntimeError('converter code mismatch: '+name)
 if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or os.environ.get('SLURM_GPUS_ON_NODE') not in ('1',None):raise RuntimeError('exactly one Slurm-visible CUDA GPU required')
 if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('conversion requires a Slurm GPU allocation')
 torch.set_num_threads(2);torch.manual_seed(1234);torch.cuda.manual_seed_all(1234)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.backends.cudnn.benchmark=False
 root=Path(args.source_root);derived=Path(args.derived_root);destination=derived/seq['source_id'];staging=derived/(seq['source_id']+'.pilot-staging')
 if destination.exists() or staging.exists():raise RuntimeError('existing derived sequence/staging: inspect before another attempt')
 staging.mkdir(parents=True);progress={'status':'RUNNING','source_id':seq['source_id'],'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'registration_sha256':sha(args.registration),'job_id':os.environ['SLURM_JOB_ID'],'completed_intervals':0,'completed_windows':0,'training_performed':False}
 save(out/'status.json',progress)
 source_files=[];rgb={};gt={}
 for view in ['l','r']:
  rgb[view]=[]
  for f in seq['frames']:
   p=resolve(root,f['rgb_'+view]);rgb[view].append(read_rgb(p));source_files.append({'path':f['rgb_'+view],'bytes':p.stat().st_size,'sha256':sha(p)})
 for center in range(1,9):
  f=seq['frames'][center];p=resolve(root,f['disparity_l']);d,meta=read_disparity_pfm(p);valid=np.isfinite(d)&(d>0)
  if not valid.any():raise RuntimeError('actual target has no finite positive disparity')
  gt[center]=d;source_files.append({'path':f['disparity_l'],'bytes':p.stat().st_size,'sha256':sha(p),'pfm':meta,'finite_positive_pixels':int(valid.sum()),'positive_gt_min':float(d[valid].min()),'positive_gt_max':float(d[valid].max()),'positive_gt_median':float(np.median(d[valid]))})
 save(out/'source-files.json',source_files)
 model=EMAInterpolator(args.ema_source,args.checkpoint,source_files=protocol['interpolator']['source_files'])
 torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();conversion_start=time.monotonic()
 voltage_traces=[];intervals=[];windows={};snapshots=[];overshoots=0
 for view in ['l','r']:
  voltage=torch.zeros((540,960),device='cuda',dtype=torch.float32);buffers={};window_sets={i:set(window_indices(i)) for i in range(1,9)}
  for interval in range(9):
   if time.monotonic()-started>registration['time_limit_seconds']-30:raise TimeoutError('pilot deadline; preserve partial, no automatic extension')
   torch.cuda.synchronize();tick=time.monotonic()
   iterator=model.interiors(rgb[view][interval],rgb[view][interval+1])
   for j in range(1,51):
    if j==50:
     frame=torch.from_numpy(rgb[view][interval+1].copy()).to('cuda',dtype=torch.float32)/255;extra={'overshoot_values':0}
    else:frame,extra=next(iterator)
    n=50*interval+j;overshoots+=extra['overshoot_values']
    intensity=.299*frame[...,0]+.587*frame[...,1]+.114*frame[...,2]
    spike,voltage=integrate_torch(voltage,intensity)
    if not torch.isfinite(voltage).all() or float(voltage.min())<0 or float(voltage.max())>=5.00001:raise RuntimeError('invalid continuous I&F state')
    voltage_traces.append({'view':view,'global_index':n,'source_time':n/50,'spike_fraction':float(spike.float().mean()),'voltage_mean':float(voltage.mean()),'voltage_min':float(voltage.min()),'voltage_max':float(voltage.max()),**extra})
    if n in [25,50,75]:snapshots.append((f'EMA/native {view} sample n={n}, source time={n/50}',(frame.clamp(0,1).cpu().numpy()*255).round().astype(np.uint8)))
    for center,indices in window_sets.items():
     if n in indices:
      buffers.setdefault(center,[]).append(spike.clone())
      if n==max(indices):
       bins=torch.stack(buffers.pop(center)).cpu().numpy()
       if bins.shape!=(50,540,960):raise RuntimeError('wrong centered window length/geometry')
       rel=f'spike_{view}/{seq["frames"][center]["source_frame"]}.dat';p=staging/rel;atomic_dat(p,bins)
       if p.stat().st_size!=3240000:raise RuntimeError('wrong packed byte count')
       windows[(center,view)]={'relative_path':rel,'sha256':sha(p),'bytes':p.stat().st_size,'spike_fraction':float(bins.mean())}
       progress['completed_windows']+=1
    del frame,spike,intensity
   try:next(iterator);raise RuntimeError('unexpected extra interpolation sample')
   except StopIteration:pass
   torch.cuda.synchronize();intervals.append({'view':view,'interval':interval,'seconds':time.monotonic()-tick,'interpolated_images':49,'exact_native_endpoint_images':1})
   progress.update(completed_intervals=len(intervals),elapsed_seconds=time.monotonic()-started);save(out/'status.json',progress)
  if buffers:raise RuntimeError('unfinished centered windows')
  del voltage
 torch.cuda.synchronize();conversion_seconds=time.monotonic()-conversion_start
 qc=warp_qc(rgb['l'][1],rgb['r'][1],gt[1]);contact_sheet(out/'contact-sheet.png',[('Native left actual center frame1',rgb['l'][1]),('Native right actual center frame1',rgb['r'][1]),*snapshots])
 rows=[]
 for center in range(1,9):
  f=seq['frames'][center];rows.append({'sample_id':seq['source_id']+'/'+f['source_frame'],'source_id':seq['source_id'],'split':'train','center_index':center,'source_frame':f['source_frame'],'source_time':center,'rgb_l':f['rgb_l'],'rgb_r':f['rgb_r'],'disparity_l':f['disparity_l'],'spike_l':str(Path(seq['source_id'])/windows[(center,'l')]['relative_path']),'spike_r':str(Path(seq['source_id'])/windows[(center,'r')]['relative_path']),'spike_shape':[50,540,960],'global_indices':window_indices(center),'vertical_flip':False,'half_frame_future_lookahead':True})
 if len(windows)!=16 or len(voltage_traces)!=900:raise RuntimeError('incomplete sequence')
 save(staging/'manifest.json',{'protocol_id':protocol['protocol_id'],'source_index_sha256':sha(args.source_index),'rows':rows,'source_root_resolution':'runtime-config','disparity_semantics':'actual-left-native-positive-pixel-PFM-not-depth','pilot_only':True})
 destination.parent.mkdir(parents=True,exist_ok=True);staging.replace(destination)
 save(out/'voltage-traces.json',voltage_traces);save(out/'interval-timing.json',intervals)
 gpu=subprocess.run(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
 result={**progress,'status':'COMPLETED','elapsed_seconds':time.monotonic()-started,'conversion_seconds':conversion_seconds,'peak_cuda_memory_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_cuda_memory_reserved_bytes':torch.cuda.max_memory_reserved(),'interpolated_view_images':882,'exact_native_endpoint_view_images':18,'selected_stereo_windows':8,'selected_dat_files':16,'selected_spike_bytes':51840000,'all450_samples_processed_per_view':True,'overshoot_channel_values_clamped':overshoots,'native_center_GT_warp_diagnostic':qc,'windows':[{ 'center':c,'view':v,**value} for (c,v),value in windows.items()],'checkpoint':model.metadata,'gpu':gpu,'python':platform.python_version(),'torch':torch.__version__,'cudnn':torch.backends.cudnn.version(),'tf32':False,'TTA':False,'batch_per_view':1,'derived_sequence':str(destination),'forecast_all150_conversion_hours_25pct_overhead':conversion_seconds*150*1.25/3600,'bulk_authorized':False,'training_performed':False}
 save(out/'result.json',result);save(out/'status.json',result)


def main():
 p=argparse.ArgumentParser()
 for name in ['registration','protocol','source-index','source-root','derived-root','ema-source','checkpoint','output']:p.add_argument('--'+name,required=True)
 a=p.parse_args()
 try:run(a)
 except Exception as exc:
  out=Path(a.output)
  if out.exists():save(out/'failure.json',{'status':'FAILED','error':str(exc),'traceback':traceback.format_exc(),'training_performed':False,'retry_automatically':False})
  raise
if __name__=='__main__':main()
