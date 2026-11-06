"""Actual RGB/PFM plus paired 50-bin spikes; never inherits old depth calibration."""
import sys
from pathlib import Path
import numpy as np
import torch
PREP=Path(__file__).resolve().parents[4]/'flyingthings3d-native-v1/scripts'
sys.path.insert(0,str(PREP))
from source_io import read_rgb,read_disparity_pfm
from spike_codec import unpack_spikes


def input_tensor(array,model):
 if model=='original':
  if array.ndim!=3 or array.shape[0]!=50 or not np.isin(array,[0,1]).all():raise ValueError('original requires all50binaryplanes')
  return torch.from_numpy(array.astype(np.float32,copy=True))
 if array.ndim!=3 or array.shape[-1]!=3 or array.dtype!=np.uint8:raise ValueError('native models require real RGBuint8')
 return torch.from_numpy(array.copy()).permute(2,0,1).float() # Native models apply RGB/255 normalization once.


def target_tensor(disparity):
 valid=np.isfinite(disparity)&(disparity>0)
 target=np.zeros(disparity.shape,dtype=np.float32);target[valid]=-disparity[valid]
 return torch.from_numpy(target[None]),torch.from_numpy(valid.copy())


def crop_coordinates(seed,update,micro,geometry,shape=(540,960)):
 if geometry=='native':return (0,0,*shape)
 if geometry!='crop384x768':raise ValueError('unregistered geometry')
 rng=np.random.default_rng(np.random.SeedSequence([seed,update,micro]));h,w=shape;ch,cw=384,768
 return int(rng.integers(h-ch+1)),int(rng.integers(w-cw+1)),ch,cw


def apply_crop(item,coords):
 y,x,h,w=coords;result=dict(item)
 for k in ['left','right','target','valid']:result[k]=item[k][...,y:y+h,x:x+w]
 result['crop']=[y,x,h,w];return result


class NativeDataset:
 def __init__(self,source_root,derived_root,rows,model):self.source_root=Path(source_root);self.derived_root=Path(derived_root);self.rows=rows;self.model=model
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  row=self.rows[i]
  def safe(root,name):
   p=Path(name)
   if p.is_absolute() or '..' in p.parts or 'TEST' in p.parts:raise ValueError('manifest path must remain relative and TRAIN-only')
   return root/p
  disparity,_=read_disparity_pfm(safe(self.source_root,row['disparity_l']));target,valid=target_tensor(disparity)
  if self.model=='original':
   pair=[unpack_spikes(safe(self.derived_root,row['spike_'+v]).read_bytes(),540,960) for v in ['l','r']]
  else:pair=[read_rgb(safe(self.source_root,row['rgb_'+v])) for v in ['l','r']]
  left,right=[input_tensor(x,self.model) for x in pair]
  return dict(left=left,right=right,target=target,valid=valid,index=i,row=row,crop=[0,0,540,960])


def validate_manifest(manifest):
 from collections import defaultdict
 cohorts={}
 for split,count,centers,sequences in [('train',960,set(range(1,9)),120),('dev',60,{3,6},30)]:
  rows=manifest[split];assert len(rows)==count and len({r['sample_id'] for r in rows})==count
  groups=defaultdict(set)
  for row in rows:
   assert row['source_id'].startswith('TRAIN/C/') and row['split']==split and row['spike_shape']==[50,540,960]
   assert row['global_indices']==list(range(50*row['center_index']-24,50*row['center_index']+26))
   assert row['source_time']==row['center_index'] and row['half_frame_future_lookahead'] is True and row['vertical_flip'] is False
   groups[row['source_id']].add(row['center_index'])
   for name in ['rgb_l','rgb_r','disparity_l','spike_l','spike_r']:
    path=Path(row[name]);assert not path.is_absolute() and '..' not in path.parts and 'TEST' not in path.parts
  assert len(groups)==sequences and all(c==centers for c in groups.values())
  cohorts[split]=set(groups)
 assert not cohorts['train']&cohorts['dev']
 return True
