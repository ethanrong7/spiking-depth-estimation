"""Native source identity/shape helpers; no geometry resampling or depth model."""
from pathlib import Path
import numpy as np
from PIL import Image


def read_rgb(path):
 with Image.open(path) as image:
  if image.mode!='RGB' or image.size!=(960,540):raise ValueError('expected native960x540 RGB PNG: '+str(path))
  return np.array(image,dtype=np.uint8)


def read_disparity_pfm(path):
 with Path(path).open('rb') as f:
  if f.readline().strip()!=b'Pf':raise ValueError('expected singlechannel SceneFlow PFM')
  width,height=map(int,f.readline().split());scale=float(f.readline().strip())
  if (width,height)!=(960,540) or abs(scale)!=1.:raise ValueError('unexpectedPFMgeometry/scale; do not resample or reinterpret')
  data=np.frombuffer(f.read(),dtype='<f4' if scale<0 else '>f4')
  if data.size!=width*height:raise ValueError('PFMbytecountmismatch')
  data=np.flipud(data.reshape(height,width)).astype(np.float32,copy=True)
 return data,dict(width=width,height=height,scale_header=scale,endian='little' if scale<0 else 'big',vertical_orientation='PFMbottom-updecodedflipud,matchingofficialRAFTloader',reference='leftpositivepixelDisparity;preservedsource,nooldfBconversion')


def validate_index(index):
 sequences=index['sequences'];assert len(sequences)==150 and index['selection_seed']==1234
 assert sum(s['split']=='train' for s in sequences)==120 and sum(s['split']=='dev' for s in sequences)==30
 ids=[s['source_id'] for s in sequences];assert len(set(ids))==150 and all(i.startswith('TRAIN/C/') for i in ids)
 for seq in sequences:
  frames=seq['frames'];assert [f['index'] for f in frames]==list(range(10)) and len({f['source_frame'] for f in frames})==10
  source_numbers=[int(f['source_frame']) for f in frames];assert source_numbers==list(range(source_numbers[0],source_numbers[0]+10))
  for frame in frames:
   for view in ['rgb_l','rgb_r']:
    path=Path(frame[view]);assert not path.is_absolute() and '..' not in path.parts and 'TRAIN' in path.parts and 'TEST' not in path.parts
    assert path.stem==frame['source_frame'] and str(path.parent.parent).endswith(seq['source_id'])
    assert path.parent.name==('left' if view=='rgb_l' else 'right')
  for center in (range(1,9) if seq['split']=='train' else [3,6]):
   value=frames[center]['disparity_l'];assert value
   path=Path(value);assert not path.is_absolute() and '..' not in path.parts and 'TRAIN' in path.parts and 'TEST' not in path.parts
 return sequences
