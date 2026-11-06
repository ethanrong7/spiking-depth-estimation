"""Tiny synthetic source-format checks; no dataset conversion/inference."""
import json,tempfile
from pathlib import Path
import numpy as np
from PIL import Image
from source_io import read_disparity_pfm,read_rgb,validate_index


def main():
 with tempfile.TemporaryDirectory() as tmp:
  root=Path(tmp);a=np.arange(540*960,dtype=np.float32).reshape(540,960)
  for endian,scale in [('<',-1),('>',1)]:
   p=root/'target.pfm';p.write_bytes(f'Pf\n960 540\n{scale}\n'.encode()+np.flipud(a).astype(endian+'f4').tobytes())
   decoded,meta=read_disparity_pfm(p);assert decoded.dtype==np.float32 and np.array_equal(decoded,a)
  rgb=np.zeros((540,960,3),dtype=np.uint8);rgb[:,:,0]=17;rgb[:,:,1]=93;rgb[:,:,2]=241
  Image.fromarray(rgb).save(root/'frame.png');assert np.array_equal(read_rgb(root/'frame.png'),rgb)
  Image.fromarray(rgb).resize((100,100)).save(root/'wrong.png')
  try:read_rgb(root/'wrong.png');raise AssertionError('must reject nonnative geometry')
  except ValueError:pass
 seq=[]
 for n in range(150):
  split='train' if n<120 else 'dev';selected=range(1,9) if split=='train' else [3,6];sid=f'TRAIN/C/{n:04d}'
  frames=[{'index':i,'source_frame':f'{i+6:04d}','rgb_l':f'frames_cleanpass/{sid}/left/{i+6:04d}.png','rgb_r':f'frames_cleanpass/{sid}/right/{i+6:04d}.png','disparity_l':f'disparity/{sid}/left/{i+6:04d}.pfm' if i in selected else None} for i in range(10)]
  seq.append({'source_id':sid,'split':split,'frames':frames})
 assert len(validate_index({'selection_seed':1234,'sequences':seq}))==150
 seq[0]['frames'][4]['source_frame']='0030'
 try:validate_index({'selection_seed':1234,'sequences':seq});raise ValueError('nonconsecutive source frames must fail')
 except AssertionError:pass
 print(json.dumps({'synthetic_only':True,'native_RGB_decode_exact':True,'PFM_both_endian_numeric_and_vertical_exact':True,'PFM_native_float32':True,'reject_resizing':True,'150_scene_manifest_nullable_unselected_GT':True,'consecutive_chronological_source_frames_enforced':True,'gpu_inference_performed':False},indent=2))
if __name__=='__main__':main()
