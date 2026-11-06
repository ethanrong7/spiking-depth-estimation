"""Pinned official EMA-VFI inference methods with safe explicit weight loading."""
import hashlib,importlib,json,sys
from pathlib import Path
import numpy as np
import torch

EXPECTED_CHECKPOINT_SHA='c09e82d193e303d7b91f1e36bf535886b820cb56105574355211d60a6b557a99'


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()


class EMAInterpolator:
 def __init__(self,source,checkpoint,device='cuda',source_files=None):
  source=Path(source).resolve();checkpoint=Path(checkpoint).resolve()
  if sha(checkpoint)!=EXPECTED_CHECKPOINT_SHA:raise RuntimeError('arbitrary-time ours_t checkpoint identity mismatch')
  if source_files:
   for name,digest in source_files.items():
    if sha(source/name)!=digest:raise RuntimeError('EMA source changed: '+name)
  sys.path.insert(0,str(source))
  cfg=importlib.import_module('config');trainer=importlib.import_module('Trainer')
  if not Path(cfg.__file__).resolve().is_relative_to(source) or not Path(trainer.__file__).resolve().is_relative_to(source):raise RuntimeError('EMA import namespace collision')
  architecture=cfg.init_model_config(F=32,depth=[2,2,2,4,4])
  backbone,multiscale=cfg.MODEL_CONFIG['MODEL_TYPE'];net=multiscale(backbone(**architecture[0]),**architecture[1])
  raw=torch.load(checkpoint,map_location='cpu',weights_only=True)
  if not isinstance(raw,dict):raise RuntimeError('expected original raw state dictionary')
  # This is the exact upstream module-prefix / derived-buffer filter, no remapping.
  weights={k.replace('module.',''):v for k,v in raw.items() if 'module.' in k and 'attn_mask' not in k and 'HW' not in k}
  if not weights or any(not torch.is_tensor(v) or v.dtype!=torch.float32 or not torch.isfinite(v).all() for v in weights.values()):raise RuntimeError('nonfinite/nonFP32 learned weights')
  net.load_state_dict(weights,strict=True)
  self.model=trainer.Model.__new__(trainer.Model)  # Inference only: no optimizer/LapLoss.
  self.model.net=net.to(device).eval().requires_grad_(False);self.model.name='ours_t'
  self.device=device
  self.metadata=dict(checkpoint_sha256=EXPECTED_CHECKPOINT_SHA,strict_load=True,source_code_commit='75b6f6a889e695df875e103374040d47a4cfac7c',architecture=dict(F=32,depth=[2,2,2,4,4]),parameters=sum(p.numel() for p in net.parameters()),loaded_keys=len(weights),dropped_derived_or_nonmodule_keys=len(raw)-len(weights),torch=torch.__version__,weight_source='thirdpartymirror; matchingLFSoidandWASreport,notindependentauthorsDriveauthentication')
  if self.metadata['parameters']!=65662359:raise RuntimeError('unexpected full ours_t architecture size')
  del raw,weights
  padder=importlib.import_module('benchmark.utils.padder');self.Padder=padder.InputPadder

 @torch.no_grad()
 def interiors(self,rgb0,rgb1):
  """Exactly49 floating RGB images; official API batch1, noTTA/downscale1."""
  if rgb0.shape!=rgb1.shape or rgb0.shape!=(540,960,3) or rgb0.dtype!=np.uint8 or rgb1.dtype!=np.uint8:raise ValueError('native lossless decoded RGB expected')
  def image(rgb):return torch.from_numpy(rgb[:,:,::-1].copy()).permute(2,0,1).unsqueeze(0).to(self.device,dtype=torch.float32)/255.
  left,right=image(rgb0),image(rgb1);assert left.shape[0]==right.shape[0]==1
  padder=self.Padder(left.shape,divisor=32);left,right=padder.pad(left,right)
  predictions=self.model.multi_inference(left,right,TTA=False,down_scale=1.,time_list=[j/50 for j in range(1,50)],fast_TTA=False)
  if len(predictions)!=49:raise RuntimeError('EMA dropped timesteps')
  for pred in predictions:
   rgb=padder.unpad(pred).flip(0).permute(1,2,0).contiguous()
   if rgb.shape!=(540,960,3) or not torch.isfinite(rgb).all():raise RuntimeError('bad EMA prediction')
   low=float(rgb.min());high=float(rgb.max());count=int(((rgb<0)|(rgb>1)).sum())
   yield rgb.clamp(0,1),dict(overshoot_values=count,min_before_clamp=low,max_before_clamp=high)
