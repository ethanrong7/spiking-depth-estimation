"""Full official architectures with isolated 50-spike-channel input adapters."""
from types import SimpleNamespace
import torch
from torch import nn

class ConventionalAdapter(nn.Module):
 """Return-interface adapter only: no spike/voltage diagnostics are invented."""
 def __init__(self,model):super().__init__();self.model=model
 @property
 def update_block(self):return self.model.update_block
 def freeze_bn(self):self.model.freeze_bn()
 def forward(self,left,right,iters=16,test_mode=False):
  result=self.model(left,right,iters=iters,test_mode=test_mode)
  if test_mode:
   low,final=result
   return low,final,[]
  return result,[],[]


def build_conventional(name,seed):
 torch.manual_seed(seed)
 args=SimpleNamespace(hidden_dims=[128]*3,hidden_dim=128,context_norm='batch',shared_backbone=False,n_gru_layers=3,n_downsample=2,corr_levels=4,corr_radius=4,corr_implementation='reg',mixed_precision=False,precision_dtype='float32',slow_fast_gru=False)
 if name=='raft_stereo':
  from vendor.raft_stereo.net.raft_stereo import RAFTStereo
  model=RAFTStereo(args)
 elif name=='selective_raft':
  from vendor.selective_raft.net.raft import RAFT
  model=RAFT(args)
 else:raise ValueError(name)
 assert model.fnet.conv1.in_channels==model.cnet.conv1.in_channels==50
 return ConventionalAdapter(model)
