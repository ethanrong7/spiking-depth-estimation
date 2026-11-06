"""Complete original spike model and untouched native-RGB official architectures."""
import sys
from pathlib import Path
from types import SimpleNamespace
import torch
PROJECT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(PROJECT))
from core.spikestereonet import SpikeStereoNet
from core.utils.utils import InputPadder
from spikingjelly.activation_based import functional


def build_model(name,seed):
 torch.manual_seed(seed)
 args=SimpleNamespace(hidden_dims=[128]*3,hidden_dim=128,context_norm='batch',shared_backbone=False,n_gru_layers=3,n_snn_layers=3,n_downsample=2,corr_levels=4,corr_radius=4,corr_implementation='reg',mixed_precision=False,precision_dtype='float32',slow_fast_gru=False,feature_encoder='cnn',context_encoder='cnn')
 if name=='original':model=SpikeStereoNet(args);channels=50
 elif name=='raft_stereo':
  from vendor.raft_stereo_rgb.net.raft_stereo import RAFTStereo
  model=RAFTStereo(args);channels=3
 elif name=='selective_raft':
  from vendor.selective_raft_rgb.net.raft import RAFT
  model=RAFT(args);channels=3
 else:raise ValueError(name)
 assert model.fnet.conv1.in_channels==model.cnet.conv1.in_channels==channels
 return model


def reset_model(model):functional.reset_net(model)


def forward(model,name,left,right,iters,training=False):
 reset_model(model)
 expected=50 if name=='original' else 3
 if left.shape!=right.shape or left.shape[1]!=expected:raise ValueError('native modality/shape mismatch')
 if not torch.isfinite(left).all() or not torch.isfinite(right).all():raise ValueError('nonfinite inputs')
 padder=InputPadder(left.shape,divis_by=32);left,right=padder.pad(left,right)
 result=model(left,right,iters=iters,test_mode=not training)
 if training:preds=result[0] if name=='original' else result;return [padder.unpad(p) for p in preds]
 return padder.unpad(result[1])
