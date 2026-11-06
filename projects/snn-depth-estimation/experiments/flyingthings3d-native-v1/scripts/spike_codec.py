"""Explicit centered-time indexing and lossless A2.3-style spike codec."""
from pathlib import Path
import numpy as np


def window_indices(center):
 if center not in range(1,9):raise ValueError('actual centers must be1..8 in10frames')
 return list(range(50*center-24,50*center+26))


def source_time(global_index):
 if not 1<=global_index<=450:raise ValueError(global_index)
 return global_index/50


def rgb_to_gray(rgb):
 """Normalized encoded RGB; no gamma conversion/contrast/noise."""
 if rgb.shape[-1]!=3:raise ValueError(rgb.shape)
 rgb=np.asarray(rgb,dtype=np.float32)
 if not np.isfinite(rgb).all() or rgb.min()<0 or rgb.max()>1:raise ValueError('RGBmustbefinite[0,1]')
 return np.float32(.299)*rgb[...,0]+np.float32(.587)*rgb[...,1]+np.float32(.114)*rgb[...,2]


def integrate_numpy(voltage,intensity):
 """Tiny CPU reference; substantial conversion belongs on authorized GPU."""
 charged=voltage+intensity;spike=charged>=np.float32(5.)
 return spike,charged-np.float32(5.)*spike.astype(np.float32)


def integrate_torch(voltage,intensity):
 """FP32 per-view accumulator; no gradients or per-window reset."""
 charged=voltage+intensity;spike=charged>=5.
 return spike,charged-5.*spike.to(charged.dtype)


def pack_spikes(spikes):
 x=np.asarray(spikes)
 if x.ndim!=3 or x.shape[0]!=50 or not np.isin(x,[0,1]).all():raise ValueError('expectedbinary[50,H,W]')
 if x.size%8:raise ValueError('bytealignedshape required')
 return np.packbits(x.reshape(-1),bitorder='little').tobytes()


def unpack_spikes(data,height,width):
 expected=50*height*width//8
 if 50*height*width%8 or len(data)!=expected:raise ValueError('exactbytecountrequired')
 return np.unpackbits(np.frombuffer(data,dtype=np.uint8),bitorder='little').reshape(50,height,width).astype(bool)


def atomic_dat(path,spikes):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 data=pack_spikes(spikes);temporary=path.with_suffix(path.suffix+'.part');temporary.write_bytes(data)
 assert np.array_equal(unpack_spikes(data,*spikes.shape[1:]),spikes)
 temporary.replace(path)
