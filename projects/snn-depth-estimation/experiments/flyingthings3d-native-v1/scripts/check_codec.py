"""Tiny synthetic CPU tests only, no datasetconversion or interpolation."""
import json,sys,tempfile
from pathlib import Path
import numpy as np
import torch
from spike_codec import *
project=Path(__file__).resolve().parents[3];sys.path.insert(0,str(project))
from core.utils.reader import get_block_spikes
for i in range(1,9):
 indices=window_indices(i);assert len(indices)==50 and len(set(indices))==50
 assert sum(n<=50*i for n in indices)==25 and sum(n>50*i for n in indices)==25
 assert source_time(indices[0])>i-.5 and source_time(indices[-1])==i+.5
assert window_indices(1)[0]==26 and window_indices(8)[-1]==425
# Continuous state across intervals/windows; not reset per targetwindow.
v=np.zeros((2,8),np.float32);t=torch.zeros(2,8);spikes=[]
for n in range(1,451):
 intensity=np.full((2,8),.5,np.float32);s,v=integrate_numpy(v,intensity);st,t=integrate_torch(t,torch.from_numpy(intensity));assert np.array_equal(s,st.numpy()) and np.array_equal(v,t.numpy());assert np.isfinite(v).all() and v.min()>=0 and v.max()<5
 spikes.append(s)
spikes=np.stack(spikes);assert np.flatnonzero(spikes[:,0,0]).tolist()==list(range(9,450,10))
window=spikes[np.array(window_indices(1))-1]
with tempfile.TemporaryDirectory() as tmp:
 p=Path(tmp)/'spikes.dat';atomic_dat(p,window);decoded=get_block_spikes(str(p),spike_height=2,spike_width=8,block_len=50,flipud=False);assert np.array_equal(decoded,window)
probe=np.array([[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]],np.float32);assert np.array_equal(rgb_to_gray(probe),np.array([[.299,.587,.114]],np.float32))
result=dict(centered50bins=True,past_including_target=25,future_bins=25,normalized_source_times=True,continuous_if_numpy_torch_exact=True,threshold5_no_dt_scaling=True,accumulator_finite_bounded=True,bt601_rgb_coefficients=True,pack_unpack_existing_reader_exact=True,packing='chronologicalrowmajorLSBfirst,headerless,noflip',scope='tinyCPUonly,noEMA/data/GPUconversion')
out=Path(__file__).parents[1]/'results/codec-checks.json';out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
