"""Small synthetic CPU diagnostic, not training or an accelerator-utilization test."""
import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1])]
from input_pipeline import read_spike_bytes, prepare_batch
from core.utils.reader import get_block_spikes
from speed_runtime import SpeedRuntime
from test_speed_runtime import legacy_loss


def timed(fn, repeats):
    fn()
    values=[]
    for _ in range(repeats):
        start=time.perf_counter(); fn(); values.append(time.perf_counter()-start)
    return dict(median_seconds=statistics.median(values), samples_seconds=values)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args()
    if args.repeats<1:parser.error('repeats must be positive')
    torch.set_num_threads(2);torch.manual_seed(42)
    target=torch.randn(4,1,250,400)
    valid=torch.rand(4,250,400)>.2
    preds=[torch.randn_like(target,requires_grad=True) for _ in range(16)]
    old=legacy_loss(); fast=SpeedRuntime()
    def run(optimized):
        for p in preds:p.grad=None
        if optimized:
            loss,okay=fast.loss(preds,target,valid)
            loss.backward()
            fast.summarize([loss.detach()],[okay],torch.tensor(1.),{})
        else:
            loss=old(preds,target,valid)
            assert torch.isfinite(loss)
            loss.backward();float(loss.detach())
    result=dict(scope='synthetic CPU only; one microbatch, 16 predictions, no model forward',
                torch_version=torch.__version__,threads=2,shape=list(target.shape),loss={})
    for name,optimized in [('legacy',False),('speed',True)]:
        result['loss'][name]=timed(lambda:run(optimized),args.repeats)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            run(optimized)
        events={e.key:e for e in profile.key_averages()}
        result['loss'][name]['operator_counts']={key:events[key].count if key in events else 0 for key in
            ('aten::_local_scalar_dense','aten::nonzero','aten::index','aten::where')}
        result['loss'][name]['cpu_profile']=profile.key_averages().table(sort_by='self_cpu_time_total',row_limit=12)
    # Actual native file format and dimensions, synthetic bytes only.
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)/'spikes.dat'
        path.write_bytes(np.random.default_rng(42).integers(0,256,625000,dtype=np.uint8).tobytes())
        result['decode']={
            'legacy':timed(lambda:get_block_spikes(str(path)).astype(np.float32),args.repeats),
            'compact':timed(lambda:read_spike_bytes(path),args.repeats)}
        spikes=read_spike_bytes(path)
        class Dataset:
            def __getitem__(self,index):return spikes,spikes,target[0],valid[0],index
        result['cached_batch_preparation']=timed(lambda:prepare_batch(Dataset(),list(range(8)),0,1234),args.repeats)
    result['loss_cpu_speedup']=result['loss']['legacy']['median_seconds']/result['loss']['speed']['median_seconds']
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='loss'},indent=2))
    for name,values in result['loss'].items():
        print(name,values['median_seconds'],values['operator_counts'])
    print('Saved',args.output)


if __name__=='__main__':main()
