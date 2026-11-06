"""Read-only native PFM numeric QA; cluster CPU only, no filtering or model work."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import platform
import resource
import socket
import sys
import time
import numpy as np

QUANTILES=[.01,.05,.25,.5,.75,.95,.99,.999]
COUNTS=['pixels','finite','finite_positive','nan','positive_inf','negative_inf','zero','finite_negative','over_192','over_512','over_700','outside_right_fov']


def decode_pfm(payload):
    stream=io.BytesIO(payload)
    if stream.readline().strip()!=b'Pf':raise ValueError('single-channel PFM required')
    width,height=map(int,stream.readline().split());scale=float(stream.readline().strip())
    if width<=0 or height<=0 or not math.isfinite(scale) or abs(scale)!=1:raise ValueError('invalid PFM geometry/scale')
    raw=stream.read()
    if len(raw)!=width*height*4:raise ValueError('PFM payload length mismatch')
    stored=np.frombuffer(raw,dtype='<f4' if scale<0 else '>f4').reshape(height,width)
    decoded=stored[::-1,:].astype(np.float32,copy=True)
    return decoded,dict(width=width,height=height,scale=scale,endian='little' if scale<0 else 'big')


def empty():
    return dict(counts={key:0 for key in COUNTS},minimum_finite=None,maximum_finite=None,minimum_positive=None,maximum_positive=None)


def merge(target,other):
    for key in COUNTS:target['counts'][key]+=other['counts'][key]
    for key,operation in [('minimum_finite',min),('maximum_finite',max),('minimum_positive',min),('maximum_positive',max)]:
        value=other[key]
        if value is not None:target[key]=value if target[key] is None else operation(target[key],value)


def measure(d):
    finite=np.isfinite(d);valid=finite&(d>0);values=d[valid];stats=empty();c=stats['counts']
    c.update(pixels=int(d.size),finite=int(finite.sum()),finite_positive=int(valid.sum()),nan=int(np.isnan(d).sum()),positive_inf=int(np.isposinf(d).sum()),negative_inf=int(np.isneginf(d).sum()),zero=int((d==0).sum()),finite_negative=int((finite&(d<0)).sum()))
    if finite.any():stats.update(minimum_finite=float(d[finite].min()),maximum_finite=float(d[finite].max()))
    if values.size:
        stats.update(minimum_positive=float(values.min()),maximum_positive=float(values.max()))
        for threshold in [192,512,700]:c[f'over_{threshold}']=int((values>threshold).sum())
        right_x=np.arange(d.shape[1],dtype=np.float32)[None,:]-d
        c['outside_right_fov']=int((valid&((right_x<0)|(right_x>d.shape[1]-1))).sum())
    return stats,values


def fractions(stats):
    result=dict(stats);c=stats['counts'];valid=c['finite_positive'];total=c['pixels']
    result['finite_coverage']=c['finite']/total
    result['finite_positive_coverage']=valid/total
    result['fractions_among_finite_positive']={key:c[key]/valid if valid else None for key in ['over_192','over_512','over_700','outside_right_fov']}
    return result


def approximate_quantiles(hist,width):
    n=int(hist.sum());cumulative=np.cumsum(hist);result={}
    if not n:return result
    for q in QUANTILES:
        h=(n-1)*q;lo_rank=int(math.floor(h))+1;hi_rank=int(math.ceil(h))+1;weight=h-math.floor(h)
        left=int(np.searchsorted(cumulative,lo_rank));right=int(np.searchsorted(cumulative,hi_rank))
        low=np.nextafter(((1-weight)*left+weight*right)*width,-np.inf)
        high=np.nextafter(((1-weight)*(left+1)+weight*(right+1))*width,np.inf)
        result[str(q)]=dict(estimate=float((low+high)/2),lower_bound=float(low),upper_bound=float(high),absolute_error_bound=float((high-low)/2))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['manifest','source-root','download-receipt','comparison-reader','output-root']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--expected-host',required=True);p.add_argument('--manifest-sha256',required=True);p.add_argument('--comparison-reader-sha256',required=True)
    p.add_argument('--histogram-target-width',type=float,default=1/64);p.add_argument('--max-histogram-bins',type=int,default=1048576);a=p.parse_args()
    if platform.system()!='Linux' or socket.gethostname()!=a.expected_host:raise RuntimeError('numeric dataset QA is cluster CPU only')
    source=a.source_root.resolve();output=a.output_root.resolve()
    if source==output or source in output.parents:raise ValueError('QA output must be outside source data')
    if hashlib.sha256(a.manifest.read_bytes()).hexdigest()!=a.manifest_sha256:raise ValueError('frozen manifest changed')
    if hashlib.sha256(a.comparison_reader.read_bytes()).hexdigest()!=a.comparison_reader_sha256:raise ValueError('comparison reader changed')
    output.mkdir(parents=True,exist_ok=False);started=time.monotonic();state=dict(status='RUNNING',phase='DECODE_AND_EXACT_STATS',pid=os.getpid(),command=sys.argv,source_root=str(source),manifest_sha256=a.manifest_sha256)
    def save():
        state['updated_utc']=datetime.now(timezone.utc).isoformat();temp=output/'progress.tmp';temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(output/'progress.json')
    save()
    try:
        manifest=json.loads(a.manifest.read_text());receipt=json.loads(a.download_receipt.read_text());assert receipt['manifest_sha256']==a.manifest_sha256 and receipt['status']=='COMPLETE'
        spec=importlib.util.spec_from_file_location('comparison_source_io',a.comparison_reader);reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
        all_stats=empty();split_stats={s:empty() for s in ['train','dev']};scenes=[];endians=Counter();scales=Counter();files=0;comparisons=0
        for sequence in manifest['sequences']:
            scene=empty();arrays=[];references=[]
            for frame in sequence['frames']:
                path=frame['disparity_l']
                if path is None:continue
                relative=Path(path);assert not relative.is_absolute() and '..' not in relative.parts and path.startswith('disparity/TRAIN/C/')
                file=source/relative;payload=file.read_bytes();assert hashlib.sha256(payload).hexdigest()==receipt['completed'][path]['sha256']
                d,metadata=decode_pfm(payload);assert d.shape==(540,960) and (metadata['width'],metadata['height'])==(960,540)
                comparison,_=reader.read_disparity_pfm(file);assert np.array_equal(d,comparison,equal_nan=True)
                comparisons+=1;stats,values=measure(d);merge(scene,stats);arrays.append(values);references.append(frame['source_frame']);files+=1;endians[metadata['endian']]+=1;scales[str(metadata['scale'])]+=1
            expected=8 if sequence['split']=='train' else 2;assert len(arrays)==expected
            values=np.concatenate(arrays);quantiles={str(q):float(v) for q,v in zip(QUANTILES,np.quantile(values,QUANTILES,method='linear'))} if values.size else {}
            merge(split_stats[sequence['split']],scene);merge(all_stats,scene);scenes.append(dict(source_id=sequence['source_id'],split=sequence['split'],source_frames=references,**fractions(scene),quantiles_exact_linear=quantiles))
            state.update(files=files,scenes=len(scenes));save();print('decoded',files,'scenes',len(scenes),flush=True)
        assert files==1020 and len(scenes)==150 and comparisons==1020
        maximum=all_stats['maximum_positive'];assert maximum is not None and maximum>0
        bins=min(a.max_histogram_bins,max(1,math.ceil(maximum/a.histogram_target_width)))
        upper=float(np.nextafter(float(maximum),np.inf));width=upper/bins;histograms={s:np.zeros(bins,dtype=np.int64) for s in ['train','dev']}
        state.update(phase='POOLED_HISTOGRAM_QUANTILES',histogram_bins=bins,histogram_width=width);save()
        for sequence in manifest['sequences']:
            for frame in sequence['frames']:
                if frame['disparity_l'] is None:continue
                d,_=decode_pfm((source/frame['disparity_l']).read_bytes());values=d[np.isfinite(d)&(d>0)]
                indices=np.floor(values.astype(np.float64)/width).astype(np.int64);np.clip(indices,0,bins-1,out=indices)
                histograms[sequence['split']]+=np.bincount(indices,minlength=bins)
        aggregate={}
        for split,stats in list(split_stats.items())+[('all',all_stats)]:
            histogram=histograms[split] if split!='all' else histograms['train']+histograms['dev'];assert int(histogram.sum())==stats['counts']['finite_positive']
            aggregate[split]=dict(**fractions(stats),quantiles_histogram_linear=approximate_quantiles(histogram,width))
        result=dict(status='PASS',manifest_sha256=a.manifest_sha256,source_root=str(source),files=files,scenes=150,comparison_reader_equal_arrays=comparisons,comparison_reader_sha256=a.comparison_reader_sha256,endian_counts=dict(endians),scale_counts=dict(scales),geometry=[960,540],aggregate=aggregate,per_scene=scenes,
            methodology=dict(valid_mask='isfinite(d) & (d > 0), no upper-disparity cutoff',threshold_comparisons='strict d>192/512/700 among finite-positive pixels',outside_native_right_fov='left reference: x_right=x_left-d; outside iff x_right<0 or x_right>959, among finite-positive pixels; reported separately, not removed from valid mask',orientation='PFM stored bottom-up; decode rows with flipud to top-down; equality to separately pinned converter reader checked for all1020arrays. Numerical QA establishes parser agreement, not visual cross-camera alignment.',per_scene_quantiles='exact NumPy quantile(method=linear) over all finite-positive pixels in selected reference frames per scene',pooled_quantiles='two-pass bounded-memory histogram; approximate same linear order-statistic quantile, explicit lower/upper bin bounds and midpoint estimate; no tail clipping',histogram_bins=bins,histogram_width_pixels=width,histogram_maximum=upper,histogram_target_width=a.histogram_target_width,max_histogram_bins=a.max_histogram_bins),
            provenance=dict(command=sys.argv,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),numpy=np.__version__,python=sys.version,hostname=socket.gethostname(),pid=os.getpid(),gpu_used=False,slurm_job=None,elapsed_seconds=time.monotonic()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,thread_environment={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']}))
        (output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');state.update(status='COMPLETE',phase='COMPLETE',result_sha256=hashlib.sha256((output/'result.json').read_bytes()).hexdigest());save();print(json.dumps({'status':'PASS','files':files,'result_sha256':state['result_sha256'],'aggregate':aggregate,'peak_rss_kib':result['provenance']['peak_rss_kib']},indent=2),flush=True)
    except Exception as error:
        state.update(status='FAILED',error_type=type(error).__name__,error=str(error));save();raise


if __name__=='__main__':main()
