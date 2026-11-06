"""Independent registered paired-pilot integrity audit; never loads a model/Test.

Bulk array/source-payload checks require a single authorized Slurm RTX GPU
allocation (CPU decoding only). --metadata-only permits cluster read-only
inspection without an allocation and opens no sensor/GT arrays. Output must be
outside the immutable dataset/source trees. No jobs are submitted here.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import time

import numpy as np


class AuditDeadline(TimeoutError):
    pass


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def safe(root,name):
    name=Path(name);root=Path(root).resolve()
    if name.is_absolute() or '..' in name.parts or 'TEST' in name.parts:
        raise ValueError('Audit assets must remain relative and never reference Test')
    result=(root/name).resolve()
    if not result.is_relative_to(root):raise ValueError('Asset escapes configured root')
    return result


def validate_metadata(manifest,config,index):
    if (manifest.get('schema')!='stereo-paired-engineering-pilot-v1'
            or manifest.get('evidence_scope')!='engineering_only' or manifest.get('test')!=[]
            or manifest.get('T')!=50 or manifest.get('time_unit')!='source_frame_interval'
            or manifest.get('rectified') is not True or manifest.get('disparity')!='positive_left_pixels'):
        raise ValueError('Expected causal engineering Train/Val only, all50 bins and positive left disparity')
    scale=config.get('output_scale',1.)
    shape=[int(540*scale),int(960*scale)]
    if (scale not in (1.,.5,.25) or manifest.get('output_scale',1.)!=scale
            or manifest.get('output_shape',shape)!=shape):raise ValueError('Registered sensor resolution changed')
    for modality in ('spike','event'):
        if manifest['simulators'][modality]['configuration']!=config['sensor']:
            raise ValueError('Sensor configuration metadata changed')
    sources={s['source_id']:s for s in index['sequences']}
    expected={};expected_sources=set();owners={}
    for request in config['sequences']:
        sid=request['source_id'];split=request['split']
        if not sid.startswith('TRAIN/') or split not in ('train','val') or sid in owners:
            raise ValueError('Repeated, Test or invalid registered scene')
        owners[sid]=split;scene=sources[sid]
        for frame in scene['frames']:
            expected_sources.update((frame['rgb_l'],frame['rgb_r']))
        for center in request['centers']:
            frame=scene['frames'][center];identifier=sid+'/'+frame['source_frame']
            expected[identifier]=(split,sid,center,frame)
            expected_sources.add(frame['disparity_l'])
    actual={};counts={}
    for split in ('train','val'):
        for row in manifest[split]:
            sid=row['scene_id'];identifier=row['sample_id']
            if identifier in actual or identifier not in expected:raise ValueError('Repeated/unregistered sample: '+identifier)
            wanted,sid_expected,center,frame=expected[identifier]
            if (split!=wanted or row['split']!=split or sid!=sid_expected or row['source_frame']!=frame['source_frame']
                    or row.get('original_split')!=sources[sid]['split']):raise ValueError('Scene/frame/split mapping changed')
            edges=np.asarray(row['bin_edges'],dtype=np.float64)
            if (edges.shape!=(51,) or not np.allclose(edges,np.linspace(center-1,center,51),rtol=0,atol=1e-12)
                    or row['target_time']!=center or row['source_support_end']!=center
                    or row.get('sensor_state_start')!=0 or row.get('window_start')!=center-1
                    or row.get('window_end')!=center or row.get('future_context') is not False):
                raise ValueError('Causal50-bin/initial-state metadata changed')
            if (row.get('event_channels')!=['ON','OFF'] or row.get('native_shape')!=[540,960]
                    or row.get('observation_shape',shape)!=shape or row.get('output_scale',scale)!=scale
                    or row.get('disparity_unit','output_horizontal_pixels')!='output_horizontal_pixels'):
                raise ValueError('Polarity/spatial/disparity-unit metadata changed')
            for field,suffix in [('spike','-spike.npz'),('event','-event.npz'),('disparity','-disparity.npy')]:
                if row[field]!=identifier+suffix:raise ValueError('Asset/target filename mapping changed')
            actual[identifier]=(row,center,frame)
        counts[split]=dict(samples=len(manifest[split]),scenes=len({r['scene_id'] for r in manifest[split]}))
    if set(actual)!=set(expected):raise ValueError('Incomplete registered samples')
    for split in ('train','val'):
        if 'sample_counts' in config and counts[split]['samples']!=config['sample_counts'][split]:raise ValueError('Registered sample count changed')
        if 'scene_counts' in config and counts[split]['scenes']!=config['scene_counts'][split]:raise ValueError('Registered scene count changed')
    return actual,owners,shape,expected_sources,counts


def validate_trace(trace,threshold):
    result={}
    if len(trace)!=900:raise ValueError('Each scene needs450 trace records per view')
    for view in ('l','r'):
        rows=[r for r in trace if r['view']==view]
        if [r['global_index'] for r in rows]!=list(range(1,451)):raise ValueError('Dropped/reordered sensor-time steps')
        np.testing.assert_allclose([r['source_time'] for r in rows],np.arange(1,451)/50,rtol=0,atol=1e-9)
        values=np.asarray([[r[k] for k in ('spike_fraction','on_events','off_events','charge_min','charge_max','charge_mean')] for r in rows],dtype=np.float64)
        if (not np.isfinite(values).all() or np.any(values[:,0]<0) or np.any(values[:,0]>1)
                or np.any(values[:,1:3]<0) or np.any(values[:,1:3]!=np.floor(values[:,1:3]))
                or np.any(values[:,3]<0) or np.any(values[:,4]>=threshold)
                or np.any(values[:,5]<values[:,3]) or np.any(values[:,5]>values[:,4])):
            raise ValueError('Nonfinite/out-of-range simulator trace')
        result[view]=rows
    return result


def check_sensor_array(array,modality,shape,trace_rows):
    channels=1 if modality=='spike' else 2
    if array.shape!=(50,channels,*shape) or not (np.issubdtype(array.dtype,np.number) or array.dtype==np.bool_):raise ValueError('Sensor tensor shape/dtype changed')
    if not np.isfinite(array).all() or np.any(array<0):raise ValueError('Invalid sensor values')
    if np.issubdtype(array.dtype,np.floating) and np.any(array!=np.floor(array)):raise ValueError('Nonintegral sensor counts')
    if modality=='spike':
        if np.any(array>1):raise ValueError('Intensity spikes are not binary')
        actual=array.sum(axis=(1,2,3),dtype=np.int64)
        # FP32 mean rounding is <half a spike at these registered resolutions.
        expected=np.rint(np.asarray([r['spike_fraction'] for r in trace_rows])*shape[0]*shape[1]).astype(np.int64)
        np.testing.assert_array_equal(actual,expected)
        return dict(total_spikes=int(actual.sum()))
    actual=array.sum(axis=(2,3),dtype=np.int64)
    expected=np.asarray([[r['on_events'],r['off_events']] for r in trace_rows],dtype=np.int64)
    np.testing.assert_array_equal(actual,expected)
    return dict(total_on=int(actual[:,0].sum()),total_off=int(actual[:,1].sum()),max_count=int(array.max()))


def source_disparity(path,shape):
    with Path(path).open('rb') as stream:
        if stream.readline().strip()!=b'Pf':raise ValueError('Original positive left PFM required')
        width,height=map(int,stream.readline().split());scale=float(stream.readline())
        if (height,width)!=(540,960) or abs(scale)!=1:raise ValueError('Unexpected original PFM geometry/scale')
        data=np.frombuffer(stream.read(),dtype='<f4' if scale<0 else '>f4')
        if data.size!=height*width:raise ValueError('PFM payload length changed')
    original=np.flipud(data.reshape(height,width)).astype(np.float32)
    # Independent nearest-exact pixel-center selection, not the generator helper.
    yy=np.floor((np.arange(shape[0])+.5)*height/shape[0]).astype(int)
    xx=np.floor((np.arange(shape[1])+.5)*width/shape[1]).astype(int)
    return original[yy[:,None],xx[None,:]]*np.float32(shape[1]/width)


def audit(args):
    started=time.monotonic();report=dict(status='STARTED',mode='metadata_only' if args.metadata_only else 'all_registered_assets',
        manifest_sha256=args.manifest_sha256,samples_checked=0,source_payloads_hashed=0,test_access=False,
        model_inference=False,gpu_compute=False,dataset_outputs_modified=False,max_seconds=args.max_seconds)
    root=args.data_root.resolve();output=args.output.resolve();source_root=args.source_root.resolve()
    if output.is_relative_to(root) or output.is_relative_to(source_root) or output.is_relative_to(Path(__file__).resolve().parents[5]):raise ValueError('Report must be outside readonly data/source roots')
    if output.exists():raise FileExistsError('Never overwrite an existing audit report')
    previous=None
    try:
        if not 0<args.max_seconds<=360:raise ValueError('Audit cap is360 seconds')
        config=json.loads(args.config.read_text())
        if socket.gethostname()!=config['expected_hostname']:raise RuntimeError('Audit runs only on registered cluster')
        if not args.metadata_only:
            if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Bulk audit requires authorized Slurm allocation')
            import torch
            if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or not any(x in torch.cuda.get_device_name(0) for x in ('4090','5090')):
                raise RuntimeError('Bulk audit requires one allocated RTX4090/5090')
            report.update(slurm_job_id=os.environ['SLURM_JOB_ID'],gpu=torch.cuda.get_device_name(0))
        def expired(number=None,frame=None):raise AuditDeadline('Bounded audit deadline; incomplete is not PASS')
        previous=signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,max(.1,args.max_seconds-10-(time.monotonic()-started)))
        manifest_path=root/'manifest.json'
        if file_sha(manifest_path)!=args.manifest_sha256:raise ValueError('Registered manifest identity changed')
        manifest=json.loads(manifest_path.read_text());result=json.loads((root/'result.json').read_text());registration=json.loads((root/'registration.json').read_text())
        if result['status']!='COMPLETED' or result['manifest_sha256']!=args.manifest_sha256 or result.get('test_samples')!=0 or result.get('all450_samples_processed_per_view') is not True or result.get('training_performed') is not False:
            raise ValueError('Generation is incomplete or manifest receipt differs')
        if registration['config']!=config or manifest['configuration_sha256']!=file_sha(args.config):raise ValueError('Generation/config binding mismatch')
        if file_sha(args.source_index)!=config['source_index_sha256'] or manifest['source_identity']!=config['source_index_sha256']:
            raise ValueError('Original source index changed')
        if args.generation_source_registry:
            registry=json.loads(args.generation_source_registry.read_text())
            if (file_sha(args.generation_source_registry)!=manifest['source_registry_sha256']
                    or registry['git_commit']!=manifest['git_commit'] or registry['source_identity_sha256']!=manifest['source_overlay_identity']):
                raise ValueError('Immutable generation source binding mismatch')
        actual,owners,shape,expected_sources,counts=validate_metadata(manifest,config,json.loads(args.source_index.read_text()))
        files=json.loads((root/'source-files.json').read_text());by_path={r['path']:r for r in files}
        if len(files)!=len(by_path) or set(by_path)!=expected_sources:raise ValueError('Generation source-file receipt incomplete/repeated')
        if any(len(r.get('sha256',''))!=64 or r['bytes']<=0 for r in files):raise ValueError('Invalid source-file identity receipt')
        traces={};report.update(counts=counts,shape=shape,T=50,configuration_sha256=file_sha(args.config),
            source_index_sha256=config['source_index_sha256'],generation_job_id=result['job_id'],trace_steps_checked=0,
            source_files_registered=len(files),spike_total=0,event_on_total=0,event_off_total=0,event_max_count=0,
            gt_valid_pixels=0,gt_total_pixels=0,gt_nonfinite_pixels=0,gt_nonpositive_finite_pixels=0,source_PFM_equal_samples=0)
        for sid in owners:
            traces[sid]=validate_trace(json.loads(safe(root,sid+'/trace.json').read_text()),config['sensor']['spike_threshold'])
            report['trace_steps_checked']+=900
        if args.metadata_only:
            report['status']='PASS_METADATA_ONLY_ARRAYS_NOT_AUDITED'
            for key in ('spike_total','event_on_total','event_off_total','event_max_count','gt_valid_pixels','gt_total_pixels',
                        'gt_nonfinite_pixels','gt_nonpositive_finite_pixels'):
                report[key]=None
        else:
            for identifier,(row,center,frame) in actual.items():
                report['current_sample']=identifier
                for modality in ('spike','event'):
                    with np.load(safe(root,row[modality]),allow_pickle=False) as packed:
                        if set(packed.files)!={'left','right'}:raise ValueError('Unexpected sensor asset keys')
                        for key,view in [('left','l'),('right','r')]:
                            window=traces[row['scene_id']][view][50*(center-1):50*center]
                            values=check_sensor_array(packed[key],modality,shape,window)
                            if modality=='spike':report['spike_total']+=values['total_spikes']
                            else:
                                report['event_on_total']+=values['total_on'];report['event_off_total']+=values['total_off']
                                report['event_max_count']=max(report['event_max_count'],values['max_count'])
                target=np.load(safe(root,row['disparity']),allow_pickle=False)
                if target.shape!=tuple(shape) or target.dtype!=np.float32:raise ValueError('Saved disparity shape/dtype changed')
                valid=np.isfinite(target)&(target>0)
                if not valid.any():raise ValueError('No finite positive GT support')
                expected=source_disparity(safe(source_root,frame['disparity_l']),shape)
                if not np.array_equal(target,expected,equal_nan=True):raise ValueError('PFM orientation/nearest-selection/horizontal disparity scaling mismatch')
                report['source_PFM_equal_samples']+=1;report['gt_valid_pixels']+=int(valid.sum());report['gt_total_pixels']+=target.size
                report['gt_nonfinite_pixels']+=int((~np.isfinite(target)).sum())
                report['gt_nonpositive_finite_pixels']+=int((np.isfinite(target)&(target<=0)).sum())
                report['samples_checked']+=1
            for record in files:
                path=safe(source_root,record['path'])
                if path.stat().st_size!=record['bytes'] or file_sha(path)!=record['sha256']:raise ValueError('Original source payload changed: '+record['path'])
                report['source_payloads_hashed']+=1
            report['gt_coverage']=report['gt_valid_pixels']/report['gt_total_pixels']
            report['status']='PASS_ALL_REGISTERED_ASSETS_TRACES_AND_SOURCE_IDENTITIES'
    except AuditDeadline as error:
        report.update(status='INCOMPLETE_TIME_CAP',error=str(error))
    except Exception as error:
        report.update(status='FAILED_INTEGRITY',error=repr(error))
    finally:
        if previous is not None:
            signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
        report['elapsed_seconds']=time.monotonic()-started
        report['limitations']=['No neural-interpolator replay or visual stereo-fidelity verdict','No physical sensor calibration claim']
        output.parent.mkdir(parents=True,exist_ok=True)
        with output.open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data-root','config','source-index','source-root','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--manifest-sha256',required=True)
    p.add_argument('--generation-source-registry',type=Path)
    p.add_argument('--max-seconds',type=int,default=360)
    p.add_argument('--metadata-only',action='store_true')
    args=p.parse_args();report=audit(args)
    print(json.dumps(report,allow_nan=False))
    if not report['status'].startswith('PASS_'):raise SystemExit(1)

if __name__=='__main__':main()
