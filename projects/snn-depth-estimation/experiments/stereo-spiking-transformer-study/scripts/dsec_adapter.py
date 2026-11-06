"""Prospective DSEC Train-only causal count adapter; no model or job launcher.

--mode preflight inspects source/header/calibration/timestamp metadata and
freezes target identities without opening GT pixels or converting events.
--mode prepare requires a registered configuration, bound preflight manifest,
frozen source registry and one authorized RTX Slurm allocation. Prepared
observations and native GT loading are deliberately separate APIs.
"""
import argparse
import datetime
import hashlib
import io
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import struct
import sys
import time
import zipfile
import zlib

import numpy as np

SCENES = ('thun_00_a', 'interlaken_00_c')
NATIVE_SHAPE = (480, 640)
OUTPUT_SHAPE = (120, 160)
T = 50
BIN_US = 1000
WINDOW_US = 50000


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def safe(root, name):
    base = Path(root).resolve(); relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts or 'test' in [part.lower() for part in relative.parts]:
        raise ValueError('Only relative, source-contained, non-Test assets are permitted')
    path = (base/relative).resolve()
    if not path.is_relative_to(base):
        raise ValueError('Asset escapes configured root')
    return path


def example_config():
    return dict(schema='dsec-causal-adapter-prospective-v1', registered=False,
        protocol_status='prospective_engineering_only', purpose='engineering_transfer_only',
        source_root_env='STUDY_DSEC_SOURCE', source_provenance='provenance.json', expected_hostname='cccl3',
        scenes=list(SCENES), targets_per_scene=10, T=T, bin_us=BIN_US, window_us=WINDOW_US,
        window='[target_us-50000,target_us)', native_shape=list(NATIVE_SHAPE), output_shape=list(OUTPUT_SHAPE),
        scale=.25, event_channels=['ON','OFF'], polarity={'ON':1,'OFF':0},
        spatial_mapping='q=(rectified+.5)/4-.5;nearest=floor(q+.5);ties_toward_positive_infinity',
        rectified_support='native_pixel_centers_0_to_width_minus1_and_0_to_height_minus1',
        count_storage_dtype='uint32_checked', prediction_to_native='bilinear_align_corners_false_times4',
        target_selection='sorted_eligible_timestamp_ordinals_floor_i_times_Nminus1_over9',
        max_seconds=120, prepared_version='dsec-quarter-train-transfer-prospective-v1',
        preflight_manifest_env='STUDY_DSEC_PREFLIGHT_MANIFEST', preflight_manifest_sha256=None,
        claim_boundary='prospective_adapter;historically_exposed_officialTrain_only;no_sensor_preference_or_officialTest_score')


def contract(config):
    keys = ('protocol_status','purpose','source_provenance','scenes','targets_per_scene','T','bin_us','window_us','window',
            'native_shape','output_shape','scale','event_channels','polarity','spatial_mapping',
            'rectified_support','count_storage_dtype','prediction_to_native','target_selection','prepared_version')
    return {key: config[key] for key in keys}


def validate_config(config, *, execution=False):
    example = example_config()
    if config.get('schema') != example['schema']:
        raise ValueError('Unknown prospective DSEC adapter schema')
    if contract(config) != contract(example):
        raise ValueError('This adapter preserves the explicit prospective20-target T50 physical/spatial contract')
    if not config.get('source_root_env') or not config.get('expected_hostname'):
        raise ValueError('Source root environment variable and cluster hostname required')
    cap = 600 if execution else 120
    if type(config.get('max_seconds')) is not int or not 10 <= config['max_seconds'] <= cap:
        raise ValueError('Bounded adapter deadline required')
    if execution and (config.get('registered') is not True or not config.get('preflight_manifest_env')
            or not isinstance(config.get('preflight_manifest_sha256'), str)
            or len(config['preflight_manifest_sha256']) != 64):
        raise ValueError('Preparation requires registered=true and exact metadata-only manifest SHA')
    return True


def open_h5(path):
    """Use existing runtime plus optional Blosc registration; never install here."""
    try:
        import hdf5plugin  # noqa: F401 -- registers the official Blosc filter
    except ImportError:
        pass
    import h5py
    return h5py.File(path, 'r')


def h5_event_metadata(path):
    with open_h5(path) as stream:
        if set(stream['events']) != {'x','y','t','p'}:
            raise ValueError('Unexpected official DSEC events group')
        t = stream['events/t']; n = len(t)
        if n < 1 or t.ndim != 1 or not np.issubdtype(t.dtype, np.integer):
            raise ValueError('Nonempty integer event timestamp vector required')
        if any(stream['events/'+key].shape != (n,) for key in ('x','y','p')):
            raise ValueError('Event columns have different lengths')
        offset = int(stream['t_offset'][()])
        result = dict(event_count=n, timestamp_dtype=str(t.dtype), timestamp_chunks=list(t.chunks) if t.chunks else None,
            timestamp_filters=t._filters, t_offset_us=offset, keys=list(stream.keys()),
            ms_index_count=len(stream['ms_to_idx']), endpoint_values_read=2)
        try:
            first, last = int(t[0]), int(t[-1])
        except OSError as error:
            raise RuntimeError('DSEC Blosc timestamp endpoints unreadable; install compatible hdf5plugin in the task runtime') from error
        if first > last:
            raise ValueError('Timestamp endpoint order invalid')
        result.update(first_local_us=first,last_local_us=last,first_global_us=offset+first,last_global_us=offset+last)
        return result


class H5EventWindow:
    """Bounded, lazy HDF5 slicing with global physical microseconds.

    Binary search is bracketed by the official ms_to_idx table. No full time
    vector is materialized. Only requested event slices are decoded. Search
    preserves duplicate timestamps at both causal boundaries.
    """
    def __init__(self, path):
        self.stream = open_h5(path)
        self.offset = int(self.stream['t_offset'][()])
        self.t = self.stream['events/t']; self.index = self.stream['ms_to_idx']
        self.boundary_receipts = []

    def close(self):
        self.stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def lower_bound(self, global_us):
        if type(global_us) is not int:
            raise ValueError('Physical timestamp must be an integer microsecond')
        local = global_us-self.offset; n = len(self.t)
        if local <= 0:
            low = 0
        else:
            ms = local//1000
            low = int(self.index[ms]) if ms < len(self.index) else n
            high = int(self.index[ms+1]) if ms+1 < len(self.index) else n
            if not 0 <= low <= high <= n:
                raise ValueError('Invalid ms_to_idx bounds')
            while low < high:
                middle = (low+high)//2
                if int(self.t[middle]) < local:
                    low = middle+1
                else:
                    high = middle
        previous = int(self.t[low-1])+self.offset if low else None
        following = int(self.t[low])+self.offset if low < n else None
        if ((previous is not None and previous >= global_us)
                or (following is not None and following < global_us)):
            raise ValueError('Actual neighboring timestamps disprove ms_to_idx lower bound')
        self.boundary_receipts.append(dict(boundary_global_us=global_us,index=low,
            previous_global_us=previous,next_global_us=following,lower_bound_verified=True))
        return low

    def iter_window(self, target_us, *, chunk_events=250000):
        if type(target_us) is not int or type(chunk_events) is not int or chunk_events <= 0:
            raise ValueError('Integer target and bounded positive chunk size required')
        first = self.lower_bound(target_us-WINDOW_US); end = self.lower_bound(target_us)
        previous=None
        for start in range(first, end, chunk_events):
            stop = min(start+chunk_events, end)
            events = {key: np.asarray(self.stream['events/'+key][start:stop]) for key in ('x','y','p')}
            times = np.asarray(self.t[start:stop],dtype=np.int64)+self.offset
            if (np.any(np.diff(times) < 0) or (previous is not None and int(times[0])<previous)
                    or np.any(times < target_us-WINDOW_US) or np.any(times >= target_us)):
                raise ValueError('HDF5 event window violates chronological causal bounds')
            previous=int(times[-1])
            events['t'] = times
            yield events


def bin_rectified_counts(events, rectify_map, target_us):
    """Integer counts until explicit checked storage cast; ON1 and OFF0.

    Raw noninteger/invalid coordinates, polarity, nonfinite maps and out-of-view
    rectified coordinates have disjoint counters. Native support is pixel-center
    [0,W-1] x [0,H-1]; quarter deposition uses floor(q+.5), including exact ties
    toward positive infinity. Time is [target-50000,target),50 equal1ms bins.
    Tiny fixtures may provide a smaller native map divisible by4.
    """
    arrays = {key: np.asarray(events[key]) for key in ('x','y','p','t')}
    n = len(arrays['t'])
    if (any(value.shape != (n,) for value in arrays.values()) or type(target_us) is not int
            or not np.issubdtype(arrays['t'].dtype,np.integer)):
        raise ValueError('Equal-length vectors and integer physical timestamps required')
    mapping = np.asarray(rectify_map)
    if mapping.ndim != 3 or mapping.shape[2] != 2 or mapping.shape[0]%4 or mapping.shape[1]%4:
        raise ValueError('Rectification map requires nativeH,W,2 divisible by4')
    h,w = mapping.shape[:2]; oh,ow = h//4,w//4
    counts = np.zeros((T,2,oh,ow),dtype=np.int64)
    stats = dict(input_events=n,time_outside=0,bad_raw_coordinate=0,bad_polarity=0,
                 nonfinite_rectification=0,outside_rectified_support=0,accepted_events=0,on_events=0,off_events=0)
    times=arrays['t'].astype(np.int64,copy=False)
    keep=(times >= target_us-WINDOW_US)&(times < target_us)
    stats['time_outside']=int((~keep).sum())
    x,y,p=arrays['x'],arrays['y'],arrays['p']
    raw=(np.isfinite(x)&np.isfinite(y)&(x==np.floor(x))&(y==np.floor(y))&(x>=0)&(x<w)&(y>=0)&(y<h))
    stats['bad_raw_coordinate']=int((keep&~raw).sum());keep &= raw
    polarity=np.isfinite(p)&((p==0)|(p==1))
    stats['bad_polarity']=int((keep&~polarity).sum());keep &= polarity
    selected=np.flatnonzero(keep)
    coords=mapping[y[selected].astype(np.int64),x[selected].astype(np.int64)].astype(np.float64)
    finite=np.isfinite(coords).all(axis=1)
    stats['nonfinite_rectification']=int((~finite).sum());selected=selected[finite];coords=coords[finite]
    inside=(coords[:,0]>=0)&(coords[:,0]<=w-1)&(coords[:,1]>=0)&(coords[:,1]<=h-1)
    stats['outside_rectified_support']=int((~inside).sum());selected=selected[inside];coords=coords[inside]
    q=np.floor((coords+.5)/4).astype(np.int64)  # floor(q_continuous+.5); exact ties upward
    if np.any(q<0) or np.any(q[:,0]>=ow) or np.any(q[:,1]>=oh):
        raise ValueError('Quarter spatial deposition escaped its registered grid')
    bins=(times[selected]-(target_us-WINDOW_US))//BIN_US
    channels=np.where(p[selected]==1,0,1)
    np.add.at(counts,(bins,channels,q[:,1],q[:,0]),1)
    stats.update(accepted_events=len(selected),on_events=int((channels==0).sum()),off_events=int((channels==1).sum()))
    discarded=sum(stats[k] for k in ('time_outside','bad_raw_coordinate','bad_polarity','nonfinite_rectification','outside_rectified_support'))
    if discarded+stats['accepted_events'] != n or int(counts.sum()) != stats['accepted_events']:
        raise RuntimeError('Event count conservation failed')
    return counts,stats


def checked_uint32(counts):
    values=np.asarray(counts)
    if values.dtype != np.int64 or np.any(values<0) or (values.size and int(values.max())>np.iinfo(np.uint32).max):
        raise ValueError('Lossless uint32 cast requires nonnegative in-range int64 counts')
    return values.astype(np.uint32)


def decode_uint16_gt(encoded):
    values=np.asarray(encoded)
    if values.dtype != np.uint16 or values.ndim != 2:
        raise ValueError('Native official single-channel uint16 disparity required')
    target=values.astype(np.float32)/256.
    valid=(values>0)&np.isfinite(target)
    return target[None],valid[None]


def predict_to_native(prediction, native_shape=NATIVE_SHAPE):
    """Positive disparity convention; finite negative outputs are never clamped."""
    import torch
    import torch.nn.functional as F
    if (prediction.ndim != 4 or prediction.shape[1] != 1
            or tuple(native_shape) != tuple(4*x for x in prediction.shape[-2:])
            or not torch.isfinite(prediction).all()):
        raise ValueError('Finite B,1,H,W prediction and exactly4x native grid required')
    return F.interpolate(prediction,size=tuple(native_shape),mode='bilinear',align_corners=False)*4


def source_receipt(root,path):
    sha=hashlib.sha256();crc=0
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            sha.update(block);crc=zlib.crc32(block,crc)
    return dict(path=str(path.resolve().relative_to(root.resolve())),bytes=path.stat().st_size,
                sha256=sha.hexdigest(),crc32=f'{crc:08x}')


def uniform_eligible_targets(timestamps, names, views, *, count=10):
    stamps=np.asarray(timestamps)
    if (stamps.ndim != 1 or not np.issubdtype(stamps.dtype,np.integer) or len(stamps)!=len(names)
            or np.any(np.diff(stamps)<=0) or len(names)!=len(set(names))):
        raise ValueError('One strictly ordered timestamp per unique sorted GT filename required')
    eligible=[i for i,t in enumerate(stamps) if all(int(t)-WINDOW_US>=v['first_global_us'] and int(t)<=v['last_global_us'] for v in views.values())]
    if len(eligible)<count or count<2:
        raise ValueError('Too few complete stereo causal windows; do not invent target identities')
    positions=[i*(len(eligible)-1)//(count-1) for i in range(count)]
    return [dict(timestamp_ordinal=eligible[j],eligible_ordinal=j,target_us=int(stamps[eligible[j]]),
                 filename=names[eligible[j]]) for j in positions],len(eligible)


def preflight(config,root):
    """No real GT pixels, rectification values or event-window conversion."""
    validate_config(config)
    root=Path(root).resolve()
    result=dict(schema='dsec-causal-preflight-v1',status='PREFLIGHT_STARTED',protocol_status=config['protocol_status'],
        contract=contract(config),contract_sha256=digest(contract(config)),source_files=[],sequences=[],targets=[],
        gt_pixels_opened=False,event_conversion=False,test_access=False,historical_exposure='existing2officialTrain_discovery_sequences',
        dataset_license='CC-BY-SA-4.0',official_format='https://dsec.ifi.uzh.ch/data-format/',
        official_download='https://dsec.ifi.uzh.ch/dsec-datasets/download/',
        availability_basis='conservative_first_and_last_recorded_event_span;internal_dropouts_not_audited',
        accessed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),failures=[])
    import importlib.metadata
    result['runtime_versions']={name:importlib.metadata.version(name) if importlib.util.find_spec(name) else None for name in ('numpy','h5py','hdf5plugin')}
    try:
        provenance_path=safe(root,config['source_provenance'])
        provenance=json.loads(provenance_path.read_text())
        original_receipts={(row['sequence'],row['suffix']):row for row in provenance['files']}
        result['source_files'].append(source_receipt(root,provenance_path))
    except Exception as error:
        result.update(status='BLOCKED_METADATA_NOT_READY',failures=[dict(scope='source_provenance',error=repr(error))])
        return result
    for scene in config['scenes']:
        try:
            folder=safe(root,'download/'+scene)
            files={suffix:folder/(scene+'_'+suffix) for suffix in ('events_left.zip','events_right.zip','disparity_event.zip','calibration.zip','disparity_timestamps.txt')}
            timestamps=np.loadtxt(files['disparity_timestamps.txt'],dtype=np.int64,ndmin=1)
            with zipfile.ZipFile(files['disparity_event.zip']) as archive:
                members=sorted(archive.infolist(),key=lambda info:info.filename)
                names=[info.filename for info in members]
                if any(Path(name).name!=name or len(name)!=10 or not name[:6].isdigit() or not name.endswith('.png') for name in names):
                    raise ValueError('Expected actual bare6digit native disparity PNG members')
                gt_members={info.filename:dict(member=info.filename,bytes=info.file_size,crc32=f'{info.CRC:08x}') for info in members}
            with zipfile.ZipFile(files['calibration.zip']) as archive:
                if set(archive.namelist()) != {'cam_to_cam.yaml','cam_to_lidar.yaml'}:
                    raise ValueError('Actual calibration ZIP layout changed')
                calibration_members=[dict(member=name,sha256=hashlib.sha256(archive.read(name)).hexdigest()) for name in archive.namelist()]
                import yaml
                result['runtime_versions']['PyYAML']=yaml.__version__
                calibration=yaml.safe_load(archive.read('cam_to_cam.yaml'))
                intrinsics={}
                for camera,view in ((0,'left'),(3,'right')):
                    raw=calibration['intrinsics']['cam'+str(camera)]
                    rect=calibration['intrinsics']['camRect'+str(camera)]
                    if (raw['camera_type']!='event' or raw['camera_location']!=view or raw['resolution']!=[640,480]
                            or rect['is_rectified'] is not True or rect['resolution']!=[640,480]
                            or len(rect['camera_matrix'])!=4 or not np.isfinite(rect['camera_matrix']).all()
                            or min(rect['camera_matrix'][:2])<=0):
                        raise ValueError('Actual event-camera calibration mapping changed')
                    intrinsics[view]=rect['camera_matrix']
                q=np.asarray(calibration['disparity_to_depth']['cams_03'])
                if q.shape!=(4,4) or not np.isfinite(q).all() or q[3,2]<=0:
                    raise ValueError('Expected finite event-stereo disparity reprojection matrix')
            views={};view_sources=[]
            for view in ('left','right'):
                event_path=safe(root,'train/'+scene+'/events/'+view+'/events.h5')
                map_path=safe(root,'train/'+scene+'/events/'+view+'/rectify_map.h5')
                if not map_path.is_file():
                    raise FileNotFoundError('Required extracted official rectification map unavailable: '+str(map_path.relative_to(root)))
                with zipfile.ZipFile(files['events_'+view+'.zip']) as archive:
                    if set(archive.namelist()) != {'events.h5','rectify_map.h5'}:
                        raise ValueError('Actual event ZIP layout changed')
                    for name,path in [('events.h5',event_path),('rectify_map.h5',map_path)]:
                        if path.stat().st_size != archive.getinfo(name).file_size:
                            raise ValueError('Extracted file size differs from source ZIP member')
                    expected_crc={name:f'{archive.getinfo(name).CRC:08x}' for name in ('events.h5','rectify_map.h5')}
                with open_h5(map_path) as stream:
                    if set(stream) != {'rectify_map'} or stream['rectify_map'].shape != (*NATIVE_SHAPE,2) or stream['rectify_map'].dtype != np.float32:
                        raise ValueError('Official rectification-map key/shape/dtype mismatch')
                views[view]=h5_event_metadata(event_path)
                receipts=[source_receipt(root,event_path),source_receipt(root,map_path)]
                if any(receipt['crc32']!=expected_crc[Path(receipt['path']).name] for receipt in receipts):
                    raise ValueError('Extracted HDF5/map CRC differs from original ZIP member')
                view_sources.extend(receipts)
                views[view].update(event_path=str(event_path.relative_to(root)),rectification_path=str(map_path.relative_to(root)))
            selected,eligible=uniform_eligible_targets(timestamps,names,views,count=10)
            originals=[source_receipt(root,path) for path in files.values()]
            for suffix,path in files.items():
                prior=original_receipts[(scene,suffix)]
                observed=next(row for row in originals if row['path']==str(path.relative_to(root)))
                url='https://download.ifi.uzh.ch/rpg/DSEC/train/'+scene+'/'+path.name
                if prior['url']!=url or prior['size_bytes']!=observed['bytes'] or prior['sha256']!=observed['sha256']:
                    raise ValueError('Existing official-download provenance disagrees with current source: '+suffix)
                observed['origin_url']=url
            result['source_files'].extend(originals+view_sources)
            gt_archive_sha=next(receipt['sha256'] for receipt in originals if receipt['path'].endswith('_disparity_event.zip'))
            result['sequences'].append(dict(scene_id=scene,original_split='train',timestamp_count=len(timestamps),gt_member_count=len(names),
                eligible_targets=eligible,views=views,calibration_members=calibration_members,
                calibration=dict(rectified_event_intrinsics=intrinsics,disparity_to_depth_cams_03=q.tolist()),
                frame_mapping='sorted_nativePNG_names_zipped_with_disparity_timestamp_rows;official_example_mapping'))
            for target in selected:
                time_us=target['target_us'];name=target.pop('filename')
                result['targets'].append(dict(**target,sample_id=scene+'/'+Path(name).stem,scene_id=scene,original_split='train',
                    bin_edges_us=list(range(time_us-WINDOW_US,time_us+1,BIN_US)),
                    native_gt=dict(archive=str(files['disparity_event.zip'].relative_to(root)),archive_sha256=gt_archive_sha,**gt_members[name]),
                    source_views={view:dict(event_path=views[view]['event_path'],rectification_path=views[view]['rectification_path']) for view in views}))
        except TimeoutError:
            raise
        except Exception as error:
            result['failures'].append(dict(scene_id=scene,error=repr(error)))
    result['status']='PREFLIGHT_READY_METADATA_ONLY' if not result['failures'] and len(result['targets'])==20 else 'BLOCKED_METADATA_NOT_READY'
    return result


def verify_preflight(manifest,config,root):
    if (manifest.get('status')!='PREFLIGHT_READY_METADATA_ONLY' or manifest.get('schema')!='dsec-causal-preflight-v1'
            or manifest.get('test_access') is not False or manifest.get('gt_pixels_opened') is not False
            or manifest.get('event_conversion') is not False or manifest.get('contract') != contract(config)
            or manifest.get('contract_sha256') != digest(contract(config)) or len(manifest.get('targets',[]))!=20
            or [row['scene_id'] for row in manifest['sequences']] != list(SCENES)):
        raise ValueError('Frozen complete metadata-only20-target contract required')
    rows=manifest['targets']
    if len({row['sample_id'] for row in rows})!=20 or any(sum(row['scene_id']==scene for row in rows)!=10 for scene in SCENES):
        raise ValueError('Require ten unique targets from each known official Train sequence')
    for row in rows:
        sid=row['scene_id'];target=row['target_us'];gt=row['native_gt']
        if (sid not in SCENES or row.get('original_split')!='train' or type(target) is not int
                or row['sample_id']!=sid+'/'+Path(gt['member']).stem
                or row['bin_edges_us']!=list(range(target-WINDOW_US,target+1,BIN_US))
                or gt['archive']!='download/'+sid+'/'+sid+'_disparity_event.zip'
                or set(row['source_views'])!={'left','right'}):
            raise ValueError('Target identity/source/causal edges changed')
        for view,source in row['source_views'].items():
            base='train/'+sid+'/events/'+view+'/'
            if source!=dict(event_path=base+'events.h5',rectification_path=base+'rectify_map.h5'):
                raise ValueError('View-specific verified source mapping changed')
    for receipt in manifest['source_files']:
        path=safe(root,receipt['path'])
        if path.stat().st_size!=receipt['bytes'] or file_sha(path)!=receipt['sha256']:
            raise ValueError('Source/calibration/rectification identity changed: '+receipt['path'])


def require_allocation(hostname=None):
    if not os.environ.get('SLURM_JOB_ID') or (hostname is not None and socket.gethostname()!=hostname):
        raise RuntimeError('Real conversion/GT decoding requires the registered cluster Slurm allocation')
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or not any(x in torch.cuda.get_device_name(0) for x in ('4090','5090')):
        raise RuntimeError('One authorized allocated RTX4090/5090 required; preparation itself usesCPU')


def prepare(config,manifest,root,output,*,binding):
    """Called only after Slurm/config/source guards. Opens no GT pixels."""
    validate_config(config,execution=True)
    require_allocation(config['expected_hostname'])
    if output.resolve().is_relative_to(root.resolve()):
        raise ValueError('Prepared output cannot enter readonly source data')
    output.mkdir(parents=True,exist_ok=False)
    result=dict(schema='dsec-causal-prepared-v1',protocol_status=config['protocol_status'],status='STARTED',
        contract=contract(config),source_manifest_sha256=config['preflight_manifest_sha256'],targets=[],
        source_files=manifest['source_files'],gt_pixels_opened=False,test_access=False,model_inference=False,
        slurm_job_id=os.environ['SLURM_JOB_ID'],binding=binding)
    try:
        verify_preflight(manifest,config,root)
        for row in manifest['targets']:
            sensors={};stats={}
            for view,sources in row['source_views'].items():
                with open_h5(safe(root,sources['rectification_path'])) as stream:
                    mapping=np.asarray(stream['rectify_map'])
                counts=np.zeros((T,2,*OUTPUT_SHAPE),dtype=np.int64)
                totals={key:0 for key in ('input_events','time_outside','bad_raw_coordinate','bad_polarity',
                    'nonfinite_rectification','outside_rectified_support','accepted_events','on_events','off_events')}
                with H5EventWindow(safe(root,sources['event_path'])) as events:
                    for chunk in events.iter_window(row['target_us']):
                        values,record=bin_rectified_counts(chunk,mapping,row['target_us'])
                        counts+=values
                        for key,value in record.items():totals[key]=totals.get(key,0)+value
                    boundaries=list(events.boundary_receipts)
                sensors[view]=checked_uint32(counts)
                totals['per_bin_ON_OFF_counts']=counts.sum(axis=(2,3),dtype=np.int64).tolist()
                totals['causal_boundary_neighbors']=boundaries
                stats[view]=totals
            asset=output/(row['sample_id'].replace('/','-')+'-events.npz')
            np.savez_compressed(asset,**sensors)
            prepared=dict(**row,sensor_asset=asset.name,sensor_sha256=file_sha(asset),count_statistics=stats)
            prepared['sensor_readback_qa']=verify_sensor_readback(output,prepared,sensors)
            result['targets'].append(prepared)
        source_faults={view:{key:sum(row['count_statistics'][view][key] for row in result['targets'])
            for key in ('time_outside','bad_raw_coordinate','bad_polarity')} for view in ('left','right')}
        result['conversion_qa']=dict(prepared_targets=20,readback_targets=20,
            all_causal_lower_bound_neighbors_verified=True,all_stored_counts_readback_exact=True,
            source_fault_counts=source_faults,
            ready_for_inference=not any(value for view in source_faults.values() for value in view.values()),
            rectification_discard_policy='nonfinite_or_outside_native_pixel_centers_counted_per_target_and_view;inspect_before_interpretation')
        result['status']='COMPLETED_PROSPECTIVE_ENGINEERING_PREPARATION'
    except BaseException as error:
        result.update(status='INCOMPLETE_PREPARATION',error=repr(error))
        raise
    finally:
        (output/'manifest.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


def load_prepared_sensor(root,row):
    """Future evaluator API: observations only, no GT. Unbatched50,2,120,160."""
    if row.get('scene_id') not in SCENES or row.get('original_split')!='train':
        raise ValueError('Prepared adapter access is restricted to the two known official Train sequences')
    path=safe(root,row['sensor_asset'])
    if file_sha(path)!=row['sensor_sha256']:
        raise ValueError('Prepared sensor asset changed')
    with np.load(path,allow_pickle=False) as archive:
        if set(archive.files)!={'left','right'}:
            raise ValueError('Exact left/right assets required')
        values={key:archive[key] for key in ('left','right')}
    for array in values.values():
        if array.shape!=(T,2,*OUTPUT_SHAPE) or array.dtype!=np.uint32 or int(array.max())>2**24:
            raise ValueError('Prepared integral counts must retain exact FP32 model-input values')
    return {key:array.astype(np.float32) for key,array in values.items()}


def verify_sensor_readback(root,row,expected):
    """Accept one newly written asset through the evaluator's actual input API."""
    observed=load_prepared_sensor(root,row)
    for view in ('left','right'):
        if not np.array_equal(observed[view],expected[view]):
            raise ValueError('Prepared sensor readback differs from accumulated integer counts')
        totals=observed[view].astype(np.uint64).sum(axis=(2,3),dtype=np.uint64)
        if totals.tolist()!=row['count_statistics'][view]['per_bin_ON_OFF_counts']:
            raise ValueError('Prepared readback changed per-bin ON/OFF counts')
        if int(totals.sum(dtype=np.uint64))!=row['count_statistics'][view]['accepted_events']:
            raise ValueError('Prepared readback violates accepted-event count conservation')
    return dict(status='PASS_EXACT_COUNTS_AND_PER_BIN_READBACK',views=['left','right'],
                float32_count_cast_exact=True,ground_truth_opened=False)


def load_native_gt(root,row):
    """Future cluster evaluator API: separately decode one native official GT."""
    require_allocation()
    from PIL import Image
    if row.get('scene_id') not in SCENES or row.get('original_split')!='train':
        raise ValueError('GT adapter access is restricted to the two known official Train sequences')
    gt=row['native_gt']
    if gt['archive']!='download/'+row['scene_id']+'/'+row['scene_id']+'_disparity_event.zip':
        raise ValueError('GT source must match its known Train sequence')
    archive_path=safe(root,gt['archive'])
    if file_sha(archive_path)!=gt['archive_sha256']:
        raise ValueError('Native GT archive identity changed')
    with zipfile.ZipFile(archive_path) as archive:
        member=archive.getinfo(gt['member'])
        if member.file_size!=gt['bytes'] or f'{member.CRC:08x}'!=gt['crc32']:
            raise ValueError('Native GT member identity changed')
        raw=archive.read(gt['member'])
    if raw[:8]!=b'\x89PNG\r\n\x1a\n' or raw[24:26]!=b'\x10\x00' or struct.unpack('>II',raw[16:24])!=(640,480):
        raise ValueError('Official native640x480 single-channel16bit PNG required')
    with Image.open(io.BytesIO(raw)) as image:
        values=np.asarray(image,dtype=np.uint16)
    return decode_uint16_gt(values)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--mode',choices=('preflight','prepare'),default='preflight')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--source-registry',type=Path)
    parser.add_argument('--print-example-config',action='store_true')
    args=parser.parse_args()
    if args.print_example_config:
        print(json.dumps(example_config(),indent=2));return
    if not args.config or not args.output:
        parser.error('--config and --output required')
    config=json.loads(args.config.read_text());validate_config(config,execution=args.mode=='prepare')
    root=Path(os.environ[config['source_root_env']]).resolve();output=args.output.resolve()
    if output.exists() or output.is_relative_to(root) or output.is_relative_to(Path(__file__).resolve().parents[3]):
        raise ValueError('Fresh output outside readonly source dataset/project required')
    binding=dict(config_sha256=file_sha(args.config),script_sha256=file_sha(__file__))
    if args.mode=='prepare':
        if not args.source_registry:
            raise RuntimeError('Preparation requires authorized registered cluster Slurm and frozen source')
        require_allocation(config['expected_hostname'])
        import torch
        registry=json.loads(args.source_registry.read_text())
        for name,sha in registry['source_files'].items():
            if file_sha(safe(args.source_registry.parent,name))!=sha:
                raise ValueError('Frozen source changed: '+name)
        if hashlib.sha256(json.dumps(registry['source_files'],sort_keys=True).encode()).hexdigest()!=registry['source_identity_sha256']:
            raise ValueError('Frozen source identity mismatch')
        for path in (Path(__file__),args.config):
            if str(path.resolve().relative_to(args.source_registry.parent.resolve())) not in registry['source_files']:
                raise ValueError('Adapter/config absent from frozen source')
        if output.is_relative_to(args.source_registry.parent.resolve()):
            raise ValueError('Prepared outputs cannot enter frozen source')
        path=Path(os.environ[config['preflight_manifest_env']])
        if file_sha(path)!=config['preflight_manifest_sha256']:
            raise ValueError('Preselected metadata manifest changed')
        manifest=json.loads(path.read_text())
        binding.update(source_registry_sha256=file_sha(args.source_registry),source_identity_sha256=registry['source_identity_sha256'],
                       git_commit=registry['git_commit'],gpu=torch.cuda.get_device_name(0),torch=torch.__version__,
                       source_dirty_status=registry.get('dirty_status'))
    started=time.monotonic()
    def expired(number,frame):raise TimeoutError('Registered bounded adapter deadline reached')
    previous=signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,config['max_seconds']-5)
    try:
        if args.mode=='preflight':
            try:
                result=preflight(config,root)
            except TimeoutError as error:
                result=dict(schema='dsec-causal-preflight-v1',status='BLOCKED_METADATA_TIME_CAP',error=repr(error),
                    gt_pixels_opened=False,event_conversion=False,test_access=False,targets=[])
            result.update(binding=binding,elapsed_seconds=time.monotonic()-started)
            output.parent.mkdir(parents=True,exist_ok=True)
            with output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
        else:
            result=prepare(config,manifest,root,output,binding=binding)
            result.update(binding=binding,elapsed_seconds=time.monotonic()-started)
            (output/'manifest.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
    print(json.dumps(dict(status=result['status'],output=str(output))))
    if result['status'] not in ('PREFLIGHT_READY_METADATA_ONLY','COMPLETED_PROSPECTIVE_ENGINEERING_PREPARATION'):
        raise SystemExit(1)


if __name__=='__main__':main()
