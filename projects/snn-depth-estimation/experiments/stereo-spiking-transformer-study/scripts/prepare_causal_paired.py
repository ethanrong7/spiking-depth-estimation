"""Prepare a NEW causal paired Train/Val engineering pilot on Slurm RTX only.

No launch occurs here. --check validates registration/metadata on CPU without
opening source pixels; --generate requires the explicitly registered cluster.
All dataset/model locations come from STUDY_* environment variables. The
historical centered derived spikes are never opened or reused.
"""
import argparse
from dataclasses import asdict
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import sys
import time

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[3]
PREP = PROJECT / 'experiments/flyingthings3d-native-v1/scripts'
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PREP))
from core.study_data import safe_path, validate_paired_manifest
from core.study_sensor_simulation import (PairedSensorState, SensorConfig, rgb_to_gray,
    registered_output_shape, resample_float_gray, resample_disparity_nearest)
from ema_adapter import EMAInterpolator, sha
from source_io import read_rgb, read_disparity_pfm


def write_new_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def check_inputs(args):
    config = json.loads(args.config.read_text())
    index = json.loads(args.source_index.read_text())
    protocol = json.loads(args.legacy_protocol.read_text())
    if config.get('stage') != 'data_preparation' or config.get('scope') != 'engineering_train_val_only':
        raise ValueError('Explicit engineering-only data preparation registration required')
    if sha(args.source_index) != config['source_index_sha256'] or sha(args.legacy_protocol) != config['legacy_protocol_sha256']:
        raise ValueError('Source index/protocol changed')
    sensor = SensorConfig(**config['sensor'])
    registered_output_shape((540,960), config.get('output_scale', 1.0))
    if protocol['interpolator']['code_commit'] != '75b6f6a889e695df875e103374040d47a4cfac7c':
        raise ValueError('Unrecognized pinned EMA interpolator')
    selected, identities = [], set()
    for request in config['sequences']:
        source_id = request['source_id']
        if source_id in identities or request['split'] not in ('train', 'val'):
            raise ValueError('Repeated source or forbidden split')
        identities.add(source_id)
        sequence = next(s for s in index['sequences'] if s['source_id'] == source_id)
        if not source_id.startswith('TRAIN/C/') or len(sequence['frames']) != 10:
            raise ValueError('Engineering pilot source must be original ten-frame TRAIN/C')
        centers = request['centers']
        if not centers or len(set(centers)) != len(centers):
            raise ValueError('Distinct actual target centers required')
        for center in centers:
            if center not in range(1,10) or not sequence['frames'][center]['disparity_l']:
                raise ValueError('No actual target disparity at registered center')
        selected.append((request, sequence))
    if {r['split'] for r,s in selected} != {'train','val'}:
        raise ValueError('Engineering pilot needs distinct Train and Val scenes; no fake Test')
    return config, index, protocol, sensor, selected


def env_path(name):
    value = os.environ.get(name)
    if not value:
        raise RuntimeError('Missing configured path: ' + name)
    return Path(value).resolve()


@torch.no_grad()
def generate(args):
    config, index, protocol, sensor, selected = check_inputs(args)
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname() != config['expected_hostname']:
        raise RuntimeError('Generation requires the registered Slurm cluster allocation')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Generation requires exactly one visible CUDA GPU')
    gpu = torch.cuda.get_device_name(0)
    if not any(card in gpu for card in ('RTX 4090', 'RTX 5090')):
        raise RuntimeError('Only authorized RTX 4090/5090 generation is supported')
    source_root = env_path('STUDY_SOURCE_DATA')
    output_parent = env_path('STUDY_PAIRED_OUTPUT')
    ema_source = env_path('STUDY_EMA_SOURCE')
    checkpoint = env_path('STUDY_EMA_CHECKPOINT')
    registry_path = env_path('STUDY_SOURCE_REGISTRY')
    registry = json.loads(registry_path.read_text())
    if not registry.get('git_commit') or not registry.get('source_identity_sha256'):
        raise RuntimeError('Immutable base Git plus source overlay identity required')
    for path in (Path(__file__), PROJECT/'core/study_sensor_simulation.py', PROJECT/'core/study_data.py',
                 PREP/'ema_adapter.py', PREP/'source_io.py', args.legacy_protocol, args.config):
        candidates = [digest for name,digest in registry['source_files'].items()
                      if str(path.resolve()).endswith(name)]
        if candidates != [sha(path)]:
            raise RuntimeError('Source/config differs from immutable registry: ' + path.name)
    name = config['dataset_version']
    if Path(name).name != name or name in ('.','..'):
        raise ValueError('Dataset version must be a single new directory name')
    output = output_parent / name
    if output.is_relative_to(source_root) or source_root.is_relative_to(output):
        raise RuntimeError('New derived output must be separate from source data')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    source_files = []
    output_scale = config.get('output_scale', 1.0)
    output_shape = registered_output_shape((540,960), output_scale)
    manifest = dict(schema='stereo-paired-engineering-pilot-v1', evidence_scope='engineering_only',
                    T=50, disparity='positive_left_pixels', rectified=True,
                    time_unit='source_frame_interval', physical_time_source=None,
                    source_identity=config['source_index_sha256'], license='SceneFlow research-only/noncommercial',
                    output_scale=output_scale, output_shape=list(output_shape),
                    spatial_operation='area_common_float_gray_before_sensor; nearest-exact_PFM_then_horizontal_disparity_scale',
                    source_registry_sha256=sha(registry_path), git_commit=registry['git_commit'],
                    source_overlay_identity=registry['source_identity_sha256'],
                    configuration_sha256=sha(args.config), sensor_model=sensor.metadata(),
                    simulators={
                        'spike':dict(implementation='core.study_sensor_simulation.PairedSensorState',
                                     version='v1-SpikeStereoNet-A2.3-style', configuration=asdict(sensor)),
                        'event':dict(implementation='core.study_sensor_simulation.PairedSensorState',
                                     version='v1-vid2e-reference-count-adaptation', configuration=asdict(sensor))},
                    train=[], val=[], test=[], test_status='not_created_requires_new_unexposed_scene_inventory',
                    source_support='preceding source interval endpoints only; no endpoint later than output target',
                    interpolation_limit='independent-view neural interpolation is not verified high-rate irradiance truth')
    write_new_json(output/'registration.json', dict(config=config, job_id=os.environ['SLURM_JOB_ID'],
                   gpu=gpu, source_registry_sha256=sha(registry_path), started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    try:
        torch.set_num_threads(2)
        torch.manual_seed(config['seed']); torch.cuda.manual_seed_all(config['seed'])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        model = EMAInterpolator(ema_source, checkpoint, source_files=protocol['interpolator']['source_files'])
        for request, sequence in selected:
            if time.monotonic() - started >= config['time_limit_seconds'] - 60:
                raise TimeoutError('Registered preparation deadline; preserve partial output')
            source_id, centers = request['source_id'], set(request['centers'])
            frames = sequence['frames']
            destination = output / source_id
            destination.mkdir(parents=True,exist_ok=False)
            observations = {center:{} for center in centers}
            trace = []
            for view in ('l','r'):
                rgb = []
                for frame in frames:
                    path = safe_path(source_root, frame['rgb_'+view])
                    source_files.append(dict(path=frame['rgb_'+view], bytes=path.stat().st_size, sha256=sha(path)))
                    rgb.append(read_rgb(path))
                initial = torch.from_numpy(rgb[0].copy()).to('cuda', dtype=torch.float32)/255
                state = PairedSensorState(resample_float_gray(rgb_to_gray(initial), output_scale), sensor)
                for interval in range(9):
                    spikes, events = [], []
                    iterator = iter(model.interiors(rgb[interval], rgb[interval+1]))
                    for j in range(1,51):
                        if time.monotonic()-started >= config['time_limit_seconds']-30:
                            raise TimeoutError('Registered preparation deadline; preserve partial output')
                        if j == 50:
                            image = torch.from_numpy(rgb[interval+1].copy()).to('cuda', dtype=torch.float32)/255
                            extra = {'overshoot_values':0}
                        else:
                            image, extra = next(iterator)
                        gray = resample_float_gray(rgb_to_gray(image), output_scale)
                        spike, event = state.step(gray)
                        trace.append(dict(view=view,global_index=50*interval+j,
                                          source_time=interval+j/50,
                                          spike_fraction=float(spike.float().mean()),
                                          on_events=int(event[0].sum()),off_events=int(event[1].sum()),
                                          charge_mean=float(state.voltage.mean()),
                                          charge_min=float(state.voltage.min()),charge_max=float(state.voltage.max()),
                                          **extra))
                        if interval+1 in centers:
                            spikes.append(spike.cpu().numpy())
                            if int(event.max()) > np.iinfo(np.uint16).max:
                                raise RuntimeError('Event count exceeds lossless uint16 storage')
                            events.append(event.cpu().numpy().astype(np.uint16))
                    try:
                        next(iterator)
                        raise RuntimeError('EMA produced extra temporal samples')
                    except StopIteration:
                        pass
                    if interval+1 in centers:
                        observations[interval+1][view] = (np.stack(spikes),np.stack(events))
                if state.steps != 450:
                    raise RuntimeError('Dropped sequence-time simulation samples')
            for center in sorted(centers):
                frame = frames[center]; stem=frame['source_frame']
                disparity_path = safe_path(source_root, frame['disparity_l'])
                disparity,_ = read_disparity_pfm(disparity_path)
                disparity = resample_disparity_nearest(torch.from_numpy(disparity), output_scale).numpy()
                source_files.append(dict(path=frame['disparity_l'],bytes=disparity_path.stat().st_size,sha256=sha(disparity_path)))
                for modality, index_in_pair in (('spike',0),('event',1)):
                    with (destination/(stem+'-'+modality+'.npz')).open('xb') as stream:
                        np.savez_compressed(stream,left=observations[center]['l'][index_in_pair],
                                            right=observations[center]['r'][index_in_pair])
                with (destination/(stem+'-disparity.npy')).open('xb') as stream:
                    np.save(stream,disparity,allow_pickle=False)
                row=dict(sample_id=source_id+'/'+stem,scene_id=source_id,split=request['split'],
                         target_time=float(center),bin_edges=np.linspace(center-1,center,51).tolist(),
                         source_support_end=float(center),event_channels=['ON','OFF'],
                         spike=source_id+'/'+stem+'-spike.npz',event=source_id+'/'+stem+'-event.npz',
                         disparity=source_id+'/'+stem+'-disparity.npy',source_frame=stem,
                         original_split=sequence['split'],historical_exposure=True,
                         sensor_state_start=0.,window_start=float(center-1),window_end=float(center),
                         future_context=False,native_shape=[540,960],observation_shape=list(output_shape),
                         output_scale=output_scale,disparity_unit='output_horizontal_pixels')
                manifest[request['split']].append(row)
            write_new_json(destination/'trace.json',trace)
        validate_paired_manifest(manifest,allow_engineering_pilot=True)
        write_new_json(output/'source-files.json',source_files)
        write_new_json(output/'manifest.json',manifest)
        write_new_json(output/'result.json',dict(status='COMPLETED',manifest_sha256=sha(output/'manifest.json'),
                       elapsed_seconds=time.monotonic()-started,train_samples=len(manifest['train']),
                       val_samples=len(manifest['val']),test_samples=0,gpu=gpu,
                       job_id=os.environ['SLURM_JOB_ID'],torch=torch.__version__,python=platform.python_version(),
                       checkpoint=model.metadata,all450_samples_processed_per_view=True,
                       peak_cuda_memory_allocated_bytes=torch.cuda.max_memory_allocated(),training_performed=False))
        for path in output.rglob('*'):
            path.chmod(0o555 if path.is_dir() else 0o444)
        output.chmod(0o555)
        print(json.dumps(dict(status='COMPLETED',output=str(output),manifest_sha256=sha(output/'manifest.json'))))
    except Exception as error:
        write_new_json(output/'failure.json',dict(status='FAILED',error=str(error),elapsed_seconds=time.monotonic()-started))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--source-index',type=Path,required=True)
    parser.add_argument('--legacy-protocol',type=Path,required=True)
    action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--check',action='store_true')
    action.add_argument('--generate',action='store_true')
    args=parser.parse_args()
    if args.check:
        config,index,protocol,sensor,selected=check_inputs(args)
        print(json.dumps(dict(status='CONFIG_CHECK_PASS',source_scenes=[r['source_id'] for r,s in selected],
                              sensor=sensor.metadata(),output_scale=config.get('output_scale',1.0),
                              output_shape=registered_output_shape((540,960),config.get('output_scale',1.0)),
                              source_pixels_opened=False,test_created=False)))
    else:
        generate(args)

if __name__ == '__main__':main()
