"""Native Train-only fidelity preparation; generation requires Slurm RTX.

--check opens metadata only. The standalone QA artifact has no Val/Test rows
and is not a training manifest. Existing model/data core and EMA are reused.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import sys
import time

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[3]
PREP = PROJECT/'experiments/flyingthings3d-native-v1/scripts'
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PREP))
from core.study_sensor_simulation import PairedSensorState, SensorConfig, rgb_to_gray
from ema_adapter import EMAInterpolator, sha
from source_io import read_rgb, read_disparity_pfm
from audit_paired_pilot import check_sensor_array, validate_trace
from prepare_train_fidelity_sources import SCENES, CENTERS, source_file, require

SENSOR = dict(T=50, spike_threshold=5., event_threshold_on=.2, event_threshold_off=.2,
              log_epsilon=.001, event_tolerance=1e-6, noise=0., refractory=0.)
POINTS = ((.25, .25), (.25, .75), (.75, .25), (.75, .75))


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def validate_config(config, require_registered=True):
    require(config.get('schema') == 'stereo-study-train-fidelity-v1'
            and config.get('scope') == 'engineering_train_fidelity_only'
            and config.get('driver') == 'train_fidelity' and config.get('stage') == 'engineering_diagnostic'
            and config.get('allocated_gpu_hour_cap') == 24 and config.get('seed') == 9471,
            'Explicit Train-only fidelity contract required')
    require(not require_registered or config.get('registered') is True, 'Generation requires registered=true')
    require(config.get('test_access') == 'sealed' and config.get('sensor') == SENSOR
            and config.get('source_ids') == list(SCENES) and config.get('centers') == list(CENTERS),
            'Fixed Train panel, sensor parameters and all50 bins required')
    require(config.get('output_scale') == 1. and config.get('max_seconds') == 900
            and config.get('expected_hostname') == 'cccl3'
            and config.get('scheduler_resources', {}).get('time') == '00:15:00'
            and config.get('native_visual_crop') == [222, 318, 416, 544]
            and config.get('visual_steps') == [0, 25, 50]
            and config.get('scalar_positions') == [list(p) for p in POINTS], 'Native fixed fidelity operating point changed')
    for key in ('source_panel_sha256', 'source_readback_sha256', 'legacy_protocol_sha256'):
        require(re.fullmatch(r'[0-9a-f]{64}', config.get(key, '')) is not None, 'Exact metadata binding required: ' + key)
    return SensorConfig(**config['sensor'])


def check_inputs(args):
    config = json.loads(args.config.read_text())
    validate_config(config, require_registered=args.generate)
    require(sha(args.panel) == config['source_panel_sha256']
            and sha(args.source_readback) == config['source_readback_sha256']
            and sha(args.legacy_protocol) == config['legacy_protocol_sha256'], 'Source/protocol metadata changed')
    panel = json.loads(args.panel.read_text())
    receipt = json.loads(args.source_readback.read_text())
    protocol = json.loads(args.legacy_protocol.read_text())
    require(panel['schema'] == 'stereo-study-train-fidelity-source-panel-v1'
            and panel['test'] == [] and panel['test_values_opened'] is False
            and [s['source_id'] for s in panel['sequences']] == list(SCENES)
            and all(s['split'] == 'train' and s['centers'] == list(CENTERS) for s in panel['sequences']),
            'Fixed Train-only source panel required')
    require(receipt['status'] == 'PASS_FIXED92_SOURCE_BYTE_READBACK' and receipt['files'] == 92
            and receipt['panel_sha256'] == sha(args.panel)
            and protocol['interpolator']['code_commit'] == '75b6f6a889e695df875e103374040d47a4cfac7c',
            'Accepted source readback and pinned EMA required')
    return config, panel, protocol


def scalar_reference(initial_gray, gray, config):
    """Independent scalar FP32 charge/log-crossing reconstruction from input gray.

    NumPy log and GPU log can differ near a crossing. No tolerance adjustment
    or substitution of recorded internal state is used to make counts agree.
    """
    f = np.float32
    gray = np.asarray(gray, dtype=np.float32)
    initial_gray = np.asarray(initial_gray, dtype=np.float32)
    require(gray.ndim == 2 and gray.shape[1:] == initial_gray.shape, 'Scalar trace shape mismatch')
    spikes = np.zeros(gray.shape, dtype=np.uint8)
    events = np.zeros((len(gray), 2, gray.shape[1]), dtype=np.uint16)
    voltage = np.zeros_like(gray)
    logs = np.zeros_like(gray)
    bound = math.ceil(math.log((1+config.log_epsilon)/config.log_epsilon)
                      / min(config.event_threshold_on, config.event_threshold_off)) + 2
    for point in range(gray.shape[1]):
        charge = f(0)
        previous = reference = f(np.log(f(initial_gray[point]+f(config.log_epsilon))))
        for step in range(len(gray)):
            charge = f(charge+gray[step, point])
            spikes[step, point] = charge >= f(config.spike_threshold)
            if spikes[step, point]:
                charge = f(charge-f(config.spike_threshold))
            voltage[step, point] = charge
            current = f(np.log(f(gray[step, point]+f(config.log_epsilon))))
            logs[step, point] = current
            upward = current >= previous
            count = 0
            if abs(f(current-previous)) > f(config.event_tolerance):
                for _ in range(bound):
                    proposed = f(reference+f(config.event_threshold_on if upward else -config.event_threshold_off))
                    inside = previous < proposed <= current if upward else previous > proposed >= current
                    if not inside:
                        break
                    reference = proposed
                    count += 1
                else:
                    raise ValueError('Independent scalar crossing bound exceeded')
            events[step, 0 if upward else 1, point] = count
            previous = current
    return spikes, events, voltage, logs


def compare_scalar_trace(trace, config):
    expected = scalar_reference(trace['initial_gray'], trace['gray'], config)
    spike_bad = np.argwhere(expected[0] != trace['spike'])
    event_bad = np.argwhere(expected[1] != trace['event'])
    voltage_error = float(np.max(np.abs(expected[2]-trace['voltage'])))
    return dict(spike_mismatches=len(spike_bad), event_mismatches=len(event_bad),
                first_spike_mismatches=spike_bad[:20].tolist(), first_event_mismatches=event_bad[:20].tolist(),
                maximum_voltage_difference=voltage_error,
                maximum_log_backend_difference=float(np.max(np.abs(expected[3]-trace['log']))),
                exact_counts_and_charge_agree=not len(spike_bad) and not len(event_bad) and voltage_error == 0.,
                scope='independent_FP32_scalar_reconstruction; no_claim_of_universal_CPU_GPU_bitwise_identity')


@torch.no_grad()
def simulate_view(frames, interpolator, sensor, device, check_deadline=lambda: None, visual=None):
    """Full450-step sequence engine; tests use tiny synthetic images only."""
    require(len(frames) == 10 and all(f.shape == frames[0].shape and f.dtype == np.uint8 for f in frames),
            'Ten lossless same-shape RGB source frames required')
    height, width = frames[0].shape[:2]
    points = [(int(y*height), int(x*width)) for y, x in POINTS]
    yy, xx = zip(*points)
    def image(frame):
        return torch.from_numpy(frame.copy()).to(device=device, dtype=torch.float32)/255.
    def take(value):
        return value[list(yy), list(xx)].cpu().numpy().copy()
    initial = image(frames[0])
    initial_gray = rgb_to_gray(initial)
    state = PairedSensorState(initial_gray, sensor)
    scalars = dict(initial_gray=take(initial_gray), gray=[], spike=[], event=[], voltage=[], log=[])
    targets, summary = {}, []
    for interval in range(1, 10):
        check_deadline()
        if visual and interval in CENTERS:
            visual(interval, 0, image(frames[interval-1]), None, None)
        interiors = iter(interpolator.interiors(frames[interval-1], frames[interval]))
        spikes, events = [], []
        for step in range(1, 51):
            check_deadline()
            if step == 50:
                rgb = image(frames[interval])
                extra = dict(overshoot_values=0, min_before_clamp=float(rgb.min()), max_before_clamp=float(rgb.max()))
            else:
                rgb, extra = next(interiors)
            gray = rgb_to_gray(rgb)
            spike, event = state.step(gray)
            require(int(event.max()) <= np.iinfo(np.uint16).max, 'Event count exceeds lossless uint16 storage')
            scalars['gray'].append(take(gray))
            scalars['spike'].append(take(spike[0]))
            scalars['event'].append(np.stack([take(event[channel]) for channel in (0, 1)]))
            scalars['voltage'].append(take(state.voltage))
            scalars['log'].append(take(state.previous_log))
            summary.append(dict(global_index=state.steps, source_time=state.steps/50,
                spike_fraction=float(spike.float().mean()), on_events=int(event[0].sum()), off_events=int(event[1].sum()),
                charge_mean=float(state.voltage.mean()), charge_min=float(state.voltage.min()),
                charge_max=float(state.voltage.max()), **extra))
            if interval in CENTERS:
                spikes.append(spike.cpu().numpy().copy())
                events.append(event.cpu().numpy().astype(np.uint16))
                if visual and step in (25, 50):
                    visual(interval, step, rgb, spike, event)
        try:
            next(interiors)
            raise ValueError('Interpolator produced extra temporal samples')
        except StopIteration:
            pass
        if interval in CENTERS:
            targets[interval] = dict(spike=np.stack(spikes), event=np.stack(events))
    require(state.steps == 450, 'Dropped source-time steps')
    for key in ('gray', 'spike', 'event', 'voltage', 'log'):
        scalars[key] = np.stack(scalars[key])
    scalars['points_row_col'] = np.asarray(points, dtype=np.int64)
    return targets, summary, scalars


def save_visual(folder, interval, step, rgb, spike, event):
    require(tuple(rgb.shape) == (540, 960, 3), 'Native visual inputs required')
    crop = np.s_[222:318, 416:544]
    gray = rgb_to_gray(rgb)
    values = dict(rgb_thumbnail=rgb[::4, ::4].cpu().numpy(), gray_crop=gray[crop].cpu().numpy(),
        sensor_output_available=np.asarray(step != 0), interval=np.asarray(interval), step=np.asarray(step))
    if step:
        values.update(spike_crop=spike[0][crop].cpu().numpy(),
                      event_crop=event[:, 222:318, 416:544].cpu().numpy().astype(np.uint16))
    with (folder/f'visual-{interval}-{step}.npz').open('xb') as stream:
        np.savez_compressed(stream, **values)


def saved_target_check(folder, stem, shape, left_trace, right_trace):
    """Independent NPZ readback; this Train-only QA schema is not a training manifest."""
    result = {}
    for modality in ('spike', 'event'):
        with np.load(folder/(stem+'-'+modality+'.npz'), allow_pickle=False) as packed:
            result[modality] = {view: check_sensor_array(packed[key], modality, shape, trace)
                for key, view, trace in [('left', 'l', left_trace), ('right', 'r', right_trace)]}
    disparity = np.load(folder/(stem+'-disparity.npy'), allow_pickle=False)
    require(disparity.shape == tuple(shape) and disparity.dtype == np.float32, 'Saved native GT layout changed')
    valid = np.isfinite(disparity) & (disparity > 0)
    xx = np.arange(shape[1])[None, :]
    result['support'] = dict(valid_fraction=float(valid.mean()),
        valid_within_D256_fraction=float((valid & (disparity < 256)).mean()),
        valid_in_frame_fraction=float((valid & (xx-disparity >= 0) & (xx-disparity < shape[1])).mean()),
        masks_applied_to_saved_GT=False)
    return result


def generation_context(args, config, panel):
    require(os.environ.get('SLURM_JOB_ID') and socket.gethostname() == config['expected_hostname'],
            'Generation requires registered cluster Slurm allocation')
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1
            and any(name in torch.cuda.get_device_name(0) for name in ('RTX 4090', 'RTX 5090')),
            'Exactly one authorized RTX GPU required')
    require(args.source_registry and args.output, 'Executing source registry and fresh output required')
    registry = json.loads(args.source_registry.read_text())
    root = args.source_registry.parent.resolve()
    for member, digest in registry['source_files'].items():
        require(sha(root/member) == digest, 'Executing frozen member changed: ' + member)
    relative_config = str(args.config.resolve().relative_to(root))
    require(registry['source_files'].get(relative_config) == sha(args.config), 'Configuration absent from frozen source')
    output = args.output.resolve()
    for source in list(panel['source_roots'].values())+[str(root)]:
        source = Path(source).resolve()
        require(not output.is_relative_to(source) and not source.is_relative_to(output), 'QA output must be separate from sources')
    return registry


@torch.no_grad()
def generate(args):
    config, panel, protocol = check_inputs(args)
    registry = generation_context(args, config, panel)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    def deadline(*unused):
        if unused or time.monotonic()-started >= config['max_seconds']-30:
            raise TimeoutError('Registered fidelity preparation deadline; preserve partial output')
    old_handler = signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, config['max_seconds']-5)
    write_json(args.output/'registration.json', dict(config=config, source_registry_sha256=sha(args.source_registry),
        source_identity=registry['source_identity_sha256'], git_commit=registry['git_commit'],
        source_panel_sha256=sha(args.panel), job_id=os.environ['SLURM_JOB_ID']))
    try:
        torch.set_num_threads(2)
        torch.manual_seed(config['seed'])
        torch.cuda.manual_seed_all(config['seed'])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        interpolator = EMAInterpolator(Path(os.environ['STUDY_EMA_SOURCE']), Path(os.environ['STUDY_EMA_CHECKPOINT']),
                                       source_files=protocol['interpolator']['source_files'])
        sensor = SensorConfig(**config['sensor'])
        targets, comparisons = [], []
        for sequence in panel['sequences']:
            deadline()
            destination = args.output/sequence['source_id']
            destination.mkdir(parents=True, exist_ok=False)
            observations, summaries = {}, {}
            for view, view_name in [('l', 'left'), ('r', 'right')]:
                folder = destination/view
                folder.mkdir()
                records = [r for r in sequence['files'] if r['role'] == 'rgb' and r['view'] == view_name]
                require([r['frame_index'] for r in records] == list(range(10)), 'Source view order changed')
                frames = []
                for record in records:
                    path = source_file(panel, sequence, record)
                    require(path.stat().st_size == record['size'] and sha(path) == record['sha256'], 'Source changed before decode')
                    frames.append(read_rgb(path))
                observations[view], summaries[view], scalar = simulate_view(frames, interpolator, sensor, 'cuda', deadline,
                    lambda interval, step, rgb, spike, event: save_visual(folder, interval, step, rgb, spike, event))
                with (folder/'scalar-traces.npz').open('xb') as stream:
                    np.savez_compressed(stream, **scalar)
                with np.load(folder/'scalar-traces.npz', allow_pickle=False) as saved:
                    comparison = compare_scalar_trace(saved, sensor)
                comparisons.append(dict(source_id=sequence['source_id'], view=view,
                    scalar_trace=sequence['source_id']+'/'+view+'/scalar-traces.npz', **comparison))
            trace = [dict(view=view, **row) for view in ('l', 'r') for row in summaries[view]]
            validate_trace(trace, sensor.spike_threshold)
            write_json(destination/'trace.json', trace)
            for center in CENTERS:
                stem = sequence['frames'][center]['source_frame']
                record = next(r for r in sequence['files'] if r['role'] == 'gt' and r['frame_index'] == center)
                path = source_file(panel, sequence, record)
                require(path.stat().st_size == record['size'] and sha(path) == record['sha256'], 'Target source changed before decode')
                disparity, _ = read_disparity_pfm(path)
                with (destination/(stem+'-disparity.npy')).open('xb') as stream:
                    np.save(stream, disparity, allow_pickle=False)
                with (destination/f'disparity-visual-{center}.npz').open('xb') as stream:
                    np.savez_compressed(stream, native_pixel_thumbnail=disparity[::4, ::4], native_crop=disparity[222:318, 416:544])
                for modality in ('spike', 'event'):
                    with (destination/(stem+'-'+modality+'.npz')).open('xb') as stream:
                        np.savez_compressed(stream, left=observations['l'][center][modality], right=observations['r'][center][modality])
                support = saved_target_check(destination, stem, (540, 960),
                    summaries['l'][50*(center-1):50*center], summaries['r'][50*(center-1):50*center])
                targets.append(dict(sample_id=sequence['source_id']+'/'+stem, split='train', center=center,
                    source_root_alias=sequence['root_alias'], historical_exposure=sequence['historical_exposure'],
                    spike=sequence['source_id']+'/'+stem+'-spike.npz',
                    event=sequence['source_id']+'/'+stem+'-event.npz',
                    disparity=sequence['source_id']+'/'+stem+'-disparity.npy',
                    bin_edges=np.linspace(center-1, center, 51).tolist(), source_support_end=center,
                    sensor_state_start=0, output_shape=[540, 960], disparity_units='native_horizontal_pixels', readback=support))
            del observations
        result = dict(status='COMPLETED_TRAIN_ONLY_FIDELITY_GENERATION', elapsed_seconds=time.monotonic()-started,
            job_id=os.environ['SLURM_JOB_ID'], source_identity=registry['source_identity_sha256'], config_sha256=sha(args.config),
            source_registry_sha256=sha(args.source_registry), source_panel_sha256=sha(args.panel),
            source_readback_sha256=sha(args.source_readback), legacy_protocol_sha256=sha(args.legacy_protocol),
            schema='stereo-study-train-fidelity-artifact-v1', training_compatible=False, test_access=False,
            model_training=False, disparity_model_inference=False, interpolation_pretrained=True,
            native_saved_assets=True, quarter_saved_asset_fidelity_assessed=False, target_count=len(targets),
            time_unit='source_frame_interval', physical_time_calibrated=False,
            source_steps_per_view=450, targets=targets, scalar_comparisons=comparisons,
            scalar_counts_and_charge_pass=all(c['exact_counts_and_charge_agree'] for c in comparisons),
            visual_review_status='PENDING_FIXED_NATIVE_VIEWS; not high_rate_stereo_ground_truth',
            interpolator=interpolator.metadata, overshoot_handling='existing_EMA_adapter_clamp_disclosed_in_all_trace_rows',
            canonical_training_loader_scope='not exercised: standalone Train-only QA schema; prior2238 remains separate',
            sensor=sensor.metadata(), peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
        require(len(targets) == 12 and len(comparisons) == 8, 'Incomplete fixed panel')
        write_json(args.output/'result.json', result)
        for path in args.output.rglob('*'):
            path.chmod(0o555 if path.is_dir() else 0o444)
        args.output.chmod(0o555)
    except BaseException as error:
        write_json(args.output/'failure.json', dict(status='FAILED_PRESERVE_PARTIAL', error_type=type(error).__name__, error=str(error)))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
    print(json.dumps(dict(status=result['status'], scalar_pass=result['scalar_counts_and_charge_pass'], visual_review='pending')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'panel', 'source-readback', 'legacy-protocol'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--source-registry', type=Path)
    parser.add_argument('--output', type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--generate', action='store_true')
    args = parser.parse_args()
    if args.check:
        config, panel, protocol = check_inputs(args)
        print(json.dumps(dict(status='TRAIN_FIDELITY_METADATA_CHECK_PASS', registered=config['registered'],
            source_arrays_opened=False, GPU_used=False, sources=len(panel['sequences']), targets=12)))
    else:
        generate(args)


if __name__ == '__main__':
    main()
