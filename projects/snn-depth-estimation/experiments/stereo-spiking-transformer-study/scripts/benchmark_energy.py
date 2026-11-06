"""Registered trained-checkpoint GPU inference energy; never submits jobs.

CLI: --config ENERGY.json --source-registry source-registry.json --output FRESH.
--validate-config and --print-example-config are CPU-only controls. Environment:
STUDY_TRAINED_PARENT, STUDY_SOURCE_PARENT, STUDY_PAIRED_PARENT and (legacy only)
STUDY_BASELINE_CAPSULE. Cases bind a best-val.pt SHA to its immutable training
source/config/manifest/seed/initial weights; no latest-checkpoint substitution.

Primary pipeline boundary is batch1 existing sensor NPZ decode/validation,
unchanged counts->FP32 CPU construction, H2D, full model API, finite check and
synchronization. No GT labels are read. Secondary gpu_resident boundary excludes
decode/H2D. Offline sensor simulation and sensor acquisition-window latency are
separate; CPU/sensor/system energy is unmeasured. Three interleaved rounds each
start with a30s model-free idle estimate. Primary energy retains total GPU joules;
idle subtraction is a separate estimate that may be negative. Raw traces and
read-only same-UUID settings/process snapshots are retained. No GPU settings
are changed, and clocks remain dynamic. OS filesystem caches are never flushed.
Registration freezes cache_priming='all_sensor_assets', warmup_all_samples=False:
pipeline primes/releases every CPU sensor asset outside measurement; resident
preload already primes them. Same-boundary API warmup needs >=3outputs AND>=2s,
while measured intervals retain whole selected-Val cycles after the30s minimum.
Study cases additionally pass an untimed first-Val checkpoint telemetry gate:
all prediction outputs with diagnostics True/False, atol1e-6/rtol1e-5, same RNG,
restored config/neuron flags, immutable inputs/weights. Optional registration
study_diagnostics_parity={atol:1e-6,rtol:1e-5} documents the fixed default; omission
does not disable the gate. Baselines have no override or extra parity forward.
"""
import argparse
import copy
import csv
import datetime
from dataclasses import replace
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import statistics
import subprocess
import sys
import time
import threading
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from core.study_energy import GPUPowerTrace, integrate_power
from core.study_training import atomic_json, capture_rng, digest, file_sha, restore_rng

PREFIX = 'projects/snn-depth-estimation/'
SHA_FIELDS = ('training_source_identity_sha256', 'training_source_registry_sha256',
              'training_config_sha256', 'checkpoint_sha256', 'manifest_sha256',
              'training_initial_weights_sha256', 'selection_validation_log_sha256')


class EnergyDeadline(TimeoutError):
    pass


def example_config():
    return {'schema': 'stereo-study-energy-v2', 'registered': False,
            'stage': 'engineering_diagnostic', 'driver': 'energy_benchmark',
            'scope': 'engineering', 'test_access': 'sealed', 'allocated_gpu_hour_cap': 24,
            'max_seconds': 2700, 'scheduler_resources': {'time': '00:45:00'},
            'measurement_seconds': 30, 'repeats': 3, 'warmup_outputs': 3,
            'warmup_seconds': 2, 'idle_seconds_per_round': 30, 'complete_val_cycles': True,
            'cache_priming': 'all_sensor_assets', 'warmup_all_samples': False,
            'study_diagnostics_parity': {'atol': 1e-6, 'rtol': 1e-5},
            'power_sampling_seconds': .05, 'context_sampling_seconds': 1,
            'max_power_sample_gap_seconds': 1.,
            'boundaries': ['pipeline', 'gpu_resident'], 'val_samples': 20,
            'resource_forecast': {'source_member': PREFIX+'experiments/stereo-spiking-transformer-study/analysis/REGISTERED-ENERGY-FORECAST.md',
                                  'sha256': '0'*64, 'reserved_gpu_seconds': 2700},
            'cases': [{'case_id': 'A-spike', 'training_run': 'REGISTERED_RUN',
                       'training_source_snapshot': 'REGISTERED_SOURCE',
                       'training_config_member': PREFIX+'experiments/stereo-spiking-transformer-study/configs/REGISTERED.json',
                       'seed': 9471, 'selection_validation_log_bytes': 1,
                       **{key: '0'*64 for key in SHA_FIELDS}}]}


def validate_energy_config(config, *, require_registered=True):
    if config.get('schema') != 'stereo-study-energy-v2':
        raise ValueError('Unknown energy benchmark schema')
    if require_registered and config.get('registered') is not True:
        raise ValueError('Energy config must be registered')
    if config.get('scope') != 'engineering' or config.get('test_access') != 'sealed':
        raise ValueError('Pilot energy is engineering only with sealed Test')
    if config.get('stage') != 'engineering_diagnostic' or config.get('driver') != 'energy_benchmark':
        raise ValueError('Explicit bounded energy driver registration required')
    if config.get('allocated_gpu_hour_cap') != 24:
        raise ValueError('Registered study allocation cap must remain24GPUhours')
    duration, maximum = config.get('measurement_seconds'), config.get('max_seconds')
    if not isinstance(duration, (float, int)) or not math.isfinite(duration) or duration < 30:
        raise ValueError('Energy intervals require at least30seconds')
    if type(maximum) is not int or not 300 <= maximum <= 2700:
        raise ValueError('Energy benchmark wall bound must be300..2700seconds')
    if config.get('repeats') != 3 or config.get('warmup_outputs') != 3:
        raise ValueError('Use three interleaved intervals and three complete warmup outputs')
    if config.get('boundaries') != ['pipeline', 'gpu_resident'] or config.get('val_samples') not in (1, 20):
        raise ValueError('Register primary pipeline and secondary gpu_resident boundaries, one or20 fixed Val inputs')
    if (config.get('warmup_seconds') != 2 or config.get('idle_seconds_per_round') != 30
            or config.get('context_sampling_seconds') != 1 or config.get('complete_val_cycles') is not True):
        raise ValueError('Use >=3outputs/2s warmup,30s round idle,1s context checks and whole Val cycles')
    if config.get('cache_priming') != 'all_sensor_assets' or config.get('warmup_all_samples') is not False:
        raise ValueError('Prime every sensor asset outside intervals; API warmup uses3outputs/2s without forcing all samples')
    if config.get('study_diagnostics_parity', {'atol': 1e-6, 'rtol': 1e-5}) != {'atol': 1e-6, 'rtol': 1e-5}:
        raise ValueError('Study telemetry parity uses fixed FP32 atol1e-6/rtol1e-5; it cannot be disabled or relaxed')
    if config.get('power_sampling_seconds') != .05:
        raise ValueError('Registered NVML cadence is50ms; samples are correlated')
    if config.get('max_power_sample_gap_seconds') != 1.:
        raise ValueError('Register at most1second unexplained power-sampling gaps')
    limit = config.get('scheduler_resources', {}).get('time', '')
    if not re.fullmatch(r'00:[0-4][0-9]:[0-5][0-9]', limit):
        raise ValueError('Energy scheduler bound must be at most45minutes')
    scheduler_seconds = sum(int(x)*scale for x, scale in zip(limit.split(':'), (3600, 60, 1)))
    if scheduler_seconds < maximum or scheduler_seconds > 2700:
        raise ValueError('Scheduler reservation must cover declared process bound')
    forecast = config.get('resource_forecast') or {}
    member = Path(forecast.get('source_member', ''))
    forecast_sha = forecast.get('sha256', '')
    if (member.is_absolute() or '..' in member.parts or not member.name
            or not isinstance(forecast_sha, str) or not re.fullmatch(r'[0-9a-f]{64}', forecast_sha)
            or (require_registered and forecast_sha == '0'*64)
            or forecast.get('reserved_gpu_seconds') != scheduler_seconds):
        raise ValueError('Freeze resource forecast SHA/member and full requested GPU reservation before execution')
    cases, identifiers = config.get('cases') or [], set()
    if not 1 <= len(cases) <= 10 or duration*6*len(cases)+3*30+30 > maximum:
        raise ValueError('Both boundaries x three rounds plus idle/cleanup must fit declared bound; at most10cases')
    for case in cases:
        for key in ('case_id', 'training_run', 'training_source_snapshot'):
            name = case.get(key, '')
            if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name):
                raise ValueError('Safe registered identifier required: '+key)
        if case['case_id'] in identifiers:
            raise ValueError('Duplicate energy case')
        identifiers.add(case['case_id'])
        member = Path(case.get('training_config_member', ''))
        if member.is_absolute() or '..' in member.parts or member.suffix != '.json':
            raise ValueError('Training config must be a relative frozen JSON member')
        if type(case.get('seed')) is not int or case['seed'] < 0:
            raise ValueError('Registered integer training seed required')
        if type(case.get('selection_validation_log_bytes')) is not int or case['selection_validation_log_bytes'] < 1:
            raise ValueError('Pin the complete final validation log byte count')
        for key in SHA_FIELDS:
            value = case.get(key, '')
            if not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{64}', value):
                raise ValueError('Pinned SHA256 required: '+key)
            if require_registered and value == '0'*64:
                raise ValueError('Placeholder identity cannot execute: '+key)
    return True


def interval_plan(config):
    """Interleave (case,boundary) pairs; reverse the complete middle round."""
    result = []
    pairs = [(case, boundary) for case in config['cases'] for boundary in config['boundaries']]
    for repeat in range(3):
        ordered = pairs if repeat != 1 else list(reversed(pairs))
        result.extend((repeat+1, boundary, case) for case,boundary in ordered)
    return result


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    index = (len(values)-1)*fraction
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo]+(values[hi]-values[lo])*(index-lo)


def counter_crosscheck(rows, start, end, integrated_joules):
    """Interpolate the optional cumulative NVML counter on the same boundaries."""
    relevant = [row for row in rows if start <= row['monotonic_seconds'] <= end]
    if any(row.get('joules_counter') is None for row in relevant):
        return {'available': False, 'reason': 'counter_unavailable_inside_interval'}
    points = [(row['monotonic_seconds'], row['joules_counter']) for row in rows
              if row.get('joules_counter') is not None]
    if (len(points) < 2 or points[0][0] > start or points[-1][0] < end
            or any(not math.isfinite(v) for point in points for v in point)
            or any(t1 <= t0 or c1 < c0 for (t0,c0),(t1,c1) in zip(points,points[1:]))):
        return {'available': False, 'reason': 'counter_unbracketed_nonfinite_or_reset'}
    def interpolate(boundary):
        for (t0,c0),(t1,c1) in zip(points,points[1:]):
            if t0 <= boundary <= t1:
                return c0+(c1-c0)*(boundary-t0)/(t1-t0)
        raise ValueError('Missing counter boundary')
    joules = interpolate(end)-interpolate(start)
    return {'available': True, 'same_window_counter_gpu_joules': joules,
            'absolute_difference_joules': abs(joules-integrated_joules),
            'relative_difference_from_trapezoid': abs(joules-integrated_joules)/integrated_joules if integrated_joules > 0 else None,
            'method': 'linearly interpolated cumulative NVML counter; crosscheck, not independent sensor'}


def verify_validation_selection(path, receipt, state, required_updates=None):
    raw = Path(path).read_bytes()[:receipt['bytes']]
    if len(raw) != receipt['bytes'] or hashlib.sha256(raw).hexdigest() != receipt['sha256']:
        raise RuntimeError('Best-Val checkpoint validation-log prefix changed')
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    metrics = [row for row in rows if type(row.get('update')) is int and row['update'] > 0
               and isinstance(row.get('mean_scene_epe'), (float, int))
               and math.isfinite(row['mean_scene_epe'])]
    selected = [row for row in metrics if row.get('update') == state.get('best_update')]
    if required_updates is not None and {row['update'] for row in metrics} != set(required_updates):
        raise RuntimeError('Final Val log does not match the complete registered evaluation schedule')
    if (not selected or state.get('update') != state.get('best_update')
            or selected[-1]['mean_scene_epe'] != state.get('best_val_epe')
            or min(row['mean_scene_epe'] for row in metrics) != state.get('best_val_epe')):
        raise RuntimeError('Checkpoint is not its recorded best scene-EPE Val selection')
    return {'selection': 'minimum trained-update mean_scene_epe over bound validation-log prefix; Val0 excluded',
            'update': state['update'], 'mean_scene_epe': state['best_val_epe'],
            'validation_prefix_sha256': receipt['sha256'], 'validation_prefix_bytes': receipt['bytes']}


def check_training_binding(case, training_config, metadata, checkpoint, registry, registry_sha):
    binding, state = checkpoint['binding'], checkpoint['state']
    expected = {'source_identity_sha256': case['training_source_identity_sha256'],
                'source_registry_sha256': case['training_source_registry_sha256'],
                'config_sha256': case['training_config_sha256'], 'manifest_sha256': case['manifest_sha256'],
                'git_commit': registry['git_commit'], 'config_identity': digest(training_config),
                'total_updates': training_config['total_updates'], 'effective_batch': 8}
    if metadata.get('binding') != binding or any(binding.get(key) != value for key,value in expected.items()):
        raise RuntimeError('Training/checkpoint immutable binding mismatch')
    if registry_sha != case['training_source_registry_sha256'] or registry['source_identity_sha256'] != case['training_source_identity_sha256']:
        raise RuntimeError('Training registry identity mismatch')
    if (training_config.get('seed') != case['seed'] or training_config.get('scope') != 'engineering'
            or training_config.get('test_access') != 'sealed' or training_config.get('pretraining') is not False
            or metadata.get('fresh_seed_model_sha256') != case['training_initial_weights_sha256']):
        raise RuntimeError('Training seed/initial weights/pilot scope binding mismatch')
    if type(state.get('update')) is not int or state['update'] < 1 or state.get('presentations') != state['update']*8:
        raise RuntimeError('Checkpoint is not a complete effective8 optimizer boundary')
    model = training_config['model']
    if model['family'] == 'study':
        if model['config'].get('architecture') not in ('A', 'B', 'conv') or model['config'].get('time_steps') != 50:
            raise RuntimeError('Study inference requires registered full50 sensor steps')
    elif model['family'] != 'baseline' or model['config'].get('architecture') not in ('original', 'original_I'):
        raise RuntimeError('Unknown pinned model family')
    return expected


def tensor_state_sha(model):
    answer = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        answer.update(name.encode())
        answer.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return answer.hexdigest()


def load_sensor_only(root, row, modality):
    """Exact PairedStereoDataset sensor checks/conversion, without GT access.

    That loader performs no count normalization. Keep binary spikes and ON/OFF
    integer counts unchanged, copying each into FP32 just as _tensor_sample.
    Geometry is checked between views without opening the disparity label.
    """
    import numpy as np
    import torch
    from core.study_data import safe_path
    if modality not in ('spike', 'event'):
        raise ValueError('Unknown sensor modality')
    with np.load(safe_path(root, row[modality]), allow_pickle=False) as packed:
        left, right = packed['left'], packed['right']
    expected_c = 1 if modality == 'spike' else 2
    for array in (left, right):
        if array.ndim != 4 or array.shape[:2] != (50, expected_c) or not np.isfinite(array).all():
            raise ValueError('Invalid temporal/channel layout or nonfinite observations')
        if np.any(array < 0) or np.any(array != np.floor(array)):
            raise ValueError('Inputs must be nonnegative integral counts')
        if modality == 'spike' and np.any(array > 1):
            raise ValueError('Intensity spikes must be binary')
    if left.shape != right.shape:
        raise ValueError('Stereo sensor geometry mismatch')
    return {'left': torch.from_numpy(left.astype(np.float32, copy=True)),
            'right': torch.from_numpy(right.astype(np.float32, copy=True)), 'sample_id': row['sample_id']}


def normalized_uuid(value):
    value = value.decode() if isinstance(value, bytes) else str(value)
    return value if value.startswith(('GPU-', 'MIG-')) else 'GPU-'+value


def gpu_snapshot(uuid, *, own_pid=None, runner=None):
    """Read-only documented nvidia-smi queries, filtered to allocated UUID.

    NVIDIA query reference: https://docs.nvidia.com/deploy/nvidia-smi/ . No
    clock/power setters are used. Observed dynamic clocks are never called locked.
    """
    runner = runner or subprocess.check_output
    uuid, own_pid = normalized_uuid(uuid), os.getpid() if own_pid is None else own_pid
    fields = ['uuid', 'name', 'driver_version', 'power.limit', 'clocks.gr', 'clocks.sm',
              'clocks.mem', 'temperature.gpu', 'pstate', 'memory.used', 'memory.total']
    began = time.monotonic()
    text = runner(['nvidia-smi', '--id='+uuid, '--query-gpu='+','.join(fields),
                   '--format=csv,noheader,nounits'], text=True, timeout=5)
    parsed = list(csv.reader(text.strip().splitlines()))
    if len(parsed) != 1 or len(parsed[0]) != len(fields):
        raise RuntimeError('Allocated GPU settings query did not return one complete row')
    settings = dict(zip(fields, [value.strip() for value in parsed[0]]))
    if settings['uuid'] != uuid:
        raise RuntimeError('GPU settings query escaped allocated UUID')
    applications = runner(['nvidia-smi', '--id='+uuid, '--query-compute-apps=pid,gpu_uuid,process_name',
                           '--format=csv,noheader,nounits'], text=True, timeout=5)
    processes = []
    for values in csv.reader(applications.strip().splitlines()):
        if not values or not any(value.strip() for value in values):
            continue
        if len(values) != 3:
            raise RuntimeError('Cannot verify allocated GPU compute process row')
        pid, observed_uuid, name = [value.strip() for value in values]
        if observed_uuid != uuid and observed_uuid.startswith(('GPU-', 'MIG-')):
            # Some driver versions may return extra device process rows even
            # with --id. Other authorized cards do not contaminate this UUID.
            continue
        if observed_uuid != uuid or not pid.isdigit():
            raise RuntimeError('Cannot attribute allocated GPU compute process')
        processes.append({'pid': int(pid), 'gpu_uuid': observed_uuid, 'process_name': name})
    return {'monotonic_seconds': began, 'query_finished_monotonic_seconds': time.monotonic(),
            'epoch_ns': time.time_ns(), 'gpu_uuid': uuid, 'settings': settings,
            'compute_processes': processes,
            'foreign_compute_processes': [row for row in processes if row['pid'] != own_pid]}


class GPUContextTrace:
    """One-second snapshots: query failure/foreign compute invalidates the interval."""
    def __init__(self, uuid, path, expected_power_limit, interval=1.):
        self.uuid, self.path = normalized_uuid(uuid), Path(path)
        self.expected_power_limit, self.interval = expected_power_limit, interval
        self.rows, self.errors, self.foreign = [], [], []
        self.event = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        try:
            with self.path.open('x') as stream:
                while not self.event.is_set():
                    row = gpu_snapshot(self.uuid)
                    self.rows.append(row)
                    self.foreign.extend(row['foreign_compute_processes'])
                    if row['settings']['power.limit'] != self.expected_power_limit:
                        self.errors.append('Observed power limit changed during same-card comparison')
                    stream.write(json.dumps(row)+'\n'); stream.flush()
                    self.event.wait(self.interval)
        except BaseException as error:
            self.errors.append(repr(error))

    def start(self):
        self.thread.start()
        wait_trace(self)
        self.check()

    def check(self):
        if self.foreign or self.errors:
            raise RuntimeError('GPU comparison contaminated/unverifiable: '+str(self.foreign or self.errors))

    def stop(self):
        self.event.set(); self.thread.join(timeout=11)
        if self.thread.is_alive():
            raise RuntimeError('GPU context monitor did not stop')
        # Final read catches a process/settings change after the last poll.
        row = gpu_snapshot(self.uuid)
        self.rows.append(row); self.foreign.extend(row['foreign_compute_processes'])
        with self.path.open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        if row['settings']['power.limit'] != self.expected_power_limit:
            self.errors.append('Observed power limit changed at interval end')
        self.check()


def context_summary(rows):
    result = {'snapshots': len(rows), 'power_limit_values': sorted({row['settings']['power.limit'] for row in rows}),
              'observed_pstates': sorted({row['settings']['pstate'] for row in rows}),
              'foreign_compute_observed': any(row['foreign_compute_processes'] for row in rows),
              'max_snapshot_gap_seconds': max((b['monotonic_seconds']-a['monotonic_seconds'] for a,b in zip(rows,rows[1:])),default=None),
              'max_query_duration_seconds': max((row['query_finished_monotonic_seconds']-row['monotonic_seconds'] for row in rows),default=None),
              'scope': 'read-only approximately1s snapshots; dynamic clocks, sampled device VRAM may miss short peaks; Torch allocator maxima separate'}
    for field in ('memory.used', 'temperature.gpu', 'clocks.gr', 'clocks.sm', 'clocks.mem'):
        values = []
        for row in rows:
            try:
                value = float(row['settings'][field])
                if math.isfinite(value): values.append(value)
            except (KeyError, ValueError, TypeError):
                pass
        result[field] = {'min': min(values), 'max': max(values)} if values else None
    if result['memory.used'] is not None:
        result['max_observed_device_memory_used_bytes'] = result['memory.used']['max']*1024**2
    return result


def host_snapshot():
    import torch
    return {'epoch_ns': time.time_ns(), 'hostname': os.uname().nodename,
            'pid': os.getpid(), 'cpu_count': os.cpu_count(),
            'cpu_affinity': sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
            'load_average': list(os.getloadavg()) if hasattr(os,'getloadavg') else None,
            'torch_threads': torch.get_num_threads(), 'torch_interop_threads': torch.get_num_interop_threads(),
            'slurm_cpus_per_task': os.environ.get('SLURM_CPUS_PER_TASK'),
            'omp_num_threads': os.environ.get('OMP_NUM_THREADS'), 'mkl_num_threads': os.environ.get('MKL_NUM_THREADS'),
            'limit': 'load/affinity snapshots expose possible contention; they do not prove an otherwise idle host'}


def power_trace_health(trace, start, end, max_gap=1.):
    rows = trace.rows
    points = [(row['monotonic_seconds'], row.get('watts')) for row in rows]
    if not points or points[0][0] > start or points[-1][0] < end:
        raise RuntimeError('Power trace does not bracket interval')
    pairs = [(a,b) for a,b in zip(points,points[1:]) if a[0] < end and b[0] > start]
    if not pairs:
        raise RuntimeError('Missing power trace coverage')
    for a,b in pairs:
        if (b[0] <= a[0] or b[0]-a[0] > max_gap
                or any(w is None or not math.isfinite(w) or w < 0 for t,w in (a,b))):
            raise RuntimeError('Unavailable/nonfinite power or unexplained >1s sampling gap')
    duration = [row.get('query_finished_monotonic_seconds',row['monotonic_seconds'])-row['monotonic_seconds']
                for row in rows if start-max_gap <= row['monotonic_seconds'] <= end+max_gap]
    return {'validated_max_gap_limit_seconds': max_gap,
            'max_interval_sample_gap_seconds': max(b[0]-a[0] for a,b in pairs),
            'max_query_duration_seconds': max(duration,default=None)}


def load_case(case, config, benchmark_registry, device, boundary='gpu_resident'):
    import numpy as np
    import torch
    from core.study_data import PairedStereoDataset, safe_path
    from train_study import verify_source
    source = safe_path(os.environ['STUDY_SOURCE_PARENT'], case['training_source_snapshot'])
    registry_path = source/'source-registry.json'
    if file_sha(registry_path) != case['training_source_registry_sha256']:
        raise RuntimeError('Training registry file hash changed')
    registry = verify_source(registry_path)
    member = case['training_config_member']
    config_path = safe_path(source, member)
    if registry['source_files'].get(member) != case['training_config_sha256'] or file_sha(config_path) != case['training_config_sha256']:
        raise RuntimeError('Frozen training config hash mismatch')
    training_config = json.loads(config_path.read_text())
    model_family = training_config['model']['family']
    model_member = PREFIX+'core/'+('study_models.py' if model_family == 'study' else 'study_baselines.py')
    training_core = {name: value for name,value in registry['source_files'].items() if name.startswith(PREFIX+'core/')}
    if (not training_core.get(model_member)
            or any(benchmark_registry['source_files'].get(name) != value for name,value in training_core.items())):
        raise RuntimeError('Benchmark frozen core bytes differ from trained source')
    run = safe_path(os.environ['STUDY_TRAINED_PARENT'], case['training_run'])
    checkpoint_path = run/'best-val.pt'
    if file_sha(checkpoint_path) != case['checkpoint_sha256']:
        raise RuntimeError('Registered best-Val checkpoint SHA mismatch')
    metadata = json.loads((run/'metadata.json').read_text())
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False, mmap=True)
    check_training_binding(case, training_config, metadata, checkpoint, registry, file_sha(registry_path))
    selection = verify_validation_selection(run/'validation.jsonl', checkpoint['logs']['validation'], checkpoint['state'])
    schedule = training_config.get('training', {}).get('evaluation_updates')
    if schedule is None:
        interval = training_config['training']['eval_interval']
        schedule = list(range(interval, training_config['total_updates']+1, interval))
        if not schedule or schedule[-1] != training_config['total_updates']:
            schedule.append(training_config['total_updates'])
    final_receipt = {'bytes': case['selection_validation_log_bytes'], 'sha256': case['selection_validation_log_sha256']}
    if checkpoint['logs']['validation']['bytes'] > final_receipt['bytes']:
        raise RuntimeError('Final selection log cannot precede its best checkpoint prefix')
    selection['final_selection_receipt'] = verify_validation_selection(
        run/'validation.jsonl', final_receipt, checkpoint['state'], required_updates=[x for x in schedule if x > 0])
    data = training_config['data']
    if data.get('manifest_sha256') != case['manifest_sha256']:
        raise RuntimeError('Training data config manifest differs')
    root = safe_path(os.environ['STUDY_PAIRED_PARENT'], data['dataset_version'])
    manifest = root/'manifest.json'
    if file_sha(manifest) != case['manifest_sha256']:
        raise RuntimeError('Val manifest changed')
    val = PairedStereoDataset(root, manifest, data['modality'], 'val', purpose='engineering', sample_ids=data.get('val_sample_ids'))
    if len(val) != checkpoint['binding']['validation_samples'] or digest([row['sample_id'] for row in val.rows]) != checkpoint['binding']['val_sample_ids_sha256']:
        raise RuntimeError('Val population/order differs from trained checkpoint')
    if config['val_samples'] == 20 and len(val) != 20:
        raise RuntimeError('Full20 mode requires exactly the registered20 Val examples')
    random.seed(case['seed']); np.random.seed(case['seed']); torch.manual_seed(case['seed'])
    torch.cuda.manual_seed_all(case['seed'])
    model_config = copy.deepcopy(training_config['model']['config'])
    variant = {'training_model_config': copy.deepcopy(model_config), 'overrides': {}, 'changes_weights': False}
    if model_family == 'study':
        from core.study_models import build_model
        model_config['collect_diagnostics'] = False
        variant['overrides'] = {'collect_diagnostics': False}
        model = build_model(model_config)
    else:
        from core.study_baselines import StudyBaseline
        model = StudyBaseline(os.environ['STUDY_BASELINE_CAPSULE'], seed=case['seed'], **model_config)
        provenance = model.provenance()
        historical = metadata.get('baseline_provenance') or {}
        for key in ('source_identity_sha256', 'archive_sha256', 'upstream_commit', 'architecture', 'input_channels'):
            if provenance.get(key) != historical.get(key):
                raise RuntimeError('Legacy source capsule provenance differs: '+key)
    if tensor_state_sha(model) != case['training_initial_weights_sha256']:
        raise RuntimeError('Rebuilt fresh-seed model differs from recorded training initializer')
    model.load_state_dict(checkpoint['model'], strict=True)
    del checkpoint
    model = model.to(device).eval()
    if any(p.is_floating_point() and p.dtype != torch.float32 for p in model.parameters()):
        raise RuntimeError('Benchmark requires original FP32 parameters')
    variant['inference_model_config'] = model_config
    state_sha = tensor_state_sha(model)
    allocated_before_inputs = torch.cuda.memory_allocated(device)
    samples, resident_bytes = [], 0
    for index in range(config['val_samples']):
        row = val.rows[index]
        if boundary == 'pipeline':
            samples.append({'sensor_root': str(root), 'row': row, 'modality': data['modality'], 'sample_id': row['sample_id']})
        else:
            sample = load_sensor_only(root, row, data['modality'])
            left, right = [sample[key].unsqueeze(0).to(device=device, dtype=torch.float32) for key in ('left','right')]
            resident_bytes += left.numel()*left.element_size()+right.numel()*right.element_size()
            samples.append({'left': left, 'right': right, 'sample_id': sample['sample_id']})
    torch.cuda.synchronize(device)
    return model, samples, {'checkpoint_sha256': case['checkpoint_sha256'], 'training_run': case['training_run'],
        'model_family': model_family,
        'training_seed': case['seed'], 'training_source_identity_sha256': registry['source_identity_sha256'],
        'training_config_sha256': case['training_config_sha256'], 'manifest_sha256': case['manifest_sha256'],
        'training_initial_weights_sha256': case['training_initial_weights_sha256'], 'trained_state_sha256': state_sha,
        'verified_training_core_files': len(training_core), 'training_core_file_map_sha256': digest(training_core),
        'selection': selection, 'variant': variant, 'sensor_semantics': '50 chronological simulation steps'
            if model_family == 'study' else '50 chronological bins as channels; ON/OFF preserved as100 time-major channels for event',
        'val_sample_ids': [row['sample_id'] for row in samples], 'input_tensor_bytes': resident_bytes,
        'allocated_before_input_preload_bytes': allocated_before_inputs,
        'allocated_after_input_preload_bytes': torch.cuda.memory_allocated(device),
        'boundary': boundary, 'ground_truth_io': 'none; sensor-only NPZ path',
        'observation_windows': [{'sample_id': row['sample_id'], 'first_bin_edge': (row.get('bin_edges') or [None])[0],
                                 'last_bin_edge': (row.get('bin_edges') or [None])[-1],
                                 'time_unit': getattr(val,'manifest',{}).get('time_unit'),
                                 'physical_fps': getattr(val,'manifest',{}).get('physical_fps')}
                                for row in val.rows[:config['val_samples']]],
        'training_torch': metadata.get('torch'), 'training_gpu': metadata.get('gpu')}


def value_fingerprint(value):
    """Exact values/structure hash for input and Python/NumPy/Torch RNG snapshots."""
    import numpy as np
    import torch
    answer = hashlib.sha256()
    def visit(item):
        if isinstance(item,torch.Tensor):
            array=item.detach().cpu().contiguous().numpy()
            answer.update(('tensor:'+str(item.dtype)+':'+repr(tuple(item.shape))).encode())
            answer.update(array.tobytes())
        elif isinstance(item,np.ndarray):
            answer.update(('numpy:'+str(item.dtype)+':'+repr(item.shape)).encode());answer.update(item.tobytes())
        elif isinstance(item,dict):
            answer.update(b'dict')
            for key in sorted(item,key=repr):visit(key);visit(item[key])
        elif isinstance(item,(tuple,list)):
            answer.update((type(item).__name__+':'+str(len(item))).encode())
            for child in item:visit(child)
        else:
            token=(type(item).__name__+':'+repr(item)).encode()
            answer.update(str(len(token)).encode()+b':'+token)
    visit(value)
    return answer.hexdigest()


def prediction_leaves(output):
    """Copy all output leaves to CPU; exclude only root diagnostics metadata."""
    import torch
    result={}
    def visit(value,path):
        if isinstance(value,torch.Tensor):result[path]=('tensor',value.detach().cpu().clone())
        elif isinstance(value,dict):
            keys=[key for key in value if not (path=='$' and key=='diagnostics')]
            result[path]=('dict',tuple(sorted((repr(key) for key in keys))))
            for key in keys:visit(value[key],path+'['+repr(key)+']')
        elif isinstance(value,(list,tuple)):
            result[path]=(type(value).__name__,len(value))
            for index,child in enumerate(value):visit(child,path+'['+str(index)+']')
        elif value is None or isinstance(value,(str,bool,int,float)):
            result[path]=(type(value).__name__,value)
        else:raise TypeError('Unsupported prediction output type at'+path)
    if not isinstance(output,dict) or 'disparity' not in output:raise RuntimeError('Missing disparity API output')
    visit(output,'$')
    return result


def compare_prediction_leaves(reference,candidate,atol,rtol):
    import torch
    if set(reference)!=set(candidate):raise RuntimeError('Diagnostics modes changed prediction output structure')
    result={}
    for path,(kind,expected) in reference.items():
        observed_kind,observed=candidate[path]
        if kind!=observed_kind:raise RuntimeError('Diagnostics modes changed output type at'+path)
        if kind!='tensor':
            if expected!=observed:raise RuntimeError('Diagnostics modes changed output metadata at'+path)
            continue
        if expected.shape!=observed.shape or expected.dtype!=observed.dtype:
            raise RuntimeError('Diagnostics modes changed prediction tensor shape/dtype at'+path)
        exact=torch.equal(expected,observed)
        if expected.is_floating_point() or expected.is_complex():
            close=bool(torch.isclose(observed,expected,atol=atol,rtol=rtol,equal_nan=False).all())
            finite=torch.isfinite(expected)&torch.isfinite(observed)
            difference=(observed[finite]-expected[finite]).abs().double()
            scale=atol+rtol*expected[finite].abs().double()
            max_abs=float(difference.max()) if difference.numel() else None
            max_scaled=float((difference/scale).max()) if difference.numel() else None
        else:
            close=exact;max_abs=float((observed.double()-expected.double()).abs().max()) if expected.numel() else 0.;max_scaled=None
        result[path]={'shape':list(expected.shape),'dtype':str(expected.dtype),'exact_equal':exact,
                      'within_tolerance':close,'max_absolute_error':max_abs,'max_scaled_tolerance_error':max_scaled,
                      'reference_nonfinite_values':int((~torch.isfinite(expected)).sum()),
                      'candidate_nonfinite_values':int((~torch.isfinite(observed)).sum())}
    return result


def run_study_diagnostics_parity(model,left,right,*,atol=1e-6,rtol=1e-5,audit=None):
    """One model, restored flags/RNG, two full forwards outside energy windows.

    Actual GPU checkpoint parity is established only when this runs in Slurm.
    This gate makes no gradient-parity claim. Fresh input clones protect the
    registered input cache; mutations of either clones or source inputs fail.
    """
    import torch
    from core.study_models import TemporalNeuron
    audit={} if audit is None else audit
    started=time.perf_counter()
    if model.training:raise RuntimeError('Diagnostics parity requires eval mode')
    original_config=model.config
    neurons=[module for module in model.modules() if isinstance(module,TemporalNeuron)]
    flags=[module.collect_diagnostics for module in neurons]
    if not neurons or original_config.collect_diagnostics is not False or any(flags):
        raise RuntimeError('Benchmark study model must begin with all diagnostics flags disabled')
    rng=capture_rng();rng_sha=value_fingerprint(rng)
    input_sha=value_fingerprint((left,right));state_sha=tensor_state_sha(model)
    audit.update(status='RUNNING',atol=atol,rtol=rtol,excluded_root_keys=['diagnostics'],
                 comparison='all recursive prediction tensor/scalar/container leaves; no gradient claim',
                 input_fingerprint_sha256=input_sha,trained_state_sha256=state_sha,
                 input_policy='identical fresh clones per mode; source input cache remains untouched',
                 initial_rng_sha256=rng_sha,forward_seconds={})
    if left.is_cuda:torch.cuda.reset_peak_memory_stats(left.device)
    outputs={};post_rng={}
    try:
        for enabled in (True,False):
            model.config=replace(original_config,collect_diagnostics=enabled)
            for module in neurons:module.collect_diagnostics=enabled
            restore_rng(rng)
            work_left,work_right=left.clone(),right.clone()
            tick=time.perf_counter()
            with torch.inference_mode():output=model(work_left,work_right)
            if left.is_cuda:torch.cuda.synchronize(left.device)
            audit['forward_seconds'][str(enabled).lower()]=time.perf_counter()-tick
            if (output['disparity'].shape!=(1,1,*left.shape[-2:])
                    or not bool(torch.isfinite(output['disparity']).all().cpu())):
                raise FloatingPointError('Parity gate did not produce a complete finite primary map')
            if value_fingerprint((work_left,work_right))!=input_sha:
                raise RuntimeError('Diagnostics parity forward mutated its input tensors')
            if value_fingerprint((left,right))!=input_sha:
                raise RuntimeError('Diagnostics parity changed registered source input tensors')
            if tensor_state_sha(model)!=state_sha:
                raise RuntimeError('Diagnostics parity forward changed trained weights/buffers')
            outputs[enabled]=prediction_leaves(output)
            post_rng[enabled]=value_fingerprint(capture_rng())
            del output,work_left,work_right
        audit['prediction_tensors']=compare_prediction_leaves(outputs[True],outputs[False],atol,rtol)
        audit['max_absolute_error']=max((row['max_absolute_error'] for row in audit['prediction_tensors'].values()
                                         if row['max_absolute_error'] is not None),default=None)
        audit['max_scaled_tolerance_error']=max((row['max_scaled_tolerance_error'] for row in audit['prediction_tensors'].values()
                                                 if row['max_scaled_tolerance_error'] is not None),default=None)
        audit['mismatching_tensor_paths']=[path for path,row in audit['prediction_tensors'].items() if not row['within_tolerance']]
        audit['post_forward_rng_equal_between_modes']=post_rng[True]==post_rng[False]
        audit['forward_rng_unchanged']=post_rng[True]==rng_sha
        if not audit['post_forward_rng_equal_between_modes']:raise RuntimeError('Diagnostics modes consumed different RNG state')
        if not all(row['within_tolerance'] for row in audit['prediction_tensors'].values()):
            raise RuntimeError('Full prediction parity exceeded registered FP32 tolerance')
        audit.update(status='PASSED_CHECKPOINT_DIAGNOSTICS_PARITY',
                     all_prediction_tensors_exact=all(row['exact_equal'] for row in audit['prediction_tensors'].values()),
                     compared_prediction_tensors=len(audit['prediction_tensors']))
    except BaseException as error:
        audit.update(status='FAILED_CHECKPOINT_DIAGNOSTICS_PARITY',error=repr(error))
        raise
    finally:
        model.config=original_config
        for module,flag in zip(neurons,flags):module.collect_diagnostics=flag
        restore_rng(rng)
        audit.update(config_object_restored=model.config is original_config,
                     neuron_flags_restored=all(module.collect_diagnostics==flag for module,flag in zip(neurons,flags)),
                     source_inputs_unchanged=value_fingerprint((left,right))==input_sha,
                     trained_state_unchanged=tensor_state_sha(model)==state_sha,
                     rng_restored=value_fingerprint(capture_rng())==rng_sha,
                     peak_allocated_bytes=torch.cuda.max_memory_allocated(left.device) if left.is_cuda else None,
                     peak_reserved_bytes=torch.cuda.max_memory_reserved(left.device) if left.is_cuda else None,
                     total_setup_seconds=time.perf_counter()-started)
        outputs.clear()
    if not all(audit[key] for key in ('config_object_restored','neuron_flags_restored','source_inputs_unchanged','trained_state_unchanged','rng_restored')):
        audit['status']='FAILED_CHECKPOINT_DIAGNOSTICS_PARITY'
        raise RuntimeError('Diagnostics parity restoration or immutable-state check failed')
    if left.is_cuda:
        # Release diagnostic-only cached blocks before the ordinary disabled-
        # telemetry warmup; leave the trained model/resident inputs allocated.
        torch.cuda.empty_cache()
    audit['allocator_cleanup']='unused diagnostic GPU cache blocks released before inference warmup'
    return audit


def ensure_diagnostics_parity(model,sample,boundary,device,*,case,provenance,cache,options=None,audit=None):
    """Reuse only a same-checkpoint/core/input/state receipt within this job."""
    import torch
    audit={} if audit is None else audit
    if provenance['model_family']!='study':
        audit.update(status='NOT_APPLICABLE_BASELINE',reason='no diagnostics override; no extra forward or sensor decode')
        return audit
    from core.study_models import TemporalNeuron
    if (model.training or model.config.collect_diagnostics is not False
            or any(module.collect_diagnostics for module in model.modules() if isinstance(module,TemporalNeuron))):
        raise RuntimeError('Parity receipt requires eval model with all inference diagnostics disabled')
    started=time.perf_counter()
    if boundary=='pipeline':
        cpu=load_sensor_only(sample['sensor_root'],sample['row'],sample['modality'])
        left,right=[cpu[key].unsqueeze(0).to(device=device,dtype=torch.float32) for key in ('left','right')]
    else:left,right=sample['left'],sample['right']
    options=options or {'atol':1e-6,'rtol':1e-5}
    binding={key:case[key] for key in ('case_id','checkpoint_sha256','manifest_sha256','seed','training_initial_weights_sha256')}
    binding.update(first_val_sample_id=sample['sample_id'],trained_state_sha256=provenance['trained_state_sha256'],
                   training_core_file_map_sha256=provenance['training_core_file_map_sha256'],
                   inference_config_fingerprint=value_fingerprint(vars(model.config)),
                   slurm_job_id=os.environ.get('SLURM_JOB_ID'),torch_version=torch.__version__,
                   gpu_uuid=normalized_uuid(torch.cuda.get_device_properties(left.device).uuid) if left.is_cuda else 'CPU_UNIT_TEST',
                   **options)
    key=digest(binding)
    state_sha=tensor_state_sha(model);input_sha=value_fingerprint((left,right))
    if state_sha!=binding['trained_state_sha256']:raise RuntimeError('Parity receipt cannot bind changed trained model state')
    if key in cache:
        receipt=cache[key]
        if (receipt['binding']!=binding or receipt['input_fingerprint_sha256']!=input_sha
                or receipt['trained_state_sha256']!=state_sha or receipt['status']!='PASSED_CHECKPOINT_DIAGNOSTICS_PARITY'):
            raise RuntimeError('Diagnostics parity receipt input/state/checkpoint binding mismatch')
        audit.update(status='REUSED_VERIFIED_JOB_RECEIPT',receipt_key=key,binding=binding,
                     input_fingerprint_sha256=input_sha,trained_state_sha256=state_sha,
                     reuse_validation_seconds=time.perf_counter()-started)
    else:
        audit.update(binding=binding,receipt_key=key,first_val_sample_id=sample['sample_id'])
        run_study_diagnostics_parity(model,left,right,**options,audit=audit)
        audit['total_setup_seconds_including_input']=time.perf_counter()-started
        cache[key]=copy.deepcopy(audit)
    return audit


def prime_sensor_assets(samples, boundary):
    """Warm repeated-file cache without forcing model inference on every input.

    Pipeline primes/reconstructs FP32 sensors one at a time and releases them.
    GPU-resident loading already decoded all sensors and placed them on GPU.
    Nothing here opens GT, performs inference or enters a measured interval.
    """
    if boundary == 'gpu_resident':
        return {'primed_sensor_assets': len(samples), 'method': 'already completed by GPU-resident input preload',
                'additional_priming_seconds': 0., 'cpu_priming_tensors_retained': False}
    if boundary != 'pipeline':
        raise ValueError('Unknown benchmark boundary')
    tick, tensor_bytes = time.perf_counter(), 0
    for sample in samples:
        cpu = load_sensor_only(sample['sensor_root'],sample['row'],sample['modality'])
        tensor_bytes += sum(cpu[key].numel()*cpu[key].element_size() for key in ('left','right'))
        del cpu
    return {'primed_sensor_assets': len(samples), 'method': 'sensor-only NPZ decode/validation+FP32 CPU construction, one-at-a-time released before API warmup',
            'additional_priming_seconds': time.perf_counter()-tick, 'total_cpu_tensor_bytes_processed': tensor_bytes,
            'cpu_priming_tensors_retained': False}


def warmup_model(model, samples, boundary, device, minimum_outputs=3, minimum_seconds=2.):
    """Same-boundary API warmup stops once both minima hold, not after all20."""
    start, outputs = time.monotonic(), 0
    while outputs < minimum_outputs or time.monotonic()-start < minimum_seconds:
        run_output(model,samples[outputs%len(samples)],boundary,device)
        outputs += 1
    return {'warmup_outputs': outputs, 'warmup_seconds': time.monotonic()-start,
            'warmup_all_samples': False}


def wait_trace(trace, boundary=None, timeout=3.):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        if trace.errors:
            raise RuntimeError('NVML trace failed: '+str(trace.errors))
        if trace.rows and (boundary is None or trace.rows[-1]['monotonic_seconds'] >= boundary):
            return
        time.sleep(.01)
    raise RuntimeError('Power trace failed to bracket measurement boundary')


def run_output(model, sample, boundary, device):
    """One complete batch1 prediction; wall starts before sensor-only decoding."""
    import torch
    tick = time.perf_counter()
    cpu_seconds, h2d_wall = 0., 0.
    copy_begin, copy_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    begin, finish = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    if boundary == 'pipeline':
        cpu = load_sensor_only(sample['sensor_root'], sample['row'], sample['modality'])
        cpu_seconds = time.perf_counter()-tick
        transfer_tick = time.perf_counter()
        copy_begin.record()
        left, right = [cpu[key].unsqueeze(0).to(device=device, dtype=torch.float32, non_blocking=False)
                       for key in ('left', 'right')]
        copy_end.record()
        h2d_wall = time.perf_counter()-transfer_tick
    else:
        left, right = sample['left'], sample['right']
    begin.record()
    output = model(left, right)
    prediction = output['disparity']
    if prediction.shape != (1,1,*left.shape[-2:]):
        raise RuntimeError('Inference API did not deliver a full-resolution primary map')
    finite = torch.isfinite(prediction).all()
    finish.record()
    torch.cuda.synchronize(device)
    if not bool(finite.cpu()):
        raise FloatingPointError('Nonfinite inference output; no positivity filtering or clamping is applied')
    wall = time.perf_counter()-tick
    row = {'sample_id': sample['sample_id'], 'wall_seconds': wall,
           'cpu_sensor_decode_tensor_seconds': cpu_seconds, 'h2d_wall_seconds': h2d_wall,
           'h2d_cuda_span_seconds': copy_begin.elapsed_time(copy_end)/1000 if boundary == 'pipeline' else 0.,
           'model_cuda_span_seconds': begin.elapsed_time(finish)/1000}
    return row


def measure_idle(trace, context, seconds, folder):
    """Model/tensor-free CUDA-context-retained round baseline; no fake outputs."""
    wait_trace(trace)
    start = time.monotonic()
    try:
        while time.monotonic()-start < seconds:
            context.check()
            time.sleep(.05)
    except BaseException:
        atomic_json(folder/'measurement-boundary.json', {'status': 'INCOMPLETE_IDLE',
                    'start_monotonic_seconds': start, 'stopped_monotonic_seconds': time.monotonic()})
        raise
    end = time.monotonic()
    wait_trace(trace, end)
    health = power_trace_health(trace,start,end)
    relevant = [row for row in trace.rows if start-trace.interval <= row['monotonic_seconds'] <= end+trace.interval]
    if any(row['watts'] is None for row in relevant):
        raise RuntimeError('Idle power sensor unavailable; no idle subtraction estimate')
    points = [(row['monotonic_seconds'], row['watts']) for row in trace.rows if row['watts'] is not None]
    joules = integrate_power(points, start, end)
    middle = (start+end)/2
    result = {'start_monotonic_seconds': start, 'end_monotonic_seconds': end, 'seconds': end-start,
              'integrated_gpu_joules': joules, 'mean_gpu_watts': joules/(end-start),
              'first_half_mean_gpu_watts': integrate_power(points,start,middle)/(middle-start),
              'second_half_mean_gpu_watts': integrate_power(points,middle,end)/(end-middle),
              'queried_power_min_watts': min(row['watts'] for row in relevant),
              'queried_power_max_watts': max(row['watts'] for row in relevant),
              'power_sample_count': len(relevant), 'counter_crosscheck': counter_crosscheck(trace.rows,start,end,joules),
              'power_trace_health': health,
              'method': 'trapezoid GPUjoules/seconds,30s before round, models+input tensors released; CUDA context retained; same monitoring cadence',
              'limitation': 'uncalibrated GPU idle estimate; dynamic clocks/temperature and monitor activity can differ from workload; correlated polls are not independent observations'}
    atomic_json(folder/'measurement-boundary.json', {'status': 'COMPLETE_IDLE_BRACKET', **result})
    return result


def measure_interval(model, samples, trace, config, device, latency_path, *, boundary='gpu_resident', context=None, idle=None):
    import torch
    latencies, delivered = [], 0
    torch.cuda.synchronize(device)
    wait_trace(trace)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()
    try:
        while (time.monotonic()-start < config['measurement_seconds']
               or (config.get('complete_val_cycles', True) and delivered % len(samples) != 0)):
            if context is not None:
                context.check()
            latency = run_output(model, samples[delivered % len(samples)], boundary, device)
            latencies.append({'output': delivered, **latency})
            delivered += 1
    except BaseException:
        latency_path.write_text(''.join(json.dumps(row)+'\n' for row in latencies))
        atomic_json(latency_path.with_name('measurement-boundary.json'),
                    {'status': 'INCOMPLETE_NO_ENERGY_CLAIM', 'boundary': boundary, 'start_monotonic_seconds': start,
                     'stopped_monotonic_seconds': time.monotonic(), 'complete_outputs': delivered,
                     'per_sample_output_counts': {s['sample_id']:sum(r['sample_id']==s['sample_id'] for r in latencies) for s in samples}})
        raise
    end = time.monotonic()
    wait_trace(trace, end)
    latency_path.write_text(''.join(json.dumps(row)+'\n' for row in latencies))
    atomic_json(latency_path.with_name('measurement-boundary.json'),
                {'status': 'INTERVAL_FINISHED_PENDING_TELEMETRY_VALIDATION', 'boundary': boundary,
                 'start_monotonic_seconds': start, 'end_monotonic_seconds': end, 'complete_outputs': delivered})
    health = power_trace_health(trace,start,end,config.get('max_power_sample_gap_seconds',1.))
    energy = trace.summary(start, end, delivered, idle_watts=idle['mean_gpu_watts'] if idle else None)
    energy['power_trace_health'] = health
    energy['counter_crosscheck'] = counter_crosscheck(trace.rows, start, end, energy['integrated_gpu_joules'])
    if idle is not None:
        energy['separately_idle_adjusted_gpu_joules'] = energy['integrated_gpu_joules']-idle['mean_gpu_watts']*(end-start)
        energy['idle_adjustment'] = {'calibration_round': idle['round'], 'watts': idle['mean_gpu_watts'],
            'method': 'total_gpu_joules minus same-round idle mean_watts times workload_seconds; supplementary uncalibrated estimate; never clamped',
            'primary_total_gpu_joules_unchanged': True}
    sample_counts = {sample['sample_id']: sum(row['sample_id'] == sample['sample_id'] for row in latencies) for sample in samples}
    energy.update(start_monotonic_seconds=start, end_monotonic_seconds=end, boundary=boundary,
                  complete_outputs_all_finite=True, batch_size=1,
                  unique_val_samples_delivered=sum(count > 0 for count in sample_counts.values()),
                  per_sample_output_counts=sample_counts, complete_val_cycles=delivered % len(samples) == 0,
                  throughput_maps_per_second=delivered/(end-start),
                  latency_wall_p50_seconds=percentile([x['wall_seconds'] for x in latencies], .5),
                  latency_wall_p95_seconds=percentile([x['wall_seconds'] for x in latencies], .95),
                  h2d_cuda_span_p50_seconds=percentile([x['h2d_cuda_span_seconds'] for x in latencies], .5),
                  h2d_cuda_span_p95_seconds=percentile([x['h2d_cuda_span_seconds'] for x in latencies], .95),
                  model_cuda_span_p50_seconds=percentile([x['model_cuda_span_seconds'] for x in latencies], .5),
                  model_cuda_span_p95_seconds=percentile([x['model_cuda_span_seconds'] for x in latencies], .95),
                  mean_cpu_sensor_decode_tensor_seconds=statistics.mean(x['cpu_sensor_decode_tensor_seconds'] for x in latencies),
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    atomic_json(latency_path.with_name('measurement-boundary.json'),
                {'status': 'COMPLETE_BRACKETED_INTERVAL', 'boundary': boundary, 'start_monotonic_seconds': start,
                 'end_monotonic_seconds': end, 'complete_outputs': delivered})
    return energy


def aggregate_intervals(rows):
    answer = []
    for case_id, boundary in dict.fromkeys((row['case_id'],row['boundary']) for row in rows):
        group = [row for row in rows if row['case_id'] == case_id and row['boundary'] == boundary]
        complete = [row for row in group if row['status'] == 'COMPLETED_ENERGY_INTERVAL']
        out = {'case_id': case_id, 'boundary': boundary, 'primary': boundary == 'pipeline', 'completed_intervals': len(complete),
               'status': 'COMPLETE_THREE_REPEATS' if len(complete) == 3 else 'INCOMPLETE_NO_THREE_REPEAT_CLAIM'}
        for key in ('gpu_joules_per_map', 'separately_idle_adjusted_gpu_joules_per_map',
                    'throughput_maps_per_second', 'latency_wall_p50_seconds', 'latency_wall_p95_seconds'):
            values = [row['measurement'][key] for row in complete if key in row['measurement']]
            out[key] = {'mean': statistics.mean(values), 'min': min(values), 'max': max(values),
                        'sample_stdev': statistics.stdev(values) if len(values)>1 else None} if values else None
        out['max_peak_allocated_bytes'] = max((row['measurement']['peak_allocated_bytes'] for row in complete), default=None)
        answer.append(out)
    return answer


def verify_resource_forecast(config, registry, root):
    from core.study_data import safe_path
    forecast = config['resource_forecast']
    path = safe_path(root, forecast['source_member'])
    if registry['source_files'].get(forecast['source_member']) != forecast['sha256'] or file_sha(path) != forecast['sha256']:
        raise RuntimeError('Registered pre-execution resource forecast changed or is absent from capsule')
    minimum = len(config['cases'])*len(config['boundaries'])*3*config['measurement_seconds']+3*config['idle_seconds_per_round']+30
    return {**forecast, 'verified_before_cuda': True, 'minimum_measurement_idle_cleanup_seconds': minimum,
            'setup_warmup_cycle_overrun_margin_seconds': config['max_seconds']-minimum,
            'account_24GPUhour_reservation_guard': 'performed by authorized launcher; this executable submits no jobs'}


def stop_monitors(power, context):
    errors = []
    for monitor in (context, power):
        if monitor is not None:
            try:
                monitor.stop()
            except BaseException as error:
                errors.append(repr(error))
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--source-registry', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--validate-config', action='store_true')
    parser.add_argument('--print-example-config', action='store_true')
    args = parser.parse_args()
    if args.print_example_config:
        print(json.dumps(example_config(), indent=2)); return
    if args.config is None:
        parser.error('--config is required')
    config = json.loads(args.config.read_text())
    validate_energy_config(config, require_registered=not args.validate_config)
    if args.validate_config:
        print(json.dumps({'status': 'VALID_CONFIG_ONLY', 'cases': len(config['cases']), 'boundaries': config['boundaries'], 'config_sha256': file_sha(args.config)})); return
    if not os.environ.get('SLURM_JOB_ID') or args.source_registry is None or args.output is None:
        raise RuntimeError('Energy execution requires authorized Slurm, immutable source registry and fresh output')
    began = time.monotonic()
    from train_study import verify_source
    registry = verify_source(args.source_registry)
    source_root = args.source_registry.parent.resolve()
    script_path = Path(__file__).resolve()
    if not script_path.is_relative_to(source_root):
        raise RuntimeError('Execute benchmark script from its verified source capsule')
    script_member = str(script_path.relative_to(source_root))
    if registry['source_files'].get(script_member) != file_sha(script_path):
        raise RuntimeError('Executing benchmark script is absent from verified source registry')
    if args.output.resolve().is_relative_to(source_root):
        raise ValueError('Benchmark artifacts must be outside immutable source')
    if registry['source_files'].get(str(args.config.resolve().relative_to(source_root))) != file_sha(args.config):
        raise RuntimeError('Energy config is absent from verified benchmark source capsule')
    resource = verify_resource_forecast(config, registry, source_root)
    args.output.mkdir(parents=True, exist_ok=False)
    result = {'schema': 'stereo-study-energy-result-v2', 'status': 'INITIALIZING',
        'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'slurm_job_id': os.environ['SLURM_JOB_ID'],
        'config_sha256': file_sha(args.config), 'benchmark_source_identity_sha256': registry['source_identity_sha256'],
        'benchmark_source_registry_sha256': file_sha(args.source_registry), 'git_commit': registry['git_commit'],
        'resource_forecast': resource, 'max_seconds': config['max_seconds'], 'measurement_seconds': config['measurement_seconds'],
        'scope': 'engineering_GPU_joules_during_declared_pipeline; CPU/sensor/total-system/neuromorphic_energy_unmeasured',
        'primary_boundary': 'pipeline', 'boundaries': {
            'pipeline': 'sensor-only NPZ read/decode+unchanged count validation+FP32 CPU tensor construction+pageable blocking H2D+full model API+finite-output reduction+synchronization',
            'gpu_resident': 'secondary: full model API+finite-output reduction+synchronization; excludes input decode/H2D'},
        'excluded_both': 'offline sensor simulation, physical sensor power, CPU/total-system energy, model loading, warmup; fixed50-bin acquisition window separately reported from data metadata',
        'cache_policy': 'no filesystem cache flush; prime every selected sensor asset outside each interval (GPU-resident preload already does this), release pipeline CPU priming tensors; same-boundary API warmup >=3outputs/2s without forcing20; pipeline reopens/decodes NPZ every measured output',
        'cache_priming': config['cache_priming'], 'warmup_all_samples': config['warmup_all_samples'],
        'transfer_policy': 'pipeline pageable FP32 CPU tensors, blocking to(cuda), no prefetch or pinned-memory optimization; decode excluded from CUDA spans',
        'latency_policy': 'wall includes complete boundary; H2D stream elapsed span includes transfer/launch gaps; model CUDA span includes full API and finite reduction',
        'delivered_map': 'one primary left map per complete API call; computed right/aux outputs included in cost; finite negative/zero maps counted',
        'interleaving': 'round1 (case,boundary) forward, round2 reverse complete pair order, round3 forward;30s model-free idle before each round; same allocated GPU UUID',
        'uncertainty': 'report spread across three interleaved intervals, not IID-poll confidence;50ms polling is not50ms independent power resolution; GPU power may be1s averaged',
        'settings_policy': 'read-only snapshots; no power/clock modifications; dynamic clocks and temperature observed, never claimed locked',
        'study_diagnostics_parity': config.get('study_diagnostics_parity',{'atol':1e-6,'rtol':1e-5}),
        'study_diagnostics_parity_policy': 'once per study case/checkpoint/firstVal/input/core/config in job; two untimed eval forwards with identical RNG, all predictions compared, flags/RNG restored; baseline not applicable; output parity only',
        'study_diagnostics_parity_receipts': {},
        'precision': 'float32', 'tf32': False, 'test_access': 'sealed', 'idle_calibrations': [], 'intervals': []}
    save = lambda: atomic_json(args.output/'result.json', result)
    save()
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one authorized CUDA GPU must be visible')
    device = torch.device('cuda:0')
    gpu = torch.cuda.get_device_properties(device)
    if not any(name in gpu.name for name in ('4090', '5090')):
        raise RuntimeError('Energy requires an authorizedRTX4090/5090')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.synchronize(device)
    initial_snapshot = gpu_snapshot(gpu.uuid)
    if initial_snapshot['foreign_compute_processes']:
        result.update(status='FAILED_FOREIGN_COMPUTE_PROCESS', initial_gpu_snapshot=initial_snapshot); save()
        raise RuntimeError('Foreign compute process already occupies allocated UUID')
    expected_power_limit = initial_snapshot['settings']['power.limit']
    result.update(status='RUNNING', gpu=gpu.name, gpu_uuid=normalized_uuid(gpu.uuid), gpu_total_bytes=gpu.total_memory,
                  initial_gpu_snapshot=initial_snapshot, torch=torch.__version__, cuda=torch.version.cuda,
                  initial_host_snapshot=host_snapshot(),
                  host_contention_policy='avoid concurrent host-intensive study jobs; sequential same-card API; affinity/thread/load snapshots do not prove host exclusivity',
                  representativeness='full20 ordered Val inputs' if config['val_samples']==20 else 'one-target diagnostic; not the primary paired-data operating point',
                  slurm_nodes=os.environ.get('SLURM_JOB_NODELIST'))
    save()
    deadline = began+config['max_seconds']-30
    def stop_signal(number, frame):
        raise EnergyDeadline('Bounded energy job reached signal/deadline; partial interval is not an energy result')
    old_handlers = {name: signal.signal(name, stop_signal) for name in (signal.SIGALRM, signal.SIGTERM, signal.SIGUSR1, signal.SIGINT)}
    signal.setitimer(signal.ITIMER_REAL, max(.1, deadline-time.monotonic()))
    failed, exhausted = set(), False
    try:
        for repeat in range(1,4):
            idle = {'round': repeat, 'status': 'INITIALIZING', 'host_before_round': host_snapshot()}
            result['idle_calibrations'].append(idle)
            folder = args.output/('idle-round'+str(repeat)); folder.mkdir(exist_ok=False)
            if exhausted or time.monotonic()+config['idle_seconds_per_round']+5 > deadline:
                exhausted = True; idle['status'] = 'SKIPPED_COLLECTIVE_TIME_BOUND'
            else:
                power = context = None
                try:
                    # Previous iteration releases all models/inputs. Keep only
                    # the initialized CUDA context for the idle calibration.
                    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(device)
                    idle['allocated_bytes_before_idle'] = torch.cuda.memory_allocated(device)
                    idle['reserved_bytes_before_idle'] = torch.cuda.memory_reserved(device)
                    power = GPUPowerTrace(gpu.uuid,folder/'gpu-power.jsonl',interval=.05)
                    context = GPUContextTrace(gpu.uuid,folder/'gpu-context.jsonl',expected_power_limit,interval=1.)
                    power.start(); context.start()
                    idle.update(measure_idle(power,context,config['idle_seconds_per_round'],folder))
                    idle['status'] = 'COMPLETED_IDLE_CALIBRATION'
                except BaseException as error:
                    (folder/'traceback.txt').write_text(traceback.format_exc())
                    idle.update(status='FAILED_IDLE_CALIBRATION',error=repr(error))
                    if isinstance(error,EnergyDeadline): exhausted=True
                finally:
                    errors = stop_monitors(power,context)
                    if errors:
                        idle.update(status='FAILED_IDLE_TELEMETRY',telemetry_errors=errors)
                    if context is not None:
                        idle['gpu_context_summary'] = context_summary(context.rows)
                    idle.update(raw_power_file=str(folder/'gpu-power.jsonl'),raw_context_file=str(folder/'gpu-context.jsonl'))
                    save()
            for planned_round,boundary,case in interval_plan(config):
                if planned_round != repeat: continue
                row = {'case_id': case['case_id'], 'boundary': boundary, 'primary': boundary == 'pipeline',
                       'repeat': repeat, 'status': 'INITIALIZING', 'gpu_uuid': normalized_uuid(gpu.uuid),
                       'host_before_setup': host_snapshot()}
                result['intervals'].append(row)
                if exhausted or time.monotonic()+config['measurement_seconds']+5 > deadline:
                    exhausted=True; row['status']='SKIPPED_COLLECTIVE_TIME_BOUND'; save(); continue
                if idle['status'] != 'COMPLETED_IDLE_CALIBRATION':
                    row['status']='SKIPPED_IDLE_CALIBRATION_FAILURE'; save(); continue
                if (case['case_id'],boundary) in failed:
                    row['status']='SKIPPED_PRIOR_CASE_FAILURE'; save(); continue
                folder = args.output/(case['case_id']+'-'+boundary+'-repeat'+str(repeat)); folder.mkdir(exist_ok=False)
                row.update(raw_power_file=str(folder/'gpu-power.jsonl'),raw_context_file=str(folder/'gpu-context.jsonl'),
                           raw_latency_file=str(folder/'latency.jsonl'),measurement_boundary_file=str(folder/'measurement-boundary.json'))
                power=context=model=samples=None
                started=time.monotonic()
                try:
                    model,samples,provenance=load_case(case,config,registry,device,boundary=boundary)
                    row.update(provenance=provenance,status='WARMING_UP'); save()
                    row['study_diagnostics_parity']={}
                    ensure_diagnostics_parity(model,samples[0],boundary,device,case=case,provenance=provenance,
                        cache=result['study_diagnostics_parity_receipts'],
                        options=config.get('study_diagnostics_parity'),audit=row['study_diagnostics_parity'])
                    row['cache_priming']=prime_sensor_assets(samples,boundary)
                    with torch.inference_mode():
                        row.update(warmup_model(model,samples,boundary,device,config['warmup_outputs'],config['warmup_seconds']))
                        row['setup_and_warmup_seconds']=time.monotonic()-started
                        power=GPUPowerTrace(gpu.uuid,folder/'gpu-power.jsonl',interval=.05)
                        context=GPUContextTrace(gpu.uuid,folder/'gpu-context.jsonl',expected_power_limit,interval=1.)
                        power.start(); context.start()
                        row['status']='MEASURING'; save()
                        row['measurement']=measure_interval(model,samples,power,config,device,folder/'latency.jsonl',boundary=boundary,context=context,idle=idle)
                    errors=stop_monitors(power,context)
                    row['gpu_context_summary']=context_summary(context.rows)
                    power=context=None
                    if errors: raise RuntimeError('GPU telemetry invalidated interval: '+str(errors))
                    if tensor_state_sha(model) != provenance['trained_state_sha256']:
                        raise RuntimeError('Inference changed model state; interval invalid')
                    row['status']='COMPLETED_ENERGY_INTERVAL'
                except BaseException as error:
                    (folder/'traceback.txt').write_text(traceback.format_exc())
                    row.update(status='INCOMPLETE_NO_ENERGY_CLAIM' if isinstance(error,EnergyDeadline) else 'FAILED_ENERGY_CASE',
                               error=repr(error),traceback_file=str(folder/'traceback.txt'))
                    row.pop('measurement',None); failed.add((case['case_id'],boundary))
                    if isinstance(error,EnergyDeadline): exhausted=True
                finally:
                    errors=stop_monitors(power,context)
                    if errors:
                        row.update(status='FAILED_ENERGY_TELEMETRY',telemetry_errors=errors)
                        row.pop('measurement',None); failed.add((case['case_id'],boundary))
                    row['total_interval_setup_measurement_seconds']=time.monotonic()-started
                    model=samples=None
                    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(device)
                    row['host_after_cleanup']=host_snapshot()
                    save()
            idle['host_after_round']=host_snapshot()
        idle_failed=any(row['status']!='COMPLETED_IDLE_CALIBRATION' for row in result['idle_calibrations'])
        result['status']='INCOMPLETE_TIME_BOUND' if exhausted else 'COMPLETED_WITH_CASE_FAILURES' if failed or idle_failed else 'COMPLETED_ENGINEERING_GPU_ENERGY'
    except EnergyDeadline as error:
        result.update(status='INCOMPLETE_TIME_BOUND',error=repr(error))
        for row in result['intervals']:
            if row['status'] in ('INITIALIZING','WARMING_UP','MEASURING'):
                row['status']='INCOMPLETE_NO_ENERGY_CLAIM'; row.pop('measurement',None)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        for name,handler in old_handlers.items(): signal.signal(name,handler)
        present={(row['repeat'],row['boundary'],row['case_id']) for row in result['intervals']}
        if result['status']=='INCOMPLETE_TIME_BOUND':
            for repeat,boundary,case in interval_plan(config):
                if (repeat,boundary,case['case_id']) not in present:
                    result['intervals'].append({'repeat':repeat,'boundary':boundary,'case_id':case['case_id'],
                                                'status':'SKIPPED_COLLECTIVE_TIME_BOUND'})
        result['elapsed_seconds']=time.monotonic()-began
        result['cases']=aggregate_intervals(result['intervals'])
        save()
    print(json.dumps({'status':result['status'],'cases':result['cases']},indent=2))


if __name__ == '__main__':
    main()
