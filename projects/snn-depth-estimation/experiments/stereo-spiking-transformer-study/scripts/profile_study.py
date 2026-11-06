"""Bounded one-GPU Train/Val engineering timings; submits no jobs or Test access."""
import argparse
import copy
import ctypes
import datetime
import gc
import gzip
import json
import os
from pathlib import Path
import random
import signal
import statistics
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from core.study_training import (atomic_json, evaluate_validation, file_sha,
                                 optimizer_and_scheduler, sample_indices, serial, train_update, validate_config)
from core.study_prefetch import OrderedBatchStream, prefetch_options
from core.study_energy import GPUPowerTrace


class ActivityTrace(GPUPowerTrace):
    """Read only the allocated UUID; utilization is correlated NVML telemetry."""
    class Utilization(ctypes.Structure):
        _fields_ = [('gpu', ctypes.c_uint), ('memory', ctypes.c_uint)]

    def read(self):
        row = super().read()
        value = self.Utilization()
        status = self.lib.nvmlDeviceGetUtilizationRates(self.handle, ctypes.byref(value))
        row.update(gpu_utilization_percent=value.gpu if status == 0 else None,
                   memory_utilization_percent=value.memory if status == 0 else None,
                   utilization_status=status)
        return row


class ProfileDeadline(TimeoutError):
    pass


class OneSample:
    def __init__(self, dataset, index=0):
        if index < 0 or index >= len(dataset):
            raise ValueError('Profile sample index outside registered cohort')
        self.dataset, self.index = dataset, index

    def __len__(self):
        return 1

    def __getitem__(self, index):
        if index != 0:
            raise IndexError(index)
        return self.dataset[self.index]


def profile_train_dataset(profile, dataset):
    """Select the registered population; full mode keeps the trainer's seeded order."""
    if profile.get('sample_mode', 'one_train_sample') == 'one_train_sample':
        return OneSample(dataset)
    expected = profile.get('expected_train_samples')
    if expected is not None and len(dataset) != expected:
        raise ValueError('Profile Train population differs from registered sample count')
    return dataset


def compact_readout_health(record):
    targets = {}
    for name, values in ((record.get('readout_optimizer_health') or {}).get('targets') or {}).items():
        if values.get('role') not in ('final_disparity_kernel', 'final_disparity_bias',
                                      'qkv_projection_kernel', 'qkv_normalization_scale'):
            continue
        row = {key: values[key] for key in ('role', 'output_channel', 'available', 'reason', 'step',
               'raw_gradient_energy', 'raw_energy_weighted_denominator_attenuation',
               'raw_energy_weighted_actual_moment_attenuation', 'raw_energy_on_changed_stored_coordinates',
               'raw_energy_on_changed_adaptive_coordinates', 'stored_coordinate_changed_fraction',
               'adaptive_coordinate_changed_fraction') if key in values}
        row['adaptive_delta_rms_excluding_ideal_decay'] = (values.get('adaptive_delta_excluding_ideal_decay') or {}).get('rms')
        row['adaptive_delta_rms_excluding_rounded_decay'] = (values.get('adaptive_delta_excluding_rounded_decay') or {}).get('rms')
        targets[name] = row
    return {'optimizer_moments_all_finite': record.get('optimizer_moments_all_finite'), 'targets': targets}


def training_config(profile, case):
    training = {**profile['training'], **case.get('training_override', {})}
    if case['model']['family'] == 'study' and profile.get('study_health_parameter_names'):
        training['health_parameter_names'] = profile['study_health_parameter_names']
    return {'schema': 'stereo-study-training-v1', 'registered': True, 'scope': 'engineering',
            'pretraining': False, 'test_access': 'sealed', 'seed': 9471,
            'total_updates': profile['recipe_total_updates'], 'model': case['model'],
            'data': {'modality': case['modality'], 'manifest_sha256': case['manifest_sha256']},
            'optimizer': profile['optimizer'], 'training': training, 'neuron_regularization': None}


def validate_profile(config, *, require_registered=True):
    comparison = config.get('schema') in ('stereo-study-prefetch-profile-v1', 'stereo-study-prefetch-parity-v1')
    parity = config.get('schema') == 'stereo-study-prefetch-parity-v1'
    if config.get('schema') not in ('stereo-study-profile-v1', 'stereo-study-prefetch-profile-v1', 'stereo-study-prefetch-parity-v1'):
        raise ValueError('Unknown runtime profile schema')
    if config.get('sample_mode', 'one_train_sample') not in ('one_train_sample', 'full_train_order'):
        raise ValueError('Unknown profile sample_mode')
    if config.get('sample_mode') == 'full_train_order':
        if config['training'].get('sampling', 'epoch_permutation') != 'epoch_permutation':
            raise ValueError('Full Train profile uses seeded epoch permutation')
        if type(config.get('expected_train_samples')) is not int or config['expected_train_samples'] < 24:
            raise ValueError('Full Train profile registers its population of at least24 samples')
    if require_registered and config.get('registered') is not True:
        raise ValueError('Profile example is unregistered')
    if config.get('seed') != 9471 or not 60 < config['max_seconds'] <= 3600:
        raise ValueError('Engineering profile requires seed9471 and at most3600seconds')
    expected = (16, 2) if comparison and not parity else (3, 1)
    if (config.get('updates'), config.get('warmup_updates')) != expected or config.get('recipe_total_updates') != 1000:
        raise ValueError('Use3updates, firstwarmup, immutable1000update recipe for timing only')
    if config['training']['effective_batch'] != 8 or config['training']['microbatch'] not in (1, 2, 4, 8):
        raise ValueError('Time effective8 with one common physicalbatch1/2/4/8 across the grid')
    if config['optimizer']['lr'] != .0002 or config['training']['clip_grad'] != 1.:
        raise ValueError('Registered engineering profile uses2e-4LR and clip1')
    head_health = config.get('purpose') == 'cross_family_head_health'
    if head_health:
        training = config['training']
        if (config.get('sample_mode') != 'full_train_order' or config.get('expected_train_samples') != 320
                or training['microbatch'] != 8 or training.get('readout_optimizer_health') is not True
                or prefetch_options(training) != {'workers': 2, 'batches': 2, 'pin_memory': True, 'device_prefetch': True}
                or training.get('health_parameter_names') or config['updates'] != 3):
            raise ValueError('Cross-family head health requires320-Train order, physical8, prefetch2, default final heads, threeupdates')
    identifiers = set()
    for case in config['cases']:
        if case['case_id'] in identifiers or case['case_id'] in ('.', '..') or Path(case['case_id']).name != case['case_id']:
            raise ValueError('Case identifiers must be unique safe directory names')
        identifiers.add(case['case_id'])
        version = Path(case['dataset_version'])
        if version.is_absolute() or '..' in version.parts:
            raise ValueError('Dataset versions must be beneath STUDY_PAIRED_PARENT')
        shape = case['shape']
        if shape not in ([270, 480], [135, 240]):
            raise ValueError('Profile only declared half/quarter regenerated sensor data')
        model = case['model']
        if head_health and (shape != [135, 240] or case.get('training_override')):
            raise ValueError('Cross-family head health uses one common quarter-resolution recipe')
        if model['family'] == 'study':
            values = model['config']
            if values['architecture'] not in ('A', 'B') or values['widths'] != [32, 64, 96, 128]:
                raise ValueError('Profile full unchanged A/B widths')
            if values.get('blocks_per_scale', 2) != 2 or values.get('attention_normalization', 'count_floor_v2') != 'count_floor_v2':
                raise ValueError('Profile complete two-block scales with countfloorv2')
            if values['max_disparity'] != (128 if shape[0] == 270 else 64):
                raise ValueError('Resolution-specific disparity support must be registered')
            if values.get('collect_diagnostics', True) is not True:
                raise ValueError('Training/Val forecast includes production GPU telemetry')
            if head_health and values.get('feature_norm_eps') != .01:
                raise ValueError('Head health keeps feature GN epsilon0.01 fixed')
        elif model['family'] != 'baseline' or model['config']['architecture'] != 'original_I':
            raise ValueError('Priority recurrent profile is pinned strongALIF+I')
        candidate = training_config(config, case)
        if comparison and case.get('training_override'):
            raise ValueError('Prefetch comparison modes are fixed by the profiler')
        if not require_registered:
            candidate['registered'] = False
        validate_config(candidate, require_registered=require_registered)
    if not config['cases']:
        raise ValueError('No profile cases')
    if comparison:
        if len(config['cases']) != 2 or {case['model']['family'] for case in config['cases']} != {'study', 'baseline'}:
            raise ValueError('Prefetch comparison pairs one complete study model and pinned baseline')
        if any(case['modality'] != 'event' for case in config['cases']):
            raise ValueError('Prefetch comparison uses larger ON/OFF inputs')
        return True
    for shape in ([270, 480], [135, 240]):
        grid = []
        for case in config['cases']:
            if case['shape'] == shape:
                model = case['model']
                family = (model['config']['architecture'], model['config'].get('spiking', True))
                grid.append((family, case['modality']))
        if not grid:
            continue
        expected = {(family, modality) for family in [('A', True), ('B', True), ('A', False), ('B', False), ('original_I', True)] for modality in ('spike', 'event')}
        if len(grid) != 10 or set(grid) != expected:
            raise ValueError('Each resolution requires the matched fivefamilies x twomodalities grid')
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--validate-config', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    validate_profile(config, require_registered=not args.validate_config)
    if args.validate_config:
        print(json.dumps({'status': 'VALID_CONFIG_ONLY', 'cases': len(config['cases']), 'config_sha256': file_sha(args.config)}))
        return
    if not os.environ.get('SLURM_JOB_ID') or args.output is None:
        raise RuntimeError('Profile requires authorized Slurm and fresh raw output location')
    began = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    result = {'status': 'INITIALIZING', 'scope': 'bounded_engineering_runtime_not_recipe_or_accuracy_verdict',
              'slurm_job_id': os.environ['SLURM_JOB_ID'], 'config_sha256': file_sha(args.config),
              'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'max_seconds': config['max_seconds'],
              'effective_batch': 8, 'physical_batch': config['training']['microbatch'],
              'test_access': 'sealed', 'cases': [], 'timing_boundary': 'CPUdecode+H2D+fullT50forward/backward+AdamW+FP64clip+telemetryCPUserialization+gzipflush/fsync',
              'sample_mode': config.get('sample_mode', 'one_train_sample'),
              'purpose': config.get('purpose', 'runtime_profile'),
              'forecast_limitation': 'first24 seeded Train presentations and one fixed Val frame; threeupdate health/timing diagnostic, not training accuracy/convergence or full-cohort IO evidence'
                  if config.get('sample_mode') == 'full_train_order' else
                  'singleTrain/Valsample; cache-warm engineering timings, not full-dataset IO/convergence guarantee'}
    save = lambda: atomic_json(args.output / 'result.json', result)
    save()
    import numpy as np
    import torch
    from core.study_data import PairedStereoDataset, safe_path
    from train_study import verify_source
    registry_path = Path(os.environ['STUDY_SOURCE_REGISTRY'])
    registry = verify_source(registry_path)
    if args.output.resolve().is_relative_to(registry_path.parent.resolve()):
        raise ValueError('Profile outputs must be outside immutable source')
    if config['schema'] == 'stereo-study-prefetch-parity-v1':
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    device = torch.device('cuda:0')
    gpu = torch.cuda.get_device_properties(device)
    if torch.cuda.device_count() != 1 or not any(x in gpu.name for x in ('4090', '5090')):
        raise RuntimeError('Exactly one authorizedRTX4090/5090 must be visible')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    result.update(status='RUNNING', gpu=gpu.name, gpu_total_bytes=gpu.total_memory, precision='float32',
                  source_identity_sha256=registry['source_identity_sha256'], git_commit=registry['git_commit'],
                  source_registry_sha256=file_sha(registry_path), torch=torch.__version__, cuda=torch.version.cuda,
                  deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                  cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                  slurm_cluster=os.environ.get('SLURM_CLUSTER_NAME'), slurm_nodes=os.environ.get('SLURM_JOB_NODELIST'))
    save()
    deadline = began + config['max_seconds'] - 60
    def deadline_handler(number, frame):
        raise ProfileDeadline('Collective profile softdeadline reached;60seconds reserved for finalization')
    previous_handler = signal.signal(signal.SIGALRM, deadline_handler)
    signal.setitimer(signal.ITIMER_REAL, max(.1, deadline - time.monotonic()))
    budget_exhausted = False
    cases = config['cases']
    if config['schema'] in ('stereo-study-prefetch-profile-v1', 'stereo-study-prefetch-parity-v1'):
        cases = []
        for original in config['cases']:
            for repeat, workers in enumerate((0, 2, 2, 0)):
                case = copy.deepcopy(original)
                case['comparison_case'] = original['case_id']
                case['case_id'] += '-repeat' + str(repeat+1) + '-workers' + str(workers)
                case['training_override'] = {'prefetch': {'workers': workers, 'batches': 2,
                                                        'pin_memory': True, 'device_prefetch': True}}
                cases.append(case)
        result['comparison_order'] = 'off,on,on,off independently initialized same seed/order on the same allocated GPU'
    try:
        for case in cases:
            if budget_exhausted or time.monotonic() >= deadline:
                result['cases'].append({'case_id': case['case_id'], 'status': 'SKIPPED_COLLECTIVE_TIME_BUDGET'})
                save()
                continue
            row = {'case_id': case['case_id'], 'status': 'RUNNING', 'dataset_version': case['dataset_version'],
                   'shape': case['shape'], 'modality': case['modality'], 'model': case['model'], 'steps': []}
            result['cases'].append(row)
            save()
            folder = args.output / case['case_id']
            folder.mkdir(exist_ok=False)
            model = optimizer = scheduler = train = val = batch_stream = activity = None
            started = time.monotonic()
            try:
                dataset_root = safe_path(os.environ['STUDY_PAIRED_PARENT'], case['dataset_version'])
                manifest_path = dataset_root / 'manifest.json'
                if file_sha(manifest_path) != case['manifest_sha256']:
                    raise RuntimeError('Profile paired manifest hash mismatch')
                train = profile_train_dataset(config, PairedStereoDataset(dataset_root, manifest_path, case['modality'], 'train', purpose='engineering'))
                val = OneSample(PairedStereoDataset(dataset_root, manifest_path, case['modality'], 'val', purpose='engineering'))
                sample = train[0]
                if list(sample['left'].shape[-2:]) != case['shape']:
                    raise ValueError('Asset shape differs from declared regenerated resolution')
                row.update(shape_probe_train_sample_id=sample['sample_id'], shape_probe_train_scene_id=sample['scene_id'],
                           train_population_samples=len(train), sample_mode=config.get('sample_mode', 'one_train_sample'),
                           sensor_steps=int(sample['left'].shape[0]), train_input_shape=list(sample['left'].shape),
                           evidence_scope=sample['metadata'].get('evidence_scope', 'engineering_only'))
                if isinstance(train, OneSample):
                    row.update(train_sample_id=sample['sample_id'], train_scene_id=sample['scene_id'])
                else:
                    first_indices = sample_indices(9471, 0, config['updates']*8, len(train))
                    row['diagnostic_train_indices'] = first_indices
                    row['diagnostic_train_sample_ids'] = [train.rows[index]['sample_id'] for index in first_indices]
                del sample
                random.seed(9471)
                np.random.seed(9471)
                torch.manual_seed(9471)
                torch.cuda.manual_seed_all(9471)
                values = copy.deepcopy(case['model']['config'])
                if case['model']['family'] == 'study':
                    from core.study_models import build_model
                    model = build_model(values)
                else:
                    from core.study_baselines import StudyBaseline
                    model = StudyBaseline(os.environ['STUDY_BASELINE_CAPSULE'], seed=9471, **values)
                    row['baseline_provenance'] = model.provenance()
                model = model.to(device)
                cfg = training_config(config, case)
                row.update(prefetch=prefetch_options(cfg['training']), comparison_case=case.get('comparison_case'))
                optimizer, scheduler = optimizer_and_scheduler(model, config['optimizer'], config['recipe_total_updates'])
                row.update(parameters=sum(p.numel() for p in model.parameters()), setup_seconds=time.monotonic()-started)
                batch_stream = OrderedBatchStream(train, seed=9471, position=0,
                    microbatch=cfg['training']['microbatch'], end_position=config['updates']*8,
                    device=device, options=prefetch_options(cfg['training']))
                activity = ActivityTrace(gpu.uuid, folder/'gpu-activity.jsonl', interval=.2)
                activity.start()
                measured_start = None
                with (folder / 'training-telemetry.jsonl.gz').open('wb') as raw:
                    with gzip.GzipFile(fileobj=raw, mode='wb') as compressed:
                        for update in range(config['updates']):
                            if time.monotonic() >= deadline:
                                raise ProfileDeadline('Collective softdeadline before next complete effective8update')
                            torch.cuda.reset_peak_memory_stats(device)
                            torch.cuda.synchronize(device)
                            tick = time.perf_counter()
                            if update == config['warmup_updates']:
                                measured_start = time.monotonic()
                            record = train_update(model, optimizer, scheduler, train, cfg, update * 8, device,
                                                  folder / ('failure-update-' + str(update+1) + '.json'), batch_stream)
                            torch.cuda.synchronize(device)
                            record.update(update=update+1, warmup=update < config['warmup_updates'],
                                          seconds_before_logging=time.perf_counter()-tick)
                            compressed.write((json.dumps(serial(record), allow_nan=False)+'\n').encode())
                            compressed.flush()
                            raw.flush()
                            os.fsync(raw.fileno())
                            step = {key: record[key] for key in ('update', 'warmup', 'loss', 'lr_used', 'preclip_global_gradient_l2', 'clipping_coefficient')}
                            step['data_pipeline'] = record['data_pipeline']
                            step['sample_indices'] = record['sample_indices']
                            step['optimizer_moments_all_finite'] = record.get('optimizer_moments_all_finite')
                            if record.get('readout_optimizer_health') is not None:
                                step['readout_optimizer_health'] = compact_readout_health(record)
                            step.update(seconds=time.perf_counter()-tick, peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                                        peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
                            row['steps'].append(step)
                            save()
                measured_end = time.monotonic()
                activity.stop()
                usable = [r['gpu_utilization_percent'] for r in activity.rows
                          if measured_start <= r['monotonic_seconds'] <= measured_end and r['gpu_utilization_percent'] is not None]
                row['gpu_activity'] = {'scope': 'correlated NVML samples, diagnostic only',
                    'gpu_uuid': activity.uuid, 'samples': len(usable), 'sampling_seconds': .2,
                    'mean_gpu_utilization_percent': statistics.mean(usable) if usable else None,
                    'fraction_samples_below_50_percent': sum(x < 50 for x in usable)/len(usable) if usable else None}
                activity = None
                import hashlib
                state_digest = hashlib.sha256()
                for name, tensor in sorted(model.state_dict().items()):
                    state_digest.update(name.encode())
                    state_digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
                row['final_parameter_sha256'] = state_digest.hexdigest()
                batch_stream.close()
                batch_stream = None
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
                tick = time.perf_counter()
                metrics = evaluate_validation(model, val, device, prefetch=prefetch_options(cfg['training']))
                torch.cuda.synchronize(device)
                row['val_one_frame_seconds'] = time.perf_counter()-tick
                row['val_peak_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
                atomic_json(folder / 'validation-engineering-only.json', metrics)
                measured = [x['seconds'] for x in row['steps'] if not x['warmup']]
                row.update(status='COMPLETED_ENGINEERING_TIMING', mean_measured_update_seconds=statistics.mean(measured),
                           measured_update_seconds=measured, measured_presentations=len(measured)*8,
                           diagnostic_optimizer_updates=config['updates'], full_recipe_completed=False, validation_frames=1,
                           diagnostic_train_presentations=config['updates']*8,
                           max_peak_allocated_bytes=max(x['peak_allocated_bytes'] for x in row['steps']))
            except BaseException as error:
                trace = traceback.format_exc()
                (folder / 'traceback.txt').write_text(trace)
                row.update(status='FAILED_ENGINEERING_CASE', error=repr(error), traceback_path=str(folder/'traceback.txt'))
                if isinstance(error, ProfileDeadline):
                    budget_exhausted = True
                    row['status'] = 'STOPPED_COLLECTIVE_TIME_BUDGET'
                save()
            finally:
                if activity is not None:
                    activity.stop()
                if batch_stream is not None:
                    batch_stream.close()
                row['total_case_seconds'] = time.monotonic()-started
                model = optimizer = scheduler = train = val = None
                gc.collect()
                torch.cuda.empty_cache()
                save()
        if config['schema'] in ('stereo-study-prefetch-profile-v1', 'stereo-study-prefetch-parity-v1'):
            comparisons = []
            for original in config['cases']:
                group = [r for r in result['cases'] if r.get('comparison_case') == original['case_id']]
                if len(group) != 4 or any(r['status'] != 'COMPLETED_ENGINEERING_TIMING' for r in group):
                    comparisons.append({'case': original['case_id'], 'status': 'INCOMPLETE_NO_SPEEDUP_CLAIM'})
                    continue
                modes = {}
                for name, workers in [('off', 0), ('on', 2)]:
                    rows = [r for r in group if r['prefetch']['workers'] == workers]
                    steps = [s for r in rows for s in r['steps'] if not s['warmup']]
                    times = sorted(s['seconds'] for s in steps)
                    utilization = [r['gpu_activity']['mean_gpu_utilization_percent'] for r in rows
                                   if r['gpu_activity']['mean_gpu_utilization_percent'] is not None]
                    modes[name] = {'mean_update_seconds': statistics.mean(times),
                        'p50_update_seconds': statistics.median(times),
                        'p95_update_seconds': times[min(len(times)-1, int(.95*len(times)))],
                        'mean_host_wait_seconds': statistics.mean(s['data_pipeline']['host_wait_seconds'] for s in steps),
                        'mean_h2d_cuda_seconds': statistics.mean(s['data_pipeline']['h2d_cuda_seconds'] for s in steps),
                        'mean_gpu_utilization_percent': statistics.mean(utilization) if utilization else None,
                        'max_peak_allocated_bytes': max(r['max_peak_allocated_bytes'] for r in rows),
                        'replicate_mean_update_seconds': [r['mean_measured_update_seconds'] for r in rows]}
                comparisons.append({'case': original['case_id'], 'status': 'COMPLETED_BOUNDED_COMPARISON',
                    'modes': modes, 'throughput_ratio_on_over_off': modes['off']['mean_update_seconds']/modes['on']['mean_update_seconds'],
                    'all_final_parameters_byte_identical': len({r['final_parameter_sha256'] for r in group}) == 1,
                    'all_loss_trajectories_identical': all([s['loss'] for s in r['steps']] == [s['loss'] for s in group[0]['steps']] for r in group)})
            result['prefetch_comparisons'] = comparisons
        result['status'] = 'COMPLETED_PROFILE_WITH_CASE_FAILURES' if any(x['status']=='FAILED_ENGINEERING_CASE' for x in result['cases']) else 'STOPPED_COLLECTIVE_TIME_BUDGET' if budget_exhausted else 'COMPLETED_ENGINEERING_PROFILE'
    except ProfileDeadline as error:
        result['status'] = 'STOPPED_COLLECTIVE_TIME_BUDGET'
        result['deadline_error'] = repr(error)
        done = {row['case_id'] for row in result['cases']}
        for row in result['cases']:
            if row['status'] == 'RUNNING':
                row['status'] = 'STOPPED_COLLECTIVE_TIME_BUDGET'
        for case in cases:
            if case['case_id'] not in done:
                result['cases'].append({'case_id': case['case_id'], 'status': 'SKIPPED_COLLECTIVE_TIME_BUDGET'})
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        result['elapsed_seconds'] = time.monotonic()-began
        result['completed_cases'] = sum(x['status']=='COMPLETED_ENGINEERING_TIMING' for x in result['cases'])
        result['failed_cases'] = sum(x['status']=='FAILED_ENGINEERING_CASE' for x in result['cases'])
        result['skipped_cases'] = sum(x['status'].startswith('SKIPPED') for x in result['cases'])
        save()


if __name__ == '__main__':
    main()
