"""Frozen-horizon Train/Val runner; authorized4090/5090 Slurm execution only.

This command submits no jobs and never constructs a Test loader. A bounded
--stop-at-update pauses the same frozen schedule; it cannot alter its horizon.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from core.study_training import (append_json, atomic_json, digest, evaluate_validation,
                                 file_sha, optimizer_and_scheduler, restore_checkpoint,
                                 save_checkpoint, train_update, training_sample_indices, validate_config, validation_due)


def verify_source(path):
    registry = json.loads(path.read_text())
    for name, expected in registry['source_files'].items():
        if file_sha(path.parent / name) != expected:
            raise RuntimeError('Frozen source member changed: ' + name)
    original = hashlib.sha256(json.dumps(registry['source_files'], sort_keys=True).encode()).hexdigest()
    if original != registry['source_identity_sha256']:
        raise RuntimeError('Frozen source registry identity mismatch')
    return registry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--source-registry', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-at-update', type=int)
    parser.add_argument('--validate-config', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    validate_config(config, require_registered=not args.validate_config)
    if args.validate_config:
        print(json.dumps({'status': 'VALID_CONFIG_ONLY', 'registered': config['registered'],
                          'scope': config['scope'], 'config_sha256': file_sha(args.config)}))
        return
    if not os.environ.get('SLURM_JOB_ID') or not args.source_registry or not args.output:
        raise RuntimeError('Execution requires authorized Slurm allocation, frozen source and output location')
    import numpy as np
    import torch
    from core.study_data import PairedStereoDataset
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    device = torch.device('cuda:0')
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one authorized GPU must be visible per training job')
    gpu = torch.cuda.get_device_properties(device)
    if not any(name in gpu.name for name in ('4090', '5090')):
        raise RuntimeError('Only authorizedRTX4090/5090 hardware is supported')
    registry = verify_source(args.source_registry)
    total = config['total_updates']
    stop_at = total if args.stop_at_update is None else args.stop_at_update
    if not 0 < stop_at <= total:
        raise ValueError('Stage boundary must be inside the frozen total horizon')
    data = config['data']
    root = Path(os.environ[data['root_env']])
    manifest = Path(os.environ[data['manifest_env']])
    if file_sha(manifest) != data['manifest_sha256']:
        raise RuntimeError('Paired manifest changed')
    purpose = 'engineering' if config['scope'] == 'engineering' else 'study'
    train = PairedStereoDataset(root, manifest, data['modality'], 'train', purpose=purpose,
                               sample_ids=data.get('train_sample_ids'))
    val = PairedStereoDataset(root, manifest, data['modality'], 'val', purpose=purpose,
                             sample_ids=data.get('val_sample_ids'))
    # Primary training requires scene-disjoint Train/Val/Test metadata; the
    # dataset validates Test identities without opening a single Test asset.
    if config['scope'] != 'engineering' and train.manifest['schema'] != 'stereo-paired-v1':
        raise ValueError('Causal pilot is engineering only, never final/tuning evidence')
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    # Determinism is recorded explicitly; exact resume means restored source,
    # state, sample sequence and RNG, without claiming all GPU kernels bitwise.
    torch.backends.cudnn.deterministic = True
    model_config = dict(config['model']['config'])
    if config['model']['family'] == 'study':
        from core.study_models import build_model
        model = build_model(model_config)
    else:
        from core.study_baselines import StudyBaseline
        model = StudyBaseline(capsule=os.environ[config['model']['capsule_env']], seed=config['seed'], **model_config)
    model = model.to(device)
    initial_state_digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        initial_state_digest.update(name.encode())
        initial_state_digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    optimizer, scheduler = optimizer_and_scheduler(model, config['optimizer'], total)
    output = args.output
    if output.resolve().is_relative_to(args.source_registry.parent.resolve()):
        raise ValueError('Run artifacts must be outside the immutable source capsule')
    if args.resume:
        if not (output / 'latest.pt').is_file():
            raise RuntimeError('Resume requires the latest complete optimizer checkpoint')
    else:
        output.mkdir(parents=True, exist_ok=False)
    logs = {'training': output / 'training.jsonl', 'validation': output / 'validation.jsonl'}
    binding = {'config_sha256': file_sha(args.config), 'config_identity': digest(config),
               'source_identity_sha256': registry['source_identity_sha256'], 'git_commit': registry['git_commit'],
               'source_registry_sha256': file_sha(args.source_registry), 'manifest_sha256': data['manifest_sha256'],
               'total_updates': total, 'effective_batch': config['training']['effective_batch'],
               'training_samples': len(train), 'validation_samples': len(val),
               'train_sample_ids_sha256': digest([row['sample_id'] for row in train.rows]),
               'val_sample_ids_sha256': digest([row['sample_id'] for row in val.rows]),
               'gpu_name': gpu.name, 'precision': 'float32', 'tf32': False}
    metadata = {'schema': 1, 'config': config, 'binding': binding,
                'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'command': sys.argv, 'stage_stop_at_update': stop_at,
                'slurm_job_id': os.environ['SLURM_JOB_ID'], 'gpu': gpu.name, 'gpu_total_bytes': gpu.total_memory,
                'slurm_cluster': os.environ.get('SLURM_CLUSTER_NAME'), 'slurm_nodes': os.environ.get('SLURM_JOB_NODELIST'),
                'slurm_job_gpus': os.environ.get('SLURM_JOB_GPUS'), 'visible_cuda_devices': torch.cuda.device_count(),
                'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                'torch': torch.__version__, 'cuda': torch.version.cuda, 'python': sys.version,
                'source_registry': str(args.source_registry.resolve()), 'source_dirty_status': registry.get('dirty_status'),
                'source_overlay_sha256': registry.get('overlay_sha256'), 'raw_output': str(output.resolve()),
                'parameters': sum(p.numel() for p in model.parameters()),
                'fresh_seed_model_sha256': initial_state_digest.hexdigest(),
                'pretraining': False, 'test_access': 'sealed', 'dataset_schema': train.manifest['schema'],
                'scope': config['scope'], 'baseline_provenance': model.provenance() if hasattr(model, 'provenance') else None,
                'occlusion_supervision': 'only_explicit_trusted_labels; absent_labels_zero_loss',
                'optimizer_boundary': 'oneAdamW+oneScheduler step per8presentations',
                'resume_claim': 'exact source/config/horizon/state/RNG/sampleposition; GPU bitwise kernel equality unverified'}
    state = {'update': 0, 'presentations': 0, 'best_val_epe': None, 'best_update': None,
             'status': 'INITIALIZED', 'last_validation_update': None}
    if args.resume:
        state = restore_checkpoint(output / 'latest.pt', model, optimizer, scheduler, binding, logs)
        append_json(output / 'resume-history.jsonl', {**metadata, 'restored_update': state['update']})
    else:
        atomic_json(output / 'metadata.json', metadata)
        save_checkpoint(output / 'latest.pt', model, optimizer, scheduler, state, binding, logs)
    if stop_at < state['update']:
        raise ValueError('Stage boundary precedes restored optimizer position')
    stopping = {'signal': None}
    def stop_handler(number, frame):
        stopping['signal'] = number
    for name in ('SIGTERM', 'SIGINT', 'SIGUSR1'):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), stop_handler)
    def save_latest(status):
        state['status'] = status
        save_checkpoint(output / 'latest.pt', model, optimizer, scheduler, state, binding, logs)
        atomic_json(output / 'status.json', {**state, 'slurm_job_id': os.environ['SLURM_JOB_ID'],
                                           'stage_stop_at_update': stop_at, 'total_horizon_updates': total,
                                           'signal': stopping['signal'], 'scope': config['scope'], 'test_access': 'sealed'})
    from core.study_prefetch import OrderedBatchStream, prefetch_options
    def make_batch_stream():
        return OrderedBatchStream(train, seed=config['seed'], position=state['presentations'],
                                  microbatch=config['training']['microbatch'],
                                  end_position=stop_at*config['training']['effective_batch'], device=device,
                                  options=prefetch_options(config['training']),
                                  indices_fn=lambda seed, pos, count, length: training_sample_indices(config, pos, count, length))
    batch_stream = make_batch_stream()
    try:
        state['status'] = 'RUNNING'
        if (config['training'].get('evaluate_initial', False) and state['update'] == 0
                and state['last_validation_update'] is None and stopping['signal'] is None):
            batch_stream.close()
            batch_stream = None
            validation_started = time.perf_counter()
            metrics = evaluate_validation(model, val, device, lambda: stopping['signal'] is not None,
                                          prefetch=prefetch_options(config['training']))
            if metrics is not None:
                append_json(logs['validation'], {'update': 0, 'presentations': 0, 'split': 'val',
                    'scope': config['scope'], 'untrained_reference': True,
                    'seconds': time.perf_counter()-validation_started, **metrics})
                state['last_validation_update'] = 0
                # Initial Val is descriptive and never eligible for best-val.
                save_latest('RUNNING')
            if stopping['signal'] is None:
                batch_stream = make_batch_stream()
        while state['update'] < stop_at and stopping['signal'] is None:
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            record = train_update(model, optimizer, scheduler, train, config, state['presentations'], device,
                                  output / ('failure-update-' + str(state['update'] + 1) + '.json'),
                                  batch_stream=batch_stream)
            torch.cuda.synchronize(device)
            state['update'] += 1
            state['presentations'] += config['training']['effective_batch']
            append_json(logs['training'], {**record, 'update': state['update'], 'total_presentations': state['presentations'],
                                           'seconds': time.perf_counter() - started, 'scope': config['scope']})
            evaluate_now = validation_due(config,state['update'])
            if evaluate_now and stopping['signal'] is None:
                # Keep the global one-device-lookahead bound during validation.
                # Speculative Train data are reconstructed at the committed
                # position afterward, exactly as on a checkpoint resume.
                batch_stream.close()
                batch_stream = None
                validation_started = time.perf_counter()
                metrics = evaluate_validation(model, val, device, lambda: stopping['signal'] is not None,
                                              prefetch=prefetch_options(config['training']))
                if metrics is not None:
                    append_json(logs['validation'], {'update': state['update'], 'presentations': state['presentations'],
                                                     'split': 'val', 'scope': config['scope'],
                                                     'seconds': time.perf_counter() - validation_started, **metrics})
                    state['last_validation_update'] = state['update']
                    if state['best_val_epe'] is None or metrics['mean_scene_epe'] < state['best_val_epe']:
                        state['best_val_epe'], state['best_update'] = metrics['mean_scene_epe'], state['update']
                        save_checkpoint(output / 'best-val.pt', model, optimizer, scheduler, state, binding, logs)
                    save_latest('RUNNING')
                if state['update'] < stop_at and stopping['signal'] is None:
                    batch_stream = make_batch_stream()
            if state['update'] % config['training']['checkpoint_interval'] == 0:
                save_latest('RUNNING')
        status = 'PAUSED_SIGNAL' if stopping['signal'] is not None else 'COMPLETED_FROZEN_HORIZON' if state['update'] == total else 'PAUSED_STAGE_BOUNDARY'
        save_latest(status)
    except BaseException as error:
        # Never replace latest with a possibly partial/bad optimizer update.
        atomic_json(output / 'status.json', {**state, 'status': 'FAILED_RESTORE_LATEST_REQUIRED', 'error': repr(error),
                                           'scope': config['scope'], 'test_access': 'sealed'})
        raise
    finally:
        if batch_stream is not None:
            batch_stream.close()


if __name__ == '__main__':
    main()
