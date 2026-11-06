"""Resumable study training primitives; GPU execution lives in train_study.py.

No Test dataset is constructed here. Tiny toy unit tests may exercise control
flow on CPU; all actual study optimization/evaluation requires authorized GPU
Slurm execution. A saved horizon, source, config and manifest cannot change on
resume. Mid-update failures preserve diagnostics and restart the last complete
optimizer boundary, with its original RNG and sample position.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from core.study_losses import disparity_loss
from core.study_prefetch import OrderedBatchStream, prefetch_options


FINAL_SEEDS = (1234, 2345, 3456)
TUNING_SEED = 9471


def serial(value):
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    if isinstance(value, torch.Tensor):
        x = value.detach().cpu()
        return serial(x.item() if x.numel() == 1 else x.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return serial(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(serial(value), indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def append_json(path, value):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(serial(value), allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def validate_config(config, *, require_registered=True):
    if config.get('schema') != 'stereo-study-training-v1':
        raise ValueError('Unrecognized training schema')
    if require_registered and config.get('registered') is not True:
        raise ValueError('Example/unregistered configuration cannot launch training')
    if config.get('scope') not in ('engineering', 'tuning', 'final', 'supporting'):
        raise ValueError('Declare engineering/tuning/final/supporting evidence scope')
    if config.get('pretraining') is not False or config.get('test_access') != 'sealed':
        raise ValueError('Primary controlled training requires no pretraining and sealed Test')
    if config['scope'] == 'final' and config['seed'] not in FINAL_SEEDS:
        raise ValueError('Fresh final seeds are1234/2345/3456')
    if config['scope'] == 'tuning' and config['seed'] != TUNING_SEED:
        raise ValueError('Tuning uses seed9471, separate from fresh final seeds')
    horizon = config['total_updates']
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError('Freeze a positive complete optimizer-update horizon')
    training = config['training']
    prefetch_options(training)
    if training['effective_batch'] != 8 or training['microbatch'] not in (1, 2, 4, 8):
        raise ValueError('Primary effectivebatch8 requires a supported exact divisor')
    if training.get('precision') != 'float32' or training.get('augmentation') != 'none':
        raise ValueError('This version registers FP32 and no undeclared augmentation')
    if training['eval_interval'] < 1 or training['checkpoint_interval'] < 1:
        raise ValueError('Positive evaluation/checkpoint intervals required')
    evaluation_updates = training.get('evaluation_updates')
    if evaluation_updates is not None:
        if (not isinstance(evaluation_updates,list) or not evaluation_updates
                or any(type(x) is not int or not 1 <= x <= horizon for x in evaluation_updates)
                or evaluation_updates != sorted(set(evaluation_updates)) or evaluation_updates[-1] != horizon):
            raise ValueError('Explicit evaluation updates must be unique sorted positive boundaries ending at the frozen horizon')
    if training.get('diagnostic_interval', 50) < 1:
        raise ValueError('Positive full-telemetry diagnostic interval required')
    if type(training.get('readout_optimizer_health', False)) is not bool:
        raise ValueError('Readout optimizer health option must be boolean')
    sampling = training.get('sampling', 'epoch_permutation')
    if sampling not in ('epoch_permutation', 'repeated_panel_fixed_order'):
        raise ValueError('Unknown training sample ordering')
    if sampling == 'repeated_panel_fixed_order':
        panel = config['data'].get('train_sample_ids', [])
        if config['scope'] != 'engineering' or len(panel) != 8 or len(set(panel)) != 8:
            raise ValueError('Repeated health panel requires eight explicit unique engineering Train IDs')
    model = config['model']
    if model['family'] == 'study':
        if model['config']['time_steps'] != 50:
            raise ValueError('All50 sensor steps with full temporal gradients required')
        if model['config'].get('collect_diagnostics', True) is not True:
            raise ValueError('Trainer retains GPU diagnostics for failure audits; disabled telemetry belongs to a separate inference/profile pass')
    elif model['family'] == 'baseline':
        if model['config']['architecture'] not in ('original', 'original_I'):
            raise ValueError('Unknown registered recurrent baseline')
    else:
        raise ValueError('Unknown family')
    if config['data']['modality'] not in ('spike', 'event'):
        raise ValueError('Unknown sensor modality')
    channels = 1 if config['data']['modality'] == 'spike' else 2
    if model['config']['input_channels'] != channels:
        raise ValueError('Preserve registered ON/OFF channels for events')
    if config['optimizer'].get('warmup_fraction') != .05 or config['optimizer'].get('minimum_lr_ratio') != .1:
        raise ValueError('This recipe uses5%warmup and cosine to0.1peakLR')
    if config['optimizer'].get('weight_decay') != 1e-4:
        raise ValueError('This recipe decays convolution/matrix weights at1e-4')
    if config.get('neuron_regularization') is not None and config['scope'] != 'supporting':
        raise ValueError('Neuron regularization belongs to a separately registered supporting arm')
    if config.get('neuron_regularization') is not None and model['family'] != 'baseline':
        raise ValueError('Registered legacy neuron regularizer only applies to baseline refinement outputs')
    if require_registered and len(config['data'].get('manifest_sha256') or '') != 64:
        raise ValueError('Registered training requires immutable paired manifest identity')
    return True


def sample_indices(seed, position, count, length):
    """Same scene presentations across architectures/modalities; no global RNG."""
    if length < 1 or min(position, count) < 0:
        raise ValueError('Invalid sample sequence request')
    result = []
    while count:
        epoch, offset = divmod(position, length)
        generator = torch.Generator().manual_seed(int(seed) + 1000003 * epoch)
        permutation = torch.randperm(length, generator=generator).tolist()
        take = min(count, length - offset)
        result.extend(permutation[offset:offset + take])
        position, count = position + take, count - take
    return result


def training_sample_indices(config, position, count, length):
    """The diagnostic panel repeats in order; ordinary study sampling is unchanged."""
    if config['training'].get('sampling', 'epoch_permutation') == 'repeated_panel_fixed_order':
        if config.get('scope') != 'engineering' or length != 8 or len(config['data'].get('train_sample_ids', [])) != 8:
            raise ValueError('Fixed panel is only an explicitly registered eight-example engineering diagnostic')
        return [(position + offset) % length for offset in range(count)]
    return sample_indices(config['seed'], position, count, length)


def validation_due(config, update):
    explicit = config['training'].get('evaluation_updates')
    if explicit is not None:
        return update in explicit
    return update % config['training']['eval_interval'] == 0 or update == config['total_updates']


def collate_samples(samples):
    """Keep None-containing metadata as records; never default-collate it."""
    if not samples:
        raise ValueError('Empty microbatch')
    tensor_keys = ('left', 'right', 'target', 'valid', 'right_target', 'right_valid', 'occlusion_target', 'occlusion_valid')
    result = {}
    for key in tensor_keys:
        present = [key in sample and sample[key] is not None for sample in samples]
        if any(present) and not all(present):
            raise ValueError('Mixed supervision availability within microbatch: ' + key)
        if all(present):
            result[key] = torch.stack([sample[key] for sample in samples])
    for key in ('sample_id', 'scene_id', 'metadata'):
        result[key] = [sample.get(key) for sample in samples]
    return result


def optimizer_and_scheduler(model, config, total_updates):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (decay if name.endswith('weight') and parameter.ndim >= 2 else no_decay).append(parameter)
    optimizer = torch.optim.AdamW([{'params': decay, 'weight_decay': config['weight_decay']},
                                   {'params': no_decay, 'weight_decay': 0.}],
                                  lr=config['lr'], eps=config.get('epsilon', 1e-8))
    warmup = max(1, math.ceil(config['warmup_fraction'] * total_updates))
    floor = config['minimum_lr_ratio']
    def multiplier(index):
        if index < warmup:
            return (index + 1) / warmup
        progress = min(1., max(0., (index - warmup) / max(1, total_updates - warmup - 1)))
        return floor + (1 - floor) * .5 * (1 + math.cos(math.pi * progress))
    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def capture_rng():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA checkpoint requires CUDA RNG restoration')
        torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])


def log_binding(path):
    path = Path(path)
    raw = path.read_bytes() if path.exists() else b''
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def reconcile_log(path, binding):
    """Preserve uncommitted crash tails, then restore exact checkpoint prefix."""
    path = Path(path)
    raw = path.read_bytes() if path.exists() else b''
    size = binding['bytes']
    if len(raw) < size or hashlib.sha256(raw[:size]).hexdigest() != binding['sha256']:
        raise RuntimeError('Missing or modified checkpoint-bound log prefix: ' + str(path))
    if len(raw) > size:
        preserved = path.with_name(path.name + '.uncommitted-tail-' + str(time.time_ns()))
        preserved.write_bytes(raw[size:])
        path.write_bytes(raw[:size])


def save_checkpoint(path, model, optimizer, scheduler, state, binding, logs):
    state = dict(state)
    if state['presentations'] != state['update'] * binding['effective_batch']:
        raise RuntimeError('Checkpoint must be at a complete effective-batch boundary')
    payload = {'schema': 1, 'binding': binding, 'state': state, 'model': model.state_dict(),
               'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'rng': capture_rng(),
               'logs': {name: log_binding(value) for name, value in logs.items()}}
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('wb') as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def restore_checkpoint(path, model, optimizer, scheduler, expected_binding, logs):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload['binding'] != expected_binding:
        raise RuntimeError('Resume changes frozen horizon/config/source/manifest/exposure binding')
    state = payload['state']
    if state['presentations'] != state['update'] * expected_binding['effective_batch']:
        raise RuntimeError('Checkpoint sample position is not an optimizer boundary')
    for name, path in logs.items():
        reconcile_log(path, payload['logs'][name])
    model.load_state_dict(payload['model'], strict=True)
    optimizer.load_state_dict(payload['optimizer'])
    scheduler.load_state_dict(payload['scheduler'])
    restore_rng(payload['rng'])
    return state


def primary_loss(output, batch, config):
    if config['model']['family'] == 'study':
        return disparity_loss(output, batch['target'], batch['valid'], config['model']['config']['max_disparity'],
                              **{k: batch[k] for k in ('right_target', 'right_valid', 'occlusion_target', 'occlusion_valid') if k in batch})
    # Legacy architecture keeps its sequence supervision, with trusted positive
    # GT and a per-sample valid mean like the new primary final-map objective.
    predictions = output['refinement_disparities']
    target = batch['target']
    valid = batch['valid'].bool() & torch.isfinite(target) & (target > 0)
    counts = valid.flatten(1).sum(1)
    if not (counts > 0).all():
        raise ValueError('Training samples require trusted positive disparity pixels')
    terms = {}
    total = target.new_zeros(())
    gamma = .9 ** (15 / max(1, len(predictions) - 1))
    for index, prediction in enumerate(predictions):
        error = (prediction - torch.where(valid, target, torch.zeros_like(target))).abs()
        term = (error * valid).flatten(1).sum(1).div(counts).mean()
        total = total + gamma ** (len(predictions) - index - 1) * term
        terms['refinement_' + str(index)] = term
    regularizer = config.get('neuron_regularization')
    if regularizer is not None:
        if not output['refinement_voltages']:
            raise ValueError('Supporting neuron regularization requires training membrane outputs')
        spikes = torch.stack(output['refinement_spikes'], 1)
        voltages = torch.stack(output['refinement_voltages'], 1)
        term = regularizer['lambda_rate'] * ((spikes.mean((0, 1)) - regularizer['target_rate']) ** 2).sum()
        term = term + regularizer['lambda_voltage'] * voltages.square().mean()
        total = total + term
        terms['supporting_neuron_regularizer'] = term
    return total, terms


def tensor_audit(tensor):
    x = tensor.detach()
    finite = torch.isfinite(x)
    safe = torch.where(finite, x, torch.zeros_like(x)).double()
    maximum = safe.abs().max() if x.numel() else safe.new_zeros(())
    # Reduce on the tensor's device and transfer four scalars in one sync;
    # ordinary predictions/parameters never need full CPU tensor copies.
    summary = torch.stack((finite.all().double(), (~finite).sum().double(), safe.square().sum().sqrt(), maximum)).cpu().tolist()
    return {'shape': list(x.shape), 'all_finite': bool(summary[0]),
            'nonfinite_count': int(summary[1]), 'finite_l2': summary[2], 'finite_abs_max': summary[3]}


def all_finite(tensors):
    """Same element coverage with one host synchronization per whole guard."""
    flags = [torch.isfinite(tensor).all() for tensor in tensors]
    return bool(torch.stack(flags).all()) if flags else True


def gradient_audit(model, left, right):
    def sensor(x):
        if x.grad is None:
            return {'missing': True}
        return {'per_sensor_step': [tensor_audit(x.grad[:, t]) for t in range(x.shape[1])]}
    return {'left': sensor(left), 'right': sensor(right),
            'parameters': {name: tensor_audit(p.grad) if p.grad is not None else {'missing': True}
                           for name, p in model.named_parameters() if p.requires_grad}}


def stable_clip_grad_norm(model, maximum, *, extreme_threshold=1e12, diagnostics=None):
    named = [(name, p.grad) for name, p in model.named_parameters() if p.grad is not None]
    gradients = [gradient for name, gradient in named]
    if not gradients:
        raise FloatingPointError('No parameter gradients')
    if maximum <= 0 or not all_finite(gradients):
        raise FloatingPointError('Invalid clipping bound or genuinely nonfinite gradient elements')
    squared_norms = torch.stack([g.detach().double().square().sum() for g in gradients])
    norm = squared_norms.sum().sqrt()
    if not bool(torch.isfinite(norm)):
        raise FloatingPointError('Nonfinite FP64 global gradient norm')
    coefficient = (maximum / (norm + 1e-12)).clamp(max=1.)
    if diagnostics is not None:
        diagnostics.update(threshold=extreme_threshold, is_extreme=bool(norm > extreme_threshold))
        if diagnostics['is_extreme']:
            values, indices = squared_norms.topk(min(10, len(named)))
            diagnostics['top10_named_preclip_gradient_l2'] = [
                {'name': named[index][0], 'gradient_l2': math.sqrt(value)}
                for index, value in zip(indices.cpu().tolist(), values.cpu().tolist())]
            parameters = dict(model.named_parameters())
            direction = {}
            for name, gradient in named:
                g = gradient.detach().cpu().double()
                p = parameters[name].detach().cpu().double()
                gn, pn = float(g.square().sum().sqrt()), float(p.square().sum().sqrt())
                dot = float((g*p).sum())
                direction[name] = {'gradient_l2': gn, 'parameter_l2': pn, 'gradient_parameter_dot': dot,
                                   'gradient_parameter_cosine': dot / (gn*pn) if gn*pn else None,
                                   'gradient_signed_mean': float(g.mean()), 'gradient_abs_mean': float(g.abs().mean()),
                                   'positive_fraction': float((g>0).double().mean()),
                                   'negative_fraction': float((g<0).double().mean()), 'zero_fraction': float((g==0).double().mean())}
            diagnostics['_raw_parameter_direction_statistics'] = direction
    for gradient in gradients:
        # Avoid a tiny FP32 scalar flushing to zero on GPU before multiplication.
        gradient.copy_((gradient.double() * coefficient).to(gradient.dtype))
    return norm, coefficient


def train_update(model, optimizer, scheduler, dataset, config, position, device, failure_path, batch_stream=None):
    """One atomic optimizer update; audits are written before propagating failure."""
    effective, micro = config['training']['effective_batch'], config['training']['microbatch']
    indices = training_sample_indices(config, position, effective, len(dataset))
    update = position // effective + 1
    detailed = (update == 1 or update % config['training'].get('diagnostic_interval', 50) == 0
                or update in config['training'].get('diagnostic_updates', []))
    optimizer.zero_grad(set_to_none=True)
    weighted_loss, terms_sum, micro_records = 0., {}, []
    last_full_diagnostics = None
    own_stream = batch_stream is None
    if own_stream:
        batch_stream = OrderedBatchStream(dataset, seed=config['seed'], position=position, microbatch=micro,
                                         end_position=position+effective, device=device,
                                         options=prefetch_options(config['training']),
                                         indices_fn=lambda seed, pos, count, length: training_sample_indices(config, pos, count, length))
    pipeline_timings, copy_events = [], []
    model.train()
    try:
        for start in range(0, effective, micro):
            selected = indices[start:start + micro]
            batch, loaded_indices, pipeline_timing, events = batch_stream.take(position+start)
            if loaded_indices != selected:
                raise RuntimeError('Prefetch changed registered sample order')
            pipeline_timings.append(pipeline_timing)
            copy_events.append(events)
            trusted = batch['valid'].bool() & torch.isfinite(batch['target']) & (batch['target'] > 0)
            if not bool((trusted.flatten(1).sum(1) > 0).all()):
                raise ValueError('Every training sample requires trusted positive disparity')
            left, right = batch['left'].requires_grad_(), batch['right'].requires_grad_()
            output = model(left, right)
            # Only a ready batch is copied; never block current GPU work on a
            # speculative decode. One device batch is the strict lookahead cap.
            batch_stream.stage_next()
            prediction = output['disparity']
            loss, terms = primary_loss(output, batch, config)
            scalar_keys = list(terms)
            scalar_values = torch.stack([loss.detach()] + [terms[key].detach() for key in scalar_keys]).double().cpu().tolist()
            loss_value = scalar_values[0]
            last_full_diagnostics = output.get('diagnostics', {})
            compact = {key: last_full_diagnostics[key] for key in ('sensor_steps', 'spiking', 'hybrid', 'attention_normalization',
                        'feature_activity', 'refinement_iterations', 'temporal_semantics') if key in last_full_diagnostics}
            record = {'indices': selected, 'sample_ids': batch['sample_id'],
                      'loss': loss_value if math.isfinite(loss_value) else None,
                      'loss_all_finite': math.isfinite(loss_value),
                      'prediction': tensor_audit(prediction), 'activity': serial(compact),
                      'terms': {key: value if math.isfinite(value) else None for key, value in zip(scalar_keys, scalar_values[1:])}}
            micro_records.append(record)
            if not record['loss_all_finite'] or not record['prediction']['all_finite']:
                raise FloatingPointError('Nonfinite forward loss/prediction')
            (loss * (len(selected) / effective)).backward()
            # Sensor gradients expose instability hidden by finite scalar loss.
            sensors_ok = left.grad is not None and right.grad is not None and all_finite((left.grad, right.grad))
            parameters_ok = all_finite(p.grad for p in model.parameters() if p.grad is not None)
            if not sensors_ok or not parameters_ok:
                record['gradients'] = gradient_audit(model, left, right)
                raise FloatingPointError('Missing/nonfinite sensor or parameter gradients before optimizer update')
            record['sensor_gradient_mean_per_time'] = {
                key: x.grad.detach().double().abs().mean((0, 2, 3, 4)).cpu().tolist() for key, x in (('left', left), ('right', right))}
            record['sensor_gradient_elements_all_finite'] = True
            record['parameter_gradient_elements_all_finite'] = True
            if detailed and start + micro >= effective:
                record['activity'] = serial(last_full_diagnostics)
            weighted_loss += loss_value * len(selected) / effective
            for key, value in record['terms'].items():
                terms_sum[key] = terms_sum.get(key, 0.) + value * len(selected) / effective
            # Release the previous microbatch checkpoint graph before building
            # the next forward graph. Retained telemetry is detached only.
            del output, loss, terms, prediction, batch, left, right, trusted
            batch_stream.stage_next()
        monitor = capture = None
        named_preclip_norms = None
        if config['training'].get('readout_optimizer_health', False):
            from core.study_optimizer_diagnostics import ReadoutAdamWDiagnostics, ReadoutTarget, default_readout_targets
            def diagnostic_role(name):
                if '.qkv.' in name:
                    return 'qkv_projection_kernel' if name.endswith('.conv.weight') else 'qkv_normalization_scale'
                return 'early_feature_normalization_bias' if name.startswith('stages.0.') else 'diagnostic_parameter'
            targets = default_readout_targets(model) + [ReadoutTarget(name, diagnostic_role(name))
                for name in config['training'].get('health_parameter_names', [])]
            monitor = ReadoutAdamWDiagnostics(model, optimizer, targets)
            capture = monitor.before_clip()
            if detailed:
                named = [(name,p.grad) for name,p in model.named_parameters() if p.grad is not None]
                values = torch.stack([g.detach().double().square().sum().sqrt() for name,g in named]).cpu().tolist()
                named_preclip_norms = dict(zip([name for name,g in named],values))
        extreme_audit = {} if update == 1 else None
        preclip, coefficient = stable_clip_grad_norm(model, config['training']['clip_grad'],
            extreme_threshold=config['training'].get('extreme_gradient_threshold', 1e12), diagnostics=extreme_audit)
        if extreme_audit and extreme_audit.get('is_extreme'):
            direction_path = Path(failure_path).parent / ('finite-extreme-gradient-directions-update-' + str(update) + '.json')
            atomic_json(direction_path, {'update': update, 'before_clipping': True,
                'global_gradient_l2': float(preclip), 'parameters': extreme_audit.pop('_raw_parameter_direction_statistics')})
            extreme_audit['raw_parameter_direction_path'] = str(direction_path)
        lr = optimizer.param_groups[0]['lr']
        if monitor is not None:
            monitor.after_clip_before_step(capture)
        optimizer.step()
        if not all_finite(model.parameters()):
            raise FloatingPointError('Nonfinite parameter after optimizer update; restore prior checkpoint')
        readout_health = monitor.after_step(capture) if monitor is not None else None
        moments_finite = None
        if monitor is not None:
            moments_finite = all_finite(value for state in optimizer.state.values()
                                       for key, value in state.items() if key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'))
            if not moments_finite:
                raise FloatingPointError('Nonfinite Adam moment after optimizer update; restore prior checkpoint')
        scheduler.step()
        pipeline = {key: sum(row[key] for row in pipeline_timings)
                    for key in ('cpu_prepare_seconds', 'host_wait_seconds', 'h2d_submit_seconds')}
        pipeline['h2d_cuda_seconds'] = sum(begin.elapsed_time(end)/1000 for begin, end in copy_events if begin is not None)
        pipeline['options'] = batch_stream.options
        return {'loss': weighted_loss, 'terms': terms_sum, 'lr_used': lr, 'preclip_global_gradient_l2': float(preclip),
                'gradient_norm_reduction_dtype': 'float64', 'clipping_coefficient': float(coefficient),
                'full_activity_logged': detailed,
                'data_pipeline': pipeline,
                'readout_optimizer_health': readout_health, 'optimizer_moments_all_finite': moments_finite,
                'named_preclip_gradient_l2': named_preclip_norms,
                'first_update_extreme_gradient_audit': extreme_audit,
                'sample_indices': indices, 'presentations': effective,
                'sensor_gradient_mean_per_time': micro_records[-1]['sensor_gradient_mean_per_time'],
                'last_microbatch_activity': micro_records[-1]['activity']}
    except BaseException as error:
        if micro_records and last_full_diagnostics is not None:
            try:
                micro_records[-1]['activity'] = serial(last_full_diagnostics)
            except BaseException as audit_error:
                micro_records[-1]['activity_audit_error'] = repr(audit_error)
        if micro_records and 'gradients' not in micro_records[-1] and 'left' in locals():
            try:
                micro_records[-1]['gradients'] = gradient_audit(model, left, right)
            except BaseException as audit_error:
                micro_records[-1]['gradient_audit_error'] = repr(audit_error)
        failure = {'status': 'FAILED_UPDATE_NOT_COMMITTED', 'presentation_position': position,
                   'indices': indices, 'error': repr(error), 'microbatches': micro_records,
                   'readout_optimizer_health': locals().get('readout_health')}
        try:
            failure['parameter_audit'] = {name: tensor_audit(p) for name, p in model.named_parameters()}
        except BaseException as audit_error:
            failure['parameter_audit_error'] = repr(audit_error)
        atomic_json(failure_path, failure)
        raise
    finally:
        if own_stream:
            batch_stream.close()


@torch.no_grad()
def evaluate_validation(model, dataset, device, should_stop=lambda: False, prefetch=None):
    """Val only; scene-equal primary EPE, no GT prediction filtering."""
    model.eval()
    scenes, frames = {}, []
    with OrderedBatchStream(dataset, seed=0, position=0, microbatch=1, end_position=len(dataset),
                            device=device, options=prefetch,
                            indices_fn=lambda seed, position, count, length: list(range(position, position+count))) as stream:
        for index in range(len(dataset)):
            if should_stop():
                return None
            sample, _, _, _ = stream.take(index)
            left, right, target, valid = [sample[k] for k in ('left', 'right', 'target', 'valid')]
            prediction = model(left, right)['disparity']
            stream.stage_next()
            if prediction.shape != target.shape or not torch.isfinite(prediction).all():
                raise FloatingPointError('Nonfinite or shape-invalid validation disparity')
            valid = valid.bool() & torch.isfinite(target) & (target > 0)
            error = (prediction - target).abs()[valid]
            count = error.numel()
            if not count:
                raise ValueError('Validation sample has no trusted positive disparity')
            row = {'sample_id': sample['sample_id'][0], 'scene_id': sample['scene_id'][0], 'valid_pixels': count,
                   'total_pixels': valid.numel(), 'absolute_error_sum': float(error.double().sum()),
                   **{'bad' + str(k) + '_count': int((error > k).sum()) for k in (1, 3, 5)}}
            frames.append(row)
            scene = scenes.setdefault(row['scene_id'], {'valid_pixels': 0, 'total_pixels': 0, 'absolute_error_sum': 0.,
                                                         'bad1_count': 0, 'bad3_count': 0, 'bad5_count': 0, 'frames': 0})
            for key in ('valid_pixels', 'total_pixels', 'absolute_error_sum', 'bad1_count', 'bad3_count', 'bad5_count'):
                scene[key] += row[key]
            scene['frames'] += 1
    if not scenes:
        raise ValueError('No validation scenes')
    for scene in scenes.values():
        scene['epe'] = scene['absolute_error_sum'] / scene['valid_pixels']
        scene['coverage'] = scene['valid_pixels'] / scene['total_pixels']
        for k in (1, 3, 5):
            scene['bad' + str(k)] = scene['bad' + str(k) + '_count'] / scene['valid_pixels']
    count = sum(s['valid_pixels'] for s in scenes.values())
    return {'mean_scene_epe': sum(s['epe'] for s in scenes.values()) / len(scenes),
            'pooled_epe': sum(s['absolute_error_sum'] for s in scenes.values()) / count,
            'coverage': count / sum(s['total_pixels'] for s in scenes.values()),
            'bad_pixel_units': 'fraction', 'aggregation': 'pooled_valid_pixels_within_scene_then_equal_scene_mean',
            'scene_metrics': scenes, 'frame_metrics': frames}
