"""Registered engineering functionality diagnostics in one Slurm RTX allocation.

No jobs are submitted. Priors fit raw Train GT on CPU and open no observations
or model. Checkpoint mode opens Val observations only. Test is never accessed.
Use --validate-config for metadata-only local configuration validation.
"""
import argparse
import contextlib
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import socket
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.study_data import safe_path, validate_paired_manifest
from core.study_diagnostics import (aggregate_scene_metrics, disparity_metrics,
    fit_train_priors, fixed_val_right_shuffle, paired_scene_bootstrap,
    require_engineering_manifest, right_view_conditions, shift_response,
    shifted_disparity_support)


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def validate_config(config, *, require_registered=True):
    if (config.get('schema') != 'stereo-study-functionality-v1'
            or config.get('purpose') != 'engineering' or config.get('mode') not in ('priors', 'checkpoint')):
        raise ValueError('Expected registered engineering functionality configuration')
    if require_registered and config.get('registered') is not True:
        raise ValueError('Execution requires registered=true')
    cap = 120 if config['mode'] == 'priors' else 600
    if type(config.get('max_seconds')) is not int or not 10 <= config['max_seconds'] <= cap:
        raise ValueError('Mode time cap exceeded')
    if type(config.get('seed')) is not int or not config.get('expected_hostname'):
        raise ValueError('Explicit seed and cluster hostname required')
    data = config['data']
    for key in ('root_env', 'manifest_env', 'audit_receipt_env'):
        if not isinstance(data.get(key), str) or not data[key]:
            raise ValueError('Configured environment variable required: ' + key)
    for key in ('manifest_sha256', 'audit_receipt_sha256'):
        if len(data.get(key, '')) != 64 or any(c not in '0123456789abcdef' for c in data[key]):
            raise ValueError('Exact immutable identity required: ' + key)
    for key, count in (('train_samples', 320), ('val_samples', 20), ('train_scenes', 40), ('val_scenes', 10)):
        if data.get(key) != count:
            raise ValueError('This bounded pilot evaluates the full registered 320/20 cohort')
    if config['mode'] == 'checkpoint':
        ckpt = config['checkpoint']
        if (ckpt.get('selection') not in ('best-val.pt', 'latest.pt')
                or type(config.get('shift_pixels')) is not int or config['shift_pixels'] <= 0):
            raise ValueError('Explicit frozen best-val/latest checkpoint and positive integer shift required')
        for name in ('path', 'metadata', 'training_config', 'training_source_registry'):
            if not ckpt.get(name + '_env') or len(ckpt.get(name + '_sha256', '')) != 64:
                raise ValueError('Exact checkpoint/source/config binding required: ' + name)
        if (not ckpt.get('selection_validation_log_env')
                or type(ckpt.get('selection_validation_log_bytes')) is not int
                or ckpt['selection_validation_log_bytes'] <= 0
                or not isinstance(ckpt.get('selection_validation_log_sha256'), str)
                or len(ckpt['selection_validation_log_sha256']) != 64
                or any(c not in '0123456789abcdef' for c in ckpt['selection_validation_log_sha256'])):
            raise ValueError('Pin the complete final validation log environment, bytes and SHA256')
        if ckpt.get('seed') != config['seed'] or type(ckpt.get('update')) is not int or ckpt['update'] <= 0:
            raise ValueError('Explicit trained checkpoint seed/update required')
    elif 'checkpoint' in config:
        raise ValueError('Prior mode must not specify or load a checkpoint')
    if 'feature_information' in config:
        from feature_information import probe_contract
        if config['mode']!='checkpoint' or config['feature_information']!=probe_contract():
            raise ValueError('Feature probes require the complete fixed descriptive-only checkpoint contract')
    return True


def verify_source(path, config_path):
    registry = json.loads(path.read_text())
    for name, expected in registry['source_files'].items():
        if file_sha(safe_path(path.parent, name)) != expected:
            raise RuntimeError('Frozen source changed: ' + name)
    identity = hashlib.sha256(json.dumps(registry['source_files'], sort_keys=True).encode()).hexdigest()
    if identity != registry['source_identity_sha256']:
        raise ValueError('Frozen source identity changed')
    required_members=[Path(__file__),config_path]
    if 'feature_information' in json.loads(config_path.read_text()):
        required_members.append(Path(__file__).with_name('feature_information.py'))
    for required in required_members:
        member = str(required.resolve().relative_to(path.parent.resolve()))
        if member not in registry['source_files']:
            raise ValueError('Diagnostic script/config absent from frozen source registry')
    return registry


def load_bound_data(config):
    data = config['data']
    root = Path(os.environ[data['root_env']]).resolve()
    path = Path(os.environ[data['manifest_env']]).resolve()
    audit_path = Path(os.environ[data['audit_receipt_env']]).resolve()
    if path != root/'manifest.json' or file_sha(path) != data['manifest_sha256']:
        raise ValueError('Immutable data manifest binding changed')
    manifest = json.loads(path.read_text())
    require_engineering_manifest(manifest)
    validate_paired_manifest(manifest, allow_engineering_pilot=True)
    if file_sha(audit_path) != data['audit_receipt_sha256']:
        raise ValueError('Full asset audit identity changed')
    audit = json.loads(audit_path.read_text())
    if (audit.get('status') != 'PASS_ALL_REGISTERED_ASSETS_TRACES_AND_SOURCE_IDENTITIES'
            or audit.get('mode') != 'all_registered_assets' or audit.get('test_access') is not False
            or audit.get('manifest_sha256') != data['manifest_sha256']
            or audit.get('samples_checked') != data['train_samples']+data['val_samples']
            or audit.get('source_PFM_equal_samples') != data['train_samples']+data['val_samples']
            or audit.get('source_payloads_hashed') != audit.get('source_files_registered')
            or not audit.get('source_payloads_hashed') or audit.get('T') != 50):
        raise ValueError('Complete registered asset/source audit PASS required')
    for split in ('train', 'val'):
        expected = dict(samples=data[split+'_samples'], scenes=data[split+'_scenes'])
        actual = dict(samples=len(manifest[split]), scenes=len({r['scene_id'] for r in manifest[split]}))
        if actual != expected or audit.get('counts', {}).get(split) != expected:
            raise ValueError('Cohort count/scene binding changed')
    if audit.get('configuration_sha256') != manifest['configuration_sha256'] or audit.get('shape') != manifest['output_shape']:
        raise ValueError('Audited generation geometry/configuration changed')
    return root, path, manifest, audit


def gt_samples(root, manifest, split):
    """Open only disparity NPY, never sensor arrays or source RGB/GT."""
    import numpy as np
    import torch
    if split not in ('train', 'val'):
        raise ValueError('Only Train/Val GT may be loaded')
    for row in manifest[split]:
        raw = np.load(safe_path(root, row['disparity']), allow_pickle=False)
        if raw.dtype != np.float32 or raw.shape != tuple(manifest['output_shape']):
            raise ValueError('Audited raw disparity shape/dtype changed')
        valid = np.isfinite(raw) & (raw > 0)
        yield dict(target=torch.from_numpy(np.where(valid, raw, 0)[None].copy()),
                   valid=torch.from_numpy(valid[None].copy()), sample_id=row['sample_id'],
                   scene_id=row['scene_id'], metadata={**row, 'crop': None})


def selected_cases(manifest):
    identifiers = sorted(row['sample_id'] for row in manifest['val'])
    return [identifiers[0], identifiers[len(identifiers)//2], identifiers[-1]]


def scene_equal_summary(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row['method'], row['condition']), []).append(row)
    return [dict(method=method, condition=condition, scenes=len(values),
                 **{'mean_scene_'+key: sum(row[key] for row in values)/len(values)
                    if all(row[key] is not None for row in values) else None
                    for key in ('epe', 'bad1', 'bad3', 'bad5', 'coverage')})
            for (method, condition), values in sorted(groups.items())]


def metric_row(sample, prediction, method, condition, target=None, valid=None, **extra):
    return dict(method=method, condition=condition, sample_id=sample['sample_id'], scene_id=sample['scene_id'],
        **disparity_metrics(prediction, sample['target'] if target is None else target,
                            sample['valid'] if valid is None else valid), **extra)


def save_case(output, index, sample, predictions):
    import numpy as np
    values = {key: value.detach().cpu().numpy() for key, value in predictions.items()}
    values.update(target=sample['target'].cpu().numpy(), valid=sample['valid'].cpu().numpy())
    name = 'selected-case-' + str(index) + '.npz'
    np.savez_compressed(output/name, **values)
    return dict(sample_id=sample['sample_id'], scene_id=sample['scene_id'], artifact=name,
                sha256=file_sha(output/name), prediction_keys=list(predictions),
                source_frame=sample['metadata'].get('source_frame'),
                disparity_asset=sample['metadata']['disparity'])


def run_priors(config, root, manifest, output, report):
    import torch
    # Deliberately no PairedStereoDataset/model construction or Train observations.
    priors = fit_train_priors(gt_samples(root, manifest, 'train'), manifest=manifest)
    artifact = output/'train-disparity-priors.pt'
    torch.save(dict(schema=1, constant_median=priors.constant_median, pixel_median=priors.pixel_median,
                    metadata=priors.metadata, binding=report['binding']), artifact)
    report.update(prior_fit=priors.metadata, constant_median=priors.constant_median,
                  prior_artifact=dict(path=artifact.name, sha256=file_sha(artifact)),
                  fit_device='cpu', model_loaded=False, train_observations_opened=False,
                  val_observations_opened=False, neural_training_seeds=0)
    rows, cases = [], []
    report.update(frame_metrics=rows, selected_cases=cases)
    wanted = report['preselected_val_case_ids']
    for sample in gt_samples(root, manifest, 'val'):
        predictions = {}
        for kind in ('constant', 'pixel'):
            pred = priors.predict(kind)
            predictions[kind] = pred
            rows.append(metric_row(sample, pred, 'train_'+kind+'_median', 'correct', reference_target='original'))
        predictions['zero'] = torch.zeros_like(sample['target'])
        rows.append(metric_row(sample, predictions['zero'], 'zero_disparity', 'correct', reference_target='original'))
        if sample['sample_id'] in wanted:
            cases.append(save_case(output, wanted.index(sample['sample_id']), sample, predictions))
    scenes = aggregate_scene_metrics(rows)
    report.update(frame_metrics=rows, scene_metrics=scenes, scene_equal_summary=scene_equal_summary(scenes), selected_cases=cases,
                  conditions_evaluated=['correct'], prior_semantics='deterministic_Train_GT_only_no_binocular_input')


def verify_checkpoint_validation_selection(path, prefix_receipt, state, training_config, selection, final_receipt):
    """Bind checkpoint prefix AND complete final log; Val0 never competes.

    Strict-less-than training selection keeps the earliest checkpoint on ties.
    A best checkpoint may precede the endpoint, but must still win over every
    final registered trained evaluation. Latest is an explicitly separate1000
    update diagnostic, carrying the same verified final best-selection history.
    """
    horizon = training_config['total_updates']
    schedule = training_config['training'].get('evaluation_updates')
    if (horizon != 1000 or schedule != [250, 500, 1000]
            or selection not in ('best-val.pt', 'latest.pt')):
        raise ValueError('Require the frozen1000-update pilot schedule250/500/1000')
    raw = Path(path).read_bytes()
    for label, receipt in [('checkpoint prefix', prefix_receipt), ('final full log', final_receipt)]:
        count, sha = receipt.get('bytes'), receipt.get('sha256')
        if type(count) is not int or count <= 0 or not isinstance(sha, str) or len(sha) != 64:
            raise ValueError('Invalid validation-log receipt: '+label)
        piece = raw[:count]
        if (len(piece) != count or hashlib.sha256(piece).hexdigest() != sha
                or not piece.endswith(b'\n') or (label == 'final full log' and len(raw) != count)):
            raise ValueError('Validation-log bytes/SHA or complete-line binding changed: '+label)
    if prefix_receipt['bytes'] > final_receipt['bytes']:
        raise ValueError('Final validation log cannot precede checkpoint prefix')
    initial = [0] if training_config['training'].get('evaluate_initial', False) else []
    def parse(piece, expected):
        rows = [json.loads(line) for line in piece.splitlines() if line.strip()]
        updates = [row.get('update') for row in rows]
        if any(type(value) is not int for value in updates) or updates != expected:
            raise ValueError('Missing, duplicate, extra or reordered registered Val evaluation')
        for row in rows:
            value = row.get('mean_scene_epe')
            if (type(value) not in (int, float) or not math.isfinite(value) or value < 0
                    or (row['update'] > 0 and row.get('untrained_reference', False))):
                raise ValueError('Every registered trained evaluation needs finite scene-mean EPE')
        return [row for row in rows if row['update'] > 0]
    if type(state.get('update')) is not int or state['update'] not in schedule:
        raise ValueError('Checkpoint must be at a registered trained evaluation')
    prefix = parse(raw[:prefix_receipt['bytes']], initial+[x for x in schedule if x <= state['update']])
    final = parse(raw, initial+schedule)
    prefix_best = min(prefix, key=lambda row: row['mean_scene_epe'])
    final_best = min(final, key=lambda row: row['mean_scene_epe'])
    if (state.get('best_update') != prefix_best['update']
            or state.get('best_val_epe') != prefix_best['mean_scene_epe']
            or state.get('best_update') != final_best['update']
            or state.get('best_val_epe') != final_best['mean_scene_epe']):
        raise ValueError('Checkpoint best-selection history disagrees with prefix or complete final log')
    if selection == 'best-val.pt' and state['update'] != final_best['update']:
        raise ValueError('Best-Val checkpoint does not win over the complete final log')
    if selection == 'latest.pt' and state['update'] != horizon:
        raise ValueError('Latest diagnostic is permitted only at the complete1000-update endpoint')
    return dict(selection=selection, trained_evaluation_updates=schedule, initial_update_excluded_from_selection=True,
                rule='minimum_trained_scene_mean_EPE;earliest_registered_update_on_ties',
                checkpoint_update=state['update'], best_update=final_best['update'], best_val_epe=final_best['mean_scene_epe'],
                checkpoint_log_prefix=dict(prefix_receipt), final_full_validation_log=dict(final_receipt),
                trained_evaluations=[dict(update=row['update'], mean_scene_epe=row['mean_scene_epe']) for row in final])


def load_checkpoint(config, manifest, device):
    import torch
    ckpt = config['checkpoint']
    paths = {name: Path(os.environ[ckpt[name+'_env']]).resolve()
             for name in ('path', 'metadata', 'training_config', 'training_source_registry')}
    for name, path in paths.items():
        if file_sha(path) != ckpt[name+'_sha256']:
            raise ValueError('Frozen checkpoint member changed: ' + name)
    if paths['path'].name != ckpt['selection']:
        raise ValueError('Checkpoint filename differs from explicit selection')
    train_config = json.loads(paths['training_config'].read_text())
    metadata = json.loads(paths['metadata'].read_text())
    registry = json.loads(paths['training_source_registry'].read_text())
    binding = metadata['binding']
    if (train_config != metadata['config'] or train_config['scope'] != 'engineering'
            or train_config.get('test_access') != 'sealed' or train_config.get('pretraining') is not False
            or train_config['seed'] != ckpt['seed'] or train_config['data']['manifest_sha256'] != config['data']['manifest_sha256']
            or binding['config_sha256'] != ckpt['training_config_sha256'] or binding['config_identity'] != digest(train_config)
            or binding['source_registry_sha256'] != ckpt['training_source_registry_sha256']
            or binding['source_identity_sha256'] != registry['source_identity_sha256'] or binding['git_commit'] != registry['git_commit']
            or binding['manifest_sha256'] != config['data']['manifest_sha256']
            or binding['train_sample_ids_sha256'] != digest([r['sample_id'] for r in manifest['train']])
            or binding['val_sample_ids_sha256'] != digest([r['sample_id'] for r in manifest['val']])
            or binding['training_samples'] != 320 or binding['validation_samples'] != 20):
        raise ValueError('Training config/source/manifest/seed/cohort binding mismatch')
    # Current imported architecture/baseline dependencies must equal the training
    # capsule, even if this diagnostic script was added in a later source freeze.
    for name, expected in registry['source_files'].items():
        if '/core/' in '/'+name:
            if file_sha(safe_path(Path(__file__).resolve().parents[5], name)) != expected:
                raise ValueError('Imported core differs from trained source: ' + name)
    payload = torch.load(paths['path'], map_location='cpu', weights_only=False)
    state = payload['state']
    if (payload.get('schema') != 1 or payload['binding'] != binding or state['update'] != ckpt['update']
            or state['presentations'] != state['update']*binding['effective_batch']
            or (ckpt['selection'] == 'best-val.pt' and state['best_update'] != state['update'])):
        raise ValueError('Explicit trained checkpoint state/binding mismatch')
    selection = verify_checkpoint_validation_selection(
        Path(os.environ[ckpt['selection_validation_log_env']]), payload['logs']['validation'], state, train_config,
        ckpt['selection'], dict(bytes=ckpt['selection_validation_log_bytes'], sha256=ckpt['selection_validation_log_sha256']))
    values = dict(train_config['model']['config'])
    if train_config['model']['family'] == 'study':
        from core.study_models import build_model
        model = build_model(values)
    elif train_config['model']['family'] == 'baseline':
        from core.study_baselines import StudyBaseline
        model = StudyBaseline(capsule=os.environ[train_config['model']['capsule_env']], seed=ckpt['seed'], **values)
    else:
        raise ValueError('Unknown exact model family')
    model.load_state_dict(payload['model'], strict=True)
    provenance = model.provenance() if hasattr(model, 'provenance') else None
    if provenance != metadata.get('baseline_provenance'):
        raise ValueError('Baseline environment/capsule provenance differs from training')
    return model.to(device).eval(), train_config, dict(binding=binding, state=state, model=train_config['model'],
        baseline_provenance=provenance, selection=ckpt['selection'], validation_selection=selection,
        checkpoint_sha256=ckpt['path_sha256'])


def run_checkpoint(config, root, manifest_path, manifest, output, report, device):
    import torch
    from core.study_data import PairedStereoDataset
    model, training, provenance = load_checkpoint(config, manifest, device)
    feature_enabled='feature_information' in config
    if feature_enabled:
        from feature_information import TemporalFeatureProbe,output_information
        if training['model']['family']!='study':
            raise ValueError('Legacy baseline feature probe explicitly unsupported; do not rerun its existing diagnostics')
    val = PairedStereoDataset(root, manifest_path, training['data']['modality'], 'val', purpose='engineering')
    if config['shift_pixels'] != int(8*manifest['output_scale']):
        raise ValueError('Primary shift must equal eight native pixels: quarter2/half4')
    indices = {row['sample_id']: i for i, row in enumerate(val.rows)}
    pairing = fixed_val_right_shuffle(manifest, seed=config['seed'])
    wanted = report['preselected_val_case_ids']
    rows, shift_rows, cases = [], [], []
    report.update(frame_metrics=rows, shifted_geometry_frame_metrics=shift_rows, selected_cases=cases)
    report.update(checkpoint=provenance, model_loaded=True, train_observations_opened=False,
                  val_observations_opened=True, neural_training_seeds=1, fixed_right_donors=pairing)
    if feature_enabled:
        report['feature_information']=dict(contract=config['feature_information'],
            interpretation='descriptive_only_not_stop_or_checkpoint_selection_criterion;no_architecture_conclusion_from_statistics_alone',
            preselected_case_ids=list(wanted),temporal_feature_cases=[],correct_forward_output_summaries=[],
            legacy_baseline='unsupported')
    with torch.inference_mode():
        for index in range(len(val)):
            sample = val[index]
            donor = val[indices[pairing[sample['sample_id']]]]
            left = sample['left'][None].to(device)
            right = sample['right'][None].to(device)
            other = donor['right'][None].to(device)
            predictions = {}
            for condition, changed in right_view_conditions(right, other, config['shift_pixels']).items():
                # Each existing builder resets state per forward; retain all50 steps.
                probe=(TemporalFeatureProbe(model,left.shape[-2:])
                    if feature_enabled and condition=='correct' and sample['sample_id'] in wanted else None)
                with probe if probe is not None else contextlib.nullcontext():
                    model_output=model(left,changed)
                pred = model_output['disparity']
                if pred.shape != (1, *sample['target'].shape) or not torch.isfinite(pred).all():
                    raise ValueError('Model must return finite disparity of exact unaugmented shape')
                if feature_enabled and condition=='correct':
                    identity=dict(sample_id=sample['sample_id'],scene_id=sample['scene_id'])
                    report['feature_information']['correct_forward_output_summaries'].append(
                        dict(**identity,**output_information(model_output,left.shape[-2:])))
                    if probe is not None:
                        report['feature_information']['temporal_feature_cases'].append(dict(**identity,**probe.serialize()))
                del model_output,probe
                predictions[condition] = pred[0].cpu()
                if condition != 'shifted':
                    rows.append(metric_row(sample, predictions[condition], 'checkpoint', condition,
                        reference_target='original', disparity_convention='d=x_left_minus_x_right',
                        nonpositive_prediction_pixels=int(((predictions[condition] <= 0)&sample['valid']).sum())))
            changed_gt, common = shifted_disparity_support(sample['target'], sample['valid'], config['shift_pixels'])
            response = shift_response(predictions['correct'], predictions['shifted'], config['shift_pixels'], common)
            for condition, pred, target in [('correct_common', predictions['correct'], sample['target']),
                                             ('shifted_geometry', predictions['shifted'], changed_gt)]:
                shift_rows.append(metric_row(sample, pred, 'checkpoint', condition, target, common,
                    reference_target='original' if condition == 'correct_common' else 'original_minus_shift',
                    shift_pixels=config['shift_pixels'], **response))
            if sample['sample_id'] in wanted:
                cases.append(save_case(output, wanted.index(sample['sample_id']), sample,
                    {**predictions, 'shifted_target': changed_gt, 'shift_common_valid': common}))
            del donor, left, right, other
    if feature_enabled:
        feature=report['feature_information']
        if (len(feature['temporal_feature_cases'])!=3
                or {row['sample_id'] for row in feature['temporal_feature_cases']}!=set(wanted)
                or len(feature['correct_forward_output_summaries'])!=len(val)):
            raise ValueError('Descriptive probe must cover exactly three fixed cases and every correct Val output')
        feature.update(temporal_feature_cases_count=3,correct_forward_output_summaries_count=len(val))
    scenes = aggregate_scene_metrics(rows)
    cis = {}
    correct = {r['scene_id']: r['epe'] for r in scenes if r['condition'] == 'correct'}
    for condition in ('shuffled', 'masked'):
        comparison = {r['scene_id']: r['epe'] for r in scenes if r['condition'] == condition}
        cis[condition+'_minus_correct_epe'] = paired_scene_bootstrap(comparison, correct, seed=config['seed'])
    shift_scenes = aggregate_scene_metrics(shift_rows)
    responses = [row['response_ratio'] for row in shift_scenes if row['condition'] == 'shifted_geometry']
    import numpy as np
    report.update(frame_metrics=rows, scene_metrics=scenes, scene_equal_summary=scene_equal_summary(scenes),
                  shifted_geometry_frame_metrics=shift_rows, shifted_geometry_scene_metrics=shift_scenes,
                  shifted_geometry_scene_equal_summary=scene_equal_summary(shift_scenes), selected_cases=cases,
                  shift_scene_response_summary=dict(scene_count=len(responses),
                    median_scene_mean_response_ratio=float(np.median(responses)) if all(v is not None for v in responses) else None,
                    positive_mean_response_scenes=sum(v > 0 for v in responses if v is not None),
                    majority_scenes_mean_positive=sum(v > 0 for v in responses if v is not None)>len(responses)/2,
                    interpretation='median_of_scene_mean_signed_pixel_responses;not_pooled_pixel_median'),
                  paired_scene_descriptive_ci=cis, state_reset='existing_model_forward_resets_each_presentation',
                  shift_geometry='right_content_plus_delta_zero_fill;d_new=d_original_minus_delta;GT_only_common_inframe_positive_support',
                  main_accuracy_conditions=['correct', 'shuffled', 'masked'], confidence_filter=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--source-registry', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--validate-config', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    validate_config(config, require_registered=not args.validate_config)
    if args.validate_config:
        print(json.dumps(dict(status='VALID_CONFIG_ONLY_NO_DATA_OR_MODEL_READ', config_sha256=file_sha(args.config))))
        return
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname() != config['expected_hostname']:
        raise RuntimeError('Diagnostics require the registered cluster Slurm allocation')
    if not args.source_registry or not args.output:
        raise ValueError('Frozen source registry and new output directory required')
    import numpy as np
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not any(n in torch.cuda.get_device_name(0) for n in ('4090', '5090')):
        raise RuntimeError('Exactly one allocated authorized RTX4090/5090 required')
    registry = verify_source(args.source_registry, args.config)
    root, manifest_path, manifest, audit = load_bound_data(config)
    output = args.output.resolve()
    if any(output.is_relative_to(path.resolve()) for path in (root, args.source_registry.parent)):
        raise ValueError('Artifacts must be outside immutable source/data')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = dict(schema=1, status='STARTED', mode=config['mode'], purpose='engineering', test_access=False,
        binding=dict(config_sha256=file_sha(args.config), manifest_sha256=config['data']['manifest_sha256'],
                     audit_receipt_sha256=config['data']['audit_receipt_sha256'], source_registry_sha256=file_sha(args.source_registry),
                     source_identity_sha256=registry['source_identity_sha256'], git_commit=registry['git_commit']),
        config=config, source_dirty_status=registry.get('dirty_status'), source_overlay_sha256=registry.get('overlay_sha256'),
        slurm_job_id=os.environ['SLURM_JOB_ID'], slurm_cluster=os.environ.get('SLURM_CLUSTER_NAME'),
        slurm_nodes=os.environ.get('SLURM_JOB_NODELIST'), gpu=torch.cuda.get_device_name(0), visible_cuda_devices=1,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), python=sys.version, torch=torch.__version__,
        numpy=np.__version__, cuda=torch.version.cuda, command=sys.argv,
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), max_seconds=config['max_seconds'],
        preselected_val_case_ids=selected_cases(manifest), selection_rule='lexicographic_Val_ids_first_middle_last_before_inference',
        fit_includes_val=False, test_selection=False, cpu_threads=1, gpu_compute=config['mode']=='checkpoint',
        data_audit_status=audit['status'], aggregation='pixels_within_scene_then_equal_scene;no_pixel_IID')
    def expired(number, frame):
        raise TimeoutError('Functionality deadline reached; incomplete is not PASS')
    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, config['max_seconds']-5)
    try:
        torch.set_num_threads(1)
        random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
        torch.cuda.manual_seed_all(config['seed'])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if config['mode'] == 'priors':
            run_priors(config, root, manifest, output, report)
        else:
            run_checkpoint(config, root, manifest_path, manifest, output, report, torch.device('cuda:0'))
        report['status'] = 'COMPLETED_REGISTERED_DIAGNOSTIC_NO_SCIENTIFIC_PASS_THRESHOLD'
    except TimeoutError as error:
        report.update(status='INCOMPLETE_TIME_CAP', error=repr(error))
    except Exception as error:
        report.update(status='FAILED_DIAGNOSTIC', error=repr(error))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, previous)
        report['elapsed_seconds'] = time.monotonic()-started
        (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'], report=str(output/'report.json'))))
    if not report['status'].startswith('COMPLETED_'):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
