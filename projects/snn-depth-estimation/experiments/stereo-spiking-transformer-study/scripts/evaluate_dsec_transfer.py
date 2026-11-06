"""Prospective fixed20-target, five-family DSEC Train development transfer.

Execution requires registered bindings and one RTX4090/5090 Slurm allocation.
No job submission, adaptation, synthetic observation/GT read, or Test access.
The example is unregistered and cannot execute. Native GT opens only after
all five synthetic checkpoint selection bindings have passed metadata checks.
"""
import argparse
import datetime
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import dsec_adapter as adapter
import evaluate_functionality as functionality
import benchmark_energy as energy

FAMILIES = ('A-event', 'B-event', 'continuous-A-event', 'continuous-B-event', 'ALIF+I-event')
SHA = re.compile(r'[0-9a-f]{64}')
file_sha = functionality.file_sha
digest = functionality.digest


def validate_config(config, *, require_registered=True):
    if (config.get('schema') != 'stereo-study-dsec-transfer-v1'
            or config.get('scope') != 'engineering' or config.get('test_access') != 'sealed'
            or config.get('driver') != 'dsec_transfer' or config.get('stage') != 'engineering_diagnostic'
            or config.get('protocol_status') != 'prospective_engineering_only'
            or config.get('allocated_gpu_hour_cap') != 24):
        raise ValueError('Expected prospective engineering DSEC Train-only registration')
    if require_registered and config.get('registered') is not True:
        raise ValueError('Example/unregistered configuration cannot execute')
    if (type(config.get('max_seconds')) is not int or not 30 <= config['max_seconds'] <= 600
            or config.get('scheduler_resources') != {'time': '00:10:00'}
            or not config.get('expected_hostname') or type(config.get('seed')) is not int
            or config.get('inference_variant') != 'trained_config_unchanged'
            or config.get('models') != list(FAMILIES)
            or config.get('scenes') != list(adapter.SCENES)
            or config.get('targets_per_scene') != 10 or config.get('predictions') != 100):
        raise ValueError('Fixed five-family20-target/full trained-config/600-second contract required')
    data = config['dsec']
    for key in ('source_root_env', 'prepared_root_env', 'prepared_manifest_env', 'preflight_manifest_env',
                'preparation_config_env', 'preparation_source_registry_env'):
        if not isinstance(data.get(key), str) or not data[key]:
            raise ValueError('Explicit DSEC deployment environment key required: '+key)
    for key in ('prepared_manifest_sha256', 'preflight_manifest_sha256', 'preparation_config_sha256',
                'preparation_source_registry_sha256', 'preparation_source_identity_sha256',
                'adapter_script_sha256', 'targets_sha256', 'source_files_sha256'):
        value = data.get(key, '')
        if not isinstance(value, str) or not SHA.fullmatch(value) or (require_registered and value == '0'*64):
            raise ValueError('Nonplaceholder immutable DSEC identity required: '+key)
    ids = data.get('sample_ids', [])
    if len(ids) != 20 or len(set(ids)) != 20 or any(
            sum(name.startswith(scene+'/') for name in ids) != 10 for scene in adapter.SCENES):
        raise ValueError('Exactly twenty fixed known-Train sample identities required')
    for name in ids:
        scene, _, frame = name.partition('/')
        if scene not in adapter.SCENES or not re.fullmatch(r'\d{6}', frame):
            raise ValueError('No unknown scene or Test identity permitted')
    if [case.get('case_id') for case in config.get('cases', [])] != list(FAMILIES):
        raise ValueError('All five event families required in fixed order; no subset selection')
    for case in config['cases']:
        ckpt = case['checkpoint']
        if ckpt.get('selection') != 'best-val.pt' or ckpt.get('update') not in (250, 500, 1000):
            raise ValueError('Only completed synthetic best-trained-Val selections permitted')
        bridge = dict(schema='stereo-study-functionality-v1', registered=config['registered'],
            purpose='engineering', mode='checkpoint', max_seconds=config['max_seconds'],
            seed=ckpt.get('seed'), expected_hostname=config['expected_hostname'], shift_pixels=2,
            data=config['synthetic_data'], checkpoint=ckpt)
        functionality.validate_config(bridge, require_registered=require_registered)
        for key in ('training_source_identity_sha256', 'training_initial_weights_sha256'):
            value = case.get(key, '')
            if not isinstance(value, str) or not SHA.fullmatch(value) or (require_registered and value == '0'*64):
                raise ValueError('Exact synthetic source/initialization binding required: '+key)
        if require_registered:
            for key, value in ckpt.items():
                if key.endswith('_sha256') and value == '0'*64:
                    raise ValueError('Placeholder checkpoint binding cannot execute')
    if require_registered and any(config['synthetic_data'][key] == '0'*64
                                  for key in ('manifest_sha256', 'audit_receipt_sha256')):
        raise ValueError('Placeholder synthetic metadata/audit identity cannot execute')
    return True


def verify_registry(path, required=()):
    registry = json.loads(path.read_text())
    if hashlib.sha256(json.dumps(registry['source_files'], sort_keys=True).encode()).hexdigest() != registry['source_identity_sha256']:
        raise ValueError('Frozen source map identity changed')
    for name, expected in registry['source_files'].items():
        if file_sha(adapter.safe(path.parent, name)) != expected:
            raise ValueError('Frozen source member changed: '+name)
    for member in required:
        name = str(Path(member).resolve().relative_to(path.parent.resolve()))
        if registry['source_files'].get(name) != file_sha(member):
            raise ValueError('Required executing source/config absent from frozen capsule')
    return registry


def validate_prepared_receipt(config, preflight, preparation_config, prepared):
    """Pure metadata gate; no source arrays or GT pixels are read."""
    data = config['dsec']
    adapter.validate_config(preparation_config, execution=True)
    if (preflight.get('status') != 'PREFLIGHT_READY_METADATA_ONLY'
            or preflight.get('test_access') is not False or preflight.get('gt_pixels_opened') is not False
            or preflight.get('event_conversion') is not False
            or preflight.get('contract') != adapter.contract(preparation_config)
            or preflight.get('contract_sha256') != digest(preflight['contract'])
            or digest(preflight.get('targets')) != data['targets_sha256']
            or digest(preflight.get('source_files')) != data['source_files_sha256']
            or [r['sample_id'] for r in preflight['targets']] != data['sample_ids']):
        raise ValueError('Reviewed preflight targets/source/contract identity changed')
    binding = prepared.get('binding', {})
    if (prepared.get('schema') != 'dsec-causal-prepared-v1'
            or prepared.get('status') != 'COMPLETED_PROSPECTIVE_ENGINEERING_PREPARATION'
            or prepared.get('protocol_status') != 'prospective_engineering_only'
            or prepared.get('test_access') is not False or prepared.get('gt_pixels_opened') is not False
            or prepared.get('model_inference') is not False
            or prepared.get('source_manifest_sha256') != data['preflight_manifest_sha256']
            or prepared.get('contract') != preflight['contract']
            or prepared.get('source_files') != preflight['source_files']
            or binding.get('config_sha256') != data['preparation_config_sha256']
            or binding.get('script_sha256') != data['adapter_script_sha256']
            or binding.get('source_registry_sha256') != data['preparation_source_registry_sha256']
            or binding.get('source_identity_sha256') != data['preparation_source_identity_sha256']
            or preparation_config.get('adapter_script_sha256') != data['adapter_script_sha256']
            or preparation_config.get('preflight_manifest_sha256') != data['preflight_manifest_sha256']):
        raise ValueError('Completed preparation and exact executing source/config binding required')
    qa = prepared.get('conversion_qa', {})
    if (qa.get('prepared_targets') != 20 or qa.get('readback_targets') != 20
            or qa.get('ready_for_inference') is not True
            or qa.get('all_causal_lower_bound_neighbors_verified') is not True
            or qa.get('all_stored_counts_readback_exact') is not True
            or set(qa.get('source_fault_counts', {})) != {'left', 'right'}
            or any(qa['source_fault_counts'][view] != dict(time_outside=0,bad_raw_coordinate=0,bad_polarity=0)
                   for view in ('left', 'right'))):
        raise ValueError('Full20-target conversion/readback/source-fault QA PASS required')
    rows = prepared.get('targets', [])
    if len(rows) != 20 or [r['sample_id'] for r in rows] != data['sample_ids']:
        raise ValueError('No missing/reordered/replacement prepared targets permitted')
    for original, row in zip(preflight['targets'], rows):
        if any(row.get(key) != value for key, value in original.items()):
            raise ValueError('Prepared identity, physical time or native GT mapping changed')
        if row.get('original_split') != 'train' or row.get('scene_id') not in adapter.SCENES:
            raise ValueError('Official Train-only targets required')
        readback = row.get('sensor_readback_qa', {})
        if (readback.get('status') != 'PASS_EXACT_COUNTS_AND_PER_BIN_READBACK'
                or readback.get('views') != ['left', 'right']
                or readback.get('float32_count_cast_exact') is not True
                or readback.get('ground_truth_opened') is not False):
            raise ValueError('Every prepared asset must pass evaluator-loader readback')
        if not SHA.fullmatch(row.get('sensor_sha256', '')):
            raise ValueError('Prepared sensor asset SHA required')
        for view in ('left', 'right'):
            stats = row['count_statistics'][view]
            counters=('input_events','time_outside','bad_raw_coordinate','bad_polarity',
                'nonfinite_rectification','outside_rectified_support','accepted_events','on_events','off_events')
            if (any(type(stats.get(key)) is not int or stats[key]<0 for key in counters)
                    or stats['input_events'] != sum(stats[key] for key in counters[1:7])):
                raise ValueError('Disjoint input/accepted/discard event counters must conserve counts')
            bins = stats['per_bin_ON_OFF_counts']
            if len(bins) != 50 or any(len(pair) != 2 or any(type(v) is not int or v < 0 for v in pair) for pair in bins):
                raise ValueError('Full50-bin nonnegative integer ON/OFF totals required')
            if ([sum(pair[c] for pair in bins) for c in (0, 1)] != [stats['on_events'], stats['off_events']]
                    or stats['on_events']+stats['off_events'] != stats['accepted_events']
                    or any(stats[key] != 0 for key in ('time_outside', 'bad_raw_coordinate', 'bad_polarity'))):
                raise ValueError('Per-bin/view event conservation or source QA failed')
            boundaries = stats['causal_boundary_neighbors']
            if len(boundaries) != 2 or [b['boundary_global_us'] for b in boundaries] != [row['target_us']-50000,row['target_us']]:
                raise ValueError('Both physical causal boundary receipts required')
            for b in boundaries:
                if (b.get('lower_bound_verified') is not True
                        or (b['previous_global_us'] is not None and b['previous_global_us'] >= b['boundary_global_us'])
                        or (b['next_global_us'] is not None and b['next_global_us'] < b['boundary_global_us'])):
                    raise ValueError('Actual timestamp neighbor lower-bound QA failed')
    return True


def load_dsec_metadata(config):
    data = config['dsec']
    paths = {key: Path(os.environ[data[key+'_env']]).resolve() for key in
             ('source_root','prepared_root','prepared_manifest','preflight_manifest','preparation_config','preparation_source_registry')}
    for name in ('prepared_manifest','preflight_manifest','preparation_config','preparation_source_registry'):
        if file_sha(paths[name]) != data[name+'_sha256']:
            raise ValueError('Bound DSEC metadata file changed: '+name)
    if paths['prepared_manifest'] != paths['prepared_root']/'manifest.json':
        raise ValueError('Prepared manifest must belong to the immutable prepared root')
    preflight=json.loads(paths['preflight_manifest'].read_text())
    preparation=json.loads(paths['preparation_config'].read_text())
    prepared=json.loads(paths['prepared_manifest'].read_text())
    validate_prepared_receipt(config,preflight,preparation,prepared)
    registry=verify_registry(paths['preparation_source_registry'])
    if (registry['source_identity_sha256'] != data['preparation_source_identity_sha256']
            or prepared['binding'].get('git_commit') != registry['git_commit']):
        raise ValueError('Preparation capsule identity changed')
    member=str(paths['preparation_config'].relative_to(paths['preparation_source_registry'].parent))
    adapter_member=str(Path(__file__).resolve().relative_to(Path(__file__).resolve().parents[5])).replace('evaluate_dsec_transfer.py','dsec_adapter.py')
    if (registry['source_files'].get(member) != data['preparation_config_sha256']
            or registry['source_files'].get(adapter_member) != data['adapter_script_sha256']
            or file_sha(adapter.__file__) != data['adapter_script_sha256']):
        raise ValueError('Frozen preparation config/adapter member mismatch')
    adapter.verify_preflight(preflight,preparation,paths['source_root'])
    # Hash every prepared observation before opening GT; no sensor decode yet.
    for row in prepared['targets']:
        if file_sha(adapter.safe(paths['prepared_root'],row['sensor_asset'])) != row['sensor_sha256']:
            raise ValueError('Prepared sensor changed')
    return paths,prepared


def validate_training_family(case_id, training):
    model=training['model'];values=model['config']
    if (training['data'].get('modality') != 'event' or training.get('total_updates') != 1000
            or training['training'].get('evaluation_updates') != [250,500,1000]
            or values.get('input_channels') != 2):
        raise ValueError('Fixed synthetic event modality and complete schedule required')
    if case_id=='ALIF+I-event':
        if model['family']!='baseline' or values.get('architecture')!='original_I' or values.get('eval_refinements')!=32:
            raise ValueError('Retain exact ALIF+I wrapper/ONOFF100-channel/sign/inferenceK adaptation')
    elif (model['family']!='study' or values.get('architecture') != case_id.removeprefix('continuous-').split('-')[0]
            or values.get('spiking') is not (not case_id.startswith('continuous-'))
            or values.get('widths') != [32,64,96,128] or values.get('time_steps')!=50
            or values.get('max_disparity')!=64):
        raise ValueError('Retain registered full-width/T50 study architecture and activation')
    return True


def verify_executing_core(registry):
    """Same member equality as load_checkpoint, before any real GT decoding."""
    checked=[]
    for name,expected in registry['source_files'].items():
        if '/core/' in '/'+name:
            if file_sha(adapter.safe(Path(__file__).resolve().parents[5],name))!=expected:
                raise ValueError('Imported core differs from trained source: '+name)
            checked.append(name)
    return dict(members_checked=len(checked),member_names=sorted(checked))


def precheck_checkpoint(case, config, manifest):
    """Verify every final synthetic selection before any DSEC GT is opened."""
    import torch
    ckpt=case['checkpoint'];paths={key:Path(os.environ[ckpt[key+'_env']]).resolve()
        for key in ('path','metadata','training_config','training_source_registry')}
    for key,path in paths.items():
        if file_sha(path)!=ckpt[key+'_sha256']:raise ValueError('Checkpoint metadata SHA changed: '+key)
    if paths['path'].name!='best-val.pt':raise ValueError('Only explicit best-val.pt permitted')
    registry=verify_registry(paths['training_source_registry'])
    if registry['source_identity_sha256']!=case['training_source_identity_sha256']:
        raise ValueError('Training source identity changed')
    core_check=verify_executing_core(registry)
    member=str(paths['training_config'].relative_to(paths['training_source_registry'].parent))
    if registry['source_files'].get(member)!=ckpt['training_config_sha256']:
        raise ValueError('Training config absent from its source capsule')
    training=json.loads(paths['training_config'].read_text());metadata=json.loads(paths['metadata'].read_text())
    validate_training_family(case['case_id'],training)
    payload=torch.load(paths['path'],map_location='cpu',weights_only=False,mmap=True)
    binding=metadata['binding'];state=payload['state']
    expected=dict(config_sha256=ckpt['training_config_sha256'],config_identity=digest(training),
        source_identity_sha256=case['training_source_identity_sha256'],source_registry_sha256=ckpt['training_source_registry_sha256'],
        git_commit=registry['git_commit'],manifest_sha256=config['synthetic_data']['manifest_sha256'],
        train_sample_ids_sha256=digest([r['sample_id'] for r in manifest['train']]),
        val_sample_ids_sha256=digest([r['sample_id'] for r in manifest['val']]),training_samples=320,validation_samples=20,
        total_updates=1000,effective_batch=8)
    if (training!=metadata['config'] or payload.get('schema')!=1 or payload['binding']!=binding
            or any(binding.get(k)!=v for k,v in expected.items()) or training['seed']!=ckpt['seed']
            or training.get('scope')!='engineering' or training.get('pretraining') is not False
            or training.get('test_access')!='sealed'
            or metadata.get('fresh_seed_model_sha256')!=case['training_initial_weights_sha256']
            or state['update']!=ckpt['update'] or state['presentations']!=state['update']*8):
        raise ValueError('Synthetic checkpoint/config/seed/initialization/cohort binding mismatch')
    selection=functionality.verify_checkpoint_validation_selection(Path(os.environ[ckpt['selection_validation_log_env']]),
        payload['logs']['validation'],state,training,'best-val.pt',
        dict(bytes=ckpt['selection_validation_log_bytes'],sha256=ckpt['selection_validation_log_sha256']))
    del payload
    return dict(case_id=case['case_id'],checkpoint_sha256=ckpt['path_sha256'],selection=selection,
        training_config_sha256=ckpt['training_config_sha256'],training_source_identity_sha256=case['training_source_identity_sha256'],
        executing_core_check=core_check)


def native_metrics(prediction, target):
    """Every finite positive native GT label, including nonpositive predictions."""
    import numpy as np
    prediction=np.asarray(prediction);target=np.asarray(target)
    if prediction.shape!=target.shape or not np.isfinite(prediction).all():
        raise ValueError('Complete finite native prediction required; no confidence filtering')
    valid=np.isfinite(target)&(target>0);n=int(valid.sum())
    if not n:raise ValueError('Frozen target has no positive native GT; do not substitute another target')
    error=np.abs(prediction[valid].astype(np.float64)-target[valid].astype(np.float64))
    return dict(valid_pixels=n,total_pixels=int(valid.size),absolute_error_sum=float(error.sum()),
        squared_error_sum=float(np.square(error).sum()),bad1_pixels=int((error>1).sum()),
        bad2_pixels=int((error>2).sum()),nonpositive_prediction_pixels=int((prediction[valid]<=0).sum()),
        epe=float(error.mean()),mae=float(error.mean()),rmse=float(np.sqrt(np.square(error).mean())),
        one_pe=float((error>1).mean()),two_pe=float((error>2).mean()),coverage=n/valid.size)


def aggregate(rows):
    """Pool trusted pixels within sequence, then give the two sequences equal weight."""
    result=[]
    for method in sorted({r['method'] for r in rows}):
        for scene in adapter.SCENES:
            values=[r for r in rows if r['method']==method and r['scene_id']==scene]
            if len(values)!=10:raise ValueError('Complete ten targets per sequence/method required')
            sums={key:sum(row[key] for row in values) for key in ('valid_pixels','total_pixels',
                'absolute_error_sum','squared_error_sum','bad1_pixels','bad2_pixels','nonpositive_prediction_pixels')}
            n=sums['valid_pixels']
            result.append(dict(method=method,scene_id=scene,targets=10,**sums,epe=sums['absolute_error_sum']/n,
                mae=sums['absolute_error_sum']/n,rmse=math.sqrt(sums['squared_error_sum']/n),
                one_pe=sums['bad1_pixels']/n,two_pe=sums['bad2_pixels']/n,coverage=n/sums['total_pixels']))
    equal=[]
    for method in sorted({r['method'] for r in result}):
        values=[r for r in result if r['method']==method]
        equal.append(dict(method=method,sequences=2,**{key:sum(v[key] for v in values)/2
            for key in ('epe','mae','rmse','one_pe','two_pe','coverage')}))
    return result,equal


def run_panel(config, paths, prepared, synthetic_manifest, output, report, device):
    import numpy as np
    import torch
    # All five metadata selections must pass before native GT decoding.
    report['checkpoint_prechecks']=[precheck_checkpoint(case,config,synthetic_manifest) for case in config['cases']]
    targets={}
    for row in prepared['targets']:
        report['gt_targets_attempted']=report.get('gt_targets_attempted',0)+1
        report.setdefault('gt_target_attempt_ids',[]).append(row['sample_id'])
        target,valid=adapter.load_native_gt(paths['source_root'],row)
        if not np.array_equal(valid,np.isfinite(target)&(target>0)):
            raise ValueError('Native GT mask must retain every finite positive label')
        targets[row['sample_id']]=target
        report['gt_targets_opened']+=1
        report['frame_metrics'].append(dict(method='zero_disparity',sample_id=row['sample_id'],scene_id=row['scene_id'],
            **native_metrics(np.zeros_like(target),target)))
    for case in config['cases']:
        bridge=dict(data=config['synthetic_data'],checkpoint=case['checkpoint'])
        model,training,provenance=functionality.load_checkpoint(bridge,synthetic_manifest,device)
        validate_training_family(case['case_id'],training)
        if config['inference_variant']!='trained_config_unchanged':raise ValueError('No silent inference overrides permitted')
        report['models'].append(dict(case_id=case['case_id'],status='STARTED',provenance=provenance,
            inference_model_config=training['model'],overrides={},predictions=0))
        current=report['models'][-1]
        current['tensor_state_sha256_before']=energy.tensor_state_sha(model)
        try:
            with torch.inference_mode():
                for row in prepared['targets']:
                    sensors=adapter.load_prepared_sensor(paths['prepared_root'],row)
                    # Existing study forward creates fresh state; baseline forward calls reset_net.
                    left=torch.from_numpy(sensors['left'])[None].to(device)
                    right=torch.from_numpy(sensors['right'])[None].to(device)
                    output_values=model(left,right);pred=output_values['disparity']
                    if tuple(pred.shape)!=(1,1,120,160) or not bool(torch.isfinite(pred).all()):
                        raise ValueError('Exact finite1,1,120,160 primary output required; no clamp or filtering')
                    native=adapter.predict_to_native(pred)[0].cpu().numpy()
                    target=targets[row['sample_id']]
                    report['frame_metrics'].append(dict(method=case['case_id'],sample_id=row['sample_id'],scene_id=row['scene_id'],
                        **native_metrics(native,target)))
                    current['predictions']+=1;report['predictions_completed']+=1
                    del left,right,output_values,pred,native,sensors
        finally:
            current['tensor_state_sha256_after']=energy.tensor_state_sha(model)
            current['weights_and_buffers_unchanged']=current['tensor_state_sha256_before']==current['tensor_state_sha256_after']
            if not current['weights_and_buffers_unchanged']:
                current['status']='FAILED_WEIGHTS_OR_BUFFERS_MUTATED'
                raise RuntimeError('Frozen zero-shot weights or persistent buffers changed during inference')
        current['status']='COMPLETED20_TARGETS'
        del model
        gc.collect();torch.cuda.empty_cache()
    if report['predictions_completed']!=100 or len(report['frame_metrics'])!=120:
        raise ValueError('Incomplete five-family panel cannot produce a completed report')
    report['sequence_metrics'],report['equal_sequence_summary']=aggregate(report['frame_metrics'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--source-registry',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--validate-config',action='store_true')
    args=parser.parse_args();config=json.loads(args.config.read_text())
    validate_config(config,require_registered=not args.validate_config)
    if args.validate_config:
        print(json.dumps(dict(status='VALID_PROSPECTIVE_CONFIG_ONLY_NO_DATA_OR_MODEL_READ',registered=config['registered'])));return
    adapter.require_allocation(config['expected_hostname'])
    if not args.source_registry or not args.output:raise ValueError('Frozen source and fresh output required')
    import numpy as np
    import torch
    output=args.output.resolve();data=config['dsec']
    readonly=[args.source_registry.parent,Path(__file__).resolve().parents[3]]
    readonly.extend(Path(os.environ[data[key]]) for key in ('source_root_env','prepared_root_env'))
    readonly.extend(Path(os.environ[data[key]]).parent for key in
        ('preflight_manifest_env','preparation_config_env','preparation_source_registry_env'))
    readonly.append(Path(os.environ[config['synthetic_data']['root_env']]))
    for case in config['cases']:
        readonly.extend(Path(os.environ[case['checkpoint'][key]]).parent for key in
            ('path_env','metadata_env','training_source_registry_env','training_config_env'))
    if output.exists() or any(output.is_relative_to(path.resolve()) for path in readonly):
        raise ValueError('Fresh output outside read-only datasets/checkpoints/frozen source required')
    output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    report=dict(schema='stereo-study-dsec-transfer-result-v1',status='STARTED',config=config,
        config_sha256=file_sha(args.config),script_sha256=file_sha(__file__),test_access=False,
        gt_targets_attempted=0,gt_target_attempt_ids=[],gt_targets_opened=0,predictions_completed=0,frame_metrics=[],models=[],
        synthetic_observations_opened=False,synthetic_GT_opened=False,adaptation=False,confidence_filter=False,
        claim='zero-shot_DSEC_officialTrain_development_transfer;historically_exposed;no_sensor_preference_or_officialTest_score',
        aggregation='pool_valid_pixels_within_sequence_then_equal_two_sequence_metrics;no_independence_or_significance_claim',
        thresholds='absolute_error_strictly_greater_than1_and2_native_pixels',native_conversion='bilinear_align_corners_false_times4',
        inference_variant='trained_config_unchanged;all50bins;existing_per_forward_reset;baseline_same_eval_refinements',
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),slurm_job_id=os.environ['SLURM_JOB_ID'],
        gpu=torch.cuda.get_device_name(0),cuda=torch.version.cuda,torch=torch.__version__,numpy=np.__version__,
        python=sys.version,hostname=config['expected_hostname'],command=sys.argv)
    def expired(number,frame):raise TimeoutError('600-second registered panel deadline; partial results are incomplete')
    previous=signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,config['max_seconds']-5)
    try:
        torch.set_num_threads(1);random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed'])
        torch.cuda.manual_seed_all(config['seed']);torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        registry=verify_registry(args.source_registry,(Path(__file__),args.config,Path(adapter.__file__),Path(functionality.__file__),Path(energy.__file__)))
        report['source_binding']=dict(source_registry_sha256=file_sha(args.source_registry),
            source_identity_sha256=registry['source_identity_sha256'],git_commit=registry['git_commit'],dirty_status=registry.get('dirty_status'))
        # This existing helper validates only manifest/audit metadata; no dataset is constructed.
        _,_,synthetic_manifest,audit=functionality.load_bound_data(dict(data=config['synthetic_data']))
        report['synthetic_asset_audit_status']=audit['status']
        paths,prepared=load_dsec_metadata(config)
        report['prepared_manifest_sha256']=config['dsec']['prepared_manifest_sha256']
        report['sample_ids']=config['dsec']['sample_ids']
        run_panel(config,paths,prepared,synthetic_manifest,output,report,torch.device('cuda:0'))
        report['status']='COMPLETED_FIXED_FIVE_FAMILY100_PREDICTION_DEVELOPMENT_PANEL'
    except BaseException as error:
        report.update(status='INCOMPLETE_TIME_CAP' if isinstance(error,TimeoutError) else 'FAILED_INCOMPLETE_PANEL',error=repr(error))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
        report['elapsed_seconds']=time.monotonic()-started
        (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'],report=str(output/'report.json'))))
    if not report['status'].startswith('COMPLETED_'):raise SystemExit(1)


if __name__=='__main__':main()
