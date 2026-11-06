"""Prospective Val-only first/second/repeat accuracy diagnostic, one fresh process/case."""
import argparse
import datetime
import json
import os
from pathlib import Path
import random
import signal
import socket
import subprocess
import sys
import time

from evaluate_functionality import file_sha, load_bound_data, load_checkpoint, verify_source

PROTOCOL = dict(forwards_per_target=3, api_warmup_forwards=0,
    input_policy='fresh_identical_clones', rng_policy='restore_same_pretarget_state_before_each_call',
    process_policy='one_baseline_modality_per_fresh_process_no_prior_forward',
    comparison='all_recursive_prediction_leaves_excluding_root_diagnostics',
    atol=1e-5, rtol=1e-5, save_primary_predictions_and_GT=True,
    checkpoint_selection=False, replace_pilot_metrics=False)


def validate_config(config, require_registered=True):
    from profile_arithmetic import validate_config as validate_bindings
    if (config.get('schema') != 'stereo-study-baseline-repeats-v1'
            or config.get('driver') != 'baseline_repeats' or config.get('repeat_protocol') != PROTOCOL
            or config.get('sample_ordinals') != list(range(20))
            or config.get('max_seconds') != 180
            or config.get('scheduler_resources') != {'time':'00:03:00'}):
        raise ValueError('Exact prospective bounded baseline repeat protocol required')
    copied = {**config, 'schema':'stereo-study-arithmetic-v1', 'driver':'arithmetic_accounting'}
    validate_bindings(copied, require_registered)
    case = config.get('case')
    if case not in ('original-I-spike', 'original-I-event'):
        raise ValueError('Exactly one fixed baseline modality per fresh process')
    if (config['training_run'] != 'pilot-'+case+'-seed9471-v1'
            or config['training_config'] != config['training_run']+'.json'
            or config['training_source_snapshot'] != 'source-prefetched-first-seed-pilot-v12'
            or len(config.get('sample_ids', [])) != 20 or len(set(config['sample_ids'])) != 20
            or any(not value.startswith('TRAIN/') for value in config['sample_ids'])):
        raise ValueError('Exact v12 baseline run and ordered20 Val identities required')
    return True


def pointwise_difference(first, second, valid):
    """FP32 subtraction matches the original prediction comparison; no masking of predictions."""
    import torch
    if first.shape != second.shape or first.shape != valid.shape or not torch.isfinite(first).all() or not torch.isfinite(second).all():
        raise ValueError('Same complete finite primary map and GT-only support required')
    delta = (second-first).abs().double()
    def summary(values):
        return dict(pixels=values.numel(), changed_pixels=int((values != 0).sum()),
                    maximum_absolute_difference=float(values.max()), mean_absolute_difference=float(values.mean()))
    if not valid.any():
        raise ValueError('Nonempty common GT support required')
    return dict(all_pixels=summary(delta), valid_GT_pixels=summary(delta[valid]))


def repeat_target(model, left, right):
    """Exactly three unobserved API calls, without prewarming or changing model state/reset."""
    import torch
    from benchmark_energy import prediction_leaves, compare_prediction_leaves, tensor_state_sha, value_fingerprint
    from core.study_training import capture_rng, restore_rng
    initial = capture_rng(); initial_sha = value_fingerprint(initial)
    inputs_sha = value_fingerprint((left, right)); state_sha = tensor_state_sha(model)
    snapshots = []; poststates = []; records = []
    with torch.inference_mode():
        for name in ('first', 'second', 'repeat'):
            restore_rng(initial)
            work_left, work_right = left.clone(), right.clone()
            output = model(work_left, work_right)
            primary = output['disparity']
            if primary.shape != (1,1,*left.shape[-2:]) or not torch.isfinite(primary).all():
                raise ValueError('Finite full unaugmented primary map required')
            poststates.append(value_fingerprint(capture_rng()))
            input_ok = value_fingerprint((work_left,work_right)) == inputs_sha
            weights_ok = tensor_state_sha(model) == state_sha
            records.append(dict(call=name, input_unchanged=input_ok, weights_buffers_unchanged=weights_ok,
                                post_rng_sha256=poststates[-1]))
            if not input_ok or not weights_ok:
                raise ValueError('Input or trained weights/buffers changed during forward')
            snapshots.append(prediction_leaves(output))
            del output
    if value_fingerprint((left,right)) != inputs_sha:
        raise ValueError('Source inputs changed')
    comparisons = {}
    for a,b in ((0,1),(0,2),(1,2)):
        leaves = compare_prediction_leaves(snapshots[a], snapshots[b], 1e-5, 1e-5)
        if any(r['reference_nonfinite_values'] or r['candidate_nonfinite_values'] for r in leaves.values()):
            raise ValueError('Nonfinite returned prediction tensor')
        comparisons[('first','second','repeat')[a]+'_vs_'+('first','second','repeat')[b]] = dict(
            prediction_tensors=leaves, exact_equal=all(r['exact_equal'] for r in leaves.values()),
            within_tolerance=all(r['within_tolerance'] for r in leaves.values()),
            maximum_absolute_difference=max(r['max_absolute_error'] for r in leaves.values()))
    primaries = [s["$['disparity']"][1][0] for s in snapshots]
    # Preserve every observed difference, including a nonrepeatable second/third call.
    # Only integrity failures reject; numerical differences are the outcome under study.
    return primaries, dict(comparisons=comparisons, calls=records, input_sha256=inputs_sha,
        pre_rng_sha256=initial_sha, post_rng_matches_between_calls=len(set(poststates))==1,
        rng_consumption_occurred=poststates[0]!=initial_sha, trained_state_sha256=state_sha)


def aggregate(rows):
    from core.study_diagnostics import aggregate_scene_metrics
    scenes = aggregate_scene_metrics(rows)
    summary = {name: sum(r['epe'] for r in scenes if r['condition']==name)/len({r['scene_id'] for r in scenes})
               for name in ('first','second','repeat')}
    paired = []
    for scene in sorted({r['scene_id'] for r in scenes}):
        values = {r['condition']:r['epe'] for r in scenes if r['scene_id']==scene}
        paired.append(dict(scene_id=scene, **values, second_minus_first=values['second']-values['first'],
                           repeat_minus_second=values['repeat']-values['second']))
    return dict(scene_metrics=scenes, equal_scene_epe=summary, paired_scene_epe_changes=paired,
        equal_scene_second_minus_first=summary['second']-summary['first'],
        equal_scene_repeat_minus_second=summary['repeat']-summary['second'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True); p.add_argument('--source-registry',type=Path)
    p.add_argument('--output',type=Path); p.add_argument('--validate-config',action='store_true')
    a=p.parse_args(); config=json.loads(a.config.read_text()); validate_config(config, not a.validate_config)
    if a.validate_config:
        print(json.dumps(dict(status='VALID_CONFIG_ONLY_NO_ARRAYS_OR_MODEL',sha256=file_sha(a.config)))); return
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname()!=config['expected_hostname']:
        raise RuntimeError('Registered authorized Slurm allocation required')
    if not a.source_registry or not a.output:
        raise ValueError('Frozen source and new output required')
    import numpy as np
    import torch
    from benchmark_energy import tensor_state_sha
    from core.study_data import PairedStereoDataset
    from core.study_diagnostics import disparity_metrics
    from core.study_training import atomic_json
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or not any(v in torch.cuda.get_device_name(0) for v in ('4090','5090')):
        raise RuntimeError('Exactly one authorized RTX4090/5090 required')
    registry=verify_source(a.source_registry,a.config)
    member=str(Path(__file__).resolve().relative_to(a.source_registry.parent.resolve()))
    if registry['source_files'].get(member)!=file_sha(Path(__file__)):
        raise ValueError('Driver missing from frozen source')
    root, path, manifest, audit=load_bound_data(config)
    if [r['sample_id'] for r in manifest['val']]!=config['sample_ids']:
        raise ValueError('Ordered common Val panel changed')
    if any(a.output.resolve().is_relative_to(v.resolve()) for v in (root,a.source_registry.parent)):
        raise ValueError('Output outside immutable source/data required')
    a.output.mkdir(parents=True,exist_ok=False)
    result=dict(schema='stereo-study-baseline-repeats-result-v1',status='INITIALIZING',case=config['case'],
        config=config,config_sha256=file_sha(a.config),source_registry_sha256=file_sha(a.source_registry),
        source_identity_sha256=registry['source_identity_sha256'],git_commit=registry['git_commit'],
        source_dirty_status=registry['dirty_status'],source_overlay_sha256=registry['overlay_sha256'],
        slurm_job_id=os.environ['SLURM_JOB_ID'],slurm_nodes=os.environ.get('SLURM_JOB_NODELIST'),
        process_id=os.getpid(),hostname=socket.gethostname(),python=sys.version,torch=torch.__version__,
        numpy=np.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
        allocated_gpu_uuid=str(getattr(torch.cuda.get_device_properties(0),'uuid',None)),
        gpu=torch.cuda.get_device_name(0),gpu_properties=str(torch.cuda.get_device_properties(0)),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),command=sys.argv,
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),test_access=False,
        prior_forward_calls=0,frame_metrics=[],samples=[],checkpoint_selection=False,replace_pilot_metrics=False,
        aggregation='GT_pixels_within_scene_then_equal10scenes;paired_scene_deltas;descriptive_no_CI',
        first_call_scope='Only ordinal0 is process-cold; subsequent first calls are first for that input in the same process')
    save=lambda:atomic_json(a.output/'result.json',result)
    save(); tick=time.monotonic()
    def expired(number,frame):raise TimeoutError('Registered180second bound')
    previous=signal.signal(signal.SIGALRM,expired); signal.setitimer(signal.ITIMER_REAL,175)
    try:
        result['gpu_readonly_snapshot']=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,driver_version,pstate,clocks.sm,clocks.mem,power.limit','--format=csv,noheader'],text=True)
        torch.set_num_threads(1)
        random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed']); torch.cuda.manual_seed_all(config['seed'])
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
        result['backend_settings']=dict(matmul_tf32=False,cudnn_tf32=False,cudnn_benchmark=False,cudnn_deterministic=True,
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),jit='unchanged',cpu_threads=1)
        model,training,checkpoint=load_checkpoint(config,manifest,torch.device('cuda:0'))
        if (training['model']['family']!='baseline' or training['model']['config']['architecture']!='original_I'
                or training['data']['modality']!=config['case'].split('-')[-1] or model.eval_refinements!=32):
            raise ValueError('Exact unchanged ALIF+I K32 baseline required')
        result.update(status='RUNNING',checkpoint=checkpoint,trained_state_sha256=tensor_state_sha(model),
            temporal_semantics='legacy_T50_time_as_channels_K32_not50_encoder_simulation_steps')
        val=PairedStereoDataset(root,path,training['data']['modality'],'val',purpose='engineering')
        if [r['sample_id'] for r in val.rows]!=config['sample_ids']:
            raise ValueError('Loader order changed')
        for ordinal in range(20):
            sample=val[ordinal]; left,right=(sample[k][None].to('cuda:0') for k in ('left','right'))
            primary,evidence=repeat_target(model,left,right)
            identity=dict(ordinal=ordinal,sample_id=sample['sample_id'],scene_id=sample['scene_id'])
            target,valid=sample['target'],sample['valid'] & torch.isfinite(sample['target']) & (sample['target']>0)
            for name,pred in zip(('first','second','repeat'),primary):
                result['frame_metrics'].append(dict(method=config['case'],condition=name,**identity,
                    **disparity_metrics(pred,target,valid)))
            artifact=a.output/('sample-'+str(ordinal)+'.npz')
            np.savez_compressed(artifact,first=primary[0].numpy(),second=primary[1].numpy(),repeat=primary[2].numpy(),
                target=target.numpy(),valid=valid.numpy())
            result['samples'].append(dict(**identity,process_cold=ordinal==0,**evidence,
                primary_first_vs_second=pointwise_difference(primary[0],primary[1],valid),
                primary_second_vs_repeat=pointwise_difference(primary[1],primary[2],valid),
                artifact=artifact.name,artifact_sha256=file_sha(artifact)))
            save(); del left,right,primary,sample
        result.update(**aggregate(result['frame_metrics']))
        if tensor_state_sha(model)!=result['trained_state_sha256']:
            raise ValueError('Final trained state changed')
        result.update(status='COMPLETED_BASELINE_REPEAT_DIAGNOSTIC',weights_buffers_unchanged=True,
            interpretation='Numerical reproducibility on this process/device/panel; no causal mechanism or replacement pilot score')
    except Exception as error:
        result.update(status='INCOMPLETE_TIME_CAP' if isinstance(error,TimeoutError) else 'FAILED_BASELINE_REPEAT_DIAGNOSTIC',error=repr(error))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
        result['elapsed_seconds_not_latency']=time.monotonic()-tick;save()
    print(json.dumps(dict(status=result['status'],samples=len(result['samples']))))
    if not result['status'].startswith('COMPLETED_'):raise SystemExit(1)

if __name__=='__main__':main()
