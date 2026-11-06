"""Bounded A/B-spike branch-resolved existing+2 shift diagnostic; no launcher.

Same selected checkpoints/full trained configs/20 Val targets. Matcher is the
stride4 estimate interpolated to the decoder's pad16 grid then cropped. Its
residual is main minus matcher. Branch scores cannot replace main predictions.
No Test access, retraining, adaptation, confidence masks or architecture edits.
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

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
sys.path.insert(0,str(Path(__file__).resolve().parent))

import benchmark_energy as energy
import evaluate_functionality as functionality
from core.study_diagnostics import (aggregate_scene_metrics,disparity_metrics,
    shift_response,shift_right_input,shifted_disparity_support)
from core.study_data import PairedStereoDataset,safe_path

BRANCHES=('main','matcher','residual')
TOLERANCE={'atol':1e-5,'rtol':1e-5}
file_sha=functionality.file_sha
digest=functionality.digest


def reference_binding(case):
    """Normalize only deployment environment names; selected bytes stay fixed."""
    return {key:value for key,value in case['checkpoint'].items() if not key.endswith('_env')}


def validate_config(config,*,require_registered=True):
    if (config.get('schema')!='stereo-study-branch-shift-v1' or config.get('driver')!='branch_shift'
            or config.get('stage')!='engineering_diagnostic' or config.get('scope')!='engineering'
            or config.get('test_access')!='sealed' or config.get('allocated_gpu_hour_cap')!=24
            or config.get('conditions')!=['correct','shifted'] or config.get('shift_pixels')!=2
            or config.get('inference_variant')!='trained_config_unchanged'
            or config.get('identity_tolerance')!=TOLERANCE
            or config.get('maximum_forwards')!=80 or config.get('batch_size')!=1 or config.get('T')!=50):
        raise ValueError('Fixed A/B-spike80-forward existing+2-shift engineering contract required')
    if require_registered and config.get('registered') is not True:
        raise ValueError('Registered configuration required for execution')
    if (type(config.get('max_seconds')) is not int or not 30<=config['max_seconds']<=300
            or config.get('scheduler_resources')!={'time':'00:05:00'}
            or config.get('main_shift_screen')!={'median_scene_mean_range':[.5,1.5],'positive_scenes_required':6,
                'changes_primary_functionality_protocol':False}):
        raise ValueError('Retain300-second bound and unchanged primary main shift margin')
    if [case.get('case_id') for case in config.get('cases',[])]!=['A-spike','B-spike']:
        raise ValueError('Same selected A and B spike checkpoints required')
    for case in config['cases']:
        bridge=dict(schema='stereo-study-functionality-v1',registered=config['registered'],purpose='engineering',
            mode='checkpoint',seed=config['seed'],expected_hostname=config['expected_hostname'],
            max_seconds=config['max_seconds'],shift_pixels=2,data=config['data'],checkpoint=case['checkpoint'])
        functionality.validate_config(bridge,require_registered=require_registered)
        if case['checkpoint']['selection']!='best-val.pt' or case['checkpoint']['update']!=1000:
            raise ValueError('Only the already selected1000-update best-Val checkpoints allowed')
        member=case['functionality_config_member']
        path=safe_path(Path(__file__).resolve().parents[5],member)
        if file_sha(path)!=case['functionality_config_sha256']:
            raise ValueError('Original selected functionality config identity changed')
        reference=json.loads(path.read_text())
        if (reference_binding(case)!=reference_binding(reference) or config['data']!=reference['data']
                or any(case.get(key)!=reference[key] for key in ('training_run','training_source_snapshot','training_config'))):
            raise ValueError('Branch diagnostic cannot replace selected checkpoints, logs, source or cohort')
        status=case.get('completed_training_receipt',{})
        if (status.get('status')!='COMPLETED_FROZEN_HORIZON' or status.get('update')!=1000
                or status.get('presentations')!=8000 or status.get('best_update')!=1000
                or status.get('total_horizon_updates')!=1000 or status.get('test_access')!='sealed'):
            raise ValueError('Completed full-horizon training receipt required')
        for key in ('training_source_identity_sha256','training_initial_weights_sha256',
                    'train_sample_ids_sha256','val_sample_ids_sha256'):
            if not re.fullmatch(r'[0-9a-f]{64}',case.get(key,'')):
                raise ValueError('Exact selected training binding required: '+key)
    return True


def verify_source(path,config_path,config):
    registry=json.loads(path.read_text())
    if hashlib.sha256(json.dumps(registry['source_files'],sort_keys=True).encode()).hexdigest()!=registry['source_identity_sha256']:
        raise ValueError('Frozen source identity changed')
    for member,expected in registry['source_files'].items():
        if file_sha(safe_path(path.parent,member))!=expected:raise ValueError('Frozen source member changed: '+member)
    required=[Path(__file__),config_path,Path(functionality.__file__),Path(energy.__file__)]
    required.extend(safe_path(path.parent,case['functionality_config_member']) for case in config['cases'])
    for item in required:
        if registry['source_files'].get(str(item.resolve().relative_to(path.parent.resolve())))!=file_sha(item):
            raise ValueError('Diagnostic/helper/selected config absent from frozen source')
    return registry


def extract_branches(output,input_shape):
    """Exact existing decoder geometry; all branches retain image-pixel units."""
    import torch
    import torch.nn.functional as F
    h,w=map(int,input_shape);ph,pw=h+(-h)%16,w+(-w)%16
    main=output['disparity'].detach();aux=output['aux_disparities_native'][4].detach()
    if tuple(main.shape)!=(1,1,h,w) or tuple(aux.shape)!=(1,1,ph//4,pw//4):
        raise ValueError('Exact batch1 main/stride4 padded matcher layout required')
    if not torch.isfinite(main).all() or not torch.isfinite(aux).all():
        raise ValueError('All main/matcher pixels must be finite; preserve finite zero/negative values')
    matcher=F.interpolate(aux,size=(ph,pw),mode='bilinear',align_corners=False)[...,:h,:w]
    return {'main':main[0].cpu(),'matcher':matcher[0].cpu(),'residual':(main-matcher)[0].cpu()}


def check_mean_identity(values):
    lhs=values['main'];rhs=values['matcher']+values['residual']
    if not all(math.isfinite(v) for v in (lhs,rhs)) or not math.isclose(lhs,rhs,abs_tol=TOLERANCE['atol'],rel_tol=TOLERANCE['rtol']):
        raise ValueError('Signed mean response identity main=matcher+residual failed')
    return dict(main_mean=lhs,matcher_plus_residual_mean=rhs,absolute_error=abs(lhs-rhs),**TOLERANCE)


def response_row(case_id,sample,correct,shifted,common,delta):
    values={name:shift_response(correct[name],shifted[name],delta,common) for name in BRANCHES}
    n=values['main']['response_pixels']
    if not n or any(row['response_pixels']!=n for row in values.values()):
        raise ValueError('Identical nonempty GT-only common response support required')
    trusted=int((sample['valid']&sample['target'].isfinite()&(sample['target']>0)).sum())
    identity=check_mean_identity({name:row['response_ratio'] for name,row in values.items()})
    return dict(case_id=case_id,sample_id=sample['sample_id'],scene_id=sample['scene_id'],
        response_pixels=n,total_pixels=common.numel(),trusted_original_pixels=trusted,
        common_image_coverage=n/common.numel(),common_trusted_GT_coverage=n/trusted,
        branches=values,mean_identity=identity)


def summarize_responses(rows):
    import statistics
    scenes=[];equal=[]
    for case_id in sorted({row['case_id'] for row in rows}):
        values=[r for r in rows if r['case_id']==case_id]
        for scene_id in sorted({r['scene_id'] for r in values}):
            frames=[r for r in values if r['scene_id']==scene_id]
            if len(frames)!=2 or len({r['sample_id'] for r in frames})!=2:
                raise ValueError('Two fixed Val frames per scene required')
            n=sum(r['response_pixels'] for r in frames)
            branches={name:dict(response_sum=sum(r['branches'][name]['response_sum'] for r in frames),
                response_pixels=n,response_ratio=sum(r['branches'][name]['response_sum'] for r in frames)/n,
                positive_response_pixels=sum(r['branches'][name]['positive_response_pixels'] for r in frames)) for name in BRANCHES}
            identity=check_mean_identity({name:value['response_ratio'] for name,value in branches.items()})
            total=sum(r['total_pixels'] for r in frames);trusted=sum(r['trusted_original_pixels'] for r in frames)
            scenes.append(dict(case_id=case_id,scene_id=scene_id,frames=2,sample_ids=sorted(r['sample_id'] for r in frames),
                response_pixels=n,total_pixels=total,trusted_original_pixels=trusted,
                common_image_coverage=n/total,common_trusted_GT_coverage=n/trusted,branches=branches,mean_identity=identity))
        selected=[r for r in scenes if r['case_id']==case_id]
        if len(selected)!=10:raise ValueError('All ten Val scenes required; no selective removal')
        means={name:sum(r['branches'][name]['response_ratio'] for r in selected)/10 for name in BRANCHES}
        main=[r['branches']['main']['response_ratio'] for r in selected]
        median=statistics.median(main);positive=sum(v>0 for v in main)
        equal.append(dict(case_id=case_id,scenes=10,mean_of_scene_mean_response_ratios=means,
            mean_identity=check_mean_identity(means),main_median_of_scene_mean_response_ratios=median,
            positive_main_response_scenes=positive,
            unchanged_main_shift_screen=dict(median_in_registered_range=.5<=median<=1.5,majority_scenes_positive=positive>=6),
            common_trusted_GT_coverage_mean=sum(r['common_trusted_GT_coverage'] for r in selected)/10,
            branch_medians='not_reported;medians_do_not_add;branches_are_diagnostic_only'))
    return scenes,equal


def validate_loaded_case(case,training,provenance):
    values=training['model']['config'];binding=provenance['binding']
    if (training['model']['family']!='study' or training['data']['modality']!='spike'
            or values.get('architecture')!=case['case_id'][0] or values.get('spiking') is not True
            or values.get('input_channels')!=1 or values.get('time_steps')!=50 or values.get('widths')!=[32,64,96,128]
            or training['total_updates']!=1000 or training['training']['evaluation_updates']!=[250,500,1000]):
        raise ValueError('Full selected spike model configuration required without inference overrides')
    if (binding['source_identity_sha256']!=case['training_source_identity_sha256']
            or binding['train_sample_ids_sha256']!=case['train_sample_ids_sha256']
            or binding['val_sample_ids_sha256']!=case['val_sample_ids_sha256']):
        raise ValueError('Selected source/cohort binding changed')
    metadata=json.loads(Path(os.environ[case['checkpoint']['metadata_env']]).read_text())
    if metadata.get('fresh_seed_model_sha256')!=case['training_initial_weights_sha256']:
        raise ValueError('Selected fresh initialization changed')


def run(config,root,manifest_path,manifest,output,report,device):
    import torch
    dataset=PairedStereoDataset(root,manifest_path,'spike','val',purpose='engineering')
    if (len(dataset)!=20 or len({r['scene_id'] for r in dataset.rows})!=10
            or config['preselected_case_ids']!=functionality.selected_cases(manifest)
            or config['shift_pixels']!=int(8*manifest['output_scale'])):
        raise ValueError('Exact20 Val/ten scenes and existing primary shift/case IDs required')
    for case in config['cases']:
        bridge=dict(data=config['data'],checkpoint=case['checkpoint'])
        model,training,provenance=functionality.load_checkpoint(bridge,manifest,device)
        validate_loaded_case(case,training,provenance)
        current=dict(case_id=case['case_id'],status='STARTED',checkpoint=provenance,
            inference_model_config=training['model'],overrides={},forwards=0,
            tensor_state_sha256_before=energy.tensor_state_sha(model))
        report['models'].append(current)
        try:
            with torch.inference_mode():
                for index in range(20):
                    sample=dataset[index];left=sample['left'][None].to(device);right=sample['right'][None].to(device)
                    predictions={}
                    for condition,value in [('correct',right),('shifted',shift_right_input(right,2))]:
                        if report['forwards_completed']>=80:raise RuntimeError('Registered80-forward cap exceeded')
                        output_values=model(left,value)
                        predictions[condition]=extract_branches(output_values,left.shape[-2:])
                        current['forwards']+=1;report['forwards_completed']+=1
                        del output_values
                    transformed,common=shifted_disparity_support(sample['target'],sample['valid'],2)
                    report['frame_responses'].append(response_row(case['case_id'],sample,predictions['correct'],predictions['shifted'],common,2))
                    for branch in ('main','matcher'):
                        for condition,target in [('correct_common',sample['target']),('shifted_geometry',transformed)]:
                            pred=predictions['correct' if condition=='correct_common' else 'shifted'][branch]
                            report['frame_metrics'].append(dict(method=case['case_id']+'/'+branch,condition=condition,
                                scene_id=sample['scene_id'],sample_id=sample['sample_id'],
                                reference_target='original' if condition=='correct_common' else 'original_minus2',
                                **disparity_metrics(pred,target,common)))
                    report['frame_metrics'].append(dict(method=case['case_id']+'/main',condition='correct_full',
                        scene_id=sample['scene_id'],sample_id=sample['sample_id'],reference_target='original',
                        **disparity_metrics(predictions['correct']['main'],sample['target'],sample['valid'])))
                    del predictions,left,right,sample
        finally:
            current['tensor_state_sha256_after']=energy.tensor_state_sha(model)
            current['weights_and_buffers_unchanged']=current['tensor_state_sha256_after']==current['tensor_state_sha256_before']
            if not current['weights_and_buffers_unchanged']:
                current['status']='FAILED_WEIGHTS_OR_BUFFERS_MUTATED'
                raise RuntimeError('Frozen weights or persistent buffers changed')
        current['status']='COMPLETED40_FORWARDS20_TARGETS'
        del model
        gc.collect();torch.cuda.empty_cache()
    if report['forwards_completed']!=80 or len(report['frame_responses'])!=40 or len(report['frame_metrics'])!=200:
        raise ValueError('Entire two-model20-target panel required before aggregation')
    report['scene_responses'],report['scene_equal_response_summary']=summarize_responses(report['frame_responses'])
    report['scene_metrics']=aggregate_scene_metrics(report['frame_metrics'])
    report['scene_equal_accuracy_summary']=functionality.scene_equal_summary(report['scene_metrics'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--source-registry',type=Path)
    parser.add_argument('--output',type=Path);parser.add_argument('--validate-config',action='store_true')
    args=parser.parse_args();config=json.loads(args.config.read_text())
    validate_config(config,require_registered=not args.validate_config)
    if args.validate_config:
        print(json.dumps(dict(status='VALID_CONFIG_ONLY_NO_ARRAYS_OR_MODELS',config_sha256=file_sha(args.config))));return
    import numpy as np
    import torch
    if not args.source_registry or not args.output or not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Frozen source/fresh output/authorized Slurm required')
    import socket
    if (socket.gethostname()!=config['expected_hostname'] or not torch.cuda.is_available() or torch.cuda.device_count()!=1
            or not any(x in torch.cuda.get_device_name(0) for x in ('4090','5090'))):
        raise RuntimeError('One allocated authorized RTX4090/5090 on the registered host required')
    output=args.output.resolve();readonly=[args.source_registry.parent,Path(__file__).resolve().parents[3],Path(os.environ[config['data']['root_env']])]
    for case in config['cases']:
        readonly.extend(Path(os.environ[case['checkpoint'][key]]).parent for key in ('path_env','metadata_env','training_source_registry_env','training_config_env'))
    if output.exists() or any(output.is_relative_to(p.resolve()) for p in readonly):
        raise ValueError('Fresh output outside read-only source/data/checkpoints required')
    output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    report=dict(schema='stereo-study-branch-shift-result-v1',status='STARTED',config=config,config_sha256=file_sha(args.config),
        script_sha256=file_sha(__file__),forwards_completed=0,models=[],frame_metrics=[],frame_responses=[],test_access=False,
        train_observations_opened=False,train_GT_opened=False,confidence_filter=False,adaptation=False,
        claim='branch_resolved_existing_shift_diagnostic_only;no_architecture_change_or_new_main_screen',
        means_identity='main=matcher+residual_on_identical_common_GT_support;medians_do_not_add',
        response_definition='signed_response=(correct_prediction-shifted_prediction)/2;response_sum_is_sum_of_these_ratios',
        matcher_geometry='aux_disparities_native[4];bilinear_align_corners_false_to_pad16_then_crop;image_pixel_units_no_stride_multiply',
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),slurm_job_id=os.environ['SLURM_JOB_ID'],
        slurm_cluster=os.environ.get('SLURM_CLUSTER_NAME'),slurm_nodes=os.environ.get('SLURM_JOB_NODELIST'),
        gpu=torch.cuda.get_device_name(0),torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,python=sys.version,command=sys.argv)
    def expired(number,frame):raise TimeoutError('Registered300-second branch diagnostic deadline')
    previous=signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,config['max_seconds']-5)
    try:
        torch.set_num_threads(1);random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed'])
        torch.cuda.manual_seed_all(config['seed']);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        registry=verify_source(args.source_registry,args.config,config)
        report['source_binding']=dict(source_registry_sha256=file_sha(args.source_registry),source_identity_sha256=registry['source_identity_sha256'],
            git_commit=registry['git_commit'],dirty_status=registry.get('dirty_status'))
        root,manifest_path,manifest,audit=functionality.load_bound_data(config)
        report.update(manifest_sha256=config['data']['manifest_sha256'],asset_audit_status=audit['status'],preselected_case_ids=config['preselected_case_ids'])
        run(config,root,manifest_path,manifest,output,report,torch.device('cuda:0'))
        report['status']='COMPLETED_BRANCH_RESOLVED80_FORWARD_DIAGNOSTIC'
    except BaseException as error:
        report.update(status='INCOMPLETE_TIME_CAP' if isinstance(error,TimeoutError) else 'FAILED_INCOMPLETE_DIAGNOSTIC',error=repr(error))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
        report['elapsed_seconds']=time.monotonic()-started
        (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'],report=str(output/'report.json'))))
    if not report['status'].startswith('COMPLETED_'):raise SystemExit(1)


if __name__=='__main__':main()
