"""Independent saved-map arithmetic readback; GT/maps open only in registered Slurm, no model."""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import socket
import time
from evaluate_functionality import file_sha,verify_source


def validate_config(c):
    if (c.get('schema')!='stereo-study-repeat-map-readback-v1' or c.get('registered') is not True
            or c.get('driver')!='baseline_repeat_readback' or c.get('stage')!='engineering_diagnostic'
            or c.get('max_seconds')!=60 or c.get('scheduler_resources')!={'time':'00:01:00'}
            or c.get('allocated_gpu_hour_cap')!=24 or c.get('test_access')!='sealed'
            or [r['run'] for r in c.get('cases',[])]!=['baseline-repeats-original-I-spike-v1','baseline-repeats-original-I-event-v1']):
        raise ValueError('Exact registered bounded two-case map readback required')
    for case in c['cases']:
        if len(case['files'])!=21 or set(case['files'])!={'result.json',*(f'sample-{i}.npz' for i in range(20))}:
            raise ValueError('Exact42 raw members required')
        for value in case['files'].values():
            if value['bytes']<=0 or len(value['sha256'])!=64 or any(v not in '0123456789abcdef' for v in value['sha256']):
                raise ValueError('Pin each exact raw member before readback')


def verify_maps(arrays,metrics,evidence):
    """Independent NumPy scalar reconstruction, preserving FP32 subtraction before FP64 sum."""
    import numpy as np
    if set(arrays)!=set(('first','second','repeat','target','valid')):
        raise ValueError('Exact stored primary predictions/GT/support required')
    target,valid=arrays['target'],arrays['valid']
    if target.dtype!=np.float32 or valid.dtype!=np.bool_ or target.shape!=valid.shape or target.ndim!=3 or target.shape[0]!=1:
        raise ValueError('Exact saved GT shape/dtype')
    if not valid.any() or np.any(valid & (~np.isfinite(target) | (target<=0))):raise ValueError('GT-only support changed')
    def check(actual,expected):
        if not np.isclose(actual,expected,atol=1e-12,rtol=1e-12):raise ValueError('Saved pixel arithmetic disagrees with scalar receipt')
    for name in ('first','second','repeat'):
        pred=arrays[name]
        if pred.shape!=target.shape or pred.dtype!=np.float32 or not np.isfinite(pred).all():raise ValueError('Invalid saved primary')
        error=np.abs(pred[valid]-target[valid]).astype(np.float64)
        row=metrics[name]
        check(error.sum(),row['absolute_error_sum']);check(error.mean(),row['epe']);check(valid.sum(),row['valid_pixels'])
        for threshold in (1,3,5):check((error>threshold).sum(),row[f'bad{threshold}_pixels'])
    for a,b,key in [('first','second','primary_first_vs_second'),('second','repeat','primary_second_vs_repeat')]:
        delta=np.abs(arrays[b]-arrays[a]).astype(np.float64)
        for scope,values in [('all_pixels',delta),('valid_GT_pixels',delta[valid])]:
            expected=evidence[key][scope]
            for name,value in [('pixels',values.size),('changed_pixels',(values!=0).sum()),
                               ('maximum_absolute_difference',values.max()),('mean_absolute_difference',values.mean())]:
                check(value,expected[name])
    return True


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--source-registry',type=Path);p.add_argument('--output',type=Path);p.add_argument('--validate-config',action='store_true')
    a=p.parse_args();c=json.loads(a.config.read_text());validate_config(c)
    if a.validate_config:print('VALID_CONFIG_ONLY_NO_ARRAYS');return
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname()!=c['expected_hostname']:raise RuntimeError('Registered Slurm only')
    import numpy as np
    import torch
    from core.study_training import atomic_json
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or not any(v in torch.cuda.get_device_name(0) for v in ('4090','5090')):
        raise RuntimeError('Exactly one authorized RTX GPU allocation required')
    registry=verify_source(a.source_registry,a.config)
    member=str(Path(__file__).resolve().relative_to(a.source_registry.parent.resolve()))
    if registry['source_files'].get(member)!=file_sha(Path(__file__)):raise ValueError('Driver absent from frozen source')
    parent=Path(os.environ['STUDY_REPEAT_PARENT']);roots=[parent/r['run'] for r in c['cases']]
    if any(a.output.resolve().is_relative_to(v.resolve()) for v in [*roots,a.source_registry.parent]):raise ValueError('Fresh output outside inputs/source')
    a.output.mkdir(parents=True,exist_ok=False)
    result=dict(schema='stereo-study-repeat-map-readback-result-v1',status='STARTED',cases=[],
        source_identity_sha256=registry['source_identity_sha256'],source_registry_sha256=file_sha(a.source_registry),
        config_sha256=file_sha(a.config),git_commit=registry['git_commit'],
        source_dirty_status=registry['dirty_status'],source_overlay_sha256=registry['overlay_sha256'],
        slurm_job_id=os.environ['SLURM_JOB_ID'],gpu=torch.cuda.get_device_name(0),
        allocated_gpu_uuid=str(getattr(torch.cuda.get_device_properties(0),'uuid',None)),
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),model_loaded=False,test_access=False,
        scope='Independent archived-primary/GT pixel arithmetic only, no new inference/checkpoint selection/pilot replacement')
    tick=time.monotonic()
    def expired(number,frame):raise TimeoutError('Readback60second bound')
    previous=signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,55)
    try:
        for case,root in zip(c['cases'],roots):
            def verify_files():
                if {v.name for v in root.iterdir()}!=set(case['files']) or any(v.is_symlink() for v in root.iterdir()):raise ValueError('Raw exact set/type mismatch')
                for name,record in case['files'].items():
                    if (root/name).stat().st_size!=record['bytes'] or file_sha(root/name)!=record['sha256']:raise ValueError('Raw member changed')
            verify_files();r=json.loads((root/'result.json').read_text())
            if r['status']!='COMPLETED_BASELINE_REPEAT_DIAGNOSTIC' or len(r['samples'])!=20:raise ValueError('Completed panel required')
            samples=[]
            for sample in r['samples']:
                metrics={v['condition']:v for v in r['frame_metrics'] if v['sample_id']==sample['sample_id']}
                if sample['artifact']!=f"sample-{sample['ordinal']}.npz":raise ValueError('Safe exact artifact required')
                with np.load(root/sample['artifact'],allow_pickle=False) as packed:
                    arrays={key:packed[key] for key in packed.files}
                verify_maps(arrays,metrics,sample)
                samples.append(dict(sample_id=sample['sample_id'],ordinal=sample['ordinal'],status='PASS_ALL_STORED_PIXEL_ARITHMETIC'))
            verify_files()
            result['cases'].append(dict(run=case['run'],source_report_sha256=case['files']['result.json']['sha256'],samples=samples,
                predictions_verified=60,pointwise_comparisons_verified=40,remote_before_after_hashes_equal=True))
        result.update(status='PASS_ALL120_STORED_PREDICTION_MAPS_AND40_GT_SUPPORTS',prediction_maps=120,GT_supports=40,pointwise_comparisons=80)
    except Exception as error:result.update(status='FAILED_SAVED_MAP_READBACK',error=repr(error))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
        result['elapsed_seconds_not_latency']=time.monotonic()-tick;atomic_json(a.output/'result.json',result)
    print(json.dumps(dict(status=result['status'])))
    if not result['status'].startswith('PASS_'):raise SystemExit(1)

if __name__=='__main__':main()
