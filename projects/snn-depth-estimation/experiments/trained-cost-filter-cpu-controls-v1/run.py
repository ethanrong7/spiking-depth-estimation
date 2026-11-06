"""Bounded user-authorized native decoder CPU intervention; no GPU or training."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
for _k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_k]='2'
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import sys
import time

import torch
from controls import CONDITIONS,intervention

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
BASE=REPO/'storage/experiment-runs/snn-depth-estimation'
STEREO=BASE/'spiking-transformer-stereo'
RAW=BASE/'trained-cost-filter-cpu-controls-v1'
SHA='d5dad669ba1ba521f36f7ede66142f163afca896cf677f7586a4f4b53829031b'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');temp.replace(path)


def state_digest(model):
    h=hashlib.sha256()
    for n,v in sorted(model.state_dict().items()):h.update(n.encode());h.update(v.cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-id',required=True)
    parser.add_argument('--cpu-only-amendment',action='store_true');args=parser.parse_args()
    assert (HERE/'STUDY.md').exists()
    prior_seconds=0.
    if args.cpu_only_amendment:
        assert (HERE/'CPU-ONLY-AMENDMENT.md').exists()
        prior=json.loads((RAW/'native-first2-20261104/result.json').read_text())
        assert prior['status']=='STOPPED_CPU_GPU_BASELINE_MISMATCH'
        prior_seconds=prior['elapsed_compute_seconds']
    out=RAW/args.run_id;out.mkdir(parents=True,exist_ok=False)
    compact=HERE/'results'/args.run_id;compact.mkdir(parents=True,exist_ok=False)
    source=RAW/'source';project=source/'projects/snn-depth-estimation'
    registry=json.loads((source/'source-registry.json').read_text())
    assert registry['source_identity_sha256']=='606dcfe39ee26b7eecb4f52768376db76a80c5a4243929795cec2d233ba82f56'
    for rel,sha in registry['source_files'].items():assert digest(source/rel)==sha,rel
    checkpoint=STEREO/'spiking-attention-placement-v1/train/A/step-30000.pt'
    assert checkpoint.stat().st_size==146168349 and digest(checkpoint)==SHA
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    sys.path[:0]=[str(project),str(project/'core'),str(project/'experiments/spiking-transformer-stereo/scripts')]
    driver=load('cpu_frozen_native_driver',project/'experiments/spiking-transformer-stereo/scripts/run_experiment.py')
    probes_path=REPO/'projects/snn-depth-estimation/experiments/spiking-attention-placement-v1/probes.py'
    assert digest(probes_path) in {'da0fa6ab64de0649af1e7afc9c586e79fc0ccc83740824431713e1a8eb766098','45876d358b3420e328ad4f5937764b2fa637109438e27a7a67d12d46b7c4e979'}
    probes=load('cpu_admitted_metrics',probes_path)
    from core.spiking_attention_placement import install
    import cv2
    cv2.setNumThreads(1)
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert ck['updates']==30000 and ck['samples_seen']==240000 and ck['source_identity']==registry['source_identity_sha256']
    assert ck['config']['arm']=='A' and ck['config']['seed']==1234
    model=install(driver.build_model('original',1234),'A');model.load_state_dict(ck['model'],strict=True)
    assert all(torch.equal(t,ck['model'][n]) for n,t in model.state_dict().items());del ck
    model.eval();initial_state=state_digest(model)
    manifest_path=project/'experiments/spiking-transformer-stereo/configs/manifest.json'
    assert digest(manifest_path)==probes.MANIFEST_SHA
    rows=json.loads(manifest_path.read_text())['dev'];cohort=[next(r for r in rows if r['scene']==s and r['frame']=='0001') for s in ('scene_0120','scene_0121')]
    dataset=driver.ManifestDataset(REPO/'data/Synthetic',cohort)
    saved=json.loads((STEREO/'spiking-attention-placement-phase2/frozen-ablation-v2/A/update-30000/baseline.json').read_text())
    official={(r['scene'],r['frame']):r for r in saved['per_frame']}
    result=dict(status='RUNNING_CPU_DIAGNOSTIC',started_utc=datetime.now(timezone.utc).isoformat(),checkpoint_sha256=SHA,
        source_identity=registry['source_identity_sha256'],source_registry_sha256=digest(source/'source-registry.json'),
        evaluator_sha256=digest(__file__),control_source_sha256=digest(HERE/'controls.py'),study_sha256=digest(HERE/'STUDY.md'),
        probes_sha256=digest(probes_path),manifest_sha256=digest(manifest_path),threads=2,iterations=32,compute_limit_seconds=300,
        CPU_GPU_gate=dict(epe_absolute_px=.005,bad_percentage_points=.05),conditions=CONDITIONS,per_frame=[],
        torch=torch.__version__,gpu_used=False,training_steps=0,
        scope='separate CPU-only within-backend causal diagnostic' if args.cpu_only_amendment else 'historical GPU parity admission',
        amendment_sha256=digest(HERE/'CPU-ONLY-AMENDMENT.md') if args.cpu_only_amendment else None,
        prior_compute_seconds=prior_seconds,
        execution_route='batch1 direct32 prefix, CPU float32; historical GPU batch2 with prefixes1,2,4,8,16,32',
        historical_gpu_environment={k:saved['receipt'].get(k) for k in ('torch','cuda','batch_size','precision')},
        backend_difference_caveat='device, library version, batchshape and prefix route differ; their separate effects are not identified')
    write(out/'result.json',result)
    def timeout(*_):raise TimeoutError('prospective300secondcomputeceiling')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(int(300-prior_seconds))
    began=time.monotonic();planned_frames=1
    try:
        with torch.no_grad():
            for index in range(2):
                if index>=planned_frames:break
                left,right,target,valid,_=dataset[index]
                left,right,target,valid=[t[None] for t in (left,right,target,valid)]
                label=cohort[index];key=(label['scene'],label['frame']);condition_rows=[]
                support_hash=hashlib.sha256(target[0].contiguous().numpy().tobytes()+valid[0].contiguous().numpy().tobytes()).hexdigest()
                assert support_hash==official[key]['target_support_sha256'], 'target/support hash mismatch'
                for condition in CONDITIONS:
                    stats=[];started=time.monotonic()
                    with intervention(model,condition,stats):
                        prediction=driver.forward(model,left,right,32)
                    elapsed=time.monotonic()-started
                    metrics=probes.frame_metrics(prediction,target,valid)[0]
                    driver.reset_model(model)
                    assert state_digest(model)==initial_state and not torch.cuda.is_initialized()
                    assert len(stats)==1
                    row=dict(scene=key[0],frame=key[1],condition=condition,seconds=elapsed,metrics=metrics,
                        cost_row_statistics=stats[0],state_unchanged=True,target_support_sha256=support_hash)
                    if condition=='baseline':
                        full=metrics['full'];ref=official[key]
                        differences={m:full[m]-ref[m] for m in ('epe','bad1','bad3','bad5')}
                        gate=full['valid_pixels']==ref['valid_pixels'] and abs(differences['epe'])<=.005 and all(abs(differences[m])<=.05 for m in ('bad1','bad3','bad5'))
                        row['CPU_GPU_differences']=differences;row['CPU_GPU_gate_passed']=gate
                        if index==0:
                            # Budget choice uses timing alone, before control outcomes.
                            planned_frames=2 if elapsed*(len(CONDITIONS)+1)*2*1.25<300-prior_seconds-(time.monotonic()-began) else 1
                            result['baseline_timing_seconds']=elapsed;result['planned_frames_from_timing']=planned_frames
                        condition_rows.append(row)
                        result['per_frame'].append(row)
                        write(out/'result.json',result);write(compact/'summary.json',result)
                        print(json.dumps(row),flush=True)
                        if args.cpu_only_amendment:
                            repeat_stats=[];repeat_started=time.monotonic()
                            with intervention(model,'baseline',repeat_stats):
                                repeated=driver.forward(model,left,right,32)
                            repeat_seconds=time.monotonic()-repeat_started
                            driver.reset_model(model)
                            row['CPU_baseline_full_tensor_exact_repeat']=torch.equal(prediction,repeated)
                            row['CPU_baseline_repeat_max_difference']=float((prediction-repeated).abs().max())
                            row['CPU_baseline_repeat_seconds']=repeat_seconds
                            assert row['CPU_baseline_full_tensor_exact_repeat'] and state_digest(model)==initial_state, 'CPU exact repeat failed'
                            torch.save(repeated.cpu(),out/f'{key[0]}-{key[1]}-baseline-repeat.pt')
                            print(json.dumps(dict(scene=key[0],condition='CPU_exact_repeat_gate',passed=True,seconds=repeat_seconds)),flush=True)
                        elif not gate:
                            result['status']='STOPPED_CPU_GPU_BASELINE_MISMATCH';raise RuntimeError('frozenCPU/GPUbaselinegatefailed; no tolerance change or intervention interpretation')
                    else:
                        condition_rows.append(row);result['per_frame'].append(row)
                        print(json.dumps(row),flush=True)
                    torch.save(prediction.cpu(),out/f'{key[0]}-{key[1]}-{condition}.pt')
                    write(out/'result.json',result);write(compact/'summary.json',result)
                assert len(condition_rows)==len(CONDITIONS)
        result['status']='COMPLETED_CPU_DIAGNOSTIC';result['completed_frames']=planned_frames
    except Exception as error:
        result['error']=repr(error)
        if result['status']=='RUNNING_CPU_DIAGNOSTIC':result['status']='INCOMPLETE_CPU_DIAGNOSTIC'
    finally:
        signal.alarm(0)
        result['elapsed_compute_seconds']=time.monotonic()-began;result['finished_utc']=datetime.now(timezone.utc).isoformat()
        result['cumulative_compute_seconds']=prior_seconds+result['elapsed_compute_seconds']
        result['checkpoint_unchanged']=digest(checkpoint)==SHA;result['state_unchanged']=state_digest(model)==initial_state
        write(out/'result.json',result);write(compact/'summary.json',result)
    print(json.dumps(dict(status=result['status'],seconds=result['elapsed_compute_seconds'],rows=len(result['per_frame']))),flush=True)


if __name__=='__main__':main()
