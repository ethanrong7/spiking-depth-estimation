"""Recompute saved-prediction metrics; no model forwards or GPU operations."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import torch

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
BASE=REPO/'storage/experiment-runs/snn-depth-estimation'
STEREO=BASE/'spiking-transformer-stereo'
RAW=BASE/'success-mechanism-cpu-audit-v1'
PROJECT=BASE/'trained-cost-filter-cpu-controls-v1/source/projects/snn-depth-estimation'
sys.path[:0]=[str(PROJECT),str(PROJECT/'core'),str(PROJECT/'experiments/spiking-transformer-stereo/scripts')]


def main():
    torch.set_num_threads(2)
    spec=importlib.util.spec_from_file_location('verification_driver',PROJECT/'experiments/spiking-transformer-stereo/scripts/run_experiment.py')
    driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
    manifest=json.loads((PROJECT/'experiments/spiking-transformer-stereo/configs/manifest.json').read_text())
    cohort=[next(r for r in manifest['dev'] if r['scene']==s and r['frame']=='0001') for s in ('scene_0120','scene_0123')]
    dataset=driver.ManifestDataset(REPO/'data/Synthetic',cohort)
    official=json.loads((STEREO/'spiking-attention-placement-v1/diagnostics/B-30000-probes.json').read_text())
    official={(r['scene'],r['frame']):r for r in official['per_frame']}
    panels=[('B.json','B-fixed-metric-shape'),('B-calibration-split.json','B-calibration-split')]
    results={f:json.loads((HERE/'results'/f).read_text()) for f,_ in panels}
    assert all(r['status']=='complete' and r['weights_unchanged'] and not r['cuda_initialized'] for r in results.values())
    verification={'saved_predictions_checked':0,'max_metric_difference':0.,'historical_GPU_drift':[], 'same_case_main_split_baseline_exact':True}
    for index,label in enumerate(cohort):
        _,_,native_target,native_valid,_=dataset[index]
        target,valid=native_target,native_valid
        valid=valid.bool();target=target.squeeze(0)
        support_hash=hashlib.sha256(native_target.contiguous().numpy().tobytes()+native_valid.contiguous().numpy().tobytes()).hexdigest()
        assert support_hash==official[(label['scene'],label['frame'])]['target_support_sha256']
        baseline=None
        for file,folder in panels:
            rows=[r for r in results[file]['rows'] if r['scene']==label['scene']]
            for row in rows:
                pred=torch.load(RAW/folder/f"{label['scene']}-{row['condition']}.pt",map_location='cpu',weights_only=True).squeeze()
                error=(pred-target).abs()[valid].double()
                actual={'epe':error.mean().item(),'bad3':(error>3).double().mean().item()*100,
                        'capped5':error.clamp(max=5).mean().item(),'excess5':(error-5).clamp(min=0).mean().item()}
                saved={'epe':row['metrics']['full']['epe'],'bad3':row['metrics']['full']['bad3'],
                       'capped5':row['capped5'],'excess5':row['excess5']}
                delta=max(abs(actual[k]-saved[k]) for k in actual)
                assert delta<1e-5
                assert error.numel()==row['metrics']['full']['valid_pixels']==100000
                verification['max_metric_difference']=max(verification['max_metric_difference'],delta)
                verification['saved_predictions_checked']+=1
                if row['condition']=='baseline':
                    assert row['exact_baseline_repeat']
                    if baseline is None:
                        baseline=pred
                        ref=official[(label['scene'],label['frame'])]['iterations']['32']['full']
                        verification['historical_GPU_drift'].append(dict(scene=label['scene'],
                            epe=saved['epe']-ref['epe'],bad3=saved['bad3']-ref['bad3']))
                    else:
                        assert torch.equal(baseline,pred)
    contrasts=[]
    for label in cohort:
        rows={r['condition']:r for r in results['B.json']['rows'] if r['scene']==label['scene']}
        rows.update({r['condition']:r for r in results['B-calibration-split.json']['rows'] if r['scene']==label['scene']})
        epe=lambda k:rows[k]['metrics']['full']['epe']
        damage=epe('bypass')-epe('baseline')
        contrasts.append(dict(scene=label['scene'],bypass_damage=damage,
            recovery={k:(epe('bypass')-epe(k))/damage for k in ('raw_match_Y_moments','shift_only','scale_only','uniform')},
            exact_code_tanh_penalty=epe('tanh')-epe('baseline'),
            bypass_excess5_share=(rows['bypass']['excess5']-rows['baseline']['excess5'])/damage))
    guide=json.loads((STEREO/'v0-guided-stereo-transformer-v1/diagnostics/C-30000-probes-summary.json').read_text())
    guided={'gate_tanh':guide['gate_tanh'],'conditions':{},'iteration32':{}}
    for name,c in guide['conditions'].items():
        guided['conditions'][name]={'epe':c['mean_scene_epe'],'bad3':c['mean_scene_bad3'],
            'scene0123':next(r for r in c['per_scene'] if r['scene']=='scene_0123')}
    anchor=guide['conditions']['canonical_native']['selected_iterations']['32']['valid']
    for k in ('branch_rms_feature_cells','native_rms_feature_cells','entropy_mean','remaining_abs_mean','branch_remaining_abs_error_mean'):
        guided['iteration32'][k]=anchor[k]
    total=sum(r['elapsed_seconds'] for r in results.values())+json.loads((HERE/'results/B-initial-metric-shape-failure.json').read_text())['elapsed_seconds']
    assert total<300
    summary=dict(verification=verification,B_contrasts=contrasts,guided_cached=guided,cumulative_B_forward_seconds=total,
                 scope='B two selected frames; guided cached full-Dev measurements; no new population inference')
    (HERE/'results/summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
