"""New local CPU analysis of saved trained-A GPU records; no new inference."""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '2'
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parents[1]
PREFIXES = (1, 2, 4, 8, 16, 32)
METRICS = ('epe', 'bad1', 'bad3', 'bad5')
SOURCE_ID = 'ac37454c00b65b324ecab02fa0f55aeb0d0e1791413cab9a8b3351c143a15a50'
A_CHECKPOINT = 'd5dad669ba1ba521f36f7ede66142f163afca896cf677f7586a4f4b53829031b'
ORIGINAL_CHECKPOINT = '81ca0f01b0b6da6cc5159eac517447618c24fbeec8f9c295cf94da890a09483e'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def keyed(rows):
    result = {(r['scene'], r['frame']): r for r in rows}
    assert len(rows) == len(result) == 60
    assert len({s for s,_ in result}) == 30
    assert all(sum(s==scene for s,_ in result)==2 for scene,_ in result)
    return result


def mean_scene(rows, scene, function):
    values = [function(v) for (s,_),v in rows.items() if s==scene]
    assert len(values)==2
    return float(np.mean(values))


def describe(values, scenes, draws):
    values = np.asarray(values, dtype=np.float64)
    total = values.sum()
    top = np.argsort(-np.abs(values), kind='stable')[:3]
    loo = (total-values)/(len(values)-1)
    return dict(mean=float(values.mean()), median=float(np.median(values)),
        positive_scenes=int((values>0).sum()), zero_scenes=int((values==0).sum()), negative_scenes=int((values<0).sum()),
        descriptive_scene_bootstrap_95_percent_interval=np.percentile(values[draws].mean(1), [2.5,97.5]).tolist(),
        three_largest_absolute_contributions=[dict(scene=scenes[i],value=float(values[i])) for i in top],
        three_largest_signed_share_of_net=float(values[top].sum()/total) if abs(total)>1e-12 else None,
        three_largest_share_of_absolute_mass=float(np.abs(values[top]).sum()/np.abs(values).sum()) if np.abs(values).sum() else None,
        leave_one_scene_out_mean_range=[float(loo.min()),float(loo.max())],
        leave_one_scene_out_means={s:float(v) for s,v in zip(scenes,loo)})


def correlation(x, y):
    if np.ptp(x)==0 or np.ptp(y)==0:
        return dict(spearman_rho=None,scenes=len(x),reason='constant scene vector')
    return dict(spearman_rho=float(spearmanr(x,y).statistic),scenes=len(x),
        interpretation='descriptive scene association, no pixel independence or causal mediation claim')


def analyze(raw_root):
    started=time.monotonic()
    assert (HERE/'STUDY.md').exists(), 'scientific specification must precede analysis'
    paths=dict(A_replay=raw_root/'spiking-attention-placement-phase2/diagnostics/seed-1234/SC/replay-30000.json',
        Original_replay=raw_root/'spiking-attention-placement-phase2/diagnostics-r2/seed-1234/Original/replay-30000.json',
        A_frozen_interventions=raw_root/'spiking-attention-placement-phase2/frozen-ablation-v2/A/update-30000/ablation.json')
    data={name:json.loads(p.read_text()) for name,p in paths.items()}
    a,original,ablation=data['A_replay'],data['Original_replay'],data['A_frozen_interventions']
    for replay,expected in [(a,A_CHECKPOINT),(original,ORIGINAL_CHECKPOINT)]:
        assert replay['seed']==1234 and replay['update']==30000 and replay['samples_seen']==240000
        assert replay['source_identity']==SOURCE_ID and replay['checkpoint_sha256']==expected
        assert replay['parity']['passed'] and replay['parity']['all_official_frame_metrics_exact']
    assert ablation['checkpoint_unchanged'] and ablation['receipt']['checkpoint_sha256']==A_CHECKPOINT
    assert ablation['receipt']['updates']==30000 and ablation['receipt']['seed']==1234
    assert ablation['receipt']['source_identity']==a['checkpoint_source_identity']
    assert ablation['receipt']['manifest_sha256']==a['manifest_sha256']==original['manifest_sha256']
    ak,ok=keyed(a['per_frame']),keyed(original['per_frame'])
    conditions=ablation['conditions']
    assert set(conditions)=={'baseline','branch_off','uniform_weights','code_tanh'}
    ck={name:keyed(value['per_frame']) for name,value in conditions.items()}
    scenes=sorted({s for s,_ in ak})
    for key,row in ak.items():
        assert key in ok
        assert (row['target_support_sha256'],row['fine_cohort_sha256'])==(ok[key]['target_support_sha256'],ok[key]['fine_cohort_sha256'])
        for field in ('pixel_count','valid_fine_count','excluded_oov_count','candidate_count'):
            assert row['rank']['pre'][field]==row['rank']['post'][field]
        for position in ('pre','post'):
            rank=row['rank'][position]
            assert abs(rank['mean_rank']-rank['rank_sum']/rank['pixel_count'])<1e-12
            assert abs(rank['mean_argmax_error_cells']-rank['argmax_error_cells_sum']/rank['pixel_count'])<1e-12
        for prefix in PREFIXES:
            for stratum in ('full','inview','oov'):
                assert row['iterations'][str(prefix)][stratum]['valid_pixels']==ok[key]['iterations'][str(prefix)][stratum]['valid_pixels']
        for name,condition in conditions.items():
            assert condition['parameters_unchanged'] and condition['state_sha256']==conditions['baseline']['state_sha256']
            assert condition['receipt']['checkpoint_sha256']==A_CHECKPOINT
            frame=ck[name][key]
            assert frame['target_support_sha256']==row['target_support_sha256']
            assert frame['valid_pixels']==row['iterations']['32']['full']['valid_pixels']
        for metric in METRICS:
            assert ck['baseline'][key][metric]==row['iterations']['32']['full'][metric]
    assert conditions['baseline']['official_parity']['status']=='OFFICIAL_PARITY_PASSED'
    assert conditions['baseline']['official_parity']['all_frame_metrics_exact']
    vectors={name:[] for name in ('g','b','u','t','r','q','e')}
    per_scene=[]
    for scene in scenes:
        row={'scene':scene}
        row['g']=mean_scene(ok,scene,lambda v:v['iterations']['32']['full']['epe'])-mean_scene(ak,scene,lambda v:v['iterations']['32']['full']['epe'])
        for label,condition in [('b','branch_off'),('u','uniform_weights'),('t','code_tanh')]:
            row[label]=mean_scene(ck[condition],scene,lambda v:v['epe'])-mean_scene(ck['baseline'],scene,lambda v:v['epe'])
        for label,metric,sign in [('r','mean_rank',-1),('q','top1_fraction',1),('e','mean_argmax_error_cells',-1)]:
            row[label]=mean_scene(ak,scene,lambda v:sign*(v['rank']['post'][metric]-v['rank']['pre'][metric]))
        for name in vectors:vectors[name].append(row[name])
        per_scene.append(row)
    draws=np.random.default_rng(20261104).integers(0,len(scenes),(10000,len(scenes)))
    descriptions={name:describe(values,scenes,draws) for name,values in vectors.items()}
    associations={x+'_versus_'+y:correlation(vectors[x],vectors[y]) for x,y in [('g','b'),('g','u'),('g','t'),('g','r'),('g','q'),('g','e'),('u','r')]}
    prefix={}
    for n in PREFIXES:
        av=[mean_scene(ak,s,lambda v:v['iterations'][str(n)]['full']['epe']) for s in scenes]
        ov=[mean_scene(ok,s,lambda v:v['iterations'][str(n)]['full']['epe']) for s in scenes]
        delta=np.asarray(ov)-av
        prefix[str(n)]=dict(A_mean_scene_epe=float(np.mean(av)),Original_mean_scene_epe=float(np.mean(ov)),
            A_advantage=describe(delta,scenes,draws))
    gap1=prefix['1']['A_advantage']['mean'];gap32=prefix['32']['A_advantage']['mean']
    rank_endpoints={name:{metric:float(np.mean([mean_scene(ak,s,lambda v:v['rank'][name][metric]) for s in scenes]))
                         for metric in ('mean_rank','top1_fraction','mean_argmax_error_cells')} for name in ('pre','post')}
    condition_metrics={name:{metric:float(np.mean([mean_scene(rows,s,lambda v:v[metric]) for s in scenes]))
                              for metric in METRICS} for name,rows in ck.items()}
    return dict(status='COMPLETED_CACHED_CPU_ANALYSIS',computed_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-started,
        execution='local CPU analysis of frozen GPU record JSON; zero new inference/training/data decoding/network calls',
        seed=1234,update=30000,frames=60,scenes=30,checkpoint_sha256=A_CHECKPOINT,
        source_identity=SOURCE_ID,ablation_checkpoint_source_identity=a['checkpoint_source_identity'],
        cached_record_identities={name:dict(path='RUN_ROOT/'+str(path.relative_to(raw_root)),sha256=digest(path),bytes=path.stat().st_size) for name,path in paths.items()},
        analysis_source_sha256=digest(Path(__file__)),study_sha256=digest(HERE/'STUDY.md'),
        audits=dict(all_60_target_hashes_and_counts_match=True,Original_A_fine_cohort_hashes_match=True,
            baseline_all_frame_metrics_exact_official=True,recorded_all_intervention_model_tensors_unchanged=True,
            note='fine-rank inview sites differ from full-resolution EPE support; paired by scene, not pixel'),
        definitions=dict(g='Original32-A32 EPEpx, positive A advantage',b='branch_off-baseline EPEpx',
            u='uniform_weights-baseline EPEpx',t='code_tanh-baseline EPEpx',r='pre-post GT competition rank; positive improves',
            q='post-pre GTtop1 fraction; positive improves',e='pre-post argmax error in featurecells; positive improves'),
        inference=dict(scope='exploratory descriptive reused one-seed Dev scenes',bootstrap_draws=10000,seed=20261104,
            unit='whole scene with both frames and all paired conditions retained',interval='uncorrected95%percentile',
            shared_term_caveat='g and intervention penalties share -A baseline; correlations can be algebraically induced',
            limits='rank associations do not identify mediation; no cached cost vectors/weights/pixel errors to measure profile calibration, lookup stability or severity split'),
        condition_metrics=condition_metrics,rank_endpoints=rank_endpoints,descriptions=descriptions,
        associations=associations,prefix=prefix,gap1_over_gap32=gap1/gap32 if abs(gap32)>1e-12 else None,
        scene_direction_tests=dict(raw_argmax_worsened_and_filter_bypass_hurt=sum(e<0 and b>0 for e,b in zip(vectors['e'],vectors['b'])),
            raw_argmax_worsened_and_uniform_weights_hurt=sum(e<0 and u>0 for e,u in zip(vectors['e'],vectors['u'])),
            scope='scene-level joined frozen records; full-EPE and fine-inview rank supports differ, not pixel causal mediation'),
        gap_ratio_interpretation='ratio of global gaps, not causal fraction; may exceed1 when later refinement narrows the gap',
        per_scene=per_scene)


def main():
    repo=HERE.parents[3]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root',type=Path,default=repo/'storage/experiment-runs/snn-depth-estimation/spiking-transformer-stereo')
    args=parser.parse_args()
    result=analyze(args.raw_root)
    out=HERE/'results';out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with (out/'per-scene.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(result['per_scene'][0]));writer.writeheader();writer.writerows(result['per_scene'])
    raw=repo/'storage/experiment-runs/snn-depth-estimation/trained-cost-filter-cached-cpu-v1'
    raw.mkdir(parents=True,exist_ok=True)
    (raw/'provenance.json').write_text(json.dumps(dict(command='scripts/analyze.py --raw-root configured RUN_ROOT',
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        dirty_status=subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).splitlines(),
        source_sha256=result['analysis_source_sha256'],cached_record_identities=result['cached_record_identities']),indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ('elapsed_seconds','condition_metrics','rank_endpoints','associations','gap1_over_gap32')},indent=2))


if __name__=='__main__':
    main()
