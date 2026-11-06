"""Analyze saved scalars/predictions only; never instantiate or run a model."""
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MPLCONFIGDIR'] = '/tmp/offline-direction-matplotlib'
import ast
import csv
import json
import math
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PROJECT = HERE.parents[1]
RAW = REPO / 'storage/experiment-runs/snn-depth-estimation'
STEREO = RAW / 'spiking-transformer-stereo'
OUT = HERE / 'results'
PREFIXES = ['1', '2', '4', '8', '16', '32']
STRATA = ['full', 'inview', 'oov']


def read(path):
    return json.loads(path.read_text())


def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def scene_values(frames, prefix, stratum):
    groups = {}
    for f in frames:
        groups.setdefault(f['scene'], []).append(f['iterations'][prefix][stratum]['epe'])
    assert len(groups) == 30 and all(len(x) == 2 for x in groups.values())
    return {s: st.mean(v) for s, v in groups.items()}


def scalar_analysis():
    models = {}
    rows = []
    inputs = []
    for arm in ['Original', 'A', 'B', 'C']:
        path = STEREO / f'spiking-attention-placement-v1/diagnostics/{arm}-30000-probes.json'
        d = read(path)
        assert d['parity']['passed'] and d['update'] == 30000 and len(d['per_frame']) == 60
        inputs.append(str(path.relative_to(REPO)))
        models[arm] = {t: {s: scene_values(d['per_frame'], t, s) for s in STRATA} for t in PREFIXES}
        for t in PREFIXES:
            for s in STRATA:
                vals = models[arm][t][s]
                assert abs(st.mean(vals.values()) - d['summary']['refinement'][t][s]['mean_scene_epe']) < 1e-6
                for scene, value in vals.items():
                    rows.append(dict(model=arm, prefix=int(t), stratum=s, scene=scene, epe=value))
    with (OUT / 'scene-refinements.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    results = {}
    for arm, trajectory in models.items():
        results[arm] = {}
        for s in STRATA:
            means = {t: st.mean(trajectory[t][s].values()) for t in PREFIXES}
            late = {scene: trajectory['32'][s][scene] - trajectory['16'][s][scene] for scene in trajectory['32'][s]}
            oracle = {scene: min(trajectory[t][s][scene] for t in PREFIXES) for scene in late}
            best = {scene: min(PREFIXES, key=lambda t: trajectory[t][s][scene]) for scene in late}
            positive = sum(max(v, 0) for v in late.values())
            sorted_late = sorted(late.items(), key=lambda x: x[1], reverse=True)
            results[arm][s] = dict(
                prefix_means=means,
                late_16_to_32=dict(mean=st.mean(late.values()), median=st.median(late.values()),
                    worsened_scenes=sum(v > 0 for v in late.values()), improved_scenes=sum(v < 0 for v in late.values()),
                    largest_deteriorations=sorted_late[:5], largest_improvements=sorted_late[-5:],
                    positive_damage_top2_share=sum(max(v, 0) for _, v in sorted_late[:2])/positive if positive else None,
                    scene_deltas=late),
                best_fixed_prefix=min(means, key=means.get),
                gt_selected_scene_prefix_epe=st.mean(oracle.values()),
                gt_selected_scene_prefix_gain_vs32=means['32']-st.mean(oracle.values()),
                gt_selected_scene_prefix_counts={t: list(best.values()).count(t) for t in PREFIXES})
    # A single full-mask oracle chooses the prefix per scene for all its strata.
    coherent = {}
    for arm, tr in models.items():
        choice = {scene: min(PREFIXES, key=lambda t: tr[t]['full'][scene]) for scene in tr['32']['full']}
        coherent[arm] = {s: st.mean(tr[choice[scene]][s][scene] for scene in choice) for s in STRATA}
    save('refinement-analysis.json', dict(inputs=inputs, models=results,
        full_mask_scene_oracle_shared_prefix=coherent,
        caveat='GT-selected scene/prefix oracle; not per-pixel stopping or an achievable trained model.'))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), layout='constrained')
    colors = dict(Original='#555555', A='#4076b5', B='#24917c', C='#c14e58')
    for ax, s in zip(axes, STRATA):
        for arm in models:
            means = results[arm][s]['prefix_means']
            ax.plot([4,8,16,32], [means[t] for t in ['4','8','16','32']], marker='o', label=arm, color=colors[arm])
        ax.set_title(dict(full='All valid pixels', inview='In-view coordinates', oov='Out-of-view coordinates')[s])
        ax.set_xlabel('Evaluation refinements'); ax.set_ylabel('Mean-scene EPE (px)')
        ax.set_xticks([4,8,16,32]); ax.grid(alpha=.2)
    axes[0].legend(frameon=False)
    fig.suptitle('Existing 30k checkpoints, seed 1234 — no new inference')
    fig.savefig(OUT / 'refinement-trajectories.png', dpi=160)
    plt.close(fig)
    return results


def history_analysis():
    logs = [json.loads(x) for x in (STEREO/'candidate-selection-pilot-v1/train/evaluations.jsonl').read_text().splitlines()]
    final = next(x for x in logs if x['update'] == 1000)
    modes = final['metrics_by_mode']
    global_fraction = (modes['top1']['candidate_epe']-modes['spatial']['candidate_epe'])/(modes['top1']['candidate_epe']-modes['oracle']['candidate_epe'])
    gates = {n: read(PROJECT/f'experiments/{n}/results/paired-gate-2500.json') for n in ['coordinate-transformer-v1','coordinate-transformer-e3']}
    hull = read(STEREO/'coordinate-transformer-v1/diagnostics/transformer-2500-hull-sweep.json')
    assert hull['official_parity']
    support = {}
    for radius in ['8','16','32']:
        support[radius] = {s: {k:v for k,v in d.items() if k in ['hull_bound_epe','actual_epe','near4_fraction','scoring_slack']} for s,d in hull['totals']['3'][radius]['coarse'].items()}
    save('historical-headroom.json', dict(
        candidate_pilot=dict(updates=1000, backbone='frozen compact S2M2 v2b; not native SpikeStereoNet',
            modes=modes, ratio_of_mean_candidate_gap_captured=global_fraction,
            note='Saved gap_captured averages per-frame ratios; the ratio of means above is a distinct statistic. GT oracle selects nearest literal candidate before a frozen decoder; not globally optimal final output.'),
        coordinate_e2={k:gates['coordinate-transformer-v1'][k] for k in ['update','decision','transformer_epe','pointwise_epe','original_epe','wins_vs_pointwise']},
        coordinate_e3={k:gates['coordinate-transformer-e3'][k] for k in ['update','decision','challenger_epe','archived_epe','original_epe']},
        e2_final_coarse_support=support,
        note='Expanded support evaluated geometrically with frozen centers. Hull bound is reachability under arbitrary convex weights, not evidence that matching identifies the target.'))


def pixel_analysis():
    import numpy as np
    import torch
    import cv2
    torch.set_num_threads(2)
    manifest = read(PROJECT/'experiments/spiking-transformer-stereo/configs/manifest.json')
    panels = [
        ('A','scene_0120',RAW/'trained-cost-filter-cpu-controls-v1/cpu-within-backend-20261104'),
        ('A','scene_0121',RAW/'trained-cost-filter-cpu-controls-v1/cpu-within-backend-20261104'),
        ('B','scene_0120',RAW/'success-mechanism-cpu-audit-v1/B-fixed-metric-shape'),
        ('B','scene_0123',RAW/'success-mechanism-cpu-audit-v1/B-fixed-metric-shape')]
    all_rows=[]
    for arm,scene,path in panels:
        row=next(r for r in manifest['dev'] if r['scene']==scene and r['frame']=='0001')
        depth=cv2.imread(str(REPO/'data/Synthetic'/row['depth']),cv2.IMREAD_ANYCOLOR|cv2.IMREAD_ANYDEPTH)[:,:,0].astype(np.float32)
        valid=np.isfinite(depth)&(depth>0)
        target=np.zeros_like(depth);target[valid]=-(289.740625*.08)/depth[valid];valid &= np.abs(target)<512
        def load(condition, folder=path):
            filename=f'{scene}-0001-{condition}.pt' if arm=='A' else f'{scene}-{condition}.pt'
            x=torch.load(folder/filename,map_location='cpu',weights_only=True)
            assert isinstance(x,torch.Tensor) and x.numel()==target.size
            return x.numpy().reshape(target.shape)
        predictions={c:load(c) for c in ['baseline','bypass','raw_match_Y_moments']}
        if arm=='B':
            split=RAW/'success-mechanism-cpu-audit-v1/B-calibration-split'
            assert np.array_equal(predictions['baseline'],load('baseline',split))
            predictions.update({c:load(c,split) for c in ['shift_only','scale_only']})
        else: predictions['smooth5_match_Y_moments']=load('smooth5_match_Y_moments')
        xx=np.arange(target.shape[1])[None,:]
        inview=valid&(xx+target>=0)&(xx+target<=target.shape[1]-1)
        boundary=np.zeros_like(valid)
        dh=(np.abs(np.diff(target,axis=1))>3)&valid[:,1:]&valid[:,:-1]
        dv=(np.abs(np.diff(target,axis=0))>3)&valid[1:]&valid[:-1]
        boundary[:,1:]|=dh;boundary[:,:-1]|=dh;boundary[1:]|=dv;boundary[:-1]|=dv
        boundary=cv2.dilate(boundary.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)&valid
        error={c:np.abs(pred-target) for c,pred in predictions.items()}
        strata={'all':valid,'inview_boundary':inview&boundary,'inview_interior':inview&~boundary,'out_of_view':valid&~inview}
        conditions={}
        for condition,e in error.items():
            conditions[condition]={s:dict(pixels=int(mask.sum()),epe=float(e[mask].mean(dtype=np.float64)),
                bad3=float((e[mask]>3).mean()),bad5=float((e[mask]>5).mean()),
                excess5_mean=float(np.maximum(e[mask]-5,0).mean(dtype=np.float64))) if mask.any() else None for s,mask in strata.items()}
        # Paired predictions reveal whether recovery occurs at the same locations.
        damaged=valid&(error['bypass']>error['baseline']+1)
        recovery={}
        for c in predictions:
            if c in ['baseline','bypass']:continue
            den=(error['bypass'][damaged]-error['baseline'][damaged]).sum(dtype=np.float64)
            num=(error['bypass'][damaged]-error[c][damaged]).sum(dtype=np.float64)
            recovery[c]=dict(damaged_pixel_count=int(damaged.sum()),damage_recovered_fraction=float(num/den) if den else None,
                restored_within1px_of_baseline_error=float((error[c][damaged]<=error['baseline'][damaged]+1).mean()) if damaged.any() else None,
                prediction_mae_to_baseline=float(np.abs(predictions[c]-predictions['baseline'])[valid].mean(dtype=np.float64)))
        donor=np.zeros_like(valid)
        donor_by_radius={}
        # Exploratory scale extension: native radius 4 is one encoder cell.
        for radius in [1,4,8]:
          for dy in range(-radius,radius+1):
            for dx in range(-radius,radius+1):
                if not(dx or dy):continue
                shifted_gt=np.roll(target,(dy,dx),(0,1));shifted_good=np.roll(valid&(error['baseline']<=1),(dy,dx),(0,1))
                ok=valid&shifted_good&(np.abs(shifted_gt-target)<=1)
                if dy>0:ok[:dy]=False
                if dy<0:ok[dy:]=False
                if dx>0:ok[:,:dx]=False
                if dx<0:ok[:,dx:]=False
                donor |= ok
          donor_by_radius[str(radius)]={s:dict(large_error_pixels=int((mask&(error['baseline']>5)).sum()),
            fraction_with_gt_compatible_accurate_neighbor=float(donor[mask&(error['baseline']>5)].mean()) if (mask&(error['baseline']>5)).any() else None) for s,mask in strata.items()}
        all_rows.append(dict(model=arm,scene=scene,frame='0001',conditions=conditions,recovery_on_bypass_damaged_pixels=recovery,gt_assisted_neighbor_diagnostic=donor_by_radius,
            donor_scope='GT-assisted native-pixel Chebyshev radii 1,4,8. Radii 4/8 are exploratory extensions. No claim about matching-feature information.'))
        if arm=='B' and scene=='scene_0123':
            import matplotlib.pyplot as plt
            fig,axes=plt.subplots(1,4,figsize=(13,3.1),layout='constrained')
            gt_im=axes[0].imshow(-target,cmap='viridis');axes[0].set_title('GT disparity')
            fig.colorbar(gt_im,ax=axes[0],label='Disparity (px)',shrink=.75)
            for ax,c,title in zip(axes[1:],['baseline','bypass','shift_only'],['B baseline error','Branch bypass error','Mean restored error']):
                im=ax.imshow(error[c],cmap='magma',vmin=0,vmax=10);ax.set_title(title)
            for ax in axes:ax.axis('off')
            fig.colorbar(im,ax=list(axes[1:]),label='Absolute error (px), clipped at 10',shrink=.75)
            fig.suptitle('Saved B predictions: scene 0123 / frame 0001 — selected example')
            fig.savefig(OUT/'selected-case-errors.png',dpi=160);plt.close(fig)
    save('selected-pixel-analysis.json',dict(scope='Four previously selected A/B cases, three unique frames; not population estimates or C trajectories.',rows=all_rows))


def verify_saved_analysis():
    records=read(OUT/'selected-pixel-analysis.json')['rows']
    references={
        'A':read(PROJECT/'experiments/trained-cost-filter-cpu-controls-v1/results/cpu-within-backend-20261104/summary.json')['per_frame'],
        'B':read(PROJECT/'experiments/success-mechanism-cpu-audit-v1/results/B.json')['rows'] + read(PROJECT/'experiments/success-mechanism-cpu-audit-v1/results/B-calibration-split.json')['rows']}
    maximum=0.;checks=0
    for r in records:
        for condition, strata in r['conditions'].items():
            old=next(x for x in references[r['model']] if x['scene']==r['scene'] and x['condition']==condition)
            m=old['metrics']['full'];new=strata['all']
            assert m['valid_pixels']==new['pixels']==sum(strata[s]['pixels'] for s in ['inview_boundary','inview_interior','out_of_view'])
            difference=abs(m['epe']-new['epe']);maximum=max(maximum,difference)
            assert difference<1e-5
            assert abs(m['bad3']/100-new['bad3'])<1e-6
            checks+=1
        for s in ['all','inview_boundary','inview_interior','out_of_view']:
            v=[r['gt_assisted_neighbor_diagnostic'][str(k)][s]['fraction_with_gt_compatible_accurate_neighbor'] for k in [1,4,8]]
            assert all(x is None for x in v) or v[0]<=v[1]<=v[2]
    save('verification.json',dict(saved_prediction_conditions_reconciled=checks,max_epe_difference_vs_saved_metrics=maximum,
        frame_to_scene_reductions_reconciled=True,disjoint_pixel_counts_reconciled=True,donor_radius_monotonicity=True,
        operator_checks='passed',new_model_forwards=0,optimizer_steps=0,gpu_used=False))


def tiny_operator_checks():
    import torch
    # Extract the existing pure code operator without importing the model stack.
    source=PROJECT/'core/epipolar_spiking_stereo.py'
    tree=ast.parse(source.read_text())
    selected=[x for x in tree.body if isinstance(x,(ast.ClassDef,ast.FunctionDef)) and x.name in ['Threshold','signed_rate']]
    ns={'torch':torch};exec(compile(ast.Module(body=selected,type_ignores=[]),str(source),'exec'),ns)
    signed=ns['signed_rate']
    x=torch.tensor([.60,.65],dtype=torch.float64,requires_grad=True)
    rates=signed(x,4);rates.sum().backward()
    assert rates[0]==rates[1] and x.grad.abs().min()>0
    q=torch.ones(16,dtype=torch.float64)
    k=x.detach()[:,None].repeat(1,16)
    continuous_logits=(k*q).sum(-1)/4
    spike_logits=(signed(k,4)*signed(q,4)).sum(-1)/4
    assert continuous_logits[1]>continuous_logits[0] and spike_logits[0]==spike_logits[1]
    # Readout facts independent of any learning claim.
    candidates=torch.tensor([0.,8.]);weights=torch.tensor([.5,.5]);estimate=(candidates*weights).sum()
    target=torch.tensor(4.)
    assert estimate==target and (candidates-target).abs().min()==4
    logits=torch.tensor([1.,2.,3.]);assert torch.allclose(logits.softmax(0),(logits+5).softmax(0))
    assert float(torch.tensor([0.,0.]).softmax(0)@torch.tensor([0.,10.]))==5
    assert abs(float(torch.zeros(3).softmax(0)@torch.tensor([0.,10.,10.]))-20/3)<1e-6
    save('tiny-operator-checks.json',dict(
        status='passed',scope='Tiny CPU algebra/gradient checks only; no trained-layer activation measurements.',
        signed_code=dict(currents=x.detach().tolist(),rates=rates.detach().tolist(),surrogate_gradients=x.grad.tolist(),continuous_logits=continuous_logits.tolist(),spike_logits=spike_logits.tolist()),
        hull_example=dict(candidates=[0,8],target=4,mean_readout=4,nearest_candidate_distance=4),
        duplicate_example=dict(unique_candidates=[0,10],unique_mean=5,duplicated_candidates=[0,10,10],duplicated_mean=20/3),
        row_shift_softmax_invariance=True,
        conclusions=['A finite code can erase local score differences while surrogate gradients remain nonzero; learned projections may separate them.',
            'Convex-hull reachability does not prove a candidate samples the true correspondence.',
            'Duplicate candidates change softmax mass unless deduplicated or appropriately weighted.',
            'Constant row shifts cancel in softmax; a downstream decoder consuming absolute costs need not share this invariance.']))


if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    r=scalar_analysis();history_analysis();pixel_analysis();tiny_operator_checks();verify_saved_analysis()
    print(json.dumps({a:{s:{'mean_late':r[a][s]['late_16_to_32']['mean'],'worsened_scenes':r[a][s]['late_16_to_32']['worsened_scenes'],'oracle_epe':r[a][s]['gt_selected_scene_prefix_epe']} for s in STRATA} for a in r},indent=2))
