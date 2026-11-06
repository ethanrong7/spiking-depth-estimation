"""Cached prediction pooling and tiny synthetic readout diagnostics; CPU only."""
import argparse
import csv
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '2'

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PROJECT = HERE.parents[1]
RAW = REPO / 'storage/experiment-runs/snn-depth-estimation'
CASES = [
    ('A', 'scene_0120', RAW/'trained-cost-filter-cpu-controls-v1/cpu-within-backend-20261104/scene_0120-0001-baseline.pt'),
    ('A', 'scene_0121', RAW/'trained-cost-filter-cpu-controls-v1/cpu-within-backend-20261104/scene_0121-0001-baseline.pt'),
    ('B', 'scene_0120', RAW/'success-mechanism-cpu-audit-v1/B-fixed-metric-shape/scene_0120-baseline.pt'),
    ('B', 'scene_0123', RAW/'success-mechanism-cpu-audit-v1/B-fixed-metric-shape/scene_0123-baseline.pt'),
]
RADII = (1, 4, 8)
STRATA = ('all', 'inview_boundary', 'inview_interior', 'out_of_view', 'baseline_error_gt5')


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def shifted_slices(h, w, dy, dx):
    """Return destination/source slices for src(y+dy,x+dx), without wrapping."""
    dst = (slice(max(0, -dy), min(h, h-dy)), slice(max(0, -dx), min(w, w-dx)))
    src = (slice(max(0, dy), min(h, h+dy)), slice(max(0, dx), min(w, w+dx)))
    return dst, src


def pool(error, target, valid, radius, compatible):
    h, w = error.shape
    sums = [np.zeros((h, w), dtype=np.float64) for _ in range(4)]
    count, se, sg, se2 = sums
    for dy in range(-radius, radius+1):
        for dx in range(-radius, radius+1):
            dst, src = shifted_slices(h, w, dy, dx)
            ok = valid[src] & valid[dst]
            if compatible:
                ok = ok & (np.abs(target[src] - target[dst]) <= 1)
            e = error[src]
            count[dst] += ok
            se[dst] += np.where(ok, e, 0)
            sg[dst] += np.where(ok, target[src]-target[dst], 0)
            se2[dst] += np.where(ok, e*e, 0)
    assert np.all(count[valid] >= 1)
    n = np.maximum(count, 1)
    return count, se/n, sg/n, se2/(n*n)


def check_pool():
    y = np.array([[0., 1., 50.], [0., 1., 50.]])
    e = np.array([[1., -1., 8.], [3., -3., 10.]])
    v = np.ones((2, 3), bool)
    n, mu, g, q = pool(e, y, v, 1, False)
    assert n[0, 0] == 4 and mu[0, 0] == 0 and g[0, 0] == .5
    assert q[0, 0] == 20/16
    nc, muc, _, _ = pool(e, y, v, 1, True)
    assert nc[0, 1] == 4 and muc[0, 1] == 0
    assert nc[0, 2] == 2 and muc[0, 2] == 9
    return True


def summarize(mask, e, n, mu, g, q):
    x, nn, mm, gg, qq = e[mask], n[mask], mu[mask], g[mask], q[mask]
    z = mm + gg
    m2, g2, cross, qmean = (float(np.mean(a)) for a in (mm*mm, gg*gg, 2*mm*gg, qq))
    ordinary_rmse2 = float(np.mean(z*z))
    assert abs(ordinary_rmse2-(m2+g2+cross)) < 1e-8 * max(1, ordinary_rmse2)
    return dict(pixels=int(mask.sum()), neighbors_mean=float(nn.mean()), neighbors_min=int(nn.min()), neighbors_max=int(nn.max()),
        residual_bias=float(mm.mean()), residual_variance=float(np.var(mm)), geometry_bias=float(gg.mean()), geometry_variance=float(np.var(gg)),
        original_epe=float(np.abs(x).mean()), original_rmse=float(np.sqrt(np.mean(x*x))),
        ordinary_epe=float(np.abs(z).mean()), ordinary_rmse=float(np.sqrt(ordinary_rmse2)),
        detrended_epe=float(np.abs(mm).mean()), detrended_rmse=float(np.sqrt(m2)),
        ordinary_fraction_improved=float(np.mean(np.abs(z)<np.abs(x))), ordinary_fraction_worsened=float(np.mean(np.abs(z)>np.abs(x))),
        detrended_fraction_improved=float(np.mean(np.abs(mm)<np.abs(x))), detrended_fraction_worsened=float(np.mean(np.abs(mm)>np.abs(x))),
        mean_mu2=m2, mean_g2=g2, mean_2mug=cross, mean_q=qmean, mean_mu2_minus_q=m2-qmean,
        coherence_ratio=m2/qmean if qmean else None)


def cached_analysis(data_root):
    manifest = json.loads((PROJECT/'experiments/spiking-transformer-stereo/configs/manifest.json').read_text())
    old = json.loads((PROJECT/'experiments/offline-direction-audit-v1/results/selected-pixel-analysis.json').read_text())
    rows, missing, input_records = [], [], []
    for arm, scene, path in CASES:
        source = next(x for x in manifest['dev'] if x['scene']==scene and x['frame']=='0001')
        depth_path = data_root/source['depth']
        if not path.exists() or not depth_path.exists():
            missing.append(dict(model=arm, scene=scene, prediction_exists=path.exists(), depth_exists=depth_path.exists()))
            continue
        depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        assert depth is not None and depth.ndim == 3
        depth = depth[:,:,0].astype(np.float64)
        valid = np.isfinite(depth) & (depth>0)
        y = np.zeros_like(depth)
        y[valid] = -(289.740625*.08)/depth[valid]
        valid &= np.abs(y)<512
        pred = torch.load(path, map_location='cpu', weights_only=True)
        assert isinstance(pred, torch.Tensor) and pred.numel()==y.size
        p = pred.numpy().reshape(y.shape).astype(np.float64)
        assert np.isfinite(p[valid]).all()
        e = p-y
        x = np.arange(y.shape[1])[None,:]
        inview = valid & (x+y>=0) & (x+y<=y.shape[1]-1)
        boundary = np.zeros_like(valid)
        dh = (np.abs(np.diff(y,axis=1))>3)&valid[:,1:]&valid[:,:-1]
        dv = (np.abs(np.diff(y,axis=0))>3)&valid[1:]&valid[:-1]
        boundary[:,1:] |= dh; boundary[:,:-1] |= dh
        boundary[1:] |= dv; boundary[:-1] |= dv
        boundary = cv2.dilate(boundary.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)&valid
        masks = {'all':valid, 'inview_boundary':inview&boundary, 'inview_interior':inview&~boundary,
                 'out_of_view':valid&~inview, 'baseline_error_gt5':valid&(np.abs(e)>5)}
        assert np.array_equal(masks['inview_boundary'] | masks['inview_interior'] | masks['out_of_view'], valid)
        assert not (masks['inview_boundary'] & masks['inview_interior']).any()
        assert not (masks['inview_boundary'] & masks['out_of_view']).any()
        assert not (masks['inview_interior'] & masks['out_of_view']).any()
        prior = next(x for x in old['rows'] if x['model']==arm and x['scene']==scene)['conditions']['baseline']['all']
        assert abs(float(np.abs(e[valid]).mean())-prior['epe'])<1e-5
        assert int(valid.sum())==prior['pixels']
        input_records.append(dict(model=arm, scene=scene, prediction=str(path.relative_to(REPO)), prediction_bytes=path.stat().st_size,
                                  depth=str(depth_path.relative_to(REPO)), depth_bytes=depth_path.stat().st_size,
                                  valid_pixels=int(valid.sum()), baseline_epe=float(np.abs(e[valid]).mean())))
        for radius in RADII:
            for neighborhood in ('all_valid','gt_compatible'):
                n, mu, g, q = pool(e,y,valid,radius,neighborhood=='gt_compatible')
                for stratum in STRATA:
                    if masks[stratum].any():
                        rows.append(dict(model=arm, scene=scene, frame='0001', radius=radius, neighborhood=neighborhood,
                                         stratum=stratum, **summarize(masks[stratum],e,n,mu,g,q)))
    return rows, missing, input_records


def gaussian(d, center, sigma):
    a = np.exp(-.5*((d-center)/sigma)**2)
    return a/a.sum()


def lowest_index_argmax(w):
    # Treat numerically symmetric masses as ties without depending on platform rounding.
    return int(np.flatnonzero(np.isclose(w, w.max(), atol=1e-15, rtol=0))[0])


def readout_analysis():
    d=np.arange(-8,9,dtype=np.float64)
    rows=[]
    for u in (0.,.25,.5,.75):
        for delta in (2,4):
            for sigma in (.35,.75):
                for alpha in (0.,.10,.25,.49,.50,.51,.75):
                    for temperature in (.5,1.,2.):
                        p=(1-alpha)*gaussian(d,u,sigma)+alpha*gaussian(d,u+delta,sigma)
                        logits=np.log(p)/temperature
                        w=np.exp(logits-logits.max());w/=w.sum()
                        assert abs(w.sum()-1)<1e-13 and np.all(w>0)
                        idx=lowest_index_argmax(w); winner=float(d[idx])
                        local=np.abs(d-winner)<=1
                        assert local.sum() in (2,3)
                        estimates={'expectation':float(np.dot(w,d)), 'argmax':winner,
                                   'local_expectation':float(np.dot(w[local],d[local])/w[local].sum())}
                        category=('unimodal' if alpha==0 else 'true_dominant' if alpha<.5 else 'balanced' if alpha==.5 else 'wrong_dominant')
                        for method, estimate in estimates.items():
                            near=min(abs(estimate-u),abs(estimate-(u+delta))) if alpha>0 else abs(estimate-u)
                            tied=(abs(abs(estimate-(u+delta))-abs(estimate-u))<1e-12) if alpha>0 else False
                            distractor=(not tied and abs(estimate-(u+delta))<abs(estimate-u)) if alpha>0 else False
                            rows.append(dict(u=u, delta=delta, sigma=sigma, alpha=alpha, temperature=temperature,
                                             category=category, subpixel=u!=0, selected_argmax=winner, method=method,
                                             estimate=estimate, absolute_error=abs(estimate-u), nearest_active_center_distance=near,
                                             closer_to_distractor=bool(distractor), center_tie=bool(tied), mass_sum=float(w.sum()),
                                             duplicate_unimodal=bool(alpha==0 and delta==4)))
    assert len(rows)==336*3
    assert lowest_index_argmax(np.array([.5,.5]))==0
    unique=[r for r in rows if not r['duplicate_unimodal']]
    partitions=[]
    for category in ('unimodal','true_dominant','balanced','wrong_dominant'):
        for subpixel in (False,True):
            for method in ('expectation','argmax','local_expectation'):
                subset=[r for r in unique if r['category']==category and r['subpixel']==subpixel and r['method']==method]
                partitions.append(dict(category=category, subpixel=subpixel, method=method, rows=len(subset),
                                       mean_absolute_error=float(np.mean([r['absolute_error'] for r in subset])),
                                       distractor_fraction=float(np.mean([r['closer_to_distractor'] for r in subset]))))
    return rows,partitions


def csv_write(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-root',type=Path,default=REPO/'data/Synthetic')
    args=parser.parse_args()
    torch.set_num_threads(2)
    assert check_pool()
    HERE.joinpath('results').mkdir(exist_ok=True)
    spatial,missing,inputs=cached_analysis(args.data_root)
    modes,partitions=readout_analysis()
    csv_write(HERE/'results/spatial.csv',spatial)
    csv_write(HERE/'results/readout-grid.csv',modes)
    write_json(HERE/'results/readout-partitions.json',partitions)
    git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip())
    write_json(HERE/'results/provenance.json',dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),git_commit=git_head,
        dirty_worktree=dirty,command='PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" scratch/spiking-attention-unit-venv/bin/python projects/snn-depth-estimation/experiments/context-readout-cpu-v1/analyze.py',
        python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,opencv=cv2.__version__,seed=None,
        deterministic=True,threads=2,split='Synthetic Dev; selected frame 0001',data_root=str(args.data_root.relative_to(REPO)),
        radii=RADII,readout_grid=dict(u=[0,.25,.5,.75],delta=[2,4],sigma=[.35,.75],alpha=[0,.10,.25,.49,.50,.51,.75],temperature=[.5,1,2],candidates=list(range(-8,9))),
        inputs=inputs,missing=missing,spatial_rows=len(spatial),readout_distributions=336,readout_rows=len(modes),
        checks=['four baseline EPE parity','valid/disjoint spatial strata','hand-computable no-wrap pool','squared-error decomposition','normalized readout masses','lowest-index argmax tie']))


if __name__=='__main__': main()
