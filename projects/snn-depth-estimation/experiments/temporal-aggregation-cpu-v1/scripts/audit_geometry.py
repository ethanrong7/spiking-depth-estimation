"""Post-run bounded first-frame loader/geometry audit requested by root.

Canonical and wrong-sign are fixed diagnostics, not orientation selection. No
cohort, scoring definition or main-result orientation is changed by this audit.
"""
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

import cv2
import numpy as np
from patch_match import zncc, score_metrics
from run import decode, write_json, digest


def main():
    here = Path(__file__).resolve().parents[1]
    repo = here.parents[3]
    output = here/'results/native-dev-20261104'
    raw = repo/'storage/experiment-runs/snn-depth-estimation/temporal-aggregation-cpu-v1/native-dev-20261104'
    result = json.loads((output/'summary.json').read_text())
    item = json.loads((raw/'provenance.json').read_text())['cohort'][0]
    data = repo/result['config']['dataset_root']
    sys.path.insert(0, str(repo/'projects/snn-depth-estimation'))
    from core.utils.reader import get_block_spikes, read_depth, depth2disp
    paths = {k: data/item[k] for k in ('spike_l','spike_r','depth')}
    spikes = {k: decode(paths[k], (50,250,400)) for k in ('spike_l','spike_r')}
    decoder_parity = {k: bool(np.array_equal(v, get_block_spikes(str(paths[k])))) for k,v in spikes.items()}
    assert all(decoder_parity.values())
    exr = cv2.imread(str(paths['depth']), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    depth = exr[:,:,0]
    assert np.array_equal(depth, read_depth(str(paths['depth'])))
    native_positive = depth2disp(depth, f=289.740625, B=.08)
    saved = np.load(raw/(item['scene']+'-'+item['frame']+'.npz'))
    queries, truth, disparities = [saved[k] for k in ('queries','truth','disparities')]
    native_truth = native_positive[queries[:,0], queries[:,1]]
    assert np.array_equal(truth, native_truth.astype(np.float64))
    left, right = [spikes[k].mean(0) for k in ('spike_l','spike_r')]
    independent = []
    for qi in (0, len(queries)//2, len(queries)-1):
        y,x=queries[qi]
        for d in (0,32,64,96):
            l=left[y-4:y+5,x-4:x+5].ravel()
            r=right[y-4:y+5,x-d-4:x-d+5].ravel()
            centered_l,centered_r=l-l.mean(),r-r.mean()
            denominator=np.linalg.norm(centered_l)*np.linalg.norm(centered_r)
            scalar=float(np.dot(centered_l,centered_r)/denominator) if denominator>1e-12 else 0.
            difference=abs(scalar-float(saved['aggregate_first_scores'][qi,d]))
            assert difference<1e-12
            independent.append(dict(query_index=qi,disparity=d,absolute_score_difference=difference))
    # Shared first-frame support fitting both x-d and intentionally wrong x+d.
    common=queries[:,1] + disparities[-1] + 4 < left.shape[1]
    q,t=queries[common],truth[common]
    canonical,_=zncc(left,right,q,disparities)
    mirrored=q.copy();mirrored[:,1]=left.shape[1]-1-q[:,1]
    wrong,_=zncc(left[:,::-1],right[:,::-1],mirrored,disparities)
    sign={}
    for name,scores in [('documented_xR_equals_xL_minus_d',canonical),('wrong_sign_xR_equals_xL_plus_d',wrong)]:
        sign[name]={k:v for k,v in score_metrics(scores,t,disparities).items() if not k.startswith('_')}
    audit=dict(observed_utc=datetime.now(timezone.utc).isoformat(),scope='first manifest Dev frame only, postrun fixed loader+sign audit',
        scene=item['scene'],frame=item['frame'],decoder_exact_native_parity=decoder_parity,
        exr_shape=list(exr.shape),depth_channel=0,depth_exact_native_reader_parity=True,
        disparity_exact_native_depth2disp_parity=True,
        depth_channel_max_differences=[float(np.max(np.abs(exr[:,:,i]-depth))) for i in range(exr.shape[2])],
        documented_geometry='native loader uses negative flow=-fB/depth; positive d matcher samples xR=xL-d; no flips in synthetic decoder',
        source_identities={name:digest(repo/name) for name in ['projects/snn-depth-estimation/core/utils/reader.py','projects/snn-depth-estimation/core/stereo_datasets.py']},
        independent_scalar_patch_checks=independent,common_sign_diagnostic=sign,
        conclusion='implementation agrees with documented native loader; one-frame sign control cannot authenticate sensor/label calibration or target timestamp',
        main_run_unchanged=True)
    write_json(output/'geometry-audit.json',audit)
    print(json.dumps(audit,indent=2))


if __name__=='__main__':
    main()
