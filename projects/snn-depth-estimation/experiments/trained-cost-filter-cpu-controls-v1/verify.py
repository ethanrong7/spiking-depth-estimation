"""Recompute saved CPU predictions' metrics; no new model forward."""
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from datetime import datetime,timezone

import cv2
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
RAW=REPO/'storage/experiment-runs/snn-depth-estimation/trained-cost-filter-cpu-controls-v1'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(2)
    run='cpu-within-backend-20261104';out=HERE/'results'/run
    j=json.loads((out/'summary.json').read_text());assert j['status']=='COMPLETED_CPU_DIAGNOSTIC'
    assert digest(HERE/'run.py')==j['evaluator_sha256'] and digest(HERE/'controls.py')==j['control_source_sha256']
    assert digest(HERE/'STUDY.md')==j['study_sha256'] and digest(HERE/'CPU-ONLY-AMENDMENT.md')==j['amendment_sha256']
    manifest=json.loads((REPO/'projects/snn-depth-estimation/experiments/spiking-transformer-stereo/configs/manifest.json').read_text())['dev']
    inputs=[]
    for scene,frame in sorted({(r['scene'],r['frame']) for r in j['per_frame']}):
        item=next(r for r in manifest if (r['scene'],r['frame'])==(scene,frame))
        paths={k:REPO/'data/Synthetic'/item[k] for k in ('spike_l','spike_r','depth')}
        inputs.append(dict(scene=scene,frame=frame,files={k:dict(path=item[k],sha256=digest(p),bytes=p.stat().st_size) for k,p in paths.items()}))
        depth=cv2.imread(str(paths['depth']),cv2.IMREAD_ANYCOLOR|cv2.IMREAD_ANYDEPTH)[:,:,0]
        valid=np.isfinite(depth)&(depth>0)
        target=np.zeros_like(depth,dtype=np.float32);target[valid]=-(289.740625*.08)/depth[valid]
        valid&=np.abs(target)<512
        target=torch.from_numpy(target);valid=torch.from_numpy(valid)
        support_hash=hashlib.sha256(target[None].contiguous().numpy().tobytes()+valid.contiguous().numpy().tobytes()).hexdigest()
        inside=(torch.arange(400)[None]+target>=0)&(torch.arange(400)[None]+target<=399)
        rows=[r for r in j['per_frame'] if (r['scene'],r['frame'])==(scene,frame)]
        for row in rows:
            assert row['target_support_sha256']==support_hash
            pred=torch.load(RAW/run/f'{scene}-{frame}-{row["condition"]}.pt',map_location='cpu',weights_only=True)[0,0]
            error=(pred-target).abs()
            for stratum,mask in [('full',valid),('inview',valid&inside),('oov',valid&~inside)]:
                values=error[mask];metric=row['metrics'][stratum]
                assert values.numel()==metric['valid_pixels']
                assert float(values.mean())==metric['epe']
                for n in (1,3,5):assert float((values>n).float().mean()*100)==metric[f'bad{n}']
            if row['condition']=='baseline':
                repeat=torch.load(RAW/run/f'{scene}-{frame}-baseline-repeat.pt',map_location='cpu',weights_only=True)[0,0]
                assert torch.equal(pred,repeat)
    checkpoint=REPO/'storage/experiment-runs/snn-depth-estimation/spiking-transformer-stereo/spiking-attention-placement-v1/train/A/step-30000.pt'
    assert digest(checkpoint)==j['checkpoint_sha256']
    source=RAW/'source';registry=json.loads((source/'source-registry.json').read_text())
    assert all(digest(source/path)==sha for path,sha in registry['source_files'].items())
    snapshot=RAW/run/'evaluator-source';snapshot.mkdir(exist_ok=True)
    for file in [HERE/'run.py',HERE/'controls.py',HERE/'STUDY.md',HERE/'CPU-ONLY-AMENDMENT.md']:shutil.copy2(file,snapshot/file.name)
    verification=dict(verified_utc=datetime.now(timezone.utc).isoformat(),prediction_rows_recomputed=12,CPU_baseline_tensor_repeats_exact=2,
        all_full_inview_oov_metrics_and_support_hashes_match=True,checkpoint_unchanged=True,frozen_source_files_unchanged=len(registry['source_files']),
        initial_GPU_admission_preserved=True,source_hashes_match_executed_evaluator=True,input_file_identities=inputs,
        cumulative_forward_compute_seconds=j['cumulative_compute_seconds'],no_new_model_forward_in_verification=True)
    (out/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    provenance=dict(command='PYTHONDONTWRITEBYTECODE=1 scratch/spiking-attention-unit-venv/bin/python -u projects/snn-depth-estimation/experiments/trained-cost-filter-cpu-controls-v1/run.py --run-id cpu-within-backend-20261104 --cpu-only-amendment',
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        dirty_status=subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).splitlines(),verification=verification)
    (RAW/run/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(verification,indent=2))


if __name__=='__main__':main()
