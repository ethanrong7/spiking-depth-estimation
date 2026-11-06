"""CPU-only saved artifact inspection, no model construction or inference."""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
torch.set_num_threads(2)

p=argparse.ArgumentParser()
p.add_argument('--raw',type=Path,required=True)
p.add_argument('--source',type=Path,required=True)
a=p.parse_args()
registration=json.loads((a.source/'configs/registration.json').read_text())
reports=[]
for config in registration['cases']:
    path=a.raw/'cases'/config['id']/'last.pt'
    ck=torch.load(path,map_location='cpu',weights_only=False)
    assert ck['config']==config and ck['updates']==480 and ck['samples_seen']==480*config['batch']
    result=json.loads((path.parent/'result.json').read_text())
    meta=json.loads((path.parent/'metadata.json').read_text())
    assert ck['order']==meta['order']
    assert ck['source']['source_identity_sha256']==meta['source']['source_identity_sha256']
    assert ck['source']['source_identity_sha256']==json.loads((a.source/'source-registry.json').read_text())['source_identity_sha256']
    assert all(bool(torch.isfinite(t).all()) for t in ck['model'].values())
    assert all(bool(torch.isfinite(t).all()) for state in ck['optimizer']['state'].values() for t in state.values() if torch.is_tensor(t))
    steps=[int(s['step'].item()) for s in ck['optimizer']['state'].values() if 'step' in s]
    assert min(steps)==max(steps)==ck['scheduler']['last_epoch']==480
    assert ck['scheduler']['total_steps']==config['schedule_total']
    assert ck['optimizer']['param_groups'][0]['lr']==ck['scheduler']['_last_lr'][0]
    assert set(ck['rng'])=={'python','numpy','torch','cuda'}
    h=hashlib.sha256()
    with path.open('rb') as f:
        while chunk:=f.read(1024*1024):h.update(chunk)
    reports.append(dict(case=config['id'],status='PASS',checkpoint_sha256=h.hexdigest(),
        model_tensors=len(ck['model']),adamw_populated_states=len(steps),optimizer_step_min_max=[min(steps),max(steps)],
        scheduler_last_epoch=480,scheduler_total_steps=config['schedule_total'],next_lr=ck['scheduler']['_last_lr'][0],
        finite_model_optimizer=True))
    del ck
out=dict(status='PASS',inspection='CPU only; CUDA hidden; no model construction,forward,training or additional Slurm allocation',cases=reports)
(a.raw/'checkpoint-audit.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(status='PASS',checkpoints=len(reports))))
