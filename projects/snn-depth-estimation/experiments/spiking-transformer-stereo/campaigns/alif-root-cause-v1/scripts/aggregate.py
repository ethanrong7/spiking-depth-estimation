"""Collect saved P0 cells; never run inference or modify job state."""
import json
from pathlib import Path

def aggregate(root, reg):
 cells={(x['model'],x['seed'],x['split'],x['iterations']):dict(x,completed_frames=0,complete=False) for x in reg['conditions']}
 workers=[]
 for seed in reg['paired_seeds']:
  path=root/('seed'+str(seed))/'result.json'
  if path.exists():
   worker=json.loads(path.read_text());workers.append(worker)
   for x in worker['conditions']:
    key=(x['model'],x['seed'],x['split'],x['iterations']);assert key in cells;cells[key]=x
 complete=len(workers)==2 and all(w['status']=='COMPLETED' for w in workers)
 result=dict(status='COMPLETED' if complete else 'RUNNING' if workers else 'PENDING',checkpoints=reg['checkpoints'],conditions=list(cells.values()),reproduction={str(w['seed']):w['reproduction'] for w in workers},horizon_diagnostics={})
 for seed in reg['paired_seeds']:
  relevant=[cells[(m,seed,'dev',k)] for m in reg['models'] for k in reg['horizons']]
  if not all(x['complete'] for x in relevant):continue
  e=lambda m,k:cells[(m,seed,'dev',k)]['metrics']['mean_scene_epe']
  d16=e('original',16)-e('spiking',16);d32=e('original',32)-e('spiking',32);r=d32-d16
  fraction=r/d32 if d32>0 else None
  gates=dict(positive_dev32_gap=d32>0,alif_worse_at32=e('original',32)>e('original',16),fraction_at_least_half=fraction is not None and fraction>=.5)
  result['horizon_diagnostics'][str(seed)]=dict(D16=d16,D32=d32,R=r,explained_fraction=fraction,gate_components=gates,horizon_dominant=all(gates.values()))
 if len(result['horizon_diagnostics'])==2:result['horizon_dominant_both']=all(x['horizon_dominant'] for x in result['horizon_diagnostics'].values())
 state=root/'state.json'
 if state.exists():
  result['jobs']=json.loads(state.read_text())['jobs']
  if any(any(word in job.get('accounting','') for word in ['FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY']) for job in result['jobs']):result['status']='FAILED'
 temp=root/'result.tmp';temp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');temp.replace(root/'result.json')
 return result
