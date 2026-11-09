"""Verify published file hashes, raw-log completeness and full-Dev aggregation; stdlib only."""
from pathlib import Path
import csv,gzip,hashlib,json,math,statistics,tarfile
ROOT=Path(__file__).resolve().parent

def obj(name):return json.loads((ROOT/name).read_text())
def same(a,b):assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10),(a,b)
def expected_lr(step):
 initial,maximum,minimum=8e-6,2e-4,8e-10
 return initial+(maximum-initial)*step/299 if step<=299 else maximum+(minimum-maximum)*(step-299)/(29999-299)

def main():
 manifest=obj('artifact-manifest.json')
 for name,r in manifest['files'].items():
  b=(ROOT/name).read_bytes();assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256'],name
 registry=obj('source-registry.json')
 with tarfile.open(ROOT/'source.tar.gz') as tar:
  for name,digest in registry['source_files'].items():assert hashlib.sha256(tar.extractfile(name).read()).hexdigest()==digest,name
 expected_evals=[0,250,500,1000,1500,2500]+list(range(5000,30001,2500))
 endpoints={r['model']:r for r in csv.DictReader((ROOT/'summaries/endpoints.csv').open())}
 logs=obj('training-log-hashes.json')
 for model in ('original-non-i','hybrid-b'):
  cell=ROOT/'seed-1234'/model;meta=json.loads((cell/'metadata.json').read_text());result=json.loads((cell/'result.json').read_text())
  assert result['status']=='COMPLETED' and result['updates']==result['scheduler_last_epoch']==30000 and result['samples_seen']==240000
  assert meta['source_registry']['source_identity_sha256']==registry['source_identity_sha256']
  raw=gzip.decompress((ROOT/logs[model]['path']).read_bytes());assert hashlib.sha256(raw).hexdigest()==logs[model]['uncompressed_sha256']
  assert len(raw)==logs[model]['uncompressed_bytes'];rows=[json.loads(s) for s in raw.splitlines()];assert len(rows)==30000
  order=meta['order'];assert len(order)==960 and len(set(order))==960
  for i,r in enumerate(rows,1):
   assert r['update']==i and r['samples_seen']==8*i
   positions=range((i-1)*8,i*8)
   assert r['indices']==[order[p%960] for p in positions]
   flags=[]
   for p in positions:
    digest=hashlib.sha256(f'recipe-hv-v1:1234:{p}'.encode()).digest();flags.append([digest[0]<128,digest[1]<128])
   assert r['augmentation_hv']==flags
   assert math.isclose(r['lr_used'],expected_lr(i-1),rel_tol=1e-10,abs_tol=1e-14)
   assert math.isclose(r['lr_next'],expected_lr(i),rel_tol=1e-10,abs_tol=1e-14)
   assert r['physical_batch']==4 and r['accumulation']==2 and r['world_size']==1
   assert math.isfinite(r['loss']) and math.isfinite(r['gradient_norm']) and r['gradient_norm']>0
  ev=[json.loads(s) for s in (cell/'evaluations.jsonl').read_text().splitlines()];assert [r['update'] for r in ev]==expected_evals
  for r in ev:
   frames=r['per_frame'];assert len(frames)==60 and len({(f['scene'],f['frame']) for f in frames})==60
   scenes={s:statistics.mean(f['epe'] for f in frames if f['scene']==s) for s in {f['scene'] for f in frames}};assert len(scenes)==30
   for s,epe in scenes.items():same(epe,r['scene_epe'][s])
   same(statistics.mean(scenes.values()),r['mean_scene_epe']);same(sum(f['bad3']*f['valid_pixels'] for f in frames)/sum(f['valid_pixels'] for f in frames),r['bad3'])
  same(ev[-1]['mean_scene_epe'],float(endpoints[model]['mean_scene_epe']));same(ev[-1]['bad3'],float(endpoints[model]['bad3_percent']))
  print(f'{model}: 30,000 rows, 240,000 presentations, sample/flip/LR stream and 17 full Dev aggregations verified')
 print(f"PASS: {len(manifest['files'])} artifact hashes and {len(registry['source_files'])} frozen source hashes verified")

if __name__=='__main__':main()
