"""Verify the published compact evidence; no remote weights or raw logs required."""
from pathlib import Path
import csv,hashlib,itertools,json,math,statistics

root=Path(__file__).resolve().parent
read=lambda name:json.loads((root/name).read_text())
rows=lambda name:list(csv.DictReader((root/'summaries'/name).open()))
manifest=read('artifact-manifest.json')
for name,record in manifest['files'].items():
    p=root/name
    assert p.stat().st_size==record['bytes'],name
    assert hashlib.sha256(p.read_bytes()).hexdigest()==record['sha256'],name
assert {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name!='artifact-manifest.json'}==set(manifest['files'])
seeds=(1234,2345)
variants=('linear-identity','linear-surrogate','signed-identity','signed-surrogate')
dates=[0,250,500,1000,1500,2500,*range(5000,30001,2500)]
endpoint=rows('endpoints.csv');assert len(endpoint)==8
by={(int(r['seed']),r['variant']):r for r in endpoint}
assert set(by)==set(itertools.product(seeds,variants))
audit=read('summaries/completion-audit.json')
assert audit['status']=='completed' and len(audit['cells'])==8
for (seed,v),e in by.items():
    assert int(e['update'])==30000 and e['reuse']=='False'
    assert all(math.isfinite(float(e[k])) for k in ('mean_scene_epe','bad3_percent'))
    m=read(f'seed-{seed}/{v}/metadata.json');r=read(f'seed-{seed}/{v}/result.json');c=m['config']
    assert r['status']=='COMPLETED' and r['updates']==r['scheduler_last_epoch']==30000
    assert r['samples_seen']==240000 and r['finite_model_optimizer']
    assert (c['physical_batch'],c['accumulation'],c['world_size'])==(4,2,1)
    assert c['seed']==seed and c['variant']==v and c['schedule_total']==30000
    assert sorted(m['order'])==list(range(960))
for seed in seeds:
    cells=[r for r in audit['cells'] if r['seed']==seed]
    assert len(cells)==4 and all(r['status']=='completed' and r['updates']==30000 for r in cells)
    for key in ('source_identity','initial_state_sha256','stream_identity'):assert len({r[key] for r in cells})==1
curve=rows('curves.csv');assert len(curve)==8*17*2
for seed,v in by:
    for metric in ('mean_scene_epe','bad3'):
        part=[r for r in curve if int(r['seed'])==seed and r['variant']==v and r['metric']==metric]
        assert [int(r['update']) for r in part]==dates
        assert all(int(r['samples_seen'])==int(r['update'])*8 for r in part)
        key='bad3_percent' if metric=='bad3' else metric
        assert math.isclose(float(part[-1]['value']),float(by[seed,v][key]),abs_tol=1e-12)
scenes=rows('scene-deltas.csv');assert len(scenes)==2*17*6*30
for seed in seeds:
    for a,b in itertools.combinations(('linear-identity','signed-surrogate','linear-surrogate','signed-identity'),2):
        for update in dates:
            part=[r for r in scenes if int(r['seed'])==seed and int(r['update'])==update and r['variant_a']==a and r['variant_b']==b]
            assert len(part)==len({r['scene'] for r in part})==30
            for r in part:assert math.isclose(float(r['epe_a'])-float(r['epe_b']),float(r['epe_a_minus_b']),abs_tol=1e-12)
            if update==30000:
                for field,v in (('epe_a',a),('epe_b',b)):assert math.isclose(statistics.mean(float(r[field]) for r in part),float(by[seed,v]['mean_scene_epe']),abs_tol=1e-12)
effects=rows('factorial-contrasts.csv');assert len(effects)==2*17*6
for seed in seeds:
    c,g,s,h=[float(by[seed,v]['mean_scene_epe']) for v in variants]
    expected=dict(zip(('hybrid_package','forward_at_identity','forward_at_surrogate','surrogate_at_linear','surrogate_at_signed','interaction'),(c-h,c-s,g-h,c-g,s-h,s+g-h-c)))
    actual={r['contrast']:float(r['epe_improvement']) for r in effects if int(r['seed'])==seed and int(r['update'])==30000}
    assert actual.keys()==expected.keys()
    for k,value in expected.items():assert math.isclose(actual[k],value,abs_tol=1e-12),k
print(f'PASS: {len(manifest["files"])} artifact hashes; 8 completed endpoints; 272 curve rows; 6120 scene rows; 204 contrasts.')
