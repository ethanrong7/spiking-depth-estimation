"""Bind S9 split using existing local metadata only; never open dataset payloads."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[7]
CAMP = Path(__file__).resolve().parents[1]
RAW = ROOT / "storage/experiment-runs/snn-depth-estimation/spiking-transformer-stereo/spike-depth-direction-v1"
DATA = ROOT / "storage/datasets/flyingthings3d-native-v1"
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p): return json.loads(p.read_text())
def main():
    contract = CAMP / "analysis/stage9-action-value-contract-astra.md"
    manifest = RAW / "stage5/input-provenance/derived-manifest.json"
    hashes = RAW / "stage5/input-provenance/generation-window-hashes.json"
    download = DATA / "source-download-receipt-v4.json"
    assert sha(contract) == "217c7f7d1e0947fab351d2068577421a67c93f1936ba8612e250d58391975d2d"
    assert sha(manifest) == "eb52ec22890423efc2d5455904c3155d6aaa8c8a96199eb8fcd371305d62534d"
    m, h, d = load(manifest), load(hashes), load(download)
    assert h['manifest_sha256'] == sha(manifest) and d['status'] == 'COMPLETE'
    assert m['source_index_sha256'] == d['manifest_sha256']
    train = sorted({r['source_id'] for r in m['train']})
    dev = sorted({r['source_id'] for r in m['dev']})
    assert len(train) == 120 and len(dev) == 30 and not set(train)&set(dev)
    calibration = [train[i] for i in range(5,120,6)]
    fit = [v for v in train if v not in calibration]
    assert (len(fit),len(calibration)) == (100,20)
    rows=[]
    for origin in ('train','dev'):
        for ordinal, r in enumerate(m[origin]):
            i=r['center_index']; n=r['global_indices']
            assert r['spike_shape'] == [50,540,960] and r['vertical_flip'] is False
            assert n == list(range(50*i-24,50*i+26)) and n[24] == 50*i
            assert r['source_time'] == i and int(r['source_frame']) == i+6
            assert r['sample_id'] == r['source_id']+'/'+r['source_frame']
            assert r['split'] == origin
            group = 'dev' if origin=='dev' else ('policy_calibration' if r['source_id'] in calibration else 'model_fit')
            checks={}
            for key in ('rgb_l','rgb_r','disparity_l'):
                path=r[key]; v=d['completed'][path]
                assert v['source_id']==r['source_id'] and len(v['sha256'])==64
                checks[key]={'path':path,'sha256':v['sha256'],'bytes':v['size']}
            for key,view in (('spike_l','l'),('spike_r','r')):
                path=r[key];v=h['DAT_expected_hashes'][path]
                assert v['source_id']==r['source_id'] and v['center']==i and v['view']==view and v['bytes']==3240000
                checks[key]={'path':path,'sha256':v['sha256'],'bytes':v['bytes']}
            for key in checks:
                view='right' if key.endswith('_r') else 'left'
                if key.startswith('spike'): assert r[key]==f"{r['source_id']}/{key}/{r['source_frame']}.dat"
                else: assert '/'+view+'/'+r['source_frame']+'.' in r[key]
            rows.append({'group':group,'manifest_split':origin,'manifest_ordinal':ordinal,'sample_id':r['sample_id'],'source_id':r['source_id'],'center_index':i,'source_frame':r['source_frame'],'global_start':n[0],'global_end':n[-1],'target_bin':24,'input_shape':[50,540,960],'files':checks})
    assert len({r['sample_id'] for r in rows})==1020
    for group,ids,count,centers in [('model_fit',fit,800,set(range(1,9))),('policy_calibration',calibration,160,set(range(1,9))),('dev',dev,60,{3,6})]:
        rr=[r for r in rows if r['group']==group];assert len(rr)==count
        for scene in ids: assert {r['center_index'] for r in rr if r['source_id']==scene}==centers
    inputs={str(p.relative_to(ROOT)):sha(p) for p in (contract,manifest,hashes,download)}
    result={'status':'BOUND_METADATA_ONLY','version':'stage9-split-v1','inputs_sha256':inputs,'selection':{'sort':'lexicographic source_id','calibration_train_scene_ordinals_zero_based':list(range(5,120,6)),'row_order':'original manifest order within each split'},'model_fit_scene_ids':fit,'policy_calibration_scene_ids':calibration,'dev_scene_ids':dev,'counts':{'model_fit_scenes':100,'model_fit_rows':800,'policy_calibration_scenes':20,'policy_calibration_rows':160,'dev_scenes':30,'dev_rows':60},'geometry':{'binary_shape':[50,540,960],'padded_shape':[50,544,960],'target_bin':24,'prefix_endpoints_inclusive':[24,32,49],'prefix_bins':[25,33,50],'lookahead_bins':[0,8,25]},'rows':rows,'payload_reads':0,'model_execution':False}
    out=CAMP/'configs/stage9-split-rows.json'
    assert not out.exists(), 'immutable binding already exists; do not overwrite'
    out.write_text(json.dumps(result,sort_keys=True,separators=(',',':'))+'\n')
    receipt={'status':'PASS','split_binding':str(out.relative_to(ROOT)),'split_binding_sha256':sha(out),'script_sha256':sha(Path(__file__)),'inputs_sha256':inputs,'counts':result['counts'],'checks':['scene disjointness','unique exact cohort','original row order','native geometry','centered index target bin24','Train1..8 Dev3,6','stereo and target paths','all5100 file hash bindings'],'scope':'Existing metadata only; no payload rehash or decode, no inference, no cluster command.'}
    dest=CAMP/'results/stage9-split-binding';dest.mkdir(parents=True,exist_ok=True)
    (dest/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))
if __name__=='__main__': main()
