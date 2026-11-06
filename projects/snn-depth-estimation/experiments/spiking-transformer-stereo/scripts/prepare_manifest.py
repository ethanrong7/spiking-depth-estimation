"""Freeze scene-disjoint cohorts by file identity without decoding held-out labels."""
import argparse, hashlib, json
from pathlib import Path

def freeze(root, output):
    def rows(split, scenes, count):
        out=[]
        for scene in scenes:
            base=root/split/f'scene_{scene:04d}'
            ids=sorted(p.name.split('_')[0] for p in (base/'depth').glob('*.exr'))
            assert len(ids)==255, (base,len(ids))
            indices=[round(i*(len(ids)-1)/(count-1)) for i in range(count)]
            for i in indices:
                frame=ids[i]
                paths={kind:str((base/kind/f'{frame}_{suffix}').relative_to(root)) for kind,suffix in [('spike_l','spike_l.dat'),('spike_r','spike_r.dat'),('depth','depth.exr')]}
                for p in paths.values(): assert (root/p).is_file(),p
                out.append({'scene':base.name,'frame':frame,**paths})
        return out
    manifest={'train':rows('Train',range(120),8),'dev':rows('Train',range(120,150),2),'confirmation':rows('Test',range(150,190),2),'overfit':rows('Train',[0],4)}
    manifest['schema_version']=1
    output.mkdir(parents=True,exist_ok=True)
    encoded=json.dumps(manifest,indent=2,sort_keys=True)+'\n'
    (output/'manifest.json').write_text(encoded)
    protocol={'manifest_sha256':hashlib.sha256(encoded.encode()).hexdigest(),'counts':{k:len(v) for k,v in manifest.items() if isinstance(v,list)},'seed':[1234,2345],'train_iters':16,'eval_iters':32,'physical_batch':2,'effective_batch':2,'native_shape':[50,250,400],'padding_divisor':32,'pilot_updates':1000,'extension_updates':3000,'overfit_updates':500,'overfit_gate_epe_reduction':0.5,'loss':'weighted disparity L1 gamma0.9 adjusted15; optional original neuron regularization','calibration':{'f':289.740625,'B':0.08},'confirmation_locked_until_development_selection':True}
    # Root protocol.json is the scientific authority; print inventory only.
    print(json.dumps(protocol))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();freeze(a.data_root,a.output)
