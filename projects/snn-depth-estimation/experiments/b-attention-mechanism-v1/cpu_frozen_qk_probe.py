"""Two-frame frozen Q/K clipping intervention, original archived model; CPU only."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['source-root','checkpoint','data-root','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--cases',nargs='+',default=['scene_0123/0001','scene_0137/0001'])
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    if len(a.cases)>4:raise ValueError('At most four selected cases per CPU probe')
    started=time.monotonic()
    signal.signal(signal.SIGALRM,lambda *unused:(_ for _ in ()).throw(TimeoutError('180-second diagnostic budget')))
    signal.alarm(180)
    registry=json.loads((a.source_root/'source-registry.json').read_text())
    for rel,digest in registry['source_files'].items():assert sha(a.source_root/rel)==digest
    project=a.source_root/'projects/snn-depth-estimation'
    script=project/'experiments/spiking-transformer-stereo/scripts/run_experiment.py'
    sys.path[:0]=[str(project),str(script.parent)]
    import numpy as np
    import torch
    import cv2
    torch.set_num_threads(2);torch.set_num_interop_threads(2);cv2.setNumThreads(1)
    spec=importlib.util.spec_from_file_location('frozen_probe_driver',script)
    driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
    from core.conventional_attention_placement import install,LinearCodes
    class ClippedQK(LinearCodes):
        def forward(self,x):
            z=self.norm(x)
            return self.q(z).clamp(-1,1),self.k(z).clamp(-1,1),self.v(z)
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    assert ck['updates']==30000 and ck['source_identity']==registry['source_identity_sha256']
    model=install(driver.build_model('original',1234),'B').eval()
    model.load_state_dict(ck['model'],strict=True);del ck
    def state_hash():
        h=hashlib.sha256()
        for name,tensor in sorted(model.state_dict().items()):
            h.update(name.encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    initial=state_hash()
    manifest=json.loads((project/'experiments/spiking-transformer-stereo/configs/manifest.json').read_text())
    cases=[next(r for r in manifest['dev'] if r['scene']+'/'+r['frame']==key) for key in a.cases]
    data=driver.ManifestDataset(a.data_root,cases)
    result=dict(status='running',scope='outcome-selected frames; frozen conventional B; no retraining; not full Dev',
                checkpoint_sha256=sha(a.checkpoint),source_identity=registry['source_identity_sha256'],
                evaluator_sha256=sha(__file__),gpu_used=False,optimizer_steps=0,iterations=32,rows=[])
    def save():
        result['cpu_wall_seconds']=time.monotonic()-started
        temp=a.output/'result.tmp';temp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');temp.replace(a.output/'result.json')
    save()
    with torch.no_grad():
        for i,case in enumerate(cases):
            left,right,target,valid,_=data[i]
            key=case['scene']+'-'+case['frame']
            np.save(a.output/(key+'-target.npy'),target.numpy())
            np.save(a.output/(key+'-valid.npy'),valid.numpy())
            for condition,cls in [('baseline',LinearCodes),('clip-qk',ClippedQK)]:
                model.placement.codes.__class__=cls
                pred=driver.forward(model,left[None],right[None],32)
                assert torch.isfinite(pred).all()
                error=(pred[0,0]-target[0]).abs()[valid]
                row=dict(scene=case['scene'],frame=case['frame'],condition=condition,valid_pixels=int(valid.sum()),**driver.metric_values(error))
                np.save(a.output/(key+'-'+condition+'.npy'),pred[0,0].numpy())
                result['rows'].append(row);driver.reset_model(model);save()
                print(json.dumps(row),flush=True)
    assert initial==state_hash()
    result.update(status='completed',weights_unchanged=True)
    signal.alarm(0);save()


if __name__=='__main__':main()
