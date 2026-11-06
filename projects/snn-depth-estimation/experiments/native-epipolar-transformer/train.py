"""User-run native Train/Dev paired entry; no Test, inference on Mac or submission."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

PROJECT=Path(__file__).resolve().parents[2]
MANIFEST_SHA='bad4369b6fd8b0cdf871a20b1727c2d9b59733a16bd33f75904227bdfef4510a'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def append(path,value):
    with path.open('a') as f:f.write(json.dumps(value,allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm',choices=('original','epipolar'),required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--augmentation',choices=('none','hv'),default='hv')
    p.add_argument('--preflight',action='store_true',help='Two native updates, checking activated branch and resident Adam; no full Dev evaluation')
    a=p.parse_args()
    started=time.monotonic()
    if sha(a.manifest)!=MANIFEST_SHA:raise ValueError('native Train/Dev manifest identity mismatch')
    manifest=json.loads(a.manifest.read_text());assert len(manifest['train'])==960 and len(manifest['dev'])==60
    assert not {r['scene'] for r in manifest['train']}&{r['scene'] for r in manifest['dev']}
    assert len({r['scene'] for r in manifest['dev']})==30
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    assert torch.cuda.is_available() and torch.cuda.device_count()==1,'Run manually in a one-GPU NVIDIA allocation'
    assert os.environ.get('SLURM_JOB_ID'),'Scientific execution requires an existing Slurm allocation'
    assert any(x in torch.cuda.get_device_name(0) for x in ('4090','5090'))
    sys.path.insert(0,str(PROJECT))
    driver_path=PROJECT/'experiments/spiking-transformer-stereo/scripts/run_experiment.py'
    sys.path.insert(0,str(driver_path.parent))
    spec=importlib.util.spec_from_file_location('native_driver',driver_path);driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
    from core.epipolar_spiking_stereo import EpipolarSpikingStereoNet
    a.output.mkdir(parents=True,exist_ok=False)
    random.seed(1234);np.random.seed(1234);torch.manual_seed(1234);torch.cuda.manual_seed_all(1234)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    baseline=driver.build_model('original',1234)
    if a.arm=='epipolar':
        model=EpipolarSpikingStereoNet(baseline.args);model.load_native_state_dict(baseline.state_dict());del baseline
    else:model=baseline
    model=model.cuda();opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-5,eps=1e-8)
    schedule=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=2e-4,total_steps=10100,pct_start=.01,cycle_momentum=False,anneal_strategy='linear')
    train=driver.ManifestDataset(a.data_root,manifest['train']);dev=driver.ManifestDataset(a.data_root,manifest['dev'])
    loader=DataLoader(dev,batch_size=2,num_workers=0,generator=torch.Generator().manual_seed(2225))
    order=torch.randperm(960,generator=torch.Generator().manual_seed(1234)).tolist()
    source_paths=list((PROJECT/'core').rglob('*.py'))+[Path(__file__),driver_path,driver_path.parent/'run_provenance.py']
    source={str(path.relative_to(PROJECT)):sha(path) for path in source_paths}
    registry_path=PROJECT.parents[1]/'source-registry.json'
    registry=json.loads(registry_path.read_text()) if registry_path.exists() else None
    if registry:
        for rel,expected in registry['source_files'].items():assert sha(PROJECT.parents[1]/rel)==expected,rel
    def common_hash():
        digest=hashlib.sha256()
        for name,tensor in sorted(model.state_dict().items()):
            if name.startswith('fnet.descriptor.'):continue
            name=name.replace('fnet.encoder.','fnet.')
            digest.update(name.encode());digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()
    initial_common_hash=common_hash()
    write(a.output/'metadata.json',dict(arm=a.arm,seed=1234,updates=2 if a.preflight else 2500,effective_batch=8,physical_batch=4,accumulation=2,augmentation=a.augmentation,schedule_total=10100,manifest_sha256=MANIFEST_SHA,order=order,source_files=source,source_registry=registry,initial_native_state_sha256=initial_common_hash,git_commit=registry['git_commit'] if registry else subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip(),torch=torch.__version__,gpu=torch.cuda.get_device_name(0),slurm={k:v for k,v in os.environ.items() if k.startswith('SLURM_')},started_utc=datetime.now(timezone.utc).isoformat(),spike_coding='4 internal static-descriptor steps, not sensor time',command=sys.argv))
    history=[]
    def evaluate(update):
        py_rng,np_rng=random.getstate(),np.random.get_state()
        try:
            with torch.random.fork_rng(devices=[0]):row=dict(update=update,**driver.evaluate(model,loader,32))
        finally:
            random.setstate(py_rng);np.random.set_state(np_rng)
        append(a.output/'evaluations.jsonl',row);history.append(row);driver.reset_model(model)
    if not a.preflight:evaluate(0)
    torch.cuda.reset_peak_memory_stats()
    for step in range(2 if a.preflight else 2500):
        torch.cuda.synchronize();update_started=time.monotonic()
        model.train();model.freeze_bn();opt.zero_grad(set_to_none=True)
        indices=[order[(step*8+j)%960] for j in range(8)];items=[];flags=[]
        for j,index in enumerate(indices):
            item=list(train[index]);h=hashlib.sha256(f'recipe-hv-v1:1234:{step*8+j}'.encode()).digest()
            horizontal,vertical=(h[0]<128,h[1]<128) if a.augmentation=='hv' else (False,False)
            if horizontal:
                item[:4]=[x.flip(-1) for x in item[:4]];item[2]=-item[2]
            if vertical:item[:4]=[x.flip(-2) for x in item[:4]]
            items.append(item);flags.append([horizontal,vertical])
        total=sum(int(item[3].sum()) for item in items);assert total>0
        loss_value=0.
        for offset in (0,4):
            micro=items[offset:offset+4]
            left,right,target,valid=[torch.stack([item[j] for item in micro]).cuda() for j in range(4)]
            preds,_,_=driver.forward(model,left,right,16,True)
            loss=driver.disparity_loss(preds,target,valid)*(int(valid.sum())/total)
            assert torch.isfinite(loss);loss.backward();loss_value+=float(loss.detach());driver.reset_model(model)
        norms={n:float(v.grad.norm()) for n,v in model.named_parameters() if 'descriptor' in n and v.grad is not None}
        assert all(np.isfinite(v) for v in norms.values())
        if a.preflight and a.arm=='epipolar':
            assert norms['fnet.descriptor.output.weight']>0
            assert norms['fnet.descriptor.ffn_out.weight']>0
            if step==1:assert norms['fnet.descriptor.qkv.weight']>0 and norms['fnet.descriptor.ffn_in.weight']>0
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm) and norm>0
        lr=opt.param_groups[0]['lr'];opt.step();schedule.step()
        torch.cuda.synchronize()
        append(a.output/'train.jsonl',dict(update=step+1,samples_seen=(step+1)*8,indices=indices,augmentation_hv=flags,loss=loss_value,gradient_norm=float(norm),branch_gradient_norms=norms,lr_used=lr,lr_next=schedule.get_last_lr()[0],seconds=time.monotonic()-update_started,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,resident_allocated_mib=torch.cuda.memory_allocated()/2**20))
        print(a.arm,'UPDATE',step+1,'LOSS',loss_value,flush=True)
        if not a.preflight and step+1 in (1000,2500):evaluate(step+1)
    assert all(torch.isfinite(v).all() for v in model.state_dict().values())
    assert all(torch.isfinite(v).all() for state in opt.state.values() for v in state.values() if torch.is_tensor(v))
    assert source=={str(path.relative_to(PROJECT)):sha(path) for path in source_paths},'source changed during run'
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=schedule.state_dict(),updates=step+1,order=order,source_files=source,rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())),a.output/'last.pt')
    write(a.output/'result.json',dict(status='COMPLETED',scope='GPU two-update memory/gradient preflight' if a.preflight else '2500-update candidate screen',updates=step+1,samples_seen=(step+1)*8,equivalent_train_passes=(step+1)*8/960,initial_native_state_sha256=initial_common_hash,process_seconds=time.monotonic()-started,finite_model_optimizer=True,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,history=history))

if __name__=='__main__':main()
