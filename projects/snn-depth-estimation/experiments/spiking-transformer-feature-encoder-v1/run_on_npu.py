"""Prepare and execute four full Transformer encoders at two paired seeds."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import datetime

EXP=Path(__file__).resolve().parent
ROOT=EXP.parents[3]
PROJECT=EXP.parents[1]
BASE=EXP.parent/'b-attention-mechanism-v1'
VARIANTS=('spikformer','qkformer','meta-spikeformer','swin')
SEEDS=(1234,2345)
EVALS=[0,1000]+list(range(2500,30001,2500))
STUDY='encoder-screen-npu-v1'


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def read_json(path):return json.loads(Path(path).read_text())


def cells(devices):
    return [dict(seed=seed,variant=variant,device=devices[i%len(devices)])
            for i,(variant,seed) in enumerate((v,s) for v in VARIANTS for s in SEEDS)]


def environment(device,seed):
    return {**os.environ,'ASCEND_RT_VISIBLE_DEVICES':str(device),'PYTHONHASHSEED':str(seed),
            'OMP_NUM_THREADS':'4','PYTHONDONTWRITEBYTECODE':'1','WORLD_SIZE':'1',
            'RANK':'0','LOCAL_RANK':'0','HCCL_DETERMINISTIC':'false'}


def freeze_source(root):
    """Only committed project/config source, never unrelated untracked user work."""
    git=lambda *args:subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
    names=git('ls-files','--','projects/snn-depth-estimation','configs','requirements.txt').splitlines()
    names=[n for n in names if Path(n).suffix in ('.py','.json','.md','.txt','.yaml','.toml')
           and not {'outputs','results','__pycache__'}.intersection(Path(n).parts)]
    if str((EXP/'run_on_npu.py').relative_to(ROOT)) not in names:
        raise RuntimeError('Commit the implementation before preparing production sources')
    files={}
    for name in names:
        data=(ROOT/name).read_bytes()
        expected=subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT)
        if data!=expected:
            raise RuntimeError(f'Experiment dependency has uncommitted edits: {name}')
        files[name]=hashlib.sha256(data).hexdigest()
    registry=dict(git_commit=git('rev-parse','HEAD'),source_files=files,
                  source_identity_sha256=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest())
    write_json(root/'source-registry.json',registry)


def verify_source(root):
    r=read_json(root/'source-registry.json')
    for name,expected in r['source_files'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:
            raise RuntimeError(f'Frozen source changed: {name}')
    from upstream import ensure_sources
    ensure_sources(root/'upstream',offline=True)
    return r['source_identity_sha256']


def prepare(args):
    output=args.output_root
    requested=dict(study=STUDY,cells=cells(args.devices),data_root=str(args.data_root),
                   manifest=str(args.manifest),updates=30000,physical_batch=args.physical_batch,
                   accumulation=8//args.physical_batch,effective_batch=8,preload_gib=args.preload_gib)
    if (output/'config.json').exists():
        if read_json(output/'config.json')!=requested:
            raise RuntimeError('Existing batch configuration differs; use a new output root')
        verify_source(output);return
    if output.exists():
        raise RuntimeError(f'Existing incomplete preparation: {output}; inspect/use a new output root')
    from upstream import ensure_sources
    source_cache=ensure_sources(args.source_cache,offline=args.offline)
    import shutil
    output.mkdir(parents=True)
    shutil.copytree(source_cache,output/'upstream')
    freeze_source(output)
    sys.path[:0]=[str(PROJECT),str(BASE),str(EXP.parent/'shared-gpu-30k-v1')]
    from settings import load
    import torch
    torch.set_num_threads(4)
    contract=load('encoder_initial_contract',EXP.parent/'spiking-attention-placement-phase2/protocol.py')
    if hashlib.sha256(args.manifest.read_bytes()).hexdigest()!=contract.MANIFEST_SHA:
        raise RuntimeError('Dataset split differs from registered NPU split')
    path=EXP.parent/'spiking-transformer-stereo/scripts/run_experiment.py'
    sys.path.insert(0,str(path.parent));driver=load('encoder_initial_driver',path)
    for seed in SEEDS:
        model=driver.build_model('original',seed)
        state=model.state_dict()
        dest=output/'initializations'/f'seed-{seed}.pt';dest.parent.mkdir(exist_ok=True)
        torch.save(dict(seed=seed,model=state,sha256=contract.tensor_hash(state)),dest)
        del state,model
    write_json(output/'matrix.json',requested['cells'])
    write_json(output/'config.json',requested)


def worker_command(args,cell,preflight=False,resume=False,pause=False):
    root=args.output_root/('preflight' if preflight else '')/f"seed-{cell['seed']}"/cell['variant']
    command=[sys.executable,str(EXP/'worker_npu.py'),'--seed',str(cell['seed']),
             '--variant',cell['variant'],'--root',str(root),'--data-root',str(args.data_root),
             '--manifest',str(args.manifest),'--source-registry',str(args.output_root/'source-registry.json'),
             '--source-dir',str(args.output_root/'upstream'),'--initialization',
             str(args.output_root/'initializations'/f"seed-{cell['seed']}.pt"),
             '--physical-batch',str(args.physical_batch),'--stop-at','200' if preflight else '30000',
             '--input-mode','preload','--preload-gib',str(args.preload_gib),'--input-workers','2','--prefetch-depth','2']
    if preflight:command.append('--preflight')
    if resume:command.append('--resume')
    if pause:command.extend(['--pause-at','2'])
    return root,command


def run_worker(args,cell,preflight=False,resume=False,pause=False):
    root,command=worker_command(args,cell,preflight,resume,pause)
    root.mkdir(parents=True,exist_ok=True)
    if (root/'checkpoints'/'last.pt').exists()!=resume:
        raise RuntimeError(f'Checkpoint and resume flag disagree: {root}')
    if (root/'pause.request').exists():
        raise RuntimeError(f'Pause request is still present: {root}')
    write_json(root/'launch.json',dict(command=command,cell=cell,time=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    print('START',cell,'preflight' if preflight else 'production',flush=True)
    with (root/'stdout.log').open('ab') as out,(root/'stderr.log').open('ab') as err:
        r=subprocess.run(command,cwd=ROOT,env=environment(cell['device'],cell['seed']),stdout=out,stderr=err)
    if r.returncode!=(75 if pause else 0):
        raise RuntimeError(f'Exit {r.returncode}: inspect {root}/stderr.log (no automatic recipe changes)')
    if pause:
        if read_json(root/'checkpoint.json')['update']!=2:raise RuntimeError('Resume probe missed step 2')
    elif read_json(root/'result.json')['updates']!=(200 if preflight else 30000):
        raise RuntimeError(f'Incomplete endpoint: {root}')


def preflight_cell(args,cell):
    root,_=worker_command(args,cell,preflight=True)
    if (root/'result.json').exists():
        if read_json(root/'result.json')['status']=='PREFLIGHT_COMPLETED' and (root/'restore-audit-2.json').exists():return
        raise RuntimeError(f'Invalid existing preflight: {root}')
    root.mkdir(parents=True,exist_ok=True)
    if not (root/'neuron-validation.json').exists():
        command=[sys.executable,str(EXP/'validate_npu.py'),'--output',str(root/'neuron-validation.json')]
        with (root/'neuron-validation.log').open('ab') as f:
            subprocess.run(command,cwd=ROOT,env=environment(cell['device'],cell['seed']),stdout=f,stderr=f,check=True)
    if read_json(root/'neuron-validation.json')['status']!='passed':raise RuntimeError('Neuron validation failed')
    if not (root/'checkpoints'/'last.pt').exists():run_worker(args,cell,preflight=True,pause=True)
    else:
        update=read_json(root/'checkpoint.json')['update']
        if update<2:run_worker(args,cell,preflight=True,resume=True,pause=True)
    run_worker(args,cell,preflight=True,resume=True)
    if not (root/'restore-audit-2.json').exists():raise RuntimeError('Missing resume audit')


def production_cell(args,cell):
    root,_=worker_command(args,cell)
    if (root/'result.json').exists():
        result=read_json(root/'result.json')
        if args.resume and result['status']=='COMPLETED' and result['updates']==30000:return
        raise RuntimeError(f'Existing result needs --resume: {root}')
    run_worker(args,cell,resume=args.resume and (root/'checkpoints'/'last.pt').exists())


def on_devices(args,function,preflight=False):
    matrix=[c for c in cells(args.devices) if not preflight or c['seed']==1234]
    def queue(device):
        for cell in matrix:
            if cell['device']==device:function(args,cell)
    with ThreadPoolExecutor(max_workers=len(args.devices)) as pool:
        futures=[pool.submit(queue,device) for device in args.devices]
        errors=[]
        for f in futures:
            try:f.result()
            except Exception as e:errors.append(str(e))
        if errors:raise RuntimeError('\n'.join(errors))


def validate_preflights(args):
    signatures=[];records=[]
    for cell in cells(args.devices):
        if cell['seed']!=1234:continue
        root,_=worker_command(args,cell,preflight=True)
        result,meta=read_json(root/'result.json'),read_json(root/'metadata.json')
        assert result['status']=='PREFLIGHT_COMPLETED' and result['updates']==200 and result['finite_model_optimizer']
        assert meta['config']['variant']==cell['variant'] and meta['config']['physical_batch']==args.physical_batch
        assert result['source_identity']==verify_source(args.output_root)
        assert read_json(root/'restore-audit-2.json')['status']=='passed'
        env=meta['rank_environments'][0]
        signature={k:env[k] for k in ('python','torch','torch_npu','npu_name','precision')}
        signatures.append(json.dumps(signature,sort_keys=True))
        records.append(dict(**cell,shared_state=meta['config']['shared_state_sha256'],environment=env))
    assert len(set(signatures))==1,'NPU preflights differ in software/hardware/precision'
    assert len({r['shared_state'] for r in records})==1,'Native initialization differs across encoders'
    write_json(args.output_root/'port-validation.json',dict(status='passed',cells=records))


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--manifest',type=Path,default=ROOT/'configs/synthetic-split.json')
    p.add_argument('--output-root',type=Path,default=EXP/'outputs'/STUDY)
    p.add_argument('--source-cache',type=Path,default=EXP/'outputs'/'upstream-cache')
    p.add_argument('--offline',action='store_true')
    p.add_argument('--devices',type=int,nargs='+',required=True)
    p.add_argument('--physical-batch',type=int,choices=(1,2),default=2)
    p.add_argument('--preload-gib',type=float,default=12.)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--preflight-only',action='store_true')
    p.add_argument('--resume',action='store_true')
    args=p.parse_args(argv)
    if len(set(args.devices))!=len(args.devices) or not 1<=len(args.devices)<=8 or min(args.devices)<0:p.error('Use 1–8 unique allocated NPU IDs')
    if args.preload_gib<=0:p.error('Preload budget must be positive')
    for k in ('data_root','manifest','output_root','source_cache'):setattr(args,k,getattr(args,k).expanduser().resolve())
    if not args.data_root.is_dir() or not args.manifest.is_file():p.error('Dataset root and manifest must exist')
    return args


def main(argv=None):
    import fcntl
    args=parse_args(argv)
    args.output_root.parent.mkdir(parents=True,exist_ok=True)
    with (args.output_root.parent/(args.output_root.name+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prepare(args)
        print(json.dumps(dict(output=str(args.output_root),cells=cells(args.devices)),indent=2),flush=True)
        if not args.execute:
            print('Prepared only. Add --execute to run NPU validation and production.');return
        verify_source(args.output_root)
        on_devices(args,preflight_cell,preflight=True)
        validate_preflights(args)
        if args.preflight_only:return
        verify_source(args.output_root)
        try:on_devices(args,production_cell)
        finally:
            from summarize import summarize
            summarize(args.output_root)


if __name__=='__main__':main()
