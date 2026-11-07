"""One arm per allocation: disposable recovery probe, then fresh 30k training."""
import argparse, fcntl, hashlib, json, os, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
EXP=Path(__file__).resolve().parent

def main(*, worker_path=None, worker_args=()):
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=('A','B'),required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    assert os.environ['SLURM_JOB_ID'] and os.environ['PLACEMENT_GPU_UUID']
    root=Path(os.environ['STEREO_RUN_ROOT']);root.mkdir(parents=True,exist_ok=True)
    lock=(root/f'{a.arm}-allocation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    registry=json.loads((EXP.parents[3]/'source-registry.json').read_text())
    def write(name,value):
        path=root/name;tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
    def launch(stage,pause=None):
        folder=root/stage;folder.mkdir(exist_ok=True)
        command=[sys.executable,'-u',str(worker_path or EXP/'worker.py'),'--root',str(folder),'--arm',a.arm,'--data-root',os.environ['STEREO_DATA'],'--manifest',str(EXP.parent/'spiking-transformer-stereo/configs/manifest.json'),'--stop-at','30000']
        command.extend(worker_args)
        env=os.environ.copy();env.pop('SHARED_PREFLIGHT_PAUSE_AT',None)
        if stage=='preflight':command+=['--preflight']
        if pause:env['SHARED_PREFLIGHT_PAUSE_AT']=str(pause)
        write(f'{a.arm}-launch.json',dict(stage=stage,command=command,job=os.environ['SLURM_JOB_ID'],gpu_uuid=os.environ['PLACEMENT_GPU_UUID'],source_identity=registry['source_identity_sha256'],time_utc=datetime.now(timezone.utc).isoformat()))
        with (folder/f'{a.arm}.out').open('a') as log:return subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
    # Default is fresh; explicit recovery retains source/config-validated worker state.
    if not a.resume:
        assert not (root/'train'/a.arm/'last.pt').exists()
        assert not (root/'preflight'/a.arm/'last.pt').exists()
        assert launch('preflight',2)==75,'preflight pause failed'
        assert launch('preflight')==0,'preflight resume failed'
        probe=json.loads((root/'preflight'/a.arm/'result.json').read_text())
        assert probe['updates']==3 and probe['finite_model_optimizer']
        recovery=[json.loads(s) for s in (root/'preflight'/a.arm/'recoveries.jsonl').read_text().splitlines()]
        assert recovery[-1]['restored_update']==2
        write(f'{a.arm}-preflight-passed.json',dict(result=probe,recovery=recovery,production_checkpoint_reuse=False))
    else:
        assert (root/'train'/a.arm/'last.pt').exists()
        assert (root/f'{a.arm}-preflight-passed.json').exists()
    assert launch('train')==0,'production failed'
    folder=root/'train'/a.arm
    result=json.loads((folder/'result.json').read_text())
    assert result['updates']==30000 and result['finite_model_optimizer']
    checkpoint=folder/'step-30000.pt'
    final,=[json.loads(s) for s in (folder/'evaluations.jsonl').read_text().splitlines() if json.loads(s)['update']==30000]
    write(f'{a.arm}-endpoint-30000.json',dict(result=result,evaluation=final,evaluations_path=str(folder/'evaluations.jsonl'),checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),source_identity=registry['source_identity_sha256'],job=os.environ['SLURM_JOB_ID']))
if __name__=='__main__':main()
