"""One Slurm GPU, two workers; actual-memory-triggered checkpointed fallback."""
from pathlib import Path
import argparse,json,os,signal,subprocess,sys,time
from datetime import datetime,timezone
from common import atomic_json,append
ARMS=('original','s2m2_v2b')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('root','data-root','manifest'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--preflight',action='store_true');p.add_argument('--sequential',action='store_true');a=p.parse_args();a.root.mkdir(parents=True,exist_ok=True)
    assert os.environ.get('SLURM_JOB_ID');script=Path(__file__).with_name('worker.py')
    stop=[False];signal.signal(signal.SIGUSR1,lambda *args:stop.__setitem__(0,True));signal.signal(signal.SIGTERM,lambda *args:stop.__setitem__(0,True))
    peak=0.;mode='shared';records=[]
    def event(kind,**kwargs):append(a.root/'controller-events.jsonl',dict(event=kind,time_utc=datetime.now(timezone.utc).isoformat(),**kwargs))
    def start(arm,probe=False):
        folder=a.root/arm;folder.mkdir(exist_ok=True)
        env=os.environ.copy()
        if probe:env['SHARED_PREFLIGHT_PAUSE_AT']='2'
        else:env.pop('SHARED_PREFLIGHT_PAUSE_AT',None)
        cmd=[sys.executable,str(script),'--root',str(a.root),'--data-root',str(a.data_root),'--manifest',str(a.manifest),'--arm',arm]
        if a.preflight:cmd.append('--preflight')
        stream=(folder/'worker.log').open('a');proc=subprocess.Popen(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT);stream.close();event('worker_started',arm=arm,pid=proc.pid,probe=probe,mode=mode);return proc
    def memory(procs):
        nonlocal peak
        metas=[a.root/arm/'metadata.json' for arm in ARMS]
        meta=next((json.loads(f.read_text()) for f in metas if f.exists()),None)
        if not meta:return False
        uuid=meta['gpu_uuid'].replace('GPU-','')
        txt=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True)
        row=next(line.split(',') for line in txt.splitlines() if uuid in line)
        used,total,util=[float(x.strip()) for x in row[1:]];peak=max(peak,used)
        status=dict(mode=mode,job=os.environ['SLURM_JOB_ID'],gpu_uuid=uuid,used_mib=used,total_mib=total,headroom_mib=total-used,peak_combined_mib=peak,utilization=util,pids={arm:proc.pid for arm,proc in procs.items()},time_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(a.root/'gpu-health.json',status);append(a.root/'gpu-memory.jsonl',status)
        return used>total-2048
    def stop_safely(procs):
        (a.root/'pause.request').touch();deadline=time.monotonic()+180
        while any(p.poll() is None for p in procs.values()) and time.monotonic()<deadline:time.sleep(1)
        for arm,proc in procs.items():
            if proc.poll() is None:proc.terminate();event('pause_deadline',arm=arm)
        for proc in procs.values():proc.wait(timeout=90)
        for f in a.root.glob('validation-*.request'):f.unlink(missing_ok=True)
    def group(arms,probe=False):
        procs={arm:start(arm,probe) for arm in arms};high=0
        while True:
            codes={arm:p.poll() for arm,p in procs.items()}
            if stop[0]:stop_safely(procs);return 'allocation_stop'
            failures={arm:code for arm,code in codes.items() if code not in (None,0,75,86)}
            if failures:stop_safely(procs);event('scientific_or_runtime_failure',codes=failures);raise RuntimeError(f'worker failure {failures}')
            if any(code==86 for code in codes.values()):
                stop_safely(procs);event('memory_fallback',reason='worker_CUDA_OOM',codes=codes);return 'memory'
            high=high+1 if memory(procs) else 0
            if high>=3 and mode=='shared':
                stop_safely(procs);event('memory_fallback',reason='actual_headroom_below2048MiB_for3samples');return 'memory'
            if all(code is not None for code in codes.values()):
                if probe and all(code==75 for code in codes.values()):return 'probe_paused'
                if all(code==0 for code in codes.values()):return 'complete'
                raise RuntimeError(f'unexpected worker state {codes}')
            time.sleep(1 if a.preflight else 3)
    def sequential():
        nonlocal mode
        mode='sequential';(a.root/'pause.request').unlink(missing_ok=True)
        for arm in ARMS:
            result=a.root/arm/'result.json'
            if result.exists() and json.loads(result.read_text())['status']=='COMPLETED':continue
            if group([arm])!='complete':raise RuntimeError('single-model memory failure; preserving checkpoints and stopping')
    if a.sequential:sequential();result='complete'
    else:result=group(ARMS,probe=a.preflight)
    if result=='allocation_stop':atomic_json(a.root/'controller-result.json',dict(status='PAUSED_COMPLETE_CHECKPOINTS'));return 75
    if result=='memory':sequential()
    elif result=='probe_paused':
        event('preflight_complete_checkpoint_restart',description='both workers resume2→3 with full model/Adam/scheduler/RNG')
        result=group(ARMS)
        if result=='memory':sequential()
        elif result!='complete':return 75
    for arm in ARMS:
        r=json.loads((a.root/arm/'result.json').read_text());assert r['status']=='COMPLETED' and r['updates']==(3 if a.preflight else 30000)
    phases=[json.loads(s) for s in (a.root/'gpu-phases.jsonl').read_text().splitlines()];active={};overlap=False
    for e in phases:
        if e['event']=='start':
            assert not (e['validation'] and active),'validation overlapped active GPU step'
            assert not any(active.values()),'training overlapped validation'
            if active:overlap=True
            active[e['arm']]=e['validation']
        else:active.pop(e['arm'],None)
    atomic_json(a.root/'controller-result.json',dict(status='PASS' if a.preflight else 'COMPLETED',mode=mode,peak_combined_mib=peak,concurrent_train_intervals_observed=overlap,validation_exclusive_verified=True,updates_per_arm=3 if a.preflight else 30000,shared_gpu_count=1))
    return 0
if __name__=='__main__':sys.exit(main())
