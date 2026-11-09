"""One registered native CUDA run; no automatic retries or configuration changes."""
import json,os,subprocess,sys,datetime
from pathlib import Path
root=Path(sys.argv[1]);model=sys.argv[2];gpu=sys.argv[3]
source=Path('/workspace/source-cuda-28ca-v2');exp=source/'projects/snn-depth-estimation/experiments/b-attention-mechanism-v1'
cell=root/'seed-1234'/model
assert json.loads((root/'validation.json').read_text())['status']=='passed'
assert not (cell/'job.json').exists() and not (cell/'last.pt').exists()
cell.mkdir(parents=True,exist_ok=True)
def write(value):
 tmp=cell/'job.tmp.json';tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(cell/'job.json')
record=dict(pid=os.getpid(),model=model,gpu_uuid=gpu,phase='starting',updates=30000,schedule_total=30000,resumed_update=0,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());write(record)
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,EXPECTED_GPU_UUID=gpu,OMP_NUM_THREADS='6',MKL_NUM_THREADS='6')
cmd=[sys.executable,str(exp/'worker_baseline_cuda.py'),'--seed','1234','--model',model,'--data-root','/datasets/Synthetic','--manifest',str(source/'configs/synthetic-split.json'),'--root',str(cell),'--source-registry',str(source/'source-registry.json'),'--initialization',str(root/'initialization/seed-1234.pt'),'--stop-at','30000','--input-mode','preload','--preload-gib','12','--input-workers','2','--prefetch-depth','2']
with (cell/'stdout.log').open('a') as stdout,(cell/'stderr.log').open('a') as stderr:
 child=subprocess.Popen(cmd,env=env,stdout=stdout,stderr=stderr);record.update(phase='running',worker_pid=child.pid,command=cmd);write(record);rc=child.wait()
record.update(phase='completed' if rc==0 else 'failed',exit_code=rc,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());write(record)
sys.exit(rc)
