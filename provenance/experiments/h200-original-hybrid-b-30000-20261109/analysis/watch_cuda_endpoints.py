"""Observe registered CUDA jobs; audit only completed 30k endpoints; never launch training."""
import datetime,json,os,subprocess,time,traceback
from pathlib import Path
root=Path('/workspace/outputs/original-hybrid-b-h200-30000-v1');root.mkdir(parents=True,exist_ok=True)
models=('original-non-i','hybrid-b');done={}
def obj(p):
 try:return json.loads(p.read_text())
 except FileNotFoundError:return None
def status(phase,**extra):
 p=root/'endpoint-watch.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(dict(pid=os.getpid(),phase=phase,models=done,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra),indent=2)+'\n');tmp.replace(p)
try:
 while len(done)<2:
  for m in models:
   if m in done:continue
   job=obj(root/'seed-1234'/m/'job.json')
   if job and job['phase']=='failed':status('training_failed',failed_model=m,job=job);raise SystemExit(1)
   if job and job['phase']=='completed':
    assert job['exit_code']==0
    with (root/f'audit-{m}.log').open('w') as log:r=subprocess.run(['/workspace/venv/bin/python','/workspace/audit_cuda_endpoint.py','--root',str(root),'--source','/workspace/source-cuda-28ca-v2','--seed','1234','--model',m,'--endpoint','30000'],stdout=log,stderr=subprocess.STDOUT)
    audit=obj(root/f'endpoint-audit-seed1234-{m}-step30000.json');assert r.returncode==0 and audit['status']=='passed',(m,audit)
    done[m]='audit_passed'
  if len(done)<2:status('waiting_for_registered_30k_completions');time.sleep(30)
 with (root/'analysis.log').open('w') as log:r=subprocess.run(['/workspace/venv/bin/python','/workspace/analyze_cuda_results.py','--root',str(root)],stdout=log,stderr=subprocess.STDOUT)
 assert r.returncode==0
 status('analysis_completed')
except Exception:status('failed',error=traceback.format_exc());raise
