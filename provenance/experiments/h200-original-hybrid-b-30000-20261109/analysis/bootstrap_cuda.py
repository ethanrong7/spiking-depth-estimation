"""Wait for declared inputs, verify native CUDA preflights, then launch exactly two fresh 30k runs."""
import concurrent.futures,datetime,hashlib,json,os,subprocess,sys,tarfile,time,traceback
from pathlib import Path
work=Path('/workspace');root=work/'outputs/original-hybrid-b-h200-30000-v1';root.mkdir(parents=True,exist_ok=True)
source=work/'source-cuda-28ca-v2';exp=source/'projects/snn-depth-estimation/experiments/b-attention-mechanism-v1'
models={'original-non-i':'GPU-ecddde34-aa9c-ff1d-dd55-b3f978f84762','hybrid-b':'GPU-b5d1f705-29d3-eafa-52ba-da03d11fbeab'}
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def atomic(name,value):
 p=root/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,indent=2)+'\n');t.replace(p)
def status(phase,**kwargs):atomic('bootstrap-status.json',dict(phase=phase,pid=os.getpid(),updated_utc=utc(),**kwargs))
def run(cmd,log,env=None,allowed=(0,)):
 with (root/log).open('a') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env)
 assert r.returncode in allowed,(cmd,r.returncode,str(root/log))
 return r.returncode
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
try:
 status('waiting_for_environment_and_dataset')
 while True:
  install=work/'install-v4.exit';archive=work/'train-dev-transfer.tar.gz'
  if install.exists():assert install.read_text().strip()=='0','Dependency installation failed; inspect install logs'
  if install.exists() and ((work/'dataset-ready.json').exists() or (archive.exists() and archive.stat().st_size==817462052)):break
  time.sleep(30)
 status('verifying_dataset')
 data=work/'datasets/Synthetic'
 if not (work/'dataset-ready.json').exists():
  assert digest(archive)=='73a57a71de52a21d8016612087c24f876eb82a4930dabc898bd3cd5f737fe0bf'
  if data.exists():data.rename(data.with_name('Synthetic-partial-preserved-'+str(int(time.time()))))
  data.mkdir()
  with tarfile.open(archive) as t:
   assert all(not Path(m.name).is_absolute() and '..' not in Path(m.name).parts and m.isfile() and (m.name.startswith('Train/') or m.name=='dataset-manifest.json') for m in t.getmembers())
   t.extractall(data,filter='data')
 dataset=json.loads((data/'dataset-manifest.json').read_text());assert len(dataset['files'])==3060
 for name,record in dataset['files'].items():assert (data/name).stat().st_size==record['bytes'] and digest(data/name)==record['sha256'],name
 atomic('dataset-verification.json',dict(status='passed',files=3060,raw_bytes=sum(x['bytes'] for x in dataset['files'].values()),source_archive_sha256='73a57a71de52a21d8016612087c24f876eb82a4930dabc898bd3cd5f737fe0bf',delivery='hash-verified public dataset mirror' if (work/'dataset-ready.json').exists() else 'SSH source archive',manifest_sha256=dataset['split_manifest_sha256'],scope=dataset['scope'],verified_utc=utc()))
 registry=json.loads((source/'source-registry.json').read_text())
 for name,sha in registry['source_files'].items():assert digest(source/name)==sha,name
 atomic('source-registry.json',registry)
 python='/workspace/venv/bin/python'
 run([python,'-m','pip','freeze'],'environment-packages.txt')
 env=os.environ.copy();gpu=models['original-non-i'];env.update(CUDA_VISIBLE_DEVICES=gpu,EXPECTED_GPU_UUID=gpu,OMP_NUM_THREADS='6',MKL_NUM_THREADS='6')
 status('initialization_and_output_parity')
 common=['--initialization',str(root/'initialization/seed-1234.pt'),'--data-root','/datasets/Synthetic','--manifest',str(source/'configs/synthetic-split.json'),'--output',str(root/'initialization-parity.json')]
 run([python,str(exp/'validate_cuda_initialization.py'),*common],'initialization-parity.log',env)
 assert json.loads((root/'initialization-parity.json').read_text())['status']=='passed'
 status('native_update_and_recovery_preflights')
 def preflight(item):
  model,gpu=item;env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,EXPECTED_GPU_UUID=gpu,OMP_NUM_THREADS='6',MKL_NUM_THREADS='6')
  cell=root/'preflight'/model
  cmd=[python,str(exp/'worker_baseline_cuda.py'),'--seed','1234','--model',model,'--data-root','/datasets/Synthetic','--manifest',str(source/'configs/synthetic-split.json'),'--root',str(cell),'--source-registry',str(source/'source-registry.json'),'--initialization',str(root/'initialization/seed-1234.pt'),'--stop-at','3','--preflight']
  assert not cell.exists(),'Preflight attempt already exists; inspect rather than overwrite'
  run(cmd+['--pause-at','2'],f'preflight-{model}.log',env,allowed=(75,))
  assert json.loads((cell/'checkpoint.json').read_text())['update']==2
  run(cmd,f'preflight-{model}.log',env)
  result=json.loads((cell/'result.json').read_text());restore=json.loads((cell/'restore-audit-2.json').read_text())
  assert result['status']=='PREFLIGHT_COMPLETED' and result['updates']==3 and result['finite_model_optimizer'] and restore['status']=='passed'
  return dict(model=model,status='passed',native_updates=3,exact_restore_update=2)
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:checks=list(pool.map(preflight,models.items()))
 config=dict(study='original-hybrid-b-h200-30000-v1',backend='cuda',source_revision=registry['source_revision'],source_identity=registry['source_identity_sha256'],models=list(models),seed=1234,updates=30000,schedule_total=30000,physical_batch=4,accumulation=2,effective_batch=8,device_count_per_model=1,allocation=models,execution_policy='historical_native_CUDA_FP32_cudnn_TF32_on_matmul_TF32_off_cudnn_deterministic_on_strict_off',optimizer='native AdamW defaults foreach/fused omitted',dataset='/datasets/Synthetic',output=str(root),npu_runs='untouched; separate host/backend/output',created_utc=utc())
 atomic('config.json',config);atomic('validation.json',dict(status='passed',source_identity=registry['source_identity_sha256'],checks=checks,initialization_parity='passed',dataset='3060 raw file hashes passed',validated_utc=utc()))
 status('launching_fresh_scientific_pair')
 launched=[]
 for model,gpu in models.items():
  assert not (root/'seed-1234'/model/'job.json').exists()
  with (root/f'runner-{model}.log').open('a') as f:
   child=subprocess.Popen([python,str(work/'train_cuda_one.py'),str(root),model,gpu],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  launched.append(dict(model=model,runner_pid=child.pid,gpu_uuid=gpu))
 atomic('production-launch.json',dict(started_utc=utc(),runs=launched));status('training',runs=launched)
except Exception:
 status('failed',error=traceback.format_exc());raise
