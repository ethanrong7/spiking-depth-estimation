import json,os,re,subprocess
from pathlib import Path
uuid='GPU-4b32e268-a510-752a-111a-84a9220c0b52';job=os.environ['SLURM_JOB_ID'];root=Path(os.environ['STEREO_RUN_ROOT'])
queue=subprocess.check_output(['squeue','-h','-o','%u'],text=True,timeout=20).splitlines()
if any(user.strip()!='rongliang' for user in queue):
    (root/'bc-yield.request').write_text('Other user present at allocation admission\n');subprocess.run(['scancel',job]);raise SystemExit(75)
info,=[p.read_text() for p in Path('/proc/driver/nvidia/gpus').glob('*/information') if uuid in p.read_text()]
minor=re.search(r'Device Minor:\s*(\d+)',info).group(1);alloc=subprocess.check_output(['scontrol','show','job','-do',job],text=True,timeout=20)
assert 'GRES=gpu:1(IDX:'+minor+')' in alloc.split(),'Wrong physical GPU; refusing CUDA'
os.environ['CUDA_VISIBLE_DEVICES']=uuid;os.environ['PLACEMENT_GPU_UUID']=uuid
import torch
assert torch.cuda.device_count()==1 and str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-')==uuid[4:]
(root/('allocation-'+job+'.json')).write_text(json.dumps(dict(job=job,gpu_index=1,gpu_uuid=uuid,device_minor=minor,arms=['B','C']),indent=2)+'\n')
python=str(Path(os.environ['STEREO_ENV'])/'bin/python');os.execv(python,[python,str(Path(os.environ['STEREO_OPS'])/'scope_supervisor.py'),'--scope','BC'])
