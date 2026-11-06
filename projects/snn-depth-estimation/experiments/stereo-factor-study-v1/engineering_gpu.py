"""Eval-only native Original versus bounded on-demand plain dot replay."""
import argparse,hashlib,json,os,sys
from pathlib import Path
import torch

def main():
 p=argparse.ArgumentParser()
 for k in ('source','data-root','manifest','shared-lock-root','reference-root','output'):p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();assert os.environ.get('SLURM_JOB_ID') and torch.cuda.device_count()==1
 if os.environ['SLURM_JOB_ID']=='2751':assert str(torch.cuda.get_device_properties(0).uuid)=='8d9598da-dfec-a680-9b2f-975c955e5130'
 torch.cuda.set_per_process_memory_fraction(.30)
 project=a.source/'projects/snn-depth-estimation';sys.path[:0]=[str(project),str(project/'experiments/shared-gpu-30k-v1'),str(project/'experiments/spiking-transformer-stereo/scripts')]
 from common import phase_lock
 from core.stereo_factor_evidence import install_evidence
 import run_experiment as d
 import cv2
 torch.set_num_threads(6);cv2.setNumThreads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
 checkpoint=a.reference_root/'original/last.pt';ck=torch.load(checkpoint,map_location='cpu',weights_only=False);assert ck['updates']==30000
 reference=d.build_model('original',1234).cuda().eval();reference.load_state_dict(ck['model']);plain=install_evidence(d.build_model('original',1234),plain=True).cuda().eval();plain.load_state_dict(ck['model']);del ck
 ds=d.ManifestDataset(a.data_root,json.loads(a.manifest.read_text())['dev'][:2]);loader=torch.utils.data.DataLoader(ds,batch_size=2,num_workers=0,generator=torch.Generator().manual_seed(2225))
 official=next(json.loads(x) for x in (a.reference_root/'original/evaluations.jsonl').read_text().splitlines() if json.loads(x)['update']==30000);expected={(r['scene'],r['frame']):r for r in official['per_frame']}
 rows=[]
 with phase_lock(a.shared_lock_root,'stereo-factor-plain-control',True,torch.cuda.empty_cache),torch.no_grad():
  for left,right,target,valid,indices in loader:
   left,right,target,valid=[v.cuda() for v in (left,right,target,valid)];native=d.forward(reference,left,right,32);d.reset_model(reference);ondemand=d.forward(plain,left,right,32);d.reset_model(plain)
   for j,index in enumerate(indices.tolist()):
    row=ds.rows[index];baseline=float((native[j,0]-target[j,0]).abs()[valid[j]].mean());probe=float((ondemand[j,0]-target[j,0]).abs()[valid[j]].mean());delta=(native[j]-ondemand[j]).abs()
    rows.append(dict(scene=row['scene'],frame=row['frame'],native_epe=baseline,plain_epe=probe,official_native_epe_difference=abs(baseline-expected[(row['scene'],row['frame'])]['epe']),plain_native_max_prediction_difference=float(delta.max()),plain_native_mean_prediction_difference=float(delta.mean())))
 result=dict(checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),rows=rows,physical_batch=2,iterations=32,native_official_parity=max(r['official_native_epe_difference'] for r in rows)<1e-5,plain_scope='Samefeatures/weights/lookupgeometry;FP32linearoperatorreorderingerrorsreportedratherthanbitwiseequivalence',peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2));assert result['native_official_parity'];print(json.dumps(result),flush=True)
if __name__=='__main__':main()
