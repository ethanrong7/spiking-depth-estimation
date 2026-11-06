"""Frozen E3 eligibility oracle: nested local candidates at held E2 centers."""
import argparse,hashlib,json,os,sys
from pathlib import Path
import torch
import torch.nn.functional as F

OFFSETS={8:[-8,-4,0,4,8],16:[-16,-8,-4,0,4,8,16],32:[-32,-16,-8,-4,0,4,8,16,32]}
def stats(candidates,truth,after,center,mask):
    n=int(mask.sum());low=candidates.min(0).values;high=candidates.max(0).values
    distance=(low-truth).clamp_min(0)+(truth-high).clamp_min(0)
    nearest=(candidates-truth).abs().min(0).values
    return dict(count=n,hull_distance_sum=float(distance[mask].sum()),near4_count=int(((nearest<=4)&mask).sum()),nearest_distance_sum=float(nearest[mask].sum()),bracketed_count=int(((truth>=low)&(truth<=high)&mask).sum()),actual_error_sum=float((after-truth).abs()[mask].sum()),center_error_sum=float((center-truth).abs()[mask].sum()))
def main():
    p=argparse.ArgumentParser()
    for key in ('source','root','data-root','manifest','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--arm',choices=('transformer','pointwise'),required=True);a=p.parse_args()
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.device_count()==1
    project=a.source/'projects/snn-depth-estimation';sys.path[:0]=[str(project),str(project/'experiments/spiking-transformer-stereo/scripts')]
    from core.coordinate_transformer import CoordinateStereo
    from core.s2m2_spiking import fixed_native_initial
    import run_experiment as driver
    import random,numpy as np,cv2
    random.seed(1234);np.random.seed(1234);torch.manual_seed(1234);torch.cuda.manual_seed_all(1234);torch.set_num_threads(6);cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    checkpoint=a.root/a.arm/'step-2500.pt';ck=torch.load(checkpoint,map_location='cpu',weights_only=False);assert ck['updates']==2500
    model=CoordinateStereo(a.arm=='transformer').cuda().eval();model.load_state_dict(ck['model']);identity=ck['source_identity'];del ck
    ds=driver.ManifestDataset(a.data_root,json.loads(a.manifest.read_text())['dev']);loader=torch.utils.data.DataLoader(ds,batch_size=2,num_workers=0,generator=torch.Generator().manual_seed(2225));frames=[]
    with torch.no_grad():
        for left,right,target,valid,indices in loader:
            assert torch.cuda.mem_get_info()[0]>3*1024**3
            left,right,target,valid=[x.cuda() for x in (left,right,target,valid)];out=model(left,right);driver.reset_model(model)
            pad=out['padding'];h,w=target.shape[-2:];truth=-target[:,0];x=torch.arange(w,device=target.device)[None,None,:];visible=valid&(x+target[:,0]>=0)&(x+target[:,0]<=w-1)
            coarse_truth=-F.pad(target,pad)[...,::4,::4][:,0];coarse_valid=F.pad(valid[:,None],pad)[...,::4,::4][:,0];coarse_visible=F.pad(visible[:,None],pad)[...,::4,::4][:,0]
            rows=[dict(scene=ds.rows[index]['scene'],frame=ds.rows[index]['frame'],epe=float((out['flow'][j,0]-target[j,0]).abs()[valid[j]].mean()),rounds={}) for j,index in enumerate(indices.tolist())]
            for stage in (1,2):
                trace=out['coordinate_trace'][stage];center=trace['before'];after=trace['after'];native_center=fixed_native_initial(center[:,None]/4,pad,h,w)[:,0];native_after=-out['sequence'][stage][:,0]
                for radius,offsets in OFFSETS.items():
                    candidates=torch.stack([(center+v).clamp(-512,512) for v in offsets])
                    native_candidates=torch.stack([fixed_native_initial(c[:,None]/4,pad,h,w)[:,0] for c in candidates])
                    for j in range(len(rows)):
                        groups={}
                        for resolution,c,t,before,pred,m,v in [('coarse',candidates[:,j],coarse_truth[j],center[j],after[j],coarse_valid[j],coarse_visible[j]),('native',native_candidates[:,j],truth[j],native_center[j],native_after[j],valid[j],visible[j])]:
                            groups[resolution]={name:stats(c,t,pred,before,mask) for name,mask in [('full',m),('visible',m&v),('out_of_view',m&~v)]}
                        rows[j]['rounds'].setdefault(str(stage+1),{})[str(radius)]=groups
            frames.extend(rows)
    official=next(json.loads(x) for x in (a.root/a.arm/'evaluations.jsonl').read_text().splitlines() if json.loads(x)['update']==2500);observed={(r['scene'],r['frame']):r['epe'] for r in frames};difference=max(abs(observed[(r['scene'],r['frame'])]-r['epe']) for r in official['per_frame'])
    totals={}
    for stage in ('2','3'):
        totals[stage]={}
        for radius in ('8','16','32'):
            totals[stage][radius]={}
            for resolution in ('coarse','native'):
                totals[stage][radius][resolution]={}
                for group in ('full','visible','out_of_view'):
                    items=[r['rounds'][stage][radius][resolution][group] for r in frames];sums={k:sum(x[k] for x in items) for k in items[0]};n=sums['count']
                    totals[stage][radius][resolution][group]=dict(**sums,hull_bound_epe=sums['hull_distance_sum']/max(1,n),near4_fraction=sums['near4_count']/max(1,n),actual_epe=sums['actual_error_sum']/max(1,n),scoring_slack=(sums['actual_error_sum']-sums['hull_distance_sum'])/max(1,n))
    result=dict(arm=a.arm,source_identity=identity,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),official_parity=difference<1e-5,official_max_frame_epe_difference=difference,batch_size=2,offsets=OFFSETS,selection_support='Pooled final-round coarse validanchors; native-map bounds descriptive; centersheldfromunmodifiedcheckpoint',totals=totals,per_frame=frames)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2));assert result['official_parity'];print(json.dumps({k:v for k,v in result.items() if k!='per_frame'}),flush=True)
if __name__=='__main__':main()
