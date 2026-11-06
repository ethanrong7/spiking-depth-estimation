"""Eval-only E2 checkpoint interventions; immutable training modules are untouched."""
import argparse,contextlib,hashlib,importlib.util,json,os,sys
from pathlib import Path
import torch
import torch.nn.functional as F

@contextlib.contextmanager
def intervention(model,mode):
    """Capture native initial matcher before changing right descriptors only."""
    original_init=model.disp_init.forward;original_expect=model.expectation
    cache={};calls=[0]
    def cache_initial(features):
        values=original_init(features)
        cache['values']=values;cache['features']=features.detach()
        cache['state']={name:getattr(model.disp_init,name) for name in ('last_logits','last_probability','last_initial')}
        return values
    def hold_initial(features):
        for name,value in cache['state'].items():setattr(model.disp_init,name,value)
        return cache['values']
    def change_features(module,args,output):
        left,right=cache['features'].chunk(2);changed=right.clone()
        if mode=='right_shift_plus1':
            changed.zero_();changed[...,1:]=right[...,:-1]
        elif mode=='right_permutation':
            valid=model.disp_init.key_valid
            columns=valid[0,0].nonzero().flatten() if valid is not None else torch.arange(right.shape[-1],device=right.device)
            order=torch.randperm(len(columns),generator=torch.Generator().manual_seed(2226)).to(right.device)
            changed[...,columns]=right[...,columns[order]]
        return torch.cat([left,changed])
    def change_expectation(candidates,weights,valid):
        final=calls[0]%3==2;calls[0]+=1
        if final and mode=='final_uniform':weights=valid.to(weights.dtype)
        if final and mode=='final_center':
            middle=candidates.shape[-1]//2;center=candidates[...,middle:middle+1];distance=(candidates-center).abs().masked_fill(~valid,float('inf'))
            weights=torch.zeros_like(weights).scatter_(-1,distance.argmin(-1,keepdim=True),1)
        return original_expect(candidates,weights,valid)
    model.disp_init.forward=cache_initial
    try:
        yield cache,hold_initial,change_features,change_expectation
    finally:
        model.disp_init.forward=original_init;model.expectation=original_expect


def main():
    p=argparse.ArgumentParser()
    for key in ('source','root','data-root','manifest','output','shared-lock-root'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--arm',choices=('transformer','pointwise'),required=True);p.add_argument('--update',type=int,default=2500);a=p.parse_args()
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.device_count()==1
    if os.environ['SLURM_JOB_ID']=='2740':assert str(torch.cuda.get_device_properties(0).uuid)=='8d9598da-dfec-a680-9b2f-975c955e5130'
    project=a.source/'projects/snn-depth-estimation';sys.path[:0]=[str(project),str(project/'experiments/shared-gpu-30k-v1'),str(project/'experiments/spiking-transformer-stereo/scripts')]
    from core.coordinate_transformer import CoordinateStereo
    from common import phase_lock
    import run_experiment as driver
    import random,numpy as np,cv2
    torch.set_num_threads(6);cv2.setNumThreads(1);random.seed(1234);np.random.seed(1234);torch.manual_seed(1234);torch.cuda.manual_seed_all(1234)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    checkpoint=a.root/a.arm/('step-2500.pt' if a.update==2500 else 'last.pt')
    if a.update!=2500:assert json.loads((a.root/a.arm/'status.json').read_text())['phase']=='completed', 'last.pt must be terminal; use retained2500 otherwise'
    checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False);assert ck['updates']==a.update
    model_class=CoordinateStereo
    if ck['config'].get('coding_variant')=='coordinate-range9-v1':
        from core.coordinate_transformer_e3 import RangeCoordinateStereo
        model_class=RangeCoordinateStereo
    model=model_class(a.arm=='transformer').cuda().eval();model.load_state_dict(ck['model']);source_identity=ck['source_identity'];del ck
    ds=driver.ManifestDataset(a.data_root,json.loads(a.manifest.read_text())['dev']);loader=torch.utils.data.DataLoader(ds,batch_size=2,num_workers=0,generator=torch.Generator().manual_seed(2225))
    modes=('right_shift_plus1','right_permutation','final_uniform','final_center');frames=[]
    with phase_lock(a.shared_lock_root,'e2-intervention-'+a.arm,True,torch.cuda.empty_cache),torch.no_grad():
        for left,right,target,valid,indices in loader:
            assert torch.cuda.mem_get_info()[0]>3*1024**3,'diagnostic memory guard'
            left,right,target,valid=[x.cuda() for x in (left,right,target,valid)]
            # Identical uninstrumented official batch2 control precedes all hooks.
            baseline=model(left,right);driver.reset_model(model)
            records=[dict(scene=ds.rows[index]['scene'],frame=ds.rows[index]['frame'],**driver.metric_values((baseline['flow'][j,0]-target[j,0]).abs()[valid[j]]),interventions={}) for j,index in enumerate(indices.tolist())]
            for mode in modes:
                with intervention(model,mode) as (cache,hold,feature_hook,expect):
                    captured=model(left,right);driver.reset_model(model)
                    model.disp_init.forward=hold
                    handle=model.transformer.register_forward_hook(feature_hook) if mode.startswith('right_') else None
                    model.expectation=expect
                    try:changed=model(left,right)
                    finally:
                        if handle:handle.remove()
                    driver.reset_model(model)
                    assert torch.equal(changed['initial_flow'],captured['initial_flow']),'initializer was not held'
                    pad=baseline['padding'];truth=-F.pad(target,pad)[...,::4,::4][:,0]
                    mask=F.pad(valid[:,None],pad)[...,::4,::4][:,0]
                    w=target.shape[-1];x=torch.arange(w,device=target.device)[None,None,None];gt_right=x+target
                    interior=valid[:,None]&(gt_right>=4)&(gt_right<=w-5)
                    support=F.pad(interior,pad)[...,::4,::4][:,0]
                    before=captured['coordinate_trace'][-1]['after'];after=changed['coordinate_trace'][-1]['after'];movement=after-before
                    for j in range(len(records)):
                        common=support[j];n=int(common.sum());error=(changed['flow'][j,0]-target[j,0]).abs()[valid[j]]
                        records[j]['interventions'][mode]=dict(**driver.metric_values(error),common_count=n,signed_disparity_movement_sum=float(movement[j][common].sum()),correct_direction_count=int((movement[j][common]<0).sum()),shift_error_sum=float((movement[j][common]+4).abs().sum()),common_before_epe_sum=float((before[j]-truth[j]).abs()[common].sum()),common_after_epe_sum=float((after[j]-truth[j]).abs()[common].sum()),full_prediction_response=float((changed['flow'][j]-captured['flow'][j]).abs().mean()),held_capture_baseline_max_difference=float((captured['flow'][j]-baseline['flow'][j]).abs().max()))
            frames.extend(records)
    official=next(json.loads(line) for line in (a.root/a.arm/'evaluations.jsonl').read_text().splitlines() if json.loads(line)['update']==a.update)
    observed={(r['scene'],r['frame']):r['epe'] for r in frames};difference=max(abs(observed[(r['scene'],r['frame'])]-r['epe']) for r in official['per_frame'])
    result=dict(arm=a.arm,update=a.update,batch_size=2,source_identity=source_identity,checkpoint=str(checkpoint),checkpoint_sha256=checkpoint_sha256,official_max_frame_epe_difference=difference,official_parity=difference<1e-5,support='Same GT-valid native stride4 anchors whose true right coordinate is at least4px inside native bounds; fullEPE retains all official validtargets',movement_unit='native signed disparity d=xL-xR; right+1featurecell predicts delta_d=-4px',per_frame=frames)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='per_frame'}),flush=True)

if __name__=='__main__':main()
