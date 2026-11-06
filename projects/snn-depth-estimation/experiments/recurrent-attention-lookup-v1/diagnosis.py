"""Eval-only original correlation instrumentation; no optimizer or dataset access.

Load the frozen Original module before importing/using this adapter. Run only
following the pilot gate, on the same fixed Dev frames. Truth is signed flow
(x_right-x_left) at feature resolution, never positive disparity.
"""
import contextlib
import torch


def summarize(coords, scores, delta, truth, valid, levels, radius):
    """Return compact counts/sums, permitting exact pixel-weighted aggregation."""
    b, _, h, w = coords.shape
    xleft = torch.arange(w, device=coords.device)[None, None, :]
    target = xleft + truth
    inview = valid.bool() & torch.isfinite(truth) & (target >= 0) & (target <= w-1)
    offsets = torch.arange(-radius, radius+1, device=coords.device)
    candidates = torch.cat([(coords[:, 0, :, :, None]/2**i + offsets)*2**i for i in range(levels)], -1)
    # Candidate center in the full feature grid; invalid padded bins excluded.
    admissible = (candidates >= 0) & (candidates <= w-1)
    distance = (candidates-target[..., None]).abs().masked_fill(~admissible, float('inf'))
    oracle = distance.min(-1).values
    covered = inview & (oracle <= 1)
    flat_scores = scores.permute(0, 2, 3, 1).masked_fill(~admissible, -float('inf'))
    chosen = candidates.gather(-1, flat_scores.argmax(-1, keepdim=True)).squeeze(-1)
    before = coords[:, 0] - xleft - truth
    after = before + delta[:, 0]
    result = {'valid': int(valid.sum()), 'inview': int(inview.sum()), 'outofview': int((valid.bool() & ~inview).sum()), 'covered': int(covered.sum())}
    for name, mask in [('all_inview', inview), ('initial_lt1', inview & (before.abs()<1)), ('initial_1to4', inview & (before.abs()>=1) & (before.abs()<4)), ('initial_ge4', inview & (before.abs()>=4))]:
        result[name] = {'n': int(mask.sum()), 'before_abs_sum': float(before.abs()[mask].sum()), 'after_abs_sum': float(after.abs()[mask].sum()), 'before_signed_sum': float(before[mask].sum()), 'after_signed_sum': float(after[mask].sum())}
    result['covered_top_error_sum'] = float((chosen-target).abs()[covered].sum())
    result['covered_oracle_error_sum'] = float(oracle[covered].sum())
    return result


@contextlib.contextmanager
def trace_original(model, corr_module, callback, iterations=(0, 1, 7, 15, 31)):
    """Patch only in an isolated single-thread eval process; restore on exit.

    callback(iteration, coords, correlation, delta) consumes detached tensors
    immediately. corr_module is frozen core.spikestereonet; no source is edited.
    """
    if model.training:
        raise ValueError('diagnosis requires model.eval()')
    original = corr_module.CorrBlock1D
    pending = {}
    class Traced(original):
        def __call__(self, coords):
            output = super().__call__(coords)
            pending['coords'], pending['scores'] = coords.detach(), output.detach()
            return output
    counter = [0]
    def hook(module, args, output):
        iteration = counter[0]
        counter[0] += 1
        if iteration in iterations:
            callback(iteration, pending['coords'], pending['scores'], output[-1].detach(),pending.get('interpretation',{}))
    corr_module.CorrBlock1D = Traced
    attention = getattr(model, 'local_attention', None)
    old_lookup = attention.lookup if attention is not None else None
    if attention is not None:
        def lookup(pyramid, coords):
            output = old_lookup(pyramid, coords)
            pending['coords'], pending['scores'] = coords.detach(), output.detach()
            b,_,h,w=coords.shape; grouped={'raw_qk':[],'attention_weight':[]}
            offsets=torch.arange(-attention.radius,attention.radius+1,device=coords.device,dtype=coords.dtype)
            for level,(qk,qv) in enumerate(pyramid):
                rw=qk.shape[-1];x=coords[:,0,:,: ,None]/2**level+offsets
                x=x[:,None].expand(-1,attention.heads,-1,-1,-1)
                grid=torch.stack((2*x/max(rw-1,1)-1,torch.zeros_like(x)),-1).reshape(-1,1,len(offsets),2)
                logits=torch.nn.functional.grid_sample(qk.reshape(-1,1,1,rw),grid,align_corners=True).reshape(b,attention.heads,h,w,-1)
                weights=logits.softmax(-1) if attention.attention else torch.ones_like(logits)/len(offsets)
                for name,value in [('raw_qk',logits),('attention_weight',weights)]:grouped[name].append(value.mean(1).permute(0,3,1,2))
            pending['interpretation']={name:torch.cat(parts,1) for name,parts in grouped.items()}
            return output
        attention.lookup = lookup
    handle = model.update_block.register_forward_hook(hook)
    try:
        with torch.no_grad():
            yield
    finally:
        handle.remove()
        corr_module.CorrBlock1D = original
        if attention is not None: attention.lookup = old_lookup


def geometry_check():
    # Known x_right=x_left+4, with correlation peak at that absolute coordinate.
    w=24;coords=torch.zeros(1,2,1,w);coords[:,0]=torch.arange(w)+4
    truth=torch.full((1,1,w),4.);valid=torch.ones_like(truth,dtype=torch.bool)
    scores=torch.zeros(1,9,1,w);scores[:,4]=1
    r=summarize(coords,scores,torch.zeros_like(coords),truth,valid,1,4)
    assert r['covered']==20 and r['covered_top_error_sum']==0
    return {'signed_horizontal_shift':4,'inview':20,'covered':20,'top_error':0}


def main():
    import argparse, importlib.util, json, sys
    from pathlib import Path
    p=argparse.ArgumentParser()
    for name in ('checkpoint','manifest','data-root','shared-lock-root','output','official'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    project=Path(__import__('os').environ['STEREO_SOURCE'])/'projects/snn-depth-estimation'
    sys.path[:0]=[str(project),str(project/'experiments/shared-gpu-30k-v1')]
    from common import phase_lock
    from core import spikestereonet
    from core.utils.utils import InputPadder
    from types import SimpleNamespace
    import numpy as np
    import random,cv2
    torch.set_num_threads(6);cv2.setNumThreads(1)
    random.seed(1234);np.random.seed(1234);torch.manual_seed(1234);torch.cuda.manual_seed_all(1234)
    path=project/'experiments/spiking-transformer-stereo/scripts/run_experiment.py'
    sys.path.insert(0,str(path.parent))
    spec=importlib.util.spec_from_file_location('diagnosis_driver',path);driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    assert ck['updates']==2500, 'fixed pilot endpoint required'
    arm=ck['config']['arm']
    args=SimpleNamespace(hidden_dims=[128]*3,context_norm='batch',shared_backbone=False,n_snn_layers=3,n_downsample=2,corr_levels=4,corr_radius=4,corr_implementation='reg',mixed_precision=False,feature_encoder='cnn',context_encoder='cnn',matched_initialization_seed=1234,slow_fast_gru=False,recurrent_evidence=arm)
    assert __import__('os').environ['SLURM_JOB_ID']=='2740'
    assert str(torch.cuda.get_device_properties(0).uuid)=='8d9598da-dfec-a680-9b2f-975c955e5130'
    model=spikestereonet.SpikeStereoNet(args).cuda().eval();model.load_state_dict(ck['model'])
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    dataset=driver.ManifestDataset(a.data_root,json.loads(a.manifest.read_text())['dev'])
    frames=[]
    with phase_lock(a.shared_lock_root,'diagnosis-'+arm,True,torch.cuda.empty_cache),torch.no_grad():
        for batch in torch.utils.data.DataLoader(dataset,batch_size=2,num_workers=0,generator=torch.Generator().manual_seed(2225)):
            assert torch.cuda.mem_get_info()[0]>3*1024**3, "memory guard: diagnostic stopped before next batch"
            left,right,target,valid,indices=batch
            index=int(indices[0])
            left,right,target,valid=[v.cuda() for v in (left,right,target,valid)]
            padder=InputPadder(left.shape,divis_by=32)
            padded_truth=padder.pad(target)[0]
            padded_valid=torch.nn.functional.pad(valid[:,None].float(),padder._pad,mode='constant',value=0)
            baseline=driver.forward(model,left,right,32)
            driver.reset_model(model)
            traces=[]
            def callback(iteration,coords,scores,delta,interpretation):
                size=coords.shape[-2:];scale=4
                # Architectural stride is four; padded width avoids scale distortion.
                scale=padded_truth.shape[-1]/size[-1]
                truth=torch.nn.functional.interpolate(padded_truth,size=size,mode='nearest')[:,0]/scale
                mask=torch.nn.functional.interpolate(padded_valid,size=size,mode='nearest')[:,0]>0
                traces.append([dict(iteration=iteration,**summarize(coords[j:j+1],scores[j:j+1],delta[j:j+1],truth[j:j+1],mask[j:j+1],4,4),score_interpretations={name:summarize(coords[j:j+1],value[j:j+1],delta[j:j+1],truth[j:j+1],mask[j:j+1],4,4) for name,value in interpretation.items()}) for j in range(2)])
            with trace_original(model,spikestereonet,callback):pred=driver.forward(model,left,right,32)
            driver.reset_model(model)
            changed_predictions={}
            for name,other in [('zero_right',torch.zeros_like(right)),('shift_right_4',torch.roll(right,4,-1))]:
                changed_predictions[name]=driver.forward(model,left,other,32)
                driver.reset_model(model)
            for j in range(2):
                row=dataset.rows[index+j];error=(baseline[j,0]-target[j,0]).abs()[valid[j]]
                frame=dict(scene=row['scene'],frame=row['frame'],**driver.metric_values(error),trace_prediction_difference=float((pred[j]-baseline[j]).abs().max()),trace=[trace[j] for trace in traces])
                for name,changed in changed_predictions.items():
                    frame[name]=dict(prediction_abs_response=float((changed[j]-pred[j]).abs().mean()),epe=float((changed[j,0]-target[j,0]).abs()[valid[j]].mean()))
                frames.append(frame)
    official=[json.loads(line) for line in a.official.read_text().splitlines() if json.loads(line)['update']==2500][-1]
    observed={ (r['scene'],r['frame']):r['epe'] for r in frames }
    max_diff=max(abs(observed[(r['scene'],r['frame'])]-r['epe']) for r in official['per_frame'])
    result=dict(readout_note='Top-score of signed readout is not attention confidence or QK logit; do not interpret it as match probability.',batch_size=2,iterations=32,geometry_shift_check=geometry_check(),arm=arm,update=2500,official_max_frame_epe_difference=max_diff,official_parity=max_diff<1e-5,per_frame=frames)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2))
    if not result['official_parity']:print('PARITY_MISMATCH',max_diff,flush=True)

if __name__=='__main__':main()
