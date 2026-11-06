"""Eval-only descriptor intervention panel; no optimizer or training entrypoint.

Call run_panel(head, context, left_features, right_features, truth_cells,
valid, inview, recurrent_centers) on a fixed padded feature-grid panel.
recurrent_centers must come from archived frozen Original (absolute B2HW).
All errors and offsets below are feature cells; multiply by native stride.
"""
import torch
from core.corr import CorrBlock1D


def summaries(error, valid, inview):
    return {name: dict(count=int(mask.sum()), defined_count=int((mask & torch.isfinite(error)).sum()), error_sum=float(error[mask & torch.isfinite(error)].sum()))
            for name, mask in [('valid',valid),('inview',valid & inview),('oov',valid & ~inview)]}


@torch.no_grad()
def run_panel(head, context, left_features, right_features, truth_cells, valid, inview, recurrent_centers, permuted_right_features=None, evaluate_native=None):
    assert not head.training
    b,_,h,w = left_features.shape
    yy,xx = torch.meshgrid(torch.arange(h,device=context.device),torch.arange(w,device=context.device),indexing='ij')
    grid = torch.stack([xx,yy]).float()[None].expand(b,-1,-1,-1)
    def volume(right):
        return CorrBlock1D(left_features.float(),right.float(),num_levels=head.levels,radius=head.radius)
    native = volume(right_features)
    evidence = native(grid)
    base,_,components = head(context,evidence)
    result = {'units':'feature_cells', 'panel_batch':b, 'interventions':{}}
    saved = head.override
    try:
        for name,override in [('native',None),('zero_residual','zero_residual'),('uniform','uniform')]:
            head.override=override
            disp,mask,comp=head(context,evidence)
            result['interventions'][name] = dict(error=summaries((disp[:,0]-truth_cells).abs(),valid,inview), coordinate_change=float((comp['coordinate']-components['coordinate']).abs().mean()), residual_change=float((comp['residual']-components['residual']).abs().mean()), weights_change=float((comp['weights']-components['weights']).abs().mean()))
            if evaluate_native is not None:
                result['interventions'][name]['native_metrics']=evaluate_native(disp,mask)
        head.override=None
        shifted=torch.zeros_like(right_features);shifted[...,1:]=right_features[...,:-1]
        assert b>1, 'cyclic sample permutation requires batch >=2'
        for name,right in [('right_shift_plus_one',shifted),('right_cyclic_permutation',right_features.roll(1,0) if permuted_right_features is None else permuted_right_features)]:
            corr=volume(right)(grid);disp,mask,comp=head(context,corr)
            result['interventions'][name]=dict(error=summaries((disp[:,0]-truth_cells).abs(),valid,inview),evidence_change=float((corr-evidence).abs().mean()),coordinate_change=float((comp['coordinate']-components['coordinate']).abs().mean()),residual_change=float((comp['residual']-components['residual']).abs().mean()),weights_change=float((comp['weights']-components['weights']).abs().mean()))
            if name == 'right_shift_plus_one':
                # Right descriptors move toward increasing right-minus-left flow.
                interior=valid & inview & (torch.arange(w,device=context.device)[None,None,:]>0) & (torch.arange(w,device=context.device)[None,None,:]<w-1)
                changes={'coordinate':comp['coordinate'][:,0]-components['coordinate'][:,0], 'residual':comp['residual'][:,0]-components['residual'][:,0], 'total_flow':disp[:,0]-base[:,0]}
                result['interventions'][name]['signed_shift_response']={key:dict(count=int(interior.sum()),signed_change_sum=float(delta[interior].sum()),positive_count=int(((delta>0)&interior).sum()),expected_plus_one_error_sum=float((delta-1).abs()[interior].sum())) for key,delta in changes.items()}
            if evaluate_native is not None:
                result['interventions'][name]['native_metrics']=evaluate_native(disp,mask)
        if b==2:
            errors=[]
            for i in range(b):
                one,_,_=head(context[i:i+1],evidence[i:i+1]);errors.append(float((one-base[i:i+1]).abs().max()))
            result['batch2_single_max_error']=max(errors)
        offsets=components['offsets']
        queries=[('initial',grid)]
        # Wider centers preserve the exact frozen native volume and its stencil.
        for delta in (-32,-16,16,32):
            center=grid.clone();center[:,0]+=delta;queries.append(('wider_'+str(delta),center))
        queries.extend(('original_center_'+str(i),center) for i,center in recurrent_centers.items())
        result['same_volume_support']={}
        for name,center in queries:
            corr=native(center); candidates=center[:,0,...,None]-grid[:,0,...,None]+offsets
            admissible=torch.cat([((center[:,0,...,None]/2**level+torch.arange(-head.radius,head.radius+1,device=center.device))>=0)&((center[:,0,...,None]/2**level+torch.arange(-head.radius,head.radius+1,device=center.device))<=w//2**level-1) for level in range(head.levels)],-1)
            distance=(candidates-truth_cells[...,None]).abs()
            nearest=distance.masked_fill(~admissible,float('inf')).min(-1).values
            scores=corr.permute(0,2,3,1).masked_fill(~admissible,-float('inf'))
            selected=distance.gather(-1,scores.argmax(-1,keepdim=True)).squeeze(-1)
            nearest_index=distance.masked_fill(~admissible,float('inf')).argmin(-1,keepdim=True)
            nearest_score=scores.gather(-1,nearest_index)
            rank=(((scores>nearest_score)&admissible).sum(-1)+1).float()
            nonempty=admissible.any(-1)
            selected=selected.masked_fill(~nonempty,float('nan'))
            rank=rank.masked_fill(~nonempty,float('nan'))
            result['same_volume_support'][name]=dict(nearest=summaries(nearest,valid,inview),score_selected=summaries(selected,valid,inview),nearest_rank=summaries(rank,valid,inview),empty_count=int((valid&~admissible.any(-1)).sum()))
    finally:
        head.override=saved
    return result


@torch.no_grad()
def probe_models(model, original, left, right, target, validity, original_exposure=30000, permuted_right_features=None):
    """Run loaded CUDA or CPU models on one native unpadded batch >=2.

    Caller loads retained checkpoints, controls GPU budget, and writes returned
    JSON. Original's checkpoint exposure is explicitly attached to its centers.
    """
    import torch.nn.functional as F
    import core.stereo_factor_feedforward as implementation
    from core.utils.utils import InputPadder
    assert not model.training and not original.training
    padder=InputPadder(left.shape,divis_by=32)
    padding=padder._pad
    captured={}; centers={}
    native_class=implementation.CorrBlock1D
    class CaptureVolume(native_class):
        def __init__(self, left_features, right_features, *args, **kwargs):
            captured['left']=left_features;captured['right']=right_features
            super().__init__(left_features,right_features,*args,**kwargs)
    def capture_context(module, args): captured['context']=args[0]
    def capture_center(module, args, kwargs):
        flow=kwargs.get('disp',args[3] if len(args)>3 else None)
        if flow is not None:
            b,_,h,w=flow.shape
            yy,xx=torch.meshgrid(torch.arange(h,device=flow.device),torch.arange(w,device=flow.device),indexing='ij')
            centers[len(centers)+1]=flow.detach().clone()+torch.stack([xx,yy]).to(flow)[None]
    context_hook=model.feedforward_head.register_forward_pre_hook(capture_context)
    center_hook=original.update_block.register_forward_pre_hook(capture_center,with_kwargs=True)
    try:
        implementation.CorrBlock1D=CaptureVolume
        l,r=padder.pad(left,right)
        model.set_native_geometry(left.shape,padding)
        model(l,r,iters=1,test_mode=True)
        from spikingjelly.activation_based import functional
        functional.reset_net(original)
        try:
            original(l,r,iters=32,test_mode=True)
        finally:
            functional.reset_net(original)
        stride=model.feedforward_head.stride
        truth=F.pad(target,padding)[...,::stride,::stride][:,0]/stride
        valid=F.pad(validity.bool(),padding)[...,::stride,::stride][:,0]
        x=torch.arange(target.shape[-1],device=target.device)[None,None,None]
        visible=validity.bool()&(x+target>=0)&(x+target<=target.shape[-1]-1)
        visible=F.pad(visible,padding)[...,::stride,::stride][:,0]
        def evaluate_native(disp,mask):
            prediction=padder.unpad(model.upsample_disp(disp,mask)[:,:1])
            error=(prediction-target).abs()
            result=summaries(error,validity.bool(),visible_native)
            result['per_frame']=[dict(batch_index=i,count=int(validity[i].sum()),error_sum=float(error[i][validity[i].bool()].sum()),bad3_count=int(((error[i]>3)&validity[i].bool()).sum())) for i in range(error.shape[0])]
            for name,m in [('valid',validity.bool()),('inview',validity.bool()&visible_native),('oov',validity.bool()&~visible_native)]:
                result[name]['bad3_count']=int(((error>3)&m).sum())
            return result
        visible_native=validity.bool()&(x+target>=0)&(x+target<=target.shape[-1]-1)
        result=run_panel(model.feedforward_head,captured['context'],captured['left'],captured['right'],truth,valid,visible,{i:c for i,c in centers.items() if i in (1,8,16,32)},permuted_right_features=permuted_right_features,evaluate_native=evaluate_native)
        result['original_center_checkpoint_updates']=original_exposure
        result['original_center_exposure_matched']=False
        result['original_center_scope']='30k archived Original centers; diagnostic only, not a matched-exposure control'
        return result
    finally:
        implementation.CorrBlock1D=native_class
        context_hook.remove();center_hook.remove()
