"""Numerical loader parity against measured historical serial-backward variability."""
import copy
import itertools
from fresh import finite,optimizer_steps,restore_rng,rng,rng_sha,state_sha,train_step,write
from prefetch import Prefetch,decode_batch


def tensor_deltas(first,second,per_tensor=False):
    """Global relative L2 and maxabs; comparisons use CPU float64 outside timed windows."""
    import torch
    def tensors(value,prefix=''):
        if torch.is_tensor(value):return {prefix:value}
        if isinstance(value,dict):return {n:t for k,v in value.items() for n,t in tensors(v,prefix+'/'+str(k)).items()}
        if isinstance(value,(list,tuple)):return {n:t for i,v in enumerate(value) for n,t in tensors(v,prefix+'/'+str(i)).items()}
        return {}
    a=tensors(first);b=tensors(second)
    if set(a)!=set(b):raise RuntimeError('tensor membership differs')
    parts=[];d2=0.;a2=0.;b2=0.;maximum=0.;different=0
    for name in sorted(a):
        x=a[name].detach().cpu().double();y=b[name].detach().cpu().double();delta=x-y
        d=float(delta.square().sum());nx=float(x.square().sum());ny=float(y.square().sum());m=float(delta.abs().max());count=int((x!=y).sum())
        d2+=d;a2+=nx;b2+=ny;maximum=max(maximum,m);different+=count
        if per_tensor:parts.append(dict(name=name,relative_l2=d**.5/max(nx**.5,ny**.5,1e-30),max_absolute_difference=m,l2_difference=d**.5,different_elements=count,numel=x.numel()))
    result=dict(exact=different==0,relative_l2=d2**.5/max(a2**.5,b2**.5,1e-30),max_absolute_difference=maximum,l2_difference=d2**.5,different_elements=different)
    if per_tensor:result['per_tensor']=parts
    return result


def native_parity(model,opt,scheduler,driver,recipe,dataset,order,seed,output):
    """One warm +4serial +4prefetch zero-step passes; report raw deltas before gating."""
    import torch
    initial_model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    initial_opt=copy.deepcopy(opt.state_dict());initial_scheduler=copy.deepcopy(scheduler.state_dict());initial_rng=rng()
    model_hash=state_sha(initial_model);opt_hash=state_sha(initial_opt);rng_hash=rng_sha(initial_rng)
    def reset():
        driver.reset_model(model);model.load_state_dict(initial_model,strict=True);opt.load_state_dict(copy.deepcopy(initial_opt));scheduler.load_state_dict(copy.deepcopy(initial_scheduler));restore_rng(initial_rng);opt.zero_grad(set_to_none=True)
    warm=decode_batch(dataset,recipe,order,seed,0);input_hash=state_sha(warm.items)
    warm_row=train_step(model,opt,scheduler,driver,warm.items,apply=False)
    warm_unchanged=state_sha(model.state_dict())==model_hash and state_sha(opt.state_dict())==opt_hash and rng_sha(rng())==rng_hash
    del warm
    original_clip=torch.nn.utils.clip_grad_norm_;captured={};passes=[];gradients=[]
    def capture_clip(parameters,*args,**kwargs):
        parameters=list(parameters);captured['gradients']={name:p.grad.detach().cpu().clone() for name,p in model.named_parameters() if p.grad is not None}
        return original_clip(parameters,*args,**kwargs)
    try:
        for mode in ('serial','prefetch'):
            for repeat in range(4):
                reset();before=rng_sha(rng())
                if mode=='serial':batch=decode_batch(dataset,recipe,order,seed,0)
                else:
                    with Prefetch(dataset,recipe,order,seed,0,1) as loader:batch=loader.get(0)
                actual_input=state_sha(batch.items);loading_rng_exact=rng_sha(rng())==before
                torch.nn.utils.clip_grad_norm_=capture_clip
                try:row=train_step(model,opt,scheduler,driver,batch.items,apply=False)
                finally:torch.nn.utils.clip_grad_norm_=original_clip
                gradient=captured.pop('gradients');gradients.append(gradient)
                record=dict(mode=mode,repeat=repeat,input_tensor_sha256=actual_input,indices=batch.indices,augmentation_hv=batch.flags,
                    scalars=row,unclipped_gradient_sha256=state_sha(gradient),all_gradients_finite=finite(gradient),
                    loading_rng_exact=loading_rng_exact,model_optimizer_rng_unchanged=state_sha(model.state_dict())==model_hash and state_sha(opt.state_dict())==opt_hash and rng_sha(rng())==rng_hash,
                    optimizer_step_min_max=optimizer_steps(opt),scheduler_epoch=scheduler.last_epoch)
                passes.append(record);write(output/'native-parity-passes.json',passes);del batch
    finally:torch.nn.utils.clip_grad_norm_=original_clip
    pairs=[]
    for i,j in itertools.combinations(range(8),2):
        pairs.append(dict(first=i,second=j,comparison='within_serial' if j<4 else 'within_prefetch' if i>=4 else 'serial_vs_prefetch',
            scalar_absolute_differences={key:abs(passes[i]['scalars'][key]-passes[j]['scalars'][key]) for key in ('loss','train_epe','gradient_norm','valid_pixels','lr_used','lr_next')},
            gradient=tensor_deltas(gradients[i],gradients[j],per_tensor=True)))
    write(output/'native-parity-pairs.json',pairs)
    within=max(p['gradient']['relative_l2'] for p in pairs if p['comparison']=='within_serial')
    cross=max(p['gradient']['relative_l2'] for p in pairs if p['comparison']=='serial_vs_prefetch')
    bound=min(1e-5,2*within+1e-7)
    exact_forward=all(all(p['scalars'][key]==passes[0]['scalars'][key] for key in ('loss','train_epe','valid_pixels','lr_used','lr_next')) for p in passes)
    exact_inputs=all(p['input_tensor_sha256']==input_hash and p['indices']==passes[0]['indices'] and p['augmentation_hv']==passes[0]['augmentation_hv'] for p in passes)
    exact_state=warm_unchanged and all(p['loading_rng_exact'] and p['model_optimizer_rng_unchanged'] and p['optimizer_step_min_max']==[0,0] and p['scheduler_epoch']==0 for p in passes)
    passed=exact_forward and exact_inputs and exact_state and all(p['all_gradients_finite'] for p in passes) and within<=1e-5 and cross<=bound
    result=dict(status='PASS' if passed else 'FAIL',native_zero_optimizer_backward_passes=9,optimizer_updates=0,
        all_nine_input_tensors_exact=exact_inputs,forward_loss_epe_exact=exact_forward,learned_state_optimizer_rng_exact=exact_state,
        maximum_within_serial_gradient_relative_l2=within,maximum_cross_gradient_relative_l2=cross,cross_gradient_relative_l2_bound=bound,
        absolute_gradient_relative_l2_cap=1e-5,criterion='cross<=min(1e-5,2*max_within_serial+1e-7); serial<=1e-5',
        warmup=warm_row,warmup_input_tensor_sha256=input_hash,all_gradients_finite=all(p['all_gradients_finite'] for p in passes),
        bitwise_gradients_required=False,reason='diagnostic1986 demonstrated historical serial-only native gradient variability')
    write(output/'native-parity.json',result)
    reset()
    if not passed:raise RuntimeError('registered numerical native parity gate failed; raw differences preserved')
    return result
