"""300k schedule and strict bounded-stage persistence around unchanged frozen runtime."""
import hashlib
import importlib.util
import json
from pathlib import Path

RUNTIME_ID='ac8d11ac3f322194135af3c57891c1407754511f71fb951a815910850b0615ed'
PREFETCH_SHA='d9bc5d5c1e34d34c38d9a90dbce8e3ec5ad133dbeca58d50cd63ccbe602b6ddb'
ARMS=('original','original_I','spiking_normalized')
SEEDS=(1234,2345,3456)
GRID=[0,1000,2500,5000,7500,10000,15000,20000,25000,30000,35000,40000,45000,50000]
SCHEDULE=dict(kind='onecycle_from_initialization',total_steps=300100,max_lr=.0002,pct_start=.01,
              anneal_strategy='linear',cycle_momentum=False,div_factor=25.,final_div_factor=10000.)


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def append(path,value):
    with Path(path).open('a') as f:f.write(json.dumps(value,allow_nan=False)+'\n')


def load_runtime(runtime_overlay,parent_overlay,base):
    runtime_overlay=Path(runtime_overlay).resolve();registry=read(runtime_overlay/'source-registry.json')
    if registry['source_identity_sha256']!=RUNTIME_ID:raise RuntimeError('unexpected frozen runtime')
    for n,h in registry['source_files'].items():
        if sha(runtime_overlay/n)!=h:raise RuntimeError('frozen runtime changed: '+n)
    if sha(Path(__file__).with_name('prefetch.py'))!=PREFETCH_SHA:raise RuntimeError('CPU prefetch implementation changed')
    spec=importlib.util.spec_from_file_location('frozen_fresh300k_runtime',runtime_overlay/'scripts/fresh.py');rt=importlib.util.module_from_spec(spec);spec.loader.exec_module(rt)
    recipe,parent,driver,mechanism,pr,br=rt.load_science(parent_overlay,base)
    return rt,recipe,parent,driver,mechanism,registry,pr,br


def verify_dependencies(source,registration,runtime,parent,base):
    for label,registry in (('runtime',runtime),('parent',parent),('base',base)):
        if source[label+'_source_identity_sha256']!=registry['source_identity_sha256'] or registration[label+'_source_identity_sha256']!=registry['source_identity_sha256']:raise RuntimeError('dependency identity differs: '+label)
        if source[label+'_archive_sha256']!=registry['archive_sha256'] or registration[label+'_archive_sha256']!=registry['archive_sha256']:raise RuntimeError('dependency archive differs: '+label)


def validate_config(c):
    expected=dict(planned_updates=300000,initial_stage_start=0,initial_stage_target=50000,physical_batch=4,effective_batch=8,
        train_refinements=16,eval_refinements=32,eval_batch=2,weight_decay=1e-5,epsilon=1e-8,clip_grad=1.,eval_updates=GRID,
        scheduler=SCHEDULE,loader=dict(kind='one_cpu_producer_thread',queue_depth=2),precision='FP32_cudnn_TF32_on_matmul_TF32_off',
        augmentation=dict(horizontal_probability=.5,vertical_probability=.5,horizontal='mirror_both_no_swap_negate_signed_flow',vertical='mirror_both_keep_signed_flow'))
    for k,v in expected.items():
        if c.get(k)!=v:raise RuntimeError('300k recipe mismatch: '+k)
    if c['seed'] not in SEEDS or c['model'] not in ARMS:raise RuntimeError('unregistered model/seed')


def verify_source(overlay,config_path,manifest):
    overlay=Path(overlay);source=read(overlay/'source-registry.json');registration=read(overlay/'configs/registration.json');config=read(config_path);validate_config(config)
    for n,h in source['source_files'].items():
        if sha(overlay/n)!=h:raise RuntimeError('new overlay changed: '+n)
    if sha(overlay/'configs/registration.json')!=source['registration_sha256'] or sha(config_path)!=registration['config_sha256'][config['run_id']]:raise RuntimeError('registration/config mismatch')
    if sha(manifest)!=registration['manifest_sha256']:raise RuntimeError('fixed manifest changed')
    return config,registration,source


def optimizer(model):
    import torch
    opt=torch.optim.AdamW(model.parameters(),lr=.0002,weight_decay=1e-5,eps=1e-8)
    sch=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=.0002,total_steps=300100,pct_start=.01,
        anneal_strategy='linear',cycle_momentum=False,div_factor=25.,final_div_factor=10000.)
    return opt,sch


def context(config,source,registration,initialization,purpose):
    if purpose not in ('scientific_stage','engineering_resume_probe'):raise RuntimeError('unknown checkpoint purpose')
    return dict(schema='fresh300k-stage-v1',purpose=purpose,config=config,source_identity_sha256=source['source_identity_sha256'],
        registration_sha256=source['registration_sha256'],manifest_sha256=registration['manifest_sha256'],initialization=initialization)


def checkpoint(path,rt,ctx,model,opt,sch,update,order,history,best,best_update):
    import torch
    if rt.optimizer_steps(opt)!=[update,update] or sch.last_epoch!=update or sch.total_steps!=300100:raise RuntimeError('checkpoint clocks differ')
    if not rt.finite(model.state_dict()) or not rt.finite(opt.state_dict()):raise FloatingPointError('checkpoint tensors nonfinite')
    log_names=('train.jsonl','loss-components.jsonl','evaluations.jsonl','history.json','sample-order.json')
    log_binding={name:sha(Path(path).parent/name) for name in log_names}
    payload=dict(log_sha256=log_binding,context=ctx,model=model.state_dict(),optimizer=opt.state_dict(),scheduler=sch.state_dict(),updates=update,
        samples_seen=update*8,next_global_index=update,next_global_presentation=update*8,order=list(order),history=history,
        best=best,best_update=best_update,rng=rt.rng())
    path=Path(path);tmp=path.with_suffix('.tmp.pt');torch.save(payload,tmp);tmp.replace(path)
    return sha(path)


def restore(path,rt,ctx,model,opt,sch,expected_update=None):
    """Construct scheduler first, then restore both optimizer groups and full scheduler."""
    import torch
    path=Path(path);ck=torch.load(path,map_location='cpu',weights_only=False)
    if ck['context']!=ctx:raise RuntimeError('checkpoint source/config/manifest/init/purpose lineage differs')
    u=ck['updates']
    if expected_update is not None and u!=expected_update:raise RuntimeError('checkpoint boundary differs')
    if not (0<=u<=300000) or (ck['samples_seen'],ck['next_global_index'],ck['next_global_presentation'])!=(u*8,u,u*8):raise RuntimeError('checkpoint global offsets differ')
    if ck['scheduler']['total_steps']!=300100 or ck['scheduler']['last_epoch']!=u:raise RuntimeError('checkpoint original scheduler differs')
    if not rt.finite(ck['model']) or not rt.finite(ck['optimizer']):raise FloatingPointError('nonfinite restored tensors')
    model.load_state_dict(ck['model'],strict=True);opt.load_state_dict(ck['optimizer']);sch.load_state_dict(ck['scheduler'])
    if rt.state_sha(model.state_dict())!=rt.state_sha(ck['model']) or rt.state_sha(opt.state_dict())!=rt.state_sha(ck['optimizer']):raise RuntimeError('restored model or AdamW differs')
    if sch.state_dict()!=ck['scheduler'] or rt.optimizer_steps(opt)!=[u,u]:raise RuntimeError('restored scheduler/AdamW clock differs')
    actual_lrs=[g['lr'] for g in opt.param_groups];saved_lrs=[g['lr'] for g in ck['optimizer']['param_groups']]
    if actual_lrs!=saved_lrs or actual_lrs!=sch.get_last_lr():raise RuntimeError('restored LR was reset')
    rt.restore_rng(ck['rng'])
    if rt.rng_sha(rt.rng())!=rt.rng_sha(ck['rng']):raise RuntimeError('RNG restore differs')
    return ck


def strict_log_boundary(output,ck):
    """No trimming, skipping or partial-terminal tolerance on a resumable boundary."""
    output=Path(output);u=ck['updates']
    for name,h in ck['log_sha256'].items():
        if sha(output/name)!=h:raise RuntimeError('checkpoint log binding differs: '+name)
    for name in ('train.jsonl','loss-components.jsonl'):
        p=output/name;rows=[json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []
        if [r['update'] for r in rows]!=list(range(1,u+1)):raise RuntimeError('resume log does not end exactly at checkpoint: '+name)
        if name=='train.jsonl' and rows and rows[-1]['samples_seen']!=u*8:raise RuntimeError('resume presentation count differs')
    evaluations=[json.loads(x) for x in (output/'evaluations.jsonl').read_text().splitlines()]
    if evaluations!=ck['history'] or read(output/'history.json')!=ck['history']:raise RuntimeError('checkpoint/evaluation history differs')
    if read(output/'sample-order.json')['order']!=ck['order']:raise RuntimeError('checkpoint/order record differs')


def forward_signature(model,driver,rt,items):
    """Exact fixed-next-batch outputs without optimizer/RNG/model-state changes."""
    import torch
    before_model=rt.state_sha(model.state_dict());state=rt.rng();signatures=[]
    try:
        model.train();model.freeze_bn()
        with torch.no_grad():
            for offset in (0,4):
                left,right=[torch.stack([item[j] for item in items[offset:offset+4]]).cuda() for j in (0,1)]
                predictions,spikes,volts=driver.forward(model,left,right,16,True)
                signatures.append(rt.state_sha(predictions));driver.reset_model(model);del left,right,predictions,spikes,volts
    finally:
        driver.reset_model(model);rt.restore_rng(state)
    if rt.state_sha(model.state_dict())!=before_model or rt.rng_sha(rt.rng())!=rt.rng_sha(state):raise RuntimeError('forward signature changed state/RNG')
    return dict(input_tensor_sha256=rt.state_sha(items),micro_prediction_sha256=signatures)
