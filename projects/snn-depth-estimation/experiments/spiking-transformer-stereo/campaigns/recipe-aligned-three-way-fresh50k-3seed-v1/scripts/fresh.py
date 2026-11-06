"""Fresh 50k controls; scientific models/data/loss/evaluation remain in frozen capsules."""
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import random
import sys
import time

PARENT_ID='83e3aadbe4f2618dd76c8bb3021b903e11e6a11a9ffbdf074aff9f1869e557f6'
SEEDS=(1234,2345,3456)
ARMS=('original','original_I','spiking_normalized')
GRID=[0,1000,2500,5000,7500,10000,15000,20000,25000,30000,35000,40000,45000,50000]
SCHEDULE=dict(kind='onecycle_from_initialization',total_steps=50100,max_lr=.0002,pct_start=.01,
              anneal_strategy='linear',cycle_momentum=False,div_factor=25.,final_div_factor=10000.)


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()


def write(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def append(path,value):
    with Path(path).open('a') as stream:stream.write(json.dumps(value,allow_nan=False)+'\n')


def load_science(parent_overlay,base):
    parent_overlay=Path(parent_overlay).resolve();registry=read(parent_overlay/'source-registry.json')
    if registry['source_identity_sha256']!=PARENT_ID:raise RuntimeError('unexpected historical recipe source')
    for name,expected in registry['source_files'].items():
        if sha(parent_overlay/name)!=expected:raise RuntimeError('historical recipe source changed: '+name)
    sys.path.insert(0,str(parent_overlay/'scripts'))
    import recipe
    import run_recipe
    driver,mechanism,base_registry=recipe.load_frozen(base)
    if registry['base_source_identity_sha256']!=base_registry['source_identity_sha256']:raise RuntimeError('historical model source changed')
    return recipe,run_recipe,driver,mechanism,registry,base_registry


def verify_dependencies(source,registration,parent,base):
    if source['parent_source_identity_sha256']!=parent['source_identity_sha256'] or parent['source_identity_sha256']!=registration['parent_recipe_source_identity_sha256']:raise RuntimeError('composed recipe source binding differs')
    if source['base_source_identity_sha256']!=base['source_identity_sha256'] or base['source_identity_sha256']!=registration['base_source_identity_sha256']:raise RuntimeError('composed model source binding differs')
    if source['parent_archive_sha256']!=parent['archive_sha256'] or source['parent_archive_sha256']!=registration['parent_archive_sha256']:raise RuntimeError('composed recipe archive differs')
    if source['base_archive_sha256']!=base['archive_sha256'] or source['base_archive_sha256']!=registration['base_archive_sha256']:raise RuntimeError('composed model archive differs')


def validate_config(c):
    expected=dict(total_updates=50000,physical_batch=4,effective_batch=8,train_refinements=16,eval_refinements=32,
        eval_batch=2,weight_decay=1e-5,epsilon=1e-8,clip_grad=1.,eval_updates=GRID,scheduler=SCHEDULE,
        loader=dict(kind='one_cpu_producer_thread',queue_depth=2),precision='FP32_cudnn_TF32_on_matmul_TF32_off',
        augmentation=dict(horizontal_probability=.5,vertical_probability=.5,horizontal='mirror_both_no_swap_negate_signed_flow',vertical='mirror_both_keep_signed_flow'))
    for key,value in expected.items():
        if c.get(key)!=value:raise RuntimeError('fresh recipe mismatch: '+key)
    if c['seed'] not in SEEDS or c['model'] not in ARMS:raise RuntimeError('unregistered arm/seed')


def verify_source(source_path,config_path,registration_path,manifest_path):
    source=read(source_path);root=Path(source_path).parent;reg=read(registration_path);c=read(config_path)
    validate_config(c)
    for name,expected in source['source_files'].items():
        if sha(root/name)!=expected:raise RuntimeError('source mismatch: '+name)
    if sha(config_path)!=reg['config_sha256'][c['run_id']] or sha(registration_path)!=source['registration_sha256']:raise RuntimeError('configuration registration changed')
    if sha(manifest_path)!=reg['manifest_sha256']:raise RuntimeError('manifest changed')
    return c,reg,source


def setup_cuda(seed):
    import numpy as np,torch,os
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('one Slurm CUDA GPU per worker required')
    if '5090' not in torch.cuda.get_device_name(0):raise RuntimeError('registered RTX5090 required')
    torch.set_num_threads(8);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)


def rng():
    import numpy as np,torch
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    import numpy as np,torch
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'].cpu())
    torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda']])


def rng_sha(state):
    return hashlib.sha256(pickle.dumps(dict(python=state['python'],numpy=state['numpy'],torch=state['torch'].cpu().numpy(),cuda=[v.cpu().numpy() for v in state['cuda']]),protocol=4)).hexdigest()


def optimizer(model):
    import torch
    opt=torch.optim.AdamW(model.parameters(),lr=.0002,weight_decay=1e-5,eps=1e-8)
    sch=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=.0002,total_steps=50100,pct_start=.01,
        anneal_strategy='linear',cycle_momentum=False,div_factor=25.,final_div_factor=10000.)
    return opt,sch


def optimizer_steps(opt):
    values=[int(v['step'].item()) for v in opt.state.values() if 'step' in v]
    return [min(values),max(values)] if values else [0,0]


def finite(value):
    import math,torch
    if torch.is_tensor(value):return bool(torch.isfinite(value).all())
    if isinstance(value,dict):return all(finite(v) for v in value.values())
    if isinstance(value,(list,tuple)):return all(finite(v) for v in value)
    return math.isfinite(value) if isinstance(value,float) else True


def state_sha(value):
    """Stable recursive tensor/value fingerprint, including AdamW moments and groups."""
    import torch
    h=hashlib.sha256()
    def feed(v):
        if torch.is_tensor(v):
            h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(v,dict):
            for key in sorted(v,key=str):h.update(repr(key).encode());feed(v[key])
        elif isinstance(v,(list,tuple)):
            h.update(type(v).__name__.encode())
            for x in v:feed(x)
        else:h.update(repr(v).encode())
    feed(value);return h.hexdigest()


def train_step(model,opt,scheduler,driver,items,apply=True,gradient_proof=False):
    """Exact historical pooled-valid effective8 objective; one clip/AdamW/scheduler."""
    import numpy as np,torch
    begun=time.monotonic();model.train();model.freeze_bn();opt.zero_grad(set_to_none=True)
    n=sum(int(item[3].sum()) for item in items)
    if n<=0:raise RuntimeError('no valid effective8 pixels')
    disparity=0.;error=0.;diagnostics=[]
    for offset in (0,4):
        micro=items[offset:offset+4]
        left,right,target,valid=[torch.stack([item[j] for item in micro]).cuda() for j in range(4)]
        preds,spikes,volts=driver.forward(model,left,right,16,True)
        loss=driver.disparity_loss(preds,target,valid)*(int(valid.sum())/n)
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite disparity loss')
        loss.backward();disparity+=float(loss.detach());error+=float((preds[-1]-target).abs()[:,0][valid].sum())/n
        if hasattr(model.update_block,'diagnostics'):diagnostics.append(model.update_block.diagnostics())
        driver.reset_model(model);del micro,left,right,target,valid,preds,spikes,volts,loss
    proof=state_sha({name:p.grad for name,p in model.named_parameters() if p.grad is not None}) if gradient_proof else None
    grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
    if not np.isfinite(grad) or grad<=0:raise FloatingPointError('invalid gradient')
    lr_used=opt.param_groups[0]['lr']
    if apply:opt.step();scheduler.step()
    torch.cuda.synchronize()
    row=dict(loss=disparity,disparity_loss=disparity,neuron_loss=0.,train_epe=error,valid_pixels=n,gradient_norm=grad,
        lr_used=lr_used,lr_next=scheduler.get_last_lr()[0],lr=scheduler.get_last_lr()[0],compute_seconds=time.monotonic()-begun,
        peak_memory_mib=torch.cuda.max_memory_allocated()/2**20,diagnostics=diagnostics)
    if gradient_proof:row['unclipped_gradient_sha256']=proof
    if not finite(row):raise FloatingPointError('nonfinite training record')
    return row
