"""Resume-only controls around the immutable parent recipe and historical models."""
import hashlib
import importlib.util
import json
import pickle
from pathlib import Path
import sys

PARENT_ID = '83e3aadbe4f2618dd76c8bb3021b903e11e6a11a9ffbdf074aff9f1869e557f6'
SCHEDULE = dict(kind='new_segment_onecycle', updates=40000, total_steps=40100, max_lr=.0002,
                pct_start=.01, anneal_strategy='linear', cycle_momentum=False, div_factor=25., final_div_factor=10000.)
GRID = [10000,15000,20000,25000,30000,35000,40000,45000,50000]


def read(path): return json.loads(Path(path).read_text())


def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):result.update(block)
    return result.hexdigest()


def load_parent(parent_overlay, base_source):
    parent_overlay=Path(parent_overlay).resolve();registry=read(parent_overlay/'source-registry.json')
    if registry['source_identity_sha256']!=PARENT_ID:raise RuntimeError('unexpected parent recipe capsule')
    for name,expected in registry['source_files'].items():
        if sha(parent_overlay/name)!=expected:raise RuntimeError('parent recipe source mismatch: '+name)
    sys.path.insert(0,str(parent_overlay/'scripts'))
    import recipe
    import run_recipe
    driver,mechanism,base=recipe.load_frozen(base_source)
    if registry['base_source_identity_sha256']!=base['source_identity_sha256']:raise RuntimeError('historical capsule mismatch')
    return recipe,run_recipe,driver,mechanism,registry,base


def validate_config(config):
    expected=dict(seed=1234,start_updates=10000,total_updates=50000,additional_updates=40000,
        physical_batch=4,effective_batch=8,train_refinements=16,eval_refinements=32,eval_batch=2,
        weight_decay=1e-5,epsilon=1e-8,clip_grad=1.,eval_updates=GRID,scheduler=SCHEDULE)
    for key,value in expected.items():
        if config.get(key)!=value:raise RuntimeError('continuation recipe mismatch: '+key)
    if config['model'] not in ('original','original_I','spiking_normalized'):raise RuntimeError('unexpected model')


def build_scheduler(optimizer,config):
    import torch
    if config['scheduler']!=SCHEDULE:raise RuntimeError('LR binding is absent or differs from the released segment')
    return torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=.0002,total_steps=40100,pct_start=.01,
        anneal_strategy='linear',cycle_momentum=False,div_factor=25.,final_div_factor=10000.)


def optimizer_steps(optimizer):
    steps=[int(state['step'].item()) for state in optimizer.state.values() if 'step' in state]
    if not steps:raise RuntimeError('missing restored AdamW moments/counters')
    return min(steps),max(steps)


def restore_rng(state):
    import random,numpy as np,torch
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'].cpu())
    torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])


def rng_fingerprint(state):
    return hashlib.sha256(pickle.dumps(dict(python=state['python'],numpy=state['numpy'],
        torch=state['torch'].cpu().numpy(),cuda=[value.cpu().numpy() for value in state['cuda']]),protocol=4)).hexdigest()


def current_rng():
    import random,numpy as np,torch
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())


def compare_baseline(actual,parent,tolerance=1e-6):
    differences={}
    for key in ('mean_scene_epe','pooled_epe'):
        differences[key]=abs(actual[key]-parent[key])
    if set(actual['scene_epe'])!=set(parent['scene_epe']):raise RuntimeError('resumed Dev scene membership differs')
    differences['maximum_scene_epe']=max(abs(actual['scene_epe'][scene]-value) for scene,value in parent['scene_epe'].items())
    af=actual['per_frame'];pf=parent['per_frame']
    if [(r['scene'],r['frame'],r['valid_pixels']) for r in af]!=[(r['scene'],r['frame'],r['valid_pixels']) for r in pf]:raise RuntimeError('resumed Dev frame membership/validity differs')
    differences['maximum_frame_epe']=max(abs(a['epe']-p['epe']) for a,p in zip(af,pf))
    for a,p in zip(af,pf):
        for k in (1,3,5):
            if round(a[f'bad{k}']*a['valid_pixels']/100)!=round(p[f'bad{k}']*p['valid_pixels']/100):raise RuntimeError('resumed Dev bad-pixel threshold flip')
    if any(actual[f'bad{k}']!=parent[f'bad{k}'] for k in (1,3,5)):raise RuntimeError('resumed pooled bad-pixel counts differ')
    if max(differences.values())>tolerance:raise RuntimeError('parent10k reproduction failed: '+str(differences))
    return dict(status='PASS',absolute_epe_tolerance=tolerance,exact_bad_pixel_counts=True,differences=differences)
