"""Independently audit a completed fixed-30k matched baseline run before stage interpretation or continuation."""
import argparse,datetime,hashlib,importlib.util,json,math,sys,traceback
from pathlib import Path
import numpy as np
import torch

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True)
p.add_argument('--source',type=Path,required=True)
p.add_argument('--seed',type=int,required=True)
p.add_argument('--model',choices=('original-non-i','hybrid-b'),required=True)
p.add_argument('--endpoint',type=int,required=True)
a=p.parse_args()
assert a.endpoint==30000 and a.seed==1234
torch.set_num_threads(4)
folder=a.root/f'seed-{a.seed}'/a.model
report=dict(status='running',seed=a.seed,model=a.model,endpoint=a.endpoint,
    observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),checks={})
def obj(path):return json.loads(path.read_text())
def lines(path):return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def exact(x,y,path):
    if torch.is_tensor(x):assert torch.is_tensor(y) and x.shape==y.shape and x.dtype==y.dtype and torch.equal(x,y),path
    elif isinstance(x,np.ndarray):assert np.array_equal(x,y),path
    elif isinstance(x,dict):
        assert x.keys()==y.keys(),path
        for k in x:exact(x[k],y[k],path+'/'+str(k))
    elif isinstance(x,(tuple,list)):
        assert type(x)==type(y) and len(x)==len(y),path
        for i,(u,v) in enumerate(zip(x,y)):exact(u,v,path+'/'+str(i))
    else:assert x==y,(path,x,y)
def finite_tree(x,path):
    if torch.is_tensor(x):assert torch.isfinite(x).all(),path
    elif isinstance(x,dict):
        for k,v in x.items():finite_tree(v,path+'/'+str(k))
    elif isinstance(x,(tuple,list)):
        for i,v in enumerate(x):finite_tree(v,path+'/'+str(i))
    elif isinstance(x,float):assert math.isfinite(x),path
try:
    study=obj(a.root/'config.json');registry=obj(a.source/'source-registry.json')
    assert study['source_identity']==registry['source_identity_sha256']
    for name,sha in registry['source_files'].items():assert digest(a.source/name)==sha,name
    report['checks']['frozen_source']=registry['source_identity_sha256']
    result=obj(folder/'result.json');checkpoint=obj(folder/'checkpoint.json');metadata=obj(folder/'metadata.json')
    expected='COMPLETED' if a.endpoint==30000 else 'SCREEN_COMPLETED'
    assert result['status']==expected and result['endpoint_kind']==('full_schedule' if a.endpoint==30000 else 'screen')
    assert result['updates']==result['scheduler_last_epoch']==checkpoint['update']==checkpoint['scheduler_last_epoch']==a.endpoint
    assert result['samples_seen']==a.endpoint*8 and checkpoint['complete'] and result['finite_model_optimizer']
    assert result['source_identity']==registry['source_identity_sha256']
    state=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
    named_path=folder/f'step-{a.endpoint}.pt'
    named=torch.load(named_path,map_location='cpu',weights_only=False)
    for key in ('model','optimizer','scheduler','rng_by_rank','order','config','lineage','source_identity','updates','samples_seen','evaluated','model_sha256'):
        exact(state[key],named[key],'named-versus-last/'+key)
    assert state['updates']==state['scheduler']['last_epoch']==a.endpoint
    assert state['samples_seen']==8*a.endpoint and state['source_identity']==registry['source_identity_sha256']
    assert len(state['rng_by_rank'])==1
    assert state['model_sha256']==checkpoint['model_sha256']
    finite_tree(state['model'],'model');finite_tree(state['optimizer'],'optimizer')
    steps=[int(v['step']) for v in state['optimizer']['state'].values() if 'step' in v]
    assert steps and min(steps)==max(steps)==a.endpoint
    report['checks']['named_and_last_state']='exact'
    report['checks']['checkpoint_files']={path.name:dict(bytes=path.stat().st_size,sha256=digest(path)) for path in (folder/'last.pt',named_path)}
    config=state['config'];assert metadata['config']==config
    assert config['seed']==a.seed and config['model']==a.model
    assert config['effective_batch']==8 and config['world_size']==1 and config['physical_batch']==4 and config['accumulation']==2
    assert config['schedule_total']==config['scheduler']['total_steps']==30000
    assert config['dtype']=='float32' and config['train_iterations']==16 and config['dev_iterations']==32
    assert config['input_pipeline']['mode']=='preload'
    assert config['execution']==dict(deterministic_algorithms=False,cudnn_deterministic=True,cudnn_benchmark=False,cudnn_allow_tf32=True,matmul_allow_tf32=False)
    assert metadata['source_registry']['source_identity_sha256']==registry['source_identity_sha256']
    assert len(metadata['rank_environments'])==1
    for env in metadata['rank_environments']:
        precision=env['precision']
        assert not precision['deterministic_algorithms'] and precision['cudnn_deterministic'] and not precision['cudnn_benchmark']
        assert precision['conv_allow_tf32'] and not precision['matmul_allow_tf32'] and not precision['mixed_precision']
    assert config['updates']==30000 and config['device_count']==1
    assert config['loss_implementation']=='native_boolean_indexed_masked_mean'
    assert config['checkpoint_every']==500 and config['diagnostic_every']==100
    assert config['optimizer']['betas']==[.9,.999] and not config['optimizer']['amsgrad']
    assert config['optimizer']['foreach'] is None and config['optimizer']['fused'] is None
    assert config['backend']=='cuda'
    assert not state.get('lineage'), 'This fixed protocol does not permit source/policy transitions'
    assert config['initial_native_state_sha256']=='8ceac15b09ce188f9ee52ad5a8eca8e68727cbbb2887a012874d7246dc6495ca'
    assert config['shared_state_sha256']=='2085df6ea20073c3a9804d79f6baee94e8e9bfe41facda47e3f292fd3cc9394d'
    if a.model=='original-non-i':assert not any(n.startswith('placement.') for n in state['model'])
    else:assert config['placement_seed']==2017 and any(n.startswith('placement.') for n in state['model'])
    retained=torch.load(folder/'step-7500.pt',map_location='cpu',weights_only=False)
    assert retained['updates']==7500 and retained['samples_seen']==60000 and retained['config']==config
    assert retained['source_identity']==registry['source_identity_sha256']
    finite_tree(retained['model'],'retained-7500-model')
    report['checks']['retained_7500_checkpoint']=dict(sha256=digest(folder/'step-7500.pt'),bytes=(folder/'step-7500.pt').stat().st_size)
    del retained
    report['checks']['current_configuration']=config
    protocol_path=a.source/'projects/snn-depth-estimation/experiments/spiking-attention-placement-phase2/protocol.py'
    spec=importlib.util.spec_from_file_location('endpoint_protocol',protocol_path)
    protocol=importlib.util.module_from_spec(spec);spec.loader.exec_module(protocol)
    assert protocol.tensor_hash(state['model'])==state['model_sha256']
    stream=protocol.audit_history(folder/'train.jsonl',state['order'],a.seed,a.endpoint)
    training=lines(folder/'train.jsonl')
    for row in training:
        finite_tree(row,'train-row')
        assert row['seconds']>0 and row['gradient_norm']>0
        assert row['physical_batch']==4 and row['accumulation']==2 and row['world_size']==1
    report['checks']['all_training_presentations_flips_and_lr']=stream
    evaluations=lines(folder/'evaluations.jsonl')
    expected_evals=[n for n in protocol.EVALS if n<=a.endpoint]
    assert [row['update'] for row in evaluations]==expected_evals
    assert state['evaluated']==checkpoint['evaluated']==result['evaluated']==expected_evals
    manifest_path=a.source/'configs/synthetic-split.json'
    assert digest(manifest_path)==config['manifest_sha256']==protocol.MANIFEST_SHA
    manifest=obj(manifest_path)
    expected_frames={(str(r['scene']),str(r['frame'])) for r in manifest['dev']}
    assert len(expected_frames)==60 and len({r['scene'] for r in manifest['dev']})==30
    for row in evaluations:
        finite_tree(row,'evaluation')
        frames=row['per_frame']
        assert len(frames)==60 and {(str(r['scene']),str(r['frame'])) for r in frames}==expected_frames
        assert row['samples_seen']==row['update']*8
        assert all(f['valid_pixels']>0 and all(0<=f[k]<=100 for k in ('bad1','bad3','bad5')) for f in frames)
        scenes={s:float(np.mean([f['epe'] for f in frames if f['scene']==s])) for s in sorted({f['scene'] for f in frames})}
        assert row['scene_epe'].keys()==scenes.keys()
        for scene,value in scenes.items():assert math.isclose(row['scene_epe'][scene],value,rel_tol=1e-10,abs_tol=1e-10)
        assert math.isclose(row['mean_scene_epe'],float(np.mean(list(scenes.values()))),rel_tol=1e-10,abs_tol=1e-10)
        count=sum(f['valid_pixels'] for f in frames)
        for metric,frame_metric in [('pooled_epe','epe'),('bad1','bad1'),('bad3','bad3'),('bad5','bad5')]:
            value=sum(f[frame_metric]*f['valid_pixels'] for f in frames)/count
            assert math.isclose(row[metric],value,rel_tol=1e-10,abs_tol=1e-10)
    report['checks']['complete_dev_evaluations']=dict(updates=expected_evals,frames_each=60,scenes_each=30)
    report['execution_history']=[dict(update=t['update'],from_source=t['from_source'],to_source=t['to_source'],reason=t['reason']) for t in state.get('lineage',[])]
    report['homogeneous_execution_policy_from_initialization']=not state.get('lineage')
    report['endpoint_metrics']={k:evaluations[-1][k] for k in ('update','mean_scene_epe','pooled_epe','bad1','bad3','bad5','scene_epe')}
    report['status']='passed'
except Exception:
    report['status']='failed';report['error']=traceback.format_exc()
finally:
    dest=a.root/f'endpoint-audit-seed{a.seed}-{a.model}-step{a.endpoint}.json'
    dest.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps(report,indent=2,allow_nan=False))
if report['status']!='passed':sys.exit(1)
