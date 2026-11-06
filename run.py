"""Fresh stereo-depth inference, training and evaluation without campaign records."""
import argparse
import json
import os
from pathlib import Path
import subprocess


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('models', 'smoke', 'infer', 'train', 'evaluate'))
    p.add_argument('--model', default='spikestereonet')
    p.add_argument('--device', choices=('cpu','cuda'), default='cpu')
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--iterations', type=int, default=32)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--left', type=Path)
    p.add_argument('--right', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--data-root', type=Path, default=os.environ.get('SPIKE_DATA_ROOT'))
    p.add_argument('--manifest', type=Path, default=Path(__file__).resolve().parent/'configs/synthetic-split.json')
    p.add_argument('--config', type=Path, default=Path(__file__).resolve().parent/'configs/train-native.json')
    return p


def main():
    p = parser(); a = p.parse_args()
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    import runtime as rt
    if a.command == 'models':
        print('\n'.join(rt.MODELS)); return
    if a.model not in rt.MODELS:
        p.error('Unknown model; use `python run.py models`')
    if a.iterations < 1:
        p.error('--iterations must be positive')
    if a.command in ('train', 'evaluate') and (a.device != 'cuda' or not a.data_root or not a.output):
        p.error('Training/evaluation require --device cuda, --data-root and --output')
    if a.command == 'train' and a.checkpoint:
        p.error('Fresh training only; use the retained experiment code for campaign-specific resume')
    if a.command in ('infer','evaluate') and not a.checkpoint:
        p.error('Inference/evaluation require --checkpoint; use smoke for random-weight checks')
    if a.command == 'infer' and not all((a.left,a.right,a.output)):
        p.error('Inference requires --left, --right and --output')
    if a.device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA is unavailable')
    torch.set_num_threads(2)
    model = rt.build_model(a.model, a.seed).to(a.device)
    if a.checkpoint:
        # Only tensor-only model weights or checkpoints supported by the safe loader.
        state = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
        if 'model_name' in state and state['model_name'] != a.model:
            p.error('Checkpoint model_name does not match --model')
        state = state.get('model', state.get('state_dict', state))
        if state and all(k.startswith('module.') for k in state):
            state = {k[7:]:v for k,v in state.items()}
        model.load_state_dict(state, strict=True)
    if a.command in ('smoke','infer'):
        model.eval()
        if a.command == 'smoke':
            left = torch.randint(0,2,(1,50,64,128)).float()
            right = torch.randint(0,2,left.shape).float()
        else:
            left, right = rt.load_spikes(a.left)[None], rt.load_spikes(a.right)[None]
        if left.shape != right.shape:
            p.error('Left and right shapes differ')
        with torch.no_grad():
            flow, _ = rt.forward(model, a.model, left.to(a.device), right.to(a.device),
                                 min(2,a.iterations) if a.command=='smoke' else a.iterations)
        if flow.shape != (left.shape[0],1,*left.shape[-2:]) or not torch.isfinite(flow).all():
            raise RuntimeError('Invalid output shape or nonfinite prediction')
        if a.command == 'infer':
            a.output.parent.mkdir(parents=True,exist_ok=True)
            np.save(a.output, -flow[0,0].cpu().numpy())
        print(json.dumps({'model':a.model,'shape':list(flow.shape),'finite':True,
                          'kind':'synthetic shape check' if a.command=='smoke' else 'disparity x_left-x_right in pixels'}))
        return
    manifest = json.loads(a.manifest.read_text())
    if set(r['scene'] for r in manifest['train']) & set(r['scene'] for r in manifest['dev']):
        raise ValueError('Train/Dev scene overlap')
    if a.command == 'evaluate':
        model.eval(); frames=[]
        with torch.no_grad():
            for left,right,target,valid,indices in DataLoader(rt.NativeDataset(a.data_root,manifest['dev']),batch_size=1):
                flow,_ = rt.forward(model,a.model,left.cuda(),right.cuda(),a.iterations)
                error=(flow.cpu()[:,0]-target[:,0]).abs()
                if not torch.isfinite(flow).all(): raise RuntimeError('Nonfinite prediction')
                for b,i in enumerate(indices.tolist()):
                    e=error[b][valid[b]]
                    if not e.numel(): raise ValueError('Empty valid frame')
                    frames.append(dict(scene=manifest['dev'][i]['scene'],frame=manifest['dev'][i]['frame'],
                        epe=float(e.mean()),bad3=float((e>3).float().mean()*100),pixels=e.numel()))
        scenes={s:float(np.mean([x['epe'] for x in frames if x['scene']==s])) for s in sorted({x['scene'] for x in frames})}
        result={'mean_scene_epe':float(np.mean(list(scenes.values()))),'scene_epe':scenes,
                'bad3':sum(x['bad3']*x['pixels'] for x in frames)/sum(x['pixels'] for x in frames),'frames':frames}
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(result,indent=2)+'\n'); print(result['mean_scene_epe']); return
    cfg=json.loads(a.config.read_text())
    a.output.mkdir(parents=True,exist_ok=False)
    repo=Path(__file__).resolve().parent
    git=lambda *args:subprocess.check_output(['git','-C',str(repo),*args],text=True).strip()
    try: commit=git('rev-parse','HEAD')
    except subprocess.CalledProcessError: commit=None
    metadata={'model_name':a.model,'seed':a.seed,'config':cfg,'commit':commit,
              'dirty':bool(git('status','--porcelain')),'torch':str(torch.__version__),
              'gpu':torch.cuda.get_device_name(0),'manifest':str(a.manifest),
              'data_root':str(a.data_root),'recipe':'fresh portable harness; not a historical campaign reproduction'}
    (a.output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    loader=DataLoader(rt.NativeDataset(a.data_root,manifest['train']),batch_size=cfg['batch_size'],
                      shuffle=True,drop_last=True,num_workers=cfg['workers'])
    if not len(loader): raise ValueError('Training split smaller than batch size')
    batches=iter(loader)
    opt=torch.optim.AdamW((v for v in model.parameters() if v.requires_grad),lr=cfg['learning_rate'],
                          weight_decay=cfg['weight_decay'],eps=1e-8)
    schedule=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=cfg['learning_rate'],total_steps=cfg['steps'],
                 pct_start=.01,cycle_momentum=False,anneal_strategy='linear')
    model.train()
    if hasattr(model,'freeze_bn'): model.freeze_bn()
    for step in range(cfg['steps']):
        opt.zero_grad(set_to_none=True); total=0.
        for _ in range(cfg['accumulation']):
            try: batch=next(batches)
            except StopIteration: batches=iter(loader); batch=next(batches)
            left,right,target,valid=[x.to(a.device) for x in batch[:4]]
            if cfg['horizontal_flip'] or cfg['vertical_flip']:
                for b in range(left.shape[0]):
                    if cfg['horizontal_flip'] and torch.rand(()).item()<.5:
                        left[b],right[b],target[b],valid[b]=[x[b].flip(-1) for x in (left,right,target,valid)]
                        target[b].neg_()
                    if cfg['vertical_flip'] and torch.rand(()).item()<.5:
                        left[b],right[b],target[b],valid[b]=[x[b].flip(-2) for x in (left,right,target,valid)]
            _,aux=rt.forward(model,a.model,left,right,cfg['train_iterations'],training=True)
            value=rt.loss(a.model,aux,target,valid)/cfg['accumulation']
            if not torch.isfinite(value): raise RuntimeError('Nonfinite loss')
            value.backward(); total+=float(value.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'],error_if_nonfinite=True)
        opt.step();schedule.step();rt.reset(model)
        if (step+1)%cfg['log_every']==0:
            row={'step':step+1,'loss':total,'lr':schedule.get_last_lr()[0]}
            with (a.output/'train.jsonl').open('a') as stream: stream.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
        if (step+1)%cfg['save_every']==0 or step+1==cfg['steps']:
            checkpoint={'model':model.state_dict(),'model_name':a.model,'step':step+1,'config':cfg,'seed':a.seed}
            temporary=a.output/'weights.tmp.pt';torch.save(checkpoint,temporary);temporary.replace(a.output/'weights.pt')


if __name__=='__main__': main()
