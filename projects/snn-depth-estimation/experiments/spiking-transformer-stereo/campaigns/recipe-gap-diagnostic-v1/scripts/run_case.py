"""Registered short recipe-factor diagnostic; scientific execution is Slurm CUDA only."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def append(path, value):
    with path.open('a') as stream:
        stream.write(json.dumps(value, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser()
    for name in ('overlay', 'parent', 'base', 'data-root', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--case', required=True)
    args = parser.parse_args()
    source = json.loads((args.overlay / 'source-registry.json').read_text())
    for rel, expected in source['source_files'].items():
        assert digest(args.overlay / rel) == expected, rel
    registration = json.loads((args.overlay / 'configs/registration.json').read_text())
    config = next(c for c in registration['cases'] if c['id'] == args.case)
    assert config['updates'] == 480 and config['seed'] == 1234
    assert config['batch'] in (2, 8) and config['augmentation'] in ('none', 'hv')
    assert config['schedule_total'] in (10100, 300100)
    parent_registry = json.loads((args.parent / 'source-registry.json').read_text())
    assert parent_registry['source_identity_sha256'] == registration['parent_identity']
    for rel, expected in parent_registry['source_files'].items():
        assert digest(args.parent / rel) == expected, rel
    spec = importlib.util.spec_from_file_location('recipe', args.parent / 'scripts/recipe.py')
    recipe = importlib.util.module_from_spec(spec); spec.loader.exec_module(recipe)
    driver, mechanism, base_registry = recipe.load_frozen(args.base)
    assert base_registry['source_identity_sha256'] == registration['base_identity']
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    assert torch.cuda.device_count() == 1 and '5090' in torch.cuda.get_device_name(0)
    assert socket.gethostname() == 'cccl3'
    manifest_path = args.base / 'projects/snn-depth-estimation/experiments/spiking-transformer-stereo/configs/manifest.json'
    assert digest(manifest_path) == registration['manifest_sha256']
    manifest = json.loads(manifest_path.read_text())
    assert len(manifest['train']) == 960 and len(manifest['dev']) == 60
    output = args.output / args.case; output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); seed = config['seed']; batch = config['batch']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = True
    model = recipe.build_model(driver, mechanism, config['model'], seed)
    def state_hash():
        h = hashlib.sha256()
        for name, tensor in sorted(model.state_dict().items()):
            h.update(name.encode()); h.update(str(tensor.dtype).encode())
            h.update(str(tuple(tensor.shape)).encode()); h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    initial_hash = state_hash(); model = model.cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-4, total_steps=config['schedule_total'],
        pct_start=.01, cycle_momentum=False, anneal_strategy='linear', div_factor=25., final_div_factor=1e4)
    train = driver.ManifestDataset(args.data_root, manifest['train'])
    dev = driver.ManifestDataset(args.data_root, manifest['dev'])
    loader = DataLoader(dev, batch_size=2, num_workers=0, generator=torch.Generator().manual_seed(seed + 991))
    order = torch.randperm(960, generator=torch.Generator().manual_seed(seed)).tolist()
    write(output / 'metadata.json', dict(config=config, source=source, registration=registration,
        parent_identity=registration['parent_identity'], base_identity=registration['base_identity'],
        manifest_sha256=digest(manifest_path), initial_state_sha256=initial_hash, order=order,
        torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
        environment=os.environ['STEREO_ENV'], python=sys.version, command=sys.argv,
        pip_freeze=subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True),
        slurm={k:v for k,v in os.environ.items() if k.startswith('SLURM_')},
        hostname=socket.gethostname(), started_utc=datetime.now(timezone.utc).isoformat()))
    # Same warm-Dev convention as the recipe-aligned campaign; verify it changes no saved tensor.
    model.eval(); warm = next(iter(loader))
    with torch.no_grad(): driver.forward(model, warm[0].cuda(), warm[1].cuda(), 32)
    driver.reset_model(model); del warm
    assert state_hash() == initial_hash
    history = []
    def evaluate(update):
        begun = time.monotonic()
        row = dict(update=update, samples_seen=update*batch, **driver.evaluate(model, loader, 32))
        row['seconds'] = time.monotonic() - begun
        append(output / 'evaluations.jsonl', row); history.append(row)
        print(args.case, 'DEV', update, row['mean_scene_epe'], flush=True)
    evaluate(0)
    def prepare(step):
        indices = recipe.batch_indices(order, step, batch)
        flags = [recipe.paired_flags(seed, step*batch+i) if config['augmentation']=='hv' else (False, False) for i in range(batch)]
        items = [recipe.augment(train[index], *flag) for index, flag in zip(indices, flags)]
        return indices, flags, items
    rows = []
    physical = min(batch, 4)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='cpu-input') as pool:
        pending = {i:pool.submit(prepare, i) for i in range(2)}
        for step in range(480):
            begun = time.monotonic(); wait = time.monotonic()
            indices, flags, items = pending.pop(step).result(); wait = time.monotonic() - wait
            if step+2 < 480: pending[step+2] = pool.submit(prepare, step+2)
            model.train(); model.freeze_bn(); opt.zero_grad(set_to_none=True)
            total_valid = sum(int(item[3].sum()) for item in items)
            assert total_valid > 0
            loss_value = 0.; epe = 0.
            for offset in range(0, batch, physical):
                micro = items[offset:offset+physical]
                left, right, target, valid = [torch.stack([item[j] for item in micro]).cuda() for j in range(4)]
                preds, spikes, volts = driver.forward(model, left, right, 16, True)
                count = int(valid.sum()); loss = driver.disparity_loss(preds, target, valid) * (count/total_valid)
                assert bool(torch.isfinite(loss))
                loss.backward(); loss_value += float(loss.detach())
                epe += float((preds[-1]-target).abs()[:,0][valid].sum())/total_valid
                driver.reset_model(model)
                del preds, spikes, volts, loss, left, right, target, valid, micro
            grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.))
            assert math.isfinite(grad) and grad > 0
            lr_used = opt.param_groups[0]['lr']; opt.step(); scheduler.step(); torch.cuda.synchronize()
            row = dict(update=step+1, samples_seen=(step+1)*batch, indices=indices, augmentation_hv=flags,
                loss=loss_value, train_epe=epe, gradient_norm=grad, clipped=grad>1., valid_pixels=total_valid,
                lr_used=lr_used, lr_next=scheduler.get_last_lr()[0], seconds=time.monotonic()-begun,
                consumer_wait_seconds=wait, peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
            append(output / 'train.jsonl', row); rows.append(row); del items
            if step+1 in (120, 480): evaluate(step+1)
    # Fixed Train perturbations: no optimizer steps. Compare gradients with identical examples.
    probe_items = [train[index] for index in order[:8]]
    gradients = {}; perturbations = {}
    for label, flags in [('identity',(False,False)), ('H',(True,False)), ('V',(False,True)), ('HV',(True,True))]:
        model.train(); model.freeze_bn(); opt.zero_grad(set_to_none=True)
        items = [recipe.augment(item, *flags) for item in probe_items]
        count_all = sum(int(item[3].sum()) for item in items)
        loss_sum=0.; error_sum=0.; sign_sum=0
        for offset in range(0,8,2):
            micro=items[offset:offset+2]
            left,right,target,valid=[torch.stack([item[j] for item in micro]).cuda() for j in range(4)]
            preds,spikes,volts=driver.forward(model,left,right,16,True)
            loss=driver.disparity_loss(preds,target,valid)*(int(valid.sum())/count_all)
            loss.backward();loss_sum+=float(loss.detach())
            error_sum+=float((preds[-1]-target).abs()[:,0][valid].sum())
            sign_sum+=int((preds[-1][:,0][valid].sign()==target[:,0][valid].sign()).sum())
            driver.reset_model(model)
            del left,right,target,valid,preds,spikes,volts,loss,micro
        g=torch.cat([(p.grad.detach().reshape(-1) if p.grad is not None else torch.zeros_like(p).reshape(-1)) for p in model.parameters()])
        assert bool(torch.isfinite(g).all())
        gradients[label]=g
        perturbations[label]=dict(loss=loss_sum,epe=error_sum/count_all,gradient_norm=float(g.norm()),
            correct_sign_fraction=sign_sum/count_all)
    for label in perturbations:
        perturbations[label]['cosine_to_identity']=float(torch.nn.functional.cosine_similarity(gradients['identity'][None],gradients[label][None]))
    write(output/'perturbations.json',perturbations)
    opt.zero_grad(set_to_none=True); del gradients, probe_items
    steps=[int(s['step'].item()) for s in opt.state.values() if 'step' in s]
    assert min(steps)==max(steps)==scheduler.last_epoch==480
    assert all(bool(torch.isfinite(t).all()) for t in model.state_dict().values())
    assert all(bool(torch.isfinite(t).all()) for s in opt.state.values() for t in s.values() if torch.is_tensor(t))
    checkpoint=dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),
        updates=480,samples_seen=480*batch,config=config,source=source,order=order,
        rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all()))
    torch.save(checkpoint,output/'last.tmp.pt');(output/'last.tmp.pt').replace(output/'last.pt')
    write(output/'result.json',dict(status='COMPLETED',config=config,updates=480,samples_seen=480*batch,
        initial_state_sha256=initial_hash,history=history,perturbations=perturbations,
        optimizer_steps=[min(steps),max(steps)],scheduler_last_epoch=scheduler.last_epoch,
        finite_model_optimizer=True,clip_fraction=sum(r['clipped'] for r in rows)/480,
        last120_loss=sum(r['loss'] for r in rows[-120:])/120,
        process_seconds=time.monotonic()-started,finished_utc=datetime.now(timezone.utc).isoformat()))


if __name__ == '__main__': main()
