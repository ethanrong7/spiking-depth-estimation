"""Registered matched 10k recipe factorial; scientific execution is Slurm CUDA only."""
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
    assert config['updates'] == 10000 and config['seed'] == 1234
    assert config['batch'] in (2, 8) and config['augmentation'] in ('none', 'hv')
    assert config['schedule_total'] == 10100
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
    initial_hash = state_hash()
    assert initial_hash == registration['initial_hashes'][config['model']]
    model = model.cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-4, total_steps=config['schedule_total'],
        pct_start=.01, cycle_momentum=False, anneal_strategy='linear', div_factor=25., final_div_factor=1e4)
    train = driver.ManifestDataset(args.data_root, manifest['train'])
    dev = driver.ManifestDataset(args.data_root, manifest['dev'])
    loader = DataLoader(dev, batch_size=2, num_workers=0, generator=torch.Generator().manual_seed(seed + 991))
    panel_rows = []
    for scene in sorted({r['scene'] for r in manifest['train']}):
        panel_rows.append(min((r for r in manifest['train'] if r['scene'] == scene), key=lambda r: r['frame']))
    assert panel_rows == json.loads((args.overlay / 'configs/train-eval-panel.json').read_text())
    train_panel = driver.ManifestDataset(args.data_root, panel_rows)
    train_loader = DataLoader(train_panel, batch_size=2, num_workers=0, generator=torch.Generator().manual_seed(seed + 992))
    write(output / 'train-panel.json', panel_rows)
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
    def save_checkpoint(update):
        checkpoint = dict(model=model.state_dict(), optimizer=opt.state_dict(), scheduler=scheduler.state_dict(),
            updates=update, samples_seen=update*batch, config=config, source=source, order=order,
            manifest_sha256=registration['manifest_sha256'], initial_state_sha256=initial_hash,
            rng=dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all()))
        torch.save(checkpoint, output/'last.tmp.pt'); (output/'last.tmp.pt').replace(output/'last.pt')
        if update == 5000:
            import shutil
            shutil.copy2(output/'last.pt', output/'step5000.pt')

    def evaluate(update):
        begun = time.monotonic()
        # Evaluation must not advance the training RNG stream.
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        try:
            with torch.random.fork_rng(devices=[0]):
                row = dict(update=update, samples_seen=update*batch, **driver.evaluate(model, loader, 32))
                panel = dict(update=update, samples_seen=update*batch, **driver.evaluate(model, train_loader, 32))
        finally:
            random.setstate(python_rng); np.random.set_state(numpy_rng)
            driver.reset_model(model)
        row['seconds_including_train_panel'] = time.monotonic() - begun
        append(output/'evaluations.jsonl', row); history.append(row)
        append(output/'train-evaluations.jsonl', panel)
        print(args.case, 'DEV', update, row['mean_scene_epe'], 'TRAIN_PANEL', panel['mean_scene_epe'], flush=True)
        if update:
            save_checkpoint(update)
    evaluate(0)
    def prepare(step):
        indices = recipe.batch_indices(order, step, batch)
        flags = [recipe.paired_flags(seed, step*batch+i) if config['augmentation']=='hv' else (False, False) for i in range(batch)]
        items = [recipe.augment(train[index], *flag) for index, flag in zip(indices, flags)]
        return indices, flags, items
    rows = []
    physical = config['physical_batch']
    assert physical == min(batch, 4)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='cpu-input') as pool:
        pending = {i:pool.submit(prepare, i) for i in range(2)}
        for step in range(10000):
            begun = time.monotonic(); wait = time.monotonic()
            indices, flags, items = pending.pop(step).result(); wait = time.monotonic() - wait
            if step+2 < 10000: pending[step+2] = pool.submit(prepare, step+2)
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
            if step+1 in registration['eval_updates']: evaluate(step+1)
    opt.zero_grad(set_to_none=True)
    steps=[int(s['step'].item()) for s in opt.state.values() if 'step' in s]
    assert min(steps)==max(steps)==scheduler.last_epoch==10000
    assert all(bool(torch.isfinite(t).all()) for t in model.state_dict().values())
    assert all(bool(torch.isfinite(t).all()) for s in opt.state.values() for t in s.values() if torch.is_tensor(t))
    save_checkpoint(10000)
    write(output/'result.json',dict(status='COMPLETED',config=config,updates=10000,samples_seen=10000*batch,
        initial_state_sha256=initial_hash,history=history,
        dev_evaluation_presentations=len(history)*60,
        train_panel_evaluation_presentations=len(history)*len(panel_rows),
        warmup_dev_presentations=2,
        optimizer_steps=[min(steps),max(steps)],scheduler_last_epoch=scheduler.last_epoch,
        finite_model_optimizer=True,clip_fraction=sum(r['clipped'] for r in rows)/10000,
        last500_loss=sum(r['loss'] for r in rows[-500:])/500,
        process_seconds=time.monotonic()-started,finished_utc=datetime.now(timezone.utc).isoformat()))


if __name__ == '__main__': main()
