"""One fresh Slurm process per arm/physical batch; no scientific optimizer updates; first fixed Train8 and Dev2."""
import argparse
import json
import os
from pathlib import Path
import time
import traceback

from recipe import ARMS, augment, batch_indices, build_model, digest, load_frozen, paired_flags
from run_recipe import initialization_proof, tensor_hash, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-source', type=Path, required=True)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--physical-batch', type=int, choices=(4, 8), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    driver, mechanism, source = load_frozen(args.base_source)
    import torch
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert any(name in torch.cuda.get_device_name(0) for name in ('4090', '5090'))
    torch.set_num_threads(8); torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {'arm': args.arm, 'physical_batch': args.physical_batch, 'effective_batch': 8,
              'job_id': os.environ['SLURM_JOB_ID'], 'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
              'base_source_identity_sha256': source['source_identity_sha256'], 'optimizer_updates': 0,
              'scientific_training_presentations': 0, 'engineering_train_presentations': 16, 'engineering_dev_presentations': 2, 'manifest_sha256': digest(args.manifest), 'input_shape': [50, 250, 400], 'train_refinements': 16, 'eval_refinements': 32,
              'moment_reservation': 'two FP32 tensors per trainable parameter before backward; no optimizer.step',
              'status': 'STARTED'}
    write(args.output, result)
    try:
        model = build_model(driver, mechanism, args.arm, 1234)
        result['initialization'] = initialization_proof(driver, mechanism, model, args.arm, 1234)
        initial_hash = tensor_hash(model.state_dict()); model = model.cuda()
        parameters = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(parameters, lr=.0002, weight_decay=1e-5, eps=1e-8)
        for parameter in parameters:
            opt.state[parameter]['step'] = torch.tensor(0.)
            opt.state[parameter]['exp_avg'] = torch.zeros_like(parameter)
            opt.state[parameter]['exp_avg_sq'] = torch.zeros_like(parameter)
        # The foreach AdamW update may create an additional parameter-sized tensorlist.
        # Reserve this too while measuring backward, without changing model weights.
        scratch = [torch.empty_like(parameter) for parameter in parameters]
        result['moment_bytes'] = sum(p.numel() * p.element_size() * 2 for p in parameters)
        result['adamw_transient_reserved_bytes'] = result['moment_bytes'] // 2
        manifest = json.loads(args.manifest.read_text())
        assert len(manifest['train']) == 960 and len(manifest['dev']) == 60
        train = driver.ManifestDataset(args.data_root, manifest['train'])
        dev = driver.ManifestDataset(args.data_root, manifest['dev'])
        order = torch.randperm(960, generator=torch.Generator().manual_seed(1234)).tolist()
        indices = batch_indices(order, 0)
        flags = [paired_flags(1234, position) for position in range(8)]
        result['indices'] = indices; result['augmentation_hv'] = flags
        measured = []
        for repetition in range(2):
            model.train(); model.freeze_bn(); opt.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); begun = time.monotonic(); loss_value = 0.
            items = [augment(train[index], *flag) for index, flag in zip(indices, flags)]
            full_count = sum(int(item[3].sum()) for item in items)
            decode_seconds = time.monotonic() - begun
            for start in range(0, 8, args.physical_batch):
                micro = items[start:start + args.physical_batch]
                left, right, target, valid = [torch.stack([item[j] for item in micro]).cuda() for j in range(4)]
                preds, spikes, volts = driver.forward(model, left, right, 16, True)
                assert len(preds) == 16 and all(pred.shape == target.shape for pred in preds)
                loss = driver.disparity_loss(preds, target, valid) * (int(valid.sum()) / full_count)
                assert torch.isfinite(loss); loss.backward(); loss_value += float(loss.detach())
                driver.reset_model(model)
                del left, right, target, valid, preds, spikes, volts, loss
            grad = float(torch.nn.utils.clip_grad_norm_(parameters, 1.))
            assert torch.isfinite(torch.tensor(grad)) and grad > 0
            if args.arm == 'original_I':
                assert all(cell.current_proj.weight.grad is not None and cell.current_proj.weight.grad.abs().sum() > 0
                           for cell in (model.update_block.snn08, model.update_block.snn16, model.update_block.snn32))
            torch.cuda.synchronize()
            measured.append({'seconds': time.monotonic() - begun, 'decode_seconds': decode_seconds, 'loss': loss_value, 'preclip_gradient_norm': grad,
                             'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
                             'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20})
        opt.zero_grad(set_to_none=True); driver.reset_model(model); model.eval()
        torch.cuda.synchronize(); begun = time.monotonic()
        dev_items = [dev[index] for index in range(2)]
        dev_left, dev_right = [torch.stack([item[j] for item in dev_items]).cuda() for j in range(2)]
        with torch.no_grad(): prediction = driver.forward(model, dev_left, dev_right, 32)
        torch.cuda.synchronize(); eval_seconds = time.monotonic() - begun
        assert prediction.shape == (2, 1, 250, 400) and torch.isfinite(prediction).all()
        driver.reset_model(model)
        assert tensor_hash(model.state_dict()) == initial_hash
        assert all(float(state['step']) == 0 for state in opt.state.values())
        result.update(status='PASS', checks=measured, eval_batch2_seconds=eval_seconds, state_exactly_unchanged=True,
                      optimizer_step_min_max=[0, 0], parameter_count=sum(p.numel() for p in model.parameters()),
                      forecast_seconds_10k=10000 * measured[-1]['seconds'] + 6 * 30 * eval_seconds,
                      recommended_wall_seconds=1.5 * (10000 * measured[-1]['seconds'] + 6 * 30 * eval_seconds) + 600.,
                      forecast_definition='measured decode+transfer+forward+weighted-backward+clip+reset with resident moments+transient reserve, plus6Dev30batches; recommended wall adds50%+600s for optimizer/checkpoints/init/variation')
    except torch.OutOfMemoryError as error:
        result.update(status='OOM', error=str(error), traceback=traceback.format_exc())
    except Exception as error:
        result.update(status='FAILED', error=str(error), traceback=traceback.format_exc())
    write(args.output, result); print(json.dumps(result, allow_nan=False), flush=True)
    if result['status'] != 'PASS': raise SystemExit(2)


if __name__ == '__main__': main()
