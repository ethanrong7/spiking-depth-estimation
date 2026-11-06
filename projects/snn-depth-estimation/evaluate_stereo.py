from __future__ import print_function, division
import sys
sys.path.append('core')
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import argparse
import logging
import numpy as np
import torch
from tqdm import tqdm
from core.spikestereonet import SpikeStereoNet, autocast
import core.stereo_datasets as datasets
from core.utils.utils import InputPadder
from spikingjelly.activation_based import functional


@torch.no_grad()
def validate_gs1bsynthspike(model, iters=32, mixed_prec=False):
    """ Peform validation using the synthetic dataset """
    model.eval()
    aug_params = {}
    val_dataset = datasets.Gs1bSynthSpikeDataset(split = 'eval')

    out_list, epe_list = [], []
    
    for val_id in tqdm(range(len(val_dataset))):
        _, spike1, spike2, flow_gt, valid_gt = val_dataset[val_id]
        
        spike1 = spike1[None].cuda()
        spike2 = spike2[None].cuda()

        padder = InputPadder(spike1.shape, divis_by=4)

        with autocast(enabled=mixed_prec):
            _, flow_pr, _ = model(spike1, spike2, iters=iters, test_mode=True)
        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)

        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()

        epe = epe.flatten()
        val = (valid_gt.flatten() >= 0.5) & (flow_gt.abs().flatten() < 192)

        out = (epe > 1.0)
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())
        
        functional.reset_net(model)

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    print("Validation Gs1b-Synth-Spike: %f, %f" % (epe, d1))
    return {'gs1bsynthspike-epe': epe, 'gs1bsynthspike-d1': d1}



@torch.no_grad()
def validate_realspike(model, iters=32, mixed_prec=False):
    """ Peform validation using the real dataset"""
    model.eval()
    val_dataset = datasets.RealSpikeDataset(split = 'eval')

    out_list, epe_list = [], []
    
    for val_id in tqdm(range(len(val_dataset))):
        _, spike1, spike2, flow_gt, valid_gt = val_dataset[val_id]
        
        spike1 = spike1[None].cuda()
        spike2 = spike2[None].cuda()

        padder = InputPadder(spike1.shape, divis_by=4)

        with autocast(enabled=mixed_prec):
            _, flow_pr, _ = model(spike1, spike2, iters=iters, test_mode=True)
        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)

        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()

        epe = epe.flatten()
        val = (valid_gt.flatten() >= 0.5) & (flow_gt.abs().flatten() < 192)

        out = (epe > 1.0)
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())
        
        functional.reset_net(model)

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    print("Validation Real-Spike: %f, %f" % (epe, d1))
    return {'realspike-epe': epe, 'realspike-d1': d1}



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default=None)
    parser.add_argument('--dataset', help="dataset for evaluation", required=True, default=['gs1b-synth-spike'])
    
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

    # Architecure choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
    parser.add_argument('--n_snn_layers', type=int, default=3, help="number of hidden GRU levels")
    args = parser.parse_args()

    model = torch.nn.DataParallel(SpikeStereoNet(args), device_ids=[0])

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        logging.info("Loading checkpoint...")
        checkpoint = torch.load(args.restore_ckpt)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        logging.info(f"Done loading checkpoint")

    model.cuda()
    model.eval()

    # The CUDA implementations of the correlation volume prevent half-precision
    # rounding errors in the correlation lookup. This allows us to use mixed precision
    # in the entire forward pass, not just in the GRUs & feature extractors. 
    use_mixed_precision = args.corr_implementation.endswith("_cuda")

    if args.dataset == 'gs1b-synth-spike':
        validate_gs1bsynthspike(model, iters=args.valid_iters, mixed_prec=use_mixed_precision)
    elif args.dataset == 'real-spike':
        validate_realspike(model, iters=args.valid_iters, mixed_prec=use_mixed_precision)
