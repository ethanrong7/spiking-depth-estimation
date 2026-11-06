import torch
from utils.utils import InputPadder


def neuron_regularization(spike_sequence, v_sequence, f0=0.05, lambda_rate=0.0001, lambda_v=0.001):
    spike = torch.stack(spike_sequence, dim=1)   # (B, T, C, H, W)
    firing_rate = torch.mean(spike, dim=(0, 1))
    v = torch.stack(v_sequence, dim=1)
    
    loss_rate = lambda_rate * torch.sum((firing_rate - f0) ** 2)
    loss_v = lambda_v * torch.mean(v ** 2)   # loss_v = v^2
    
    return loss_rate + loss_v


def sequence_loss(s_list, v_list, disp_preds, disp_gt, valid, spike_size, loss_gamma=0.9, max_disp=700):
    """ Loss function defined over sequence of disp predictions """

    n_predictions = len(disp_preds)
    assert n_predictions >= 1
    disp_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(disp_gt**2, dim=1).sqrt()

    # exclude extremly large displacements
    valid = ((valid >= 0.5) & (mag < max_disp)).unsqueeze(1)
    assert valid.shape == disp_gt.shape, [valid.shape, disp_gt.shape]
    assert not torch.isinf(disp_gt[valid.bool()]).any()

    # compute neuron regularization
    loss_neuron = neuron_regularization(s_list, v_list)

    for i in range(n_predictions):
        assert not torch.isnan(disp_preds[i]).any() and not torch.isinf(disp_preds[i]).any()
        adjusted_loss_gamma = loss_gamma**(15/(n_predictions - 1))
        i_weight = adjusted_loss_gamma**(n_predictions - i - 1)

        padder = InputPadder(torch.Size(spike_size), divis_by=4)
        disp_preds[i] = padder.unpad(disp_preds[i])

        i_loss = (disp_preds[i] - disp_gt).abs()
        assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, disp_gt.shape, disp_preds[i].shape]
        disp_loss += i_weight * i_loss[valid.bool()].mean()

    epe = torch.sum((disp_preds[-1] - disp_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return disp_loss + loss_neuron, metrics