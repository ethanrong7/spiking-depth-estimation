"""Bounded final-readout AdamW health capture, without changing optimization.

Timing contract (one accumulated optimizer update):
  capture = monitor.before_clip()                 # raw grads, old parameters
  ... unchanged global clipping ...
  monitor.after_clip_before_step(capture)         # clipped grads, group settings
  optimizer.step()
  metrics = monitor.after_step(capture)           # actual moments/new parameters
  ... scheduler.step(), zero_grad(), next forward ...

Only selected readout tensors are copied. No all-parameter snapshots are made.
Default selectors cover the shared study decoder and the pinned recurrent
disparity head. Legacy output channel1 is vertical flow forced to zero, so its
disparity diagnostics select output channel0 rather than averaging it in.
"""
from dataclasses import dataclass, field
import math
from typing import Optional

import torch


@dataclass(frozen=True)
class ReadoutTarget:
    name: str
    role: str
    output_channel: Optional[int] = None


def default_readout_targets(model):
    targets = []
    for name, parameter in model.named_parameters():
        if name.endswith('decoder.residual.2.weight'):
            targets.append(ReadoutTarget(name, 'final_disparity_kernel'))
        elif name.endswith('decoder.residual.2.bias'):
            targets.append(ReadoutTarget(name, 'final_disparity_bias'))
        elif name.endswith('update_block.disp_head.conv2.weight'):
            targets.append(ReadoutTarget(name, 'final_disparity_kernel', 0))
        elif name.endswith('update_block.disp_head.conv2.bias'):
            targets.append(ReadoutTarget(name, 'final_disparity_bias', 0))
    return targets


def _scalar(value):
    return float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)


def _slice(value, target):
    return value.select(0, target.output_channel) if target.output_channel is not None else value


def _cpu(value, target, *, double=True):
    result = _slice(value.detach(), target).cpu().clone()
    return result.double() if double else result


def _summary(value, magnitude=False):
    flat = value.double().reshape(-1)
    finite = torch.isfinite(flat)
    if not bool(finite.all()):
        return {'all_finite': False, 'nonfinite_count': int((~finite).sum()), 'count': flat.numel()}
    if not flat.numel():
        return {'all_finite': True, 'count': 0, 'rms': None, 'abs_max': None, 'quantiles': None}
    qs = torch.tensor([0., .1, .25, .5, .75, .9, 1.], dtype=torch.float64)
    return {'all_finite': True, 'count': flat.numel(), 'rms': float(flat.square().mean().sqrt()),
            'abs_max': float(flat.abs().max()), 'signed_mean': float(flat.mean()),
            'quantile_basis': 'absolute_values' if magnitude else 'signed_values',
            'quantiles': dict(zip(('q0', 'q10', 'q25', 'q50', 'q75', 'q90', 'q100'), torch.quantile(flat.abs() if magnitude else flat, qs).tolist()))}


def _energy_average(value, raw_gradient):
    energy = raw_gradient.double().square()
    total = energy.sum()
    if not bool(torch.isfinite(total)) or float(total) == 0:
        return None
    answer = (value.double()*energy).sum()/total
    return float(answer) if bool(torch.isfinite(answer)) else None


def _finite_scalar(value):
    number = float(value)
    return number if math.isfinite(number) else None


@dataclass
class _Capture:
    owner: int
    phase: str = 'raw'
    tensors: dict = field(default_factory=dict)


class ReadoutAdamWDiagnostics:
    """Read actual optimizer moments after step; never substitute current g for v.

    Returned values are CPU Python scalars/dictionaries. The private capture
    holds only chosen small readout arrays and is consumed by after_step().
    Group LR/betas/eps/decay are frozen at the pre-step observation. AMSGrad
    reports its actual max-second-moment denominator separately from v_hat.
    """
    def __init__(self, model, optimizer, targets=None):
        if not isinstance(optimizer, torch.optim.AdamW):
            raise TypeError('This diagnostic implements AdamW moment/decay semantics only')
        self.optimizer = optimizer
        self.parameters = dict(model.named_parameters())
        self.targets = list(default_readout_targets(model) if targets is None else targets)
        if not self.targets:
            raise ValueError('No known final disparity targets; supply explicit ReadoutTarget selectors')
        if len({target.name for target in self.targets}) != len(self.targets):
            raise ValueError('Duplicate readout target')
        self.groups = {}
        for group in optimizer.param_groups:
            for parameter in group['params']:
                self.groups[id(parameter)] = group
        for target in self.targets:
            if target.name not in self.parameters or id(self.parameters[target.name]) not in self.groups:
                raise ValueError('Selected target is absent from model/optimizer: '+target.name)
            if self.parameters[target.name].is_complex():
                raise ValueError('Complex optimizer coordinates are outside this diagnostic contract')

    def _check(self, capture, phase):
        if capture.owner != id(self) or capture.phase != phase:
            raise RuntimeError('Optimizer diagnostic capture timing/order mismatch')

    @torch.no_grad()
    def before_clip(self):
        capture = _Capture(id(self))
        for target in self.targets:
            parameter = self.parameters[target.name]
            capture.tensors[target.name] = {
                'parameter_before_native': _cpu(parameter, target, double=False),
                'raw_gradient': _cpu(parameter.grad, target) if parameter.grad is not None else None}
        return capture

    @torch.no_grad()
    def after_clip_before_step(self, capture):
        self._check(capture, 'raw')
        for target in self.targets:
            parameter = self.parameters[target.name]
            values = capture.tensors[target.name]
            group = self.groups[id(parameter)]
            beta1, beta2 = [_scalar(x) for x in group['betas']]
            values.update(lr=_scalar(group['lr']), weight_decay=_scalar(group['weight_decay']),
                          epsilon=_scalar(group['eps']), beta1=beta1, beta2=beta2,
                          amsgrad=bool(group.get('amsgrad', False)), maximize=bool(group.get('maximize', False)),
                          step_before=int(_scalar(self.optimizer.state.get(parameter, {}).get('step', 0))),
                          clipped_gradient=_cpu(parameter.grad, target) if parameter.grad is not None else None)
            if values['raw_gradient'] is None and values['clipped_gradient'] is not None:
                raise RuntimeError('Gradient appeared between raw and postclip observations')
            if not (0 <= beta1 < 1 and 0 <= beta2 < 1 and values['epsilon'] > 0):
                raise ValueError('Unsupported AdamW group coefficients')
        capture.phase = 'clipped'

    @torch.no_grad()
    def after_step(self, capture):
        self._check(capture, 'clipped')
        result = {'schema': 1, 'moment_source': 'actual_post_optimizer_step_state', 'targets': {}}
        for target in self.targets:
            parameter = self.parameters[target.name]
            values = capture.tensors[target.name]
            raw, clipped = values['raw_gradient'], values['clipped_gradient']
            row = {'role': target.role, 'output_channel': target.output_channel,
                   'lr_used': values['lr'], 'weight_decay': values['weight_decay'],
                   'epsilon': values['epsilon'], 'betas': [values['beta1'], values['beta2']],
                   'amsgrad': values['amsgrad'], 'maximize': values['maximize']}
            result['targets'][target.name] = row
            if raw is None or clipped is None:
                row.update(available=False, reason='missing_gradient; no denominator or update-health pass inferred')
                continue
            row.update(raw_gradient=_summary(raw, True), clipped_gradient=_summary(clipped, True), raw_gradient_energy=_finite_scalar(raw.square().sum()))
            state = self.optimizer.state.get(parameter, {})
            if 'step' not in state or int(_scalar(state['step'])) != values['step_before']+1:
                row.update(available=False, reason='AdamW step did not advance exactly once after capture')
                continue
            step = int(_scalar(state['step']))
            m = _cpu(state['exp_avg'], target)
            v = _cpu(state['exp_avg_sq'], target)
            correction1 = 1-values['beta1']**step
            correction2 = 1-values['beta2']**step
            m_hat, v_hat = m/correction1, v/correction2
            denominator_v = _cpu(state['max_exp_avg_sq'], target)/correction2 if values['amsgrad'] else v_hat
            root_v = denominator_v.sqrt()
            actual_attenuation = root_v/(root_v+values['epsilon'])
            first_direct = clipped.abs()/(clipped.abs()+values['epsilon']) if step == 1 else None
            gate_attenuation = first_direct if step == 1 else actual_attenuation
            before_native = values['parameter_before_native']
            before, after = before_native.double(), _cpu(parameter, target)
            ideal_decay_reference = before*(1-values['lr']*values['weight_decay'])
            rounded_decay_reference = (before_native*(1-values['lr']*values['weight_decay'])).double()
            adaptive_delta = after-ideal_decay_reference
            rounded_adaptive_delta = after-rounded_decay_reference
            stored_changed = after != before
            adaptive_changed = after != rounded_decay_reference
            raw_energy = _finite_scalar(raw.square().sum())
            all_finite = all(bool(torch.isfinite(x).all()) for x in (raw, clipped, m_hat, v_hat, denominator_v, root_v, gate_attenuation, after, adaptive_delta))
            row.update(available=all_finite, step=step, step_before=values['step_before'],
                raw_gradient_energy_nonzero=raw_energy>0 if raw_energy is not None else None,
                denominator_attenuation=_summary(gate_attenuation),
                raw_energy_weighted_denominator_attenuation=_energy_average(gate_attenuation, raw),
                actual_moment_denominator_attenuation=_summary(actual_attenuation),
                raw_energy_weighted_actual_moment_attenuation=_energy_average(actual_attenuation, raw),
                first_step_direct_attenuation=_summary(first_direct) if first_direct is not None else None,
                first_step_direct_vs_actual_moment_abs_max=_finite_scalar((first_direct-actual_attenuation).abs().max()) if first_direct is not None else None,
                bias_corrected_first_moment=_summary(m_hat), bias_corrected_second_moment=_summary(v_hat),
                actual_denominator_second_moment=_summary(denominator_v),
                actual_parameter_delta=_summary(after-before),
                expected_ideal_decay_delta=_summary(ideal_decay_reference-before),
                adaptive_delta_excluding_ideal_decay=_summary(adaptive_delta),
                adaptive_delta_excluding_rounded_decay=_summary(rounded_adaptive_delta),
                stored_coordinate_changed_fraction=float(stored_changed.double().mean()),
                adaptive_coordinate_changed_fraction=float(adaptive_changed.double().mean()),
                raw_energy_on_changed_stored_coordinates=_energy_average(stored_changed, raw),
                raw_energy_on_changed_adaptive_coordinates=_energy_average(adaptive_changed, raw),
                bias_correction_first=correction1, bias_correction_second=correction2,
                denominator_ratio_definition='abs(current_clipped_gradient)/(abs(current_clipped_gradient)+eps)' if step==1 else 'sqrt(actual_bias_corrected_denominator_second_moment)/(sqrt(...)+eps)')
            if not all_finite:
                row['reason'] = 'nonfinite captured gradient/moment/parameter; no health pass inferred'
            elif raw_energy == 0:
                row['reason'] = 'zero current raw gradient energy; weighted health metric unavailable'
        capture.phase = 'complete'
        capture.tensors.clear()
        return result
