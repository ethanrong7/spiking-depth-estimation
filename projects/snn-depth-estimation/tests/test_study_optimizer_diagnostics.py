"""Tiny CPU AdamW arithmetic/state tests; no study training experiments."""
import json
import unittest

import torch
from torch import nn

from core.study_optimizer_diagnostics import ReadoutAdamWDiagnostics, default_readout_targets


class Head(nn.Module):
    def __init__(self, baseline=False):
        super().__init__()
        if baseline:
            self.model=nn.Module();self.model.update_block=nn.Module();self.model.update_block.disp_head=nn.Module()
            self.model.update_block.disp_head.conv2=nn.Linear(2,2)
            self.head=self.model.update_block.disp_head.conv2
        else:
            self.decoder=nn.Module();self.decoder.residual=nn.Sequential(nn.Identity(),nn.Identity(),nn.Linear(2,1))
            self.head=self.decoder.residual[2]
        self.unrelated=nn.Parameter(torch.ones(1000))
        self.double()
        with torch.no_grad():
            self.head.weight.zero_();self.head.bias.zero_()


class OptimizerDiagnosticsTests(unittest.TestCase):
    def first(self, coefficient):
        model=Head()
        optimizer=torch.optim.AdamW(model.parameters(),lr=.01,betas=(.9,.999),eps=1e-8,weight_decay=.1,foreach=False)
        monitor=ReadoutAdamWDiagnostics(model,optimizer)
        model.head.weight.grad=torch.tensor([[.01,.02]],dtype=torch.float64)
        model.head.bias.grad=torch.tensor([.005],dtype=torch.float64)
        capture=monitor.before_clip()
        self.assertEqual(len(capture.tensors),2)
        for parameter in (model.head.weight,model.head.bias):parameter.grad.mul_(coefficient)
        monitor.after_clip_before_step(capture)
        optimizer.step()
        row=monitor.after_step(capture)['targets']['decoder.residual.2.weight']
        self.assertEqual(capture.tensors,{})
        return model,row

    def test_first_step_uses_clipped_gradient_and_raw_energy_weights(self):
        low,row=self.first(1e-8)
        g=torch.tensor([1e-10,2e-10],dtype=torch.float64)
        attenuation=g/(g+1e-8)
        expected=float((attenuation*torch.tensor([1.,4.])).sum()/5)
        self.assertAlmostEqual(row['raw_energy_weighted_denominator_attenuation'],expected,places=14)
        self.assertAlmostEqual(row['raw_gradient_energy'],.0005,places=14)
        self.assertEqual(row['step'],1)
        self.assertTrue(torch.allclose(low.head.weight.detach()[0],-.01*g/(g+1e-8),atol=1e-15,rtol=1e-12))
        self.assertEqual(row['raw_energy_on_changed_stored_coordinates'],1.)
        self.assertLess(row['raw_energy_weighted_denominator_attenuation'],.1)
        high,healthy=self.first(1e-5)
        self.assertGreater(healthy['raw_energy_weighted_denominator_attenuation'],.9)
        self.assertGreater(healthy['adaptive_delta_excluding_ideal_decay']['rms'],row['adaptive_delta_excluding_ideal_decay']['rms'])
        json.dumps(row,allow_nan=False)

    def test_second_step_uses_actual_history_and_correct_bias_corrections(self):
        model=Head();optimizer=torch.optim.AdamW(model.parameters(),lr=.01,betas=(.5,.75),eps=1e-8,weight_decay=0,foreach=False)
        monitor=ReadoutAdamWDiagnostics(model,optimizer)
        gradients=[torch.tensor([[1e-6,2e-6]],dtype=torch.float64),torch.tensor([[0.,1e-10]],dtype=torch.float64)]
        for gradient in gradients:
            optimizer.zero_grad(set_to_none=True);model.head.weight.grad=gradient.clone()
            capture=monitor.before_clip();monitor.after_clip_before_step(capture);optimizer.step()
            row=monitor.after_step(capture)['targets']['decoder.residual.2.weight']
        v1=.25*gradients[0].square();v2=.75*v1+.25*gradients[1].square()
        vhat=v2/(1-.75**2)
        expected=float((vhat.sqrt()/(vhat.sqrt()+1e-8))[0,1])
        self.assertAlmostEqual(row['raw_energy_weighted_denominator_attenuation'],expected,places=14)
        self.assertGreater(expected,.99)
        mhat=(.5*(.5*gradients[0])+.5*gradients[1])/(1-.5**2)
        self.assertAlmostEqual(row['bias_corrected_first_moment']['signed_mean'],float(mhat.mean()),places=14)
        self.assertIsNone(row['first_step_direct_attenuation'])

    def test_zero_energy_missing_gradient_and_decay_only_are_not_health_passes(self):
        model=Head();optimizer=torch.optim.AdamW(model.parameters(),lr=.1,weight_decay=.2,foreach=False)
        with torch.no_grad():model.head.weight.fill_(2.)
        model.head.weight.grad=torch.zeros_like(model.head.weight)
        monitor=ReadoutAdamWDiagnostics(model,optimizer)
        capture=monitor.before_clip();monitor.after_clip_before_step(capture);optimizer.step()
        result=monitor.after_step(capture)['targets'];row=result['decoder.residual.2.weight']
        self.assertIsNone(row['raw_energy_weighted_denominator_attenuation'])
        self.assertIn('zero current raw',row['reason'])
        self.assertEqual(row['adaptive_delta_excluding_ideal_decay']['rms'],0.)
        self.assertEqual(row['adaptive_coordinate_changed_fraction'],0.)
        self.assertEqual(row['stored_coordinate_changed_fraction'],1.)
        self.assertFalse(result['decoder.residual.2.bias']['available'])

    def test_baseline_selector_excludes_forced_zero_vertical_flow_channel(self):
        model=Head(baseline=True)
        self.assertTrue(all(target.output_channel==0 for target in default_readout_targets(model)))
        optimizer=torch.optim.AdamW(model.parameters(),lr=.01,foreach=False)
        model.head.weight.grad=torch.tensor([[1e-4,2e-4],[100.,100.]],dtype=torch.float64)
        monitor=ReadoutAdamWDiagnostics(model,optimizer)
        capture=monitor.before_clip();monitor.after_clip_before_step(capture);optimizer.step()
        row=monitor.after_step(capture)['targets']['model.update_block.disp_head.conv2.weight']
        self.assertAlmostEqual(row['raw_gradient_energy'],5e-8,places=15)
        self.assertEqual(row['raw_gradient']['count'],2)

    def test_phase_order_and_nonfinite_metrics_are_explicit_and_json_safe(self):
        model=Head();optimizer=torch.optim.AdamW(model.parameters(),lr=.01,foreach=False)
        monitor=ReadoutAdamWDiagnostics(model,optimizer)
        model.head.weight.grad=torch.full_like(model.head.weight,float('inf'))
        capture=monitor.before_clip()
        with self.assertRaises(RuntimeError):monitor.after_step(capture)
        monitor.after_clip_before_step(capture)
        optimizer.step()
        result=monitor.after_step(capture)
        self.assertFalse(result['targets']['decoder.residual.2.weight']['available'])
        json.dumps(result,allow_nan=False)
        with self.assertRaises(RuntimeError):monitor.after_step(capture)

    def test_amsgrad_uses_actual_max_second_moment(self):
        model=Head();optimizer=torch.optim.AdamW(model.parameters(),lr=.01,betas=(.5,.5),eps=.1,amsgrad=True,foreach=False)
        monitor=ReadoutAdamWDiagnostics(model,optimizer)
        for gradient in (2.,.1):
            model.head.weight.grad=torch.full_like(model.head.weight,gradient)
            capture=monitor.before_clip();monitor.after_clip_before_step(capture);optimizer.step()
            row=monitor.after_step(capture)['targets']['decoder.residual.2.weight']
        state=optimizer.state[model.head.weight]
        vhat=state['max_exp_avg_sq']/(1-.5**2)
        expected=float((vhat.sqrt()/(vhat.sqrt()+.1)).mean())
        self.assertAlmostEqual(row['raw_energy_weighted_denominator_attenuation'],expected,places=14)
        self.assertGreater(row['actual_denominator_second_moment']['rms'],row['bias_corrected_second_moment']['rms'])


if __name__=='__main__':
    unittest.main()
