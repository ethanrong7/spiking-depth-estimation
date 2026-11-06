"""Tiny CPU resume/schedule/pooled-objective checks; no dataset or experiment."""
import copy
import importlib.util
from pathlib import Path
import random
import unittest
import warnings
import numpy as np
import torch
from continuation import SCHEDULE,build_scheduler,current_rng,restore_rng,rng_fingerprint,optimizer_steps


class Checks(unittest.TestCase):
    def test_optimizer_resume_and_segment_clock(self):
        model=torch.nn.Linear(2,1);opt=torch.optim.AdamW(model.parameters(),lr=.0002,weight_decay=1e-5,eps=1e-8)
        for p in model.parameters():opt.state[p]=dict(step=torch.tensor(10000.),exp_avg=torch.ones_like(p)*.1,exp_avg_sq=torch.ones_like(p)*.2)
        weights=copy.deepcopy(model.state_dict());state=copy.deepcopy(opt.state_dict());rng=current_rng()
        restored=torch.nn.Linear(2,1);restored.load_state_dict(weights);resumed=torch.optim.AdamW(restored.parameters(),lr=.0002);resumed.load_state_dict(state)
        self.assertEqual(optimizer_steps(resumed),(10000,10000))
        for x,y in zip(opt.state_dict()['state'].values(),resumed.state_dict()['state'].values()):
            for key in x:self.assertTrue(torch.equal(x[key],y[key]))
        schedule=build_scheduler(resumed,dict(scheduler=SCHEDULE));self.assertEqual(schedule.last_epoch,0)
        self.assertEqual(optimizer_steps(resumed),(10000,10000));self.assertEqual(resumed.param_groups[0]['lr'],8e-6)
        restore_rng(rng);self.assertEqual(rng_fingerprint(current_rng()),rng_fingerprint(rng))
        # One scalar-gradient unit check preserves the global AdamW clock.
        restored(torch.ones(1,2)).sum().backward();resumed.step();schedule.step()
        self.assertEqual(optimizer_steps(resumed),(10001,10001));self.assertEqual(schedule.last_epoch,1)

    def test_segment_schedule_milestones(self):
        p=torch.nn.Parameter(torch.ones(()));opt=torch.optim.AdamW([p],lr=.0002);scheduler=build_scheduler(opt,dict(scheduler=SCHEDULE))
        for local in (0,1,399,400,401,39999,40000):
            expected=8e-6+(2e-4-8e-6)*local/400 if local<=400 else 2e-4+(8e-10-2e-4)*(local-400)/39699
            scheduler.last_epoch=local
            with warnings.catch_warnings():warnings.simplefilter('ignore');actual=scheduler.get_lr()[0]
            self.assertAlmostEqual(actual,expected,places=15)
        self.assertEqual(10001+400,10401)

    def test_global_order_and_flags(self):
        # Resolve repository by instruction marker; no mutable scientific imports.
        repo=next(p for p in Path(__file__).resolve().parents if (p/'projects/AGENTS.md').exists())
        parent=repo/'storage/experiment-runs/snn-depth-estimation/spiking-transformer-stereo/source-recipe-aligned-three-way-v1/scripts/recipe.py'
        spec=importlib.util.spec_from_file_location('parent_recipe_unit',parent);recipe=importlib.util.module_from_spec(spec);spec.loader.exec_module(recipe)
        order=torch.randperm(960,generator=torch.Generator().manual_seed(1234)).tolist()
        self.assertEqual(recipe.batch_indices(order,10000),order[320:328]);self.assertEqual(recipe.batch_indices(order,49999),order[632:640])
        self.assertEqual(recipe.paired_flags(1234,80000),recipe.paired_flags(1234,10000*8))
        self.assertEqual(divmod(50000,120),(416,80));self.assertEqual(divmod(10000,120),(83,40))

    def test_pooled_unequal_valid_gradient(self):
        x=torch.arange(12,dtype=torch.float64).reshape(4,3);mask=torch.ones_like(x,dtype=torch.bool);mask[0]=False
        full=torch.tensor(.4,dtype=torch.float64,requires_grad=True);loss=((full*x+1).abs()[mask]).mean();loss.backward()
        part=torch.tensor(.4,dtype=torch.float64,requires_grad=True);total=0.
        for sl in (slice(0,2),slice(2,4)):
            value=(part*x[sl]+1).abs()[mask[sl]].mean()*(int(mask[sl].sum())/int(mask.sum()));value.backward();total+=float(value.detach())
        self.assertAlmostEqual(total,float(loss.detach()),places=12);self.assertTrue(torch.allclose(part.grad,full.grad,atol=1e-12,rtol=1e-12))


if __name__=='__main__':unittest.main()
