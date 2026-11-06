"""Tiny CPU contract tests; no training data or GPU execution."""
import copy
import ast
from contextlib import nullcontext
import random
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np
import json
from pathlib import Path
import tempfile
import unittest
import torch
from protocol import SEEDS, Prefetch, audit_history, expected_lr, presentation, validate_row
from common import decode


class ProtocolTests(unittest.TestCase):
    def test_actual_worker_validation_collects_frames_and_masked_metrics(self):
        # Execute the production nested evaluate() body on two tiny CPU frames.
        # Only device transfers and lock/RNG contexts are replaced; collection,
        # masking, scene/pooled aggregation and append/checkpoint calls are real.
        source=ast.parse(Path(__file__).with_name('worker.py').read_text())
        main=next(node for node in source.body if isinstance(node,ast.FunctionDef) and node.name=='main')
        evaluate=copy.deepcopy(next(node for node in main.body if isinstance(node,ast.FunctionDef) and node.name=='evaluate'))
        evaluate.decorator_list=[]
        module=ast.fix_missing_locations(ast.Module(body=[evaluate],type_ignores=[]))
        pred=torch.tensor([[[[1.,4.]]],[[[8.,2.]]]])
        valid=torch.tensor([[[True,True]],[[True,False]]])
        batch=(torch.zeros_like(pred),torch.zeros_like(pred),torch.zeros_like(pred),valid,torch.tensor([0,1]))
        def metrics(error):
            return dict(epe=float(error.mean()),**{f'bad{k}':float((error>k).float().mean()*100) for k in (1,3,5)})
        captured=[];checkpoint=Mock()
        namespace=dict(torch=torch,np=np,random=random,status=Mock(),phase_lock=lambda *args:nullcontext(),phase_root=Path('.'),lock_name='test',model=SimpleNamespace(eval=Mock()),loader=[batch],driver=SimpleNamespace(forward=lambda *args:pred,reset_model=Mock(),metric_values=metrics),dev=SimpleNamespace(rows=[dict(scene='a',frame='1'),dict(scene='b',frame='1')]),completed=0,evaluated=set(),outdir=Path('.'),append=lambda path,row:captured.append(row),checkpoint=checkpoint,a=SimpleNamespace(arm='SP'))
        exec(compile(module,'worker-validation-cpu','exec'),namespace)
        with patch.object(torch.Tensor,'cuda',lambda self:self),patch.object(torch.random,'fork_rng',lambda **kwargs:nullcontext()):
            namespace['evaluate']()
        self.assertEqual(len(captured),1)
        result=captured[0]
        self.assertEqual([row['valid_pixels'] for row in result['per_frame']],[2,1])
        self.assertEqual(result['scene_epe'],dict(a=2.5,b=8.))
        self.assertEqual(result['mean_scene_epe'],5.25)
        self.assertAlmostEqual(result['pooled_epe'],13/3)
        self.assertAlmostEqual(result['bad3'],200/3)
        checkpoint.assert_called_once()
        self.assertEqual(namespace['evaluated'],{0})

    def test_prefetch_exact_legacy_and_resume(self):
        class Dataset:
            def __getitem__(self,index):
                x=torch.arange(6).reshape(1,2,3)+index
                return x,x+1,x.float(),x.bool(),index
        data=Dataset();order=list(range(960))
        old=decode(data,order,1)
        for seed in SEEDS:
            stream=Prefetch(data,order,1,3,seed)
            try:
                actual=stream.get(1)
                self.assertEqual((actual['indices'],actual['flags']),presentation(order,1,seed))
                if seed==1234:
                    for a,b in zip(actual['items'],old['items']):
                        for x,y in zip(a[:4],b[:4]):self.assertTrue(torch.equal(x,y))
                resumed=Prefetch(data,order,2,3,seed)
                try:
                    straight=stream.get(2);restart=resumed.get(2)
                    self.assertEqual(straight['flags'],restart['flags'])
                    for a,b in zip(straight['items'],restart['items']):
                        for x,y in zip(a[:4],b[:4]):self.assertTrue(torch.equal(x,y))
                finally:resumed.close()
            finally:stream.close()

    def test_full_onecycle_and_resume(self):
        parameter=torch.nn.Parameter(torch.tensor(0.))
        opt=torch.optim.AdamW([parameter],lr=2e-4)
        schedule=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=2e-4,total_steps=30000,pct_start=.01,cycle_momentum=False,anneal_strategy='linear')
        for step in range(30001):
            self.assertAlmostEqual(opt.param_groups[0]['lr'],expected_lr(step),places=15)
            if step==7500:
                state=copy.deepcopy((opt.state_dict(),schedule.state_dict()))
                opt.load_state_dict(state[0]);schedule.load_state_dict(state[1])
                self.assertEqual(schedule.last_epoch,7500)
            if step<30000:opt.step();schedule.step()

    def test_actual_history_requires_indices_flags_lr_and_contiguity(self):
        order=list(range(960));indices,hv=presentation(order,0,2345)
        row=dict(update=1,samples_seen=8,indices=indices,augmentation_hv=hv,lr_used=expected_lr(0),lr_next=expected_lr(1))
        validate_row(row,order,2345)
        for field,value in [('indices',[7]*8),('augmentation_hv',[[False,False]]*8),('lr_used',.1)]:
            bad=dict(row,**{field:value})
            with self.assertRaises(AssertionError):validate_row(bad,order,2345)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'train.jsonl';path.write_text(json.dumps(row)+'\n')
            self.assertTrue(audit_history(path,order,2345,1)['actual_indices_hv_lr_verified'])
            with self.assertRaises(AssertionError):audit_history(path,order,2345,2)


if __name__=='__main__':unittest.main()
