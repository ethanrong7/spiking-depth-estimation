"""Tiny CPU checks of exact neurons/blocks, stereo adapters and launch contracts."""
import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import sys
import torch
from torch import nn
from encoders import StereoEncoder,VARIANTS,dynamic_swin,install
from neuron import MultiStepLIFNode
from upstream import load_definitions,ensure_sources,SOURCES
from run_on_npu import cells,worker_command,EVALS

CACHE=None


class EncoderTests(unittest.TestCase):
    def test_neuron_reference_and_temporal_gradient(self):
        from spikingjelly.clock_driven.neuron import MultiStepLIFNode as Reference
        for dtype in (torch.float32,torch.float64):
            for threshold in (.5,1.):
                for detach in (True,False):
                    torch.manual_seed(11)
                    x=(torch.randn(5,2,8,dtype=dtype)+1).requires_grad_()
                    a=Reference(tau=2.,detach_reset=detach,v_threshold=threshold,backend='torch')(x)
                    g=torch.autograd.grad(a.sum(),x)[0]
                    y=x.detach().clone().requires_grad_()
                    node=MultiStepLIFNode(tau=2.,detach_reset=detach,v_threshold=threshold)
                    b=node(y);h=torch.autograd.grad(b.sum(),y)[0]
                    self.assertTrue(torch.equal(a,b))
                    torch.testing.assert_close(g,h,rtol=1e-6,atol=1e-7)
                    self.assertTrue(torch.equal(node(y),b))
        x=torch.full((5,1,1),.8,requires_grad=True)
        MultiStepLIFNode(detach_reset=True)(x)[-1].sum().backward()
        self.assertGreater(float(x.grad[0].abs().sum()),0)

    def test_attention_blocks_match_author_torch_reference(self):
        from spikingjelly.clock_driven.neuron import MultiStepLIFNode as Reference
        class TorchReference(Reference):
            def __init__(self,*args,**kwargs):
                kwargs['backend']='torch';super().__init__(*args,**kwargs)
        specs=[('spikformer','SSA'),('qkformer','Token_QK_Attention'),
               ('qkformer','Spiking_Self_Attention'),('meta-spikeformer','MS_Attention_RepConv_qkv_id')]
        for variant,name in specs:
            with self.subTest(variant=variant,block=name):
                torch.manual_seed(101)
                a=load_definitions(variant,CACHE)[name](dim=16,num_heads=4).double().train()
                b=load_definitions(variant,CACHE,TorchReference)[name](dim=16,num_heads=4).double().train()
                b.load_state_dict(a.state_dict())
                x=torch.randn(5,2,16,4,6,dtype=torch.float64,requires_grad=True);z=x.detach().clone().requires_grad_()
                ya=a(x,None) if variant=='spikformer' else a(x)
                yb=b(z,None) if variant=='spikformer' else b(z)
                if isinstance(ya,tuple):ya=ya[0];yb=yb[0]
                torch.testing.assert_close(ya,yb,atol=1e-6,rtol=1e-5)
                probe=torch.randn_like(ya)
                ga=torch.autograd.grad((ya*probe).sum(),x)[0]
                gb=torch.autograd.grad((yb*probe).sum(),z)[0]
                torch.testing.assert_close(ga,gb,atol=1e-8,rtol=1e-8)
                if variant=='qkformer':
                    swapped=a(x.detach().flip(1))
                    torch.testing.assert_close(swapped,ya.detach().flip(1),atol=1e-8,rtol=1e-8)

    def test_swin_rectangular_port_matches_unpadded_reference(self):
        ns=load_definitions('swin',CACHE)
        orig=ns['SwinTransformerBlock'](dim=16,input_resolution=(14,14),num_heads=4,window_size=7,shift_size=3)
        dynamic_swin(ns)
        adapted=ns['SwinTransformerBlock'](dim=16,input_resolution=(14,14),num_heads=4,window_size=7,shift_size=3)
        adapted.load_state_dict(orig.state_dict())
        x=torch.randn(2,196,16,requires_grad=True);y=x.detach().clone().requires_grad_()
        a,b=orig(x),adapted(y)
        torch.testing.assert_close(a,b,atol=2e-6,rtol=1e-5)
        torch.testing.assert_close(torch.autograd.grad(a.sum(),x)[0],torch.autograd.grad(b.sum(),y)[0],atol=2e-6,rtol=1e-5)
        adapted.input_resolution=(8,13)
        self.assertTrue(torch.isfinite(adapted(torch.randn(2,104,16))).all())

    def test_full_encoders_stereo_shape_gradients_and_reset(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                torch.manual_seed(7)
                model=StereoEncoder(variant,CACHE).train()
                x=torch.randint(0,2,(1,50,32,48)).float()*2-1
                y=torch.randint(0,2,x.shape).float()*2-1
                left,right=model([x,y])
                self.assertEqual(left.shape,(1,256,8,12));self.assertEqual(right.shape,left.shape)
                (left.square().mean()+right.square().mean()).backward()
                grads=[p.grad for p in model.backbone.parameters() if p.grad is not None]
                self.assertTrue(all(torch.isfinite(g).all() for g in grads))
                self.assertGreater(sum(float(g.square().sum()) for g in grads),0)
                model.eval()
                with torch.no_grad():
                    a,b=model([x,y]);bb,aa=model([y,x]);again=model([x,y])
                torch.testing.assert_close(a,aa);torch.testing.assert_close(b,bb)
                self.assertTrue(torch.equal(a,again[0]))
                self.assertEqual(model._taps,[])
                del model,left,right,grads

    def test_bin_order_preserved(self):
        model=StereoEncoder('swin',CACHE).eval()
        captured=[]
        handle=model.backbone.patch_embed.proj.register_forward_pre_hook(lambda m,args:captured.append(args[0].clone()))
        raw=torch.arange(50).view(1,50,1,1).float().expand(1,50,32,32)/49
        with torch.no_grad():model([raw*2-1,raw*2-1])
        observed=captured[0][:,:,0,0].reshape(5,2,10)
        torch.testing.assert_close(observed[:,0].flatten(),torch.arange(50)/49)
        handle.remove()

    def test_native_bn_scope(self):
        model=nn.Module();model.native_bn=nn.BatchNorm2d(2);model.fnet=nn.Identity()
        install(model,'spikformer',1234,CACHE)
        model.train();model.freeze_bn()
        self.assertFalse(model.native_bn.training)
        self.assertTrue(all(m.training for m in model.fnet.modules() if isinstance(m,nn.modules.batchnorm._BatchNorm)))
        model.eval()
        self.assertTrue(all(not m.training for m in model.modules()))

    def test_matrix_and_command(self):
        from types import SimpleNamespace
        matrix=cells(list(range(8)))
        self.assertEqual(len(matrix),8)
        self.assertEqual([(c['variant'],c['seed']) for c in matrix],[(v,s) for v in VARIANTS for s in (1234,2345)])
        self.assertEqual(len({c['device'] for c in matrix}),8)
        args=SimpleNamespace(output_root=Path('/tmp/output'),data_root=Path('/tmp/data'),manifest=Path('/tmp/split.json'),physical_batch=1,preload_gib=4)
        root,cmd=worker_command(args,matrix[0],preflight=True,resume=True)
        self.assertIn('--resume',cmd)
        self.assertEqual(cmd[cmd.index('--stop-at')+1],'200')
        self.assertEqual(cmd[cmd.index('--physical-batch')+1],'1')
        self.assertEqual(EVALS[-1],30000)

    def test_source_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'spikformer.py';p.write_text('raise RuntimeError("must not execute")')
            with self.assertRaisesRegex(RuntimeError,'Pinned source mismatch'):
                load_definitions('spikformer',tmp)

    def test_metric_export_rejects_corrupt_evaluation(self):
        from summarize import summarize
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cell=root/'seed-1234'/'swin';cell.mkdir(parents=True)
            meta=dict(order=list(range(960)),config=dict(seed=1234,variant='swin',physical_batch=2,
                      accumulation=4,shared_state_sha256='shared'),source_registry=dict(source_identity_sha256='source'))
            (cell/'metadata.json').write_text(json.dumps(meta))
            frames=[dict(scene=f'{i:04}',frame=str(j),valid_pixels=1000,epe=2.,bad3=20.,bad3_pixels=200) for i in range(30) for j in range(2)]
            event=dict(update=0,samples_seen=0,per_frame=frames,scene_epe={f'{i:04}':2. for i in range(30)},mean_scene_epe=2.,bad3=20.)
            (cell/'eval.jsonl').write_text(json.dumps(event)+'\n')
            self.assertEqual(summarize(root),'partial')
            self.assertEqual(len((root/'summaries/scenes.csv').read_text().splitlines()),31)
            event['mean_scene_epe']=3.
            (cell/'eval.jsonl').write_text(json.dumps(event)+'\n')
            with self.assertRaisesRegex(RuntimeError,'Audit failed'):summarize(root)
            self.assertEqual(len((root/'summaries/curves.csv').read_text().splitlines()),1)

    def test_metric_export_partial_and_audit(self):
        from summarize import summarize
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(summarize(tmp),'partial')
            p=Path(tmp)/'summaries'
            self.assertTrue((p/'curves.csv').exists())
            self.assertEqual(len(json.loads((p/'completion-audit.json').read_text())['cells']),8)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-cache',type=Path,required=True)
    args,rest=p.parse_known_args()
    CACHE=ensure_sources(args.source_cache)
    torch.set_num_threads(2)
    unittest.main(argv=[sys.argv[0]]+rest)
