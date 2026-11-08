import copy
import unittest

import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

import nmrf_runtime  # Registers the vendored parent path.
from nmrf.models.NMP import SwinNMP, WindowAttention, fourier_coord_embed
from nmrf_binary_support import (LocalSupportSwinNMP, begin_support_observation, end_support_observation,
                                 install_binary_support, normalize_content, support_block_items,
                                 support_features, take_support_activity)


def fixture(shift=3, batch=1, size=12, dtype=torch.float64, device='cpu'):
    torch.manual_seed(78)
    upstream = SwinNMP(128,159,4,window_size=6,shift_size=shift,normalize_before=True)
    upstream = upstream.to(device=device,dtype=dtype).eval()
    upstream.H = upstream.W = size
    with torch.no_grad():
        upstream.attn.relative_position_enc_table.normal_(0,.1)
    x = torch.randn(batch*size**2,4,128,device=device,dtype=dtype)
    d = torch.rand(batch*size**2,4,device=device,dtype=dtype)*2
    d[:,1] = d[:,0]
    d[0,:2] = 0
    geometry = fourier_coord_embed(d.unsqueeze(-1),15,normalizer=3.14/64).to(dtype)
    if size==12:
        padded = geometry.reshape(batch,size,size,4,31)
        padded[:, :, -1] = 0
        padded[:, -2:, :] = 0
    mask = (WindowAttention.gen_shift_window_attn_mask((size,size),(6,6,4),shift,device)
            if shift else WindowAttention.gen_window_attn_mask((6,6,4),device)[None]).to(dtype)
    return upstream,x,geometry,mask


def pack_reference(features):
    shifts = torch.arange(32,device=features.device)
    return ((features.ge(1).long() << shifts).sum(-1)).to(torch.int32)


class LocalSupportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_hard_events_surrogate_and_packed_integer_reference(self):
        z = torch.tensor([-1., .999, 1., 1.001],requires_grad=True)
        hard = support_features(z,'binary',True)
        self.assertTrue(hard.equal(torch.tensor([0.,0.,1.,1.])))
        self.assertTrue(hard.equal(support_features(z,'binary',False)))
        hard.sum().backward()
        p = torch.sigmoid(z.detach()-1)
        torch.testing.assert_close(z.grad,p*(1-p))
        self.assertGreater(float(z.grad[0]),0.)
        bits = torch.stack((torch.zeros(32),torch.ones(32),torch.arange(32)%2))
        packed = [int(x)&0xffffffff for x in pack_reference(bits)]
        self.assertEqual(packed,[0,0xffffffff,0xaaaaaaaa])
        for i,first in enumerate(packed):
            for j,second in enumerate(packed):
                self.assertEqual(bin(first&second).count('1'),int((bits[i]*bits[j]).sum()))

    def test_all_five_blocks_keep_weights_and_pool_activity(self):
        shifts = (0, 3, 0, 3, 0)
        model = nn.Module()
        model.parent = nn.Module()
        model.parent.num_proposals = 4
        model.parent.inference = nn.Module()
        stack = nn.ModuleList()
        inputs = {}
        for shift in shifts:
            block, x, geometry, mask = fixture(shift=shift, dtype=torch.float32)
            layer = nn.Module()
            layer.nmp = block
            stack.append(layer)
            inputs[shift] = (x, geometry, mask)
        model.parent.inference.layers = stack
        original_keys = list(model.state_dict())
        prior = copy.deepcopy(model)
        default = install_binary_support(prior, support_mode='binary')
        self.assertEqual(default['layers'], [1, 2, 3])
        self.assertEqual(default['shifts'], [3, 0, 3])
        with self.assertRaises(ValueError):
            install_binary_support(model, support_mode='binary', layers=(0, 4, 4))

        meta = install_binary_support(model, support_mode='binary', layers=range(5))
        blocks = support_block_items(model)
        self.assertEqual(meta['layers'], [i for i, _ in blocks])
        self.assertEqual(meta['shifts'], list(shifts))
        self.assertEqual(list(model.state_dict()), original_keys)
        for (index, block) in blocks[1:4]:
            x, geometry, mask = inputs[block.shift_size]
            with torch.no_grad():
                self.assertTrue(torch.equal(block(x, geometry, mask), prior.parent.inference.layers[index].nmp(
                    x, geometry, mask)))

        x, geometry, _ = inputs[0]
        x = x.requires_grad_()
        model.train()
        with FlopCounterMode(display=False) as counter:
            for _, block in blocks:
                x = block(x, geometry, inputs[block.shift_size][2])
        raw_counts = [block._activity[1].item() for _, block in blocks]
        expected_events = 2 * int(geometry.ne(0).any(-1).sum()) * 4 * 32
        self.assertEqual(raw_counts, [expected_events]*5)
        activity = take_support_activity(model)
        (x.square().mean() + activity).backward()
        self.assertTrue(torch.isfinite(x).all())
        self.assertTrue(all(torch.isfinite(block.qkv.weight.grad).all() for _, block in blocks))
        tokens = x.shape[0]*4
        linear = 2*tokens*(2*128*128+159*128+128*128+2*128*512)
        edge_contractions = 2*2*tokens*4*20*32
        self.assertEqual(counter.get_total_flops(), 5*(linear+edge_contractions))

        model.eval()
        begin_support_observation(model)
        with torch.no_grad():
            for _, block in blocks:
                block(x, geometry, inputs[block.shift_size][2])
        stats = end_support_observation(model)
        self.assertEqual(list(stats), ['0', '1', '2', '3', '4'])
        self.assertEqual(sum(row['event_denominator'] for row in stats.values()), expected_events*5//2)

    def test_shifted_reference_messages_and_original_legality(self):
        parent,x,geometry,mask = fixture()
        block = LocalSupportSwinNMP(parent,support_mode='binary')
        labels,geom = block.partition(x),block.partition(geometry)
        q,k,value = block._projection(labels,geom)
        q,k = q.ge(1).double(),k.ge(1).double()
        prior,valid,legal = block._legal_prior(geom,mask)
        relative = parent.attn.relative_position_enc_table.reshape(-1,4,3,32)[:,:,2]
        actual = block._reference(q,k,value,prior,relative,valid,legal)

        for w,pixel,query in ((0,0,0),(0,7,2),(1,18,1),(3,35,3)):
            reference = torch.zeros_like(actual[w,pixel,query])
            mass = torch.zeros(4,dtype=x.dtype)
            y,z = divmod(pixel,6)
            for ay,az in ((y,z),(y-1,z),(y+1,z),(y,z-1),(y,z+1)):
                if not (0<=ay<6 and 0<=az<6):
                    continue
                donor = ay*6+az
                for candidate in range(4):
                    if (not bool(valid[w,pixel,query]) or not bool(valid[w,donor,candidate])
                            or not torch.isfinite(mask[w,pixel*4+query,donor*4+candidate])):
                        continue
                    d = (geom[w,pixel,query,-1]-geom[w,donor,candidate,-1]).abs()*64/3.14
                    c = (q[w,pixel,query]*k[w,donor,candidate]).sum(-1)
                    weight = c*(1-d).clamp_min(0)
                    ve = relative[parent.attn.relative_position_index[pixel,donor]]
                    reference += weight[:,None]*(value[w,donor,candidate]+ve)
                    mass += weight
            torch.testing.assert_close(actual[w,pixel,query],reference/mass.clamp_min(1)[:,None])
        self.assertTrue(legal[:,:,:,0:4].equal(
            torch.eye(4,dtype=torch.bool)[None,None] & valid[...,None] & valid[:,:,None,:]))
        self.assertEqual(int(prior[~legal].count_nonzero()),0)
        self.assertTrue(geom[..., -1][~valid].eq(0).all())
        self.assertTrue(((geom[..., -1] == 0)&valid).any())

    def test_continuous_and_binary_gradients_dense_ffn_and_permutation(self):
        parent,x,geometry,mask = fixture(batch=2)
        for mode in ('binary','continuous'):
            block = LocalSupportSwinNMP(copy.deepcopy(parent),support_mode=mode)
            inputs = x.detach().requires_grad_()
            block.train()
            shapes = []
            hook = block.mlp.register_forward_pre_hook(lambda _, args: shapes.append(args[0].shape))
            actual = block(inputs,geometry,mask)
            hook.remove()
            numerator,denominator = block.take_activity()
            objective = actual.square().mean()+.01*numerator/denominator
            objective.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                for p in (inputs,*block.parameters())))
            self.assertEqual(shapes,[x.shape])
            self.assertEqual(set(block.state_dict()),set(parent.state_dict()))
            block.eval()
            with torch.no_grad():
                baseline = block(x,geometry,mask)
                torch.testing.assert_close(actual.detach(),baseline,atol=2e-6,rtol=1e-5)
                perm = torch.rand(x.shape[:2]).argsort(-1)
                alt = block(x.gather(1,perm[...,None].expand_as(x)),
                            geometry.gather(1,perm[...,None].expand_as(geometry)),mask)
                torch.testing.assert_close(alt,baseline.gather(1,perm[...,None].expand_as(baseline)),
                                           atol=2e-6,rtol=1e-5)

    def test_no_support_keeps_projection_bias_and_own_ffn(self):
        parent,x,g,mask = fixture(shift=0,size=6)
        block = LocalSupportSwinNMP(parent,support_mode='binary')
        with torch.no_grad():
            expected_state = x + parent.proj.bias
            expected = expected_state + parent.mlp(parent.norm2(expected_state))
            actual = block(x,torch.zeros_like(g),mask)
            torch.testing.assert_close(actual,expected)
        x = torch.tensor([[[[[2.]]]]],requires_grad=True)
        normalize_content(x).sum().backward()
        self.assertEqual(float(x.grad),.5)

    def test_weak_sole_support_preserves_small_amplitude(self):
        parent,_,_,_ = fixture(shift=0,size=6)
        block = LocalSupportSwinNMP(parent,support_mode='binary')
        shape = (1,36,4,4,32)
        q,k,value = [torch.zeros(shape,dtype=torch.double) for _ in range(3)]
        q[0,0,0,0,0] = k[0,1,0,0,0] = 1.
        value[0,1,0,0] = 3.
        prior = torch.zeros((1,36,4,20),dtype=torch.double)
        prior[0,0,0,16] = .05
        relative = torch.zeros((121,4,32),dtype=torch.double)
        valid = torch.ones((1,36,4),dtype=torch.bool)
        message = block._reference(q,k,value,prior,relative,valid,prior>0)
        torch.testing.assert_close(message[0,0,0,0],torch.full((32,),.15,dtype=torch.double))
        self.assertEqual(int(message[0,0,0,1:].count_nonzero()),0)

    def test_registered_local_count_matches_shapes(self):
        parent,x,geometry,mask = fixture(shift=0,size=6,dtype=torch.float32)
        block = LocalSupportSwinNMP(parent,support_mode='binary')
        with torch.no_grad(),FlopCounterMode(display=False) as count:
            block(x,geometry,mask)
        tokens = 144
        inherited = 2*tokens*(128*256+159*128+128**2+2*128*512)
        local = 2*2*tokens*4*20*32
        self.assertEqual(count.get_total_flops(),inherited+local)


if __name__ == '__main__':
    unittest.main()
