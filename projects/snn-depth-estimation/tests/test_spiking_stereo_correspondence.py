"""Tiny CPU operator and gradient checks, not experimental results."""
import unittest
import torch
from core.spiking_stereo_correspondence import (SpikingStereoCorrespondence,
    correspondence_loss, lif, native_lattice, row_scores, row_values, window_values)


class CorrespondenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def model(self, checkpoint=False):
        return SpikingStereoCorrespondence(time_steps=3, widths=(4, 8, 8), heads=2,
            max_width=16, activation_checkpoint=checkpoint)

    def test_signed_row_geometry_and_no_cross_row(self):
        right = torch.eye(4).T.reshape(1, 4, 1, 4)
        left = torch.roll(right, 1, -1)
        scores = row_scores(left, right, 1).mean(2)
        self.assertEqual(scores[0, 0, 2].argmax().item(), 1)  # signed flow -stride
        flipped = row_scores(left.flip(-1), right.flip(-1), 1).mean(2)
        self.assertEqual(flipped[0, 0, 1].argmax().item(), 2)  # mirrored flow +stride

    def test_padding_key_mask_and_empty_counts(self):
        scores = torch.ones(1, 1, 1, 4, 4)
        value = torch.tensor([1., 2., 3., 999.]).reshape(1, 1, 1, 4)
        valid = torch.tensor([True, True, True, False]).reshape(1, 1, 1, 4)
        mixed = row_values(scores, value, 1, valid)
        self.assertTrue(torch.equal(mixed, torch.tensor([2., 2., 2., 0.]).reshape_as(mixed)))
        self.assertTrue(torch.equal(row_values(scores * 0, value, 1, valid), value * 0))
        local = window_values(torch.ones_like(value), torch.ones_like(value), value, 1, valid, 4)
        self.assertTrue(torch.equal(local, mixed))

    def test_full_temporal_gradient_reset_and_checkpoint_parity(self):
        torch.manual_seed(7)
        model = self.model()
        left = torch.rand(1, 3, 8, 16, requires_grad=True)
        right = torch.rand_like(left, requires_grad=True)
        target = torch.full((1, 1, 8, 16), -1.25)
        valid = torch.ones(1, 8, 16, dtype=torch.bool)
        output = model(left, right)
        loss, pieces = correspondence_loss(output, target, valid)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(left.grad[:, t].abs().sum() > 0 for t in range(3)))
        self.assertTrue(all(right.grad[:, t].abs().sum() > 0 for t in range(3)))
        self.assertGreater(model.cross_qkv[0][0].weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.cross_qkv[1][0].weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.temporal_logits.grad.abs().sum().item(), 0)
        repeat = model(left.detach(), right.detach())
        self.assertTrue(torch.equal(output['flow'], repeat['flow']))
        checked = self.model(True); checked.load_state_dict(model.state_dict())
        checked_output = checked(left.detach(), right.detach())
        checked_loss, _ = correspondence_loss(checked_output, target, valid)
        checked_loss.backward()
        self.assertTrue(torch.equal(repeat['flow'], checked_output['flow']))
        self.assertTrue(torch.allclose(model.cross_qkv[0][0].weight.grad,
                                      checked.cross_qkv[0][0].weight.grad, atol=1e-6))
        self.assertEqual(len(pieces), 6)

    def test_actual_height_padding_and_input_contract(self):
        model = self.model()
        output = model(torch.rand(1, 3, 9, 9), torch.rand(1, 3, 9, 9))
        self.assertEqual(output['flow'].shape, (1, 1, 9, 9))
        self.assertTrue(torch.isfinite(output['flow']).all())
        self.assertTrue((output['probabilities'][..., -1] == 0).all())  # padded key x=12
        with self.assertRaises(ValueError):
            model(torch.rand(1, 2, 8, 16), torch.rand(1, 2, 8, 16))

    def test_nonfinite_invalid_targets_are_excluded_before_indexing(self):
        model = self.model()
        output = model(torch.rand(1, 3, 8, 16), torch.rand(1, 3, 8, 16))
        target = torch.full((1, 1, 8, 16), -1.)
        target[0, 0, 0, 0] = float('nan')
        target[0, 0, 0, 4] = float('inf')
        target[0, 0, 0, 8] = -float('inf')
        valid = torch.ones(1, 8, 16, dtype=torch.bool)
        loss, terms = correspondence_loss(output, target, valid)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(terms['valid_pixels'].item(), 125)
        loss.backward()

    def test_unequal_mask_microbatch_accumulation_matches_pooled_objective(self):
        torch.manual_seed(17)
        whole = self.model(); split = self.model(); split.load_state_dict(whole.state_dict())
        left, right = torch.rand(2, 3, 8, 16), torch.rand(2, 3, 8, 16)
        target = torch.full((2, 1, 8, 16), -1.25)
        target[1] = 5.5
        valid = torch.ones(2, 8, 16, dtype=torch.bool)
        valid[1, :, :7] = False
        x = torch.arange(16)[None, None, None, :]
        inside = valid[:, None] & (x + target >= 0) & (x + target <= 15)
        denominators = dict(valid=int(valid.sum()), inframe=int(inside.sum()),
                            correspondence=int(inside[..., ::4, ::4].sum()))
        pooled, _ = correspondence_loss(whole(left, right), target, valid)
        pooled.backward()
        summed = 0.
        for i in range(2):
            loss, _ = correspondence_loss(split(left[i:i+1], right[i:i+1]),
                target[i:i+1], valid[i:i+1], denominators)
            loss.backward(); summed += loss.detach()
        self.assertTrue(torch.allclose(pooled, summed, atol=1e-5))
        for a, b in zip(whole.parameters(), split.parameters()):
            self.assertTrue(torch.allclose(a.grad, b.grad, atol=2e-4, rtol=1e-4))

    def test_stride_lattice_has_no_half_pixel_offset(self):
        field = torch.tensor([0., 4., 8., 12.]).reshape(1, 1, 1, 4)
        native = native_lattice(field, 1, 16)[0, 0, 0]
        self.assertTrue(torch.allclose(native[:13], torch.arange(13.).float(), atol=1e-6))
        self.assertTrue(torch.equal(native[13:], torch.tensor([12., 12., 12.])))
        self.assertTrue(torch.allclose(native_lattice(field + 4, 1, 16), native_lattice(field, 1, 16) + 4))

    def test_actual_fifty_bins_keep_gradient(self):
        model = SpikingStereoCorrespondence(time_steps=50, widths=(4, 8, 8), heads=2,
            max_width=8, activation_checkpoint=True)
        left = torch.rand(1, 50, 8, 8, requires_grad=True)
        right = torch.rand(1, 50, 8, 8, requires_grad=True)
        output = model(left, right)
        loss, _ = correspondence_loss(output, torch.full((1, 1, 8, 8), -1.), torch.ones(1, 8, 8, dtype=torch.bool))
        loss.backward()
        self.assertTrue(all(left.grad[:, t].abs().sum() > 0 for t in range(50)))
        self.assertTrue(all(right.grad[:, t].abs().sum() > 0 for t in range(50)))
        self.assertTrue(torch.equal(output['residual'], torch.zeros_like(output['residual'])))
        self.assertTrue(torch.equal(output['null_flow'], torch.zeros_like(output['null_flow'])))
        self.assertTrue(torch.allclose(output['null_probability'], torch.full_like(output['null_probability'], torch.sigmoid(torch.tensor(-4.)))))

    def test_default_forward_and_gradient_match_immutable_p1_core(self):
        import hashlib
        from pathlib import Path
        import tarfile
        import types
        repository=Path(__file__).resolve().parents[3]
        archive=repository/'storage/experiment-runs/snn-depth-estimation/spiking-stereo-correspondence/source-ssc-sanity-v2/source.tar.gz'
        if not archive.exists():self.skipTest('immutable local P1 capsule unavailable')
        with tarfile.open(archive,'r:gz') as capsule:
            source=capsule.extractfile('projects/snn-depth-estimation/core/spiking_stereo_correspondence.py').read()
        self.assertEqual(hashlib.sha256(source).hexdigest(),'37faece983ad1ff65ee4d74a9af37a446d7305cab5d815ffe0342175fddc0246')
        frozen=types.ModuleType('immutable_p1_core_fixture');exec(compile(source,'immutable_p1_core','exec'),frozen.__dict__)
        torch.manual_seed(19)
        current=SpikingStereoCorrespondence(time_steps=3,widths=(4,8,8),heads=2,max_width=24,activation_checkpoint=True)
        with torch.no_grad():
            current.readout[-1].weight.normal_(0,.01)
        reference=frozen.SpikingStereoCorrespondence(time_steps=3,widths=(4,8,8),heads=2,max_width=24,activation_checkpoint=True)
        reference.load_state_dict(current.state_dict())
        left,right=torch.rand(1,3,16,24),torch.rand(1,3,16,24)
        actual=current(left,right);expected=reference(left,right)
        for key in ('flow','coarse_flow','logits','probabilities','null_logit','null_probability','null_flow','residual','activity'):
            self.assertTrue(torch.equal(actual[key],expected[key]),key)
        target=torch.full((1,1,16,24),-1.25);valid=torch.ones(1,16,24,dtype=torch.bool)
        loss,_=correspondence_loss(actual,target,valid);loss.backward()
        old_loss,_=frozen.correspondence_loss(expected,target,valid);old_loss.backward()
        for (name,a),(_,b) in zip(current.named_parameters(),reference.named_parameters()):
            self.assertTrue(torch.equal(a.grad,b.grad),name)

    def test_value_probe_preserves_matching_and_content_probe_retains_bias(self):
        model=self.model();model.eval()
        with torch.no_grad():model.relative_bias.copy_(torch.linspace(-.5,.5,len(model.relative_bias)))
        left,right=torch.rand(1,3,8,16),torch.rand(1,3,8,16)
        intact=model(left,right);value_zero=model(left,right,zero_values=True)
        explicit_default=model(left,right,zero_content=False,zero_values=False)
        self.assertTrue(torch.equal(intact['flow'],explicit_default['flow']))
        for key in ('logits','probabilities','coarse_flow'):
            self.assertTrue(torch.equal(intact[key],value_zero[key]),key)
        content_zero=model(left,right,zero_content=True)
        x=torch.arange(content_zero['logits'].shape[-1]);relative=x[None,:]-x[:,None]
        expected=model.relative_bias[relative+model.max_width//4-1]
        self.assertTrue(torch.equal(content_zero['logits'],expected[None,None].expand_as(content_zero['logits'])))
        self.assertEqual(intact['activity_per_time'].shape,(3,7))
        self.assertTrue(torch.allclose(intact['activity_per_time'].mean(0),intact['activity']))

    def test_lif_state_has_untruncated_gradient(self):
        first = torch.tensor([.2], requires_grad=True)
        _, state = lif(first, 0.)
        spike, state = lif(torch.tensor([1.]), state)
        spike.sum().backward()
        self.assertGreater(first.grad.abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
