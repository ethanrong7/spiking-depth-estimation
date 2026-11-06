"""Tiny synthetic CPU geometry/state checks, never scientific inference."""
import unittest
import torch
from s2_geometry import lookup_support, score_evidence, nearest_native_lattice
from s2_state_hooks import clone, snapshot, restore, equivalent, replay_primary, capture_native


class Checks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_exact_native_centers_and_right_exclusion(self):
        target = torch.full((1, 1, 250, 400), -8.)
        valid = torch.ones(1, 250, 400, dtype=torch.bool)
        data = nearest_native_lattice(target, valid, (64, 104))
        self.assertEqual(data["native_center_x"][2].item(), 0)
        self.assertEqual(data["native_center_y"][1].item(), 1)
        self.assertEqual(data["qgt"][0, 1, 4].item(), 2)
        self.assertFalse(data["valid"][0, 1, 2].item())
        self.assertTrue(data["valid"][0, 1, 4].item())
        self.assertTrue(torch.equal(data["valid"], data["nonboundary"]))

    def test_valid_keys_flat_tied_and_separated_scores(self):
        scores = torch.zeros(4, 9)
        scores[1, 4] = 5
        scores[2, 4:6] = 5
        scores[3, 0] = 100  # padded unrestricted maximum
        scores[3, 4] = 5
        valid = torch.tensor([False, True, True, True, True, True, True, True, False])
        evidence = score_evidence(scores, torch.full((4,), 4.), valid)
        self.assertEqual(evidence["evidence_available"].tolist(), [False, True, False, True])
        self.assertTrue(evidence["unrestricted_peak_padded"][3])
        self.assertEqual(evidence["nearest_gt_rank"][1].item(), 1)
        self.assertEqual(evidence["row_std"][0].item(), 0)

    def test_lookup_matches_actual_basis_and_odd_pool_truncation(self):
        # Compare the support/mass to a complete identity basis through the
        # actual CPU kernel; future GPU calls use that GPU's kernel separately.
        import torch.nn.functional as F
        width = 17
        q = torch.tensor([-1., 0., 3.5, 8., 16., 19.])
        group = torch.ones(len(q), width, dtype=torch.bool)
        result, taps = lookup_support(q, group, [17, 8, 4, 2])
        identity = torch.eye(width)[None, :, None]
        for level, tap in enumerate(taps):
            pooled = identity
            for _ in range(level):
                pooled = F.avg_pool2d(pooled, (1, 2), (1, 2))
            centers = q[:, None] / (2 ** level) + torch.arange(-4, 5)
            grid = torch.stack((2 * centers / (pooled.shape[-1] - 1) - 1, torch.zeros_like(centers)), -1)[None]
            actual = F.grid_sample(pooled, grid, align_corners=True)[0].permute(1, 2, 0)
            expected = torch.zeros_like(actual)
            indices = tap["indices"].reshape(len(q), 9, 3)
            weights = tap["weights"].reshape(len(q), 9, 3)
            active = tap["active"].reshape(len(q), 9, 3)
            for offset in range(2 ** level):
                fine = indices * (2 ** level) + offset
                expected.scatter_add_(-1, fine.clamp(0, width - 1), active * weights / (2 ** level))
            torch.testing.assert_close(expected, actual, atol=1e-7, rtol=1e-6)
            torch.testing.assert_close(tap["GT_mass_per_sample"], actual.sum(-1))
        self.assertEqual(result["support_level_0"][3].item(), 1)

    def test_snapshot_restores_mutable_hidden_and_all_neuron_fields(self):
        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.membrane = torch.tensor([.2])
                self._memories = {"v": torch.tensor([.3]), "spike": torch.tensor([0.])}
                self.v_threshold = torch.tensor([.5])
                self._diagnostics = [torch.tensor([.7])]
        model = Module()
        saved = snapshot(model)
        model.membrane += 2
        model._memories["v"] += 3
        model._diagnostics.append(torch.tensor([1.]))
        restore(model, saved)
        equivalent(snapshot(model), saved)
        mutable = [torch.tensor([0.])]
        copied = clone(mutable)
        mutable[0] += 1
        self.assertEqual(copied[0].item(), 0)
        with self.assertRaises(RuntimeError):
            equivalent(torch.tensor([0., 1.]), torch.tensor([1., 0.]))

    def test_one_step_replay_restores_native_exit_state(self):
        class FakeUpdate(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.membrane = torch.full((1, 1, 2, 3), .2)
                self._diagnostics = []
            def forward(self, net, context, corr, disp):
                charge = self.membrane + corr
                spike = (charge >= .5).float()
                self.membrane = charge - .5 * spike
                net[0] = spike
                self._diagnostics.append(corr.mean())
                return net, self.membrane, self.membrane * .25, torch.cat((self.membrane, self.membrane), 1)
        model = torch.nn.Module()
        model.update_block = FakeUpdate()
        args = ([torch.zeros(1, 1, 2, 3)], [torch.zeros(1, 1, 2, 3)],
                torch.full((1, 1, 2, 3), .4), torch.zeros(1, 2, 2, 3))
        before = snapshot(model)
        output = clone(model.update_block(*clone(args)))
        after = snapshot(model)
        calls = [dict(coordinates=torch.full((1, 2, 2, 3), .3), vector=torch.full((1, 1, 2, 3), .3)) for _ in range(16)]
        calls[15] = dict(coordinates=torch.full((1, 2, 2, 3), .4), vector=clone(args[2]))
        record = dict(steps={16: dict(arguments=clone(args), before=before, output=output, after=after)},
                      calls=calls, correlation=lambda coords: coords[:, :1])
        model.update_block.membrane.fill_(1.7)
        native_exit = snapshot(model)
        result = replay_primary(model, record)
        self.assertEqual(result["factual"]["factual_equivalence_max_abs"], 0)
        self.assertFalse(torch.equal(result["plus"]["corr"], result["minus"]["corr"]))
        equivalent(snapshot(model), native_exit)

    def test_endpoint_counts_are_exact_and_float_tolerance_stays_fixed(self):
        from run_s2_diagnostics import endpoint_gate
        reference = dict(mean_scene_epe=2., pooled_epe=2., valid_pixels=6000000, valid_fraction=1.,
                         bad1=20., bad3=10., bad5=5., scene_epe={"scene_0120": 2.},
                         per_frame=[dict(scene=f"scene_{120+i//2:04d}", frame="0001" if i % 2 == 0 else "0255",
                                         valid_pixels=100000, epe=2., bad1=20., bad3=10., bad5=5.) for i in range(60)])
        changed = clone(reference)
        changed["valid_pixels"] += 1
        with self.assertRaisesRegex(RuntimeError, "exact endpoint valid-pixel"):
            endpoint_gate(changed, reference)
        changed = clone(reference)
        changed["mean_scene_epe"] += 1e-5
        endpoint_gate(changed, reference)
        changed["mean_scene_epe"] += 1e-3
        with self.assertRaisesRegex(RuntimeError, "endpoint mismatch"):
            endpoint_gate(changed, reference)

    def test_native_hook_counts_flags_and_cleanup(self):
        class FakeCorr:
            def __init__(self, left, right):
                pass
            def __call__(self, coordinates):
                return coordinates[:, :1]
        class FakeUpdate(torch.nn.Module):
            def forward(self, net, inp, corr, disp, iter16=True, iter32=True):
                return net, corr, corr, disp
        model = torch.nn.Module()
        model.update_block = FakeUpdate()
        tensor = torch.zeros(1, 2, 2, 3)
        original_call = FakeCorr.__call__
        with capture_native(model, FakeCorr) as captured:
            corr = FakeCorr(tensor, tensor)
            for _ in range(32):
                model.update_block([tensor], [tensor], corr(tensor), tensor, iter16=True, iter32=True)
        self.assertEqual(captured["updater_calls"], 32)
        self.assertEqual(captured["steps"][16]["kwargs"], dict(iter16=True, iter32=True))
        with self.assertRaisesRegex(RuntimeError, "full three-scale"):
            with capture_native(model, FakeCorr):
                corr = FakeCorr(tensor, tensor)
                model.update_block([tensor], [tensor], corr(tensor), tensor, iter32=False)
        self.assertIs(FakeCorr.__call__, original_call)
        self.assertEqual(len(model.update_block._forward_pre_hooks), 0)
        self.assertEqual(len(model.update_block._forward_hooks), 0)


if __name__ == "__main__":
    unittest.main()
