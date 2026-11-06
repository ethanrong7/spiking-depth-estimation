"""Tiny named rational examples only; never invokes the full lattice search."""
from fractions import Fraction as F
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import exact_compatibility_search as s6


class ExactCompatibilityChecks(unittest.TestCase):
    def test_integer_threshold_reset_and_domain(self):
        self.assertEqual(s6.encode(0, 499), ())
        self.assertEqual(s6.encode(100, 0), tuple(range(4, 50, 5)))
        self.assertEqual(s6.encode(75, 475)[:2], (0, 6))  # retained overshoot
        for pair in ((101, 0), (0, 500), (-1, 0)):
            with self.assertRaises(ValueError):
                s6.encode(*pair)

    def test_target_tie_absolute_endpoints_and_fixed_windows(self):
        events = s6.encode(50, 250)
        self.assertEqual(events, (4, 14, 24, 34, 44))
        summary, missing = s6.signature(events)
        self.assertEqual(missing, ())
        self.assertEqual(summary, (5, 4, 44, 4, 14, 24, 34, 1, 1, 3, 5))
        self.assertEqual(s6.baseline_values(events)["target_one_interval"], F(1, 2))
        self.assertEqual(s6.signature(())[1], ("a1", "a2", "b1", "b2"))

    def test_projection_zero_record_open_upper_closed_domain_zero(self):
        interval = s6.project(())
        self.assertEqual(interval, s6.Interval(F(0), F(1, 10), True, False))
        self.assertTrue(interval.contains(F(0)))
        self.assertFalse(interval.contains(F(1, 10)))
        self.assertEqual(s6.phase_interval((), F(1, 20)), s6.Interval(F(0), F(5, 2), True, False))
        self.assertFalse(s6.feasible_pair((), F(0), F(5)))

    def test_record_strict_domain_bound_and_closed_domain_one(self):
        last_event = s6.project((49,))
        self.assertEqual(last_event.lower, 0)
        self.assertFalse(last_event.lower_closed)
        self.assertFalse(last_event.contains(F(0)))
        self.assertTrue(s6.phase_interval((49,), F(1, 100)).nonempty())
        unit = s6.project(s6.encode(100, 0))
        self.assertEqual(unit.upper, 1)
        self.assertTrue(unit.upper_closed and unit.contains(F(1)))
        self.assertFalse(s6.project(tuple(range(50))).nonempty())

    def test_projection_matches_direct_phase_envelope_at_fixed_intensities(self):
        for events in ((), (49,), s6.encode(100, 0), s6.encode(49, 240)):
            projection = s6.project(events)
            for intensity in (F(0), F(1, 100), F(1, 20), F(1, 10), F(49, 100), F(1)):
                phase = s6.phase_interval(events, intensity)
                self.assertEqual(projection.contains(intensity), phase.nonempty())
                if phase.nonempty():
                    self.assertTrue(s6.feasible_pair(events, intensity, (phase.lower + phase.upper) / 2))

    def test_equal_endpoint_flags_and_strict_gap_reject_touching(self):
        self.assertTrue(s6.Interval(F(1, 2), F(1, 2), True, True).nonempty())
        self.assertFalse(s6.Interval(F(1, 2), F(1, 2), True, False).nonempty())
        left = s6.Interval(F(0), F(1, 10), True, False)
        touching = s6.Interval(F(1, 10), F(1, 5), False, False)
        separated = s6.Interval(F(11, 100), F(1, 5), False, True)
        self.assertIsNone(s6.strict_gap(left, touching))
        self.assertEqual(s6.strict_gap(left, separated), (F(1, 100), "L_below_D"))

    def test_supplied_weak_case_is_rejected_by_strong_signature(self):
        left, right, distractor = (s6.encode(*pair) for pair in ((49, 240), (49, 290), (51, 205)))
        self.assertEqual(left, (5, 15, 25, 35, 46))
        self.assertEqual(right, (4, 14, 24, 34, 45))
        self.assertEqual(distractor, (5, 15, 25, 35, 44))
        self.assertEqual(s6.baseline_values(left)["target_three_intervals"], s6.baseline_values(distractor)["target_three_intervals"])
        self.assertNotEqual(s6.baseline_values(left)["first_to_last"], s6.baseline_values(distractor)["first_to_last"])
        self.assertNotEqual(s6.signature(left)[0][1:3], s6.signature(distractor)[0][1:3])
        generators = {left: [(49, 240)], right: [(49, 290)], distractor: [(51, 205)]}
        with self.assertRaisesRegex(ValueError, "strong signature mismatch"):
            s6.certificate(left, right, distractor, generators, {e: s6.project(e) for e in generators}, {})

    def test_common_intensity_phase_order_and_hard_counter_stop(self):
        self.assertEqual(s6.shared_pair([(50, 20), (49, 240), (49, 230)], [(50, 40), (49, 290)]), (49, 230, 290))
        # Deliberately artificial grouping tests ONLY the pre-operation cap;
        # no candidate is accepted or interpreted as a mathematical witness.
        events = ((), (49,))
        counters = {"ordered_L_D_comparisons": 0, "R_record_candidate_checks": 0}
        with self.assertRaisesRegex(s6.StopSearch, "ordered_L_D_comparison_limit"):
            s6.find_witness({(0,): events}, {}, {e: s6.project(e) for e in events}, counters, lambda: None, ld_limit=0)
        self.assertEqual(counters["ordered_L_D_comparisons"], 0)


if __name__ == "__main__":
    unittest.main()
