"""Tiny audit-path fixtures; no real dataset bulk check or model inference."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

P=Path(__file__).resolve().parents[1]/'experiments/stereo-spiking-transformer-study/scripts/audit_paired_pilot.py'
spec=importlib.util.spec_from_file_location('audit_paired_pilot',P);audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


class PairedAuditTests(unittest.TestCase):
    def test_saved_sensor_counts_multiplicity_and_corruption(self):
        spikes=np.zeros((50,1,2,4),dtype=np.uint8);spikes[0,0,0,0]=1;spikes[-1,0,1,3]=1
        events=np.zeros((50,2,2,4),dtype=np.uint16);events[0,0,0,0]=3;events[-1,1,1,3]=2
        trace=[dict(spike_fraction=float(s.mean()),on_events=int(e[0].sum()),off_events=int(e[1].sum())) for s,e in zip(spikes,events)]
        self.assertEqual(audit.check_sensor_array(spikes,'spike',[2,4],trace)['total_spikes'],2)
        self.assertEqual(audit.check_sensor_array(events,'event',[2,4],trace)['max_count'],3)
        damaged=events.copy();damaged[-1,1,1,3]=1
        with self.assertRaises(AssertionError):audit.check_sensor_array(damaged,'event',[2,4],trace)
        bad=spikes.copy();bad[0,0,0,0]=2
        with self.assertRaises(ValueError):audit.check_sensor_array(bad,'spike',[2,4],trace)
        with self.assertRaises(ValueError):audit.safe('/tmp/source','../escape')
        with self.assertRaises(ValueError):audit.safe('/tmp/source','TEST/C/0001/file')

    def test_trace_requires_all450_steps_both_views_and_bounded_charge(self):
        trace=[]
        for view in ('l','r'):
            trace.extend(dict(view=view,global_index=i,source_time=i/50,spike_fraction=0.,on_events=0,
              off_events=0,charge_min=0.,charge_max=4.9,charge_mean=1.) for i in range(1,451))
        self.assertEqual(len(audit.validate_trace(trace,5.)['l']),450)
        bad=[dict(r) for r in trace];bad[-1]['charge_max']=5.
        with self.assertRaises(ValueError):audit.validate_trace(bad,5.)
        with self.assertRaises(ValueError):audit.validate_trace(trace[:-1],5.)

    def test_independent_PFM_orientation_nearest_centers_horizontal_scale(self):
        # PFM reader shape is intentionally pinned to original native geometry;
        # this fixture creates one metadata/simple value file, no model/data run.
        values=np.broadcast_to(np.arange(960,dtype=np.float32),(540,960)).copy()
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'d.pfm';p.write_bytes(b'Pf \n960 540\n-1\n'+np.flipud(values).astype('<f4').tobytes())
            half=audit.source_disparity(p,[270,480]);quarter=audit.source_disparity(p,[135,240])
            self.assertEqual(half[0,0],.5) # nearest-exact source x=1, d*=.5
            self.assertEqual(quarter[0,0],.5) # nearest-exact source x=2, d*=.25
            self.assertEqual(quarter[0,1],1.5)
            self.assertEqual(half.shape,(270,480));self.assertEqual(quarter.shape,(135,240))

if __name__=='__main__':unittest.main()
