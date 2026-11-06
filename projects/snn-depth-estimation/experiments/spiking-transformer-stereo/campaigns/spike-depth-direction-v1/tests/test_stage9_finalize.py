"""Tiny serializer/guard checks only; no native/model/dataset imports."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
P=Path(__file__).resolve().parents[1]/'scripts/stage9_finalize.py'
spec=importlib.util.spec_from_file_location('finalize',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class FinalizeTests(unittest.TestCase):
 def test_new_output_atomic_no_overwrite(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'packet.json';m.dump_json(p,{'x':[1,2]});before=p.read_bytes()
   self.assertEqual(json.loads(before),{'x':[1,2]});self.assertFalse(p.with_suffix('.json.part').exists())
   with self.assertRaises(FileExistsError):m.dump_json(p,{'x':[3]})
   self.assertEqual(p.read_bytes(),before)
 def test_reject_nonfinite_json(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'bad.json'
   with self.assertRaises(ValueError):m.dump_json(p,{'cost':float('nan')})
   self.assertFalse(p.exists())
 def test_no_local_or_nonfinal_execution(self):
  with patch.dict('os.environ',{},clear=True):
   with self.assertRaises(RuntimeError):m.finalize(None,'V','unused','unused',1,{'updates':5000})
  with patch.dict('os.environ',{'SLURM_JOB_ID':'toy','SLURM_JOB_PARTITION':'debug'}):
   with self.assertRaises(ValueError):m.finalize(None,'V','unused','unused',1,{'updates':2500})
   with self.assertRaises(ValueError):m.finalize(None,'V','unused','unused',1,{'updates':5000})
if __name__=='__main__':unittest.main()
