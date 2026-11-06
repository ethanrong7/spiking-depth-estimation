import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1] / 'experiments/sew-t50-training/scripts/continue_budget.py'
SPEC = importlib.util.spec_from_file_location('continue_budget', PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ContinuationTests(unittest.TestCase):
    def test_nineteen_hours_has_four_bounded_children_and_no_deadline_inheritance(self):
        calls = []
        class Child:
            def wait(self): return 0
        def spawn(command, env):
            self.assertNotIn('SPIKESTEREO_SEGMENT_DEADLINE', env)
            calls.append(command)
            return Child()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'status.json').write_text(json.dumps({'status': 'PAUSED_BUDGET'}))
            times = iter([0, 0, 18000, 36000, 54000, 68400])
            with patch.dict(os.environ, {'SPIKESTEREO_SEGMENT_DEADLINE': 'old'}), \
                 patch.object(MODULE.signal, 'signal'):
                os.environ.pop('SLURM_JOB_END_TIME', None)
                result = MODULE.run_budget(['trainer'], root, 68400,
                    clock=lambda: next(times), spawn=spawn)
        self.assertEqual(result, 0)
        self.assertEqual([int(x[-1]) for x in calls], [18000, 18000, 18000, 14400])

    def test_failure_is_not_automatically_retried(self):
        calls = []
        class Child:
            def wait(self): return 17
        def spawn(command, env):
            calls.append(command)
            return Child()
        with tempfile.TemporaryDirectory() as temp, patch.object(MODULE.signal, 'signal'):
            result = MODULE.run_budget(['trainer'], Path(temp), 68400,
                clock=lambda: 0, spawn=spawn)
        self.assertEqual(result, 17)
        self.assertEqual(len(calls), 1)


if __name__ == '__main__':
    unittest.main()
