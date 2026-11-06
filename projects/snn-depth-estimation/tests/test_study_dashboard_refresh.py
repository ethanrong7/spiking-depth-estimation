"""Read-only monitor process cleanup; tiny local processes, no cluster calls."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

P=Path(__file__).resolve().parents[1]/'experiments/stereo-spiking-transformer-study/scripts/serve_dashboard.py'
spec=importlib.util.spec_from_file_location('study_dashboard_refresh_test',P)
dashboard=importlib.util.module_from_spec(spec);spec.loader.exec_module(dashboard)


class RefreshTests(unittest.TestCase):
    def test_completed_refresh_keeps_output_and_returncode(self):
        result=dashboard.run_refresh([sys.executable,'-c','print("saved");raise SystemExit(2)'],timeout=2)
        self.assertEqual(result.returncode,2)
        self.assertEqual(result.stdout,'saved\n')

    def test_timeout_kills_child_that_inherits_pipes(self):
        with tempfile.TemporaryDirectory() as folder:
            marker=Path(folder)/'unexpected-child-survived'
            child='import time,pathlib;time.sleep(0.8);pathlib.Path('+repr(str(marker))+').touch()'
            parent='import subprocess,sys,time;subprocess.Popen([sys.executable,"-c",'+repr(child)+']);time.sleep(10)'
            started=time.monotonic()
            with self.assertRaises(subprocess.TimeoutExpired):
                dashboard.run_refresh([sys.executable,'-c',parent],timeout=.3)
            self.assertLess(time.monotonic()-started,2)
            time.sleep(.85)
            self.assertFalse(marker.exists())


if __name__=='__main__':unittest.main()
