"""Allocated-GPU energy telemetry; never CPU/sensor/total-system energy.

Uses the existing project's read-only ctypes NVML interface, extended with
explicit interval integration. No NVML/CUDA work occurs at import time.
"""
import ctypes as C
import json
import math
from pathlib import Path
import threading
import time


def integrate_power(samples, start, end):
    """Trapezoid watts over monotonic seconds, with interpolated endpoints.

    Bracketing samples are mandatory; do not silently extrapolate uncovered
    portions of a measured interval or bridge an unavailable sensor reading.
    """
    if not start < end or len(samples) < 2:
        raise ValueError('Positive interval and bracketing power samples required')
    points = sorted((float(t), float(w)) for t, w in samples)
    if any(not math.isfinite(t) or not math.isfinite(w) or w < 0 for t, w in points):
        raise ValueError('Invalid power sample')
    if points[0][0] > start or points[-1][0] < end:
        raise ValueError('Power trace does not bracket measured interval')
    energy, covered = 0., 0.
    for (t0, p0), (t1, p1) in zip(points, points[1:]):
        if t1 <= t0:
            raise ValueError('Strictly increasing sample timestamps required')
        a, b = max(t0, start), min(t1, end)
        if a < b:
            pa = p0 + (p1 - p0) * (a - t0) / (t1 - t0)
            pb = p0 + (p1 - p0) * (b - t0) / (t1 - t0)
            energy += (pa + pb) * .5 * (b - a)
            covered += b - a
    if not math.isclose(covered, end - start, rel_tol=1e-9, abs_tol=1e-6):
        raise ValueError('Incomplete power trace coverage')
    return energy


class GPUPowerTrace:
    """Read-only per-UUID NVML samples and optional cumulative-energy counter."""
    def __init__(self, uuid, path, interval=.05):
        uuid = uuid.decode() if isinstance(uuid, bytes) else str(uuid)
        self.uuid = uuid if uuid.startswith(('GPU-', 'MIG-')) else 'GPU-' + uuid
        self.path, self.interval = Path(path), interval
        self.rows, self.errors = [], []
        self.stop_event = threading.Event()
        self.lib = C.CDLL('libnvidia-ml.so.1')
        self._check(self.lib.nvmlInit_v2())
        self.handle = C.c_void_p()
        self.lib.nvmlDeviceGetHandleByUUID.argtypes = [C.c_char_p, C.POINTER(C.c_void_p)]
        self._check(self.lib.nvmlDeviceGetHandleByUUID(self.uuid.encode(), C.byref(self.handle)))
        for name in ('nvmlDeviceGetPowerUsage', 'nvmlDeviceGetTotalEnergyConsumption'):
            function = getattr(self.lib, name, None)
            if function:
                function.argtypes = [C.c_void_p, C.c_void_p]
                function.restype = C.c_int
        self.thread = threading.Thread(target=self._loop, daemon=True)

    @staticmethod
    def _check(code):
        if code:
            raise RuntimeError('NVML status ' + str(code))

    def read(self):
        row = {'monotonic_seconds': time.monotonic(), 'epoch_ns': time.time_ns(), 'uuid': self.uuid}
        for name, label, kind, unit in [('nvmlDeviceGetPowerUsage', 'watts', C.c_uint, 1000.),
                                        ('nvmlDeviceGetTotalEnergyConsumption', 'joules_counter', C.c_ulonglong, 1000.)]:
            function = getattr(self.lib, name, None)
            value = kind()
            status = function(self.handle, C.byref(value)) if function else 'symbol_missing'
            row[label] = value.value / unit if status == 0 else None
            row[label + '_status'] = status
        row['query_finished_monotonic_seconds'] = time.monotonic()
        return row

    def _loop(self):
        try:
            with self.path.open('x') as stream:
                while not self.stop_event.is_set():
                    tick = time.monotonic()
                    row = self.read()
                    self.rows.append(row)
                    stream.write(json.dumps(row) + '\n')
                    stream.flush()
                    self.stop_event.wait(max(0., self.interval - (time.monotonic() - tick)))
        except Exception as error:
            self.errors.append(repr(error))

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=5)
        self.lib.nvmlShutdown()
        if self.thread.is_alive() or self.errors:
            raise RuntimeError('Telemetry failed: ' + str(self.errors))

    def summary(self, start, end, delivered, *, idle_watts=None):
        if end - start < 30 or delivered <= 0:
            raise ValueError('Scientific GPU energy intervals require >=30seconds and delivered outputs')
        rows = [r for r in self.rows if start - self.interval <= r['monotonic_seconds'] <= end + self.interval]
        if any(row['watts'] is None for row in rows):
            raise RuntimeError('Power sampling unavailable inside measured interval')
        samples = [(r['monotonic_seconds'], r['watts']) for r in self.rows if r['watts'] is not None]
        joules = integrate_power(samples, start, end)
        result = {'scope': 'GPU joules during declared pipeline; CPU/sensor excluded',
                  'gpu_uuid': self.uuid, 'seconds': end - start, 'delivered_maps': delivered,
                  'integrated_gpu_joules': joules, 'gpu_joules_per_map': joules / delivered,
                  'power_sample_count': len(rows), 'sampling_interval_seconds': self.interval,
                  'power_value_count': len(set(r['watts'] for r in rows)),
                  'max_sample_gap_seconds': max(b[0] - a[0] for a, b in zip(samples, samples[1:])),
                  'uncertainty': 'Report repeated interleaved interval variation; samples are temporally correlated'}
        if idle_watts is not None:
            result['separately_idle_adjusted_gpu_joules_per_map'] = (joules - idle_watts * (end - start)) / delivered
        return result
