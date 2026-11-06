"""Assigned-GPU telemetry; active training windows are separate from inspections."""
import os
import statistics
import subprocess
import threading
import time


class Sampler:
    def __init__(self):
        rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True).splitlines()
        uuids={r.split(',')[1].strip() for r in rows if r.split(',')[0].strip()==str(os.getpid())}
        if len(uuids)!=1:raise RuntimeError('cannot resolve unique assigned GPU UUID')
        self.uuid=next(iter(uuids));self.samples=[];self.stop=threading.Event();self.thread=threading.Thread(target=self._run,daemon=True);self.thread.start()
    def _run(self):
        while not self.stop.is_set():
            try:
                x=subprocess.check_output(['nvidia-smi','-i',self.uuid,'--query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu,clocks.sm,pstate','--format=csv,noheader,nounits'],text=True,timeout=5).strip().split(',')
                self.samples.append(dict(monotonic=time.monotonic(),gpu_util=float(x[0]),memory_util=float(x[1]),memory_mib=float(x[2]),power_w=float(x[3]),temperature_c=float(x[4]),sm_clock_mhz=float(x[5]),pstate=x[6].strip()))
            except Exception as error:self.samples.append(dict(monotonic=time.monotonic(),error=str(error)))
            self.stop.wait(1.)
    def close(self):self.stop.set();self.thread.join()
    def summary(self,windows):
        selected=[s for s in self.samples if 'gpu_util' in s and any(w['started_monotonic']<=s['monotonic']<=w['ended_monotonic'] for w in windows)]
        if not selected:raise RuntimeError('no active-window telemetry')
        values=[s['gpu_util'] for s in selected]
        return dict(samples=len(selected),gpu_util_mean=statistics.mean(values),gpu_util_median=statistics.median(values),gpu_util_low_under10_fraction=sum(v<10 for v in values)/len(values),gpu_util_zero_fraction=sum(v==0 for v in values)/len(values),
            maximum_memory_mib=max(s['memory_mib'] for s in selected),power_w_mean=statistics.mean(s['power_w'] for s in selected),temperature_c_max=max(s['temperature_c'] for s in selected),temperature_c_mean=statistics.mean(s['temperature_c'] for s in selected),sm_clock_mhz_mean=statistics.mean(s['sm_clock_mhz'] for s in selected),pstates=sorted({s['pstate'] for s in selected}),gpu_uuid=self.uuid)
