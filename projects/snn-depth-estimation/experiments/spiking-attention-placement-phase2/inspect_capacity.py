"""Read-only cccl3 admission snapshot. Never imports torch or starts CUDA.

Run through the documented SSH connection using the cluster Python. This is an
observation, not launch authorization: a launch must repeat the UUID, Slurm,
duplicate-worker and capacity checks inside its allocation.
"""
import csv
import json
import os
import re
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ALLOWED = {
    0: "GPU-91144565-173b-4884-43c0-1cbff4ebdb05",
    1: "GPU-4b32e268-a510-752a-111a-84a9220c0b52",
    2: "GPU-70b4f15e-8b35-76b6-2845-90bf68430ff1",
    3: "GPU-783f12a1-f199-8f24-a7ee-472669f5e017",
    4: "GPU-a0c434d5-a0b5-6155-790d-3dd7bfa5d82c",
}


def output(*args):
    return subprocess.check_output(args, text=True, timeout=25).strip()


def read(path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None


def process(pid):
    path = Path("/proc") / str(pid)
    try:
        return dict(pid=int(pid), uid=path.stat().st_uid,
                    command=(path / "cmdline").read_bytes().replace(b"\0", b" ").decode(),
                    cgroup=(path / "cgroup").read_text().strip())
    except FileNotFoundError:
        return dict(pid=int(pid), exited_during_snapshot=True)


def snapshot(run_base=None):
    raw = Path(run_base if run_base is not None else os.environ['STEREO_RUN_BASE'])
    assert socket.gethostname().split(".")[0] == "cccl3"
    assert output("id", "-un") == "rongliang"
    queue = output("squeue", "-h", "-r", "-o", "%i|%u|%j|%T|%N|%b")
    jobs = {}
    for row in queue.splitlines():
        job = row.split('|')[0]
        try:
            jobs[job] = output('scontrol', 'show', 'job', '-do', job)
        except subprocess.CalledProcessError:
            # Queue and detail reads are not atomic; an ending job is retried
            # on the next observation rather than terminating the watcher.
            if job in output('squeue', '-h', '-r', '-o', '%i').splitlines():
                raise
    def allocated_minors(text):
        indices = set()
        for group in re.findall(r'\bGRES=gpu:\d+\(IDX:([0-9,-]+)\)', text):
            for part in group.split(','):
                bounds = part.split('-')
                indices.update(range(int(bounds[0]), int(bounds[-1]) + 1))
        return indices
    gpu_rows = list(csv.reader(output("nvidia-smi", "--query-gpu=index,uuid,pci.bus_id,name,memory.total,memory.used,memory.free,utilization.gpu",
                                      "--format=csv,noheader,nounits").splitlines(), skipinitialspace=True))
    app_rows = list(csv.reader(output("nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory,process_name",
                                      "--format=csv,noheader,nounits").splitlines(), skipinitialspace=True))
    info = [p.read_text() for p in Path("/proc/driver/nvidia/gpus").glob("*/information")]
    gpus = []
    for index, uuid, pci, name, total, used, free, utilization in gpu_rows:
        index = int(index)
        details, = [s for s in info if uuid in s]
        minor = int(re.search(r"Device Minor:\s*(\d+)", details).group(1))
        owners = [job for job, text in jobs.items() if minor in allocated_minors(text)]
        apps = [dict(process(pid), memory_mib=int(memory), executable=exe)
                for app_uuid, pid, memory, exe in app_rows if app_uuid == uuid]
        allowed = ALLOWED.get(index) == uuid
        # Historical A screen includes actual training and official validation.
        # Unknown co-tenant future peaks preclude co-location admission.
        reasons = []
        if not allowed:
            reasons.append("outside physical GPU 0..4 authorization")
        if len(owners) != 1:
            reasons.append("requires exactly one verified existing Slurm allocation")
        if apps:
            reasons.append("active co-tenants; future validation headroom not reserved")
        if int(free) < 21860 + 4096:
            reasons.append("free memory below historical A 21860 MiB plus 4096 MiB reserve")
        gpus.append(dict(physical_index=index, uuid=uuid, pci_bus_id=pci, device_minor=minor,
                         name=name, total_mib=int(total), used_mib=int(used), free_mib=int(free),
                         utilization_percent=int(utilization),
                         authorized_device=allowed, jobs=owners, processes=apps,
                         admission_candidate=not reasons, admission_reasons=reasons))
    duplicates = []
    for path in Path("/proc").iterdir():
        if path.name.isdigit():
            p = process(path.name)
            if "spiking-attention-placement-phase2" in p.get("command", ""):
                duplicates.append(p)
    endpoints = {}
    for study, arms in (("spiking-attention-placement-v1", ("A", "B", "C")),
                        ("v0-guided-stereo-transformer-v1", ("C",))):
        for arm in arms:
            folder = raw / study / "train" / arm
            endpoints[f"{study}/{arm}"] = dict(
                status=read(folder / "status.json"), checkpoint=read(folder / "checkpoint.json"),
                result=read(folder / "result.json"),
                step30000_exists=(folder / "step-30000.pt").exists(),
                final_probe=read(raw / study / "diagnostics" / f"{arm}-30000-probes.status.json"))
    return dict(observed_utc=datetime.now(timezone.utc).isoformat(), hostname=socket.gethostname(),
                user=output("id", "-un"), queue=queue, slurm_jobs=jobs, gpus=gpus,
                phase2_processes=duplicates, endpoints=endpoints,
                historical_A_peak_reserved_mib=21860, headroom_reserve_mib=4096,
                observation_only=True, cuda_initialized=False)


if __name__ == "__main__":
    print(json.dumps(snapshot(), indent=2))
