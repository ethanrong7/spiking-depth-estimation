"""Check the scheduler allocation independently of CUDA enumeration before training."""
import csv
import json
import os
import re
import subprocess
from pathlib import Path

ALLOWED = {"1": "GPU-4b32e268-a510-752a-111a-84a9220c0b52",
           "2": "GPU-70b4f15e-8b35-76b6-2845-90bf68430ff1"}


def output(*args):
    return subprocess.check_output(args, text=True).strip()


def main():
    job = os.environ["SLURM_JOB_ID"]
    devices = list(csv.reader(output("nvidia-smi", "--query-gpu=index,uuid",
                                     "--format=csv,noheader").splitlines(), skipinitialspace=True))
    info = [p.read_text() for p in Path("/proc/driver/nvidia/gpus").glob("*/information")]
    alloc = output("scontrol", "show", "job", "-do", job).split()
    print("NVML devices", devices, "allocated GRES", [s for s in alloc if "GRES=" in s], flush=True)
    for index, uuid in ALLOWED.items():
        if uuid not in {visible_uuid for _, visible_uuid in devices}:
            continue
        target_info, = [s for s in info if uuid in s]
        minor = re.search(r"Device Minor:\s*(\d+)", target_info).group(1)
        if f"GRES=gpu:1(IDX:{minor})" in alloc:
            break
    else:
        raise SystemExit("Slurm allocation is neither physical GPU 1 nor authorized fallback GPU 2")
    # GRES uses device minors here, while CUDA enumeration follows another order.
    # The UUID names only the device Slurm has explicitly allocated and verified.
    os.environ["CUDA_VISIBLE_DEVICES"] = uuid
    os.environ["PLACEMENT_GPU_UUID"] = uuid
    import torch
    props = torch.cuda.get_device_properties(0)
    if torch.cuda.device_count() != 1 or str(props.uuid).removeprefix("GPU-") != uuid[4:]:
        raise SystemExit(f"CUDA UUID disagrees with allocation: {props.uuid}")
    receipt = dict(job=job, gpu_index=int(index), gpu_uuid=uuid, device_minor=minor, gpu_name=props.name)
    path = Path(os.environ["STEREO_RUN_ROOT"]) / f"allocation-{job}.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)
    controller = Path(__file__).resolve().parents[1] / "run_on_gpu.py"
    python = str(Path(os.environ["STEREO_ENV"]) / "bin/python")
    os.execv(python, [python, str(controller), "--stage", "all"])


if __name__ == "__main__":
    main()
