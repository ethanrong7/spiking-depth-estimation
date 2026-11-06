#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))

from core.corr import CorrBlock1D, PytorchAlternateCorrBlock1D


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("Run in an authorized Slurm GPU allocation")
    if torch.cuda.device_count() != 1 or "5090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Expected one RTX 5090")

    torch.manual_seed(23)
    device = torch.device("cuda")
    fmap1_regular = torch.randn(1, 32, 4, 64, device=device, requires_grad=True)
    fmap2_regular = torch.randn(1, 32, 4, 64, device=device, requires_grad=True)
    fmap1_alternate = fmap1_regular.detach().clone().requires_grad_(True)
    fmap2_alternate = fmap2_regular.detach().clone().requires_grad_(True)
    x = torch.linspace(12.0, 51.0, 64, device=device).view(1, 1, 1, 64)
    x = x.expand(1, 1, 4, 64)
    y = torch.arange(4.0, device=device).view(1, 1, 4, 1).expand(1, 1, 4, 64)
    coords = torch.cat([x, y], dim=1)

    regular = CorrBlock1D(
        fmap1_regular, fmap2_regular, num_levels=4, radius=4
    )(coords)
    alternate = PytorchAlternateCorrBlock1D(
        fmap1_alternate, fmap2_alternate, num_levels=4, radius=4
    )(coords)
    output_difference = (regular - alternate).abs()
    regular.square().mean().backward()
    alternate.square().mean().backward()
    fmap1_difference = (fmap1_regular.grad - fmap1_alternate.grad).abs()
    fmap2_difference = (fmap2_regular.grad - fmap2_alternate.grad).abs()

    def stats(value):
        return {"max_abs": float(value.max()), "mean_abs": float(value.mean())}

    report = {
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "dtype": str(regular.dtype),
        "autocast_enabled": False,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "shape": {"features": [1, 32, 4, 64], "correlation": list(regular.shape)},
        "output_difference": stats(output_difference),
        "fmap1_gradient_difference": stats(fmap1_difference),
        "fmap2_gradient_difference": stats(fmap2_difference),
        "allclose": {
            "output": bool(torch.allclose(regular, alternate, atol=2e-6, rtol=2e-5)),
            "fmap1_gradient": bool(
                torch.allclose(
                    fmap1_regular.grad,
                    fmap1_alternate.grad,
                    atol=2e-6,
                    rtol=2e-5,
                )
            ),
            "fmap2_gradient": bool(
                torch.allclose(
                    fmap2_regular.grad,
                    fmap2_alternate.grad,
                    atol=2e-6,
                    rtol=2e-5,
                )
            ),
        },
    }
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
