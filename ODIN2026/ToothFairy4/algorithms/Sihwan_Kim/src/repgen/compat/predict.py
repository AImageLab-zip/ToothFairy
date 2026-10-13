"""Run bundled nnU-Net with an sm75-safe Mamba2 reference fallback."""

import json
import os

import torch

from .mamba_reference import install


print(json.dumps({"event": "r237_mamba_fallback", **install()}, sort_keys=True), flush=True)

memory_fraction = os.environ.get("ODIN_CUDA_MEMORY_FRACTION")
if memory_fraction:
    torch.cuda.set_per_process_memory_fraction(float(memory_fraction))
    print(
        json.dumps(
            {
                "event": "r237_cuda_memory_fraction",
                "fraction": float(memory_fraction),
                "device_total_bytes": int(torch.cuda.get_device_properties(0).total_memory),
            },
            sort_keys=True,
        ),
        flush=True,
    )

from nnunetv2.inference.predict_from_raw_data import (  # noqa: E402
    nnUNetPredictor,
    predict_entry_point,
)


use_cpu_aggregation = (
    os.environ.get("ODIN_T4_CPU_AGGREGATION") == "1"
    or torch.cuda.get_device_capability() < (8, 0)
)
if use_cpu_aggregation:
    original_predictor_init = nnUNetPredictor.__init__

    def t4_predictor_init(self, *args, **kwargs):
        kwargs["perform_everything_on_device"] = False
        return original_predictor_init(self, *args, **kwargs)

    nnUNetPredictor.__init__ = t4_predictor_init
    print(
        json.dumps(
            {"event": "r237_cpu_aggregation", "enabled": True},
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    predict_entry_point()
    print(
        json.dumps(
            {
                "event": "r237_cuda_peak",
                "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            },
            sort_keys=True,
        ),
        flush=True,
    )
