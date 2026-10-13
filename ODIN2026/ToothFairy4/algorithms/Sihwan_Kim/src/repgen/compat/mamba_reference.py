"""Install the packaged Mamba2 reference path on Turing/sm75 GPUs."""

from __future__ import annotations

import os
from typing import Any


_INSTALLED = False


def should_install() -> bool:
    if os.environ.get("ODIN_FORCE_MAMBA_REFERENCE") == "1":
        return True
    import torch

    return torch.cuda.is_available() and torch.cuda.get_device_capability() < (8, 0)


def install() -> dict[str, Any]:
    global _INSTALLED
    if _INSTALLED or not should_install():
        return {"installed": _INSTALLED, "reason": "already_or_not_required"}

    import mamba_ssm.modules.mamba2 as mamba2
    import mamba_ssm.modules.mamba2_simple as mamba2_simple
    import mamba_ssm.ops.triton.layernorm_gated as layernorm_gated
    import mamba_ssm.ops.triton.ssd_combined as ssd_combined
    import torch.nn.functional as F

    reference = ssd_combined.mamba_split_conv1d_scan_ref

    def torch_causal_conv1d(x, weight, bias=None, seq_idx=None, activation=None):
        if seq_idx is not None:
            raise RuntimeError("R237 torch causal-conv fallback does not support seq_idx")
        width = int(weight.shape[-1])
        output = F.conv1d(
            x.contiguous(),
            weight.contiguous().unsqueeze(1),
            bias,
            padding=width - 1,
            groups=int(x.shape[1]),
        )[..., : x.shape[-1]]
        if activation in {"silu", "swish"}:
            output = F.silu(output)
        elif activation is not None:
            raise ValueError(f"unsupported R237 causal-conv activation: {activation}")
        return output

    def t4_reference(
        zxbcdt,
        conv1d_weight,
        conv1d_bias,
        dt_bias,
        A,
        D,
        chunk_size,
        initial_states=None,
        seq_idx=None,
        dt_limit=(0.0, float("inf")),
        return_final_states=False,
        activation="silu",
        rmsnorm_weight=None,
        rmsnorm_eps=1e-6,
        outproj_weight=None,
        outproj_bias=None,
        headdim=None,
        ngroups=1,
        norm_before_gate=True,
    ):
        if initial_states is not None or seq_idx is not None or return_final_states:
            raise RuntimeError("R237 reference fallback supports only full-sequence inference")
        original_rmsnorm = ssd_combined.rmsnorm_fn
        original_conv1d = ssd_combined.causal_conv1d_fn
        current_selective_scan = ssd_combined.ssd_selective_scan

        def pure_torch_selective_scan(
            x,
            dt,
            scan_A,
            B,
            C,
            D=None,
            z=None,
            dt_bias=None,
            dt_softplus=False,
            dt_limit=(0.0, float("inf")),
        ):
            if dt_limit != (0.0, float("inf")):
                raise RuntimeError("R237 pure scan requires the default dt_limit")
            sequence_length = int(x.shape[1])
            pad = (-sequence_length) % int(chunk_size)
            if pad:
                x = F.pad(x, (0, 0, 0, 0, 0, pad))
                dt = F.pad(dt, (0, 0, 0, pad))
                B = F.pad(B, (0, 0, 0, 0, 0, pad))
                C = F.pad(C, (0, 0, 0, 0, 0, pad))
                if z is not None:
                    z = F.pad(z, (0, 0, 0, 0, 0, pad))
            output = ssd_combined.ssd_chunk_scan_combined_ref(
                x,
                dt,
                scan_A,
                B,
                C,
                chunk_size=chunk_size,
                D=D,
                z=z,
                dt_bias=None if dt_bias is None else dt_bias.float(),
                dt_softplus=dt_softplus,
            )
            return output[:, :sequence_length].to(x.dtype)

        ssd_combined.rmsnorm_fn = layernorm_gated.rms_norm_ref
        ssd_combined.causal_conv1d_fn = torch_causal_conv1d
        ssd_combined.ssd_selective_scan = pure_torch_selective_scan
        try:
            return reference(
                zxbcdt,
                conv1d_weight,
                conv1d_bias,
                dt_bias,
                A,
                D,
                chunk_size,
                dt_limit=dt_limit,
                activation=activation,
                rmsnorm_weight=rmsnorm_weight,
                rmsnorm_eps=rmsnorm_eps,
                outproj_weight=outproj_weight,
                outproj_bias=outproj_bias,
                headdim=headdim,
                ngroups=ngroups,
                norm_before_gate=norm_before_gate,
            )
        finally:
            ssd_combined.rmsnorm_fn = original_rmsnorm
            ssd_combined.causal_conv1d_fn = original_conv1d
            ssd_combined.ssd_selective_scan = current_selective_scan

    mamba2.mamba_split_conv1d_scan_combined = t4_reference
    mamba2_simple.mamba_split_conv1d_scan_combined = t4_reference
    _INSTALLED = True
    return {
        "installed": True,
        "path": "pure_torch_chunk_scan_plus_causal_conv_and_rms_norm",
    }
