"""torch-scatter compatible CSR segment reduction using native PyTorch."""

from __future__ import annotations

import torch


def segment_csr(src: torch.Tensor, indptr: torch.Tensor, reduce: str = "sum") -> torch.Tensor:
    return torch.segment_reduce(src, reduce=reduce, offsets=indptr, axis=0)
