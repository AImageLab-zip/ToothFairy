"""Multi-GPU data-parallel helpers — torchrun entry point, gradient averaging, rank utilities.

Run as a single process (i.e. without `torchrun`), every function here becomes a no-op and the
code behaves exactly like the original single-GPU path.

**Why not `nn.parallel.DistributedDataParallel`**
DDP all-reduces gradient buckets from autograd hooks inside backward. This model's training step,
however, calls backward **several times within one step** in order to bound memory
(`ReportModel.train_step_chunked`: once for the findings loss and once per region chunk). The DDP
reducer raises an error when a parameter becomes ready twice in one iteration, so it would need
every backward wrapped in `no_sync()` plus a manual synchronisation at the end — and even then the
final backward would have to touch **all** parameters (otherwise find_unused_parameters is
required), which is fragile against the chunk layout and against a region count that varies per
patient. Instead the gradients are all-reduced (averaged) directly, just before the optimizer step.
That is mathematically the same data-parallel average as DDP, and it is safe regardless of a
varying graph, multiple backwards, or a 4-bit quantized base (broadcasting bitsandbytes
`Params4bit`). The only cost is that communication does not overlap with backward, which is minor
here: the step is dominated by perception and LLM decode, while the traffic is just the few tens of
megabytes of trainable parameters.

Ranks are kept in agreement by (a) identical initialisation from the same seed, (b) broadcasting
the trainable parameters before training starts, and (c) averaging gradients at every step.
Decisions (best checkpoint, early stopping, stage transitions) are taken on rank 0 and broadcast as
small integer tensors so that the ranks cannot diverge.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class DistInfo:
    enabled: bool
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def setup() -> DistInfo:
    """Initialise the NCCL process group under torchrun; otherwise return single-process info."""
    ws = int(os.environ.get("WORLD_SIZE", "1"))
    if ws <= 1:
        # Name the index explicitly - comparing "cuda" (index=None) against "cuda:0" elsewhere
        # would report a spurious mismatch.
        dev = (torch.device(f"cuda:{torch.cuda.current_device()}") if torch.cuda.is_available()
               else torch.device("cpu"))
        return DistInfo(False, 0, 1, 0, dev)
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    dev = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    return DistInfo(True, rank, ws, local_rank, dev)


def cleanup(info: DistInfo) -> None:
    if info.enabled and dist.is_initialized():
        dist.destroy_process_group()


def barrier(info: DistInfo) -> None:
    if info.enabled:
        dist.barrier()


def broadcast_parameters(info: DistInfo, params) -> None:
    """Align every rank with rank 0's parameters (guards against divergent initialisation)."""
    if not info.enabled:
        return
    for p in params:
        dist.broadcast(p.data, src=0)


def average_gradients(info: DistInfo, params) -> None:
    """Average the gradients just before the optimizer step (same data-parallel average as DDP).

    A parameter whose grad is None on some rank is filled with zeros so that it still takes part in
    the all-reduce: the number of generated regions and the set of loss terms differ per rank, so
    whether a gradient exists can differ too, and a rank that skips the collective makes it
    mismatch and hang. The zeros contribute exactly nothing to the average (that rank's
    contribution is zero).
    """
    if not info.enabled:
        return
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad /= info.world_size


def reduce_scalar_dict(info: DistInfo, sums: dict[str, float], count: float
                       ) -> tuple[dict[str, float], float]:
    """Sum the per-rank (weighted sum, sample count) into the numerator and denominator of the
    overall mean.

    Splitting validation across ranks **without padding** (so duplicated samples cannot skew the
    mean) and summing here reproduces exactly the value of a single-rank evaluation. The key set
    can differ per rank, so the union is agreed on first.
    """
    if not info.enabled:
        return sums, count
    keys = [None] * info.world_size
    dist.all_gather_object(keys, sorted(sums.keys()))
    union = sorted({k for ks in keys for k in ks})
    dev = info.device
    vec = torch.tensor([sums.get(k, 0.0) for k in union] + [count],
                       dtype=torch.float64, device=dev)
    dist.all_reduce(vec, op=dist.ReduceOp.SUM)
    out = {k: float(vec[i]) for i, k in enumerate(union)}
    return out, float(vec[-1])


