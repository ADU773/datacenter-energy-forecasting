"""VRAM-aware automatic batch-size finder.

The RTX 3050 Laptop GPU has 4 GB that is shared with the desktop and other processes, so the
finder plans against a *budget* -- the smaller of ``deep.vram_budget_mb`` and
``deep.vram_budget_fraction`` x currently free memory -- rather than against the card's total.
It probes power-of-two batch sizes with a real forward + backward pass (under AMP if enabled),
measures ``torch.cuda.max_memory_allocated`` and keeps the largest size whose peak, plus a 25 %
safety margin, fits. A CUDA OOM during probing simply ends the search.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from torch import nn

from dcforecast.config import DeepConfig
from dcforecast.models.deep.networks import pinball_loss
from dcforecast.utils import get_logger

log = get_logger(__name__)

SAFETY = 1.25
CPU_DEFAULT_BATCH = 256


@dataclass
class BatchSizeResult:
    """Outcome of the search (for logging / tracking)."""

    batch_size: int
    budget_mb: float | None
    probes: list[dict[str, float]] = field(default_factory=list)
    reason: str = ""


def vram_budget_mb(cfg: DeepConfig, device: torch.device) -> float:
    """Memory we allow ourselves to plan for on ``device``.

    Cached-but-unused blocks of this process are released first so they count as free.
    """
    torch.cuda.empty_cache()
    free, _total = torch.cuda.mem_get_info(device)
    return float(min(cfg.vram_budget_mb, cfg.vram_budget_fraction * free / 2**20))


def find_batch_size(
    model: nn.Module,
    make_batch: Callable[[int], tuple[torch.Tensor, ...]],
    device: torch.device,
    cfg: DeepConfig,
    amp: bool,
) -> BatchSizeResult:
    """Find the largest batch size that fits the VRAM budget (CUDA) or use a CPU default.

    Args:
        model: the network (already on ``device``); its weights are not modified.
        make_batch: returns ``(enc, dec, site, target, mask)`` CPU tensors for a batch size.
        device: target device.
        cfg: deep-learning config (budget, min/max batch size).
        amp: whether training will use autocast.
    """
    if device.type != "cuda":
        bs = min(cfg.max_batch_size, max(cfg.min_batch_size, CPU_DEFAULT_BATCH))
        return BatchSizeResult(bs, None, [], f"non-CUDA device: fixed {bs}")
    budget = vram_budget_mb(cfg, device)
    probes: list[dict[str, float]] = []
    best: int | None = None
    bs = cfg.min_batch_size
    was_training = model.training
    model.train()
    reason = f"reached max_batch_size={cfg.max_batch_size}"
    while bs <= cfg.max_batch_size:
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        try:
            enc, dec, site, tgt, msk = (t.to(device) for t in make_batch(bs))
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                pred = model(enc, dec, site)
            loss = pinball_loss(pred.float(), tgt, msk)
            loss.backward()
            torch.cuda.synchronize(device)
            peak_mb = (torch.cuda.max_memory_allocated(device) - base) / 2**20
        except torch.OutOfMemoryError:
            reason = f"CUDA OOM at batch {bs}"
            break
        finally:
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        probes.append({"batch_size": bs, "peak_mb": round(peak_mb, 1)})
        if peak_mb * SAFETY > budget:
            reason = f"batch {bs} needs {peak_mb:.0f} MB x {SAFETY} > budget {budget:.0f} MB"
            break
        best = bs
        bs *= 2
    model.train(was_training)
    chosen = best or cfg.min_batch_size
    log.info(
        "batch-size finder: budget %.0f MB, probes %s -> %d (%s)",
        budget,
        [(p["batch_size"], p["peak_mb"]) for p in probes],
        chosen,
        reason,
    )
    return BatchSizeResult(chosen, round(budget, 1), probes, reason)
