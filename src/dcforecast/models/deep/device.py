"""Device selection (CUDA -> MPS -> CPU) and DataLoader settings."""

from __future__ import annotations

import os

import torch

from dcforecast.config import DeepConfig
from dcforecast.utils import get_logger

log = get_logger(__name__)


def mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def resolve_device(preference: str | None = "auto") -> torch.device:
    """Pick a torch device.

    ``auto`` prefers CUDA, then Apple MPS, then CPU. Asking explicitly for an accelerator that is
    not present falls back to the CPU with a warning instead of crashing a long pipeline.
    """
    pref = (preference or "auto").lower()
    if pref in {"cuda_fp32", "cuda_amp"}:
        pref = "cuda"
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if mps_available():
            return torch.device("mps")
        return torch.device("cpu")
    if pref.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(pref)
        log.warning("CUDA requested but not available -- falling back to CPU")
        return torch.device("cpu")
    if pref == "mps":
        if mps_available():
            return torch.device("mps")
        log.warning("MPS requested but not available -- falling back to CPU")
        return torch.device("cpu")
    if pref == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unknown device '{preference}' (use auto, cuda, mps or cpu)")


def amp_enabled(device: torch.device, cfg: DeepConfig) -> bool:
    """Automatic mixed precision is used on CUDA only (fp16 autocast + GradScaler)."""
    return bool(cfg.amp and device.type == "cuda")


def num_workers(cfg: DeepConfig) -> int:
    """DataLoader workers.

    Windows uses *spawn* for worker processes; each worker re-imports torch (hundreds of MB of
    RAM) and start-up dominates for our cheap, vectorised batch fetches, so ``auto`` means 0
    workers on Windows and 2 elsewhere.
    """
    if isinstance(cfg.num_workers, int):
        return max(0, cfg.num_workers)
    return 0 if os.name == "nt" else 2


def pin_memory(device: torch.device, cfg: DeepConfig) -> bool:
    """Pinned (page-locked) host memory speeds up async host-to-device copies on CUDA."""
    return bool(cfg.pin_memory and device.type == "cuda")
