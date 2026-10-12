"""Training loop with the GPU engineering checklist.

* automatic mixed precision on CUDA (``torch.autocast`` fp16 + ``GradScaler``),
* pinned host memory and non-blocking host-to-device copies,
* gradient clipping, AdamW, OneCycle (or cosine) learning-rate schedule,
* early stopping on validation pinball loss with best-model checkpointing,
* per-epoch timing / throughput and peak GPU memory for the benchmark and tracker.
"""

from __future__ import annotations

import copy
import math
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from dcforecast.config import DeepConfig
from dcforecast.models.deep.data import WindowDataset, passthrough_collate
from dcforecast.models.deep.device import amp_enabled, num_workers, pin_memory
from dcforecast.models.deep.networks import pinball_loss
from dcforecast.utils import get_logger

log = get_logger(__name__)


@dataclass
class TrainResult:
    """What happened during training."""

    epochs_run: int = 0
    best_epoch: int = 0
    best_val_loss: float = math.inf
    batch_size: int = 0
    device: str = "cpu"
    amp: bool = False
    history: list[dict[str, float]] = field(default_factory=list)
    train_seconds: float = 0.0
    samples_seen: int = 0
    peak_gpu_mem_mb: float | None = None
    stopped_early: bool = False

    @property
    def samples_per_s(self) -> float:
        return self.samples_seen / self.train_seconds if self.train_seconds > 0 else 0.0

    @property
    def mean_epoch_s(self) -> float:
        times = [h["epoch_s"] for h in self.history]
        return float(np.mean(times)) if times else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "epochs_run": self.epochs_run,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "batch_size": self.batch_size,
            "device": self.device,
            "amp": self.amp,
            "train_seconds": round(self.train_seconds, 3),
            "samples_per_s": round(self.samples_per_s, 1),
            "mean_epoch_s": round(self.mean_epoch_s, 3),
            "peak_gpu_mem_mb": self.peak_gpu_mem_mb,
            "stopped_early": self.stopped_early,
            "history": self.history,
        }


def make_loader(
    ds: WindowDataset,
    batch_size: int,
    device: torch.device,
    cfg: DeepConfig,
    shuffle: bool,
    num_samples: int | None = None,
    seed: int = 0,
) -> DataLoader:
    """DataLoader over pre-collated batches (pinned memory on CUDA)."""
    if shuffle:
        gen = torch.Generator().manual_seed(seed)
        n = min(len(ds), num_samples) if num_samples else len(ds)
        sampler: Any = RandomSampler(ds, replacement=False, num_samples=n, generator=gen)
    else:
        sampler = SequentialSampler(ds)
    workers = num_workers(cfg)
    return DataLoader(
        ds,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=passthrough_collate,
        num_workers=workers,
        pin_memory=pin_memory(device, cfg),
        persistent_workers=workers > 0,
        drop_last=False,
    )


@torch.no_grad()
def evaluate_loss(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool) -> float:
    """Mask-weighted mean pinball loss over a loader."""
    model.eval()
    total, weight = 0.0, 0.0
    for enc, dec, site, tgt, msk, _scale in loader:
        enc, dec, site = (t.to(device, non_blocking=True) for t in (enc, dec, site))
        tgt, msk = tgt.to(device, non_blocking=True), msk.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            pred = model(enc, dec, site)
        w = float(msk.sum())
        if w > 0:
            total += float(pinball_loss(pred.float(), tgt, msk)) * w
            weight += w
    return total / weight if weight else math.inf


def train_model(
    model: nn.Module,
    train_ds: WindowDataset,
    val_ds: WindowDataset | None,
    cfg: DeepConfig,
    device: torch.device,
    batch_size: int,
    epochs: int,
    seed: int = 0,
    checkpoint_path: Path | None = None,
    samples_per_epoch: int | None = None,
    patience: int | None = None,
    amp: bool | None = None,
    log_prefix: str = "",
) -> TrainResult:
    """Train ``model`` in place and restore the best validation checkpoint."""
    use_amp = amp_enabled(device, cfg) if amp is None else bool(amp and device.type == "cuda")
    patience = cfg.patience if patience is None else patience
    n_per_epoch = min(len(train_ds), samples_per_epoch or cfg.samples_per_epoch)
    steps_per_epoch = max(1, math.ceil(n_per_epoch / batch_size))
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = max(1, epochs * steps_per_epoch)
    # OneCycle needs a few steps per phase (warm-up / anneal); tiny runs fall back to cosine.
    if cfg.scheduler == "onecycle" and total_steps >= 8:
        sched: Any = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=cfg.lr, total_steps=total_steps, pct_start=0.25
        )
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    val_loader = make_loader(val_ds, batch_size * 2, device, cfg, shuffle=False) if val_ds else None
    result = TrainResult(batch_size=batch_size, device=str(device), amp=use_amp)
    best_state = copy.deepcopy(model.state_dict())
    bad_epochs = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, epochs + 1):
        loader = make_loader(
            train_ds, batch_size, device, cfg, True, n_per_epoch, seed=seed * 1000 + epoch
        )
        model.train()
        t0 = time.perf_counter()
        running = torch.zeros((), device=device)  # accumulate on-device: no per-step sync
        seen = 0
        for enc, dec, site, tgt, msk, _scale in loader:
            enc = enc.to(device, non_blocking=True)
            dec = dec.to(device, non_blocking=True)
            site = site.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            msk = msk.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                pred = model(enc, dec, site)
            loss = pinball_loss(pred.float(), tgt, msk)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            with warnings.catch_warnings():
                # GradScaler may skip a step on overflow; the scheduler warning is harmless.
                warnings.simplefilter("ignore", UserWarning)
                sched.step()
            running += loss.detach() * len(enc)
            seen += len(enc)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_s = time.perf_counter() - t0
        result.train_seconds += epoch_s
        result.samples_seen += seen
        train_loss = float(running) / max(seen, 1)
        val_loss = evaluate_loss(model, val_loader, device, use_amp) if val_loader else train_loss
        result.history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
                "epoch_s": round(epoch_s, 3),
                "samples_per_s": round(seen / epoch_s, 1) if epoch_s > 0 else 0.0,
                "lr": float(opt.param_groups[0]["lr"]),
            }
        )
        result.epochs_run = epoch
        improved = val_loss < result.best_val_loss - 1e-6
        if improved:
            result.best_val_loss = val_loss
            result.best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, checkpoint_path)
            bad_epochs = 0
        else:
            bad_epochs += 1
        log.info(
            "%sepoch %2d/%d train %.4f val %.4f %s(%.1fs, %.0f samples/s)",
            log_prefix,
            epoch,
            epochs,
            train_loss,
            val_loss,
            "* " if improved else "",
            epoch_s,
            seen / epoch_s if epoch_s > 0 else 0.0,
        )
        if bad_epochs >= patience:
            result.stopped_early = True
            log.info("%searly stopping after %d epochs without improvement", log_prefix, patience)
            break
    model.load_state_dict(best_state)
    if device.type == "cuda":
        result.peak_gpu_mem_mb = round(torch.cuda.max_memory_allocated(device) / 2**20, 1)
    return result
