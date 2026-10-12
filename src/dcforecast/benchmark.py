"""``dcforecast benchmark``: CPU vs GPU (fp32 / AMP) training and inference of the deep models.

For each model and mode the benchmark trains a fresh network for a few epochs on the real
windowed dataset with the *same* batch size, then measures inference latency. Reported:

* training throughput (samples/s, steady state = last epoch) and the implied time of one full
  ``deep.samples_per_epoch`` epoch,
* inference latency per batch (p50 / p95 ms) at ``benchmark.inference_batch_size`` and for a
  single window (batch 1, the API case),
* peak GPU memory (``torch.cuda.max_memory_allocated``),
* GPU utilisation sampled with ``nvidia-smi`` during training (device-wide), and process CPU %
  and peak RSS sampled with ``psutil``.

CPU runs use ``benchmark.cpu_threads`` intra-op threads and fewer samples (they are slow).
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any

import numpy as np
import torch

from dcforecast.config import Config
from dcforecast.data.prepare import PanelData
from dcforecast.hardware import ResourceSampler, collect_hardware
from dcforecast.models import create_model
from dcforecast.models.deep.data import NormStats
from dcforecast.models.deep.device import num_workers
from dcforecast.models.deep.forecaster import DeepForecaster
from dcforecast.models.deep.networks import count_parameters
from dcforecast.models.deep.trainer import train_model
from dcforecast.tracking.tracker import Tracker
from dcforecast.utils import get_logger, seed_everything, write_json

MODE_ORDER = ["cpu", "cuda_fp32", "cuda_amp"]

log = get_logger(__name__)


def _latency_ms(
    net: torch.nn.Module,
    batch: tuple[torch.Tensor, ...],
    device: torch.device,
    amp: bool,
    repeats: int,
) -> dict[str, float]:
    enc, dec, site = (t.to(device) for t in batch[:3])
    net.eval()
    times = []
    with torch.no_grad():
        for i in range(repeats + 5):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                net(enc, dec, site)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            if i >= 5:  # warm-up iterations excluded
                times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(times)
    return {
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "mean_ms": round(float(arr.mean()), 3),
    }


def benchmark_mode(cfg: Config, model: DeepForecaster, train_ds: Any, mode: str) -> dict[str, Any]:
    """Train + infer one (model, mode) combination and return the measurements."""
    b = cfg.benchmark
    on_gpu = mode.startswith("cuda")
    device = torch.device("cuda" if on_gpu else "cpu")
    amp = mode == "cuda_amp"
    batch_size = b.batch_size if on_gpu else b.cpu_batch_size
    n_samples = b.gpu_train_samples if on_gpu else b.cpu_train_samples
    n_samples = min(n_samples, len(train_ds))
    prev_threads = torch.get_num_threads()
    if not on_gpu:
        torch.set_num_threads(max(1, b.cpu_threads))
    seed_everything(cfg.seed, deterministic=False)
    net = model.build_net(train_ds.enc_dim, train_ds.dec_dim)
    if on_gpu:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    with ResourceSampler(b.sample_interval_s, gpu=on_gpu) as sampler:
        result = train_model(
            net,
            train_ds,
            None,
            cfg.deep,
            device,
            batch_size,
            b.epochs,
            seed=cfg.seed,
            samples_per_epoch=n_samples,
            patience=b.epochs + 1,
            amp=amp,
        )
    steady = result.history[-1]["samples_per_s"]
    rng = np.random.default_rng(0)
    idx = rng.choice(len(train_ds), size=min(b.inference_batch_size, len(train_ds)), replace=False)
    batch = train_ds.__getitems__(idx.tolist())
    single = train_ds.__getitems__(idx[:1].tolist())
    if on_gpu:
        torch.cuda.reset_peak_memory_stats(device)
    lat = _latency_ms(net, batch, device, amp, b.inference_repeats)
    lat1 = _latency_ms(net, single, device, amp, b.inference_repeats)
    infer_peak = round(torch.cuda.max_memory_allocated(device) / 2**20, 1) if on_gpu else None
    if not on_gpu:
        torch.set_num_threads(prev_threads)
    del net
    if on_gpu:
        torch.cuda.empty_cache()
    res = {
        "model": model.name,
        "mode": mode,
        "device": str(device),
        "amp": amp,
        "batch_size": batch_size,
        "train_samples": n_samples * b.epochs,
        "epochs": b.epochs,
        "train_samples_per_s": round(float(steady), 1),
        "train_samples_per_s_incl_warmup": round(result.samples_per_s, 1),
        "epoch_s_measured": round(result.history[-1]["epoch_s"], 3),
        "epoch_s_full": round(cfg.deep.samples_per_epoch / steady, 2) if steady else None,
        "train_peak_gpu_mem_mb": result.peak_gpu_mem_mb,
        "infer_batch_size": len(idx),
        "infer_latency": lat,
        "infer_latency_batch1": lat1,
        "infer_samples_per_s": round(len(idx) / (lat["p50_ms"] / 1000.0), 1),
        "infer_peak_gpu_mem_mb": infer_peak,
        "cpu_threads": b.cpu_threads if not on_gpu else None,
        **{f"sampled_{k}": v for k, v in sampler.summary().items()},
    }
    log.info(
        "benchmark %-4s %-9s train %8.0f samples/s | infer p50 %.2f ms / batch %d | GPU util %s%%",
        model.name,
        mode,
        res["train_samples_per_s"],
        lat["p50_ms"],
        len(idx),
        res["sampled_gpu_util_percent_mean"],
    )
    return res


def _speedups(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for name in sorted({r["model"] for r in results}):
        by_mode = {r["mode"]: r for r in results if r["model"] == name}
        cpu = by_mode.get("cpu")
        row: dict[str, Any] = {"model": name}
        for mode in ("cuda_fp32", "cuda_amp"):
            if cpu and mode in by_mode:
                row[f"train_speedup_{mode}_vs_cpu"] = round(
                    by_mode[mode]["train_samples_per_s"] / cpu["train_samples_per_s"], 2
                )
                row[f"infer_speedup_{mode}_vs_cpu"] = round(
                    cpu["infer_latency"]["p50_ms"] / by_mode[mode]["infer_latency"]["p50_ms"], 2
                )
        if "cuda_fp32" in by_mode and "cuda_amp" in by_mode:
            row["train_speedup_amp_vs_fp32"] = round(
                by_mode["cuda_amp"]["train_samples_per_s"]
                / by_mode["cuda_fp32"]["train_samples_per_s"],
                2,
            )
            row["train_mem_ratio_amp_vs_fp32"] = round(
                (by_mode["cuda_amp"]["train_peak_gpu_mem_mb"] or 0)
                / max(by_mode["cuda_fp32"]["train_peak_gpu_mem_mb"] or 1, 1e-9),
                2,
            )
        out.append(row)
    return out


def _benchmark_with_retry(
    cfg: Config, model: DeepForecaster, train_ds: Any, mode: str
) -> dict[str, Any]:
    """Run one mode; on CUDA OOM (the GPU is shared) halve the batch size and try again."""
    bs = cfg.benchmark.batch_size
    while True:
        bench_cfg = dataclasses.replace(cfg.benchmark, batch_size=bs)
        try:
            return benchmark_mode(
                dataclasses.replace(cfg, benchmark=bench_cfg), model, train_ds, mode
            )
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if not mode.startswith("cuda") or bs <= 64:
                raise
            log.warning(
                "CUDA OOM at batch %d (%s/%s) -- retrying with %d", bs, model.name, mode, bs // 2
            )
            bs //= 2


def run_benchmark(
    cfg: Config,
    data: PanelData,
    models: list[str] | None = None,
    modes: list[str] | None = None,
    tracker: Tracker | None = None,
) -> dict[str, Any]:
    """Run the benchmark matrix and write ``reports/benchmark.json``."""
    models = models or cfg.benchmark.models
    modes = modes or cfg.benchmark.modes
    hw = collect_hardware()
    results, skipped = [], []
    prepared: dict[str, tuple[DeepForecaster, Any]] = {}
    for name in models:
        model = create_model(name, cfg)
        if not isinstance(model, DeepForecaster):
            raise ValueError(f"benchmark supports deep models only, got '{name}'")
        model.stats = NormStats.fit(data.history)
        prepared[name] = (model, model.train_val_datasets(data.history)[0])
    # CPU modes first: run standalone, their CPU%/RSS are measured before CUDA is initialised
    order = sorted(modes, key=lambda m: (m.startswith("cuda"), modes.index(m)))
    for mode in order:
        for name in models:
            model, train_ds = prepared[name]
            if mode.startswith("cuda") and not torch.cuda.is_available():
                skipped.append({"model": name, "mode": mode, "reason": "CUDA not available"})
                log.warning("skipping %s/%s: CUDA not available", name, mode)
                continue
            if tracker is not None:
                with tracker.start_run("benchmark", model=name, params={"mode": mode}) as run:
                    res = _benchmark_with_retry(cfg, model, train_ds, mode)
                    run.set_device(res["device"])
                    run.log_metrics(
                        {
                            k: v
                            for k, v in res.items()
                            if isinstance(v, (int, float)) and not isinstance(v, bool)
                        }
                    )
                    run.log_metrics(infer_p50_ms=res["infer_latency"]["p50_ms"])
            else:
                res = _benchmark_with_retry(cfg, model, train_ds, mode)
            res["params"] = count_parameters(model.build_net(train_ds.enc_dim, train_ds.dec_dim))
            results.append(res)
    rank = {m: i for i, m in enumerate(MODE_ORDER)}
    results.sort(key=lambda r: (models.index(r["model"]), rank.get(r["mode"], 99)))
    payload = {
        "hardware": hw.to_dict(),
        "config": {
            "batch_size": cfg.benchmark.batch_size,
            "cpu_batch_size": cfg.benchmark.cpu_batch_size,
            "epochs": cfg.benchmark.epochs,
            "gpu_train_samples_per_epoch": cfg.benchmark.gpu_train_samples,
            "cpu_train_samples_per_epoch": cfg.benchmark.cpu_train_samples,
            "cpu_threads": cfg.benchmark.cpu_threads,
            "lookback": cfg.deep.lookback,
            "horizon": cfg.forecast.horizon,
            "dataloader_workers": num_workers(cfg.deep),
        },
        "results": results,
        "speedups": _speedups(results),
        "skipped": skipped,
    }
    write_json(cfg.reports_dir / "benchmark.json", payload)
    return payload
