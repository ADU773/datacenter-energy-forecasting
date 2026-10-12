"""Pipeline steps that combine several modules: train + register, promote, run-all."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from dcforecast.config import Config
from dcforecast.data.generate import run_generate
from dcforecast.data.prepare import PanelData, load_panel, prepare
from dcforecast.models import DEEP_MODELS, create_model, load_model
from dcforecast.tracking.registry import PRODUCTION, ModelNotFoundError, ModelRegistry
from dcforecast.tracking.tracker import Tracker
from dcforecast.utils import (
    dir_size_mb,
    get_logger,
    log_duration,
    peak_rss_mb,
    read_json,
    write_json,
)

log = get_logger(__name__)


def backtest_dir(cfg: Config) -> Path:
    return cfg.reports_dir / "backtest"


def train_models(
    cfg: Config,
    data: PanelData,
    models: list[str] | None = None,
    device: str | None = None,
    tracker: Tracker | None = None,
    registry: ModelRegistry | None = None,
) -> dict[str, dict[str, Any]]:
    """Train final models on the full history and register a new version of each."""
    models = models or cfg.train.models
    tracker = tracker or Tracker(cfg.runs_dir, cfg.profile)
    registry = registry or ModelRegistry(cfg.runs_dir)
    out: dict[str, dict[str, Any]] = {}
    for name in models:
        model = create_model(name, cfg)
        warm = None
        warm_dir = cfg.runs_dir / "warmstart" / name
        if name in DEEP_MODELS and cfg.train.warm_start and (warm_dir / "meta.json").exists():
            warm = load_model(warm_dir, device=device or cfg.deep.device)
        with tracker.start_run(
            "train", model=name, params={"warm_start": warm is not None, "rows": len(data.history)}
        ) as run:
            if name in DEEP_MODELS:
                model.checkpoint_path = cfg.runs_dir / "checkpoints" / f"final_{name}.pt"  # type: ignore[attr-defined]
            t0 = time.perf_counter()
            if warm is not None:
                log.info("%s: warm-starting from the last backtest fold", name)
                model.fit(  # type: ignore[call-arg]
                    data.history,
                    device=device,
                    init_from=warm,
                    epochs=cfg.backtest.warm_start_epochs,
                )
            else:
                model.fit(data.history, device=device)
            fit_s = time.perf_counter() - t0
            info = dict(model.fit_info_)  # includes the per-epoch history (small)
            metrics: dict[str, Any] = {"fit_s": round(fit_s, 3)}
            for key in (
                "best_val_loss",
                "epochs_run",
                "batch_size",
                "samples_per_s",
                "mean_epoch_s",
                "peak_gpu_mem_mb",
                "params",
                "rows",
            ):
                if isinstance(info.get(key), (int, float)):
                    metrics[key] = info[key]
            run.log_metrics(metrics)
            run.set_device(str(getattr(model, "device", "cpu")))
            run.tags["fit_info"] = info
            version = registry.register(model, run.run_id, metrics)
            run.log_artifact("model", registry.get(name, version)["abs_path"])
            out[name] = {"version": version, "run_id": run.run_id, **metrics}
            log.info("trained %s in %.1fs -> registry v%d", name, fit_s, version)
    return out


def promote(
    cfg: Config,
    model: str | None = None,
    version: int | None = None,
    metric: str | None = None,
    registry: ModelRegistry | None = None,
) -> dict[str, Any]:
    """Point the ``production`` alias at a model.

    Without ``model`` the best registered model of the latest backtest (by
    ``backtest.selection_metric``, MASE by default) is promoted.
    """
    registry = registry or ModelRegistry(cfg.runs_dir)
    metric = metric or cfg.backtest.selection_metric
    reason = "manual promotion"
    if model is None:
        path = backtest_dir(cfg) / "leaderboard.json"
        if not path.exists():
            raise FileNotFoundError("No backtest leaderboard -- run `dcforecast backtest` first")
        board = sorted(read_json(path)["leaderboard"], key=lambda r: r[metric])
        for row in board:
            try:
                registry.latest_version(row["model"])
            except ModelNotFoundError:
                continue
            model = str(row["model"])
            reason = f"best backtest {metric} = {row[metric]:.4f}"
            break
        if model is None:
            raise ModelNotFoundError("None of the backtested models is registered -- run train")
    entry = registry.set_alias(PRODUCTION, model, version, reason)
    write_json(
        cfg.reports_dir / "production_model.json",
        {"alias": PRODUCTION, "name": entry["name"], "version": entry["version"], "reason": reason},
    )
    return {**entry, "reason": reason}


def run_all(cfg: Config, device: str | None = None, skip_benchmark: bool = False) -> dict[str, Any]:
    """generate -> prepare -> backtest -> train -> promote -> benchmark -> decide -> report."""
    from dcforecast.benchmark import run_benchmark
    from dcforecast.decision.run import load_holdout, run_decision, save_decision
    from dcforecast.evaluation.backtest import run_backtest
    from dcforecast.report.build import build_report

    timings: dict[str, float] = {}
    peaks: dict[str, float | None] = {}
    tracker = Tracker(cfg.runs_dir, cfg.profile)
    registry = ModelRegistry(cfg.runs_dir)
    t_all = time.perf_counter()
    with log_duration(log, "generate") as t:
        run_generate(cfg)
    timings["generate"] = t.elapsed
    peaks["generate"] = peak_rss_mb()
    with log_duration(log, "prepare") as t:
        prepare(cfg)
    timings["prepare"] = t.elapsed
    peaks["prepare"] = peak_rss_mb()
    data = load_panel(cfg)
    with log_duration(log, "backtest") as t:
        bt = run_backtest(cfg, data, device=device, tracker=tracker, out_dir=backtest_dir(cfg))
    timings["backtest"] = t.elapsed
    peaks["backtest"] = peak_rss_mb()
    with log_duration(log, "train") as t:
        train_models(cfg, data, device=device, tracker=tracker, registry=registry)
    timings["train"] = t.elapsed
    peaks["train"] = peak_rss_mb()
    prod = promote(cfg, registry=registry)
    if not skip_benchmark:
        with log_duration(log, "benchmark") as t:
            run_benchmark(cfg, data, tracker=tracker)
        timings["benchmark"] = t.elapsed
        peaks["benchmark"] = peak_rss_mb()
    with log_duration(log, "decision") as t:
        model, _ = registry.load(PRODUCTION, device=cfg.serve.device)
        result = run_decision(cfg, data, model, holdout=load_holdout(cfg))
        save_decision(result, cfg.reports_dir)
    timings["decision"] = t.elapsed
    peaks["decision"] = peak_rss_mb()
    timings["total_before_report"] = time.perf_counter() - t_all
    write_json(
        cfg.reports_dir / "pipeline_timings.json", {k: round(v, 2) for k, v in timings.items()}
    )
    write_json(cfg.reports_dir / "resources.json", resource_snapshot(cfg, peaks))
    with log_duration(log, "report") as t:
        build_report(cfg)
    timings["report"] = t.elapsed
    timings["total"] = time.perf_counter() - t_all
    final = {k: round(v, 2) for k, v in timings.items()}
    peaks["report"] = peak_rss_mb()
    resources = resource_snapshot(cfg, peaks)
    write_json(cfg.reports_dir / "pipeline_timings.json", final)
    write_json(cfg.reports_dir / "resources.json", resources)
    # the published metrics were written mid-step; patch in the final wall-clock and footprint
    metrics_path = cfg.docs_dir / "metrics.json"
    if metrics_path.exists():
        published = read_json(metrics_path)
        published["pipeline_timings_s"] = final
        published["resources"] = resources
        write_json(metrics_path, published)
    return {"best_model": bt.best_model(), "production": prod, "timings": timings}


def resource_snapshot(cfg: Config, peaks: dict[str, float | None] | None = None) -> dict[str, Any]:
    """Peak process memory (overall and after each step), GPU peaks and disk footprint."""
    out: dict[str, Any] = {"peak_rss_mb": peak_rss_mb(), "peak_rss_mb_after_step": peaks or {}}
    try:
        runs = Tracker(cfg.runs_dir).list_runs(kind="train", limit=50)
        gpu = [r.get("peak_gpu_mem_mb") for r in runs["metrics"]] if not runs.empty else []
        gpu = [g for g in gpu if isinstance(g, (int, float))]
        if gpu:
            out["gpu_peak_allocated_train_mb"] = max(gpu)
    except Exception as exc:  # pragma: no cover - the tracker is optional here
        log.debug("no GPU stats: %s", exc)
    out["disk_mb"] = {
        "data": dir_size_mb(cfg.data_dir),
        "runs": dir_size_mb(cfg.runs_dir),
        "reports": dir_size_mb(cfg.reports_dir),
        "docs": dir_size_mb(cfg.docs_dir),
    }
    return out
