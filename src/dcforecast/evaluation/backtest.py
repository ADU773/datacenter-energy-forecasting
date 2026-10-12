"""Rolling-origin backtesting.

The last ``folds x test_days`` days of history are split into consecutive test windows. For
fold k every model is trained on all data before the window (expanding window) and issues a
24 h forecast every day at 23:00 UTC for the next calendar day, so each fold has ``test_days``
origins per site. Deep models train from scratch on fold 1; with ``backtest.deep_warm_start``
later folds fine-tune the previous fold's weights for a few epochs (a common production refit
pattern that keeps the backtest within the laptop time budget without ever peeking ahead).

Hours whose actual value was imputed during cleaning are excluded from scoring.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from dcforecast.config import Config
from dcforecast.data.prepare import PanelData
from dcforecast.evaluation.metrics import evaluate, naive_scale
from dcforecast.models import DEEP_MODELS, create_model
from dcforecast.models.base import Forecaster, daily_origins
from dcforecast.tracking.tracker import Tracker
from dcforecast.utils import get_logger, write_json

log = get_logger(__name__)


@dataclass
class Fold:
    """One rolling-origin split."""

    k: int
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    origins: list[pd.Timestamp]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold": self.k,
            "train_end": self.train_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
            "n_origins": len(self.origins),
        }


def make_folds(history_end: pd.Timestamp, folds: int, test_days: int, horizon: int) -> list[Fold]:
    """Consecutive test windows ending at ``history_end`` (oldest fold first)."""
    out = []
    for k in range(1, folds + 1):
        test_end = history_end - pd.Timedelta(days=test_days * (folds - k))
        test_start = test_end - pd.Timedelta(days=test_days) + pd.Timedelta(hours=1)
        train_end = test_start - pd.Timedelta(hours=1)
        origins = [
            o
            for o in daily_origins(train_end, test_end)
            if o + pd.Timedelta(hours=horizon) <= test_end
        ]
        out.append(Fold(k, train_end, test_start, test_end, origins))
    return out


@dataclass
class BacktestResult:
    """Forecasts and metric tables of a backtest."""

    forecasts: pd.DataFrame
    leaderboard: pd.DataFrame
    per_site: pd.DataFrame
    per_fold: pd.DataFrame
    per_horizon: pd.DataFrame
    folds: list[Fold]
    timings: dict[str, dict[str, Any]] = field(default_factory=dict)
    run_ids: dict[str, str] = field(default_factory=dict)

    def best_model(self, metric: str = "mase") -> str:
        return str(self.leaderboard.sort_values(metric).iloc[0]["model"])


def _score(frame: pd.DataFrame, target: str, by: list[str]) -> pd.DataFrame:
    rows = []
    for key, grp in frame.groupby(by, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(by, key, strict=True)), **evaluate(grp, target)})
    return pd.DataFrame(rows)


def score_forecasts(forecasts: pd.DataFrame, target: str) -> dict[str, pd.DataFrame]:
    """Leaderboards overall / per site / per fold / per horizon."""
    scored = forecasts[~forecasts["imputed"]].dropna(subset=[target])
    overall = _score(scored, target, ["model"]).sort_values("mase", ignore_index=True)
    overall.insert(0, "rank", range(1, len(overall) + 1))
    return {
        "leaderboard": overall,
        "per_site": _score(scored, target, ["model", "site_id"]),
        "per_fold": _score(scored, target, ["model", "fold"]),
        "per_horizon": _score(scored, target, ["model", "horizon"])[
            ["model", "horizon", "mae", "mape", "mase", "coverage_80"]
        ],
    }


def _fit(
    model: Forecaster,
    train: pd.DataFrame,
    device: str | None,
    warm_from: Forecaster | None,
    warm_epochs: int,
) -> None:
    if warm_from is not None:
        model.fit(train, device=device, init_from=warm_from, epochs=warm_epochs)  # type: ignore[call-arg]
    else:
        model.fit(train, device=device)


def run_backtest(
    cfg: Config,
    data: PanelData,
    models: list[str] | None = None,
    device: str | None = None,
    tracker: Tracker | None = None,
    out_dir: Path | None = None,
) -> BacktestResult:
    """Run the rolling-origin backtest for ``models`` and persist forecasts + leaderboards."""
    models = models or cfg.backtest.models
    target = cfg.forecast.target
    history = data.history
    folds = make_folds(
        data.history_end, cfg.backtest.folds, cfg.backtest.test_days, cfg.forecast.horizon
    )
    actual_cols = ["site_id", "timestamp", target, "facility_imputed"]
    actuals = history[actual_cols].rename(columns={"facility_imputed": "imputed"})
    all_fc: list[pd.DataFrame] = []
    timings: dict[str, dict[str, Any]] = {m: {"fit_s": 0.0, "predict_s": 0.0} for m in models}
    prev: dict[str, Forecaster] = {}
    fit_infos: dict[str, list[dict[str, Any]]] = {m: [] for m in models}
    for fold in folds:
        train = history[history["timestamp"] <= fold.train_end]
        scales = naive_scale(train, target)
        log.info(
            "fold %d/%d: train <= %s, test %s .. %s (%d origins)",
            fold.k,
            len(folds),
            fold.train_end,
            fold.test_start,
            fold.test_end,
            len(fold.origins),
        )
        for name in models:
            model = create_model(name, cfg)
            warm = (
                prev.get(name) if (name in DEEP_MODELS and cfg.backtest.deep_warm_start) else None
            )
            if name in DEEP_MODELS:
                model.checkpoint_path = (
                    cfg.runs_dir / "checkpoints" / f"backtest_{name}_fold{fold.k}.pt"
                )  # type: ignore[attr-defined]
            t0 = time.perf_counter()
            _fit(model, train, device, warm, cfg.backtest.warm_start_epochs)
            t1 = time.perf_counter()
            fc = model.predict(history, fold.origins)
            t2 = time.perf_counter()
            timings[name]["fit_s"] += t1 - t0
            timings[name]["predict_s"] += t2 - t1
            if model.uses_device:
                timings[name]["device"] = str(getattr(model, "device", "cpu"))
            fit_infos[name].append({k: v for k, v in model.fit_info_.items() if k != "history"})
            fc["model"] = name
            fc["fold"] = fold.k
            fc["scale"] = fc["site_id"].map(scales)
            all_fc.append(fc)
            log.info(
                "  %-8s fit %.1fs predict %.1fs (%d forecasts)", name, t1 - t0, t2 - t1, len(fc)
            )
            if name in DEEP_MODELS:
                prev[name] = model
        gc.collect()

    forecasts = pd.concat(all_fc, ignore_index=True).merge(
        actuals, on=["site_id", "timestamp"], how="left"
    )
    forecasts["imputed"] = forecasts["imputed"].fillna(False).astype(bool)
    tables = score_forecasts(forecasts, target)
    result = BacktestResult(folds=folds, forecasts=forecasts, timings=timings, **tables)

    # keep the last fold's deep models: `train` can warm-start from them
    for name, model in prev.items():
        model.save(cfg.runs_dir / "warmstart" / name)

    if tracker is not None:
        lb = result.leaderboard.set_index("model")
        for name in models:
            if name not in lb.index:
                continue
            with tracker.start_run("backtest", model=name) as run:
                run.log_params(
                    folds=cfg.backtest.folds,
                    test_days=cfg.backtest.test_days,
                    horizon=cfg.forecast.horizon,
                    warm_start=bool(name in DEEP_MODELS and cfg.backtest.deep_warm_start),
                )
                metrics = {
                    k: float(v) for k, v in lb.loc[name].items() if isinstance(v, (int, float))
                }
                metrics.update(
                    {f"{k}_s": round(v, 3) for k, v in timings[name].items() if k != "device"}
                )
                per_fold = result.per_fold[result.per_fold["model"] == name]
                for _, row in per_fold.iterrows():
                    metrics[f"fold{int(row['fold'])}_mase"] = float(row["mase"])
                run.log_metrics(metrics)
                run.set_device(timings[name].get("device", "cpu"))
                run.tags["fit_info"] = fit_infos[name]
                result.run_ids[name] = run.run_id
    if out_dir is not None:
        save_backtest(result, out_dir)
    return result


def save_backtest(result: BacktestResult, out_dir: Path) -> None:
    """Write forecasts (Parquet) and leaderboards (JSON + CSV)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    result.forecasts.to_parquet(out_dir / "forecasts.parquet", index=False, compression="zstd")
    result.leaderboard.to_csv(out_dir / "leaderboard.csv", index=False)
    write_json(
        out_dir / "leaderboard.json",
        {
            "folds": [f.to_dict() for f in result.folds],
            "leaderboard": result.leaderboard.to_dict(orient="records"),
            "per_site": result.per_site.to_dict(orient="records"),
            "per_fold": result.per_fold.to_dict(orient="records"),
            "per_horizon": result.per_horizon.to_dict(orient="records"),
            "timings": result.timings,
            "run_ids": result.run_ids,
            "best_model": result.best_model(),
        },
    )
