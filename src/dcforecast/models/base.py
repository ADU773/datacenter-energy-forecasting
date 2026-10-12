"""Common forecaster interface (fit / predict / save / load) and shared helpers.

The forecasting task: at an origin ``t`` (the last observed hour) predict the target for
``t+1 ... t+H`` (H = 24) for every site, as P10 / P50 / P90 quantiles.

``predict(panel, origins)`` receives a panel that may contain *future* target values (this is
how backtests work: one frame, many origins). Every model must only use target values at or
before each origin; ``tests/test_models.py`` enforces this by scrambling post-origin targets
and checking that forecasts do not change.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from dcforecast.config import Config, _from_dict
from dcforecast.utils import to_jsonable

QUANTILE_COLUMNS = ["p10", "p50", "p90"]
FORECAST_COLUMNS = ["site_id", "origin", "timestamp", "horizon", *QUANTILE_COLUMNS]


class Forecaster(ABC):
    """Base class for every model."""

    name: ClassVar[str] = "base"
    #: True for models that can use a GPU (deep models).
    uses_device: ClassVar[bool] = False

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.horizon = cfg.forecast.horizon
        self.target = cfg.forecast.target
        self.sites_: list[str] = []
        self.fit_info_: dict[str, Any] = {}

    # ------------------------------------------------------------------ API
    @abstractmethod
    def fit(self, history: pd.DataFrame, device: str | None = None) -> Forecaster:
        """Fit on a clean history panel (all sites, hourly, target observed)."""

    @abstractmethod
    def predict(self, panel: pd.DataFrame, origins: Sequence[pd.Timestamp]) -> pd.DataFrame:
        """Forecast ``horizon`` hours after each origin for every fitted site.

        Returns a frame with columns :data:`FORECAST_COLUMNS`.
        """

    # ------------------------------------------------------------------ persistence
    def save(self, path: Path) -> Path:
        """Persist the model into directory ``path`` (``meta.json`` + model payload)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        meta = {
            "name": self.name,
            "horizon": self.horizon,
            "target": self.target,
            "sites": self.sites_,
            "fit_info": self.fit_info_,
            "config": self.cfg.to_dict(),
        }
        with (path / "meta.json").open("w", encoding="utf-8") as fh:
            json.dump(to_jsonable(meta), fh, indent=2)
        self._save_payload(path)
        return path

    @classmethod
    def load(cls, path: Path, device: str | None = None) -> Forecaster:
        """Load a model saved with :meth:`save`."""
        path = Path(path)
        with (path / "meta.json").open(encoding="utf-8") as fh:
            meta = json.load(fh)
        cfg = _from_dict(Config, meta["config"])
        model = cls(cfg)
        model.sites_ = list(meta.get("sites", []))
        model.fit_info_ = dict(meta.get("fit_info", {}))
        model._load_payload(path, device=device)
        return model

    @abstractmethod
    def _save_payload(self, path: Path) -> None: ...

    @abstractmethod
    def _load_payload(self, path: Path, device: str | None = None) -> None: ...


# ---------------------------------------------------------------------- helpers
def target_lookup(panel: pd.DataFrame, target: str) -> dict[str, pd.Series]:
    """Per-site target series indexed by timestamp."""
    return {
        str(site): grp.set_index("timestamp")[target].sort_index()
        for site, grp in panel.groupby("site_id", sort=False)
    }


def origin_grid(
    sites: Sequence[str], origins: Sequence[pd.Timestamp], horizon: int
) -> pd.DataFrame:
    """Cartesian product site x origin x horizon with target timestamps."""
    origins_idx = pd.DatetimeIndex(origins)
    hs = np.arange(1, horizon + 1)
    n_o, n_h = len(origins_idx), len(hs)
    rows = []
    for site in sites:
        origin_rep = np.repeat(origins_idx.values, n_h)
        h_rep = np.tile(hs, n_o)
        rows.append(
            pd.DataFrame(
                {
                    "site_id": site,
                    "origin": origin_rep,
                    "horizon": h_rep,
                    "timestamp": origin_rep + (h_rep * np.timedelta64(1, "h")),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def finalize_quantiles(frame: pd.DataFrame) -> pd.DataFrame:
    """Sort P10 <= P50 <= P90 row-wise (no crossing), clip at zero, order columns."""
    q = np.sort(frame[QUANTILE_COLUMNS].to_numpy(dtype=float), axis=1)
    frame = frame.copy()
    frame[QUANTILE_COLUMNS] = np.clip(q, 0.0, None)
    return frame[FORECAST_COLUMNS].reset_index(drop=True)


class ResidualQuantiles:
    """Empirical P10/P90 of relative residuals per site and horizon bucket.

    Turns a point forecaster into a probabilistic one: ``p_q = p50 * (1 + r_q)`` where ``r_q``
    is the q-quantile of ``(y - p50) / p50`` on calibration origins from the training period.
    """

    def __init__(self, bucket_hours: int = 6) -> None:
        self.bucket_hours = bucket_hours
        self.table: dict[str, dict[int, tuple[float, float]]] = {}

    def fit(self, forecasts: pd.DataFrame, actuals: pd.DataFrame, target: str) -> ResidualQuantiles:
        merged = forecasts.merge(
            actuals[["site_id", "timestamp", target]], on=["site_id", "timestamp"], how="inner"
        ).dropna(subset=[target])
        merged["rel"] = (merged[target] - merged["p50"]) / merged["p50"].where(merged["p50"] > 0)
        merged["bucket"] = (merged["horizon"] - 1) // self.bucket_hours
        for (site, bucket), grp in merged.groupby(["site_id", "bucket"]):
            rel = grp["rel"].dropna().to_numpy()
            if len(rel) < 5:
                continue
            self.table.setdefault(str(site), {})[int(bucket)] = (
                float(np.quantile(rel, 0.1)),
                float(np.quantile(rel, 0.9)),
            )
        return self

    def apply(self, frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()
        bucket = (frame["horizon"].to_numpy() - 1) // self.bucket_hours
        lo = np.full(len(frame), -0.05)
        hi = np.full(len(frame), 0.05)
        sites = frame["site_id"].to_numpy()
        for site, buckets in self.table.items():
            site_mask = sites == site
            for b, (q10, q90) in buckets.items():
                m = site_mask & (bucket == b)
                lo[m], hi[m] = q10, q90
        frame["p10"] = frame["p50"] * (1.0 + lo)
        frame["p90"] = frame["p50"] * (1.0 + hi)
        return frame

    def to_dict(self) -> dict[str, Any]:
        return {"bucket_hours": self.bucket_hours, "table": self.table}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResidualQuantiles:
        obj = cls(int(data.get("bucket_hours", 6)))
        obj.table = {
            str(s): {int(b): (float(v[0]), float(v[1])) for b, v in buckets.items()}
            for s, buckets in data.get("table", {}).items()
        }
        return obj


def daily_origins(start: pd.Timestamp, end: pd.Timestamp, hour: int = 23) -> list[pd.Timestamp]:
    """Daily forecast origins at ``hour`` UTC between ``start`` and ``end`` (inclusive)."""
    first = pd.Timestamp(start).normalize() + pd.Timedelta(hours=hour)
    if first < start:
        first += pd.Timedelta(days=1)
    return list(pd.date_range(first, end, freq="24h"))
