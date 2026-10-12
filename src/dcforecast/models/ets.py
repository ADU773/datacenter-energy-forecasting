"""Holt-Winters exponential smoothing (statsmodels), one model per site.

To stay fast on ~3 years of hourly data the model works on a *trailing window*
(``ets.window_days``, default 6 weeks): smoothing parameters are optimised once per site on the
last window of the training history; at prediction time the model is re-filtered with those
fixed parameters on the window that ends at each origin (about 30 ms per origin), which is the
classic "refit state, keep parameters" production pattern. Seasonality defaults to 168 h so
that both the daily and the weekend cycle are captured. Damped additive trend.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from dcforecast.models.base import (
    Forecaster,
    ResidualQuantiles,
    daily_origins,
    finalize_quantiles,
    origin_grid,
    target_lookup,
)
from dcforecast.utils import get_logger

log = get_logger(__name__)

PARAM_KEYS = ("smoothing_level", "smoothing_trend", "smoothing_seasonal", "damping_trend")


def _hw(y: np.ndarray, period: int):  # type: ignore[no-untyped-def]
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    return ExponentialSmoothing(
        y,
        trend="add",
        damped_trend=True,
        seasonal="add",
        seasonal_periods=period,
        initialization_method="heuristic",
    )


class ETSForecaster(Forecaster):
    """Damped Holt-Winters with weekly seasonality on a trailing window."""

    name = "ets"

    def __init__(self, cfg) -> None:  # type: ignore[no-untyped-def]
        super().__init__(cfg)
        self.window = cfg.ets.window_days * 24
        self.period = cfg.ets.seasonal_periods
        self.params: dict[str, dict[str, float]] = {}
        self.residuals = ResidualQuantiles()

    def _forecast_one(self, y: np.ndarray, params: dict[str, float]) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = _hw(y, self.period).fit(optimized=False, **params)
            return np.asarray(res.forecast(self.horizon), dtype=float)

    def _point(self, panel: pd.DataFrame, origins: Sequence[pd.Timestamp]) -> pd.DataFrame:
        grid = origin_grid(self.sites_, origins, self.horizon)
        lookup = target_lookup(panel, self.target)
        p50 = np.full(len(grid), np.nan)
        for site in self.sites_:
            series = lookup[site]
            values = series.to_numpy(dtype=float)
            pos = series.index
            for k, origin in enumerate(pd.DatetimeIndex(origins)):
                end = pos.searchsorted(origin, side="right")  # rows <= origin
                y = values[max(0, end - self.window) : end]
                if len(y) < 2 * self.period or np.isnan(y).any():
                    continue
                fc = self._forecast_one(y, self.params[site])
                start = (self.sites_.index(site) * len(origins) + k) * self.horizon
                p50[start : start + self.horizon] = fc
        grid["p50"] = p50
        return grid

    def fit(self, history: pd.DataFrame, device: str | None = None) -> ETSForecaster:
        wanted = self.cfg.ets.sites
        self.sites_ = [s for s in dict.fromkeys(history["site_id"]) if not wanted or s in wanted]
        lookup = target_lookup(history, self.target)
        for site in self.sites_:
            y = lookup[site].to_numpy(dtype=float)[-self.window :]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = _hw(y, self.period).fit(optimized=True)
            self.params[site] = {k: float(res.params[k]) for k in PARAM_KEYS}
        end = history["timestamp"].max()
        n_cal = self.cfg.ets.calibration_origins
        cal_start = end - pd.Timedelta(days=n_cal + 1)
        origins = [
            o for o in daily_origins(cal_start, end) if o + pd.Timedelta(hours=self.horizon) <= end
        ][-n_cal:]
        cal = self._point(history, origins)
        self.residuals = ResidualQuantiles().fit(cal, history, self.target)
        self.fit_info_ = {
            "window_hours": self.window,
            "seasonal_periods": self.period,
            "calibration_origins": len(origins),
            "params": self.params,
        }
        return self

    def predict(self, panel: pd.DataFrame, origins: Sequence[pd.Timestamp]) -> pd.DataFrame:
        point = self._point(panel, origins).dropna(subset=["p50"])
        return finalize_quantiles(self.residuals.apply(point))

    def _save_payload(self, path: Path) -> None:
        payload = {
            "window": self.window,
            "period": self.period,
            "params": self.params,
            "residuals": self.residuals.to_dict(),
        }
        with (path / "ets.json").open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    def _load_payload(self, path: Path, device: str | None = None) -> None:
        with (path / "ets.json").open(encoding="utf-8") as fh:
            payload = json.load(fh)
        self.window = int(payload["window"])
        self.period = int(payload["period"])
        self.params = {s: dict(p) for s, p in payload["params"].items()}
        self.residuals = ResidualQuantiles.from_dict(payload["residuals"])
