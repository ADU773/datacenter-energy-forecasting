"""Seasonal-naive baselines: ``y_hat(t+h) = y(t+h-m)`` with m = 24 h or 168 h.

Quantiles come from empirical relative residuals on the last weeks of the training history
(:class:`~dcforecast.models.base.ResidualQuantiles`).
"""

from __future__ import annotations

import json
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

CALIBRATION_DAYS = 56


class SeasonalNaive(Forecaster):
    """Repeat the value from one season (``season`` hours) earlier."""

    season: int = 24

    def __init__(self, cfg, season: int | None = None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(cfg)
        if season is not None:
            self.season = season
        if self.season < self.horizon:
            raise ValueError("season must be >= horizon for a leakage-free seasonal naive")
        self.residuals = ResidualQuantiles()

    def _point(self, panel: pd.DataFrame, origins: Sequence[pd.Timestamp]) -> pd.DataFrame:
        sites = self.sites_ or list(dict.fromkeys(panel["site_id"]))
        grid = origin_grid(sites, origins, self.horizon)
        lookup = target_lookup(panel, self.target)
        p50 = np.full(len(grid), np.nan)
        for site in sites:
            mask = (grid["site_id"] == site).to_numpy()
            src = pd.DatetimeIndex(grid.loc[mask, "timestamp"]) - pd.Timedelta(hours=self.season)
            p50[mask] = lookup[site].reindex(src).to_numpy()
        grid["p50"] = p50
        return grid

    def fit(self, history: pd.DataFrame, device: str | None = None) -> SeasonalNaive:
        self.sites_ = list(dict.fromkeys(history["site_id"]))
        end = history["timestamp"].max()
        start = end - pd.Timedelta(days=CALIBRATION_DAYS)
        origins = [
            o for o in daily_origins(start, end) if o + pd.Timedelta(hours=self.horizon) <= end
        ]
        cal = self._point(history, origins)
        self.residuals = ResidualQuantiles().fit(cal, history, self.target)
        self.fit_info_ = {"season": self.season, "calibration_origins": len(origins)}
        return self

    def predict(self, panel: pd.DataFrame, origins: Sequence[pd.Timestamp]) -> pd.DataFrame:
        return finalize_quantiles(self.residuals.apply(self._point(panel, origins)))

    def _save_payload(self, path: Path) -> None:
        payload = {"season": self.season, "residuals": self.residuals.to_dict()}
        with (path / "naive.json").open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    def _load_payload(self, path: Path, device: str | None = None) -> None:
        with (path / "naive.json").open(encoding="utf-8") as fh:
            payload = json.load(fh)
        self.season = int(payload["season"])
        self.residuals = ResidualQuantiles.from_dict(payload["residuals"])


class SeasonalNaive24(SeasonalNaive):
    """Same hour yesterday."""

    name = "naive24"
    season = 24


class SeasonalNaive168(SeasonalNaive):
    """Same hour last week."""

    name = "naive168"
    season = 168
