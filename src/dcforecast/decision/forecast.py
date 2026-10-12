"""Multi-day forecasts beyond the model horizon, plus an empirical PUE curve per site.

Models forecast 24 h directly. For 48-72 h decision horizons we roll forward in 24 h blocks,
feeding each block's P50 back as pseudo-observations (and the weather forecast in place of
observed weather). Interval half-widths of block ``b`` (0-based) are widened by ``sqrt(b+1)``
-- a simple random-walk style heuristic for the extra uncertainty of recursive steps.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from dcforecast.data.prepare import PanelData
from dcforecast.models.base import Forecaster


class PUECurve:
    """Mean PUE as a function of outside temperature, per site (binned + interpolated)."""

    def __init__(self, bin_width_c: float = 2.0) -> None:
        self.bin_width_c = bin_width_c
        self.curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def fit(self, history: pd.DataFrame, days: int = 365) -> PUECurve:
        recent = history[
            history["timestamp"] > history["timestamp"].max() - pd.Timedelta(days=days)
        ]
        for site, grp in recent.groupby("site_id"):
            t = grp["temp_c"].to_numpy(dtype=float)
            p = grp["pue"].to_numpy(dtype=float)
            bins = np.floor(t / self.bin_width_c)
            frame = (
                pd.DataFrame({"bin": bins, "t": t, "p": p})
                .groupby("bin")
                .agg(t=("t", "mean"), p=("p", "median"), n=("p", "size"))
            )
            frame = frame[frame["n"] >= 10]
            self.curves[str(site)] = (frame["t"].to_numpy(), frame["p"].to_numpy())
        return self

    def predict(self, site: str, temp_c: np.ndarray) -> np.ndarray:
        x, y = self.curves[site]
        return np.interp(np.asarray(temp_c, dtype=float), x, y)


def forecast_ahead(
    model: Forecaster,
    data: PanelData,
    hours: int,
    sites: list[str] | None = None,
    pue_curve: PUECurve | None = None,
) -> pd.DataFrame:
    """Forecast ``hours`` ahead of the end of history (recursively in model-horizon blocks)."""
    target = model.target
    h = model.horizon
    origin = data.history_end
    work = data.combined()
    if sites:
        work = work[work["site_id"].isin(sites)].reset_index(drop=True)
    future = work["timestamp"] > origin
    pue_curve = pue_curve or PUECurve().fit(data.history)
    # observed-weather columns are unknown in the future: use the weather forecast instead
    work.loc[future, "temp_c"] = work.loc[future, "temp_fcst_c"]
    work.loc[future, "rh_pct"] = work.loc[future, "rh_fcst_pct"]
    for site in work.loc[future, "site_id"].unique():
        m = future & (work["site_id"] == site)
        work.loc[m, "pue"] = pue_curve.predict(site, work.loc[m, "temp_fcst_c"].to_numpy())
    work.loc[future, "facility_imputed"] = False
    max_ts = work["timestamp"].max()
    blocks = math.ceil(hours / h)
    out = []
    key = pd.MultiIndex.from_frame(work[["site_id", "timestamp"]])
    for b in range(blocks):
        o = origin + pd.Timedelta(hours=b * h)
        if o + pd.Timedelta(hours=h) > max_ts:
            break
        fc = model.predict(work, [o])
        if fc.empty:
            break
        if b > 0:
            widen = math.sqrt(b + 1)
            fc["p10"] = fc["p50"] - (fc["p50"] - fc["p10"]) * widen
            fc["p90"] = fc["p50"] + (fc["p90"] - fc["p50"]) * widen
        fc["horizon"] = fc["horizon"] + b * h
        fc["origin"] = origin
        out.append(fc)
        # feed P50 back as pseudo-observations for the next block
        idx = key.get_indexer(pd.MultiIndex.from_frame(fc[["site_id", "timestamp"]]))
        work.loc[idx[idx >= 0], target] = fc["p50"].to_numpy()[idx >= 0]
    if not out:
        raise ValueError(
            "No forecast could be produced (is the future covariate feed long enough?)"
        )
    res = pd.concat(out, ignore_index=True)
    return res[res["horizon"] <= hours].reset_index(drop=True)
