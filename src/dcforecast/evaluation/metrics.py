"""Point and probabilistic forecast metrics.

* MAE, RMSE (kW), MAPE and sMAPE (%),
* MASE: MAE scaled by the in-sample MAE of the 24 h seasonal-naive forecast on the training
  data of the same site (values < 1 beat "same hour yesterday" on average),
* pinball loss averaged over P10/P50/P90 (kW) and its MASE-style scaled version,
* empirical coverage of the P10-P90 interval (nominal 80 %) and its mean width.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

QUANTILE_LEVELS = {"p10": 0.1, "p50": 0.5, "p90": 0.9}


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean(np.abs(y - yhat)))


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def mape(y: np.ndarray, yhat: np.ndarray) -> float:
    """Mean absolute percentage error in % (targets are strictly positive power)."""
    return float(np.mean(np.abs(y - yhat) / np.abs(y)) * 100)


def smape(y: np.ndarray, yhat: np.ndarray) -> float:
    """Symmetric MAPE in % (0-200 scale)."""
    denom = np.abs(y) + np.abs(yhat)
    return float(np.mean(np.where(denom > 0, 2 * np.abs(y - yhat) / denom, 0.0)) * 100)


def pinball(y: np.ndarray, pred: np.ndarray, q: float) -> float:
    """Mean pinball loss of quantile ``q``."""
    diff = y - pred
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def mean_pinball(y: np.ndarray, quantiles: Mapping[float, np.ndarray]) -> float:
    """Pinball loss averaged over quantile levels."""
    return float(np.mean([pinball(y, pred, q) for q, pred in quantiles.items()]))


def coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Share of observations inside ``[lo, hi]``."""
    return float(np.mean((y >= lo) & (y <= hi)))


def naive_scale(history: pd.DataFrame, target: str, season: int = 24) -> dict[str, float]:
    """In-sample MAE of the seasonal-naive forecast per site (the MASE denominator)."""
    out: dict[str, float] = {}
    for site, grp in history.sort_values("timestamp").groupby("site_id"):
        y = grp[target].to_numpy(dtype=float)
        d = np.abs(y[season:] - y[:-season])
        out[str(site)] = float(np.nanmean(d)) if len(d) else float("nan")
    return out


def evaluate(frame: pd.DataFrame, target: str) -> dict:
    """All metrics for a frame with ``target`` and ``p10/p50/p90`` columns.

    If the frame carries a per-row ``scale`` column (the site's seasonal-naive MAE on the
    training data of the fold), scaled metrics (MASE, scaled pinball) are computed per row and
    then averaged, so large sites do not dominate.
    """
    y = frame[target].to_numpy(dtype=float)
    p10 = frame["p10"].to_numpy(dtype=float)
    p50 = frame["p50"].to_numpy(dtype=float)
    p90 = frame["p90"].to_numpy(dtype=float)
    res = {
        "n": len(frame),
        "mae": mae(y, p50),
        "rmse": rmse(y, p50),
        "mape": mape(y, p50),
        "smape": smape(y, p50),
        "pinball": mean_pinball(y, {0.1: p10, 0.5: p50, 0.9: p90}),
        "coverage_80": coverage(y, p10, p90),
        "interval_width": float(np.mean(p90 - p10)),
        "bias": float(np.mean(p50 - y)),
    }
    if "scale" in frame.columns:
        s = frame["scale"].to_numpy(dtype=float)
        res["mase"] = float(np.mean(np.abs(y - p50) / s))
        pin = np.mean(
            [
                np.maximum(q * (y - p), (q - 1) * (y - p))
                for q, p in ((0.1, p10), (0.5, p50), (0.9, p90))
            ],
            axis=0,
        )
        res["pinball_scaled"] = float(np.mean(pin / s))
    return res
