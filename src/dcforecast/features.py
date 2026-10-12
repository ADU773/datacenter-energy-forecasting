"""Leakage-safe feature engineering.

Every feature for a target hour ``tau`` uses only information available at the forecast origin
``t = tau - h`` for any horizon ``h <= H`` (H = 24):

* calendar, holiday and Fourier terms of ``tau`` -- deterministic, known in advance;
* target lags of at least H hours (24, 48, 168) and rolling statistics computed on the target
  shifted by H hours, i.e. strictly from the past;
* weather: the *day-ahead forecast* for ``tau`` (``temp_fcst_c``, ``rh_fcst_pct``). We assume
  a weather forecast is available at prediction time, as it is for any real operator; the
  generator gives it realistic errors (about 1.3 C) so models do not get oracle weather.
  Actual weather is only used lagged by H hours.

Power features are expressed relative to a trailing *level* (mean of the target over the week
ending H hours before ``tau``), which lets tree models cope with load growth they have never
seen in training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dcforecast.config import Config
from dcforecast.data.calendar import local_calendar
from dcforecast.data.sites import SITES, site_index

CDH_BASE_C = 18.0


def add_calendar(panel: pd.DataFrame) -> pd.DataFrame:
    """Add site-local calendar columns (local hour, weekday, holiday ...) to a panel."""
    parts = []
    for site_id, part in panel.groupby("site_id", sort=False):
        cal = local_calendar(part["timestamp"], SITES[str(site_id)].utc_offset_h)
        cal.index = part.index
        local = pd.DatetimeIndex(part["timestamp"]) + pd.Timedelta(
            hours=SITES[str(site_id)].utc_offset_h
        )
        cal["is_month_end"] = np.asarray(local.is_month_end)
        parts.append(cal)
    cal_all = pd.concat(parts).loc[panel.index]
    out = panel.copy()
    for col in cal_all.columns:
        out[col] = cal_all[col].to_numpy()
    return out


def fourier_terms(panel: pd.DataFrame, daily: int, weekly: int, yearly: int) -> pd.DataFrame:
    """Sine/cosine terms for daily, weekly and yearly seasonality (site-local time)."""
    hour = panel["local_hour"].to_numpy(dtype=float)
    week_pos = panel["dow"].to_numpy(dtype=float) * 24 + hour
    year_pos = panel["doy"].to_numpy(dtype=float) - 1 + hour / 24
    cols: dict[str, np.ndarray] = {}
    for k in range(1, daily + 1):
        cols[f"fd_sin{k}"] = np.sin(2 * np.pi * k * hour / 24)
        cols[f"fd_cos{k}"] = np.cos(2 * np.pi * k * hour / 24)
    for k in range(1, weekly + 1):
        cols[f"fw_sin{k}"] = np.sin(2 * np.pi * k * week_pos / 168)
        cols[f"fw_cos{k}"] = np.cos(2 * np.pi * k * week_pos / 168)
    for k in range(1, yearly + 1):
        cols[f"fy_sin{k}"] = np.sin(2 * np.pi * k * year_pos / 365.25)
        cols[f"fy_cos{k}"] = np.cos(2 * np.pi * k * year_pos / 365.25)
    return pd.DataFrame(cols, index=panel.index)


def build_features(
    panel: pd.DataFrame, cfg: Config, target: str | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """Build the tabular feature matrix for every row of ``panel``.

    ``panel`` must hold complete hourly series per site (history and, optionally, future rows
    with a NaN target). Returns the augmented frame and the ordered list of feature columns.
    The column ``level`` is the normaliser used for the relative target ``y_rel``.
    """
    target = target or cfg.forecast.target
    h = cfg.forecast.horizon
    fc = cfg.features
    df = panel.sort_values(["site_id", "timestamp"]).reset_index(drop=True)
    if "local_hour" not in df.columns:
        df = add_calendar(df)
    df = pd.concat(
        [df, fourier_terms(df, fc.fourier_daily, fc.fourier_weekly, fc.fourier_yearly)], axis=1
    )
    df["site_code"] = df["site_id"].map(site_index).astype(np.int16)

    g = df.groupby("site_id", sort=False)
    y_past = g[target].shift(h)  # the most recent value usable for every horizon <= h
    level = (
        y_past.groupby(df["site_id"], sort=False)
        .rolling(168, min_periods=24)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["level"] = level
    df["log_level"] = np.log(level)
    new: dict[str, pd.Series] = {}
    for lag in fc.lags:
        new[f"lag{lag}_rel"] = g[target].shift(lag) / level
    for w in fc.rolling_windows:
        roll = y_past.groupby(df["site_id"], sort=False).rolling(w, min_periods=max(2, w // 4))
        new[f"roll_mean{w}_rel"] = roll.mean().reset_index(level=0, drop=True) / level
        new[f"roll_std{w}_rel"] = roll.std().reset_index(level=0, drop=True) / level
    if "pue" in df.columns:
        new["pue_lag24"] = g["pue"].shift(max(h, 24))
    if "temp_c" in df.columns:
        new["temp_lag24"] = g["temp_c"].shift(max(h, 24))
    new["cdh_fcst"] = np.clip(df["temp_fcst_c"] - CDH_BASE_C, 0.0, None)
    df = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)
    if target in df.columns:
        df["y_rel"] = df[target] / df["level"]

    calendar_cols = [
        "local_hour",
        "dow",
        "month",
        "is_weekend",
        "is_holiday",
        "is_workday",
        "is_month_end",
    ]
    fourier_cols = [c for c in df.columns if c.startswith(("fd_", "fw_", "fy_"))]
    weather_cols = ["temp_fcst_c", "rh_fcst_pct", "cdh_fcst"]
    target_cols = [k for k in new if k != "cdh_fcst"]
    features = [
        "site_code",
        *calendar_cols,
        *fourier_cols,
        *target_cols,
        "log_level",
        *weather_cols,
    ]
    for col in ("is_weekend", "is_holiday", "is_workday", "is_month_end"):
        df[col] = df[col].astype(np.int8)
    return df, features
