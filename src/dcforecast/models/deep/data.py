"""Windowed tensors for the global multi-site sequence models.

All sites are stacked into flat arrays; a sample is identified by the row ``i`` of its forecast
origin. For a sample:

* encoder rows ``i-L+1 .. i``: target (scaled), observed covariates (actual weather, PUE,
  imputation flag) and known covariates (calendar, weather forecast),
* decoder rows ``i+1 .. i+H``: known covariates plus seasonal target lags
  ``y[tau-24]``, ``y[tau-168]`` (all <= i because every lag >= H),
* scale: mean target over the encoder window (instance normalisation, robust to growth),
* target: ``y[tau] / scale`` and a mask that is 0 for unknown or imputed targets.

Batches are gathered with vectorised NumPy fancy indexing in ``__getitems__`` (one call per
batch instead of one per sample), which is why DataLoader workers are optional.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from dcforecast.data.sites import site_index
from dcforecast.features import add_calendar

KNOWN_FEATURES = [
    "temp_fcst",
    "rh_fcst",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "doy_sin",
    "doy_cos",
    "is_holiday",
    "is_workday",
    "is_month_end",
]
PAST_FEATURES = ["temp", "rh", "pue", "imputed"]


@dataclass
class NormStats:
    """Global standardisation statistics fitted on training history."""

    temp_mean: float
    temp_std: float
    rh_mean: float
    rh_std: float
    pue_mean: float
    pue_std: float

    @classmethod
    def fit(cls, history: pd.DataFrame) -> NormStats:
        def ms(col: str) -> tuple[float, float]:
            v = history[col].to_numpy(dtype=float)
            return float(np.nanmean(v)), float(np.nanstd(v) or 1.0)

        t, r, p = ms("temp_c"), ms("rh_pct"), ms("pue")
        return cls(t[0], t[1], r[0], r[1], p[0], max(p[1], 1e-3))

    def to_dict(self) -> dict[str, float]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NormStats:
        return cls(**{k: float(v) for k, v in data.items()})


@dataclass
class DeepArrays:
    """Flat per-row arrays for all sites (sorted by site, then time)."""

    y: np.ndarray  # (N,) float32, raw target (NaN where unknown)
    mask: np.ndarray  # (N,) float32, 1 = usable training target
    past: np.ndarray  # (N, Fp + Fk) float32: observed + known covariates (encoder input)
    known: np.ndarray  # (N, Fk) float32: known-in-advance covariates (decoder input)
    site: np.ndarray  # (N,) int64 global site code
    timestamps: np.ndarray  # (N,) datetime64[ns]
    bounds: dict[str, tuple[int, int]]  # site -> [start, end) row range

    @property
    def past_dim(self) -> int:
        return int(self.past.shape[1])

    @property
    def known_dim(self) -> int:
        return int(self.known.shape[1])


def build_arrays(panel: pd.DataFrame, stats: NormStats, target: str) -> DeepArrays:
    """Convert a (history [+ future]) panel into :class:`DeepArrays`."""
    df = panel.sort_values(["site_id", "timestamp"], ignore_index=True)
    df = add_calendar(df)
    n = len(df)
    y = df[target].to_numpy(dtype=np.float32) if target in df else np.full(n, np.nan, np.float32)
    imputed = (
        df["facility_imputed"].fillna(False).to_numpy(dtype=bool)
        if "facility_imputed" in df
        else np.zeros(n, dtype=bool)
    )
    mask = (~np.isnan(y) & ~imputed).astype(np.float32)

    def col(name: str, fill: float = np.nan) -> np.ndarray:
        return df[name].to_numpy(dtype=float) if name in df else np.full(n, fill)

    hour = df["local_hour"].to_numpy(dtype=float)
    dow = df["dow"].to_numpy(dtype=float)
    doy = df["doy"].to_numpy(dtype=float)
    known = np.column_stack(
        [
            (col("temp_fcst_c") - stats.temp_mean) / stats.temp_std,
            (col("rh_fcst_pct") - stats.rh_mean) / stats.rh_std,
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * dow / 7),
            np.cos(2 * np.pi * dow / 7),
            np.sin(2 * np.pi * doy / 365.25),
            np.cos(2 * np.pi * doy / 365.25),
            df["is_holiday"].to_numpy(dtype=float),
            df["is_workday"].to_numpy(dtype=float),
            df["is_month_end"].to_numpy(dtype=float),
        ]
    ).astype(np.float32)
    past = np.column_stack(
        [
            (col("temp_c") - stats.temp_mean) / stats.temp_std,
            (col("rh_pct") - stats.rh_mean) / stats.rh_std,
            (col("pue") - stats.pue_mean) / stats.pue_std,
            imputed.astype(float),
        ]
    ).astype(np.float32)
    # Unknown observed covariates (future rows) are zero = "average"; never used by the encoder
    # for rows after the origin anyway.
    past = np.nan_to_num(past, nan=0.0)
    known = np.nan_to_num(known, nan=0.0)
    site_ids = df["site_id"].astype(str).to_numpy()
    bounds: dict[str, tuple[int, int]] = {}
    for s in dict.fromkeys(site_ids):
        idx = np.flatnonzero(site_ids == s)
        bounds[str(s)] = (int(idx[0]), int(idx[-1]) + 1)
    return DeepArrays(
        y=y,
        mask=mask,
        past=np.ascontiguousarray(np.concatenate([past, known], axis=1)),
        known=np.ascontiguousarray(known),
        site=np.array([site_index(s) for s in site_ids], dtype=np.int64),
        timestamps=df["timestamp"].to_numpy(),
        bounds=bounds,
    )


def origin_rows(
    arrays: DeepArrays,
    lookback: int,
    horizon: int,
    first: pd.Timestamp | None = None,
    last: pd.Timestamp | None = None,
    require_target: bool = True,
    stride: int = 1,
) -> np.ndarray:
    """Rows usable as forecast origins (full lookback, decoder rows inside the panel).

    ``first`` / ``last`` bound the origin timestamp; with ``require_target`` the whole decoder
    window must have known targets (training / validation).
    """
    rows = []
    for start, end in arrays.bounds.values():
        cand = np.arange(start + lookback - 1, end - horizon, stride)
        ts = arrays.timestamps[cand]
        keep = np.ones(len(cand), dtype=bool)
        if first is not None:
            keep &= ts >= np.datetime64(first)
        if last is not None:
            keep &= ts <= np.datetime64(last)
        cand = cand[keep]
        if require_target and len(cand):
            dec = cand[:, None] + np.arange(1, horizon + 1)[None, :]
            enc = cand[:, None] + np.arange(-lookback + 1, 1)[None, :]
            ok = ~np.isnan(arrays.y[dec]).any(axis=1) & ~np.isnan(arrays.y[enc]).any(axis=1)
            cand = cand[ok]
        rows.append(cand)
    return np.concatenate(rows) if rows else np.empty(0, dtype=np.int64)


class WindowDataset(Dataset):
    """Map-style dataset of forecast windows with batched ``__getitems__``."""

    def __init__(
        self,
        arrays: DeepArrays,
        rows: np.ndarray,
        lookback: int,
        horizon: int,
        decoder_lags: list[int],
    ) -> None:
        if any(lag < horizon for lag in decoder_lags):
            raise ValueError("decoder lags must be >= horizon (leakage)")
        self.a = arrays
        self.rows = np.asarray(rows, dtype=np.int64)
        self.enc_off = np.arange(-lookback + 1, 1, dtype=np.int64)
        self.dec_off = np.arange(1, horizon + 1, dtype=np.int64)
        self.lags = list(decoder_lags)

    @property
    def enc_dim(self) -> int:
        return 1 + self.a.past_dim

    @property
    def dec_dim(self) -> int:
        return self.a.known_dim + len(self.lags)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, k: int) -> tuple[torch.Tensor, ...]:
        return tuple(t[0] for t in self.__getitems__([k]))

    def __getitems__(self, ks: list[int]) -> tuple[torch.Tensor, ...]:
        a = self.a
        i = self.rows[np.asarray(ks, dtype=np.int64)]
        enc_idx = i[:, None] + self.enc_off[None, :]
        dec_idx = i[:, None] + self.dec_off[None, :]
        y_enc = a.y[enc_idx]
        scale = np.nanmean(y_enc, axis=1, keepdims=True).astype(np.float32)
        scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0).astype(np.float32)
        y_enc_s = np.nan_to_num(y_enc / scale, nan=1.0)
        enc = np.concatenate([y_enc_s[..., None], a.past[enc_idx]], axis=-1, dtype=np.float32)
        lag_feats = [np.nan_to_num(a.y[dec_idx - lag] / scale, nan=1.0) for lag in self.lags]
        dec = np.concatenate(
            [a.known[dec_idx], *(f[..., None] for f in lag_feats)], axis=-1, dtype=np.float32
        )
        tgt = a.y[dec_idx] / scale
        msk = a.mask[dec_idx] * np.isfinite(tgt)
        tgt = np.nan_to_num(tgt, nan=0.0).astype(np.float32)
        return (
            torch.from_numpy(enc),
            torch.from_numpy(dec),
            torch.from_numpy(a.site[i]),
            torch.from_numpy(tgt),
            torch.from_numpy(msk.astype(np.float32)),
            torch.from_numpy(scale[:, 0]),
        )


def passthrough_collate(batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    """``__getitems__`` already returns a collated batch."""
    return batch
