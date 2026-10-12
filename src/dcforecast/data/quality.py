"""Sensor-fault injection, detection, causal imputation and the data-quality (QA) report.

Fault model (per site and metered column): multi-hour gaps, isolated missing readings, spikes
(surges and drop-outs), stuck meters (value frozen for hours) and data-logger outages (whole
rows missing). Detection is rule-based (range checks, run-length of identical readings and a
two-sided neighbour test for single-hour spikes). Imputation is *causal* -- it only uses values
before the gap -- so cleaned history never leaks future information into model inputs:

1. cross-sensor: facility = IT x PUE_est (or IT = facility / PUE_est) when only one meter is bad,
   with PUE_est the trailing median of observed PUE,
2. seasonal persistence with level adjustment: ``y[t] = y[t-L] * y[last] / y[last-L]``
   (L = 168 h for power, additive with L = 24 h for weather),
3. last observation carried forward as a final fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from dcforecast.config import FaultConfig
from dcforecast.data.sites import Site

POWER_COLUMNS = ["it_load_kw", "facility_kw"]
WEATHER_COLUMNS = ["temp_c", "rh_pct"]
METERED_COLUMNS = POWER_COLUMNS + WEATHER_COLUMNS

# fault codes used in the ground-truth log
OK, GAP, MISSING, SPIKE, STUCK, OUTAGE = 0, 1, 2, 3, 4, 5
FAULT_NAMES = {GAP: "gap", MISSING: "missing", SPIKE: "spike", STUCK: "stuck", OUTAGE: "outage"}

# detection reasons
R_OK, R_MISSING, R_RANGE, R_STUCK, R_SPIKE = 0, 1, 2, 3, 4
REASON_NAMES = {R_MISSING: "missing", R_RANGE: "out_of_range", R_STUCK: "stuck", R_SPIKE: "spike"}


# ============================================================================ injection
def inject_faults(
    truth: pd.DataFrame, faults: FaultConfig, rng: np.random.Generator, sites: list[Site]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Corrupt fault-free telemetry with realistic sensor problems.

    Returns the faulty telemetry (outage rows removed) and a ground-truth fault log with one row
    per corrupted cell.
    """
    out_frames: list[pd.DataFrame] = []
    logs: list[pd.DataFrame] = []
    for site in sites:
        df = truth[truth["site_id"] == site.site_id].sort_values("timestamp").reset_index(drop=True)
        n = len(df)
        years = n / 8760.0
        codes = {c: np.zeros(n, dtype=np.int8) for c in METERED_COLUMNS}
        vals = {c: df[c].to_numpy(dtype=float).copy() for c in METERED_COLUMNS}
        for col in METERED_COLUMNS:
            v, code = vals[col], codes[col]
            for _ in range(rng.poisson(faults.stuck_episodes_per_year * years)):
                s = int(rng.integers(1, max(2, n - 8)))
                length = int(rng.integers(6, max(7, faults.stuck_max_hours + 1)))
                e = min(n, s + length)
                v[s + 1 : e] = v[s]
                code[s + 1 : e] = STUCK
            idx = np.flatnonzero(rng.random(n) < faults.spike_rate)
            idx = idx[(idx > 0) & (idx < n - 1)]
            if len(idx):
                sign = rng.choice([-1.0, 1.0], size=len(idx))
                if col in POWER_COLUMNS:
                    surge = rng.random(len(idx)) < 0.6
                    factor = np.where(
                        surge, rng.uniform(1.5, 2.6, len(idx)), rng.uniform(0.0, 0.08, len(idx))
                    )
                    v[idx] = v[idx] * factor
                elif col == "temp_c":
                    v[idx] = v[idx] + sign * rng.uniform(10.0, 25.0, len(idx))
                else:
                    v[idx] = np.clip(v[idx] + sign * rng.uniform(25.0, 50.0, len(idx)), 0.0, 100.0)
                code[idx] = SPIKE
            for _ in range(rng.poisson(faults.gap_episodes_per_year * years)):
                s = int(rng.integers(0, n))
                length = int(rng.integers(2, max(3, faults.gap_max_hours + 1)))
                v[s : s + length] = np.nan
                code[s : s + length] = GAP
            iso = rng.random(n) < faults.isolated_missing_rate
            v[iso] = np.nan
            code[iso] = MISSING
        # whole-row data-logger outages
        drop = np.zeros(n, dtype=bool)
        for _ in range(rng.poisson(2.0 * years)):
            s = int(rng.integers(0, n))
            drop[s : s + int(rng.integers(1, 7))] = True
        for col in METERED_COLUMNS:
            codes[col][drop] = OUTAGE
            vals[col][drop] = np.nan

        faulty = df.copy()
        for col in METERED_COLUMNS:
            faulty[col] = np.round(vals[col], 1)
            bad = np.flatnonzero(codes[col])
            if len(bad):
                logs.append(
                    pd.DataFrame(
                        {
                            "site_id": site.site_id,
                            "timestamp": df["timestamp"].to_numpy()[bad],
                            "column": col,
                            "fault": [FAULT_NAMES[int(c)] for c in codes[col][bad]],
                            "true_value": df[col].to_numpy()[bad],
                            "observed_value": faulty[col].to_numpy()[bad],
                        }
                    )
                )
        out_frames.append(faulty[~drop])
    telemetry = pd.concat(out_frames, ignore_index=True)
    fault_log = (
        pd.concat(logs, ignore_index=True)
        if logs
        else pd.DataFrame(
            columns=["site_id", "timestamp", "column", "fault", "true_value", "observed_value"]
        )
    )
    return telemetry, fault_log


# ============================================================================ detection
def _run_flags(x: np.ndarray, min_run: int) -> np.ndarray:
    """Flag repeated identical readings (all but the first of runs >= ``min_run``)."""
    n = len(x)
    flags = np.zeros(n, dtype=bool)
    if n == 0:
        return flags
    same = np.zeros(n, dtype=bool)
    same[1:] = (x[1:] == x[:-1]) & ~np.isnan(x[1:])
    # run ids: a new run starts wherever same is False
    run_id = np.cumsum(~same)
    counts = np.bincount(run_id)
    long_run = counts[run_id] >= min_run
    flags = same & long_run
    return flags


def _spike_flags(x: np.ndarray, k: float, abs_floor: float, rel_floor: float) -> np.ndarray:
    """Single-hour spikes: the reading jumps away from BOTH neighbours in the same direction."""
    s = pd.Series(x)
    prev = s.ffill().shift(1).to_numpy()
    nxt = s.bfill().shift(-1).to_numpy()
    d1 = x - prev
    d2 = x - nxt
    scale = (
        pd.Series(np.abs(np.diff(x, prepend=np.nan)))
        .rolling(168, min_periods=24)
        .median()
        .shift(1)
        .bfill()
        .to_numpy()
    )
    thr = np.maximum(k * 1.4826 * np.nan_to_num(scale, nan=abs_floor), abs_floor)
    thr = np.maximum(thr, rel_floor * np.abs(np.nan_to_num(prev, nan=0.0)))
    with np.errstate(invalid="ignore"):
        spike = (np.sign(d1) == np.sign(d2)) & (np.abs(d1) > thr) & (np.abs(d2) > thr)
    return np.nan_to_num(spike, nan=0).astype(bool)


def detect_anomalies(x: np.ndarray, column: str, capacity_kw: float) -> np.ndarray:
    """Return a per-reading reason code (0 = OK) for one metered column."""
    reason = np.zeros(len(x), dtype=np.int8)
    reason[np.isnan(x)] = R_MISSING
    work = x.copy()
    if column in POWER_COLUMNS:
        bad = (x <= 0) | (x > 1.6 * capacity_kw)
        min_run, k, abs_floor, rel_floor = 3, 6.0, 0.02 * capacity_kw, 0.12
    elif column == "temp_c":
        bad = (x < -60) | (x > 65)
        min_run, k, abs_floor, rel_floor = 6, 6.0, 6.0, 0.0
    else:
        bad = (x < 0) | (x > 100)
        min_run, k, abs_floor, rel_floor = 6, 6.0, 20.0, 0.0
    bad &= ~np.isnan(x)
    reason[bad] = R_RANGE
    work[bad] = np.nan
    stuck = _run_flags(work, min_run) & (reason == R_OK)
    if column == "rh_pct":  # saturated humidity (fog) legitimately reads 100 % for hours
        stuck &= work < 99.95
    reason[stuck] = R_STUCK
    work[stuck] = np.nan
    spike = _spike_flags(work, k, abs_floor, rel_floor) & (reason == R_OK)
    reason[spike] = R_SPIKE
    return reason


# ============================================================================ imputation
def seasonal_fill(y: np.ndarray, period: int, multiplicative: bool) -> np.ndarray:
    """Causally impute NaNs with level-adjusted seasonal persistence (see module docstring)."""
    y = y.astype(float).copy()
    nan_idx = np.flatnonzero(np.isnan(y))
    if len(nan_idx) == 0:
        return y
    anchor: int | None = None
    prev_t = -2
    for t in nan_idx:
        if t != prev_t + 1:
            # a new gap starts; y[t-1] is an original (valid) observation
            anchor = t - 1 if t > 0 else None
        prev_t = t
        ref = y[t - period] if t - period >= 0 else np.nan
        anchor_ref = y[anchor - period] if anchor is not None and anchor - period >= 0 else np.nan
        if not np.isnan(ref) and anchor is not None and not np.isnan(anchor_ref):
            if not multiplicative:
                y[t] = ref + (y[anchor] - anchor_ref)
            elif anchor_ref > 0:
                y[t] = ref * (y[anchor] / anchor_ref)
            else:
                y[t] = ref
        elif not np.isnan(ref):
            y[t] = ref
        elif anchor is not None:
            y[t] = y[anchor]
    # leading NaNs (no history at all): back-fill from the first valid observation
    if np.isnan(y).any():
        y = pd.Series(y).ffill().bfill().to_numpy()
    return y


def _trailing_median(x: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(x).rolling(window, min_periods=1).median().shift(1).ffill().bfill().to_numpy()


@dataclass
class CleaningResult:
    """Output of :func:`clean_telemetry`."""

    clean: pd.DataFrame
    reasons: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)


def clean_site(df: pd.DataFrame, site: Site, index: pd.DatetimeIndex) -> tuple[pd.DataFrame, dict]:
    """Clean one site's telemetry on a complete hourly grid."""
    df = df.set_index("timestamp").reindex(index)
    raw = {c: df[c].to_numpy(dtype=float) for c in METERED_COLUMNS}
    reasons = {c: detect_anomalies(raw[c], c, site.it_capacity_kw) for c in METERED_COLUMNS}
    vals = {c: np.where(reasons[c] == R_OK, raw[c], np.nan) for c in METERED_COLUMNS}

    it, fac = vals["it_load_kw"], vals["facility_kw"]
    pue_obs = np.where(~np.isnan(it) & ~np.isnan(fac), fac / it, np.nan)
    pue_obs[(pue_obs < 1.0) | (pue_obs > 3.0)] = np.nan
    pue_est = _trailing_median(pd.Series(pue_obs).ffill().to_numpy(), 48)
    pue_est = np.nan_to_num(pue_est, nan=1.3)

    it_missing, fac_missing = np.isnan(it), np.isnan(fac)
    # 1) cross-sensor imputation when exactly one power meter is bad
    it_f = np.where(it_missing & ~fac_missing, fac / pue_est, it)
    # 2) seasonal persistence for hours where both are bad
    it_f = seasonal_fill(it_f, 168, multiplicative=True)
    fac_f = np.where(fac_missing, it_f * pue_est, fac)
    pue = fac_f / it_f
    implausible = (pue < 1.02) | (pue > 2.5)
    fac_f = np.where(implausible, it_f * pue_est, fac_f)

    temp_f = seasonal_fill(vals["temp_c"], 24, multiplicative=False)
    rh_f = np.clip(seasonal_fill(vals["rh_pct"], 24, multiplicative=False), 0.0, 100.0)

    out = pd.DataFrame(
        {
            "timestamp": index,
            "site_id": site.site_id,
            "it_load_kw": it_f,
            "facility_kw": fac_f,
            "pue": fac_f / it_f,
            "temp_c": temp_f,
            "rh_pct": rh_f,
            "it_imputed": it_missing,
            "facility_imputed": fac_missing | implausible,
            "temp_imputed": np.isnan(vals["temp_c"]),
            "rh_imputed": np.isnan(vals["rh_pct"]),
        }
    )
    return out, reasons


def clean_telemetry(
    telemetry: pd.DataFrame, sites: list[Site], start: pd.Timestamp, end: pd.Timestamp
) -> CleaningResult:
    """Detect anomalies and impute every site on the complete hourly grid ``[start, end]``."""
    index = pd.date_range(start, end, freq="h")
    frames, reasons = [], {}
    for site in sites:
        part = telemetry[telemetry["site_id"] == site.site_id]
        cleaned, rs = clean_site(part, site, index)
        frames.append(cleaned)
        reasons[site.site_id] = rs
    return CleaningResult(clean=pd.concat(frames, ignore_index=True), reasons=reasons)


# ============================================================================ QA report
def qa_report(
    result: CleaningResult, telemetry: pd.DataFrame, fault_log: pd.DataFrame
) -> dict[str, Any]:
    """Data-quality report: what was found, how well detection worked, imputation error."""
    clean = result.clean
    per_site: list[dict[str, Any]] = []
    totals: dict[str, int] = {
        "rows_expected": 0,
        "rows_received": 0,
        "missing": 0,
        "out_of_range": 0,
        "stuck": 0,
        "spike": 0,
    }
    for site_id, rs in result.reasons.items():
        n = int((clean["site_id"] == site_id).sum())
        received = int((telemetry["site_id"] == site_id).sum())
        row: dict[str, Any] = {
            "site_id": site_id,
            "rows_expected": n,
            "rows_received": received,
        }
        totals["rows_expected"] += n
        totals["rows_received"] += received
        for name, code in (
            ("missing", R_MISSING),
            ("out_of_range", R_RANGE),
            ("stuck", R_STUCK),
            ("spike", R_SPIKE),
        ):
            cnt = int(sum(int((r == code).sum()) for r in rs.values()))
            row[name] = cnt
            totals[name] += cnt
        sub = clean[clean["site_id"] == site_id]
        row["facility_imputed_pct"] = round(100 * float(sub["facility_imputed"].mean()), 3)
        per_site.append(row)

    # detection quality vs ground truth (spike / stuck faults; NaNs are trivially detected)
    detected_rows = []
    for site_id, rs in result.reasons.items():
        ts = clean.loc[clean["site_id"] == site_id, "timestamp"].to_numpy()
        for col, r in rs.items():
            hit = np.flatnonzero(np.isin(r, [R_RANGE, R_STUCK, R_SPIKE]))
            if len(hit):
                detected_rows.append(
                    pd.DataFrame({"site_id": site_id, "timestamp": ts[hit], "column": col})
                )
    detected = (
        pd.concat(detected_rows, ignore_index=True)
        if detected_rows
        else pd.DataFrame(columns=["site_id", "timestamp", "column"])
    )
    injected = fault_log[fault_log["fault"].isin(["spike", "stuck"])]
    key = ["site_id", "timestamp", "column"]
    merged = injected.merge(detected.assign(found=True), on=key, how="left")
    merged["found"] = merged["found"].fillna(False).astype(bool)
    detection: dict[str, Any] = {}
    for fault in ("spike", "stuck"):
        sub = merged[merged["fault"] == fault]
        detection[f"{fault}_injected"] = len(sub)
        detection[f"{fault}_recall"] = round(float(sub["found"].mean()), 4) if len(sub) else None
    tp = int(detected.merge(injected[key], on=key, how="inner").shape[0])
    detection["flagged"] = len(detected)
    detection["precision"] = round(tp / len(detected), 4) if len(detected) else None

    # imputation error on every corrupted cell, against the fault-free truth
    imputation: dict[str, Any] = {}
    long_clean = clean.melt(
        id_vars=["site_id", "timestamp"], value_vars=METERED_COLUMNS, var_name="column"
    )
    cells = fault_log.merge(long_clean, on=key, how="inner")
    for col in METERED_COLUMNS:
        sub = cells[cells["column"] == col]
        if not len(sub):
            continue
        err = (sub["value"] - sub["true_value"]).abs()
        entry = {"cells": len(sub), "mae": round(float(err.mean()), 3)}
        if col in POWER_COLUMNS:
            entry["mape_pct"] = round(float((err / sub["true_value"].abs()).mean() * 100), 3)
        imputation[col] = entry

    fault_counts = (
        fault_log.groupby(["fault"]).size().astype(int).to_dict() if len(fault_log) else {}
    )
    return {
        "totals": totals,
        "injected_faults": {str(k): int(v) for k, v in fault_counts.items()},
        "detection": detection,
        "imputation_error": imputation,
        "per_site": per_site,
        "facility_imputed_pct": round(100 * float(clean["facility_imputed"].mean()), 3),
    }
