"""Physically-motivated synthetic telemetry for the Helios Cloud fleet.

The generative model (documented in ``docs/data-generation.md``) produces, per site and hour:

* outside temperature and relative humidity: seasonal climate + diurnal cycle + AR(1) weather,
* IT load: organic growth, product launches, business-hour and weekly cycles, nightly batch
  windows (bigger at month end), holidays, random incidents and AR(1) noise,
* PUE: free-cooling vs chiller regime driven by (thermally lagged) temperature and humidity,
  plus fixed overheads that make PUE worse at low utilisation,
* facility power = IT load x PUE,
* grid carbon intensity: site-specific mix with a solar midday dip, evening peak and AR(1)
  wind/cloud variability,
* a time-of-use electricity tariff,
* day-ahead *forecast* feeds for weather and carbon intensity with realistic errors.

Sensor faults are injected afterwards by :mod:`dcforecast.data.quality`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter
from scipy.special import expit

from dcforecast.config import Config
from dcforecast.data.calendar import local_calendar
from dcforecast.data.quality import inject_faults
from dcforecast.data.sites import Site, get_sites, site_index
from dcforecast.utils import get_logger, write_json

log = get_logger(__name__)

TELEMETRY_COLUMNS = ["it_load_kw", "facility_kw", "temp_c", "rh_pct"]


def ar1(rng: np.random.Generator, n: int, phi: float, sigma: float) -> np.ndarray:
    """Stationary AR(1) process with lag-1 autocorrelation ``phi`` and marginal std ``sigma``."""
    eps = rng.standard_normal(n) * sigma * np.sqrt(1.0 - phi**2)
    eps[0] = rng.standard_normal() * sigma
    return lfilter([1.0], [1.0, -phi], eps)


def _smooth_hinge(x: np.ndarray, width: float = 2.0) -> np.ndarray:
    """Smooth version of max(0, x) (softplus with a temperature of ``width``)."""
    return width * np.logaddexp(0.0, x / width)


def simulate_site(site: Site, timestamps: pd.DatetimeIndex, seed: int) -> pd.DataFrame:
    """Simulate the *true* (fault-free) hourly series of one site.

    Args:
        site: site parameters.
        timestamps: hourly UTC timestamps.
        seed: base seed; combined with the site index so subsets stay reproducible.
    """
    rng = np.random.default_rng([seed, site_index(site.site_id)])
    n = len(timestamps)
    cal = local_calendar(timestamps, site.utc_offset_h)
    h = cal["local_hour"].to_numpy(dtype=float)
    doy = cal["doy"].to_numpy(dtype=float)
    hours_since_start = np.arange(n, dtype=float)
    years = hours_since_start / (24 * 365.25)

    # ---------------------------------------------------------------- climate
    season = np.cos(2 * np.pi * (doy - site.temp_peak_doy) / 365.25)  # +1 at the warmest day
    diurnal = np.cos(2 * np.pi * (h - 15.0) / 24.0)  # +1 at 15:00 local
    weather = ar1(rng, n, 0.985, site.temp_noise_c)  # synoptic weather systems (days)
    temp = (
        site.temp_mean_c
        + site.temp_annual_amp_c * season
        + site.temp_diurnal_amp_c * (1.0 + 0.25 * season) * diurnal
        + weather
        + ar1(rng, n, 0.6, 0.35)
    )
    rh = (
        site.rh_mean_pct
        - site.rh_annual_amp_pct * season
        - site.rh_diurnal_amp_pct * diurnal
        - 1.2 * weather
        + ar1(rng, n, 0.95, 5.0)
    )
    rh = np.clip(rh, 4.0, 100.0)

    # ---------------------------------------------------------------- IT load
    cap = site.it_capacity_kw
    growth = (1.0 + site.annual_growth) ** years
    launch = np.ones(n)
    n_launch = max(1, rng.poisson(0.8 * years[-1]))
    for _ in range(n_launch):
        t0 = rng.uniform(0.05, 0.95) * n
        size = rng.uniform(0.02, 0.06)
        ramp_h = rng.uniform(3, 10) * 24
        launch *= 1.0 + size * expit((hours_since_start - t0) / (ramp_h / 6.0))
    utc_doy = np.asarray(timestamps.dayofyear, dtype=float)
    annual_demand = 1.0 + 0.025 * np.cos(2 * np.pi * (utc_doy - 335) / 365.25)
    level = site.base_utilization * cap * growth * launch * annual_demand

    workday = cal["is_workday"].to_numpy()
    off_day = ~workday
    business_shape = np.exp(-0.5 * ((h - 13.5) / 3.2) ** 2)
    business = site.business_amp * business_shape * np.where(workday, 1.0, 0.3)
    weekday_factor = np.where(off_day, 1.0 - site.weekend_drop, 1.0)

    hb = (h - site.batch_start_local) % 24
    batch_shape = np.where(
        hb < site.batch_hours, 0.5 * (1 - np.cos(2 * np.pi * (hb + 0.5) / site.batch_hours)), 0.0
    )
    local = timestamps + pd.Timedelta(hours=site.utc_offset_h)
    month_end = np.asarray(local.is_month_end)
    batch = site.batch_amp * batch_shape * np.where(month_end, 1.8, 1.0)

    noise = 1.0 + ar1(rng, n, 0.85, 0.012)
    it = level * weekday_factor * (1.0 + business + batch) * noise

    n_incidents = rng.poisson(4.0 * years[-1])
    for _ in range(n_incidents):
        start = int(rng.integers(0, n))
        dur = int(rng.integers(2, 9))
        depth = rng.uniform(0.12, 0.40)
        it[start : start + dur] *= 1.0 - depth
    it = np.clip(it, 0.08 * cap, 0.97 * cap)

    # ---------------------------------------------------------------- PUE
    alpha = 0.4  # thermal inertia of the building / cooling plant
    temp_eff = lfilter([alpha], [1.0, -(1.0 - alpha)], temp, zi=[(1.0 - alpha) * temp[0]])[0]
    excess = temp_eff - site.free_cooling_c
    cooling = (
        site.cooling_floor
        + site.chiller_slope * _smooth_hinge(excess)
        + site.humidity_coef * np.clip(rh - 50.0, 0.0, None) * expit(excess / 2.0)
    )
    fixed = site.fixed_overhead_frac * cap / it
    pue = np.clip(1.0 + cooling + fixed + ar1(rng, n, 0.9, 0.004), 1.03, 2.5)
    facility = it * pue

    # ---------------------------------------------------------------- grid carbon intensity
    solar_season = np.cos(2 * np.pi * (doy - (site.temp_peak_doy - 25)) / 365.25)
    sun = np.clip(np.sin(np.pi * (h + 0.5 - 6.0) / 12.0), 0.0, None) ** 1.3
    clear_sky = expit(ar1(rng, n, 0.97, 1.2) + 0.8)
    solar = sun * (0.75 + 0.25 * solar_season) * (0.45 + 0.55 * clear_sky)
    evening = np.exp(-0.5 * ((h - 19.5) / 1.8) ** 2)
    winter = -solar_season
    ci_noise = ar1(rng, n, 0.96, site.ci_noise)
    ci = (
        site.ci_base
        * (1.0 + site.ci_seasonal_amp * winter)
        * (1.0 - site.ci_solar_share * solar)
        * (1.0 + site.ci_evening_peak * evening)
        * np.exp(ci_noise)
    )
    ci = np.clip(ci, 5.0, 1200.0)

    # ---------------------------------------------------------------- tariff
    peak_window = (h >= site.peak_start_local) & (h < site.peak_end_local)
    shoulder = ((h >= 7) & (h < site.peak_start_local)) | ((h >= site.peak_end_local) & (h < 22))
    hot = season > 0.5
    peak_price = site.price_peak * np.where(hot, site.summer_peak_mult, 1.0)
    price = np.where(
        workday & peak_window,
        peak_price,
        np.where(
            (workday & shoulder) | (off_day & (h >= 17) & (h < 21)),
            site.price_mid,
            site.price_offpeak,
        ),
    )

    # ---------------------------------------------------------------- forecast feeds
    temp_fcst = temp + ar1(rng, n, 0.92, 1.3)
    rh_fcst = np.clip(rh + ar1(rng, n, 0.92, 5.0), 3.0, 100.0)
    ci_fcst = ci * np.exp(ar1(rng, n, 0.9, 0.04 + 0.3 * site.ci_noise))

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "site_id": site.site_id,
            "it_load_kw": it,
            "pue": pue,
            "facility_kw": facility,
            "temp_c": temp,
            "rh_pct": rh,
            "temp_fcst_c": temp_fcst,
            "rh_fcst_pct": rh_fcst,
            "carbon_g_per_kwh": ci,
            "carbon_fcst_g_per_kwh": ci_fcst,
            "price_usd_per_mwh": price.astype(float),
        }
    )


@dataclass
class GeneratedData:
    """All raw artefacts produced by the generator."""

    telemetry: pd.DataFrame  # metered + weather-station data WITH sensor faults (history only)
    weather_forecast: pd.DataFrame  # day-ahead weather feed (history + future)
    grid: pd.DataFrame  # carbon intensity (actual + forecast) and tariff (history + future)
    fault_log: pd.DataFrame  # ground truth of every injected fault
    holdout_future: pd.DataFrame  # true future values, hidden from models (decision audit)
    truth: pd.DataFrame  # fault-free history (for QA evaluation)
    history_end: pd.Timestamp


def generate_dataset(cfg: Config) -> GeneratedData:
    """Generate the full synthetic dataset for the configured sites."""
    sites = get_sites(cfg.data.sites)
    n_hist = cfg.data.days * 24
    n_fut = cfg.data.future_days * 24
    timestamps = pd.date_range(cfg.data.start, periods=n_hist + n_fut, freq="h")
    history_end = timestamps[n_hist - 1]
    frames = []
    for site in sites:
        frames.append(simulate_site(site, timestamps, cfg.seed))
        log.info("simulated site %-10s (%d hours)", site.site_id, len(timestamps))
    full = pd.concat(frames, ignore_index=True)
    full["site_id"] = full["site_id"].astype(str)

    hist = full[full["timestamp"] <= history_end].reset_index(drop=True)
    fut = full[full["timestamp"] > history_end].reset_index(drop=True)

    truth = hist[["timestamp", "site_id", *TELEMETRY_COLUMNS]].copy()
    # Meters add a little measurement noise and report with 0.1 precision (kW, C, %RH).
    rng = np.random.default_rng([cfg.seed, 999])
    n = len(truth)
    truth["it_load_kw"] *= 1.0 + rng.normal(0.0, 5e-4, n)
    truth["facility_kw"] *= 1.0 + rng.normal(0.0, 5e-4, n)
    truth["temp_c"] += rng.normal(0.0, 0.05, n)
    truth["rh_pct"] = np.clip(truth["rh_pct"] + rng.normal(0.0, 0.2, n), 0.0, 100.0)
    for col in TELEMETRY_COLUMNS:
        truth[col] = truth[col].round(1)
    telemetry, fault_log = inject_faults(truth, cfg.data.faults, rng, sites)

    weather_forecast = full[["timestamp", "site_id", "temp_fcst_c", "rh_fcst_pct"]].copy()
    grid = full[
        ["timestamp", "site_id", "carbon_g_per_kwh", "carbon_fcst_g_per_kwh", "price_usd_per_mwh"]
    ].copy()
    # Actual carbon intensity is only known after the fact.
    grid.loc[grid["timestamp"] > history_end, "carbon_g_per_kwh"] = np.nan
    holdout = fut[
        ["timestamp", "site_id", "it_load_kw", "pue", "facility_kw", "temp_c", "rh_pct"]
    ].copy()
    holdout["carbon_g_per_kwh"] = fut["carbon_g_per_kwh"].to_numpy()
    return GeneratedData(
        telemetry=telemetry,
        weather_forecast=weather_forecast,
        grid=grid,
        fault_log=fault_log,
        holdout_future=holdout,
        truth=truth,
        history_end=history_end,
    )


def write_raw(data: GeneratedData, cfg: Config) -> dict[str, Path]:
    """Persist raw artefacts as Parquet under ``data/raw`` and return their paths."""
    raw = cfg.raw_dir
    raw.mkdir(parents=True, exist_ok=True)
    paths = {
        "telemetry": raw / "telemetry.parquet",
        "weather_forecast": raw / "weather_forecast.parquet",
        "grid": raw / "grid.parquet",
        "fault_log": raw / "fault_log.parquet",
        "holdout_future": raw / "holdout_future.parquet",
        "truth": raw / "truth.parquet",
    }
    for key, path in paths.items():
        getattr(data, key).to_parquet(path, index=False, compression="zstd")
    sites = get_sites(cfg.data.sites)
    write_json(
        raw / "manifest.json",
        {
            "start": cfg.data.start,
            "history_end": data.history_end,
            "days": cfg.data.days,
            "future_days": cfg.data.future_days,
            "seed": cfg.seed,
            "sites": [s.to_dict() for s in sites],
            "rows": {k: len(getattr(data, k)) for k in paths},
            "faults_injected": len(data.fault_log),
        },
    )
    return paths


def run_generate(cfg: Config) -> dict[str, Path]:
    """CLI entry: generate and write the raw dataset."""
    data = generate_dataset(cfg)
    paths = write_raw(data, cfg)
    log.info(
        "wrote %d telemetry rows (%d sites), %d injected faults -> %s",
        len(data.telemetry),
        data.telemetry["site_id"].nunique(),
        len(data.fault_log),
        cfg.raw_dir.as_posix(),
    )
    return paths
