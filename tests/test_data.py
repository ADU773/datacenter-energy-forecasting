"""Synthetic generator physics, fault injection, detection, imputation and the QA report."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dcforecast.config import Config
from dcforecast.data.calendar import holiday_dates, local_calendar
from dcforecast.data.generate import ar1, simulate_site
from dcforecast.data.prepare import PanelData
from dcforecast.data.quality import (
    R_MISSING,
    R_OK,
    R_RANGE,
    R_SPIKE,
    R_STUCK,
    detect_anomalies,
    seasonal_fill,
)
from dcforecast.data.sites import SITES, get_sites
from dcforecast.utils import read_json

YEAR = pd.date_range("2024-01-01", periods=366 * 24, freq="h")


@pytest.fixture(scope="module")
def desert() -> pd.DataFrame:
    return simulate_site(SITES["dunehaven"], YEAR, seed=7)


def test_six_fictional_sites_with_distinct_climates() -> None:
    sites = get_sites(None)
    assert len(sites) == 6
    assert len({s.climate for s in sites}) == 6
    with pytest.raises(KeyError):
        get_sites(["atlantis"])


def test_facility_power_is_it_load_times_pue(desert: pd.DataFrame) -> None:
    np.testing.assert_allclose(
        desert["facility_kw"], desert["it_load_kw"] * desert["pue"], rtol=1e-6
    )
    assert desert["pue"].between(1.0, 2.5).all()
    assert (desert["it_load_kw"] > 0).all()
    assert desert[["carbon_g_per_kwh", "price_usd_per_mwh"]].gt(0).all().all()


def test_pue_rises_in_the_chiller_regime(desert: pd.DataFrame) -> None:
    site = SITES["dunehaven"]
    cool = desert.loc[desert["temp_c"] < site.free_cooling_c - 3, "pue"].mean()
    hot = desert.loc[desert["temp_c"] > site.free_cooling_c + 10, "pue"].mean()
    assert hot > cool + 0.05


def test_load_has_weekly_cycle_and_growth(desert: pd.DataFrame) -> None:
    cal = local_calendar(desert["timestamp"], SITES["dunehaven"].utc_offset_h)
    workday = desert.loc[cal["is_workday"].to_numpy() == 1, "it_load_kw"].mean()
    weekend = desert.loc[cal["is_weekend"].to_numpy() == 1, "it_load_kw"].mean()
    assert workday > weekend
    first, last = (
        desert["it_load_kw"].iloc[: 30 * 24].mean(),
        desert["it_load_kw"].iloc[-30 * 24 :].mean(),
    )
    assert last > first  # long-term growth


def test_solar_dip_in_carbon_intensity(desert: pd.DataFrame) -> None:
    local_hour = (desert["timestamp"].dt.hour + SITES["dunehaven"].utc_offset_h) % 24
    by_hour = desert.groupby(local_hour)["carbon_g_per_kwh"].mean()
    assert by_hour.loc[11:14].mean() < by_hour.loc[19:22].mean()


def test_generator_is_deterministic() -> None:
    idx = YEAR[: 24 * 14]
    a = simulate_site(SITES["frostholm"], idx, seed=3)
    b = simulate_site(SITES["frostholm"], idx, seed=3)
    c = simulate_site(SITES["frostholm"], idx, seed=4)
    pd.testing.assert_frame_equal(a, b)
    assert not np.allclose(a["temp_c"], c["temp_c"])


def test_ar1_noise_is_autocorrelated() -> None:
    x = ar1(np.random.default_rng(0), 20_000, phi=0.9, sigma=1.0)
    lag1 = np.corrcoef(x[:-1], x[1:])[0, 1]
    assert 0.85 < lag1 < 0.95


def test_holidays_cover_new_year_and_christmas() -> None:
    days = holiday_dates([2024])
    assert pd.Timestamp("2024-01-01").date() in days
    assert pd.Timestamp("2024-12-25").date() in days


def test_detect_anomalies_finds_each_fault_type() -> None:
    rng = np.random.default_rng(0)
    t = np.arange(24 * 30)
    x = 5000 + 800 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 30, len(t))
    x[100] = np.nan
    x[200] = 9_000.0  # single-hour surge
    x[300:310] = x[300]  # stuck meter for 10 h
    x[400] = -5.0  # physically impossible
    reason = detect_anomalies(x, "facility_kw", capacity_kw=10_000)
    assert reason[100] == R_MISSING
    assert reason[200] == R_SPIKE
    assert (reason[301:310] == R_STUCK).all()
    assert reason[400] == R_RANGE
    false_alarms = np.flatnonzero(reason != R_OK)
    assert len(set(false_alarms) - {100, 200, 400, *range(300, 310)}) <= 2


def test_seasonal_fill_is_causal_and_complete() -> None:
    t = np.arange(24 * 21)
    y = 100 + 10 * np.sin(2 * np.pi * t / 24)
    gappy = y.copy()
    gappy[400:420] = np.nan
    filled = seasonal_fill(gappy, period=24, multiplicative=False)
    assert not np.isnan(filled).any()
    np.testing.assert_allclose(filled[400:420], y[400:420], atol=1.0)
    # causal: changing the future after the gap does not change the imputed values
    later = gappy.copy()
    later[430:] += 50
    np.testing.assert_allclose(seasonal_fill(later, 24, False)[400:420], filled[400:420])


def test_cleaned_panel_and_qa_report(tiny_cfg: Config, tiny_data: PanelData) -> None:
    hist = tiny_data.history
    n_sites = len(tiny_cfg.data.sites or [])
    assert len(hist) == tiny_cfg.data.days * 24 * n_sites  # outage rows were restored
    assert hist[["it_load_kw", "facility_kw", "temp_c", "rh_pct"]].notna().all().all()
    assert not hist.duplicated(["site_id", "timestamp"]).any()
    assert len(tiny_data.future) == tiny_cfg.data.future_days * 24 * n_sites
    # the known-future feed carries covariates only -- no target or measured values leak in
    assert not {"facility_kw", "it_load_kw", "temp_c", "pue"} & set(tiny_data.future.columns)
    assert {"temp_fcst_c", "carbon_fcst_g_per_kwh", "price_usd_per_mwh"} <= set(
        tiny_data.future.columns
    )
    qa = read_json(tiny_cfg.reports_dir / "qa_report.json")
    assert qa["detection"]["precision"] > 0.9
    assert qa["detection"]["spike_recall"] > 0.8
    assert qa["detection"]["stuck_recall"] > 0.8
    assert qa["imputation_error"]["facility_kw"]["mape_pct"] < 10
    assert (tiny_cfg.raw_dir / "telemetry.parquet").exists()
