"""Every model behind the common fit / predict / save / load interface."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dcforecast.config import Config
from dcforecast.data.prepare import PanelData
from dcforecast.models import MODEL_NAMES, create_model, load_model
from dcforecast.models.base import FORECAST_COLUMNS, Forecaster

MEASURED = ["facility_kw", "it_load_kw", "pue", "temp_c", "rh_pct", "carbon_g_per_kwh"]


@pytest.fixture(scope="module")
def split(tiny_data: PanelData) -> tuple[pd.DataFrame, pd.DataFrame, list[pd.Timestamp]]:
    hist = tiny_data.history
    train_end = tiny_data.history_end - pd.Timedelta(days=7)
    train = hist[hist["timestamp"] <= train_end]
    origins = [train_end, train_end + pd.Timedelta(days=2)]
    return train, hist, origins


@pytest.fixture(scope="module")
def fitted(tiny_cfg: Config, split: tuple) -> dict[str, Forecaster]:
    train, _hist, _origins = split
    return {name: create_model(name, tiny_cfg).fit(train, device="cpu") for name in MODEL_NAMES}


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_predict_contract(
    name: str, fitted: dict[str, Forecaster], split: tuple, tiny_cfg: Config
) -> None:
    _train, hist, origins = split
    fc = fitted[name].predict(hist, origins)
    h = tiny_cfg.forecast.horizon
    n_sites = hist["site_id"].nunique()
    assert list(fc.columns) == FORECAST_COLUMNS
    assert len(fc) == n_sites * len(origins) * h
    assert sorted(fc["horizon"].unique()) == list(range(1, h + 1))
    q = fc[["p10", "p50", "p90"]].to_numpy()
    assert np.isfinite(q).all()
    assert (q[:, 0] <= q[:, 1]).all() and (q[:, 1] <= q[:, 2]).all()  # no quantile crossing
    # forecasts are in the right ballpark (facility power of a 5-15 MW site)
    actual = fc.merge(hist, on=["site_id", "timestamp"])["facility_kw"]
    mape = float(np.mean(np.abs(fc["p50"].to_numpy() - actual.to_numpy()) / actual.to_numpy()))
    assert mape < 0.25


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_no_lookahead_at_prediction_time(
    name: str, fitted: dict[str, Forecaster], split: tuple
) -> None:
    """Corrupting measurements after the forecast origin must not change the forecast."""
    _train, hist, origins = split
    origin = origins[-1]
    before = fitted[name].predict(hist, [origin])
    corrupted = hist.copy()
    later = corrupted["timestamp"] > origin
    for col in MEASURED:
        corrupted.loc[later, col] = corrupted.loc[later, col] * 3.0 + 7.0
    after = fitted[name].predict(corrupted, [origin])
    pd.testing.assert_frame_equal(before, after)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_save_load_roundtrip(
    name: str, fitted: dict[str, Forecaster], split: tuple, tmp_path: Path
) -> None:
    _train, hist, origins = split
    model = fitted[name]
    model.save(tmp_path / name)
    assert (tmp_path / name / "meta.json").exists()
    loaded = load_model(tmp_path / name, device="cpu")
    assert type(loaded) is type(model)
    pd.testing.assert_frame_equal(
        model.predict(hist, origins), loaded.predict(hist, origins), atol=1e-3
    )


def test_seasonal_naive_repeats_the_past(fitted: dict[str, Forecaster], split: tuple) -> None:
    _train, hist, origins = split
    fc = fitted["naive24"].predict(hist, origins[:1])
    lookup = hist.set_index(["site_id", "timestamp"])["facility_kw"]
    past = [
        lookup[(s, t - pd.Timedelta(hours=24))]
        for s, t in zip(fc["site_id"], fc["timestamp"], strict=True)
    ]
    np.testing.assert_allclose(fc["p50"].to_numpy(), np.asarray(past), rtol=1e-6)


def test_unknown_model_name(tiny_cfg: Config) -> None:
    with pytest.raises(KeyError, match="Unknown model"):
        create_model("prophet", tiny_cfg)
