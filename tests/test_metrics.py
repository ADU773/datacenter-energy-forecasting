"""Forecast metrics and the rolling-origin fold layout."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from dcforecast.evaluation import metrics as m
from dcforecast.evaluation.backtest import make_folds

Y = np.array([100.0, 200.0, 300.0, 400.0])
P = np.array([110.0, 190.0, 330.0, 400.0])


def test_point_metrics_known_values() -> None:
    assert m.mae(Y, P) == pytest.approx(12.5)
    assert m.rmse(Y, P) == pytest.approx(np.sqrt((100 + 100 + 900 + 0) / 4))
    assert m.mape(Y, P) == pytest.approx((10 + 5 + 10 + 0) / 4)
    smape = np.mean(2 * np.abs(Y - P) / (np.abs(Y) + np.abs(P))) * 100
    assert m.smape(Y, P) == pytest.approx(smape)


def test_pinball_and_coverage() -> None:
    # under-forecast by 10 everywhere: loss = q * 10
    assert m.pinball(Y, Y - 10, 0.9) == pytest.approx(9.0)
    assert m.pinball(Y, Y - 10, 0.1) == pytest.approx(1.0)
    # over-forecast by 10: loss = (1 - q) * 10
    assert m.pinball(Y, Y + 10, 0.9) == pytest.approx(1.0)
    assert m.mean_pinball(Y, {0.1: Y, 0.5: Y, 0.9: Y}) == 0.0
    assert m.coverage(Y, Y - 1, Y + 1) == 1.0
    assert m.coverage(Y, Y + 1, Y + 2) == 0.0


def test_naive_scale_and_mase() -> None:
    t = pd.date_range("2024-01-01", periods=24 * 10, freq="h")
    y = 1000 + 100 * np.sin(2 * np.pi * np.arange(len(t)) / 24) + np.arange(len(t))
    hist = pd.DataFrame({"timestamp": t, "site_id": "a", "facility_kw": y})
    scale = m.naive_scale(hist, "facility_kw", season=24)
    assert scale["a"] == pytest.approx(24.0)  # the trend adds exactly 24 per day
    frame = pd.DataFrame(
        {
            "site_id": "a",
            "facility_kw": y[:48],
            "p10": y[:48] - 50,
            "p50": y[:48] + 12,
            "p90": y[:48] + 50,
            "scale": scale["a"],
        }
    )
    out = m.evaluate(frame, "facility_kw")
    assert out["mae"] == pytest.approx(12.0)
    assert out["mase"] == pytest.approx(0.5)
    assert out["coverage_80"] == 1.0


def test_folds_are_consecutive_and_leak_free() -> None:
    end = pd.Timestamp("2025-12-31 23:00")
    folds = make_folds(end, folds=4, test_days=28, horizon=24)
    assert [f.k for f in folds] == [1, 2, 3, 4]
    assert folds[-1].test_end == end
    for a, b in pairwise(folds):
        assert b.test_start == a.test_end + pd.Timedelta(hours=1)
    for f in folds:
        assert f.train_end < f.test_start
        assert (f.test_end - f.test_start) == pd.Timedelta(days=28) - pd.Timedelta(hours=1)
        assert all(f.train_end <= o and o + pd.Timedelta(hours=24) <= f.test_end for o in f.origins)
        assert len(f.origins) == 28
