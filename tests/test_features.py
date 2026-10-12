"""Leakage-safety of the tabular features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dcforecast.config import Config
from dcforecast.data.prepare import PanelData
from dcforecast.features import add_calendar, build_features, fourier_terms


def test_feature_matrix_shape_and_columns(tiny_cfg: Config, tiny_data: PanelData) -> None:
    df, names = build_features(tiny_data.history, tiny_cfg)
    assert len(df) == len(tiny_data.history)
    assert len(names) == len(set(names))
    for col in (
        "site_code",
        "is_holiday",
        "lag24_rel",
        "lag168_rel",
        "roll_mean24_rel",
        "temp_fcst_c",
    ):
        assert col in names
    # the raw target, measured weather and the label are never features
    assert not {"facility_kw", "it_load_kw", "temp_c", "pue", "y_rel"} & set(names)
    valid = df.dropna(subset=["lag168_rel", "roll_std168_rel"])
    assert np.isfinite(valid[names].to_numpy(dtype=float)).all()


def test_features_only_use_information_older_than_the_horizon(
    tiny_cfg: Config, tiny_data: PanelData
) -> None:
    """Perturbing the target at time T must not change any feature before T + horizon."""
    h = tiny_cfg.forecast.horizon
    hist = tiny_data.history.sort_values(["site_id", "timestamp"]).reset_index(drop=True)
    base, names = build_features(hist, tiny_cfg)
    cut = hist["timestamp"].iloc[len(hist) // 4]
    changed = hist.copy()
    later = changed["timestamp"] >= cut
    changed.loc[later, "facility_kw"] *= 3.0
    changed.loc[later, "pue"] += 0.5
    changed.loc[later, "temp_c"] += 20
    pert, _ = build_features(changed, tiny_cfg)
    safe = base["timestamp"] < cut + pd.Timedelta(hours=h)
    pd.testing.assert_frame_equal(base.loc[safe, names], pert.loc[safe, names])
    # ...and it does change features afterwards (the test is not vacuous)
    after = base["timestamp"] >= cut + pd.Timedelta(hours=h + 1)
    assert not np.allclose(
        base.loc[after, "lag24_rel"].to_numpy(dtype=float),
        pert.loc[after, "lag24_rel"].to_numpy(dtype=float),
        equal_nan=True,
    )


def test_fourier_terms_are_bounded_and_periodic(tiny_data: PanelData) -> None:
    panel = add_calendar(
        tiny_data.history[tiny_data.history["site_id"] == "frostholm"].reset_index(drop=True)
    )
    f = fourier_terms(panel, daily=2, weekly=1, yearly=1)
    assert f.shape[1] == 2 * (2 + 1 + 1)
    assert f.abs().max().max() <= 1.0 + 1e-9
    np.testing.assert_allclose(
        f["fd_sin1"].iloc[:24].to_numpy(), f["fd_sin1"].iloc[24:48].to_numpy()
    )
