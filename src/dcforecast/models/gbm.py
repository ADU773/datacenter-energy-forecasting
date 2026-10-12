"""Gradient boosting with quantile loss (sklearn ``HistGradientBoostingRegressor``).

One global model per quantile (P10, P50, P90) across all sites (``site_code`` is a categorical
feature). The target is the load relative to its trailing weekly level (``y_rel``), which
removes the trend so the trees never have to extrapolate growth. Because all target-derived
features are lagged by at least the horizon, a single direct model serves every horizon
1..24 h without leakage.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from dcforecast.features import build_features
from dcforecast.models.base import Forecaster, finalize_quantiles, origin_grid
from dcforecast.utils import get_logger

log = get_logger(__name__)

QUANTILES = (0.1, 0.5, 0.9)


class GBMForecaster(Forecaster):
    """Global quantile gradient-boosting model on leakage-safe tabular features."""

    name = "gbm"

    def __init__(self, cfg) -> None:  # type: ignore[no-untyped-def]
        super().__init__(cfg)
        self.models: dict[float, HistGradientBoostingRegressor] = {}
        self.features_: list[str] = []

    def fit(self, history: pd.DataFrame, device: str | None = None) -> GBMForecaster:
        g = self.cfg.gbm
        self.sites_ = list(dict.fromkeys(history["site_id"]))
        feats, names = build_features(history, self.cfg)
        self.features_ = names
        mask = feats["y_rel"].notna() & feats["level"].notna()
        if "facility_imputed" in feats.columns:
            mask &= ~feats["facility_imputed"].astype(bool)  # never learn from imputed targets
        train = feats.loc[mask]
        if g.max_train_rows and len(train) > g.max_train_rows:
            train = train.sample(g.max_train_rows, random_state=self.cfg.seed)
        X = train[names].to_numpy(dtype=np.float32)
        y = train["y_rel"].to_numpy(dtype=np.float32)
        cat_mask = np.array([n == "site_code" for n in names])
        for q in QUANTILES:
            model = HistGradientBoostingRegressor(
                loss="quantile",
                quantile=q,
                max_iter=g.max_iter,
                learning_rate=g.learning_rate,
                max_leaf_nodes=g.max_leaf_nodes,
                min_samples_leaf=g.min_samples_leaf,
                l2_regularization=g.l2_regularization,
                categorical_features=cat_mask,
                early_stopping=False,
                random_state=self.cfg.seed,
            )
            model.fit(X, y)
            self.models[q] = model
        self.fit_info_ = {"rows": len(train), "features": len(names), "quantiles": list(QUANTILES)}
        log.info("gbm fitted on %d rows x %d features", len(train), len(names))
        return self

    def predict(self, panel: pd.DataFrame, origins: Sequence[pd.Timestamp]) -> pd.DataFrame:
        last_needed = pd.DatetimeIndex(origins).max() + pd.Timedelta(hours=self.horizon)
        first_needed = pd.DatetimeIndex(origins).min() - pd.Timedelta(days=9)
        window = panel[(panel["timestamp"] >= first_needed) & (panel["timestamp"] <= last_needed)]
        window = window[window["site_id"].isin(self.sites_)]
        feats, _ = build_features(window, self.cfg)
        grid = origin_grid(self.sites_, origins, self.horizon)
        rows = grid.merge(
            feats[["site_id", "timestamp", "level", *self.features_]],
            on=["site_id", "timestamp"],
            how="left",
        )
        X = rows[self.features_].to_numpy(dtype=np.float32)
        level = rows["level"].to_numpy(dtype=float)
        for q, col in zip(QUANTILES, ("p10", "p50", "p90"), strict=True):
            rows[col] = self.models[q].predict(X) * level
        return finalize_quantiles(rows.dropna(subset=["p50"]))

    def _save_payload(self, path: Path) -> None:
        joblib.dump({"models": self.models, "features": self.features_}, path / "gbm.joblib")

    def _load_payload(self, path: Path, device: str | None = None) -> None:
        payload = joblib.load(path / "gbm.joblib")
        self.models = payload["models"]
        self.features_ = list(payload["features"])
