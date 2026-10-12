"""Forecasting models behind one interface (see :class:`~dcforecast.models.base.Forecaster`)."""

from __future__ import annotations

import json
from pathlib import Path

from dcforecast.config import Config
from dcforecast.models.base import Forecaster
from dcforecast.models.baselines import SeasonalNaive24, SeasonalNaive168
from dcforecast.models.ets import ETSForecaster
from dcforecast.models.gbm import GBMForecaster

MODEL_NAMES = ["naive24", "naive168", "ets", "gbm", "lstm", "tcn"]
DEEP_MODELS = {"lstm", "tcn"}


def model_class(name: str) -> type[Forecaster]:
    """Resolve a model name to its class (deep models are imported lazily)."""
    if name == "naive24":
        return SeasonalNaive24
    if name == "naive168":
        return SeasonalNaive168
    if name == "ets":
        return ETSForecaster
    if name == "gbm":
        return GBMForecaster
    if name in DEEP_MODELS:
        from dcforecast.models.deep.forecaster import LSTMForecaster, TCNModelForecaster

        return LSTMForecaster if name == "lstm" else TCNModelForecaster
    raise KeyError(f"Unknown model '{name}'. Choose from: {', '.join(MODEL_NAMES)}")


def create_model(name: str, cfg: Config) -> Forecaster:
    """Instantiate an unfitted model by name."""
    return model_class(name)(cfg)


def load_model(path: Path, device: str | None = None) -> Forecaster:
    """Load any saved model directory (dispatches on ``meta.json``)."""
    with (Path(path) / "meta.json").open(encoding="utf-8") as fh:
        name = json.load(fh)["name"]
    return model_class(name).load(Path(path), device=device)


__all__ = ["DEEP_MODELS", "MODEL_NAMES", "Forecaster", "create_model", "load_model", "model_class"]
