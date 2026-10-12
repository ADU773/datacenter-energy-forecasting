"""Shared chart styling: one validated categorical palette, recessive chrome, fixed entity colors.

Colors follow the entity (a site or a model always has the same hue), never its rank. The
categorical order is a colorblind-validated reference palette (adjacent CVD separation >= 9).
"""

from __future__ import annotations

from dcforecast.data.sites import SITE_ORDER

CATEGORICAL = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
]
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
DE_EMPHASIS = "#c3c2b7"

MODEL_ORDER = ["naive24", "naive168", "ets", "gbm", "lstm", "tcn"]
MODEL_LABELS = {
    "naive24": "Seasonal naive 24h",
    "naive168": "Seasonal naive 168h",
    "ets": "Holt-Winters (ETS)",
    "gbm": "Gradient boosting",
    "lstm": "LSTM enc-dec",
    "tcn": "TCN",
}
MODE_ORDER = ["cpu", "cuda_fp32", "cuda_amp"]
MODE_LABELS = {"cpu": "CPU", "cuda_fp32": "GPU fp32", "cuda_amp": "GPU AMP (fp16)"}


def site_color(site_id: str) -> str:
    return CATEGORICAL[SITE_ORDER.index(site_id) % len(CATEGORICAL)]


def model_color(name: str) -> str:
    idx = MODEL_ORDER.index(name) if name in MODEL_ORDER else len(MODEL_ORDER)
    return CATEGORICAL[idx % len(CATEGORICAL)]


def mode_color(mode: str) -> str:
    return CATEGORICAL[MODE_ORDER.index(mode) % len(CATEGORICAL)]


def model_label(name: str) -> str:
    return MODEL_LABELS.get(name, name)
