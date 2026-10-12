"""``dcforecast prepare``: clean raw telemetry, join external feeds, build features.

Outputs (all gitignored):

* ``data/processed/panel.parquet``    -- clean hourly history per site (target + covariates)
* ``data/processed/future.parquet``   -- known covariates for the hours after the history
* ``data/processed/features.parquet`` -- the tabular feature matrix (for inspection / reuse)
* ``reports/qa_report.json``          -- data-quality report
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from dcforecast.config import Config
from dcforecast.data.quality import clean_telemetry, qa_report
from dcforecast.data.sites import get_sites
from dcforecast.features import build_features
from dcforecast.utils import get_logger, read_json, write_json

log = get_logger(__name__)

EXOG_COLUMNS = [
    "temp_fcst_c",
    "rh_fcst_pct",
    "carbon_g_per_kwh",
    "carbon_fcst_g_per_kwh",
    "price_usd_per_mwh",
]


class DataNotFoundError(FileNotFoundError):
    """Raised when a pipeline step needs data that has not been generated yet."""


@dataclass
class PanelData:
    """Clean history plus the known-future covariates."""

    history: pd.DataFrame
    future: pd.DataFrame
    history_end: pd.Timestamp

    @property
    def sites(self) -> list[str]:
        return list(dict.fromkeys(self.history["site_id"]))

    def combined(self) -> pd.DataFrame:
        """History and future rows in one frame (future target values are NaN)."""
        return pd.concat([self.history, self.future], ignore_index=True).sort_values(
            ["site_id", "timestamp"], ignore_index=True
        )


def _require(path: Path, hint: str) -> Path:
    if not path.exists():
        raise DataNotFoundError(f"Missing {path.as_posix()} -- run `dcforecast {hint}` first")
    return path


def load_manifest(cfg: Config) -> dict:
    """Read the generator manifest (start, end, sites)."""
    return read_json(_require(cfg.raw_dir / "manifest.json", "generate"))


def prepare(cfg: Config) -> dict[str, Path]:
    """Clean telemetry, merge weather/grid feeds, write the processed panel and QA report."""
    manifest = load_manifest(cfg)
    site_ids = [s["site_id"] for s in manifest["sites"]]
    sites = get_sites(site_ids)
    start = pd.Timestamp(manifest["start"])
    end = pd.Timestamp(manifest["history_end"])
    raw = cfg.raw_dir
    telemetry = pd.read_parquet(_require(raw / "telemetry.parquet", "generate"))
    fault_log = pd.read_parquet(raw / "fault_log.parquet")
    weather = pd.read_parquet(raw / "weather_forecast.parquet")
    grid = pd.read_parquet(raw / "grid.parquet")

    result = clean_telemetry(telemetry, sites, start, end)
    feeds = weather.merge(grid, on=["timestamp", "site_id"], how="outer")
    history = result.clean.merge(feeds, on=["timestamp", "site_id"], how="left")
    future = feeds[feeds["timestamp"] > end].copy()
    future = future.sort_values(["site_id", "timestamp"], ignore_index=True)
    if history[EXOG_COLUMNS[:2]].isna().any().any():
        raise ValueError("Weather forecast feed has gaps inside the history window")

    out_dir = cfg.processed_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "panel": out_dir / "panel.parquet",
        "future": out_dir / "future.parquet",
        "features": out_dir / "features.parquet",
        "qa_report": cfg.reports_dir / "qa_report.json",
    }
    history.to_parquet(paths["panel"], index=False, compression="zstd")
    future.to_parquet(paths["future"], index=False, compression="zstd")

    feats, names = build_features(history, cfg)
    keep = ["timestamp", "site_id", cfg.forecast.target, "level", "y_rel", *names]
    feats[keep].astype({n: "float32" for n in names if feats[n].dtype.kind == "f"}).to_parquet(
        paths["features"], index=False, compression="zstd"
    )

    qa = qa_report(result, telemetry, fault_log)
    qa["history_start"] = start
    qa["history_end"] = end
    qa["n_features"] = len(names)
    qa["feature_names"] = names
    write_json(paths["qa_report"], qa)
    det = qa["detection"]
    log.info(
        "cleaned %d rows: %d missing, %d stuck, %d spikes flagged (precision %.3f, spike recall "
        "%.3f, stuck recall %.3f); facility imputed %.2f%%",
        qa["totals"]["rows_expected"],
        qa["totals"]["missing"],
        qa["totals"]["stuck"],
        qa["totals"]["spike"],
        det.get("precision") or 0.0,
        det.get("spike_recall") or 0.0,
        det.get("stuck_recall") or 0.0,
        qa["facility_imputed_pct"],
    )
    return paths


def load_panel(cfg: Config) -> PanelData:
    """Load the processed panel produced by :func:`prepare`."""
    hist_path = _require(cfg.processed_dir / "panel.parquet", "prepare")
    history = pd.read_parquet(hist_path)
    future = pd.read_parquet(_require(cfg.processed_dir / "future.parquet", "prepare"))
    history["site_id"] = history["site_id"].astype(str)
    future["site_id"] = future["site_id"].astype(str)
    history = history.sort_values(["site_id", "timestamp"], ignore_index=True)
    return PanelData(history=history, future=future, history_end=history["timestamp"].max())
