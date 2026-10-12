"""Typed configuration loaded from YAML, with environment-variable and CLI overrides.

Resolution order (later wins):

1. packaged profile (``default`` or ``tiny``) or an explicit YAML path,
2. environment variables ``DCF__SECTION__KEY=value`` (double underscores separate levels),
3. ``--set section.key=value`` CLI overrides.

Values from (2) and (3) are parsed with YAML rules, so ``3`` becomes an int, ``true`` a bool
and ``[a, b]`` a list.
"""

from __future__ import annotations

import copy
import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from importlib import resources
from pathlib import Path
from typing import Any, get_type_hints

import yaml

ENV_PREFIX = "DCF__"
PROFILES = ("default", "tiny")


class ConfigError(ValueError):
    """Raised for invalid configuration files or overrides."""


@dataclass
class PathsConfig:
    """Where generated data, runs, reports and the static site live (relative to the CWD)."""

    data_dir: str = "data"
    runs_dir: str = "runs"
    reports_dir: str = "reports"
    docs_dir: str = "docs"


@dataclass
class FaultConfig:
    """Rates of injected sensor problems (per site, per metered column)."""

    gap_episodes_per_year: float = 10.0
    gap_max_hours: int = 18
    isolated_missing_rate: float = 0.002
    spike_rate: float = 0.0015
    stuck_episodes_per_year: float = 4.0
    stuck_max_hours: int = 36


@dataclass
class DataConfig:
    """Synthetic dataset settings."""

    start: str = "2023-01-01"
    days: int = 1096
    future_days: int = 7
    sites: list[str] | None = None
    faults: FaultConfig = field(default_factory=FaultConfig)


@dataclass
class ForecastConfig:
    """The forecasting task itself."""

    target: str = "facility_kw"
    horizon: int = 24
    quantiles: list[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])


@dataclass
class FeatureConfig:
    """Leakage-safe feature engineering for the tabular (GBM) model."""

    lags: list[int] = field(default_factory=lambda: [24, 48, 168])
    rolling_windows: list[int] = field(default_factory=lambda: [24, 168])
    fourier_daily: int = 3
    fourier_weekly: int = 2
    fourier_yearly: int = 2


@dataclass
class GBMConfig:
    """sklearn HistGradientBoostingRegressor (one model per quantile)."""

    max_iter: int = 250
    learning_rate: float = 0.06
    max_leaf_nodes: int = 48
    min_samples_leaf: int = 40
    l2_regularization: float = 0.1
    max_train_rows: int | None = None


@dataclass
class ETSConfig:
    """statsmodels Holt-Winters, refit on a trailing window."""

    window_days: int = 42
    seasonal_periods: int = 168
    calibration_origins: int = 14
    sites: list[str] | None = None


@dataclass
class DeepConfig:
    """Global multi-site PyTorch models (LSTM encoder-decoder and TCN)."""

    lookback: int = 168
    decoder_lags: list[int] = field(default_factory=lambda: [24, 168])
    hidden_size: int = 64
    num_layers: int = 2
    embedding_dim: int = 8
    dropout: float = 0.1
    tcn_channels: int = 48
    tcn_kernel_size: int = 3
    tcn_levels: int = 6
    epochs: int = 25
    patience: int = 4
    lr: float = 3e-3
    weight_decay: float = 1e-4
    scheduler: str = "onecycle"
    grad_clip: float = 1.0
    batch_size: int | str = "auto"
    min_batch_size: int = 64
    max_batch_size: int = 1024
    vram_budget_mb: int = 1024
    vram_budget_fraction: float = 0.5
    samples_per_epoch: int = 60000
    val_days: int = 28
    amp: bool = True
    num_workers: int | str = "auto"
    pin_memory: bool = True
    deterministic: bool = False
    device: str = "auto"


@dataclass
class BacktestConfig:
    """Rolling-origin evaluation."""

    folds: int = 4
    test_days: int = 28
    models: list[str] = field(
        default_factory=lambda: ["naive24", "naive168", "ets", "gbm", "lstm", "tcn"]
    )
    deep_warm_start: bool = True
    warm_start_epochs: int = 8
    selection_metric: str = "mase"


@dataclass
class TrainConfig:
    """Final models trained on all history and registered."""

    models: list[str] = field(
        default_factory=lambda: ["naive24", "naive168", "ets", "gbm", "lstm", "tcn"]
    )
    warm_start: bool = True


@dataclass
class BenchmarkConfig:
    """CPU vs GPU training / inference benchmark for the deep models."""

    models: list[str] = field(default_factory=lambda: ["lstm", "tcn"])
    modes: list[str] = field(default_factory=lambda: ["cpu", "cuda_fp32", "cuda_amp"])
    gpu_train_samples: int = 32768
    cpu_train_samples: int = 4096
    epochs: int = 2
    batch_size: int = 1024
    cpu_batch_size: int = 256
    inference_batch_size: int = 256
    inference_repeats: int = 40
    cpu_threads: int = 8
    sample_interval_s: float = 0.25


@dataclass
class DecisionConfig:
    """Energy / cost / carbon roll-up and carbon-aware scheduling."""

    horizon_hours: int = 72
    deferrable_mwh: float = 120.0
    objective: str = "carbon"
    carbon_price_usd_per_t: float = 85.0
    portable_fraction: float = 0.5
    pool_rate_multiple: float = 3.0
    baseline_hours: int = 12


@dataclass
class ServeConfig:
    """FastAPI service."""

    host: str = "127.0.0.1"
    port: int = 8000
    device: str = "cpu"


@dataclass
class Config:
    """Top-level configuration object."""

    seed: int = 42
    profile: str = "default"
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    forecast: ForecastConfig = field(default_factory=ForecastConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    gbm: GBMConfig = field(default_factory=GBMConfig)
    ets: ETSConfig = field(default_factory=ETSConfig)
    deep: DeepConfig = field(default_factory=DeepConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    serve: ServeConfig = field(default_factory=ServeConfig)

    # ------------------------------------------------------------------ paths
    @property
    def data_dir(self) -> Path:
        return Path(self.paths.data_dir)

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def runs_dir(self) -> Path:
        return Path(self.paths.runs_dir)

    @property
    def reports_dir(self) -> Path:
        return Path(self.paths.reports_dir)

    @property
    def docs_dir(self) -> Path:
        return Path(self.paths.docs_dir)

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict view (for logging into the experiment tracker)."""
        return asdict(self)


# ---------------------------------------------------------------------- loading
def _from_dict(cls: type, data: dict[str, Any], where: str = "") -> Any:
    if not isinstance(data, dict):
        raise ConfigError(f"Expected a mapping for section '{where or 'root'}', got {data!r}")
    hints = get_type_hints(cls)
    names = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - names)
    if unknown:
        raise ConfigError(f"Unknown config key(s) in '{where or 'root'}': {', '.join(unknown)}")
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        ftype = hints[f.name]
        if is_dataclass(ftype) and isinstance(ftype, type):
            value = _from_dict(ftype, value or {}, f"{where}.{f.name}".strip("."))
        kwargs[f.name] = value
    return cls(**kwargs)


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _set_dotted(tree: dict[str, Any], dotted: str, value: Any) -> None:
    parts = [p for p in dotted.split(".") if p]
    if not parts:
        raise ConfigError(f"Empty override key in '{dotted}'")
    node = tree
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise ConfigError(f"Cannot set '{dotted}': '{part}' is not a section")
    node[parts[-1]] = value


def parse_override(text: str) -> tuple[str, Any]:
    """Parse ``section.key=value`` into a key and a YAML-typed value."""
    if "=" not in text:
        raise ConfigError(f"Override must look like section.key=value, got '{text}'")
    key, raw = text.split("=", 1)
    try:
        value = yaml.safe_load(raw) if raw.strip() else None
    except yaml.YAMLError as exc:  # pragma: no cover - defensive
        raise ConfigError(f"Cannot parse value in override '{text}': {exc}") from exc
    return key.strip(), value


def env_overrides(environ: dict[str, str] | None = None) -> list[tuple[str, Any]]:
    """Collect ``DCF__SECTION__KEY`` environment overrides."""
    env = os.environ if environ is None else environ
    out: list[tuple[str, Any]] = []
    for name, raw in sorted(env.items()):
        if not name.startswith(ENV_PREFIX):
            continue
        dotted = ".".join(p.lower() for p in name[len(ENV_PREFIX) :].split("__") if p)
        out.append(parse_override(f"{dotted}={raw}"))
    return out


def _read_profile(name_or_path: str) -> tuple[dict[str, Any], str]:
    if name_or_path in PROFILES:
        text = (
            resources.files("dcforecast.configs")
            .joinpath(f"{name_or_path}.yaml")
            .read_text(encoding="utf-8")
        )
        return yaml.safe_load(text) or {}, name_or_path
    path = Path(name_or_path)
    if not path.is_file():
        raise ConfigError(
            f"Config '{name_or_path}' is neither a profile ({', '.join(PROFILES)}) nor a file"
        )
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data, path.stem


def load_config(
    name_or_path: str | None = None,
    overrides: list[str] | None = None,
    environ: dict[str, str] | None = None,
) -> Config:
    """Load a configuration profile/file and apply env + CLI overrides.

    Args:
        name_or_path: ``"default"``, ``"tiny"`` or a path to a YAML file. Falls back to the
            ``DCF_CONFIG`` environment variable and then to ``"default"``.
        overrides: ``section.key=value`` strings (highest precedence).
        environ: environment mapping (defaults to ``os.environ``); handy for tests.
    """
    env = os.environ if environ is None else environ
    source = name_or_path or env.get("DCF_CONFIG") or "default"
    # Explicit files are layered on top of the packaged defaults so they can stay short.
    base, _ = _read_profile("default")
    data, profile = _read_profile(source)
    merged = _deep_merge(base, data) if source != "default" else base
    merged.setdefault("profile", profile)
    for key, value in env_overrides(env):
        _set_dotted(merged, key, value)
    for text in overrides or []:
        key, value = parse_override(text)
        _set_dotted(merged, key, value)
    cfg = _from_dict(Config, merged)
    validate(cfg)
    return cfg


def validate(cfg: Config) -> None:
    """Sanity-check cross-field constraints with actionable messages."""
    h = cfg.forecast.horizon
    if h < 1 or h > 24:
        raise ConfigError(
            "forecast.horizon must be in [1, 24] (lags >= 24h keep features leak-free)"
        )
    if min(cfg.features.lags) < h:
        raise ConfigError("features.lags must all be >= forecast.horizon to stay leakage-safe")
    if any(lag < h for lag in cfg.deep.decoder_lags):
        raise ConfigError("deep.decoder_lags must all be >= forecast.horizon")
    if cfg.deep.decoder_lags and max(cfg.deep.decoder_lags) > cfg.deep.lookback:
        raise ConfigError("deep.lookback must cover the largest decoder lag")
    if sorted(cfg.forecast.quantiles) != [0.1, 0.5, 0.9]:
        raise ConfigError("forecast.quantiles must be [0.1, 0.5, 0.9] (P10/P50/P90)")
    if cfg.decision.objective not in {"carbon", "cost", "blended"}:
        raise ConfigError("decision.objective must be one of carbon, cost, blended")
    if cfg.deep.scheduler not in {"onecycle", "cosine"}:
        raise ConfigError("deep.scheduler must be 'onecycle' or 'cosine'")
    min_days = cfg.backtest.folds * cfg.backtest.test_days + cfg.deep.val_days + 21
    if cfg.data.days < min_days:
        raise ConfigError(
            f"data.days={cfg.data.days} is too short for {cfg.backtest.folds} folds of "
            f"{cfg.backtest.test_days} days (need >= {min_days})"
        )
