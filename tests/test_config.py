from __future__ import annotations

from pathlib import Path

import pytest

from dcforecast.config import ConfigError, env_overrides, load_config, parse_override


def test_default_profile_loads() -> None:
    cfg = load_config("default", environ={})
    assert cfg.profile == "default"
    assert cfg.forecast.horizon == 24
    assert cfg.forecast.quantiles == [0.1, 0.5, 0.9]
    assert set(cfg.backtest.models) == {"naive24", "naive168", "ets", "gbm", "lstm", "tcn"}
    assert cfg.data.days >= 3 * 365


def test_tiny_profile_is_layered_on_default() -> None:
    cfg = load_config("tiny", environ={})
    assert cfg.profile == "tiny"
    assert cfg.data.sites == ["frostholm", "dunehaven"]
    assert cfg.deep.device == "cpu"
    # keys not mentioned in tiny.yaml keep their default values
    assert cfg.features.lags == [24, 48, 168]


def test_cli_and_env_overrides_are_typed_and_ordered() -> None:
    env = {"DCF__DEEP__EPOCHS": "7", "DCF__DATA__SITES": "[frostholm]", "UNRELATED": "x"}
    cfg = load_config("tiny", overrides=["deep.epochs=3", "deep.amp=false"], environ=env)
    assert cfg.deep.epochs == 3  # CLI beats environment
    assert cfg.deep.amp is False
    assert cfg.data.sites == ["frostholm"]
    assert env_overrides(env) == [("data.sites", ["frostholm"]), ("deep.epochs", 7)]


def test_config_file_and_env_profile(tmp_path: Path) -> None:
    path = tmp_path / "mine.yaml"
    path.write_text("deep:\n  epochs: 5\n", encoding="utf-8")
    cfg = load_config(str(path), environ={})
    assert cfg.profile == "mine"
    assert cfg.deep.epochs == 5
    assert load_config(None, environ={"DCF_CONFIG": "tiny"}).profile == "tiny"


@pytest.mark.parametrize(
    "override, message",
    [
        ("deep.nope=1", "Unknown config key"),
        ("features.lags=[12, 24]", "leakage-safe"),
        ("forecast.horizon=48", "horizon"),
        ("decision.objective=speed", "objective"),
        ("data.days=30", "too short"),
    ],
)
def test_invalid_overrides_raise(override: str, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        load_config("default", overrides=[override], environ={})


def test_parse_override_requires_equals() -> None:
    assert parse_override("a.b=[1, 2]") == ("a.b", [1, 2])
    with pytest.raises(ConfigError):
        parse_override("no-equals-sign")
