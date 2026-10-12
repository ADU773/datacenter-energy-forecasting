"""Shared fixtures: a tiny synthetic dataset and one tiny end-to-end pipeline run per session.

Everything is written to pytest temp directories; nothing depends on previously generated data,
the network or a GPU.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dcforecast.config import Config, load_config
from dcforecast.data.generate import run_generate
from dcforecast.data.prepare import PanelData, load_panel, prepare


def path_overrides(root: Path) -> list[str]:
    """``--set`` overrides that redirect every output directory below ``root``."""
    return [
        f"paths.{key}={(root / name).as_posix()}"
        for key, name in (
            ("data_dir", "data"),
            ("runs_dir", "runs"),
            ("reports_dir", "reports"),
            ("docs_dir", "docs"),
        )
    ]


def tiny_config(root: Path, extra: list[str] | None = None) -> Config:
    """The CI-sized ``tiny`` profile with all outputs below ``root``."""
    return load_config("tiny", overrides=path_overrides(root) + list(extra or []), environ={})


@pytest.fixture(scope="session")
def tiny_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("tiny-data")


@pytest.fixture(scope="session")
def tiny_cfg(tiny_root: Path) -> Config:
    return tiny_config(tiny_root)


@pytest.fixture(scope="session")
def tiny_data(tiny_cfg: Config) -> PanelData:
    """Generated + cleaned tiny dataset (two sites, ~5 months)."""
    run_generate(tiny_cfg)
    prepare(tiny_cfg)
    return load_panel(tiny_cfg)


@pytest.fixture(scope="session")
def pipeline_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run the whole tiny pipeline once through the CLI (``run-all``) and return its root."""
    from dcforecast.cli import main

    root = tmp_path_factory.mktemp("tiny-pipeline")
    args = ["-c", "tiny", "--log-level", "WARNING"]
    for item in path_overrides(root):
        args += ["--set", item]
    args += [
        "--set",
        "backtest.models=[naive24, naive168, gbm, lstm, tcn]",
        "--set",
        "train.models=[naive24, gbm, lstm, tcn]",
        "run-all",
        "--skip-benchmark",
    ]
    assert main(args) == 0
    return root


@pytest.fixture(scope="session")
def pipeline_cfg(pipeline_root: Path) -> Config:
    return tiny_config(pipeline_root)
