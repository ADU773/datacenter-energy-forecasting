"""End-to-end: the tiny pipeline via the CLI (``run-all`` in a fixture) plus every subcommand."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from dcforecast.cli import build_parser, main
from dcforecast.config import Config
from dcforecast.utils import read_json

from .conftest import path_overrides


def cli(root: Path, *args: str) -> int:
    base = ["-c", "tiny", "--log-level", "WARNING"]
    for item in path_overrides(root):
        base += ["--set", item]
    return main([*base, *args])


def test_parser_has_every_subcommand() -> None:
    sub = next(a for a in build_parser()._actions if a.dest == "command")
    expected = {
        "hw", "generate", "prepare", "train", "backtest", "benchmark", "runs", "promote",
        "forecast", "schedule", "serve", "report", "run-all",
    }  # fmt: skip
    assert expected <= set(sub.choices)


def test_run_all_produces_reports(pipeline_root: Path, pipeline_cfg: Config) -> None:
    rep = pipeline_cfg.reports_dir
    board = read_json(rep / "backtest" / "leaderboard.json")
    assert {r["model"] for r in board["leaderboard"]} == {
        "naive24",
        "naive168",
        "gbm",
        "lstm",
        "tcn",
    }
    assert len(board["folds"]) == pipeline_cfg.backtest.folds
    # the learned models beat the 24 h seasonal-naive baseline
    mase = {r["model"]: r["mase"] for r in board["leaderboard"]}
    assert mase["gbm"] < mase["naive24"]
    prod = read_json(rep / "production_model.json")
    assert prod["name"] == board["best_model"]
    decision = read_json(rep / "decision.json")
    plans = {p["plan"]: p for p in decision["plans"]}
    assert plans["carbon-optimal"]["tco2_saved"] > 0
    timings = read_json(rep / "pipeline_timings.json")
    assert timings["total"] > 0


def test_report_is_small_and_privacy_safe(pipeline_cfg: Config) -> None:
    docs = pipeline_cfg.docs_dir
    html = (docs / "index.html").read_text(encoding="utf-8")
    assert "cdn.plot.ly" in html and "Helios Cloud" in html
    assert (docs / "index.html").stat().st_size < 2_000_000
    assert (docs / ".nojekyll").exists()
    pngs = list((docs / "images").glob("*.png"))
    assert len(pngs) >= 8
    published = html + (docs / "metrics.json").read_text(encoding="utf-8")
    assert not re.search(r"[A-Za-z]:[\\/]+Users", published)
    assert "/home/" not in published and str(Path.home()) not in published
    assert '"host": "host-01"' in (docs / "metrics.json").read_text(encoding="utf-8")


def test_runs_promote_forecast_schedule(
    pipeline_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli(pipeline_root, "runs", "list", "--kind", "backtest") == 0
    assert "backtest" in capsys.readouterr().out
    assert cli(pipeline_root, "runs", "models") == 0
    assert "production" in capsys.readouterr().out
    assert cli(pipeline_root, "promote", "--model", "gbm") == 0
    assert "production -> gbm" in capsys.readouterr().out
    out_csv = pipeline_root / "fc.csv"
    assert (
        cli(
            pipeline_root,
            "forecast",
            "--site",
            "dunehaven",
            "--horizon",
            "48",
            "--output",
            str(out_csv),
        )
        == 0
    )
    fc = pd.read_csv(out_csv)
    assert len(fc) == 48 and set(fc["site_id"]) == {"dunehaven"}
    assert cli(pipeline_root, "schedule", "--mwh", "8", "--hours", "24", "--objective", "cost") == 0
    assert "cost-optimal" in capsys.readouterr().out


def test_benchmark_and_hw_json(
    pipeline_root: Path, pipeline_cfg: Config, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli(pipeline_root, "benchmark", "--model", "lstm", "--modes", "cpu") == 0
    bench = read_json(pipeline_cfg.reports_dir / "benchmark.json")
    row = bench["results"][0]
    assert row["mode"] == "cpu" and row["train_samples_per_s"] > 0
    assert row["infer_latency"]["p95_ms"] >= row["infer_latency"]["p50_ms"] > 0
    assert bench["hardware"]["host"] == "host-01"
    capsys.readouterr()
    assert cli(pipeline_root, "hw", "--json") == 0
    hw = json.loads(capsys.readouterr().out)
    assert hw["cpu_logical_cores"] >= 1 and hw["ram_total_gb"] > 0


def test_missing_data_gives_a_clear_error(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    assert cli(tmp_path, "train", "--model", "gbm") == 1
    assert "dcforecast generate" in caplog.text or "dcforecast prepare" in caplog.text


def test_python_dash_m_entry_point() -> None:
    out = subprocess.run(
        [sys.executable, "-m", "dcforecast", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.startswith("dcforecast ")
