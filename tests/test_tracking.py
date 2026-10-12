"""SQLite experiment tracker and model registry."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from dcforecast.config import Config
from dcforecast.data.prepare import PanelData
from dcforecast.models import create_model
from dcforecast.tracking.registry import PRODUCTION, ModelNotFoundError, ModelRegistry
from dcforecast.tracking.tracker import Tracker


def test_runs_are_recorded_listed_and_compared(tmp_path: Path) -> None:
    tracker = Tracker(tmp_path / "runs", profile="test")
    with tracker.start_run("train", model="gbm", params={"lr": 0.1}) as run:
        run.log_metrics({"mae": 12.5}, mase=0.8)
        run.set_device("cpu")
        run.log_artifact("model", tmp_path / "runs" / "registry" / "gbm")
    with pytest.raises(RuntimeError), tracker.start_run("backtest", model="lstm") as failing:
        failing.log_metrics(mase=9.9)
        raise RuntimeError("boom")
    runs = tracker.list_runs()
    assert len(runs) == 2
    assert set(runs["status"]) == {"finished", "failed"}
    ok = tracker.get_run(run.run_id)
    assert ok["params"] == {"lr": 0.1}
    assert ok["metrics"] == {"mae": 12.5, "mase": 0.8}
    assert ok["device"] == "cpu" and ok["profile"] == "test"
    assert ok["duration_s"] >= 0
    assert tracker.list_runs(kind="train", model="gbm")["run_id"].tolist() == [run.run_id]
    cmp = tracker.compare([run.run_id, failing.run_id])
    assert cmp.loc["mase"].tolist() == [0.8, 9.9]
    with pytest.raises(KeyError):
        tracker.get_run("does-not-exist")


def test_registry_versions_aliases_and_loading(
    tmp_path: Path, tiny_cfg: Config, tiny_data: PanelData
) -> None:
    reg = ModelRegistry(tmp_path / "runs")
    with pytest.raises(ModelNotFoundError):
        reg.resolve(PRODUCTION)
    model = create_model("naive168", tiny_cfg).fit(tiny_data.history)
    assert reg.register(model, "run-a", {"mase": 1.0}) == 1
    assert reg.register(model, "run-b", {"mase": 0.9}) == 2
    assert reg.latest_version("naive168") == 2
    reg.set_alias(PRODUCTION, "naive168", 1, reason="test")
    entry = reg.resolve(PRODUCTION)
    assert entry["version"] == 1 and entry["alias"] == PRODUCTION
    assert reg.resolve("naive168:2")["run_id"] == "run-b"
    loaded, _ = reg.load(PRODUCTION)
    origins = [tiny_data.history_end - pd.Timedelta(days=1)]
    pd.testing.assert_frame_equal(
        loaded.predict(tiny_data.history, origins), model.predict(tiny_data.history, origins)
    )
    listing = reg.list_models()
    assert len(listing) == 2
    assert PRODUCTION in listing.loc[listing["version"] == 1, "aliases"].iloc[0]
