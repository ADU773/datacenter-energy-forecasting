"""FastAPI service on top of the tiny end-to-end pipeline."""

from __future__ import annotations

import warnings

import pytest

from dcforecast.api.app import create_app
from dcforecast.config import Config

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(pipeline_cfg: Config) -> TestClient:
    return TestClient(create_app(pipeline_cfg))


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["production_model"]


def test_models_lists_registry_with_production_alias(client: TestClient) -> None:
    r = client.get("/models")
    assert r.status_code == 200
    models = r.json()["models"]
    assert {m["name"] for m in models} >= {"naive24", "gbm", "lstm", "tcn"}
    assert sum("production" in m["aliases"] for m in models) == 1


def test_forecast_returns_ordered_quantiles(client: TestClient) -> None:
    r = client.get("/forecast", params={"site": "frostholm", "horizon": 24})
    assert r.status_code == 200
    body = r.json()
    assert body["site_id"] == "frostholm"
    assert len(body["points"]) == 24
    assert [p["horizon"] for p in body["points"]] == list(range(1, 25))
    assert all(p["p10"] <= p["p50"] <= p["p90"] for p in body["points"])
    assert body["energy_mwh"] > 0 and body["tco2"] > 0 and body["cost_usd"] > 0


def test_forecast_validates_input(client: TestClient) -> None:
    assert client.get("/forecast", params={"site": "atlantis"}).status_code == 404
    assert client.get("/forecast", params={"site": "frostholm", "horizon": 500}).status_code == 422


def test_schedule_get_and_post(client: TestClient) -> None:
    r = client.get("/schedule", params={"deferrable_mwh": 10, "horizon_hours": 48})
    assert r.status_code == 200
    body = r.json()
    plans = {p["plan"]: p for p in body["plans"]}
    assert body["chosen_plan"] == "carbon-optimal"
    assert plans["carbon-optimal"]["tco2"] <= plans["run immediately (baseline)"]["tco2"]
    r = client.post(
        "/schedule", json={"deferrable_mwh": 5, "horizon_hours": 24, "objective": "cost"}
    )
    assert r.status_code == 200
    assert r.json()["chosen_plan"] == "cost-optimal"
    bad = client.post("/schedule", json={"deferrable_mwh": -1})
    assert bad.status_code == 422
