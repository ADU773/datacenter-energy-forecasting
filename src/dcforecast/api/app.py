"""FastAPI service: health, model registry, forecasts and carbon-aware schedules.

The production model (registry alias ``production``) is loaded lazily on first use and runs on
the CPU by default (``serve.device``), which is the cheap, always-available serving target;
GPUs are used for training.
"""

from __future__ import annotations

import threading
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from dcforecast import __version__
from dcforecast.api.schemas import (
    ForecastPoint,
    ForecastResponse,
    HealthResponse,
    HourlyAllocation,
    PlanSummary,
    ScheduleRequest,
    ScheduleResponse,
)
from dcforecast.config import Config, load_config
from dcforecast.data.prepare import DataNotFoundError, PanelData, load_panel
from dcforecast.decision.energy import energy_rollup
from dcforecast.decision.forecast import PUECurve, forecast_ahead
from dcforecast.decision.run import run_decision
from dcforecast.decision.scheduler import plan_frame
from dcforecast.models.base import Forecaster
from dcforecast.tracking.registry import PRODUCTION, ModelNotFoundError, ModelRegistry
from dcforecast.utils import get_logger, to_jsonable

log = get_logger(__name__)


class ServiceState:
    """Lazily loaded data + production model, shared by request handlers."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.registry = ModelRegistry(cfg.runs_dir)
        self._lock = threading.Lock()
        self._data: PanelData | None = None
        self._model: Forecaster | None = None
        self._entry: dict[str, Any] | None = None
        self._pue: PUECurve | None = None
        self._cache: dict[int, pd.DataFrame] = {}

    def data(self) -> PanelData:
        with self._lock:
            if self._data is None:
                self._data = load_panel(self.cfg)
                self._pue = PUECurve().fit(self._data.history)
            return self._data

    def model(self) -> tuple[Forecaster, dict[str, Any]]:
        with self._lock:
            if self._model is None:
                self._model, self._entry = self.registry.load(
                    PRODUCTION, device=self.cfg.serve.device
                )
                log.info(
                    "loaded production model %s v%s", self._entry["name"], self._entry["version"]
                )
            assert self._entry is not None
            return self._model, self._entry

    def forecast(self, hours: int) -> pd.DataFrame:
        """All-site forecast for ``hours`` (cached; covers every horizon <= 72 h)."""
        data = self.data()
        model, _ = self.model()
        key = 72 if hours <= 72 else hours
        with self._lock:
            if key not in self._cache:
                self._cache[key] = forecast_ahead(model, data, key, pue_curve=self._pue)
            fc = self._cache[key]
        return fc[fc["horizon"] <= hours]

    @property
    def loaded(self) -> bool:
        return self._model is not None


def create_app(cfg: Config | None = None) -> FastAPI:
    """Application factory (tests pass their own config)."""
    cfg = cfg or load_config()
    state = ServiceState(cfg)
    app = FastAPI(
        title="Helios Cloud power forecasting API",
        version=__version__,
        description="P10/P50/P90 facility-power forecasts and carbon-aware batch scheduling "
        "for the (fictional) Helios Cloud fleet.",
    )
    app.state.service = state

    def _require_model() -> tuple[Forecaster, dict[str, Any]]:
        try:
            return state.model()
        except ModelNotFoundError as exc:
            raise HTTPException(503, f"no production model: {exc}") from exc

    def _require_data() -> PanelData:
        try:
            return state.data()
        except DataNotFoundError as exc:
            raise HTTPException(503, str(exc)) from exc

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        prod = None
        try:
            entry = state.registry.resolve(PRODUCTION)
            prod = f"{entry['name']}:v{entry['version']}"
        except ModelNotFoundError:
            pass
        data = state._data
        return HealthResponse(
            status="ok",
            version=__version__,
            model_loaded=state.loaded,
            production_model=prod,
            history_end=data.history_end if data is not None else None,
            sites=data.sites if data is not None else [],
        )

    @app.get("/models")
    def models() -> dict[str, Any]:
        df = state.registry.list_models()
        records = [] if df.empty else df.drop(columns=["path"]).to_dict(orient="records")
        aliases = {
            k: {"name": v["name"], "version": v["version"], "reason": v["reason"]}
            for k, v in state.registry.aliases().items()
        }
        return to_jsonable({"models": records, "aliases": aliases})

    @app.get("/forecast", response_model=ForecastResponse)
    def forecast(
        site: str = Query(..., description="site id, e.g. frostholm"),
        horizon: int = Query(24, ge=1, le=72, description="hours ahead"),
    ) -> ForecastResponse:
        data = _require_data()
        if site not in data.sites:
            raise HTTPException(404, f"unknown site '{site}'; known: {', '.join(data.sites)}")
        _, entry = _require_model()
        fc = state.forecast(horizon)
        fc = fc[fc["site_id"] == site]
        roll = energy_rollup(fc, data.future).iloc[0]
        return ForecastResponse(
            site_id=site,
            model=entry["name"],
            model_version=int(entry["version"]),
            origin=pd.Timestamp(fc["origin"].iloc[0]).to_pydatetime(),
            energy_mwh=float(roll["energy_mwh_p50"]),
            cost_usd=float(roll["cost_usd_p50"]),
            tco2=float(roll["tco2_p50"]),
            points=[
                ForecastPoint(
                    timestamp=pd.Timestamp(r.timestamp).to_pydatetime(),
                    horizon=int(r.horizon),
                    p10=float(r.p10),
                    p50=float(r.p50),
                    p90=float(r.p90),
                )
                for r in fc.itertuples()
            ],
        )

    def _schedule(req: ScheduleRequest) -> ScheduleResponse:
        data = _require_data()
        model, entry = _require_model()
        res = run_decision(
            cfg,
            data,
            model,
            hours=req.horizon_hours,
            deferrable_mwh=req.deferrable_mwh,
            objective=req.objective,
            carbon_price=req.carbon_price_usd_per_t,
            portable_fraction=req.portable_fraction,
            forecast=state.forecast(req.horizon_hours),
        )
        alloc = plan_frame(res.problem, res.chosen)
        alloc = alloc[alloc["deferrable_it_kw"] > 1e-6]
        keys = PlanSummary.model_fields.keys()
        return ScheduleResponse(
            model=entry["name"],
            model_version=int(entry["version"]),
            origin=pd.Timestamp(res.forecast["origin"].iloc[0]).to_pydatetime(),
            objective=req.objective,
            chosen_plan=res.chosen.name,
            plans=[PlanSummary(**{k: row[k] for k in keys}) for row in res.comparison],
            allocation=[
                HourlyAllocation(
                    site_id=r.site_id,
                    timestamp=pd.Timestamp(r.timestamp).to_pydatetime(),
                    deferrable_it_kw=float(r.deferrable_it_kw),
                    carbon_fcst_g_per_kwh=float(r.carbon_fcst_g_per_kwh),
                    price_usd_per_mwh=float(r.price_usd_per_mwh),
                )
                for r in alloc.itertuples()
            ],
        )

    @app.post("/schedule", response_model=ScheduleResponse)
    def schedule_post(req: ScheduleRequest) -> ScheduleResponse:
        return _schedule(req)

    @app.get("/schedule", response_model=ScheduleResponse)
    def schedule_get(
        deferrable_mwh: float = Query(120.0, gt=0, le=5000),
        horizon_hours: int = Query(72, ge=24, le=72),
        objective: str = Query("carbon", pattern="^(carbon|cost|blended)$"),
        carbon_price_usd_per_t: float = Query(85.0, ge=0, le=5000),
        portable_fraction: float = Query(0.5, ge=0.0, le=1.0),
    ) -> ScheduleResponse:
        return _schedule(
            ScheduleRequest(
                deferrable_mwh=deferrable_mwh,
                horizon_hours=horizon_hours,
                objective=objective,  # type: ignore[arg-type]
                carbon_price_usd_per_t=carbon_price_usd_per_t,
                portable_fraction=portable_fraction,
            )
        )

    return app
