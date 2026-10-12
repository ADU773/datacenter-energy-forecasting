"""Pydantic request / response models for the API."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    production_model: str | None = None
    history_end: datetime | None = None
    sites: list[str] = []


class ForecastPoint(BaseModel):
    timestamp: datetime
    horizon: int
    p10: float
    p50: float
    p90: float


class ForecastResponse(BaseModel):
    site_id: str
    model: str
    model_version: int
    origin: datetime
    unit: str = "kW (facility power, hourly mean)"
    energy_mwh: float
    cost_usd: float
    tco2: float
    points: list[ForecastPoint]


class ScheduleRequest(BaseModel):
    deferrable_mwh: float = Field(120.0, gt=0, le=5000, description="IT energy to place (MWh)")
    horizon_hours: int = Field(72, ge=24, le=72)
    objective: Literal["carbon", "cost", "blended"] = "carbon"
    carbon_price_usd_per_t: float = Field(85.0, ge=0, le=5000)
    portable_fraction: float = Field(0.5, ge=0.0, le=1.0)


class PlanSummary(BaseModel):
    plan: str
    facility_mwh: float
    tco2: float
    cost_usd: float
    tco2_saved: float
    tco2_saved_pct: float
    usd_saved: float
    usd_saved_pct: float
    unserved_mwh: float
    share_moved_offsite: float


class HourlyAllocation(BaseModel):
    site_id: str
    timestamp: datetime
    deferrable_it_kw: float
    carbon_fcst_g_per_kwh: float
    price_usd_per_mwh: float


class ScheduleResponse(BaseModel):
    model: str
    model_version: int
    origin: datetime
    objective: str
    chosen_plan: str
    plans: list[PlanSummary]
    allocation: list[HourlyAllocation]
