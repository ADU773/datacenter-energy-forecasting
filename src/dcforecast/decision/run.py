"""``dcforecast forecast`` / ``dcforecast schedule``: the decision layer end to end."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd

from dcforecast.config import Config
from dcforecast.data.prepare import PanelData
from dcforecast.decision.energy import energy_rollup, realized_check
from dcforecast.decision.forecast import PUECurve, forecast_ahead
from dcforecast.decision.scheduler import (
    Plan,
    ScheduleProblem,
    baseline_plan,
    build_problem,
    compare_plans,
    plan_frame,
    solve_plan,
)
from dcforecast.models.base import Forecaster
from dcforecast.utils import get_logger, write_json

log = get_logger(__name__)

PARETO_CARBON_PRICES = [0, 25, 50, 100, 200, 400, 800, 1600]


def load_holdout(cfg: Config) -> pd.DataFrame | None:
    """Realised future values hidden from all models (used only to audit decisions)."""
    path = cfg.raw_dir / "holdout_future.parquet"
    return pd.read_parquet(path) if path.exists() else None


@dataclass
class DecisionResult:
    """Forecast roll-up and scheduling results."""

    forecast: pd.DataFrame
    rollup: pd.DataFrame
    problem: ScheduleProblem
    baseline: Plan
    plans: list[Plan]
    comparison: list[dict[str, Any]]
    pareto: list[dict[str, float]]
    realized_forecast_check: dict[str, float]
    chosen: Plan

    def summary(self) -> dict[str, Any]:
        return {
            "origin": self.forecast["origin"].iloc[0],
            "horizon_hours": int(self.forecast["horizon"].max()),
            "sites": self.problem.sites,
            "energy_rollup": self.rollup.to_dict(orient="records"),
            "forecast_vs_realized": self.realized_forecast_check,
            "deferrable_mwh": float(self.problem.volume_mwh.sum()),
            "portable_fraction": self.problem.portable_fraction,
            "plans": self.comparison,
            "pareto": self.pareto,
            "chosen_plan": self.chosen.name,
        }


def forecast_and_rollup(
    cfg: Config, data: PanelData, model: Forecaster, hours: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, PUECurve]:
    """Forecast ``hours`` ahead and roll up energy / cost / carbon per site."""
    hours = hours or cfg.decision.horizon_hours
    pue_curve = PUECurve().fit(data.history)
    fc = forecast_ahead(model, data, hours, pue_curve=pue_curve)
    return fc, energy_rollup(fc, data.future), pue_curve


def run_decision(
    cfg: Config,
    data: PanelData,
    model: Forecaster,
    hours: int | None = None,
    deferrable_mwh: float | None = None,
    objective: str | None = None,
    carbon_price: float | None = None,
    holdout: pd.DataFrame | None = None,
    portable_fraction: float | None = None,
    forecast: pd.DataFrame | None = None,
) -> DecisionResult:
    """Forecast, roll up and schedule deferrable compute; compare against running now.

    ``forecast`` may be passed in (e.g. a cached API forecast) to skip re-forecasting.
    """
    dc = cfg.decision
    if portable_fraction is not None:
        dc = replace(dc, portable_fraction=portable_fraction)
    objective = objective or dc.objective
    carbon_price = dc.carbon_price_usd_per_t if carbon_price is None else carbon_price
    if forecast is None:
        fc, rollup, pue_curve = forecast_and_rollup(cfg, data, model, hours)
    else:
        hours = hours or dc.horizon_hours
        fc = forecast[forecast["horizon"] <= hours].reset_index(drop=True)
        rollup = energy_rollup(fc, data.future)
        pue_curve = PUECurve().fit(data.history)
    realized = realized_check(fc, holdout, cfg.forecast.target) if holdout is not None else {}
    problem = build_problem(fc, data.future, dc, pue_curve, holdout, deferrable_mwh)
    base = baseline_plan(problem)
    plans = [
        solve_plan(problem, "carbon", allow_shift=False),
        solve_plan(problem, "carbon"),
        solve_plan(problem, "cost"),
        solve_plan(problem, "blended", carbon_price, name=f"blended (${carbon_price:.0f}/tCO2)"),
    ]
    comparison = compare_plans(base, plans)
    pareto = []
    for price in PARETO_CARBON_PRICES:
        p = solve_plan(problem, "blended", float(price))
        pareto.append(
            {
                "carbon_price_usd_per_t": float(price),
                "cost_usd": p.metrics["cost_usd"],
                "tco2": p.metrics["tco2"],
            }
        )
    chosen = {"carbon": plans[1], "cost": plans[2], "blended": plans[3]}[objective]
    return DecisionResult(fc, rollup, problem, base, plans, comparison, pareto, realized, chosen)


def save_decision(result: DecisionResult, out_dir: Path) -> dict[str, Path]:
    """Persist the decision summary (JSON) and hourly detail (CSV) under ``reports/``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": out_dir / "decision.json",
        "forecast": out_dir / "decision_forecast.csv",
        "plan": out_dir / "decision_plan.csv",
        "baseline": out_dir / "decision_baseline.csv",
    }
    write_json(paths["summary"], result.summary())
    result.forecast.to_csv(paths["forecast"], index=False)
    plan_frame(result.problem, result.chosen).to_csv(paths["plan"], index=False)
    plan_frame(result.problem, result.baseline).to_csv(paths["baseline"], index=False)
    return paths
