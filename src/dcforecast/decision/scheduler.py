"""Carbon-aware (or cost-aware) scheduling of deferrable batch compute as a linear program.

Problem
-------
Each site ``s`` has a deferrable batch workload of ``V_s`` MWh of IT energy (fleet total split
by IT capacity) that must finish inside the decision window. A share ``phi`` of it is
*portable* (stateless jobs that may run at any site); the rest must run at its home site.

Decision variables (IT kW, hourly): ``y[s,t]`` home work at site s, ``z[s,t]`` portable work
executed at site s, plus slack ``u`` for work that cannot be placed (heavily penalised).

    minimise   sum_{s,t} (y + z)[s,t] * PUE[s,t] * c[s,t]  +  M * sum(u)
    subject to sum_t y[s,t] + u_s       = V_s * (1 - phi)        for every site s
               sum_{s,t} z[s,t] + u_p   = phi * sum_s V_s
               y[s,t] + z[s,t] <= min(headroom[s,t], pool_s)
               y, z, u >= 0

``c`` is the carbon-intensity forecast (carbon objective), the tariff (cost objective) or the
tariff plus a carbon price (blended). ``headroom = (power_cap - P90 facility forecast) / PUE``,
so the plan respects each site's contracted power cap even in a high-load (P90) scenario.
``pool_s`` caps the batch cluster at ``pool_rate_multiple`` x its nominal rate.

The baseline runs every site's work immediately at its home site at the nominal rate
(``V_s / baseline_hours``). Solved with ``scipy.optimize.linprog`` (HiGHS).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix

from dcforecast.config import DecisionConfig
from dcforecast.data.sites import SITES
from dcforecast.decision.forecast import PUECurve

UNSERVED_PENALTY = 1e6


@dataclass
class ScheduleProblem:
    """Inputs of the LP, all as (sites x hours) arrays."""

    sites: list[str]
    hours: pd.DatetimeIndex
    headroom_kw: np.ndarray
    pue: np.ndarray
    carbon: np.ndarray  # gCO2/kWh (forecast)
    price: np.ndarray  # $/MWh
    volume_mwh: np.ndarray  # per home site
    pool_kw: np.ndarray
    nominal_kw: np.ndarray
    portable_fraction: float
    carbon_actual: np.ndarray | None = None
    pue_actual: np.ndarray | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.sites), len(self.hours)

    @property
    def capacity_kw(self) -> np.ndarray:
        return np.minimum(self.headroom_kw, self.pool_kw[:, None])


@dataclass
class Plan:
    """A placement of deferrable IT load (kW) per site and hour."""

    name: str
    x_kw: np.ndarray
    unserved_mwh: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)
    realized: dict[str, float] = field(default_factory=dict)


def build_problem(
    forecast: pd.DataFrame,
    future: pd.DataFrame,
    cfg: DecisionConfig,
    pue_curve: PUECurve,
    holdout: pd.DataFrame | None = None,
    total_mwh: float | None = None,
) -> ScheduleProblem:
    """Assemble LP inputs from a facility-power forecast and the future covariate feed."""
    f = forecast.merge(
        future[
            ["site_id", "timestamp", "temp_fcst_c", "carbon_fcst_g_per_kwh", "price_usd_per_mwh"]
        ],
        on=["site_id", "timestamp"],
        how="left",
    )
    sites = list(dict.fromkeys(f["site_id"]))
    hours = pd.DatetimeIndex(sorted(f["timestamp"].unique()))

    def grid(col: str, frame: pd.DataFrame = f) -> np.ndarray:
        piv = frame.pivot_table(index="site_id", columns="timestamp", values=col, aggfunc="mean")
        return piv.reindex(index=sites, columns=hours).to_numpy(dtype=float)

    p90 = grid("p90")
    temp = grid("temp_fcst_c")
    pue = np.vstack([pue_curve.predict(s, temp[i]) for i, s in enumerate(sites)])
    cap = np.array([SITES[s].power_cap_kw for s in sites])
    headroom = np.clip((cap[:, None] - p90) / pue, 0.0, None)
    it_cap = np.array([SITES[s].it_capacity_kw for s in sites])
    total = cfg.deferrable_mwh if total_mwh is None else total_mwh
    volume = total * it_cap / it_cap.sum()
    nominal = volume * 1000.0 / cfg.baseline_hours
    carbon_actual = pue_actual = None
    if holdout is not None and not holdout.empty:
        h = holdout[holdout["site_id"].isin(sites) & holdout["timestamp"].isin(hours)]
        if len(h) == len(sites) * len(hours):
            carbon_actual = grid("carbon_g_per_kwh", h)
            pue_actual = grid("pue", h)
    return ScheduleProblem(
        sites=sites,
        hours=hours,
        headroom_kw=np.nan_to_num(headroom),
        pue=pue,
        carbon=np.nan_to_num(grid("carbon_fcst_g_per_kwh")),
        price=np.nan_to_num(grid("price_usd_per_mwh")),
        volume_mwh=volume,
        pool_kw=nominal * cfg.pool_rate_multiple,
        nominal_kw=nominal,
        portable_fraction=cfg.portable_fraction,
        carbon_actual=carbon_actual,
        pue_actual=pue_actual,
    )


def unit_cost(problem: ScheduleProblem, objective: str, carbon_price: float = 0.0) -> np.ndarray:
    """Objective coefficient per kWh of IT energy placed at (s, t)."""
    tco2_per_kwh = problem.pue * problem.carbon / 1e6
    usd_per_kwh = problem.pue * problem.price / 1000.0
    if objective == "carbon":
        return tco2_per_kwh
    if objective == "cost":
        return usd_per_kwh
    if objective == "blended":
        return usd_per_kwh + carbon_price * tco2_per_kwh
    raise ValueError(f"unknown objective '{objective}'")


def evaluate_plan(problem: ScheduleProblem, x_kw: np.ndarray) -> tuple[dict, dict]:
    """Energy / cost / emissions of a plan with forecast and (if known) realised inputs."""
    facility_kwh = x_kw * problem.pue
    metrics = {
        "it_mwh": float(x_kw.sum() / 1000.0),
        "facility_mwh": float(facility_kwh.sum() / 1000.0),
        "tco2": float((facility_kwh * problem.carbon).sum() / 1e6),
        "cost_usd": float((facility_kwh * problem.price).sum() / 1000.0),
        "peak_kw": float(x_kw.sum(axis=0).max()),
        "share_moved_offsite": 0.0,
    }
    realized: dict[str, float] = {}
    if problem.carbon_actual is not None and problem.pue_actual is not None:
        fac = x_kw * problem.pue_actual
        realized = {
            "facility_mwh": float(fac.sum() / 1000.0),
            "tco2": float((fac * problem.carbon_actual).sum() / 1e6),
            "cost_usd": float((fac * problem.price).sum() / 1000.0),
        }
    return metrics, realized


def baseline_plan(problem: ScheduleProblem) -> Plan:
    """Run everything immediately at the home site at the nominal rate."""
    n_s, n_t = problem.shape
    cap = problem.capacity_kw
    x = np.zeros((n_s, n_t))
    unserved = 0.0
    for s in range(n_s):
        remaining = problem.volume_mwh[s] * 1000.0
        for t in range(n_t):
            if remaining <= 1e-9:
                break
            step = min(problem.nominal_kw[s], cap[s, t], remaining)
            x[s, t] = step
            remaining -= step
        unserved += remaining / 1000.0
    plan = Plan("run immediately (baseline)", x, unserved)
    plan.metrics, plan.realized = evaluate_plan(problem, x)
    return plan


def solve_plan(
    problem: ScheduleProblem,
    objective: str = "carbon",
    carbon_price: float = 0.0,
    allow_shift: bool = True,
    name: str | None = None,
) -> Plan:
    """Solve the LP and return the optimal plan."""
    n_s, n_t = problem.shape
    n = n_s * n_t
    phi = problem.portable_fraction if allow_shift else 0.0
    local = problem.volume_mwh * (1.0 - phi) * 1000.0
    portable = float(problem.volume_mwh.sum() * phi * 1000.0)
    c_unit = unit_cost(problem, objective, carbon_price).ravel()
    # variable layout: [y (n), z (n), u_local (n_s), u_portable (1)]
    c = np.concatenate([c_unit, c_unit, np.full(n_s + 1, UNSERVED_PENALTY)])
    n_var = 2 * n + n_s + 1
    # equality: per-site home volume, total portable volume
    rows, cols, vals = [], [], []
    for s in range(n_s):
        idx = np.arange(s * n_t, (s + 1) * n_t)
        rows += [s] * n_t
        cols += idx.tolist()
        vals += [1.0] * n_t
        rows.append(s)
        cols.append(2 * n + s)
        vals.append(1.0)
    rows += [n_s] * n
    cols += list(range(n, 2 * n))
    vals += [1.0] * n
    rows.append(n_s)
    cols.append(2 * n + n_s)
    vals.append(1.0)
    a_eq = csr_matrix(coo_matrix((vals, (rows, cols)), shape=(n_s + 1, n_var)))
    b_eq = np.concatenate([local, [portable]])
    # capacity: y[s,t] + z[s,t] <= cap[s,t]
    eye = np.arange(n)
    a_ub = csr_matrix(
        coo_matrix(
            (np.ones(2 * n), (np.concatenate([eye, eye]), np.concatenate([eye, eye + n]))),
            shape=(n, n_var),
        )
    )
    b_ub = problem.capacity_kw.ravel()
    res = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    if not res.success:  # pragma: no cover - the slack variables keep it feasible
        raise RuntimeError(f"LP failed: {res.message}")
    y = res.x[:n].reshape(n_s, n_t)
    z = res.x[n : 2 * n].reshape(n_s, n_t)
    unserved = float(res.x[2 * n :].sum() / 1000.0)
    label = name or f"{objective}-optimal" + ("" if allow_shift else " (time-shift only)")
    plan = Plan(label, y + z, unserved)
    plan.metrics, plan.realized = evaluate_plan(problem, plan.x_kw)
    # share of all deferrable work that executed away from its home site
    moved = np.clip(z.sum(axis=1) / 1000.0 - phi * problem.volume_mwh, 0.0, None).sum()
    plan.metrics["share_moved_offsite"] = float(moved / problem.volume_mwh.sum()) if phi else 0.0
    return plan


def compare_plans(baseline: Plan, plans: list[Plan]) -> list[dict[str, Any]]:
    """Savings of each plan versus the baseline (forecast-based and realised)."""
    out = []
    for p in [baseline, *plans]:
        row: dict[str, Any] = {"plan": p.name, "unserved_mwh": p.unserved_mwh, **p.metrics}
        row["tco2_saved"] = baseline.metrics["tco2"] - p.metrics["tco2"]
        row["tco2_saved_pct"] = 100.0 * row["tco2_saved"] / baseline.metrics["tco2"]
        row["usd_saved"] = baseline.metrics["cost_usd"] - p.metrics["cost_usd"]
        row["usd_saved_pct"] = 100.0 * row["usd_saved"] / baseline.metrics["cost_usd"]
        if p.realized and baseline.realized:
            row["realized_tco2"] = p.realized["tco2"]
            row["realized_tco2_saved"] = baseline.realized["tco2"] - p.realized["tco2"]
            row["realized_tco2_saved_pct"] = (
                100.0 * row["realized_tco2_saved"] / baseline.realized["tco2"]
            )
            row["realized_usd_saved"] = baseline.realized["cost_usd"] - p.realized["cost_usd"]
        out.append(row)
    return out


def plan_frame(problem: ScheduleProblem, plan: Plan) -> pd.DataFrame:
    """Long hourly frame of a plan with its inputs (for charts and the API)."""
    n_s, n_t = problem.shape
    return pd.DataFrame(
        {
            "site_id": np.repeat(problem.sites, n_t),
            "timestamp": np.tile(problem.hours.values, n_s),
            "deferrable_it_kw": plan.x_kw.ravel(),
            "headroom_it_kw": problem.capacity_kw.ravel(),
            "pue_est": problem.pue.ravel(),
            "carbon_fcst_g_per_kwh": problem.carbon.ravel(),
            "price_usd_per_mwh": problem.price.ravel(),
        }
    )
