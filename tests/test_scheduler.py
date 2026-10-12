"""Carbon-aware scheduling LP on a hand-built problem with a known optimum."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dcforecast.decision.scheduler import (
    ScheduleProblem,
    baseline_plan,
    compare_plans,
    plan_frame,
    solve_plan,
    unit_cost,
)


def make_problem(portable: float = 0.5, cap: float = 2000.0) -> ScheduleProblem:
    n_t = 24
    hours = pd.date_range("2025-06-01", periods=n_t, freq="h")
    t = np.arange(n_t)
    clean = 100 + 300 * (np.abs(t - 12) / 12)  # solar dip around noon
    dirty = np.full(n_t, 700.0)
    price = np.where((t >= 16) & (t < 21), 180.0, 60.0)
    return ScheduleProblem(
        sites=["clean", "dirty"],
        hours=pd.DatetimeIndex(hours),
        headroom_kw=np.full((2, n_t), cap),
        pue=np.full((2, n_t), 1.2),
        carbon=np.vstack([clean, dirty]),
        price=np.vstack([price, price * 0.8]),
        volume_mwh=np.array([6.0, 6.0]),
        pool_kw=np.array([3000.0, 3000.0]),
        nominal_kw=np.array([1000.0, 1000.0]),
        portable_fraction=portable,
    )


def test_baseline_runs_immediately_at_nominal_rate() -> None:
    p = make_problem()
    base = baseline_plan(p)
    assert base.unserved_mwh == 0
    np.testing.assert_allclose(base.x_kw[:, :6], 1000.0)
    assert base.x_kw[:, 6:].sum() == 0


def test_carbon_plan_conserves_work_and_respects_caps() -> None:
    p = make_problem(cap=1500.0)
    plan = solve_plan(p, "carbon")
    assert plan.unserved_mwh == pytest.approx(0, abs=1e-6)
    assert plan.x_kw.sum() / 1000 == pytest.approx(p.volume_mwh.sum())
    assert (plan.x_kw <= p.capacity_kw + 1e-6).all()
    assert (plan.x_kw >= -1e-9).all()
    # the clean site keeps all its own work and receives the portable share of the dirty site
    assert plan.x_kw[0].sum() / 1000 == pytest.approx(6.0 + 3.0)
    assert plan.metrics["share_moved_offsite"] == pytest.approx(0.25)


def test_optimal_plans_beat_the_baseline() -> None:
    p = make_problem()
    base = baseline_plan(p)
    carbon = solve_plan(p, "carbon")
    shift_only = solve_plan(p, "carbon", allow_shift=False)
    cost = solve_plan(p, "cost")
    assert carbon.metrics["tco2"] <= shift_only.metrics["tco2"] <= base.metrics["tco2"]
    assert cost.metrics["cost_usd"] <= base.metrics["cost_usd"]
    rows = {r["plan"]: r for r in compare_plans(base, [carbon, shift_only, cost])}
    assert rows["carbon-optimal"]["tco2_saved_pct"] > 30
    assert rows["cost-optimal"]["usd_saved"] >= 0
    frame = plan_frame(p, carbon)
    assert {"site_id", "timestamp"} <= set(frame.columns)


def test_blended_objective_interpolates() -> None:
    p = make_problem()
    np.testing.assert_allclose(unit_cost(p, "blended", 0.0), unit_cost(p, "cost"))
    huge = unit_cost(p, "blended", 1e6)
    assert np.argmin(huge) == np.argmin(unit_cost(p, "carbon"))
    with pytest.raises(ValueError):
        unit_cost(p, "speed")


def test_insufficient_capacity_is_reported_not_hidden() -> None:
    p = make_problem(portable=0.0, cap=100.0)  # 24 h x 100 kW = 2.4 MWh < 6 MWh per site
    plan = solve_plan(p, "carbon")
    assert plan.unserved_mwh == pytest.approx(2 * (6.0 - 2.4), rel=1e-6)
