"""Turn hourly power forecasts into energy (MWh), cost ($) and emissions (tCO2) per site.

Units: an hourly mean power of ``P`` kW for one hour is ``P`` kWh. Cost uses the time-of-use
tariff ($/MWh); emissions use the grid operator's carbon-intensity forecast (gCO2/kWh), so
``tCO2 = sum(P_kW * CI) / 1e6``.
"""

from __future__ import annotations

import pandas as pd

from dcforecast.data.sites import SITES


def attach_prices(forecast: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
    """Join tariff and carbon-intensity forecasts onto an hourly forecast frame."""
    cols = ["site_id", "timestamp", "price_usd_per_mwh", "carbon_fcst_g_per_kwh"]
    return forecast.merge(future[cols], on=["site_id", "timestamp"], how="left")


def energy_rollup(forecast: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
    """Expected energy, cost and emissions per site (P50 plus P10/P90 bands)."""
    f = attach_prices(forecast, future)
    rows = []
    for site, g in f.groupby("site_id", sort=False):
        row: dict[str, float | str] = {
            "site_id": str(site),
            "site_name": SITES[str(site)].name,
            "hours": len(g),
        }
        for q in ("p10", "p50", "p90"):
            row[f"energy_mwh_{q}"] = float(g[q].sum() / 1000.0)
            row[f"cost_usd_{q}"] = float((g[q] / 1000.0 * g["price_usd_per_mwh"]).sum())
            row[f"tco2_{q}"] = float((g[q] * g["carbon_fcst_g_per_kwh"]).sum() / 1e6)
        row["avg_price_usd_per_mwh"] = row["cost_usd_p50"] / max(row["energy_mwh_p50"], 1e-9)  # type: ignore[operator]
        row["avg_carbon_g_per_kwh"] = (
            row["tco2_p50"] * 1e6 / max(row["energy_mwh_p50"] * 1000, 1e-9)
        )  # type: ignore[operator]
        rows.append(row)
    out = pd.DataFrame(rows)
    total = {"site_id": "TOTAL", "site_name": "Fleet", "hours": int(out["hours"].max())}
    for col in out.columns:
        if col.startswith(("energy_", "cost_", "tco2_")):
            total[col] = float(out[col].sum())
    total["avg_price_usd_per_mwh"] = total["cost_usd_p50"] / total["energy_mwh_p50"]
    total["avg_carbon_g_per_kwh"] = total["tco2_p50"] * 1e6 / (total["energy_mwh_p50"] * 1000)
    return pd.concat([out, pd.DataFrame([total])], ignore_index=True)


def realized_check(forecast: pd.DataFrame, holdout: pd.DataFrame, target: str) -> dict[str, float]:
    """Compare forecast energy with the realised (held-out) energy over the same hours."""
    m = forecast.merge(holdout[["site_id", "timestamp", target]], on=["site_id", "timestamp"])
    if m.empty:
        return {}
    actual = float(m[target].sum() / 1000.0)
    pred = float(m["p50"].sum() / 1000.0)
    inside = float(((m[target] >= m["p10"]) & (m[target] <= m["p90"])).mean())
    return {
        "hours": int(m["timestamp"].nunique()),
        "energy_mwh_forecast": pred,
        "energy_mwh_actual": actual,
        "energy_error_pct": (pred - actual) / actual * 100.0,
        "hourly_mape_pct": float(((m["p50"] - m[target]).abs() / m[target]).mean() * 100.0),
        "p10_p90_coverage": inside,
    }
