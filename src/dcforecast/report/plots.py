"""Static matplotlib figures for the README (``docs/images/*.png``, white background, 110 dpi)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from dcforecast.data.sites import SITES
from dcforecast.report.style import (
    AXIS,
    DE_EMPHASIS,
    GRID,
    INK,
    INK_2,
    MODE_LABELS,
    MODE_ORDER,
    MUTED,
    SEQUENTIAL_BLUE,
    SURFACE,
    mode_color,
    model_color,
    model_label,
    site_color,
)

DPI = 110

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titleweight": "semibold",
        "axes.titlesize": 11,
        "axes.labelsize": 9.5,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "text.color": INK,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round",
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

THOUSANDS = FuncFormatter(lambda v, _: f"{v:,.0f}")


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


def _suptitle(fig: plt.Figure, title: str, subtitle: str | None = None) -> float:
    """Left-aligned title (+ subtitle) placed in inches; returns the top of the plotting area."""
    h = fig.get_figheight()
    fig.suptitle(
        title,
        x=0.01,
        y=1 - 0.08 / h,
        ha="left",
        va="top",
        fontsize=13,
        fontweight="semibold",
        color=INK,
    )
    if subtitle:
        fig.text(0.01, 1 - 0.36 / h, subtitle, ha="left", va="top", fontsize=9, color=INK_2)
    return 1 - (0.52 if subtitle else 0.34) / h


# ---------------------------------------------------------------------- data overview
def dataset_overview(history: pd.DataFrame, path: Path) -> Path:
    """Daily mean facility power per site (small multiples)."""
    sites = list(dict.fromkeys(history["site_id"]))
    n = len(sites)
    cols = 3 if n > 2 else n
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 2.6 * rows + 0.6), sharex=True, squeeze=False)
    daily = (
        history.set_index("timestamp")
        .groupby("site_id")["facility_kw"]
        .resample("D")
        .mean()
        .reset_index()
    )
    for ax, site in zip(axes.ravel(), sites, strict=False):
        d = daily[daily["site_id"] == site]
        ax.plot(d["timestamp"], d["facility_kw"] / 1000, color=site_color(site), lw=1.4)
        ax.set_title(f"{SITES[site].name}  ({SITES[site].climate})", fontsize=10, loc="left")
        ax.set_ylabel("MW")
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    fig.autofmt_xdate()
    top = _suptitle(
        fig, "Facility power, daily mean per site", "Growth, launches, seasonal cooling load"
    )
    fig.tight_layout(rect=(0, 0, 1, top))
    return _save(fig, path)


def pue_vs_temperature(history: pd.DataFrame, path: Path, max_points: int = 2500) -> Path:
    """PUE against outside temperature per site: free cooling vs chiller regime."""
    sites = list(dict.fromkeys(history["site_id"]))
    n = len(sites)
    cols = 3 if n > 2 else n
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 2.8 * rows + 0.6), sharey=True, squeeze=False)
    rng = np.random.default_rng(0)
    for ax, site in zip(axes.ravel(), sites, strict=False):
        d = history[(history["site_id"] == site) & ~history["facility_imputed"]]
        if len(d) > max_points:
            d = d.iloc[rng.choice(len(d), max_points, replace=False)]
        ax.scatter(d["temp_c"], d["pue"], s=4, alpha=0.25, color=site_color(site), linewidths=0)
        ax.axvline(SITES[site].free_cooling_c, color=MUTED, lw=1)
        ax.set_title(SITES[site].name, fontsize=10, loc="left")
        ax.set_xlabel("outside temperature (C)")
        ax.set_ylabel("PUE")
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    top = _suptitle(
        fig,
        "PUE vs outside temperature",
        "Vertical line = free-cooling threshold; above it chillers kick in and PUE rises",
    )
    fig.tight_layout(rect=(0, 0, 1, top))
    return _save(fig, path)


# ---------------------------------------------------------------------- forecasts
def forecast_vs_actual(
    fc: pd.DataFrame, target: str, model_name: str, path: Path, sites: list[str], days: int = 7
) -> Path:
    """P10-P90 band, P50 and actuals for the last ``days`` of the final backtest fold."""
    fig, axes = plt.subplots(len(sites), 1, figsize=(12, 2.9 * len(sites) + 0.5), squeeze=False)
    for ax, site in zip(axes[:, 0], sites, strict=True):
        d = fc[(fc["site_id"] == site)].sort_values("timestamp")
        d = d[d["timestamp"] > d["timestamp"].max() - pd.Timedelta(days=days)]
        ax.fill_between(
            d["timestamp"],
            d["p10"] / 1000,
            d["p90"] / 1000,
            color="#2a78d6",
            alpha=0.15,
            linewidth=0,
            label="P10-P90 interval",
        )
        ax.plot(d["timestamp"], d[target] / 1000, color=INK, lw=1.4, label="actual")
        ax.plot(d["timestamp"], d["p50"] / 1000, color="#2a78d6", lw=2, label="P50 forecast")
        ax.set_title(f"{SITES[site].name}", loc="left", fontsize=10)
        ax.set_ylabel("facility MW")
    axes[0, 0].legend(loc="upper left", ncol=3)
    top = _suptitle(
        fig,
        f"Day-ahead forecast vs actual - {model_label(model_name)}",
        "Out-of-sample: each day forecast at 23:00 UTC for the next 24 h (last backtest week)",
    )
    fig.tight_layout(rect=(0, 0, 1, top))
    return _save(fig, path)


def leaderboard_chart(board: pd.DataFrame, best: str, path: Path) -> Path:
    """MASE and P10-P90 coverage per model (best model emphasised)."""
    b = board.sort_values("mase", ascending=False)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.6), gridspec_kw={"width_ratios": [3, 2]})
    y = np.arange(len(b))
    colors = ["#2a78d6" if m == best else DE_EMPHASIS for m in b["model"]]
    ax1.barh(y, b["mase"], height=0.55, color=colors)
    ax1.set_yticks(y, [model_label(m) for m in b["model"]])
    for yi, v in zip(y, b["mase"], strict=True):
        ax1.text(v + 0.01, yi, f"{v:.3f}", va="center", fontsize=8.5, color=INK_2)
    ax1.axvline(1.0, color=MUTED, lw=1)
    ax1.set_xlabel("MASE (lower is better; 1.0 = same hour yesterday)")
    ax1.set_title("Point accuracy", loc="left")
    ax1.grid(axis="y", visible=False)
    ax2.barh(y, b["coverage_80"] * 100, height=0.55, color=colors)
    ax2.axvline(80, color=MUTED, lw=1)
    for yi, v in zip(y, b["coverage_80"] * 100, strict=True):
        ax2.text(v + 1, yi, f"{v:.0f}%", va="center", fontsize=8.5, color=INK_2)
    ax2.set_yticks(y, [""] * len(y))
    ax2.set_xlim(0, 105)
    ax2.set_xlabel("P10-P90 coverage (%) - nominal 80%")
    ax2.set_title("Interval calibration", loc="left")
    ax2.grid(axis="y", visible=False)
    top = _suptitle(fig, "Backtest leaderboard (rolling origin)", None)
    fig.tight_layout(rect=(0, 0, 1, top))
    return _save(fig, path)


def site_heatmap(per_site: pd.DataFrame, path: Path) -> Path:
    """MASE per model x site (sequential blue; annotated cells)."""
    piv = per_site.pivot(index="model", columns="site_id", values="mase")
    order = [m for m in ["naive24", "naive168", "ets", "gbm", "lstm", "tcn"] if m in piv.index]
    piv = piv.loc[order]
    fig, ax = plt.subplots(figsize=(1.5 * piv.shape[1] + 3, 0.55 * piv.shape[0] + 1.6))
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("blue", SEQUENTIAL_BLUE)
    ax.imshow(piv.to_numpy(), cmap=cmap, aspect="auto")
    ax.set_xticks(range(piv.shape[1]), [SITES[s].name for s in piv.columns])
    ax.set_yticks(range(piv.shape[0]), [model_label(m) for m in piv.index])
    vmax = np.nanmax(piv.to_numpy())
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iat[i, j]
            ax.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if v > 0.55 * vmax else INK,
            )
    ax.grid(False)
    ax.set_title("MASE by site (darker = larger error)", loc="left")
    fig.tight_layout()
    return _save(fig, path)


def residual_diagnostics(
    fc: pd.DataFrame, per_horizon: pd.DataFrame, target: str, model_name: str, path: Path
) -> Path:
    """Residual histogram, ACF, error by horizon (all models), bias by hour of day."""
    d = fc[(fc["model"] == model_name) & ~fc["imputed"]].copy()
    d["resid_pct"] = (d[target] - d["p50"]) / d[target] * 100
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2))
    ax = axes[0, 0]
    ax.hist(d["resid_pct"].clip(-10, 10), bins=60, color="#2a78d6", alpha=0.85, rwidth=0.9)
    ax.axvline(0, color=INK, lw=1)
    ax.set_title(
        f"Residuals of {model_label(model_name)} (% of actual, clipped at +/-10)", loc="left"
    )
    ax.set_xlabel("actual - P50 (%)")
    ax.yaxis.set_major_formatter(THOUSANDS)

    ax = axes[0, 1]
    lags = np.arange(1, 49)
    acfs = []
    for (_site, _fold), g in d.sort_values("timestamp").groupby(["site_id", "fold"]):
        r = g["resid_pct"].to_numpy()
        r = r - r.mean()
        denom = (r**2).sum()
        if denom > 0 and len(r) > 60:
            acfs.append([(r[:-k] * r[k:]).sum() / denom for k in lags])
    acf = np.mean(acfs, axis=0) if acfs else np.zeros(len(lags))
    ax.vlines(lags, 0, acf, color="#2a78d6", lw=2)
    conf = 1.96 / np.sqrt(max(len(d) / max(len(acfs), 1), 1))
    ax.axhspan(-conf, conf, color=GRID, alpha=0.6, lw=0)
    ax.axhline(0, color=AXIS, lw=1)
    ax.set_title("Residual autocorrelation (mean over site x fold)", loc="left")
    ax.set_xlabel("lag (hours)")

    ax = axes[1, 0]
    for m, g in per_horizon.groupby("model"):
        ax.plot(g["horizon"], g["mape"], color=model_color(str(m)), lw=2, label=model_label(str(m)))
    ax.set_title("MAPE by forecast horizon", loc="left")
    ax.set_xlabel("hours ahead")
    ax.set_ylabel("MAPE (%)")
    ax.legend(ncol=2, fontsize=8)

    ax = axes[1, 1]
    from dcforecast.data.calendar import local_calendar

    hours = []
    for site, g in d.groupby("site_id"):
        cal = local_calendar(g["timestamp"], SITES[str(site)].utc_offset_h)
        hours.append(
            pd.DataFrame({"hour": cal["local_hour"].to_numpy(), "r": g["resid_pct"].to_numpy()})
        )
    hh = pd.concat(hours).groupby("hour")["r"].agg(["mean", "std"])
    ax.bar(hh.index, hh["mean"], width=0.7, color="#2a78d6")
    ax.axhline(0, color=INK, lw=1)
    ax.set_title("Mean residual by local hour of day (bias check)", loc="left")
    ax.set_xlabel("local hour")
    ax.set_ylabel("mean residual (%)")
    top = _suptitle(fig, "Residual diagnostics (backtest, out-of-sample)", None)
    fig.tight_layout(rect=(0, 0, 1, top))
    return _save(fig, path)


# ---------------------------------------------------------------------- benchmark
def benchmark_chart(bench: dict[str, Any], path: Path) -> Path:
    """Training throughput and inference latency, CPU vs GPU fp32 vs GPU AMP."""
    res = pd.DataFrame(bench["results"])
    models = list(dict.fromkeys(res["model"]))
    modes = [m for m in MODE_ORDER if m in set(res["mode"])]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    width = 0.8 / max(len(modes), 1)
    x = np.arange(len(models))
    for i, mode in enumerate(modes):
        vals = [
            float(res[(res["model"] == m) & (res["mode"] == mode)]["train_samples_per_s"].iloc[0])
            if ((res["model"] == m) & (res["mode"] == mode)).any()
            else np.nan
            for m in models
        ]
        cpu = [
            float(res[(res["model"] == m) & (res["mode"] == "cpu")]["train_samples_per_s"].iloc[0])
            if ((res["model"] == m) & (res["mode"] == "cpu")).any()
            else np.nan
            for m in models
        ]
        pos = x + (i - (len(modes) - 1) / 2) * width
        ax1.bar(pos, vals, width=width * 0.92, color=mode_color(mode), label=MODE_LABELS[mode])
        for p, v, c in zip(pos, vals, cpu, strict=True):
            if np.isfinite(v):
                lab = f"{v:,.0f}" if mode == "cpu" else f"{v:,.0f}\n(x{v / c:.1f})"
                ax1.text(p, v, lab, ha="center", va="bottom", fontsize=7.5, color=INK_2)
        lat = [
            float(
                res[(res["model"] == m) & (res["mode"] == mode)]["infer_latency"].iloc[0]["p50_ms"]
            )
            if ((res["model"] == m) & (res["mode"] == mode)).any()
            else np.nan
            for m in models
        ]
        ax2.bar(pos, lat, width=width * 0.92, color=mode_color(mode), label=MODE_LABELS[mode])
        for p, v in zip(pos, lat, strict=True):
            if np.isfinite(v):
                ax2.text(p, v, f"{v:.1f}", ha="center", va="bottom", fontsize=7.5, color=INK_2)
    for ax in (ax1, ax2):
        ax.set_xticks(x, [m.upper() for m in models])
        ax.grid(axis="x", visible=False)
    ax1.set_ylabel("training samples / s")
    ax1.yaxis.set_major_formatter(THOUSANDS)
    ax1.set_title("Training throughput (higher is better)", loc="left")
    ax1.margins(y=0.25)
    ax2.set_ylabel(f"ms per batch of {res['infer_batch_size'].iloc[0]}")
    ax2.set_title("Inference latency p50 (lower is better)", loc="left")
    ax2.margins(y=0.2)
    ax1.set_ylim(top=ax1.get_ylim()[1] * 1.18)
    ax1.legend(loc="upper left", ncol=len(modes))
    hw = bench.get("hardware", {})
    gpu = hw.get("gpus", [{}])[0].get("name", "no GPU") if hw.get("gpus") else "no GPU"
    top = _suptitle(fig, "CPU vs GPU benchmark", f"{hw.get('cpu_model', '')} | {gpu}")
    fig.tight_layout(rect=(0, 0, 1, top))
    return _save(fig, path)


def training_curves(histories: dict[str, list[dict[str, float]]], path: Path) -> Path:
    """Validation pinball loss per epoch for the deep models (final training runs)."""
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for name, hist in histories.items():
        if not hist:
            continue
        ep = [h["epoch"] for h in hist]
        ax.plot(
            ep, [h["val_loss"] for h in hist], color=model_color(name), label=f"{name.upper()} val"
        )
        ax.plot(
            ep,
            [h["train_loss"] for h in hist],
            color=model_color(name),
            lw=1,
            alpha=0.5,
            label=f"{name.upper()} train",
        )
    ax.set_xlabel("epoch")
    ax.set_ylabel("pinball loss (scaled)")
    ax.set_title("Deep model training curves (final models)", loc="left")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(ncol=2)
    fig.tight_layout()
    return _save(fig, path)


# ---------------------------------------------------------------------- decisions
def schedule_chart(plan: pd.DataFrame, baseline: pd.DataFrame, path: Path, title_plan: str) -> Path:
    """Carbon-intensity forecast per site and where the deferrable load runs (baseline vs plan)."""
    sites = list(dict.fromkeys(plan["site_id"]))
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    ax = axes[0]
    for s in sites:
        d = plan[plan["site_id"] == s]
        ax.plot(
            d["timestamp"],
            d["carbon_fcst_g_per_kwh"],
            color=site_color(s),
            lw=2,
            label=SITES[s].name,
        )
    ax.set_ylabel("gCO2 / kWh")
    ax.set_title("Grid carbon-intensity forecast", loc="left")
    ax.legend(ncol=6, loc="upper left", bbox_to_anchor=(0, 1.0))
    ax.margins(y=0.15)
    for ax, frame, ttl in (
        (axes[1], baseline, "Baseline: run immediately at home site"),
        (axes[2], plan, title_plan),
    ):
        piv = (
            frame.pivot(index="timestamp", columns="site_id", values="deferrable_it_kw")[sites]
            / 1000
        )
        ax.stackplot(
            piv.index,
            piv.T.to_numpy(),
            colors=[site_color(s) for s in sites],
            edgecolor=SURFACE,
            linewidth=0.6,
        )
        ax.set_ylabel("deferrable IT MW")
        ax.set_title(ttl, loc="left")
    top = _suptitle(fig, "Carbon-aware scheduling of deferrable batch compute (next 72 h)", None)
    fig.tight_layout(rect=(0, 0, 1, top))
    return _save(fig, path)


def savings_chart(plans: list[dict[str, Any]], path: Path) -> Path:
    """Emissions and cost of each plan, with savings vs the baseline."""
    df = pd.DataFrame(plans)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.8))
    y = np.arange(len(df))[::-1]
    colors = [DE_EMPHASIS if i == 0 else "#2a78d6" for i in range(len(df))]
    ax1.barh(y, df["tco2"], height=0.55, color=colors)
    ax2.barh(y, df["cost_usd"], height=0.55, color=colors)
    for yi, v, pct in zip(y, df["tco2"], df["tco2_saved_pct"], strict=True):
        tag = "baseline" if abs(pct) < 1e-9 else f"{-pct:+.0f}%"
        ax1.text(v, yi, f"  {v:,.1f} t ({tag})", va="center", fontsize=8.5, color=INK_2)
    for yi, v, pct in zip(y, df["cost_usd"], df["usd_saved_pct"], strict=True):
        tag = "baseline" if abs(pct) < 1e-9 else f"{-pct:+.0f}%"
        ax2.text(v, yi, f"  ${v:,.0f} ({tag})", va="center", fontsize=8.5, color=INK_2)
    ax1.set_yticks(y, df["plan"])
    ax2.set_yticks(y, [""] * len(y))
    ax1.set_title("Emissions of the deferrable work (tCO2)", loc="left")
    ax2.set_title("Energy cost of the deferrable work ($)", loc="left")
    for ax in (ax1, ax2):
        ax.grid(axis="y", visible=False)
        ax.margins(x=0.35)
    ax2.xaxis.set_major_formatter(THOUSANDS)
    top = _suptitle(fig, "Scheduling plans vs running immediately (forecast-based)", None)
    fig.tight_layout(rect=(0, 0, 1, top))
    return _save(fig, path)
