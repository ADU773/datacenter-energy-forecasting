"""``dcforecast report``: static HTML report (Plotly, CDN) + README PNGs + a metrics JSON.

Everything is rebuilt from the machine-readable outputs under ``reports/`` and the processed
panel, so the published numbers always come from an actual run.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import jinja2
import pandas as pd
import plotly.graph_objects as go

from dcforecast import __version__
from dcforecast.config import Config
from dcforecast.data.prepare import load_panel
from dcforecast.data.sites import SITES
from dcforecast.hardware import collect_hardware
from dcforecast.report import plots
from dcforecast.report.style import (
    AXIS,
    DE_EMPHASIS,
    GRID,
    INK,
    INK_2,
    MODE_LABELS,
    MODE_ORDER,
    MUTED,
    mode_color,
    model_color,
    model_label,
    site_color,
)
from dcforecast.tracking.tracker import Tracker
from dcforecast.utils import get_logger, read_json, write_json

log = get_logger(__name__)


@dataclass
class ReportInputs:
    """Everything the report needs (missing optional parts are simply skipped)."""

    cfg: Config
    history: pd.DataFrame
    qa: dict[str, Any]
    leaderboard: dict[str, Any] | None
    forecasts: pd.DataFrame | None
    benchmark: dict[str, Any] | None
    decision: dict[str, Any] | None
    plan: pd.DataFrame | None
    baseline: pd.DataFrame | None
    decision_fc: pd.DataFrame | None
    production: dict[str, Any] | None
    timings: dict[str, Any] | None
    histories: dict[str, list[dict[str, float]]] = field(default_factory=dict)


def _maybe_json(path: Path) -> Any:
    return read_json(path) if path.exists() else None


def _maybe_csv(path: Path, parse: tuple[str, ...] = ("timestamp",)) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    for col in parse:
        if col in df:
            df[col] = pd.to_datetime(df[col])
    return df


def load_inputs(cfg: Config) -> ReportInputs:
    rep = cfg.reports_dir
    bt = rep / "backtest"
    fc_path = bt / "forecasts.parquet"
    histories: dict[str, list[dict[str, float]]] = {}
    try:
        runs = Tracker(cfg.runs_dir).list_runs(kind="train", limit=50)
        for name in ("lstm", "tcn"):
            sub = runs[runs["model"] == name] if not runs.empty else runs
            if not sub.empty:
                histories[name] = sub.iloc[0]["tags"].get("fit_info", {}).get("history", [])
    except Exception as exc:  # pragma: no cover - tracker is optional for the report
        log.warning("could not read training histories: %s", exc)
    return ReportInputs(
        cfg=cfg,
        history=load_panel(cfg).history,
        qa=read_json(rep / "qa_report.json"),
        leaderboard=_maybe_json(bt / "leaderboard.json"),
        forecasts=pd.read_parquet(fc_path) if fc_path.exists() else None,
        benchmark=_maybe_json(rep / "benchmark.json"),
        decision=_maybe_json(rep / "decision.json"),
        plan=_maybe_csv(rep / "decision_plan.csv"),
        baseline=_maybe_csv(rep / "decision_baseline.csv"),
        decision_fc=_maybe_csv(rep / "decision_forecast.csv", ("timestamp", "origin")),
        production=_maybe_json(rep / "production_model.json"),
        timings=_maybe_json(rep / "pipeline_timings.json"),
        histories=histories,
    )


# ---------------------------------------------------------------------- plotly helpers
def _layout(
    fig: go.Figure, title: str, height: int = 380, ytitle: str = "", xtitle: str = ""
) -> go.Figure:
    fig.update_layout(
        title={"text": title, "x": 0.0, "xanchor": "left", "font": {"size": 15, "color": INK}},
        height=height,
        margin={"l": 60, "r": 20, "t": 60, "b": 45},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font={
            "family": "system-ui, -apple-system, Segoe UI, sans-serif",
            "color": INK_2,
            "size": 12,
        },
        legend={"orientation": "h", "y": -0.18, "x": 0},
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False, linecolor=AXIS, tickfont={"color": MUTED}, title=xtitle)
    fig.update_yaxes(
        gridcolor=GRID, linecolor=AXIS, tickfont={"color": MUTED}, title=ytitle, zeroline=False
    )
    return fig


class FigureSet:
    """Converts figures to HTML; the first one pulls plotly.js from the CDN."""

    def __init__(self) -> None:
        self.first = True

    def html(self, fig: go.Figure) -> str:
        out = fig.to_html(
            full_html=False,
            include_plotlyjs="cdn" if self.first else False,
            config={"displaylogo": False, "responsive": True},
        )
        self.first = False
        return out


def fig_overview(history: pd.DataFrame) -> go.Figure:
    daily = (
        history.set_index("timestamp")
        .groupby("site_id")["facility_kw"]
        .resample("D")
        .mean()
        .reset_index()
    )
    fig = go.Figure()
    for site in dict.fromkeys(history["site_id"]):
        d = daily[daily["site_id"] == site]
        fig.add_trace(
            go.Scatter(
                x=d["timestamp"],
                y=(d["facility_kw"] / 1000).round(3),
                name=SITES[site].name,
                line={"color": site_color(site), "width": 1.6},
                hovertemplate="%{y:.2f} MW",
            )
        )
    return _layout(fig, "Facility power per site (daily mean, MW)", 400, "MW")


def fig_forecast(fc: pd.DataFrame, target: str, model_name: str) -> go.Figure:
    last_fold = fc["fold"].max()
    d = fc[(fc["model"] == model_name) & (fc["fold"] == last_fold)].sort_values("timestamp")
    sites = list(dict.fromkeys(d["site_id"]))
    fig = go.Figure()
    for i, site in enumerate(sites):
        s = d[d["site_id"] == site]
        vis = i == 0
        x = s["timestamp"]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=(s["p90"] / 1000).round(3),
                line={"width": 0},
                showlegend=False,
                hoverinfo="skip",
                visible=vis,
                name="P90",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=(s["p10"] / 1000).round(3),
                fill="tonexty",
                line={"width": 0},
                fillcolor="rgba(42,120,214,0.15)",
                name="P10-P90",
                visible=vis,
                hovertemplate="P10 %{y:.2f}",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=(s[target] / 1000).round(3),
                name="actual",
                line={"color": INK, "width": 1.4},
                visible=vis,
                hovertemplate="actual %{y:.2f}",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=(s["p50"] / 1000).round(3),
                name="P50 forecast",
                line={"color": "#2a78d6", "width": 2},
                visible=vis,
                hovertemplate="P50 %{y:.2f}",
            )
        )
    buttons = []
    for i, site in enumerate(sites):
        visible = [False] * (4 * len(sites))
        visible[4 * i : 4 * i + 4] = [True] * 4
        buttons.append(
            {"label": SITES[site].name, "method": "update", "args": [{"visible": visible}]}
        )
    fig.update_layout(
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "x": 1.0,
                "xanchor": "right",
                "y": 1.18,
                "showactive": True,
            }
        ]
    )
    return _layout(
        fig,
        f"Out-of-sample day-ahead forecasts, last backtest fold - {model_label(model_name)} (MW)",
        440,
        "facility MW",
    )


def fig_leaderboard(board: pd.DataFrame, best: str) -> go.Figure:
    b = board.sort_values("mase", ascending=True)
    fig = go.Figure(
        go.Bar(
            x=b["mase"].round(4),
            y=[model_label(m) for m in b["model"]],
            orientation="h",
            marker={"color": ["#2a78d6" if m == best else DE_EMPHASIS for m in b["model"]]},
            text=[f"{v:.3f}" for v in b["mase"]],
            textposition="outside",
            hovertemplate="MASE %{x:.3f}<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed")
    fig = _layout(fig, "MASE by model (lower is better; 1.0 = seasonal naive 24h)", 340)
    fig.update_layout(hovermode="closest")
    return fig


def fig_horizon(per_h: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for m, g in per_h.groupby("model"):
        fig.add_trace(
            go.Scatter(
                x=g["horizon"],
                y=g["mape"].round(3),
                name=model_label(str(m)),
                line={"color": model_color(str(m)), "width": 2},
            )
        )
    return _layout(fig, "MAPE by forecast horizon (%)", 380, "MAPE %", "hours ahead")


def fig_bench(bench: dict[str, Any]) -> tuple[go.Figure, go.Figure]:
    res = pd.DataFrame(bench["results"])
    models = list(dict.fromkeys(res["model"]))
    f1, f2 = go.Figure(), go.Figure()
    for mode in MODE_ORDER:
        sub = res[res["mode"] == mode].set_index("model").reindex(models)
        if sub["mode"].isna().all():
            continue
        f1.add_trace(
            go.Bar(
                x=[m.upper() for m in models],
                y=sub["train_samples_per_s"],
                name=MODE_LABELS[mode],
                marker={"color": mode_color(mode)},
                text=[f"{v:,.0f}" for v in sub["train_samples_per_s"]],
                textposition="outside",
            )
        )
        f2.add_trace(
            go.Bar(
                x=[m.upper() for m in models],
                y=[r["p50_ms"] if isinstance(r, dict) else None for r in sub["infer_latency"]],
                name=MODE_LABELS[mode],
                marker={"color": mode_color(mode)},
            )
        )
    for f in (f1, f2):
        f.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08)
    _layout(f1, "Training throughput (samples/s, higher is better)", 380, "samples / s")
    _layout(f2, "Inference latency p50 per batch (ms, lower is better)", 380, "ms")
    f1.update_layout(hovermode="closest")
    f2.update_layout(hovermode="closest")
    return f1, f2


def fig_schedule(plan: pd.DataFrame, name: str) -> tuple[go.Figure, go.Figure]:
    sites = list(dict.fromkeys(plan["site_id"]))
    f1, f2 = go.Figure(), go.Figure()
    for s in sites:
        d = plan[plan["site_id"] == s]
        f1.add_trace(
            go.Scatter(
                x=d["timestamp"],
                y=d["carbon_fcst_g_per_kwh"].round(1),
                name=SITES[s].name,
                line={"color": site_color(s), "width": 2},
            )
        )
        f2.add_trace(
            go.Scatter(
                x=d["timestamp"],
                y=(d["deferrable_it_kw"] / 1000).round(3),
                name=SITES[s].name,
                stackgroup="one",
                line={"color": site_color(s), "width": 0.5},
                fillcolor=site_color(s),
            )
        )
    _layout(f1, "Grid carbon-intensity forecast (gCO2/kWh)", 380, "gCO2/kWh")
    _layout(f2, f"Deferrable load placement - {name} (IT MW)", 380, "IT MW")
    return f1, f2


def fig_pareto(pareto: list[dict[str, float]]) -> go.Figure:
    p = pd.DataFrame(pareto)
    fig = go.Figure(
        go.Scatter(
            x=p["cost_usd"].round(0),
            y=p["tco2"].round(2),
            mode="lines+markers+text",
            text=[f"${v:.0f}/t" for v in p["carbon_price_usd_per_t"]],
            textposition="top right",
            line={"color": "#2a78d6", "width": 2},
            marker={"size": 9, "line": {"color": "#fff", "width": 2}},
            hovertemplate="cost $%{x:,.0f}<br>%{y:.2f} tCO2<extra></extra>",
        )
    )
    fig = _layout(
        fig, "Cost vs emissions as the internal carbon price rises", 380, "tCO2", "energy cost ($)"
    )
    fig.update_layout(hovermode="closest")
    return fig


# ---------------------------------------------------------------------- tables
def _table(df: pd.DataFrame, fmt: dict[str, str] | None = None) -> str:
    fmt = fmt or {}
    out = df.copy()
    for col in out.columns:
        if col in fmt:
            out[col] = out[col].map(lambda v, f=fmt[col]: f.format(v) if pd.notna(v) else "-")
    return out.to_html(index=False, classes="tbl", border=0, escape=True)


def _kpis(inp: ReportInputs) -> list[dict[str, str]]:
    k: list[dict[str, str]] = []
    if inp.leaderboard:
        board = pd.DataFrame(inp.leaderboard["leaderboard"]).set_index("model")
        best = inp.leaderboard["best_model"]
        k.append(
            {
                "label": "Best model (backtest MASE)",
                "value": f"{board.loc[best, 'mase']:.3f}",
                "note": f"{model_label(best)}; MAPE {board.loc[best, 'mape']:.2f}%",
            }
        )
        if "naive24" in board.index:
            gain = (1 - board.loc[best, "mae"] / board.loc["naive24", "mae"]) * 100
            k.append(
                {
                    "label": "MAE reduction vs naive 24h",
                    "value": f"{gain:.0f}%",
                    "note": f"P10-P90 coverage {board.loc[best, 'coverage_80'] * 100:.0f}%",
                }
            )
    if inp.benchmark and inp.benchmark.get("speedups"):
        sp = [s for s in inp.benchmark["speedups"] if "train_speedup_cuda_amp_vs_cpu" in s]
        if sp:
            best_sp = max(sp, key=lambda s: s["train_speedup_cuda_amp_vs_cpu"])
            k.append(
                {
                    "label": "GPU training speed-up",
                    "value": f"x{best_sp['train_speedup_cuda_amp_vs_cpu']:.1f}",
                    "note": f"{best_sp['model'].upper()}, AMP vs CPU, 2-epoch benchmark",
                }
            )
    if inp.decision:
        plans = {p["plan"]: p for p in inp.decision["plans"]}
        carbon = plans.get("carbon-optimal")
        if carbon:
            note = f"{carbon['tco2_saved']:.1f} tCO2 on {inp.decision['deferrable_mwh']:.0f} MWh of batch work"
            if "realized_tco2_saved_pct" in carbon:
                note += f"; {carbon['realized_tco2_saved_pct']:.0f}% with realised carbon"
            k.append(
                {
                    "label": "Carbon saved by scheduling",
                    "value": f"{carbon['tco2_saved_pct']:.0f}%",
                    "note": note,
                }
            )
        total = [r for r in inp.decision["energy_rollup"] if r["site_id"] == "TOTAL"]
        if total:
            t = total[0]
            k.append(
                {
                    "label": f"Fleet energy, next {inp.decision['horizon_hours']} h",
                    "value": f"{t['energy_mwh_p50']:,.0f} MWh",
                    "note": f"${t['cost_usd_p50']:,.0f}; {t['tco2_p50']:,.0f} tCO2 (P50)",
                }
            )
    return k


def build_report(cfg: Config) -> dict[str, Path]:
    """Render ``docs/index.html``, ``docs/images/*.png`` and ``docs/metrics.json``."""
    inp = load_inputs(cfg)
    target = cfg.forecast.target
    docs = cfg.docs_dir
    img = docs / "images"
    img.mkdir(parents=True, exist_ok=True)
    (docs / ".nojekyll").touch()
    paths: dict[str, Path] = {}
    figs = FigureSet()
    ctx: dict[str, Any] = {
        "version": __version__,
        "generated": dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "profile": cfg.profile,
        "n_sites": inp.history["site_id"].nunique(),
        "start": inp.history["timestamp"].min().strftime("%Y-%m-%d"),
        "end": inp.history["timestamp"].max().strftime("%Y-%m-%d"),
        "n_rows": len(inp.history),
        "sections": {},
    }
    ctx["kpis"] = _kpis(inp)

    # ---- data
    paths["dataset_overview.png"] = plots.dataset_overview(
        inp.history, img / "dataset_overview.png"
    )
    paths["pue_vs_temperature.png"] = plots.pue_vs_temperature(
        inp.history, img / "pue_vs_temperature.png"
    )
    ctx["fig_overview"] = figs.html(fig_overview(inp.history))
    qa = inp.qa
    qa_rows = pd.DataFrame(qa["per_site"]).rename(columns={"site_id": "site"})
    ctx["qa_table"] = _table(qa_rows)
    ctx["qa"] = qa
    site_rows = [
        {
            "site": s.name,
            "id": s.site_id,
            "climate": s.climate,
            "grid": s.grid_mix,
            "UTC offset": f"{s.utc_offset_h:+d}",
            "IT capacity (MW)": s.it_capacity_kw / 1000,
            "power cap (MW)": s.power_cap_kw / 1000,
        }
        for s in (SITES[x] for x in dict.fromkeys(inp.history["site_id"]))
    ]
    ctx["sites_table"] = _table(
        pd.DataFrame(site_rows), {"IT capacity (MW)": "{:.1f}", "power cap (MW)": "{:.1f}"}
    )

    # ---- backtest
    best = None
    if inp.leaderboard and inp.forecasts is not None:
        lb = inp.leaderboard
        board = pd.DataFrame(lb["leaderboard"])
        best = lb["best_model"]
        prod = (inp.production or {}).get("name", best)
        per_site = pd.DataFrame(lb["per_site"])
        per_h = pd.DataFrame(lb["per_horizon"])
        last = inp.forecasts[inp.forecasts["fold"] == inp.forecasts["fold"].max()]
        show_sites = [s for s in ("dunehaven", "marisport") if s in set(last["site_id"])] or list(
            dict.fromkeys(last["site_id"])
        )[:2]
        paths["forecast_vs_actual.png"] = plots.forecast_vs_actual(
            last[last["model"] == prod], target, prod, img / "forecast_vs_actual.png", show_sites
        )
        paths["leaderboard.png"] = plots.leaderboard_chart(board, best, img / "leaderboard.png")
        paths["site_mase.png"] = plots.site_heatmap(per_site, img / "site_mase.png")
        paths["residual_diagnostics.png"] = plots.residual_diagnostics(
            inp.forecasts, per_h, target, prod, img / "residual_diagnostics.png"
        )
        ctx["fig_forecast"] = figs.html(fig_forecast(inp.forecasts, target, prod))
        ctx["fig_leaderboard"] = figs.html(fig_leaderboard(board, best))
        ctx["fig_horizon"] = figs.html(fig_horizon(per_h))
        cols = [
            "rank",
            "model",
            "mae",
            "rmse",
            "mape",
            "smape",
            "mase",
            "pinball",
            "pinball_scaled",
            "coverage_80",
            "interval_width",
        ]
        tb = board[cols].copy()
        tb["model"] = tb["model"].map(model_label)
        tb["coverage_80"] = tb["coverage_80"] * 100
        ctx["leaderboard_table"] = _table(
            tb,
            {
                "mae": "{:,.1f}",
                "rmse": "{:,.1f}",
                "mape": "{:.2f}",
                "smape": "{:.2f}",
                "mase": "{:.3f}",
                "pinball": "{:,.1f}",
                "pinball_scaled": "{:.3f}",
                "coverage_80": "{:.1f}",
                "interval_width": "{:,.0f}",
            },
        )
        piv = per_site.pivot(index="model", columns="site_id", values="mase")
        piv.index = [model_label(m) for m in piv.index]
        piv.columns = [SITES[c].name for c in piv.columns]
        ctx["site_table"] = (
            piv.round(3).reset_index(names="model").to_html(index=False, classes="tbl", border=0)
        )
        ctx["folds"] = lb["folds"]
        ctx["timings_table"] = _table(
            pd.DataFrame(
                [
                    {
                        "model": model_label(m),
                        "fit (s, all folds)": v["fit_s"],
                        "predict (s)": v["predict_s"],
                        "device": v.get("device", "cpu"),
                    }
                    for m, v in lb["timings"].items()
                ]
            ),
            {"fit (s, all folds)": "{:.1f}", "predict (s)": "{:.1f}"},
        )
        ctx["best_model"] = model_label(best)
        ctx["prod_model"] = model_label(prod)
        ctx["prod_reason"] = (inp.production or {}).get("reason", "")
    if inp.histories:
        paths["training_curves.png"] = plots.training_curves(
            inp.histories, img / "training_curves.png"
        )

    # ---- benchmark
    hw = (inp.benchmark or {}).get("hardware") or collect_hardware().to_dict()
    ctx["hw"] = hw
    if inp.benchmark and inp.benchmark.get("results"):
        paths["benchmark_speedup.png"] = plots.benchmark_chart(
            inp.benchmark, img / "benchmark_speedup.png"
        )
        f1, f2 = fig_bench(inp.benchmark)
        ctx["fig_bench_train"] = figs.html(f1)
        ctx["fig_bench_infer"] = figs.html(f2)
        rows = [
            {
                "model": r["model"].upper(),
                "mode": MODE_LABELS.get(r["mode"], r["mode"]),
                "train samples/s": r["train_samples_per_s"],
                "epoch (s, 60k samples)": r["epoch_s_full"],
                "infer p50 (ms)": r["infer_latency"]["p50_ms"],
                "infer p95 (ms)": r["infer_latency"]["p95_ms"],
                "batch-1 p50 (ms)": r["infer_latency_batch1"]["p50_ms"],
                "GPU peak (MB)": r["train_peak_gpu_mem_mb"],
                "GPU util (%)": r["sampled_gpu_util_percent_mean"],
                "CPU (%)": r["sampled_cpu_percent_mean"],
                "RSS max (MB)": r["sampled_rss_mb_max"],
            }
            for r in inp.benchmark["results"]
        ]
        ctx["bench_table"] = _table(
            pd.DataFrame(rows),
            {
                "train samples/s": "{:,.0f}",
                "epoch (s, 60k samples)": "{:.1f}",
                "infer p50 (ms)": "{:.2f}",
                "infer p95 (ms)": "{:.2f}",
                "batch-1 p50 (ms)": "{:.2f}",
                "GPU peak (MB)": "{:,.0f}",
                "GPU util (%)": "{:.0f}",
                "CPU (%)": "{:.0f}",
                "RSS max (MB)": "{:,.0f}",
            },
        )
        ctx["bench_cfg"] = inp.benchmark["config"]

    # ---- decision
    if inp.decision and inp.plan is not None and inp.baseline is not None:
        d = inp.decision
        chosen = d["chosen_plan"]
        paths["carbon_schedule.png"] = plots.schedule_chart(
            inp.plan, inp.baseline, img / "carbon_schedule.png", f"Plan: {chosen}"
        )
        paths["scheduling_savings.png"] = plots.savings_chart(
            d["plans"], img / "scheduling_savings.png"
        )
        f1, f2 = fig_schedule(inp.plan, chosen)
        ctx["fig_ci"] = figs.html(f1)
        ctx["fig_plan"] = figs.html(f2)
        ctx["fig_pareto"] = figs.html(fig_pareto(d["pareto"]))
        roll = pd.DataFrame(d["energy_rollup"])[
            [
                "site_name",
                "energy_mwh_p10",
                "energy_mwh_p50",
                "energy_mwh_p90",
                "cost_usd_p50",
                "tco2_p50",
                "avg_price_usd_per_mwh",
                "avg_carbon_g_per_kwh",
            ]
        ]
        ctx["rollup_table"] = _table(
            roll.rename(columns={"site_name": "site"}),
            {c: "{:,.1f}" for c in roll.columns if c != "site_name"} | {"cost_usd_p50": "${:,.0f}"},
        )
        pl = pd.DataFrame(d["plans"])
        keep = [
            "plan",
            "facility_mwh",
            "tco2",
            "cost_usd",
            "tco2_saved_pct",
            "usd_saved_pct",
            "share_moved_offsite",
        ]
        if "realized_tco2_saved_pct" in pl:
            keep.append("realized_tco2_saved_pct")
        pl = pl[keep].copy()
        pl["share_moved_offsite"] = pl["share_moved_offsite"] * 100
        ctx["plans_table"] = _table(
            pl,
            {
                "facility_mwh": "{:,.1f}",
                "tco2": "{:,.2f}",
                "cost_usd": "${:,.0f}",
                "tco2_saved_pct": "{:+.1f}",
                "usd_saved_pct": "{:+.1f}",
                "share_moved_offsite": "{:.0f}",
                "realized_tco2_saved_pct": "{:+.1f}",
            },
        )
        ctx["decision"] = d
    ctx["timings"] = inp.timings
    ctx["resources"] = _maybe_json(cfg.reports_dir / "resources.json")

    tpl_text = (
        resources.files("dcforecast.report")
        .joinpath("templates/report.html.j2")
        .read_text(encoding="utf-8")
    )
    env = jinja2.Environment(
        autoescape=jinja2.select_autoescape(default=True), undefined=jinja2.ChainableUndefined
    )
    html = env.from_string(tpl_text).render(**ctx)
    out = docs / "index.html"
    out.write_text(html, encoding="utf-8")
    paths["index.html"] = out
    paths["metrics.json"] = write_json(docs / "metrics.json", _published_metrics(inp))
    log.info("report written to %s (%d KB)", out.as_posix(), out.stat().st_size // 1024)
    return paths


def _published_metrics(inp: ReportInputs) -> dict[str, Any]:
    """Small, privacy-safe JSON of the headline numbers (the README quotes these)."""
    out: dict[str, Any] = {"generated_by": f"dcforecast {__version__}", "profile": inp.cfg.profile}
    out["hardware"] = (inp.benchmark or {}).get("hardware") or collect_hardware().to_dict()
    out["data"] = {
        "sites": int(inp.history["site_id"].nunique()),
        "rows": len(inp.history),
        "start": inp.history["timestamp"].min(),
        "end": inp.history["timestamp"].max(),
        "qa": {
            k: inp.qa[k]
            for k in (
                "totals",
                "injected_faults",
                "detection",
                "imputation_error",
                "facility_imputed_pct",
            )
        },
    }
    if inp.leaderboard:
        out["backtest"] = {
            "folds": inp.leaderboard["folds"],
            "leaderboard": inp.leaderboard["leaderboard"],
            "best_model": inp.leaderboard["best_model"],
            "timings": inp.leaderboard["timings"],
        }
    if inp.production:
        out["production_model"] = inp.production
    if inp.benchmark:
        out["benchmark"] = {
            k: inp.benchmark[k] for k in ("hardware", "config", "results", "speedups")
        }
    if inp.decision:
        out["decision"] = inp.decision
    if inp.timings:
        out["pipeline_timings_s"] = inp.timings
    resources = _maybe_json(inp.cfg.reports_dir / "resources.json")
    if resources:
        out["resources"] = resources
    return out
