"""Command-line interface: ``dcforecast <command>`` (also ``python -m dcforecast``)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence

import pandas as pd

from dcforecast import __version__
from dcforecast.config import Config, ConfigError, load_config
from dcforecast.utils import get_logger, setup_logging, to_jsonable

log = get_logger("cli")

DEVICE_CHOICES = ["auto", "cuda", "mps", "cpu"]


# ---------------------------------------------------------------------- helpers
def _print_table(df: pd.DataFrame, floatfmt: str = "{:,.3f}") -> None:
    with pd.option_context("display.max_columns", 30, "display.width", 200):
        print(df.to_string(index=False, float_format=lambda v: floatfmt.format(v)))


def _models_arg(value: str | None, default: list[str]) -> list[str]:
    if not value or value == "all":
        return list(default)
    return [m.strip() for m in value.split(",") if m.strip()]


# ---------------------------------------------------------------------- commands
def cmd_hw(cfg: Config, args: argparse.Namespace) -> int:
    from dcforecast.hardware import collect_hardware, format_hardware

    info = collect_hardware()
    print(json.dumps(to_jsonable(info.to_dict()), indent=2) if args.json else format_hardware(info))
    return 0


def cmd_generate(cfg: Config, args: argparse.Namespace) -> int:
    from dcforecast.data.generate import run_generate

    paths = run_generate(cfg)
    for key, path in paths.items():
        print(f"{key:<17} {path.as_posix()}")
    return 0


def cmd_prepare(cfg: Config, args: argparse.Namespace) -> int:
    from dcforecast.data.prepare import prepare

    paths = prepare(cfg)
    for key, path in paths.items():
        print(f"{key:<10} {path.as_posix()}")
    return 0


def cmd_train(cfg: Config, args: argparse.Namespace) -> int:
    from dcforecast.data.prepare import load_panel
    from dcforecast.pipeline import train_models

    if args.no_warm_start:
        cfg.train.warm_start = False
    models = _models_arg(args.model, cfg.train.models)
    out = train_models(cfg, load_panel(cfg), models, device=args.device)
    rows = [{"model": k, **{kk: vv for kk, vv in v.items()}} for k, v in out.items()]
    _print_table(pd.DataFrame(rows))
    return 0


def cmd_backtest(cfg: Config, args: argparse.Namespace) -> int:
    from dcforecast.data.prepare import load_panel
    from dcforecast.evaluation.backtest import run_backtest
    from dcforecast.pipeline import backtest_dir
    from dcforecast.tracking.tracker import Tracker

    models = _models_arg(args.model, cfg.backtest.models)
    res = run_backtest(
        cfg,
        load_panel(cfg),
        models,
        device=args.device,
        tracker=Tracker(cfg.runs_dir, cfg.profile),
        out_dir=backtest_dir(cfg),
    )
    cols = ["rank", "model", "mae", "rmse", "mape", "smape", "mase", "pinball", "coverage_80"]
    _print_table(res.leaderboard[cols])
    print(f"\nbest model by MASE: {res.best_model()}")
    return 0


def cmd_benchmark(cfg: Config, args: argparse.Namespace) -> int:
    from dcforecast.benchmark import run_benchmark
    from dcforecast.data.prepare import load_panel
    from dcforecast.tracking.tracker import Tracker

    modes = _models_arg(args.modes, cfg.benchmark.modes)
    models = _models_arg(args.model, cfg.benchmark.models)
    payload = run_benchmark(
        cfg, load_panel(cfg), models, modes, tracker=Tracker(cfg.runs_dir, cfg.profile)
    )
    rows = [
        {
            "model": r["model"],
            "mode": r["mode"],
            "train_samples_s": r["train_samples_per_s"],
            "epoch_s(full)": r["epoch_s_full"],
            "infer_p50_ms": r["infer_latency"]["p50_ms"],
            "infer_p95_ms": r["infer_latency"]["p95_ms"],
            "gpu_peak_mb": r["train_peak_gpu_mem_mb"],
            "gpu_util_%": r["sampled_gpu_util_percent_mean"],
            "cpu_%": r["sampled_cpu_percent_mean"],
            "rss_mb": r["sampled_rss_mb_max"],
        }
        for r in payload["results"]
    ]
    _print_table(pd.DataFrame(rows), "{:,.1f}")
    for s in payload["speedups"]:
        print(json.dumps(s))
    print(f"\nwrote {(cfg.reports_dir / 'benchmark.json').as_posix()}")
    return 0


def cmd_runs(cfg: Config, args: argparse.Namespace) -> int:
    from dcforecast.tracking.registry import ModelRegistry
    from dcforecast.tracking.tracker import Tracker

    tracker = Tracker(cfg.runs_dir, cfg.profile)
    if args.runs_cmd == "list":
        df = tracker.list_runs(kind=args.kind, model=args.model, limit=args.limit)
        if df.empty:
            print("no runs recorded yet")
            return 0
        key = args.metric
        df["metric"] = df["metrics"].map(lambda m: m.get(key))
        cols = ["run_id", "kind", "model", "status", "device", "duration_s", "metric"]
        _print_table(df[cols].rename(columns={"metric": key}))
    elif args.runs_cmd == "compare":
        _print_table(tracker.compare(args.run_ids).reset_index().rename(columns={"index": "field"}))
    elif args.runs_cmd == "show":
        print(json.dumps(to_jsonable(tracker.get_run(args.run_id)), indent=2))
    elif args.runs_cmd == "models":
        df = ModelRegistry(cfg.runs_dir).list_models()
        if df.empty:
            print("no registered models")
            return 0
        df["aliases"] = df["aliases"].map(lambda a: ",".join(a))
        _print_table(df[["name", "version", "run_id", "created_at", "aliases", "path"]])
    return 0


def cmd_promote(cfg: Config, args: argparse.Namespace) -> int:
    from dcforecast.pipeline import promote

    entry = promote(cfg, model=args.model, version=args.version, metric=args.metric)
    print(f"production -> {entry['name']} v{entry['version']} ({entry['reason']})")
    return 0


def cmd_forecast(cfg: Config, args: argparse.Namespace) -> int:
    from dcforecast.data.prepare import load_panel
    from dcforecast.decision.run import forecast_and_rollup
    from dcforecast.tracking.registry import ModelRegistry

    data = load_panel(cfg)
    model, entry = ModelRegistry(cfg.runs_dir).load(args.model, device=args.device)
    if args.site:
        data.history = data.history[data.history["site_id"] == args.site]
        data.future = data.future[data.future["site_id"] == args.site]
        if data.history.empty:
            raise SystemExit(f"unknown site '{args.site}'")
    fc, rollup, _ = forecast_and_rollup(cfg, data, model, args.horizon)
    print(f"model: {entry['name']} v{entry['version']} | origin {fc['origin'].iloc[0]} (UTC)")
    if args.output:
        fc.to_csv(args.output, index=False)
        print(f"wrote {args.output}")
    show = fc[fc["horizon"] <= args.show_hours] if args.show_hours else fc
    _print_table(show[["site_id", "timestamp", "horizon", "p10", "p50", "p90"]], "{:,.1f}")
    print()
    _print_table(rollup[["site_id", "energy_mwh_p50", "cost_usd_p50", "tco2_p50"]], "{:,.2f}")
    return 0


def cmd_schedule(cfg: Config, args: argparse.Namespace) -> int:
    from dcforecast.data.prepare import load_panel
    from dcforecast.decision.run import load_holdout, run_decision, save_decision
    from dcforecast.tracking.registry import ModelRegistry

    data = load_panel(cfg)
    model, entry = ModelRegistry(cfg.runs_dir).load(args.model, device=args.device)
    res = run_decision(
        cfg,
        data,
        model,
        hours=args.hours,
        deferrable_mwh=args.mwh,
        objective=args.objective,
        carbon_price=args.carbon_price,
        holdout=load_holdout(cfg),
        portable_fraction=args.portable_fraction,
    )
    paths = save_decision(res, cfg.reports_dir)
    df = pd.DataFrame(res.comparison)
    cols = ["plan", "facility_mwh", "tco2", "cost_usd", "tco2_saved_pct", "usd_saved_pct"]
    if "realized_tco2_saved_pct" in df:
        cols.append("realized_tco2_saved_pct")
    print(
        f"model: {entry['name']} v{entry['version']}; deferrable {res.problem.volume_mwh.sum():.1f} MWh"
    )
    _print_table(df[cols], "{:,.2f}")
    print(f"\nchosen plan: {res.chosen.name}; wrote {paths['summary'].as_posix()}")
    return 0


def cmd_serve(cfg: Config, args: argparse.Namespace) -> int:
    import uvicorn

    from dcforecast.api.app import create_app

    if args.bootstrap:
        from dcforecast.tracking.registry import PRODUCTION, ModelNotFoundError, ModelRegistry

        try:
            ModelRegistry(cfg.runs_dir).resolve(PRODUCTION)
        except ModelNotFoundError:
            log.info("no production model yet -- bootstrapping a quick pipeline (gbm)")
            _bootstrap(cfg)
    host = args.host or cfg.serve.host
    port = args.port or cfg.serve.port
    uvicorn.run(create_app(cfg), host=host, port=port, log_level="info")
    return 0


def _bootstrap(cfg: Config) -> None:
    from dcforecast.data.generate import run_generate
    from dcforecast.data.prepare import load_panel, prepare
    from dcforecast.pipeline import train_models
    from dcforecast.tracking.registry import PRODUCTION, ModelRegistry

    run_generate(cfg)
    prepare(cfg)
    train_models(cfg, load_panel(cfg), ["gbm"], device="cpu")
    ModelRegistry(cfg.runs_dir).set_alias(PRODUCTION, "gbm", reason="bootstrap")


def cmd_report(cfg: Config, args: argparse.Namespace) -> int:
    from dcforecast.report.build import build_report

    paths = build_report(cfg)
    for key, path in paths.items():
        print(f"{key:<22} {path.as_posix()}")
    return 0


def cmd_run_all(cfg: Config, args: argparse.Namespace) -> int:
    from dcforecast.pipeline import run_all

    out = run_all(cfg, device=args.device, skip_benchmark=args.skip_benchmark)
    print(json.dumps(to_jsonable(out), indent=2))
    return 0


# ---------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dcforecast",
        description="Helios Cloud data-center power forecasting and carbon-aware scheduling.",
    )
    p.add_argument("--version", action="version", version=f"dcforecast {__version__}")
    p.add_argument(
        "-c",
        "--config",
        default=None,
        help="profile name (default, tiny) or YAML path; env DCF_CONFIG also works",
    )
    p.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config value, e.g. --set deep.epochs=5 (repeatable)",
    )
    p.add_argument("--log-level", default="INFO", help="DEBUG, INFO, WARNING (default INFO)")
    sub = p.add_subparsers(dest="command", required=True, metavar="<command>")

    s = sub.add_parser("hw", help="print CPU / RAM / GPU information")
    s.add_argument("--json", action="store_true", help="machine-readable output")
    s.set_defaults(func=cmd_hw)

    sub.add_parser("generate", help="generate the synthetic raw dataset").set_defaults(
        func=cmd_generate
    )
    sub.add_parser("prepare", help="clean telemetry, build features, QA report").set_defaults(
        func=cmd_prepare
    )

    s = sub.add_parser("train", help="train final models on all history and register them")
    s.add_argument("--model", default=None, help="comma-separated names or 'all'")
    s.add_argument("--device", choices=DEVICE_CHOICES, default=None)
    s.add_argument("--no-warm-start", action="store_true", help="train deep models from scratch")
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("backtest", help="rolling-origin backtest + leaderboard")
    s.add_argument("--model", default=None, help="comma-separated names or 'all'")
    s.add_argument("--device", choices=DEVICE_CHOICES, default=None)
    s.set_defaults(func=cmd_backtest)

    s = sub.add_parser("benchmark", help="CPU vs GPU (fp32 / AMP) training + inference benchmark")
    s.add_argument("--model", default=None, help="deep models, e.g. lstm,tcn")
    s.add_argument("--modes", default=None, help="comma-separated: cpu,cuda_fp32,cuda_amp")
    s.set_defaults(func=cmd_benchmark)

    s = sub.add_parser("runs", help="experiment tracker: list / compare / show runs, models")
    rs = s.add_subparsers(dest="runs_cmd", required=True, metavar="<runs-command>")
    r = rs.add_parser("list", help="recent runs")
    r.add_argument("--kind", default=None, help="train, backtest, benchmark")
    r.add_argument("--model", default=None)
    r.add_argument("--limit", type=int, default=30)
    r.add_argument("--metric", default="mase", help="metric column to display")
    r = rs.add_parser("compare", help="side-by-side metrics of runs")
    r.add_argument("run_ids", nargs="+")
    r = rs.add_parser("show", help="one run as JSON")
    r.add_argument("run_id")
    rs.add_parser("models", help="model registry with aliases")
    s.set_defaults(func=cmd_runs)

    s = sub.add_parser(
        "promote", help="set the 'production' alias (best backtest model by default)"
    )
    s.add_argument("--model", default=None)
    s.add_argument("--version", type=int, default=None)
    s.add_argument("--metric", default=None, help="selection metric (default: mase)")
    s.set_defaults(func=cmd_promote)

    s = sub.add_parser("forecast", help="forecast the next hours with a registered model")
    s.add_argument("--site", default=None, help="one site id (default: all)")
    s.add_argument("--horizon", type=int, default=24, help="hours ahead (<= 72)")
    s.add_argument("--model", default="production", help="alias, name or name:version")
    s.add_argument("--device", choices=DEVICE_CHOICES, default="cpu")
    s.add_argument("--show-hours", type=int, default=6, help="rows per site to print (0 = all)")
    s.add_argument("--output", default=None, help="write the full forecast to this CSV")
    s.set_defaults(func=cmd_forecast)

    s = sub.add_parser("schedule", help="carbon-aware placement of deferrable compute")
    s.add_argument("--mwh", type=float, default=None, help="deferrable IT energy (MWh)")
    s.add_argument("--hours", type=int, default=None, help="decision window (hours)")
    s.add_argument("--objective", choices=["carbon", "cost", "blended"], default=None)
    s.add_argument("--carbon-price", type=float, default=None, help="$/tCO2 for 'blended'")
    s.add_argument("--portable-fraction", type=float, default=None)
    s.add_argument("--model", default="production")
    s.add_argument("--device", choices=DEVICE_CHOICES, default="cpu")
    s.set_defaults(func=cmd_schedule)

    s = sub.add_parser("serve", help="run the FastAPI service (uvicorn)")
    s.add_argument("--host", default=None)
    s.add_argument("--port", type=int, default=None)
    s.add_argument(
        "--bootstrap", action="store_true", help="run a quick pipeline first if no model exists"
    )
    s.set_defaults(func=cmd_serve)

    sub.add_parser("report", help="build docs/index.html and docs/images/*.png").set_defaults(
        func=cmd_report
    )

    s = sub.add_parser("run-all", help="the whole pipeline end to end")
    s.add_argument("--device", choices=DEVICE_CHOICES, default=None)
    s.add_argument("--skip-benchmark", action="store_true")
    s.set_defaults(func=cmd_run_all)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.log_level)
    try:
        cfg = load_config(args.config, args.overrides)
        return int(args.func(cfg, args) or 0)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    except (FileNotFoundError, LookupError, ValueError) as exc:
        # one clear line by default; the full traceback with --log-level DEBUG
        log.error("%s", exc, exc_info=log.isEnabledFor(logging.DEBUG))
        return 1
    except KeyboardInterrupt:  # pragma: no cover
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
