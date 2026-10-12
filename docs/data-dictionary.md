# Data dictionary

All timestamps are hourly, timezone-naive **UTC**, and mark the start of the hour; power values
are hourly means in kW. Generated files live under `data/` (gitignored) and are rebuilt by
`dcforecast generate` and `dcforecast prepare`.

## Raw layer (`data/raw/`, written by `generate`)

| File | Grain | Columns | Notes |
|---|---|---|---|
| `telemetry.parquet` | site x hour (history) | `timestamp`, `site_id`, `it_load_kw`, `facility_kw`, `temp_c`, `rh_pct` | **With** injected sensor faults; outage hours are missing rows |
| `weather_forecast.parquet` | site x hour (history + future) | `timestamp`, `site_id`, `temp_fcst_c`, `rh_fcst_pct` | Day-ahead weather forecast feed (noisy) |
| `grid.parquet` | site x hour (history + future) | `timestamp`, `site_id`, `carbon_g_per_kwh`*, `carbon_fcst_g_per_kwh`, `price_usd_per_mwh` | *Actual carbon intensity for history hours only |
| `fault_log.parquet` | corrupted cell | `site_id`, `timestamp`, `column`, `fault` (gap, missing, spike, stuck, outage), `true_value`, `observed_value` | Ground truth used to score the cleaning step |
| `truth.parquet` | site x hour (history) | fault-free version of every simulated column incl. `pue` | Only used for QA scoring |
| `holdout_future.parquet` | site x hour (future) | true future values | Hidden from all models; used to audit decisions afterwards |
| `manifest.json` | run | seed, date range, sites and their parameters, row counts | |

## Processed layer (`data/processed/`, written by `prepare`)

### `panel.parquet` - clean history (one row per site and hour, no gaps)

| Column | Type | Unit | Description |
|---|---|---|---|
| `timestamp` | datetime | UTC hour | Start of the hour |
| `site_id` | string | - | `frostholm`, `dunehaven`, `marisport`, `ironvale`, `tidewater`, `highmesa` |
| `it_load_kw` | float | kW | IT (server) power after cleaning |
| `facility_kw` | float | kW | Total facility power = IT x PUE. **Forecast target** |
| `pue` | float | ratio | Power usage effectiveness = facility / IT (>= 1) |
| `temp_c` | float | deg C | Measured outside temperature (cleaned) |
| `rh_pct` | float | % | Measured relative humidity (cleaned) |
| `it_imputed`, `facility_imputed`, `temp_imputed`, `rh_imputed` | bool | - | True where the value was detected as bad or missing and imputed |
| `temp_fcst_c`, `rh_fcst_pct` | float | deg C, % | Day-ahead weather forecast for that hour |
| `carbon_g_per_kwh` | float | gCO2/kWh | Actual grid carbon intensity |
| `carbon_fcst_g_per_kwh` | float | gCO2/kWh | Day-ahead carbon-intensity forecast |
| `price_usd_per_mwh` | float | $/MWh | Time-of-use electricity tariff |

### `future.parquet` - known-future covariates

Same keys plus `temp_fcst_c`, `rh_fcst_pct`, `carbon_fcst_g_per_kwh`, `price_usd_per_mwh` for
the hours after the end of history. It deliberately contains **no** measured values.

### `features.parquet` - tabular feature matrix (gradient boosting)

| Group | Columns | Leakage rule |
|---|---|---|
| Site | `site_code` | static |
| Calendar (local time) | `local_hour`, `dow`, `month`, `is_weekend`, `is_holiday`, `is_workday`, `is_month_end` | deterministic |
| Fourier | `fd_sin/cos{1..3}` (daily), `fw_*` (weekly), `fy_*` (yearly) | deterministic |
| Target lags | `lag24_rel`, `lag48_rel`, `lag168_rel` (divided by `level`) | every lag >= horizon (24 h) |
| Rolling stats | `roll_mean{24,168}_rel`, `roll_std{24,168}_rel` | computed on the target shifted by the horizon (past only) |
| Level | `log_level` = log of the trailing 168 h mean of the 24 h-shifted target | past only |
| Weather | `temp_fcst_c`, `rh_fcst_pct`, `cdh_fcst` (cooling degree hours from the forecast) | forecast feed, available at issue time |
| Physics lags | `pue_lag24`, `temp_lag24` | 24 h old |
| Label | `y_rel` = `facility_kw / level` | never a feature |

The tests in `tests/test_features.py` perturb the target after a cut-off and assert that no
feature before `cut-off + 24 h` changes; `tests/test_models.py` does the same for every model's
predictions.

## Outputs (`reports/`, gitignored; `docs/`, committed)

| File | Content |
|---|---|
| `reports/qa_report.json` | Rows expected/received, flags per reason, detection precision/recall, imputation error, per site |
| `reports/backtest/forecasts.parquet` | Every backtest forecast: `site_id`, `origin`, `timestamp`, `horizon`, `p10`, `p50`, `p90`, `model`, `fold`, `scale`, actual |
| `reports/backtest/leaderboard.json` | Leaderboard overall, per site, per fold, per horizon; folds; fit/predict timings |
| `reports/benchmark.json` | Hardware (host anonymised as `host-01`), per model x mode throughput, latency, GPU memory/utilisation, CPU%, RSS; speed-ups |
| `reports/decision.json` | Energy / cost / carbon roll-up, forecast-vs-realised audit, scheduling plans and savings, Pareto front |
| `reports/decision_plan.csv`, `decision_baseline.csv`, `decision_forecast.csv` | Hourly plan, baseline and forecast behind the decision |
| `reports/pipeline_timings.json`, `reports/resources.json` | Wall-clock per step, peak memory, GPU peak, disk footprint |
| `docs/metrics.json` | Small privacy-safe copy of the headline numbers quoted in the README |

## Experiment tracker (`runs/tracker.db`, SQLite)

| Table | Key | Columns |
|---|---|---|
| `runs` | `run_id` | `kind` (train, backtest, benchmark), `model`, `status`, `started_at`, `finished_at`, `duration_s`, `device`, `git_commit`, `profile`, `params`, `metrics`, `artifacts`, `tags` (JSON text) |
| `models` | (`name`, `version`) | `run_id`, `path` (relative to `runs/`), `created_at`, `metrics` |
| `aliases` | `alias` | `name`, `version`, `updated_at`, `reason` (e.g. `production` -> best backtest MASE) |

Model artefacts live in `runs/registry/<name>/v<version>/` (`meta.json` + payload: `model.pt`,
`model.joblib` or JSON), checkpoints in `runs/checkpoints/`.
