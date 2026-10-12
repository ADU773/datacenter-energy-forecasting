# Synthetic data: the generative model

Helios Cloud is a **fictional** cloud operator. Its telemetry is produced by
`dcforecast generate` (`src/dcforecast/data/generate.py`) from a small, physically-motivated
model so that every effect a forecaster has to learn has a known cause. The run is fully
reproducible from `seed` (default 42); each site draws from its own random stream
(`default_rng([seed, site_index])`), so generating a subset of sites gives identical series.

Default size: 6 sites x 1,096 days x 24 h = **157,824 site-hours** of history (2023-01-01 to
2025-12-31, UTC) plus a 7-day future feed. On disk this is about 30 MB of Parquet.

## Sites

| Site | Climate | Grid mix | UTC offset | IT capacity (MW) | Power cap (MW) | Mean temp (C) | Free-cooling limit (C) | Base carbon (g/kWh) | Tariff off/mid/peak ($/MWh) | Growth / yr |
|---|---|---|---|---|---|---|---|---|---|---|
| Frostholm | sub-arctic | hydro-dominated | +1 | 12 | 14.5 | 3.0 | 17 | 38 | 42 / 58 / 88 | 9% |
| Dunehaven | hot desert | solar + gas | +4 | 18 | 22.5 | 27.0 | 18 | 470 | 55 / 78 / 145 | 12% |
| Marisport | temperate maritime | wind + gas | +0 | 15 | 18.5 | 11.0 | 17 | 230 | 68 / 96 / 152 | 11% |
| Ironvale | continental | coal-heavy | -6 | 20 | 24.0 | 8.0 | 16 | 680 | 40 / 55 / 98 | 7% |
| Tidewater Key | tropical humid | gas-heavy | +8 | 14 | 17.5 | 27.5 | 18 | 510 | 86 / 112 / 168 | 16% |
| High Mesa | semi-arid, southern hemisphere | solar + hydro | -4 | 10 | 12.0 | 12.0 | 18 | 330 | 50 / 70 / 122 | 12% |

All parameters live in `src/dcforecast/data/sites.py`.

## 1. Weather (per site)

With `h` the local hour and `doy` the day of year:

```
season(t)  = cos(2 pi (doy - peak_doy) / 365.25)          # +1 on the warmest day
diurnal(t) = cos(2 pi (h - 15) / 24)                       # +1 at 15:00 local
w(t)       = AR(1), phi = 0.985, sigma = temp_noise        # synoptic weather systems (days)
temp(t)    = mean + A_year * season + A_day * (1 + 0.25 season) * diurnal + w(t) + AR(1)(0.6, 0.35)
rh(t)      = rh_mean - B_year * season - B_day * diurnal - 1.2 w(t) + AR(1)(0.95, 5)   clipped to [4, 100]
```

AR(1) noise `x[t] = phi x[t-1] + e[t]` (stationary start) gives realistic multi-day warm or cold
spells. High Mesa peaks in January (southern hemisphere).

## 2. IT load

```
level(t)   = base_util * capacity * (1 + growth)^years * launches(t) * (1 + 0.025 cos(annual))
business   = business_amp * exp(-((h - 13.5) / 3.2)^2 / 2) * (1 on workdays, 0.3 otherwise)
weekday    = 1 - weekend_drop on weekends and holidays
batch      = batch_amp * raised-cosine window starting at batch_start_local for batch_hours
             (x1.8 on the last day of the month: month-end reporting jobs)
it(t)      = level * weekday * (1 + business + batch) * (1 + AR(1)(0.85, 0.012))
```

* **Product launches**: a Poisson number (0.8 per year) of logistic step-ups of +2-6 % that
  ramp in over 3-10 days.
* **Incidents**: Poisson (4 per year) drops of 12-40 % lasting 2-8 hours.
* **Holidays**: a fixed set of fictional public holidays (New Year, Christmas, nth-weekday
  holidays) computed in each site's local time (`data/calendar.py`).
* IT load is clipped to [8 %, 97 %] of capacity.

## 3. PUE and facility power

Cooling responds to a thermally-lagged temperature (exponential smoothing, `alpha = 0.4`):

```
excess   = temp_eff - free_cooling_limit
cooling  = floor + chiller_slope * softplus_hinge(excess) + humidity_coef * max(rh - 50, 0) * sigmoid(excess / 2)
fixed    = fixed_overhead_frac * capacity / it          # overheads weigh more at low utilisation
PUE      = clip(1 + cooling + fixed + AR(1)(0.9, 0.004), 1.03, 2.5)
facility = it * PUE                                      # the forecast target (kW, hourly mean)
```

Below the free-cooling limit PUE stays near the floor (outside air / economisers); above it the
chillers switch in and PUE climbs roughly linearly. The `pue_vs_temperature.png` chart shows the
two regimes per site.

## 4. Grid carbon intensity and tariff

```
sun      = max(sin(pi (h + 0.5 - 6) / 12), 0)^1.3
solar    = sun * (0.75 + 0.25 solar_season) * (0.45 + 0.55 clear_sky(t))     # clear_sky = sigmoid(AR(1))
evening  = exp(-((h - 19.5) / 1.8)^2 / 2)
CI(t)    = base * (1 + seasonal_amp * winter) * (1 - solar_share * solar) * (1 + evening_peak * evening) * exp(AR(1))
```

This gives the typical "duck curve": a midday solar dip and an evening peak, whose depth depends
on each site's mix (Frostholm's hydro grid is clean all day; Ironvale's coal grid barely moves).

The time-of-use tariff has off-peak, shoulder and peak prices on workdays (peak windows are
site-specific and 15-45 % more expensive in the hot season), and a short evening shoulder on
weekends.

## 5. Forecast feeds (known at prediction time)

The models may use weather and carbon intensity *at the forecast horizon*, because operators buy
day-ahead weather forecasts and grid operators publish carbon forecasts. To avoid an unrealistic
advantage the generator stores **noisy forecasts**, not the truth:

```
temp_fcst = temp + AR(1)(0.92, 1.3 C)
rh_fcst   = rh   + AR(1)(0.92, 5 %)
ci_fcst   = ci * exp(AR(1)(0.9, 0.04 + 0.3 ci_noise))
```

Models only ever see `temp_fcst_c`, `rh_fcst_pct` and `carbon_fcst_g_per_kwh` for future hours.
Measured temperature and humidity are used only as *past* inputs.

## 6. Sensor faults and cleaning

`data/quality.py` corrupts the fault-free series (per site and metered column: IT load,
facility power, temperature, humidity) with:

| Fault | Default rate |
|---|---|
| multi-hour gaps | 10 episodes / year, up to 18 h |
| isolated missing readings | 0.2 % of readings |
| spikes (surges and drop-outs) | 0.15 % of readings |
| stuck meter (value frozen) | 4 episodes / year, up to 36 h |
| data-logger outage | whole rows missing |

The ground truth of every injected fault is written to `data/raw/fault_log.parquet`.

`dcforecast prepare` then detects and repairs them **without looking at the future**:

1. range checks (negative power, > 160 % of capacity, impossible weather),
2. stuck detection: runs of identical readings (>= 3 h for power, >= 6 h for weather; saturated
   100 % humidity is allowed),
3. spike detection: a two-sided neighbour test with robust (MAD) thresholds,
4. causal imputation: cross-sensor first (facility = IT x trailing-median PUE, or the inverse),
   then level-adjusted seasonal persistence (`y[t] = y[t-L] * y[last] / y[last-L]`, L = 168 h
   for power, additive with L = 24 h for weather), then last-observation-carried-forward.

Because the truth is known, `reports/qa_report.json` measures the cleaning itself: detection
precision and recall per fault type and the imputation error (MAE / MAPE) on repaired cells.
Imputed targets are masked out of the deep models' loss.
