"""The six (fictional) Helios Cloud data-center sites and their physical parameters.

All names, places and numbers are invented. Parameters are chosen to span very different
climates (sub-arctic to desert to tropical), grid mixes (hydro to coal) and tariffs, which is
what makes carbon-aware scheduling across sites interesting.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Site:
    """Physical, climate, grid and tariff parameters of one site."""

    site_id: str
    name: str
    climate: str
    grid_mix: str
    utc_offset_h: int
    # --- IT load ---------------------------------------------------------------
    it_capacity_kw: float
    base_utilization: float
    annual_growth: float
    business_amp: float
    weekend_drop: float
    batch_amp: float
    batch_start_local: int
    batch_hours: int
    # --- climate ---------------------------------------------------------------
    temp_mean_c: float
    temp_annual_amp_c: float
    temp_diurnal_amp_c: float
    temp_peak_doy: int
    temp_noise_c: float
    rh_mean_pct: float
    rh_annual_amp_pct: float
    rh_diurnal_amp_pct: float
    # --- cooling / PUE ---------------------------------------------------------
    free_cooling_c: float
    cooling_floor: float
    chiller_slope: float
    humidity_coef: float
    fixed_overhead_frac: float
    # --- grid carbon intensity (gCO2/kWh) ----------------------------------------
    ci_base: float
    ci_solar_share: float
    ci_evening_peak: float
    ci_seasonal_amp: float
    ci_noise: float
    # --- time-of-use tariff ($/MWh) ----------------------------------------------
    price_offpeak: float
    price_mid: float
    price_peak: float
    peak_start_local: int
    peak_end_local: int
    summer_peak_mult: float
    # --- contracted facility power cap (kW), used for scheduling headroom ---------
    power_cap_kw: float

    @property
    def southern_hemisphere(self) -> bool:
        return self.temp_peak_doy < 90 or self.temp_peak_doy > 300

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


SITES: dict[str, Site] = {
    s.site_id: s
    for s in [
        Site(
            site_id="frostholm",
            name="Frostholm",
            climate="sub-arctic",
            grid_mix="hydro-dominated",
            utc_offset_h=1,
            it_capacity_kw=12_000,
            base_utilization=0.50,
            annual_growth=0.09,
            business_amp=0.10,
            weekend_drop=0.05,
            batch_amp=0.07,
            batch_start_local=1,
            batch_hours=4,
            temp_mean_c=3.0,
            temp_annual_amp_c=11.0,
            temp_diurnal_amp_c=3.0,
            temp_peak_doy=200,
            temp_noise_c=3.5,
            rh_mean_pct=80.0,
            rh_annual_amp_pct=6.0,
            rh_diurnal_amp_pct=8.0,
            free_cooling_c=17.0,
            cooling_floor=0.06,
            chiller_slope=0.010,
            humidity_coef=0.0008,
            fixed_overhead_frac=0.035,
            ci_base=38.0,
            ci_solar_share=0.04,
            ci_evening_peak=0.12,
            ci_seasonal_amp=0.25,
            ci_noise=0.10,
            price_offpeak=42.0,
            price_mid=58.0,
            price_peak=88.0,
            peak_start_local=16,
            peak_end_local=20,
            summer_peak_mult=1.0,
            power_cap_kw=14_500,
        ),
        Site(
            site_id="dunehaven",
            name="Dunehaven",
            climate="hot desert",
            grid_mix="solar + gas",
            utc_offset_h=4,
            it_capacity_kw=18_000,
            base_utilization=0.48,
            annual_growth=0.12,
            business_amp=0.12,
            weekend_drop=0.06,
            batch_amp=0.06,
            batch_start_local=0,
            batch_hours=4,
            temp_mean_c=27.0,
            temp_annual_amp_c=8.5,
            temp_diurnal_amp_c=6.5,
            temp_peak_doy=205,
            temp_noise_c=2.0,
            rh_mean_pct=28.0,
            rh_annual_amp_pct=8.0,
            rh_diurnal_amp_pct=10.0,
            free_cooling_c=18.0,
            cooling_floor=0.08,
            chiller_slope=0.0115,
            humidity_coef=0.0010,
            fixed_overhead_frac=0.04,
            ci_base=470.0,
            ci_solar_share=0.55,
            ci_evening_peak=0.10,
            ci_seasonal_amp=0.08,
            ci_noise=0.07,
            price_offpeak=55.0,
            price_mid=78.0,
            price_peak=145.0,
            peak_start_local=14,
            peak_end_local=19,
            summer_peak_mult=1.25,
            power_cap_kw=22_500,
        ),
        Site(
            site_id="marisport",
            name="Marisport",
            climate="temperate maritime",
            grid_mix="wind + gas",
            utc_offset_h=0,
            it_capacity_kw=15_000,
            base_utilization=0.55,
            annual_growth=0.11,
            business_amp=0.15,
            weekend_drop=0.08,
            batch_amp=0.08,
            batch_start_local=1,
            batch_hours=5,
            temp_mean_c=11.0,
            temp_annual_amp_c=6.0,
            temp_diurnal_amp_c=3.5,
            temp_peak_doy=210,
            temp_noise_c=2.5,
            rh_mean_pct=78.0,
            rh_annual_amp_pct=5.0,
            rh_diurnal_amp_pct=9.0,
            free_cooling_c=17.0,
            cooling_floor=0.07,
            chiller_slope=0.011,
            humidity_coef=0.0012,
            fixed_overhead_frac=0.04,
            ci_base=230.0,
            ci_solar_share=0.15,
            ci_evening_peak=0.15,
            ci_seasonal_amp=0.15,
            ci_noise=0.28,
            price_offpeak=68.0,
            price_mid=96.0,
            price_peak=152.0,
            peak_start_local=16,
            peak_end_local=20,
            summer_peak_mult=1.0,
            power_cap_kw=18_500,
        ),
        Site(
            site_id="ironvale",
            name="Ironvale",
            climate="continental",
            grid_mix="coal-heavy",
            utc_offset_h=-6,
            it_capacity_kw=20_000,
            base_utilization=0.52,
            annual_growth=0.07,
            business_amp=0.13,
            weekend_drop=0.07,
            batch_amp=0.07,
            batch_start_local=0,
            batch_hours=4,
            temp_mean_c=8.0,
            temp_annual_amp_c=14.0,
            temp_diurnal_amp_c=5.5,
            temp_peak_doy=200,
            temp_noise_c=3.5,
            rh_mean_pct=62.0,
            rh_annual_amp_pct=8.0,
            rh_diurnal_amp_pct=12.0,
            free_cooling_c=16.0,
            cooling_floor=0.07,
            chiller_slope=0.012,
            humidity_coef=0.0012,
            fixed_overhead_frac=0.04,
            ci_base=680.0,
            ci_solar_share=0.10,
            ci_evening_peak=0.06,
            ci_seasonal_amp=0.08,
            ci_noise=0.05,
            price_offpeak=40.0,
            price_mid=55.0,
            price_peak=98.0,
            peak_start_local=15,
            peak_end_local=20,
            summer_peak_mult=1.2,
            power_cap_kw=24_000,
        ),
        Site(
            site_id="tidewater",
            name="Tidewater Key",
            climate="tropical humid",
            grid_mix="gas-heavy",
            utc_offset_h=8,
            it_capacity_kw=14_000,
            base_utilization=0.46,
            annual_growth=0.16,
            business_amp=0.14,
            weekend_drop=0.06,
            batch_amp=0.06,
            batch_start_local=1,
            batch_hours=4,
            temp_mean_c=27.5,
            temp_annual_amp_c=1.8,
            temp_diurnal_amp_c=3.5,
            temp_peak_doy=130,
            temp_noise_c=1.2,
            rh_mean_pct=81.0,
            rh_annual_amp_pct=5.0,
            rh_diurnal_amp_pct=10.0,
            free_cooling_c=18.0,
            cooling_floor=0.09,
            chiller_slope=0.010,
            humidity_coef=0.0022,
            fixed_overhead_frac=0.045,
            ci_base=510.0,
            ci_solar_share=0.18,
            ci_evening_peak=0.08,
            ci_seasonal_amp=0.04,
            ci_noise=0.05,
            price_offpeak=86.0,
            price_mid=112.0,
            price_peak=168.0,
            peak_start_local=11,
            peak_end_local=17,
            summer_peak_mult=1.0,
            power_cap_kw=17_500,
        ),
        Site(
            site_id="highmesa",
            name="High Mesa",
            climate="high-altitude semi-arid (southern hemisphere)",
            grid_mix="solar + hydro",
            utc_offset_h=-4,
            it_capacity_kw=10_000,
            base_utilization=0.54,
            annual_growth=0.12,
            business_amp=0.12,
            weekend_drop=0.06,
            batch_amp=0.08,
            batch_start_local=2,
            batch_hours=4,
            temp_mean_c=12.0,
            temp_annual_amp_c=7.5,
            temp_diurnal_amp_c=8.0,
            temp_peak_doy=20,
            temp_noise_c=2.2,
            rh_mean_pct=38.0,
            rh_annual_amp_pct=10.0,
            rh_diurnal_amp_pct=14.0,
            free_cooling_c=18.0,
            cooling_floor=0.06,
            chiller_slope=0.011,
            humidity_coef=0.0008,
            fixed_overhead_frac=0.035,
            ci_base=330.0,
            ci_solar_share=0.45,
            ci_evening_peak=0.12,
            ci_seasonal_amp=0.10,
            ci_noise=0.08,
            price_offpeak=50.0,
            price_mid=70.0,
            price_peak=122.0,
            peak_start_local=17,
            peak_end_local=21,
            summer_peak_mult=1.1,
            power_cap_kw=12_000,
        ),
    ]
}


def get_sites(site_ids: list[str] | None = None) -> list[Site]:
    """Return the selected sites (all six when ``site_ids`` is None)."""
    if not site_ids:
        return list(SITES.values())
    unknown = [s for s in site_ids if s not in SITES]
    if unknown:
        raise KeyError(f"Unknown site(s): {', '.join(unknown)}. Known: {', '.join(SITES)}")
    return [SITES[s] for s in site_ids]


SITE_ORDER: list[str] = list(SITES)


def site_index(site_id: str) -> int:
    """Stable integer code of a site (used for categorical features and embeddings)."""
    return SITE_ORDER.index(site_id)
