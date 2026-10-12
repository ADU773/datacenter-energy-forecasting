"""Corporate holiday calendar and local-time calendar fields.

All timestamps in the project are UTC. Each site has a fixed UTC offset (no daylight saving),
and every calendar effect (business hours, batch windows, holidays, tariffs, solar position)
is evaluated in *site-local* time.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable
from functools import lru_cache

import numpy as np
import pandas as pd

HOLIDAY_NAMES = {
    "new_year": "New Year's Day",
    "spring": "Spring Holiday (last Monday of May)",
    "labour": "Labour Day (first Monday of September)",
    "harvest": "Harvest Day (fourth Thursday of November) + bridge Friday",
    "winter": "Winter Break (Dec 24, 25, 26, 31)",
}


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> dt.date:
    """n-th (1-based) given weekday of a month; n=-1 means the last one."""
    if n > 0:
        first = dt.date(year, month, 1)
        offset = (weekday - first.weekday()) % 7
        return first + dt.timedelta(days=offset + 7 * (n - 1))
    nxt = dt.date(year + (month == 12), month % 12 + 1, 1)
    last = nxt - dt.timedelta(days=1)
    return last - dt.timedelta(days=(last.weekday() - weekday) % 7)


@lru_cache(maxsize=64)
def _holidays_for_year(year: int) -> frozenset[dt.date]:
    harvest = _nth_weekday(year, 11, 3, 4)
    days = {
        dt.date(year, 1, 1),
        _nth_weekday(year, 5, 0, -1),
        _nth_weekday(year, 9, 0, 1),
        harvest,
        harvest + dt.timedelta(days=1),
        dt.date(year, 12, 24),
        dt.date(year, 12, 25),
        dt.date(year, 12, 26),
        dt.date(year, 12, 31),
    }
    return frozenset(days)


def holiday_dates(years: Iterable[int]) -> set[dt.date]:
    """All Helios corporate holidays for the given years."""
    out: set[dt.date] = set()
    for y in years:
        out |= _holidays_for_year(int(y))
    return out


def local_calendar(timestamps_utc: pd.Series | pd.DatetimeIndex, utc_offset_h: int) -> pd.DataFrame:
    """Local-time calendar fields for UTC timestamps at a site with a fixed UTC offset.

    Returns a frame (same length/order) with ``local_hour``, ``dow`` (Mon=0), ``doy``,
    ``month``, ``is_weekend``, ``is_holiday`` and ``is_workday``.
    """
    ts = pd.DatetimeIndex(timestamps_utc)
    local = ts + pd.Timedelta(hours=utc_offset_h)
    years = np.unique(local.year)
    hol = np.array(sorted(holiday_dates(years)), dtype="datetime64[D]")
    local_days = local.values.astype("datetime64[D]")
    is_hol = np.isin(local_days, hol)
    dow = np.asarray(local.dayofweek)
    is_weekend = dow >= 5
    return pd.DataFrame(
        {
            "local_hour": np.asarray(local.hour, dtype=np.int16),
            "dow": dow.astype(np.int16),
            "doy": np.asarray(local.dayofyear, dtype=np.int16),
            "month": np.asarray(local.month, dtype=np.int16),
            "is_weekend": is_weekend,
            "is_holiday": is_hol,
            "is_workday": ~(is_weekend | is_hol),
        }
    )
