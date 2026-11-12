"""Inclusive complete-month reporting intervals, separate from model lookbacks."""

import calendar
from dataclasses import dataclass

import pandas as pd

FIRST_MONTH = pd.Period("2016-10", freq="M")
LAST_MONTH = pd.Period("2026-09", freq="M")
MINIMUM_MONTHS = 12


def month_number(name):
    """Accept an English month name or abbreviation, without fuzzy date parsing."""
    if not isinstance(name, str):
        raise ValueError("Supply a month name, for example January or Jan.")
    names = {value.casefold(): i for i in range(1, 13)
             for value in (calendar.month_name[i], calendar.month_abbr[i])}
    try:
        return names[name.strip().casefold()]
    except KeyError as exc:
        raise ValueError(f"Invalid month name: {name!r}.") from exc


@dataclass(frozen=True)
class MonthWindow:
    start: pd.Period
    end: pd.Period

    def __post_init__(self):
        if any(not isinstance(p, pd.Period) or p.freqstr != "M" for p in (self.start, self.end)):
            raise ValueError("Month boundaries must be monthly periods.")
        if self.start < FIRST_MONTH or self.end > LAST_MONTH:
            raise ValueError("Select complete months between October 2016 and September 2026.")
        if self.months < MINIMUM_MONTHS:
            raise ValueError("Select at least 12 inclusive months; January through December is 12.")

    @classmethod
    def from_names(cls, start_month, start_year, end_month, end_year):
        for year in (start_year, end_year):
            if type(year) is not int or not 1 <= year <= 9999:
                raise ValueError("Years must be calendar-year integers.")
        return cls(pd.Period(year=start_year, month=month_number(start_month), freq="M"),
                   pd.Period(year=end_year, month=month_number(end_month), freq="M"))

    @property
    def months(self):
        return self.end.ordinal - self.start.ordinal + 1

    @property
    def interval_id(self):
        return f"{self.start}_{self.end}"

    def resolve(self, dates, inception_date):
        """Keep the prior close as opening valuation; include the first earned return."""
        dates = pd.DatetimeIndex(dates)
        if dates.empty or dates.hasnans or not dates.is_unique or not dates.is_monotonic_increasing:
            raise ValueError("A sorted unique nonmissing return calendar is required.")
        if dates.tz is not None or not dates.equals(dates.normalize()):
            raise ValueError("Return sessions must be timezone-free calendar dates.")
        first = self.start.start_time.normalize()
        last = self.end.end_time.normalize()
        left, right = dates.searchsorted(first), dates.searchsorted(last, side="right")
        included = dates[left:right]
        expected = pd.period_range(self.start, self.end, freq="M")
        if included.empty or not included.to_period("M").unique().equals(expected):
            raise ValueError("The selected interval is missing one or more complete reporting months.")
        opening = dates[left - 1] if left else pd.Timestamp(inception_date)
        if opening >= included[0] or opening.to_period("M") != self.start - 1:
            raise ValueError("The last available close in the preceding month is required.")
        return dict(interval_id=self.interval_id, start_month=str(self.start), end_month=str(self.end),
                    inclusive_months=self.months, calendar_start=first, calendar_end=last,
                    opening_valuation_date=opening, first_return_date=included[0],
                    last_return_date=included[-1], observation_count=len(included),
                    start_index=int(left), stop_index=int(right),
                    boundary_policy="PRIOR_MONTH_CLOSE_THROUGH_END_MONTH_CLOSE")


def enumerate_month_windows(dates, inception_date):
    """Enumerate every >=12-month selection in the approved ten-year history."""
    months = pd.period_range(FIRST_MONTH, LAST_MONTH, freq="M")
    return pd.DataFrame([
        MonthWindow(months[i], months[j]).resolve(dates, inception_date)
        for j in range(MINIMUM_MONTHS - 1, len(months))
        for i in range(j - MINIMUM_MONTHS + 2)
    ])
