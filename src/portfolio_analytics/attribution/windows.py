"""Attribution windows: validated research calculations."""

from __future__ import annotations

import pandas as pd



# Research reference: 08_portfolio_attribution.ipynb, code cell index 12.
def build_attribution_reporting_windows(
    daily_returns: pd.DataFrame,
    *,
    portfolio_id: str,
    benchmark_id: str,
    inception_valuation_date: str | pd.Timestamp,
    trailing_years: tuple[int, ...] = (3, 5, 10),
) -> pd.DataFrame:
    """
    Construct calendar and anniversary-based attribution windows.

    Output grain:
        portfolio_id, benchmark_id, window_id.

    start_date:
        First included daily return interval end.

    valuation_anchor_date:
        Portfolio valuation date immediately preceding the
        first included return interval.

    coverage_status:
        FULL, PARTIAL_START, PARTIAL_END, PARTIAL_BOTH,
        or INSUFFICIENT_HISTORY.

    FULL indicates sample-window eligibility, not external
    exchange-calendar certification.
    """

    daily = daily_returns.loc[
        daily_returns["portfolio_id"].eq(portfolio_id)
        & daily_returns["benchmark_id"].eq(benchmark_id),
        ["date"],
    ].copy()

    dates = pd.DatetimeIndex(
        pd.to_datetime(daily["date"]).sort_values()
    )

    inception = pd.Timestamp(
        inception_valuation_date
    ).normalize()

    if (
        dates.empty
        or dates.has_duplicates
        or dates.hasnans
        or inception >= dates.min()
    ):
        raise ValueError(
            "Invalid return calendar or inception valuation date."
        )

    cutoff = dates.max()
    records = []

    def add_window(
        window_id,
        window_type,
        window_label,
        calendar_start,
        calendar_end,
        anchor,
        first_date,
        last_date,
        observations,
        coverage,
    ):
        records.append(
            {
                "portfolio_id": portfolio_id,
                "benchmark_id": benchmark_id,
                "window_id": window_id,
                "window_type": window_type,
                "window_label": window_label,
                "calendar_start": calendar_start,
                "calendar_end": calendar_end,
                "valuation_anchor_date": anchor,
                "start_date": first_date,
                "end_date": last_date,
                "observations": observations,
                "coverage_status": coverage,
                "is_full_period": coverage == "FULL",
            }
        )

    # --------------------------------------------------------
    # 1. Monthly, quarterly and annual calendar windows
    # --------------------------------------------------------

    for window_type, frequency in (
        ("MONTHLY", "M"),
        ("QUARTERLY", "Q"),
        ("ANNUAL", "Y"),
    ):
        periods = dates.to_period(frequency)

        for period in periods.unique():
            period_dates = dates[periods == period]

            calendar_start = (
                period.start_time.normalize()
            )

            calendar_end = (
                period.end_time.normalize()
            )

            previous_position = (
                dates.searchsorted(
                    calendar_start,
                    side="left",
                )
                - 1
            )

            anchor = (
                dates[previous_position]
                if previous_position >= 0
                else inception
            )

            partial_start = (
                inception
                > calendar_start - pd.Timedelta(days=1)
            )

            partial_end = calendar_end > cutoff

            if partial_start and partial_end:
                status = "PARTIAL_BOTH"
            elif partial_start:
                status = "PARTIAL_START"
            elif partial_end:
                status = "PARTIAL_END"
            else:
                status = "FULL"

            add_window(
                f"{window_type}:{period}",
                window_type,
                str(period),
                calendar_start,
                calendar_end,
                anchor,
                period_dates.min(),
                period_dates.max(),
                len(period_dates),
                status,
            )

    # --------------------------------------------------------
    # 2. Trailing calendar-anniversary windows
    # --------------------------------------------------------
    #
    # Example:
    #   Reporting cutoff = 2026-09-30
    #   3Y anniversary   = 2023-09-30 (Saturday)
    #   Valuation anchor = 2023-09-29 (previous trading date)
    #   First return     = 2023-10-02
    #
    # The starting valuation date is excluded from returns.
    # --------------------------------------------------------

    for years in trailing_years:
        if not isinstance(years, int) or years <= 0:
            raise ValueError(
                "Trailing horizons must be positive integer years."
            )

        anniversary = cutoff - pd.DateOffset(
            years=years
        )

        previous_position = (
            dates.searchsorted(
                anniversary,
                side="right",
            )
            - 1
        )

        anchor = (
            dates[previous_position]
            if previous_position >= 0
            else inception
        )

        if anchor < inception:
            anchor = inception

        eligible = inception <= anniversary

        selected_dates = dates[dates > anchor]

        if selected_dates.empty:
            raise ValueError(
                "Trailing reporting window contains no returns."
            )

        add_window(
            f"TRAILING:{years}Y",
            "TRAILING",
            f"{years}Y",
            anniversary,
            cutoff,
            anchor,
            selected_dates.min(),
            selected_dates.max(),
            len(selected_dates),
            (
                "FULL"
                if eligible
                else "INSUFFICIENT_HISTORY"
            ),
        )

    # --------------------------------------------------------
    # 3. Full-history attribution window
    # --------------------------------------------------------

    add_window(
        "FULL_HISTORY",
        "FULL_HISTORY",
        "FULL_HISTORY",
        inception,
        cutoff,
        inception,
        dates.min(),
        cutoff,
        len(dates),
        "FULL",
    )

    return (
        pd.DataFrame(records)
        .sort_values(
            ["window_type", "calendar_start", "window_id"]
        )
        .reset_index(drop=True)
    )
