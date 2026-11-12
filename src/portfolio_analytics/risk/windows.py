"""Risk windows: validated research calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd

RISK_FINAL_DDOF = 1

# Research reference: 09_risk_analytics.ipynb, code cell index 15.
def select_calendar_risk_windows(
    aligned_returns: pd.DataFrame,
    as_of_date: pd.Timestamp,
    reporting_years: tuple[int, ...],
) -> dict[str, dict]:
    """
    Select reporting windows using calendar anniversaries.

    Returns are close-to-close observations labeled by
    their return-interval end date.

    A return is included only when:
        anniversary_anchor < return_date <= as_of_date.

    The first observed return date must follow the anchor
    by no more than seven calendar days. This is a basic
    boundary-coverage diagnostic, not certification of
    the underlying exchange calendar.

    Incomplete anniversary coverage is rejected rather
    than relabeled as a complete reporting period.
    """

    required = {
        "date",
        "portfolio_id",
        "benchmark_portfolio_id",
        "portfolio_return_total",
        "benchmark_return_total",
        "active_return_arithmetic",
    }

    if not required.issubset(aligned_returns.columns):
        raise ValueError(
            "Missing required aligned reporting input fields."
        )

    if (
        not reporting_years
        or len(set(reporting_years)) != len(reporting_years)
        or any(
            not isinstance(years, (int, np.integer))
            or isinstance(years, (bool, np.bool_))
            or years <= 0
            for years in reporting_years
        )
    ):
        raise ValueError(
            "Reporting years must be unique positive integers."
        )

    frame = aligned_returns.copy()

    frame["date"] = pd.to_datetime(
        frame["date"],
        errors="raise",
    )

    if (
        frame.empty
        or frame["date"].isna().any()
        or frame["date"].duplicated().any()
    ):
        raise ValueError(
            "Aligned reporting dates are missing or duplicated."
        )

    frame = (
        frame.sort_values("date")
        .reset_index(drop=True)
    )

    cutoff = pd.Timestamp(as_of_date)

    if pd.isna(cutoff) or cutoff not in pd.DatetimeIndex(
        frame["date"]
    ):
        raise ValueError(
            "Reporting cutoff must be an observed return date."
        )

    eligible = frame.loc[
        frame["date"].le(cutoff)
    ].copy()

    if (
        eligible["portfolio_id"].nunique() != 1
        or eligible["benchmark_portfolio_id"].nunique() != 1
    ):
        raise ValueError(
            "Reporting input must contain one portfolio/"
            "benchmark pair."
        )

    windows = {}

    for years in sorted(reporting_years):

        window_id = f"TRAILING_{years}Y"

        anniversary_anchor = (
            cutoff - pd.DateOffset(years=years)
        )

        window = eligible.loc[
            eligible["date"].gt(anniversary_anchor)
        ].copy()

        if len(window) <= RISK_FINAL_DDOF:
            raise ValueError(
                f"Insufficient {window_id} observations."
            )

        first_date = pd.Timestamp(
            window["date"].iloc[0]
        )

        last_date = pd.Timestamp(
            window["date"].iloc[-1]
        )

        days_after_anchor = int(
            (first_date - anniversary_anchor).days
        )

        if not 1 <= days_after_anchor <= 7:
            raise ValueError(
                f"Calendar boundary coverage is "
                f"insufficient for {window_id}."
            )

        if last_date != cutoff:
            raise AssertionError(
                f"Incorrect reporting cutoff: {window_id}"
            )

        windows[window_id] = {
            "reporting_years": years,
            "anniversary_anchor": anniversary_anchor,
            "first_return_date": first_date,
            "last_return_date": last_date,
            "first_return_gap_calendar_days": (
                days_after_anchor
            ),
            "observations": len(window),
            "coverage_status": (
                "OBSERVED_ANNIVERSARY_BOUNDARY"
            ),
            "returns": window.reset_index(drop=True),
        }

    return windows


# Research reference: 09_risk_analytics.ipynb, code cell index 15.
def calculate_window_max_drawdown(
    daily_returns: np.ndarray,
) -> tuple[float, float]:
    """
    Calculate drawdown from an initial wealth baseline of 1.

    The cumulative wealth and running peak are reset
    at the beginning of the reporting window.

    Returns:
        (maximum_drawdown, ending_drawdown)
    """

    values = np.asarray(
        daily_returns,
        dtype=np.float64,
    )

    if (
        values.ndim != 1
        or len(values) == 0
        or not np.isfinite(values).all()
        or np.any(values <= -1.0)
    ):
        raise ValueError(
            "Drawdown requires nonempty finite returns "
            "strictly greater than -100%."
        )

    wealth = np.cumprod(
        1.0 + values
    )

    if (
        not np.isfinite(wealth).all()
        or np.any(wealth <= 0.0)
    ):
        raise ArithmeticError(
            "Window wealth is invalid."
        )

    peaks = np.maximum.accumulate(
        np.concatenate(
            ([1.0], wealth)
        )
    )[1:]

    drawdown = (
        wealth / peaks - 1.0
    )

    if np.any(drawdown > 1e-12):
        raise ArithmeticError(
            "Drawdown cannot be positive."
        )

    return (
        float(drawdown.min()),
        float(drawdown[-1]),
    )
