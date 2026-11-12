"""Performance periods: validated research calculations."""

from __future__ import annotations

from .inputs import validate_performance_returns
import numpy as np
import pandas as pd



# Research reference: 07_performance_analytics.ipynb, code cell index 3.
def aggregate_period_returns(
    returns_df: pd.DataFrame,
    *,
    return_column: str = "return_total",
    period_frequencies: tuple[str, ...] = (
        "M",
        "Q",
        "Y",
    ),
) -> pd.DataFrame:
    """
    Compound daily returns into calendar periods.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Required columns:
            portfolio_id
            date
            return_column

    return_column : str
        Daily simple decimal return field.

    period_frequencies : tuple[str, ...]
        Supported calendar frequencies:
            M = monthly
            Q = quarterly
            Y = annually

    Returns
    -------
    pd.DataFrame
        One observation per portfolio and calendar period.

    Output fields
    -------------
    portfolio_id
    period_type
    period_start
    period_end
    first_return_date
    last_return_date
    observation_count
    observed_calendar_months
    expected_calendar_months
    period_return
    month_span_status
    session_coverage_status
    return_column
    return_units

    Notes
    -----
    Period boundaries are calendar dates.

    Period_end does not necessarily represent
    a trading session.

    Month-span coverage cannot independently
    certify exchange-session completeness.

    All returns retain full computational precision.
    """

    supported_frequencies = {
        "M": ("monthly", 1),
        "Q": ("quarterly", 3),
        "Y": ("annual", 12),
    }

    if not isinstance(returns_df, pd.DataFrame):
        raise TypeError(
            "returns_df must be a pandas DataFrame."
        )

    if not isinstance(return_column, str):
        raise TypeError(
            "return_column must be a string."
        )

    if return_column not in returns_df.columns:
        raise ValueError(
            f"Missing return column: {return_column}"
        )

    if return_column in {"portfolio_id", "date"}:
        raise ValueError(
            "return_column cannot be an identifier field."
        )

    if (
        not isinstance(period_frequencies, tuple)
        or not period_frequencies
    ):
        raise ValueError(
            "period_frequencies must be a nonempty tuple."
        )

    if len(set(period_frequencies)) != len(
        period_frequencies
    ):
        raise ValueError(
            "Duplicate period frequencies are not allowed."
        )

    unsupported = set(period_frequencies).difference(
        supported_frequencies
    )

    if unsupported:
        raise ValueError(
            f"Unsupported frequencies: {sorted(unsupported)}"
        )

    # Reuse the notebook's validated return contract.
    # The temporary canonical name allows equivalent
    # price or total-return inputs to use the same math.

    canonical_input = returns_df[
        ["portfolio_id", "date", return_column]
    ].rename(
        columns={return_column: "return_total"}
    )

    validated = validate_performance_returns(
        canonical_input
    )

    validated["gross_return"] = (
        1.0 + validated["return_total"]
    )

    validated["calendar_month"] = (
        validated["date"].dt.to_period("M")
    )

    results = []

    # --------------------------------------------------------
    # Aggregate each requested calendar frequency.
    # --------------------------------------------------------

    for frequency in period_frequencies:

        period_type, expected_months = (
            supported_frequencies[frequency]
        )

        working = validated.copy()

        working["calendar_period"] = (
            working["date"].dt.to_period(frequency)
        )

        summary = (
            working
            .groupby(
                ["portfolio_id", "calendar_period"],
                as_index=False,
                sort=True,
            )
            .agg(
                first_return_date=("date", "min"),
                last_return_date=("date", "max"),
                observation_count=("date", "size"),
                observed_calendar_months=(
                    "calendar_month",
                    "nunique",
                ),
                gross_period_return=(
                    "gross_return",
                    "prod",
                ),
            )
        )

        summary["period_type"] = period_type

        # Calendar boundaries are distinct from
        # the actual first and last return dates.

        summary["period_start"] = (
            summary["calendar_period"]
            .dt.to_timestamp(how="start")
        )

        summary["period_end"] = (
            summary["calendar_period"]
            .dt.to_timestamp(how="end")
            .dt.normalize()
        )

        summary["expected_calendar_months"] = (
            expected_months
        )

        summary["period_return"] = (
            summary["gross_period_return"] - 1.0
        )

        complete_month_span = (
            summary["observed_calendar_months"]
            .eq(expected_months)
        )

        summary["month_span_status"] = np.where(
            complete_month_span,
            "FULL_MONTH_SPAN_ONLY",
            "PARTIAL_MONTH_SPAN",
        )

        # Observing all calendar months does not prove
        # that every expected trading session is present.

        summary["session_coverage_status"] = (
            "NOT_ASSESSED_EXCHANGE_CALENDAR"
        )

        summary["return_column"] = return_column
        summary["return_units"] = "simple_decimal"

        results.append(
            summary[
                [
                    "portfolio_id",
                    "period_type",
                    "period_start",
                    "period_end",
                    "first_return_date",
                    "last_return_date",
                    "observation_count",
                    "observed_calendar_months",
                    "expected_calendar_months",
                    "period_return",
                    "month_span_status",
                    "session_coverage_status",
                    "return_column",
                    "return_units",
                ]
            ]
        )

    result_df = pd.concat(
        results,
        ignore_index=True,
    )

    result_df = (
        result_df
        .sort_values(
            ["portfolio_id", "period_type", "period_end"]
        )
        .reset_index(drop=True)
    )

    output_keys = [
        "portfolio_id",
        "period_type",
        "period_end",
    ]

    if result_df.duplicated(output_keys).any():
        raise ValueError(
            "Duplicate calendar-period output keys."
        )

    if not np.isfinite(
        result_df["period_return"].to_numpy(
            dtype=np.float64
        )
    ).all():
        raise ValueError(
            "Non-finite compounded period returns."
        )

    return result_df
