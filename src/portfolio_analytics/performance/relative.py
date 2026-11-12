"""Performance relative: validated research calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd



# Research reference: 07_performance_analytics.ipynb, code cell index 4.
def compute_relative_performance(
    aligned_returns_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate daily arithmetic and geometric relative performance.

    Required columns
    ----------------
    portfolio_id
    benchmark_portfolio_id
    date
    portfolio_return_total
    benchmark_return_total

    Returns
    -------
    pd.DataFrame
        One row per portfolio/benchmark/date combination.

    Mathematical conventions
    ------------------------
    Arithmetic active return:
        r_p - r_b

    Daily geometric relative return:
        (1 + r_p) / (1 + r_b) - 1

    Relative wealth:
        cumulative_product(1 + r_p)
        / cumulative_product(1 + r_b)

    Cumulative return spread:
        cumulative_portfolio_return
        - cumulative_benchmark_return

    Notes
    -----
    Each portfolio/benchmark pair is compounded separately.

    The input must already contain explicitly aligned dates.

    The function does not depend on another performance
    function, analytical module, or external calculation.
    """

    if not isinstance(aligned_returns_df, pd.DataFrame):
        raise TypeError(
            "aligned_returns_df must be a DataFrame."
        )

    if aligned_returns_df.empty:
        raise ValueError(
            "Aligned return input cannot be empty."
        )

    if not aligned_returns_df.columns.is_unique:
        raise ValueError(
            "Duplicate input column names detected."
        )

    required_columns = [
        "portfolio_id",
        "benchmark_portfolio_id",
        "date",
        "portfolio_return_total",
        "benchmark_return_total",
    ]

    missing = set(required_columns).difference(
        aligned_returns_df.columns
    )

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    data = aligned_returns_df[
        required_columns
    ].copy()

    if data.isna().any().any():
        raise ValueError(
            "Null values detected in aligned returns."
        )

    for identifier in [
        "portfolio_id",
        "benchmark_portfolio_id",
    ]:
        if not data[identifier].map(
            lambda value: isinstance(value, str)
            and bool(value.strip())
        ).all():
            raise ValueError(
                f"{identifier} must contain nonempty strings."
            )

    if data["portfolio_id"].eq(
        data["benchmark_portfolio_id"]
    ).any():
        raise ValueError(
            "Portfolio and benchmark IDs must differ."
        )

    data["date"] = pd.to_datetime(
        data["date"],
        errors="raise",
    )

    if data["date"].dt.tz is not None:
        raise ValueError(
            "Return dates must be timezone-naive."
        )

    if not data["date"].eq(
        data["date"].dt.normalize()
    ).all():
        raise ValueError(
            "Return dates must contain dates only."
        )

    key_columns = [
        "portfolio_id",
        "benchmark_portfolio_id",
        "date",
    ]

    if data.duplicated(key_columns).any():
        raise ValueError(
            "Duplicate portfolio/benchmark/date keys."
        )

    return_columns = [
        "portfolio_return_total",
        "benchmark_return_total",
    ]

    for column in return_columns:

        if not pd.api.types.is_numeric_dtype(
            data[column]
        ):
            raise TypeError(
                f"{column} must be numeric."
            )

        values = data[column].to_numpy(
            dtype=np.float64
        )

        if not np.isfinite(values).all():
            raise ValueError(
                f"{column} contains non-finite values."
            )

        if (values < -1.0).any():
            raise ValueError(
                f"{column} contains returns below -100%."
            )

    data = (
        data
        .sort_values(key_columns)
        .reset_index(drop=True)
    )

    results = []

    pair_columns = [
        "portfolio_id",
        "benchmark_portfolio_id",
    ]

    # Each pair forms a distinct investment comparison.

    for pair_ids, group in data.groupby(
        pair_columns,
        sort=False,
    ):

        portfolio_id, benchmark_id = pair_ids

        group = group.copy()

        portfolio_returns = group[
            "portfolio_return_total"
        ].to_numpy(dtype=np.float64)

        benchmark_returns = group[
            "benchmark_return_total"
        ].to_numpy(dtype=np.float64)

        portfolio_gross = 1.0 + portfolio_returns
        benchmark_gross = 1.0 + benchmark_returns

        # Geometric relative returns require a strictly
        # positive benchmark gross return on every date.

        if (benchmark_gross <= 0.0).any():
            raise ValueError(
                "Relative wealth is undefined after a "
                "zero benchmark wealth event for "
                f"{portfolio_id} versus {benchmark_id}."
            )

        # Independently compound both return series.

        portfolio_wealth = np.cumprod(
            portfolio_gross,
            dtype=np.float64,
        )

        benchmark_wealth = np.cumprod(
            benchmark_gross,
            dtype=np.float64,
        )

        if (
            not np.isfinite(portfolio_wealth).all()
            or not np.isfinite(benchmark_wealth).all()
            or (benchmark_wealth <= 0.0).any()
        ):
            raise ValueError(
                "Non-finite or numerically invalid wealth "
                f"for {portfolio_id} versus {benchmark_id}."
            )

        arithmetic_active_return = (
            portfolio_returns - benchmark_returns
        )

        geometric_relative_return = (
            portfolio_gross / benchmark_gross - 1.0
        )

        relative_wealth = (
            portfolio_wealth / benchmark_wealth
        )

        if not np.isfinite(relative_wealth).all():
            raise ValueError(
                "Non-finite relative wealth detected."
            )

        # Build standardized results.

        group["arithmetic_active_return"] = (
            arithmetic_active_return
        )

        group["daily_geometric_relative_return"] = (
            geometric_relative_return
        )

        group["portfolio_wealth_eop"] = (
            portfolio_wealth
        )

        group["benchmark_wealth_eop"] = (
            benchmark_wealth
        )

        group["cumulative_portfolio_return"] = (
            portfolio_wealth - 1.0
        )

        group["cumulative_benchmark_return"] = (
            benchmark_wealth - 1.0
        )

        group["cumulative_return_spread"] = (
            portfolio_wealth - benchmark_wealth
        )

        group["relative_wealth_index"] = (
            relative_wealth
        )

        group["cumulative_geometric_relative_return"] = (
            relative_wealth - 1.0
        )

        results.append(group)

    return pd.concat(
        results,
        ignore_index=True,
    )


# Research reference: 07_performance_analytics.ipynb, code cell index 5.
def compute_relative_period_performance(
    aligned_returns_df: pd.DataFrame,
    *,
    period_frequencies: tuple[str, ...] = (
        "M", "Q", "Y"
    ),
) -> pd.DataFrame:
    """
    Calculate calendar-period benchmark-relative performance.

    Required columns
    ----------------
    portfolio_id
    benchmark_portfolio_id
    date
    portfolio_return_total
    benchmark_return_total

    Parameters
    ----------
    aligned_returns_df:
        DataFrame containing explicitly paired daily returns.

    period_frequencies:
        M = calendar month
        Q = calendar quarter
        Y = calendar year

    Returns
    -------
    pd.DataFrame
        Calendar-period portfolio and benchmark returns,
        arithmetic active returns, geometric relative returns,
        cumulative wealth, and observation coverage.

    Notes
    -----
    All return observations are simple decimal fractions.

    Period return:
        product(1 + daily_return) - 1

    Arithmetic period active return:
        portfolio_period_return - benchmark_period_return

    Geometric period relative return:
        (1 + portfolio_period_return)
        / (1 + benchmark_period_return) - 1

    Summed daily active return:
        sum(portfolio_daily_return - benchmark_daily_return)

    Summed daily active returns are retained only as a
    diagnostic and are not equivalent to compounded
    period active returns.

    Each portfolio/benchmark pair is independent.

    No dependency on another analytics calculation.
    """

    frequency_mapping = {
        "M": ("monthly", 1),
        "Q": ("quarterly", 3),
        "Y": ("annual", 12),
    }

    if not isinstance(aligned_returns_df, pd.DataFrame):
        raise TypeError(
            "aligned_returns_df must be a DataFrame."
        )

    if aligned_returns_df.empty:
        raise ValueError(
            "Aligned performance input is empty."
        )

    if not aligned_returns_df.columns.is_unique:
        raise ValueError(
            "Duplicate input column names detected."
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
        frequency_mapping
    )

    if unsupported:
        raise ValueError(
            f"Unsupported frequencies: {sorted(unsupported)}"
        )

    required_columns = [
        "portfolio_id",
        "benchmark_portfolio_id",
        "date",
        "portfolio_return_total",
        "benchmark_return_total",
    ]

    missing = set(required_columns).difference(
        aligned_returns_df.columns
    )

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    data = aligned_returns_df[
        required_columns
    ].copy()

    if data.isna().any().any():
        raise ValueError(
            "Null required performance fields detected."
        )

    for column in [
        "portfolio_id",
        "benchmark_portfolio_id",
    ]:
        if not data[column].map(
            lambda value: isinstance(value, str)
            and bool(value.strip())
        ).all():
            raise ValueError(
                f"{column} must contain nonempty strings."
            )

    if data["portfolio_id"].eq(
        data["benchmark_portfolio_id"]
    ).any():
        raise ValueError(
            "Portfolio and benchmark IDs must differ."
        )

    data["date"] = pd.to_datetime(
        data["date"],
        errors="raise",
    )

    if data["date"].dt.tz is not None:
        raise ValueError(
            "Dates must be timezone-naive."
        )

    if not data["date"].eq(
        data["date"].dt.normalize()
    ).all():
        raise ValueError(
            "Dates must contain dates only."
        )

    pair_keys = [
        "portfolio_id",
        "benchmark_portfolio_id",
    ]

    if data.duplicated(
        pair_keys + ["date"]
    ).any():
        raise ValueError(
            "Duplicate portfolio/benchmark/date keys."
        )

    for column in [
        "portfolio_return_total",
        "benchmark_return_total",
    ]:
        if not pd.api.types.is_numeric_dtype(
            data[column]
        ):
            raise TypeError(
                f"{column} must be numeric."
            )

        values = data[column].to_numpy(
            dtype=np.float64
        )

        if not np.isfinite(values).all():
            raise ValueError(
                f"{column} contains non-finite returns."
            )

        if (values < -1.0).any():
            raise ValueError(
                f"{column} contains returns below -100%."
            )

    # A zero benchmark wealth path makes geometric
    # relative wealth mathematically undefined.

    if (
        data["benchmark_return_total"] <= -1.0
    ).any():
        raise ValueError(
            "Geometric relative performance is undefined "
            "after a complete benchmark loss."
        )

    data = (
        data
        .sort_values(pair_keys + ["date"])
        .reset_index(drop=True)
    )

    data["gross_portfolio_return"] = (
        1.0 + data["portfolio_return_total"]
    )

    data["gross_benchmark_return"] = (
        1.0 + data["benchmark_return_total"]
    )

    data["daily_arithmetic_active_return"] = (
        data["portfolio_return_total"]
        - data["benchmark_return_total"]
    )

    data["calendar_month"] = (
        data["date"].dt.to_period("M")
    )

    results = []

    # --------------------------------------------------------
    # Calculate each calendar frequency independently.
    # --------------------------------------------------------

    for frequency in period_frequencies:

        period_type, expected_months = (
            frequency_mapping[frequency]
        )

        working = data.copy()

        working["calendar_period"] = (
            working["date"].dt.to_period(frequency)
        )

        summary = (
            working
            .groupby(
                pair_keys + ["calendar_period"],
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
                portfolio_gross_period=(
                    "gross_portfolio_return",
                    "prod",
                ),
                benchmark_gross_period=(
                    "gross_benchmark_return",
                    "prod",
                ),
                sum_daily_arithmetic_active_return=(
                    "daily_arithmetic_active_return",
                    "sum",
                ),
            )
        )

        summary["period_type"] = period_type

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

        summary["portfolio_period_return"] = (
            summary["portfolio_gross_period"] - 1.0
        )

        summary["benchmark_period_return"] = (
            summary["benchmark_gross_period"] - 1.0
        )

        summary["arithmetic_active_period_return"] = (
            summary["portfolio_period_return"]
            - summary["benchmark_period_return"]
        )

        benchmark_gross = (
            summary["benchmark_gross_period"].to_numpy()
        )

        if (
            not np.isfinite(benchmark_gross).all()
            or (benchmark_gross <= 0.0).any()
        ):
            raise ValueError(
                "Undefined compounded benchmark wealth."
            )

        summary["geometric_relative_period_return"] = (
            summary["portfolio_gross_period"]
            / summary["benchmark_gross_period"]
            - 1.0
        )

        summary["month_span_status"] = np.where(
            summary["observed_calendar_months"].eq(
                expected_months
            ),
            "FULL_MONTH_SPAN_ONLY",
            "PARTIAL_MONTH_SPAN",
        )

        summary["session_coverage_status"] = (
            "NOT_ASSESSED_EXCHANGE_CALENDAR"
        )

        summary["return_units"] = "simple_decimal"

        results.append(
            summary[
                [
                    *pair_keys,
                    "period_type",
                    "period_start",
                    "period_end",
                    "first_return_date",
                    "last_return_date",
                    "observation_count",
                    "observed_calendar_months",
                    "expected_calendar_months",
                    "portfolio_period_return",
                    "benchmark_period_return",
                    "arithmetic_active_period_return",
                    "geometric_relative_period_return",
                    "sum_daily_arithmetic_active_return",
                    "month_span_status",
                    "session_coverage_status",
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
            pair_keys + ["period_type", "period_end"]
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Independently compound calendar-period wealth.
    # --------------------------------------------------------

    wealth_group_keys = (
        pair_keys + ["period_type"]
    )

    result_df["portfolio_wealth_eop"] = (
        1.0 + result_df["portfolio_period_return"]
    ).groupby(
        [
            result_df[column]
            for column in wealth_group_keys
        ]
    ).cumprod()

    result_df["benchmark_wealth_eop"] = (
        1.0 + result_df["benchmark_period_return"]
    ).groupby(
        [
            result_df[column]
            for column in wealth_group_keys
        ]
    ).cumprod()

    if (
        not np.isfinite(
            result_df["benchmark_wealth_eop"]
        ).all()
        or (
            result_df["benchmark_wealth_eop"] <= 0.0
        ).any()
    ):
        raise ValueError(
            "Undefined cumulative benchmark wealth."
        )

    result_df["relative_wealth_eop"] = (
        result_df["portfolio_wealth_eop"]
        / result_df["benchmark_wealth_eop"]
    )

    result_df["cumulative_geometric_relative_return"] = (
        result_df["relative_wealth_eop"] - 1.0
    )

    numeric_fields = [
        "portfolio_period_return",
        "benchmark_period_return",
        "arithmetic_active_period_return",
        "geometric_relative_period_return",
        "sum_daily_arithmetic_active_return",
        "portfolio_wealth_eop",
        "benchmark_wealth_eop",
        "relative_wealth_eop",
        "cumulative_geometric_relative_return",
    ]

    if not np.isfinite(
        result_df[numeric_fields].to_numpy(
            dtype=np.float64
        )
    ).all():
        raise ValueError(
            "Non-finite period-relative results detected."
        )

    output_keys = (
        pair_keys + ["period_type", "period_end"]
    )

    if result_df.duplicated(output_keys).any():
        raise ValueError(
            "Duplicate period-relative output keys."
        )

    return result_df
