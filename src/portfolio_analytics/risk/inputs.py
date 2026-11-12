"""Risk inputs: validated research calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd



# Research reference: 09_risk_analytics.ipynb, code cell index 2.
def align_portfolio_risk_returns(
    returns_df: pd.DataFrame,
    portfolio_id: str,
    benchmark_portfolio_id: str,
) -> pd.DataFrame:
    """
    Align daily portfolio and benchmark total returns.

    Parameters
    ----------
    returns_df:
        Standardized portfolio-return DataFrame.

        Required columns:
            portfolio_id
            date
            return_total

        Returns must be daily simple decimal fractions.
        Dates identify close-to-close return-interval ends.

    portfolio_id:
        Portfolio identifier being evaluated.

    benchmark_portfolio_id:
        Reference benchmark identifier.

    Returns
    -------
    pd.DataFrame
        One row per matched return date containing:

            portfolio_id
            benchmark_portfolio_id
            date
            portfolio_return_total
            benchmark_return_total
            active_return_arithmetic

        The active-return column is the arithmetic
        difference between contemporaneous total returns.

    Raises
    ------
    ValueError
        Missing identifiers, duplicate dates, unmatched
        portfolio calendars, or invalid return values.

    TypeError
        Non-numeric return data.

    Notes
    -----
    No missing observations are filled or discarded.

    The function does not calculate relative compounded
    wealth, volatility, tracking error, or other risk
    statistics.

    No source DataFrame is modified.
    """

    required = {
        "portfolio_id",
        "date",
        "return_total",
    }

    missing_columns = required.difference(
        returns_df.columns
    )

    if missing_columns:
        raise ValueError(
            "Missing required return columns: "
            f"{sorted(missing_columns)}"
        )

    if portfolio_id == benchmark_portfolio_id:
        raise ValueError(
            "Portfolio and benchmark identifiers "
            "must be distinct."
        )

    # Extract only the selected portfolio and benchmark.

    portfolio = returns_df.loc[
        returns_df["portfolio_id"].eq(portfolio_id),
        ["date", "return_total"],
    ].copy()

    benchmark = returns_df.loc[
        returns_df["portfolio_id"].eq(
            benchmark_portfolio_id
        ),
        ["date", "return_total"],
    ].copy()

    if portfolio.empty:
        raise ValueError(
            f"No returns found for portfolio: {portfolio_id}"
        )

    if benchmark.empty:
        raise ValueError(
            "No returns found for benchmark: "
            f"{benchmark_portfolio_id}"
        )

    # Normalize date representation without altering
    # the source DataFrame.

    for frame, label in (
        (portfolio, "Portfolio"),
        (benchmark, "Benchmark"),
    ):

        frame["date"] = pd.to_datetime(
            frame["date"],
            errors="raise",
        )

        if frame["date"].isna().any():
            raise ValueError(
                f"{label}: null return dates detected."
            )

        if frame["date"].duplicated().any():
            raise ValueError(
                f"{label}: duplicate return dates detected."
            )

        if not pd.api.types.is_numeric_dtype(
            frame["return_total"]
        ):
            raise TypeError(
                f"{label}: return_total must be numeric."
            )

    # Rename return columns to preserve their meaning.

    portfolio = portfolio.rename(
        columns={
            "return_total": "portfolio_return_total",
        }
    )

    benchmark = benchmark.rename(
        columns={
            "return_total": "benchmark_return_total",
        }
    )

    # Outer alignment detects missing dates on either side.

    matched = portfolio.merge(
        benchmark,
        on="date",
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    unmatched = matched.loc[
        matched["_merge"].ne("both")
    ]

    if not unmatched.empty:

        unmatched_counts = (
            unmatched["_merge"]
            .value_counts()
            .to_dict()
        )

        raise ValueError(
            "Portfolio and benchmark return calendars "
            "are not identical. "
            f"Unmatched observations: {unmatched_counts}"
        )

    matched = matched.drop(
        columns="_merge"
    )

    matched = (
        matched
        .sort_values("date")
        .reset_index(drop=True)
    )

    # Reject non-finite or economically invalid returns.

    return_columns = [
        "portfolio_return_total",
        "benchmark_return_total",
    ]

    return_values = matched[
        return_columns
    ].to_numpy(dtype=np.float64)

    if not np.isfinite(return_values).all():
        raise ValueError(
            "Matched returns contain non-finite values."
        )

    if (return_values < -1.0).any():
        raise ValueError(
            "Simple returns cannot be below -100%."
        )

    if len(matched) < 2:
        raise ValueError(
            "At least two matched observations are "
            "required for sample variance estimation."
        )

    # Calculate arithmetic active returns.

    matched["active_return_arithmetic"] = (
        matched["portfolio_return_total"]
        - matched["benchmark_return_total"]
    )

    # Preserve both portfolio identities explicitly.

    matched.insert(
        0,
        "benchmark_portfolio_id",
        benchmark_portfolio_id,
    )

    matched.insert(
        0,
        "portfolio_id",
        portfolio_id,
    )

    return matched[
        [
            "portfolio_id",
            "benchmark_portfolio_id",
            "date",
            "portfolio_return_total",
            "benchmark_return_total",
            "active_return_arithmetic",
        ]
    ]
