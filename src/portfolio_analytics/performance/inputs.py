"""Performance inputs: validated research calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd



# Research reference: 07_performance_analytics.ipynb, code cell index 1.
def validate_performance_returns(
    returns_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate daily portfolio simple returns.

    Required columns:
        portfolio_id
        date
        return_total

    Optional:
        return_price

    Units:
        Daily simple decimal returns.

    Returns:
        Validated, date-normalized, sorted DataFrame.

    Raises:
        TypeError:
            Invalid input or numeric data types.

        ValueError:
            Missing fields, duplicate keys, invalid dates,
            non-finite returns, or returns below -100%.

    Notes:
        Does not require a benchmark, holdings, metadata,
        risk-free series, or a particular portfolio identity.
    """

    if not isinstance(returns_df, pd.DataFrame):
        raise TypeError(
            "returns_df must be a pandas DataFrame."
        )

    if returns_df.empty:
        raise ValueError(
            "Performance return input is empty."
        )

    if not returns_df.columns.is_unique:
        raise ValueError(
            "Duplicate DataFrame column names detected."
        )

    required = {
        "portfolio_id",
        "date",
        "return_total",
    }

    missing = required.difference(returns_df.columns)

    if missing:
        raise ValueError(
            f"Missing performance columns: {sorted(missing)}"
        )

    validated = returns_df.copy()

    if validated[
        ["portfolio_id", "date", "return_total"]
    ].isna().any().any():

        raise ValueError(
            "Null values detected in required performance fields."
        )

    if not validated["portfolio_id"].map(
        lambda value: isinstance(value, str)
        and bool(value.strip())
    ).all():

        raise ValueError(
            "portfolio_id must contain nonempty strings."
        )

    # Normalize dates without changing return observations.

    validated["date"] = pd.to_datetime(
        validated["date"],
        errors="raise",
    )

    if validated["date"].dt.tz is not None:
        raise ValueError(
            "Performance dates must be timezone-naive."
        )

    if not validated["date"].eq(
        validated["date"].dt.normalize()
    ).all():

        raise ValueError(
            "Performance dates must contain dates only."
        )

    # Enforce one portfolio observation per date.

    key_columns = ["portfolio_id", "date"]

    if validated.duplicated(key_columns).any():
        raise ValueError(
            "Duplicate (portfolio_id, date) keys detected."
        )

    # Validate total returns.

    if not pd.api.types.is_numeric_dtype(
        validated["return_total"]
    ):
        raise TypeError(
            "return_total must contain numeric values."
        )

    total_returns = validated[
        "return_total"
    ].to_numpy(dtype=np.float64)

    if not np.isfinite(total_returns).all():
        raise ValueError(
            "Non-finite total returns detected."
        )

    if (total_returns < -1.0).any():
        raise ValueError(
            "Simple total returns cannot be below -100%."
        )

    # Price returns are optional.
    # Missing optional observations do not invalidate
    # otherwise complete total-return analytics.

    if "return_price" in validated.columns:

        price_series = validated["return_price"]

        if price_series.notna().any():

            if not pd.api.types.is_numeric_dtype(
                price_series
            ):
                raise TypeError(
                    "return_price must contain numeric values."
                )

            observed_prices = price_series.dropna().to_numpy(
                dtype=np.float64
            )

            if not np.isfinite(observed_prices).all():
                raise ValueError(
                    "Non-finite price returns detected."
                )

            if (observed_prices < -1.0).any():
                raise ValueError(
                    "Simple price returns cannot be below -100%."
                )

    return (
        validated
        .sort_values(key_columns)
        .reset_index(drop=True)
    )


# Research reference: 07_performance_analytics.ipynb, code cell index 1.
def align_performance_pair(
    validated_returns_df: pd.DataFrame,
    portfolio_id: str,
    benchmark_portfolio_id: str,
) -> pd.DataFrame:
    """
    Prepare aligned portfolio and benchmark daily returns.

    Required input:
        Validated performance returns DataFrame.

    Output grain:
        One row per common return-interval end date.

    Output fields:
        portfolio_id
        benchmark_portfolio_id
        date
        portfolio_return_total
        benchmark_return_total

    Policy:
        Exact date alignment is required.
        Unmatched dates are not silently dropped.

    No active-return calculations are performed here.
    """

    if portfolio_id == benchmark_portfolio_id:
        raise ValueError(
            "Portfolio and benchmark IDs must differ."
        )

    portfolio = validated_returns_df.loc[
        validated_returns_df["portfolio_id"].eq(
            portfolio_id
        ),
        ["date", "return_total"],
    ].rename(
        columns={
            "return_total": "portfolio_return_total"
        }
    )

    benchmark = validated_returns_df.loc[
        validated_returns_df["portfolio_id"].eq(
            benchmark_portfolio_id
        ),
        ["date", "return_total"],
    ].rename(
        columns={
            "return_total": "benchmark_return_total"
        }
    )

    if portfolio.empty or benchmark.empty:
        raise ValueError(
            "Portfolio or benchmark return observations missing."
        )

    aligned = portfolio.merge(
        benchmark,
        on="date",
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    unmatched = aligned["_merge"].ne("both")

    if unmatched.any():
        raise ValueError(
            "Portfolio/benchmark calendar mismatch: "
            f"{int(unmatched.sum())} unmatched dates."
        )

    aligned = (
        aligned
        .drop(columns="_merge")
        .sort_values("date")
        .reset_index(drop=True)
    )

    aligned.insert(
        0,
        "portfolio_id",
        portfolio_id,
    )

    aligned.insert(
        1,
        "benchmark_portfolio_id",
        benchmark_portfolio_id,
    )

    return aligned
