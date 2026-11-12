"""Performance wealth: validated research calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd



# Research reference: 07_performance_analytics.ipynb, code cell index 2.
def compute_wealth_index(
    returns_df: pd.DataFrame,
    *,
    return_column: str = "return_total",
    initial_wealth: float = 1.0,
) -> pd.DataFrame:
    """
    Compute independently compounded wealth paths.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Required columns:
            portfolio_id
            date
            return_column

    return_column : str
        Column containing simple decimal returns.

    initial_wealth : float
        Strictly positive, finite starting wealth.

    Returns
    -------
    pd.DataFrame
        portfolio_id
        date
        <return_column>
        wealth_bop
        wealth_eop
        cumulative_return

    Conventions
    -----------
    wealth_eop[t] = wealth_bop[t] * (1 + return[t])

    cumulative_return[t] = wealth_eop[t] /
                           initial_wealth - 1

    Each portfolio is independently compounded.

    The first observed date is the first return-interval
    end date, not the wealth initialization date.

    No input DataFrame mutation or external dependencies.
    """

    # Validate configuration.

    if not np.isscalar(initial_wealth):
        raise TypeError(
            "initial_wealth must be a numeric scalar."
        )

    if isinstance(initial_wealth, (bool, np.bool_)):
        raise TypeError(
            "initial_wealth must not be Boolean."
        )

    try:
        initial_wealth = float(initial_wealth)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(
            "initial_wealth must be numeric."
        ) from exc

    if not np.isfinite(initial_wealth):
        raise ValueError(
            "initial_wealth must be finite."
        )

    if initial_wealth <= 0:
        raise ValueError(
            "initial_wealth must be strictly positive."
        )

    # Validate required input structure.

    if not isinstance(returns_df, pd.DataFrame):
        raise TypeError(
            "returns_df must be a pandas DataFrame."
        )

    if returns_df.empty:
        raise ValueError(
            "Cannot compound an empty return series."
        )

    if not returns_df.columns.is_unique:
        raise ValueError(
            "Duplicate input column names detected."
        )

    required_columns = {
        "portfolio_id",
        "date",
        return_column,
    }

    missing_columns = required_columns.difference(
        returns_df.columns
    )

    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    if return_column in {
        "portfolio_id",
        "date",
        "wealth_bop",
        "wealth_eop",
        "cumulative_return",
    }:
        raise ValueError(
            "return_column conflicts with an output field."
        )

    if not pd.api.types.is_numeric_dtype(
        returns_df[return_column]
    ):
        raise TypeError(
            "Return column must contain numeric values."
        )

    # Work on a separate, minimal input projection.

    data = returns_df[
        ["portfolio_id", "date", return_column]
    ].copy()

    if data.isna().any().any():
        raise ValueError(
            "Null values detected in required input fields."
        )

    if not data["portfolio_id"].map(
        lambda value: isinstance(value, str)
        and bool(value.strip())
    ).all():
        raise ValueError(
            "portfolio_id must contain nonempty strings."
        )

    data["date"] = pd.to_datetime(
        data["date"],
        errors="raise",
    )

    if data["date"].isna().any():
        raise ValueError(
            "Invalid or missing return dates."
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

    if data.duplicated(
        ["portfolio_id", "date"]
    ).any():
        raise ValueError(
            "Duplicate portfolio-date observations."
        )

    returns = data[
        return_column
    ].to_numpy(dtype=np.float64)

    if not np.isfinite(returns).all():
        raise ValueError(
            "Returns must be finite."
        )

    if (returns < -1.0).any():
        raise ValueError(
            "Simple returns cannot be below -100%."
        )

    # Explicitly sort the return intervals.

    data = (
        data
        .sort_values(["portfolio_id", "date"])
        .reset_index(drop=True)
    )

    # Compound each portfolio independently.

    results = []

    for portfolio_id, group in data.groupby(
        "portfolio_id",
        sort=False,
    ):

        daily_returns = group[
            return_column
        ].to_numpy(dtype=np.float64)

        gross_returns = 1.0 + daily_returns

        cumulative_growth = np.cumprod(
            gross_returns,
            dtype=np.float64,
        )

        wealth_eop = (
            initial_wealth * cumulative_growth
        )

        if not np.isfinite(wealth_eop).all():
            raise ValueError(
                f"Non-finite wealth values: {portfolio_id}"
            )

        wealth_bop = np.concatenate(
            (
                [initial_wealth],
                wealth_eop[:-1],
            )
        )

        cumulative_return = (
            wealth_eop / initial_wealth - 1.0
        )

        result = group.copy()

        result["wealth_bop"] = wealth_bop
        result["wealth_eop"] = wealth_eop
        result["cumulative_return"] = cumulative_return

        results.append(result)

    return pd.concat(
        results,
        ignore_index=True,
    )
