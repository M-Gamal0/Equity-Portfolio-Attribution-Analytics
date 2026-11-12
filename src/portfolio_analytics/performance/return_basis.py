"""Performance return basis: validated research calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd



# Research reference: 07_performance_analytics.ipynb, code cell index 17.
def compute_return_basis_windows(
    returns_df: pd.DataFrame,
    windows_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compound price and total returns over explicit windows."""

    required_returns = {
        "portfolio_id", "date",
        "return_total", "return_price",
    }

    required_windows = {
        "portfolio_id", "window_id", "as_of_date",
        "resolved_start_date", "first_return_date",
        "last_return_date", "observation_count",
        "start_anchor_source", "window_status",
    }

    if not required_returns.issubset(returns_df.columns):
        raise ValueError("Missing portfolio return fields.")

    if not required_windows.issubset(windows_df.columns):
        raise ValueError("Missing reporting-window fields.")

    returns = returns_df[
        ["portfolio_id", "date", "return_total", "return_price"]
    ].copy()

    windows = windows_df[list(required_windows)].copy()

    returns["date"] = pd.to_datetime(returns["date"])

    for column in (
        "as_of_date", "resolved_start_date",
        "first_return_date", "last_return_date",
    ):
        windows[column] = pd.to_datetime(windows[column])

    if returns.duplicated(["portfolio_id", "date"]).any():
        raise ValueError("Duplicate portfolio return keys.")

    if windows.duplicated(
        ["portfolio_id", "as_of_date", "window_id"]
    ).any():
        raise ValueError("Duplicate reporting-window keys.")

    values = returns[
        ["return_total", "return_price"]
    ].to_numpy(dtype=float)

    if not np.isfinite(values).all() or (values < -1).any():
        raise ValueError("Invalid price or total returns.")

    records = []

    for portfolio_id, group in returns.groupby(
        "portfolio_id", sort=True
    ):

        group = group.sort_values("date")

        portfolio_windows = windows.loc[
            windows["portfolio_id"].eq(portfolio_id)
        ]

        for _, window in portfolio_windows.iterrows():

            start = window["resolved_start_date"]
            end = window["as_of_date"]

            record = {
                "portfolio_id": portfolio_id,
                "as_of_date": end,
                "window_id": window["window_id"],
                "resolved_start_date": start,
                "observation_count": window["observation_count"],
                "period_return_total": np.nan,
                "period_return_price": np.nan,
                "total_return_premium_pp": np.nan,
                "total_to_price_wealth_ratio": np.nan,
                "total_to_price_wealth_relative_return": np.nan,
                "calculation_status": "WINDOW_UNAVAILABLE",
                "return_units": "simple_decimal",
            }

            if window["window_status"] != "AVAILABLE":
                records.append(record)
                continue

            if (
                pd.isna(start)
                or pd.isna(end)
                or start >= end
                or window["last_return_date"] != end
            ):
                raise ValueError("Invalid reporting-window dates.")

            anchor_source = window["start_anchor_source"]

            if anchor_source == "OBSERVED_RETURN_DATE":

                if not group["date"].eq(start).any():
                    raise ValueError("Missing observed start anchor.")

            elif anchor_source == "EXPLICIT_INCEPTION_BASELINE":

                if start >= group["date"].min():
                    raise ValueError("Invalid inception baseline.")

            else:
                raise ValueError("Unsupported start-anchor source.")

            sample = group.loc[
                group["date"].gt(start)
                & group["date"].le(end)
            ]

            if (
                sample.empty
                or len(sample) != int(window["observation_count"])
                or sample["date"].iloc[0]
                != window["first_return_date"]
                or sample["date"].iloc[-1] != end
            ):
                raise ValueError("Invalid window observation coverage.")

            # Do not reinitialize an investment whose
            # starting wealth has already been extinguished.

            prior = group.loc[
                group["date"].le(start),
                ["return_total", "return_price"],
            ]

            if prior.eq(-1.0).any().any():
                record["calculation_status"] = (
                    "UNAVAILABLE_ZERO_START_WEALTH"
                )
                records.append(record)
                continue

            gross_total = float(
                np.prod(
                    1.0 + sample["return_total"].to_numpy(),
                    dtype=np.float64,
                )
            )

            gross_price = float(
                np.prod(
                    1.0 + sample["return_price"].to_numpy(),
                    dtype=np.float64,
                )
            )

            if not np.isfinite([gross_total, gross_price]).all():
                raise ValueError("Non-finite compounded wealth.")

            record["period_return_total"] = gross_total - 1.0
            record["period_return_price"] = gross_price - 1.0

            record["total_return_premium_pp"] = (
                100.0 * (gross_total - gross_price)
            )

            if gross_price > 0.0:

                record["total_to_price_wealth_ratio"] = (
                    gross_total / gross_price
                )

                record[
                    "total_to_price_wealth_relative_return"
                ] = gross_total / gross_price - 1.0

                record["calculation_status"] = "CALCULATED"

            else:

                record["calculation_status"] = (
                    "UNDEFINED_ZERO_PRICE_WEALTH"
                )

            records.append(record)

    return (
        pd.DataFrame(records)
        .sort_values(
            ["portfolio_id", "as_of_date", "window_id"]
        )
        .reset_index(drop=True)
    )
