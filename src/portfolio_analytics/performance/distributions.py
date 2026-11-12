"""Performance distributions: validated research calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd



# Research reference: 07_performance_analytics.ipynb, code cell index 20.
def prepare_rolling_relative_visuals(
    rolling_relative_df: pd.DataFrame,
    *,
    portfolio_id: str,
    benchmark_portfolio_id: str,
    horizons: tuple[str, ...] = ("1M", "3M", "12M"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare rolling chart series and descriptive statistics."""

    required = {
        "portfolio_id",
        "benchmark_portfolio_id",
        "date",
        "horizon_id",
        "calculation_status",
        "arithmetic_active_return",
        "geometric_relative_return",
    }

    if not required.issubset(rolling_relative_df.columns):
        raise ValueError("Missing rolling relative-performance fields.")

    if not horizons or len(set(horizons)) != len(horizons):
        raise ValueError("Rolling horizons must be unique.")

    data = rolling_relative_df.loc[
        rolling_relative_df["portfolio_id"].eq(portfolio_id)
        & rolling_relative_df[
            "benchmark_portfolio_id"
        ].eq(benchmark_portfolio_id)
        & rolling_relative_df["horizon_id"].isin(horizons),
        list(required),
    ].copy()

    if data.empty:
        raise ValueError("No matching rolling relative observations.")

    data["date"] = pd.to_datetime(data["date"])

    if (
        data[["date", "horizon_id"]].isna().any().any()
        or data.duplicated(["date", "horizon_id"]).any()
    ):
        raise ValueError("Invalid rolling observation keys.")

    if set(data["horizon_id"]) != set(horizons):
        raise ValueError("One or more requested horizons are missing.")

    available = data["calculation_status"].eq("AVAILABLE")

    numeric_fields = [
        "arithmetic_active_return",
        "geometric_relative_return",
    ]

    if not np.isfinite(
        data.loc[available, numeric_fields].to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError("Non-finite available relative returns.")

    # A benchmark-relative wealth return cannot fall
    # below -100% when benchmark ending wealth is positive.

    if (
        data.loc[
            available, "geometric_relative_return"
        ] < -1.0
    ).any():
        raise ValueError("Invalid geometric relative return.")

    # Mask unavailable windows explicitly.

    data["chart_relative_return"] = (
        data["geometric_relative_return"].where(available)
    )

    chart_df = (
        data.pivot(
            index="date",
            columns="horizon_id",
            values="chart_relative_return",
        )
        .sort_index()
        .reindex(columns=list(horizons))
    )

    chart_df.columns.name = None

    records = []

    for horizon in horizons:

        sample = (
            data.loc[
                data["horizon_id"].eq(horizon)
                & data["calculation_status"].eq("AVAILABLE")
            ]
            .sort_values("date")
        )

        if sample.empty:
            raise ValueError(
                f"No available windows for {horizon}."
            )

        geometric = sample[
            "geometric_relative_return"
        ].to_numpy(dtype=float)

        arithmetic = sample[
            "arithmetic_active_return"
        ].to_numpy(dtype=float)

        percentiles = np.quantile(
            geometric,
            [0.05, 0.25, 0.50, 0.75, 0.95],
        )

        records.append(
            {
                "horizon_id": horizon,
                "available_windows": len(sample),
                "outperforming_windows": int(
                    np.count_nonzero(arithmetic > 0)
                ),
                "underperforming_windows": int(
                    np.count_nonzero(arithmetic < 0)
                ),
                "tied_windows": int(
                    np.count_nonzero(arithmetic == 0)
                ),
                "outperformance_frequency": float(
                    np.mean(arithmetic > 0)
                ),
                "p05_geometric_relative": percentiles[0],
                "p25_geometric_relative": percentiles[1],
                "median_geometric_relative_return": (
                    percentiles[2]
                ),
                "p75_geometric_relative": percentiles[3],
                "p95_geometric_relative": percentiles[4],
                "first_available_date": sample["date"].iloc[0],
                "last_available_date": sample["date"].iloc[-1],
                "terminal_geometric_relative_return": (
                    geometric[-1]
                ),
            }
        )

    summary_df = pd.DataFrame.from_records(records)

    return chart_df, summary_df
