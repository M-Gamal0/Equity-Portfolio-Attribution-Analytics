"""Attribution carino: validated research calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd



# Research reference: 08_portfolio_attribution.ipynb, code cell index 8.
def carino_scaling_factor(
    portfolio_returns: np.ndarray,
    benchmark_returns: np.ndarray,
) -> np.ndarray:
    """Calculate stable Carino logarithmic scaling factors."""

    rp = np.asarray(portfolio_returns, dtype=float)
    rb = np.asarray(benchmark_returns, dtype=float)

    if rp.shape != rb.shape:
        raise ValueError("Portfolio/benchmark shapes differ.")

    if (
        not np.isfinite(rp).all()
        or not np.isfinite(rb).all()
        or (rp <= -1).any()
        or (rb <= -1).any()
    ):
        raise ValueError(
            "Carino requires finite returns strictly above -100%."
        )

    difference = rp - rb
    factors = np.empty_like(difference)

    unequal = difference != 0

    factors[unequal] = (
        np.log1p(
            difference[unequal] / (1 + rb[unequal])
        )
        / difference[unequal]
    )

    factors[~unequal] = 1 / (1 + rb[~unequal])

    return factors


# Research reference: 08_portfolio_attribution.ipynb, code cell index 8.
def link_attribution_carino(
    daily_sector_attribution: pd.DataFrame,
    *,
    portfolio_id: str,
    benchmark_id: str,
    start_date=None,
    end_date=None,
    tolerance: float = 1e-10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Link daily Brinson effects using Carino methodology.

    Input grain:
        portfolio_id, benchmark_id, date, sector.

    Required effects:
        allocation_effect, selection_effect, interaction_effect.

    Returns:
        Sector-level linked attribution and one period summary.

    The compounded active-return target is:
        compounded_portfolio_return - compounded_benchmark_return.

    No date/return observations are fabricated.
    """

    effect_columns = [
        "allocation_effect",
        "selection_effect",
        "interaction_effect",
    ]

    required_columns = {
        "portfolio_id",
        "benchmark_id",
        "date",
        "sector",
        "portfolio_return_total",
        "benchmark_return_total",
        *effect_columns,
    }

    missing = required_columns - set(
        daily_sector_attribution.columns
    )

    if missing:
        raise ValueError(
            f"Missing Carino input columns: {sorted(missing)}"
        )

    data = daily_sector_attribution.loc[
        daily_sector_attribution["portfolio_id"].eq(
            portfolio_id
        )
        & daily_sector_attribution["benchmark_id"].eq(
            benchmark_id
        )
    ].copy()

    data["date"] = pd.to_datetime(data["date"])

    if start_date is not None:
        data = data.loc[
            data["date"] >= pd.Timestamp(start_date)
        ]

    if end_date is not None:
        data = data.loc[
            data["date"] <= pd.Timestamp(end_date)
        ]

    if data.empty:
        raise ValueError("No observations in selected period.")

    keys = ["date", "sector"]

    if data.duplicated(keys).any():
        raise ValueError("Duplicate daily sector attribution keys.")

    # --------------------------------------------------------
    # 1. Extract unique daily portfolio/benchmark returns
    # --------------------------------------------------------

    daily_returns = (
        data[
            [
                "date",
                "portfolio_return_total",
                "benchmark_return_total",
            ]
        ]
        .drop_duplicates()
        .sort_values("date")
        .reset_index(drop=True)
    )

    if daily_returns["date"].duplicated().any():
        raise ValueError(
            "Inconsistent portfolio/benchmark returns "
            "within a daily sector group."
        )

    rp = daily_returns[
        "portfolio_return_total"
    ].to_numpy(dtype=float)

    rb = daily_returns[
        "benchmark_return_total"
    ].to_numpy(dtype=float)

    effects = data[effect_columns].to_numpy(dtype=float)

    if not np.isfinite(effects).all():
        raise ValueError("Non-finite daily attribution effects.")

    # --------------------------------------------------------
    # 2. Calculate compounded period returns
    # --------------------------------------------------------

    portfolio_log_growth = np.log1p(rp).sum()
    benchmark_log_growth = np.log1p(rb).sum()

    portfolio_period_return = np.expm1(
        portfolio_log_growth
    )

    benchmark_period_return = np.expm1(
        benchmark_log_growth
    )

    compounded_active_return = (
        portfolio_period_return
        - benchmark_period_return
    )

    relative_wealth_return = (
        np.expm1(
            portfolio_log_growth - benchmark_log_growth
        )
    )

    # --------------------------------------------------------
    # 3. Calculate daily and period Carino scaling factors
    # --------------------------------------------------------

    daily_returns["carino_daily_factor"] = (
        carino_scaling_factor(rp, rb)
    )

    period_factor = carino_scaling_factor(
        np.array([portfolio_period_return]),
        np.array([benchmark_period_return]),
    )[0]

    daily_returns["carino_linking_multiplier"] = (
        daily_returns["carino_daily_factor"]
        / period_factor
    )

    # --------------------------------------------------------
    # 4. Link daily attribution effects
    # --------------------------------------------------------

    data = data.merge(
        daily_returns[
            ["date", "carino_linking_multiplier"]
        ],
        on="date",
        how="left",
        validate="many_to_one",
    )

    for column in effect_columns:
        data[f"linked_{column}"] = (
            data[column]
            * data["carino_linking_multiplier"]
        )

    linked_columns = [
        f"linked_{column}"
        for column in effect_columns
    ]

    sector_linked = (
        data.groupby("sector", as_index=False)[
            linked_columns
        ]
        .sum()
    )

    sector_linked["linked_total_effect"] = (
        sector_linked[linked_columns].sum(axis=1)
    )

    # --------------------------------------------------------
    # 5. Reconcile total linked attribution
    # --------------------------------------------------------

    total_linked = sector_linked[
        linked_columns
    ].sum()

    total_effect = float(total_linked.sum())

    reconciliation_error = (
        total_effect - compounded_active_return
    )

    if abs(reconciliation_error) > tolerance:
        raise ValueError(
            "Carino linking reconciliation failed. "
            f"Residual: {reconciliation_error:.12e}"
        )

    # --------------------------------------------------------
    # 6. Prepare standardized period outputs
    # --------------------------------------------------------

    period_start = daily_returns["date"].min()
    period_end = daily_returns["date"].max()

    sector_linked.insert(
        0, "portfolio_id", portfolio_id
    )

    sector_linked.insert(
        1, "benchmark_id", benchmark_id
    )

    sector_linked["period_start"] = period_start
    sector_linked["period_end"] = period_end
    sector_linked["linking_method"] = "carino"

    sector_linked["linked_relative_wealth_effect"] = (
        sector_linked["linked_total_effect"]
        / (1 + benchmark_period_return)
    )

    summary = pd.DataFrame(
        [
            {
                "portfolio_id": portfolio_id,
                "benchmark_id": benchmark_id,
                "period_start": period_start,
                "period_end": period_end,
                "observations": len(daily_returns),
                "portfolio_compounded_return":
                    portfolio_period_return,
                "benchmark_compounded_return":
                    benchmark_period_return,
                "compounded_active_return":
                    compounded_active_return,
                "relative_wealth_return":
                    relative_wealth_return,
                "linked_allocation_effect":
                    total_linked["linked_allocation_effect"],
                "linked_selection_effect":
                    total_linked["linked_selection_effect"],
                "linked_interaction_effect":
                    total_linked["linked_interaction_effect"],
                "linked_total_effect": total_effect,
                "reconciliation_error": reconciliation_error,
                "linking_method": "carino",
            }
        ]
    )

    return sector_linked, summary
