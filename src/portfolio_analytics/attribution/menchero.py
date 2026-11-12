"""Attribution menchero: validated research calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd



# Research reference: 08_portfolio_attribution.ipynb, code cell index 10.
def menchero_linking_coefficients(
    portfolio_returns: np.ndarray,
    benchmark_returns: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """
    Calculate Menchero optimized linking coefficients.

    Returns
    -------
    beta:
        Daily optimized linking multipliers.
    M:
        Common scaling factor.
    C:
        Residual correction coefficient.
    """

    rp = np.asarray(portfolio_returns, dtype=float)
    rb = np.asarray(benchmark_returns, dtype=float)

    if (
        rp.ndim != 1
        or rp.shape != rb.shape
        or rp.size == 0
    ):
        raise ValueError("Invalid Menchero return-array shapes.")

    if (
        not np.isfinite(rp).all()
        or not np.isfinite(rb).all()
        or (rp <= -1).any()
        or (rb <= -1).any()
    ):
        raise ValueError(
            "Menchero requires finite returns above -100%."
        )

    n = rp.size
    active = rp - rb

    portfolio_log_growth = np.log1p(rp).sum()
    benchmark_log_growth = np.log1p(rb).sum()

    log_growth_difference = (
        portfolio_log_growth - benchmark_log_growth
    )

    # Stable evaluation of:
    #
    # M = [(exp(Lp) - exp(Lb)) / n] /
    #     [exp(Lp/n) - exp(Lb/n)]
    #
    # The rearrangement avoids subtracting nearly equal
    # compounded gross returns and geometric mean returns.

    if log_growth_difference == 0.0:
        scaling_ratio = 1.0
    else:
        scaling_ratio = (
            np.expm1(log_growth_difference)
            / (
                n
                * np.expm1(log_growth_difference / n)
            )
        )

    M = (
        np.exp(
            benchmark_log_growth * (n - 1) / n
        )
        * scaling_ratio
    )

    compounded_active_difference = (
        np.exp(benchmark_log_growth)
        * np.expm1(log_growth_difference)
    )

    # Residual after common scaling.
    residual = (
        compounded_active_difference
        - M * active.sum()
    )

    active_squared_sum = float(
        np.dot(active, active)
    )

    if active_squared_sum == 0.0:
        # All daily active returns are exactly zero.
        # No correction is necessary.
        C = 0.0
    else:
        C = residual / active_squared_sum

    # Important: a zero compounded active difference
    # does not imply C = 0 when daily active returns
    # offset one another.

    beta = M + C * active

    if not np.isfinite(beta).all():
        raise ValueError(
            "Non-finite Menchero linking coefficients."
        )

    return beta, float(M), float(C)


# Research reference: 08_portfolio_attribution.ipynb, code cell index 10.
def link_attribution_menchero(
    daily_sector_attribution: pd.DataFrame,
    *,
    portfolio_id: str,
    benchmark_id: str,
    start_date=None,
    end_date=None,
    tolerance: float = 1e-10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Link daily Brinson effects using Menchero optimization.

    Input grain:
        portfolio_id, benchmark_id, date, sector.

    Output:
        Sector-level linked effects and period summary.

    Target:
        Compounded portfolio return minus compounded
        benchmark return, in decimal return units.
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
            f"Missing Menchero columns: {sorted(missing)}"
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
        raise ValueError(
            "No attribution data in the selected period."
        )

    if data.duplicated(["date", "sector"]).any():
        raise ValueError(
            "Duplicate daily sector attribution keys."
        )

    # --------------------------------------------------------
    # 1. Obtain unique daily portfolio/benchmark returns
    # --------------------------------------------------------

    daily = (
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

    if daily["date"].duplicated().any():
        raise ValueError(
            "Inconsistent daily portfolio/benchmark returns."
        )

    rp = daily[
        "portfolio_return_total"
    ].to_numpy(dtype=float)

    rb = daily[
        "benchmark_return_total"
    ].to_numpy(dtype=float)

    # --------------------------------------------------------
    # 2. Calculate Menchero optimized linking coefficients
    # --------------------------------------------------------

    beta, M, C = menchero_linking_coefficients(
        rp, rb
    )

    daily["menchero_linking_multiplier"] = beta

    # --------------------------------------------------------
    # 3. Calculate compounded performance
    # --------------------------------------------------------

    portfolio_log_growth = np.log1p(rp).sum()
    benchmark_log_growth = np.log1p(rb).sum()

    portfolio_period_return = float(
        np.expm1(portfolio_log_growth)
    )

    benchmark_period_return = float(
        np.expm1(benchmark_log_growth)
    )

    compounded_active_return = (
        portfolio_period_return
        - benchmark_period_return
    )

    relative_wealth_return = float(
        np.expm1(
            portfolio_log_growth - benchmark_log_growth
        )
    )

    # --------------------------------------------------------
    # 4. Apply optimized multipliers to daily sector effects
    # --------------------------------------------------------

    data = data.merge(
        daily[
            ["date", "menchero_linking_multiplier"]
        ],
        on="date",
        how="left",
        validate="many_to_one",
    )

    for column in effect_columns:
        data[f"linked_{column}"] = (
            data[column]
            * data["menchero_linking_multiplier"]
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
    # 5. Reconcile optimized linked effects
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
            "Menchero linking reconciliation failed. "
            f"Residual: {reconciliation_error:.12e}"
        )

    # --------------------------------------------------------
    # 6. Standardized sector and period outputs
    # --------------------------------------------------------

    period_start = daily["date"].min()
    period_end = daily["date"].max()

    sector_linked.insert(
        0, "portfolio_id", portfolio_id
    )

    sector_linked.insert(
        1, "benchmark_id", benchmark_id
    )

    sector_linked["period_start"] = period_start
    sector_linked["period_end"] = period_end
    sector_linked["linking_method"] = "menchero"

    # Relative wealth is supplementary reporting,
    # not the target of Menchero arithmetic linking.
    sector_linked["linked_relative_wealth_effect"] = (
        sector_linked["linked_total_effect"]
        / (1.0 + benchmark_period_return)
    )

    summary = pd.DataFrame(
        [
            {
                "portfolio_id": portfolio_id,
                "benchmark_id": benchmark_id,
                "period_start": period_start,
                "period_end": period_end,
                "observations": len(daily),
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
                "linking_method": "menchero",
                "menchero_base_scale": M,
                "menchero_correction_coefficient": C,
            }
        ]
    )

    return sector_linked, summary
