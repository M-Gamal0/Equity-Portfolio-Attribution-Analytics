"""Attribution linking: validated research calculations."""

from __future__ import annotations

from .carino import carino_scaling_factor
from .menchero import menchero_linking_coefficients
import numpy as np
import pandas as pd



# Research reference: 08_portfolio_attribution.ipynb, code cell index 15.
def link_security_contributions_by_period(
    security_daily: pd.DataFrame,
    daily_returns: pd.DataFrame,
    period_attribution: pd.DataFrame,
    *,
    tolerance: float = 1e-10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Link daily security active contributions by reporting period.

    Returns
    -------
    security_period:
        Grain:
        portfolio_id, benchmark_id, window_id,
        linking_method, security_id, sector.

    reconciliation:
        Grain:
        portfolio_id, benchmark_id, window_id, linking_method.

    Notes
    -----
    Linked portfolio and benchmark legs are components
    of an active-return decomposition. Their individual
    sums are not standalone compounded portfolio returns.
    """

    pair_keys = ["portfolio_id", "benchmark_id"]

    period_keys = pair_keys + [
        "window_id",
        "linking_method",
    ]

    security_columns = pair_keys + [
        "date",
        "security_id",
        "sector",
        "portfolio_weight_bop",
        "benchmark_weight_bop",
        "portfolio_contribution_total",
        "benchmark_contribution_total",
        "active_contribution_total",
    ]

    return_columns = pair_keys + [
        "date",
        "portfolio_return_total",
        "benchmark_return_total",
    ]

    if period_attribution.duplicated(period_keys).any():
        raise ValueError("Duplicate period attribution keys.")

    # --------------------------------------------------------
    # 1. Index validated inputs by portfolio/benchmark pair
    # --------------------------------------------------------

    security_sources = {
        pair: frame[security_columns[2:]]
        .set_index("date")
        .sort_index()
        for pair, frame in security_daily.groupby(
            pair_keys, sort=False
        )
    }

    return_sources = {
        pair: frame[return_columns[2:]]
        .set_index("date")
        .sort_index()
        for pair, frame in daily_returns.groupby(
            pair_keys, sort=False
        )
    }

    security_results = []
    reconciliation_results = []

    # --------------------------------------------------------
    # 2. Process each approved reporting window and method
    # --------------------------------------------------------

    for pair, periods in period_attribution.groupby(
        pair_keys, sort=False
    ):

        if pair not in security_sources or pair not in return_sources:
            raise ValueError(f"Missing daily data for {pair}")

        security_source = security_sources[pair]
        return_source = return_sources[pair]

        for window in periods.itertuples(index=False):

            start = pd.Timestamp(window.period_start)
            end = pd.Timestamp(window.period_end)

            period_returns = return_source.loc[start:end]

            if len(period_returns) != window.observations:
                raise ValueError(
                    f"Return coverage mismatch: {window.window_id}"
                )

            rp = period_returns[
                "portfolio_return_total"
            ].to_numpy(dtype=float)

            rb = period_returns[
                "benchmark_return_total"
            ].to_numpy(dtype=float)

            # ------------------------------------------------
            # 3. Reuse the validated linking methodologies
            # ------------------------------------------------

            if window.linking_method == "carino":

                daily_factor = carino_scaling_factor(rp, rb)

                period_factor = carino_scaling_factor(
                    np.array([
                        window.portfolio_compounded_return
                    ]),
                    np.array([
                        window.benchmark_compounded_return
                    ]),
                )[0]

                multipliers = daily_factor / period_factor

            elif window.linking_method == "menchero":

                multipliers, _, _ = (
                    menchero_linking_coefficients(rp, rb)
                )

            else:
                raise ValueError(
                    f"Unsupported method: {window.linking_method}"
                )

            # ------------------------------------------------
            # 4. Select corresponding security contributions
            # ------------------------------------------------

            selected = security_source.loc[start:end].copy()

            daily_multipliers = pd.Series(
                multipliers,
                index=period_returns.index,
            )

            selected["linking_multiplier"] = (
                selected.index.map(daily_multipliers)
            )

            if selected["linking_multiplier"].isna().any():
                raise ValueError(
                    f"Missing daily security multipliers: "
                    f"{window.window_id}"
                )

            # ------------------------------------------------
            # 5. Link security contribution components
            # ------------------------------------------------

            selected["linked_portfolio_leg"] = (
                selected["portfolio_contribution_total"]
                * selected["linking_multiplier"]
            )

            selected["linked_benchmark_leg"] = (
                selected["benchmark_contribution_total"]
                * selected["linking_multiplier"]
            )

            selected["linked_active_contribution"] = (
                selected["active_contribution_total"]
                * selected["linking_multiplier"]
            )

            selected["portfolio_held"] = (
                selected["portfolio_weight_bop"] > 0
            )

            selected["benchmark_held"] = (
                selected["benchmark_weight_bop"] > 0
            )

            # ------------------------------------------------
            # 6. Aggregate by security and historical sector
            # ------------------------------------------------

            linked_securities = (
                selected.groupby(
                    ["security_id", "sector"],
                    as_index=False,
                    observed=True,
                )
                .agg(
                    linked_portfolio_leg=(
                        "linked_portfolio_leg", "sum"
                    ),
                    linked_benchmark_leg=(
                        "linked_benchmark_leg", "sum"
                    ),
                    linked_active_contribution=(
                        "linked_active_contribution", "sum"
                    ),
                    security_observation_days=(
                        "linked_active_contribution", "size"
                    ),
                    portfolio_held_days=(
                        "portfolio_held", "sum"
                    ),
                    benchmark_held_days=(
                        "benchmark_held", "sum"
                    ),
                )
            )

            # Preserve reporting metadata.

            for column in pair_keys:
                linked_securities[column] = getattr(
                    window, column
                )

            for column in [
                "window_id",
                "window_type",
                "window_label",
                "coverage_status",
                "linking_method",
                "period_start",
                "period_end",
            ]:
                linked_securities[column] = getattr(
                    window, column
                )

            security_results.append(linked_securities)

            # ------------------------------------------------
            # 7. Essential financial reconciliation
            # ------------------------------------------------

            security_total = float(
                linked_securities[
                    "linked_active_contribution"
                ].sum()
            )

            compounded_active = float(
                window.compounded_active_return
            )

            brinson_total = float(
                window.linked_total_effect
            )

            daily_multiplier_total = float(
                np.dot(multipliers, rp - rb)
            )

            security_error = (
                security_total - compounded_active
            )

            brinson_agreement_error = (
                security_total - brinson_total
            )

            multiplier_error = (
                daily_multiplier_total - compounded_active
            )

            if max(
                abs(security_error),
                abs(brinson_agreement_error),
                abs(multiplier_error),
            ) > tolerance:
                raise ValueError(
                    "Security linking reconciliation failed: "
                    f"{window.window_id} / "
                    f"{window.linking_method}"
                )

            reconciliation_results.append(
                {
                    "portfolio_id": pair[0],
                    "benchmark_id": pair[1],
                    "window_id": window.window_id,
                    "window_type": window.window_type,
                    "coverage_status": window.coverage_status,
                    "linking_method": window.linking_method,
                    "period_start": start,
                    "period_end": end,
                    "observations": len(period_returns),
                    "security_count": len(linked_securities),
                    "linked_security_total": security_total,
                    "compounded_active_return":
                        compounded_active,
                    "linked_brinson_total": brinson_total,
                    "security_reconciliation_error":
                        security_error,
                    "brinson_agreement_error":
                        brinson_agreement_error,
                    "multiplier_reconciliation_error":
                        multiplier_error,
                }
            )

    # --------------------------------------------------------
    # 8. Standardize final reporting outputs
    # --------------------------------------------------------

    security_period = (
        pd.concat(
            security_results,
            ignore_index=True,
        )
        .sort_values(
            period_keys + ["sector", "security_id"]
        )
        .reset_index(drop=True)
    )

    reconciliation = (
        pd.DataFrame(reconciliation_results)
        .sort_values(period_keys)
        .reset_index(drop=True)
    )

    return security_period, reconciliation
