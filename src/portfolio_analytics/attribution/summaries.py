"""Attribution summaries: validated research calculations."""

from __future__ import annotations

import pandas as pd



# Research reference: 08_portfolio_attribution.ipynb, code cell index 16.
def build_attribution_reporting_tables(
    period_attribution: pd.DataFrame,
    sector_attribution: pd.DataFrame,
    security_attribution: pd.DataFrame,
    *,
    tolerance: float = 1e-10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build reconciled period, sector and security reports."""

    keys = [
        "portfolio_id",
        "benchmark_id",
        "window_id",
        "linking_method",
    ]

    sector_keys = keys + ["sector"]

    # --------------------------------------------------------
    # 1. Portfolio-level attribution report
    # --------------------------------------------------------

    period_report = period_attribution.copy()

    period_metrics = {
        "portfolio_compounded_return": "portfolio_return_pp",
        "benchmark_compounded_return": "benchmark_return_pp",
        "compounded_active_return": "active_return_pp",
        "linked_allocation_effect": "allocation_pp",
        "linked_selection_effect": "selection_pp",
        "linked_interaction_effect": "interaction_pp",
        "linked_total_effect": "total_attribution_pp",
    }

    for source, target in period_metrics.items():
        period_report[target] = 100.0 * period_report[source]

    # --------------------------------------------------------
    # 2. Aggregate linked security contributions by sector
    # --------------------------------------------------------

    security_sector_totals = (
        security_attribution.groupby(
            sector_keys,
            as_index=False,
            observed=True,
        )
        .agg(
            linked_security_active_contribution=(
                "linked_active_contribution", "sum"
            ),
            security_count=("security_id", "nunique"),
            portfolio_held_security_count=(
                "portfolio_held_days",
                lambda values: int((values > 0).sum()),
            ),
            benchmark_held_security_count=(
                "benchmark_held_days",
                lambda values: int((values > 0).sum()),
            ),
        )
    )

    # --------------------------------------------------------
    # 3. Integrate Brinson and security attribution
    # --------------------------------------------------------

    sector_report = sector_attribution.merge(
        security_sector_totals,
        on=sector_keys,
        how="left",
        validate="one_to_one",
    )

    if sector_report[
        "linked_security_active_contribution"
    ].isna().any():
        raise ValueError(
            "A Brinson reporting sector has no linked "
            "security contribution data."
        )

    # Brinson effects differ from direct sector active
    # contributions by the linked benchmark-reference term:
    #
    #   sum_t beta_t * (wP_sector,t - wB_sector,t) * RB_total,t
    #
    # This is a methodology bridge, not an unexplained residual.

    sector_report["linked_benchmark_reference_bridge"] = (
        sector_report["linked_security_active_contribution"]
        - sector_report["linked_total_effect"]
    )

    sector_metrics = {
        "linked_allocation_effect": "allocation_pp",
        "linked_selection_effect": "selection_pp",
        "linked_interaction_effect": "interaction_pp",
        "linked_total_effect": "brinson_total_pp",
        "linked_security_active_contribution":
            "security_active_contribution_pp",
        "linked_benchmark_reference_bridge":
            "benchmark_reference_bridge_pp",
    }

    for source, target in sector_metrics.items():
        sector_report[target] = 100.0 * sector_report[source]

    # --------------------------------------------------------
    # 4. Security-level contribution report and rankings
    # --------------------------------------------------------

    security_report = security_attribution.copy()

    security_report["active_contribution_pp"] = (
        100.0
        * security_report["linked_active_contribution"]
    )

    security_report["portfolio_leg_pp"] = (
        100.0 * security_report["linked_portfolio_leg"]
    )

    security_report["benchmark_leg_pp"] = (
        100.0 * security_report["linked_benchmark_leg"]
    )

    # Rank deterministically within each reporting window.
    # Ties are resolved by security identifier.

    security_report = security_report.sort_values(
        keys + ["linked_active_contribution", "security_id"],
        ascending=[True] * len(keys) + [False, True],
        kind="mergesort",
    )

    security_report["positive_contribution_rank"] = (
        security_report.groupby(keys).cumcount() + 1
    )

    security_report = security_report.sort_values(
        keys + ["linked_active_contribution", "security_id"],
        ascending=[True] * len(keys) + [True, True],
        kind="mergesort",
    )

    security_report["negative_contribution_rank"] = (
        security_report.groupby(keys).cumcount() + 1
    )

    # --------------------------------------------------------
    # 5. Essential reporting reconciliation
    # --------------------------------------------------------

    sector_totals = (
        sector_report.groupby(
            keys,
            as_index=False,
        )
        .agg(
            sector_brinson_total=(
                "linked_total_effect", "sum"
            ),
            sector_security_total=(
                "linked_security_active_contribution", "sum"
            ),
            benchmark_reference_bridge_total=(
                "linked_benchmark_reference_bridge", "sum"
            ),
        )
    )

    reconciliation = period_report.merge(
        sector_totals,
        on=keys,
        how="left",
        validate="one_to_one",
    )

    reconciliation_errors = pd.DataFrame(
        {
            "brinson": (
                reconciliation["sector_brinson_total"]
                - reconciliation["linked_total_effect"]
            ),
            "security": (
                reconciliation["sector_security_total"]
                - reconciliation["compounded_active_return"]
            ),
            "sector_bridge": (
                reconciliation[
                    "benchmark_reference_bridge_total"
                ]
            ),
        }
    )

    maximum_error = reconciliation_errors.abs().max().max()

    if (
        reconciliation_errors.isna().any().any()
        or maximum_error > tolerance
    ):
        raise ValueError(
            "Consolidated attribution reporting "
            "failed financial reconciliation."
        )

    # --------------------------------------------------------
    # 6. Standardized, deterministic reporting outputs
    # --------------------------------------------------------

    period_report = (
        period_report.sort_values(keys)
        .reset_index(drop=True)
    )

    sector_report = (
        sector_report.sort_values(sector_keys)
        .reset_index(drop=True)
    )

    security_report = (
        security_report.sort_values(
            keys + ["positive_contribution_rank"]
        )
        .reset_index(drop=True)
    )

    return period_report, sector_report, security_report
