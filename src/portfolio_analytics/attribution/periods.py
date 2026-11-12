"""Attribution periods: validated research calculations."""

from __future__ import annotations

import pandas as pd



# Research reference: 08_portfolio_attribution.ipynb, code cell index 13.
def build_period_attribution(
    daily_sector_attribution: pd.DataFrame,
    reporting_windows: pd.DataFrame,
    *,
    linkers: dict,
    tolerance: float = 1e-10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate linked sector and total attribution by window."""

    period_metadata = [
        "window_id",
        "window_type",
        "window_label",
        "calendar_start",
        "calendar_end",
        "valuation_anchor_date",
        "coverage_status",
        "is_full_period",
    ]

    period_keys = [
        "portfolio_id",
        "benchmark_id",
        "window_id",
        "linking_method",
    ]

    # Pre-index returns to avoid rescanning the full history
    # for every monthly, quarterly, and annual window.
    sources = {
        pair: group.set_index("date").sort_index()
        for pair, group in daily_sector_attribution.groupby(
            ["portfolio_id", "benchmark_id"],
            sort=False,
        )
    }

    sector_results = []
    total_results = []
    excluded_windows = []

    for _, window in reporting_windows.iterrows():

        if window["coverage_status"] == "INSUFFICIENT_HISTORY":
            excluded_windows.append(window["window_id"])
            continue

        portfolio_id = window["portfolio_id"]
        benchmark_id = window["benchmark_id"]

        pair = (portfolio_id, benchmark_id)

        if pair not in sources:
            raise ValueError(
                f"Missing attribution data for {pair}"
            )

        window_data = (
            sources[pair]
            .loc[
                window["start_date"]:window["end_date"]
            ]
            .reset_index()
        )

        metadata = {
            column: window[column]
            for column in period_metadata
        }

        for method, linking_function in linkers.items():

            sector_result, total_result = linking_function(
                window_data,
                portfolio_id=portfolio_id,
                benchmark_id=benchmark_id,
                tolerance=tolerance,
            )

            actual = total_result.iloc[0]

            # Essential reporting-window integrity check.
            if (
                actual["period_start"] != window["start_date"]
                or actual["period_end"] != window["end_date"]
                or actual["observations"] != window["observations"]
                or actual["linking_method"] != method
            ):
                raise ValueError(
                    "Reporting-window mismatch: "
                    f"{window['window_id']} / {method}"
                )

            for column, value in metadata.items():
                sector_result[column] = value
                total_result[column] = value

            sector_results.append(sector_result)
            total_results.append(total_result)

    if not total_results:
        raise ValueError(
            "No eligible attribution reporting windows."
        )

    sector_period_df = pd.concat(
        sector_results,
        ignore_index=True,
    )

    total_period_df = pd.concat(
        total_results,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Financial validation: period attribution reconciliation
    # --------------------------------------------------------

    if total_period_df.duplicated(period_keys).any():
        raise ValueError(
            "Duplicate period attribution keys."
        )

    sector_keys = period_keys + ["sector"]

    if sector_period_df.duplicated(sector_keys).any():
        raise ValueError(
            "Duplicate sector-period attribution keys."
        )

    maximum_reconciliation_error = (
        total_period_df["reconciliation_error"]
        .abs()
        .max()
    )

    if maximum_reconciliation_error > tolerance:
        raise ValueError(
            "Period attribution reconciliation failed."
        )

    # Verify the aggregated sector effects reproduce
    # the corresponding period-level linked totals.
    sector_sums = (
        sector_period_df.groupby(
            period_keys,
            as_index=False,
        )["linked_total_effect"]
        .sum()
        .rename(
            columns={
                "linked_total_effect":
                    "summed_sector_linked_effect"
            }
        )
    )

    comparison = total_period_df.merge(
        sector_sums,
        on=period_keys,
        how="left",
        validate="one_to_one",
    )

    if comparison["summed_sector_linked_effect"].isna().any():
        raise ValueError(
            "Missing sector attribution for a reporting period."
        )

    sector_reconciliation_error = (
        comparison["summed_sector_linked_effect"]
        - comparison["linked_total_effect"]
    )

    if sector_reconciliation_error.abs().max() > tolerance:
        raise ValueError(
            "Sector-period attribution does not reconcile."
        )

    # --------------------------------------------------------
    # Cross-method consistency
    # --------------------------------------------------------

    performance_columns = [
        "portfolio_compounded_return",
        "benchmark_compounded_return",
        "compounded_active_return",
        "relative_wealth_return",
    ]

    paired_returns = total_period_df.pivot(
        index=period_keys[:-1],
        columns="linking_method",
        values=performance_columns,
    )

    expected_methods = set(linkers)

    observed_methods = set(
        total_period_df["linking_method"]
    )

    if observed_methods != expected_methods:
        raise ValueError(
            "Incomplete attribution linking methods."
        )

    if {"carino", "menchero"}.issubset(expected_methods):

        for metric in performance_columns:

            difference = (
                paired_returns[(metric, "carino")]
                - paired_returns[(metric, "menchero")]
            )

            if (
                difference.isna().any()
                or difference.abs().max() > tolerance
            ):
                raise ValueError(
                    "Carino/Menchero performance mismatch: "
                    f"{metric}"
                )

    sector_period_df = (
        sector_period_df.sort_values(
            period_keys + ["sector"]
        )
        .reset_index(drop=True)
    )

    total_period_df = (
        total_period_df.sort_values(period_keys)
        .reset_index(drop=True)
    )

    return sector_period_df, total_period_df


# Research reference: 08_portfolio_attribution.ipynb, code cell index 14.
def compare_attribution_linking_methods(
    period_attribution: pd.DataFrame,
    sector_attribution: pd.DataFrame,
    *,
    tolerance: float = 1e-10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare Carino and Menchero attribution by period and sector."""

    effects = [
        "linked_allocation_effect",
        "linked_selection_effect",
        "linked_interaction_effect",
        "linked_total_effect",
    ]

    performance = [
        "portfolio_compounded_return",
        "benchmark_compounded_return",
        "compounded_active_return",
        "relative_wealth_return",
    ]

    period_keys = [
        "portfolio_id",
        "benchmark_id",
        "window_id",
    ]

    sector_keys = period_keys + ["sector"]

    def pair_methods(frame, keys, metadata, metrics):
        carino = frame.loc[
            frame["linking_method"].eq("carino"),
            keys + metadata + metrics,
        ].copy()

        menchero = frame.loc[
            frame["linking_method"].eq("menchero"),
            keys + metrics,
        ].copy()

        paired = carino.merge(
            menchero,
            on=keys,
            how="outer",
            validate="one_to_one",
            suffixes=("_carino", "_menchero"),
            indicator=True,
        )

        if not paired["_merge"].eq("both").all():
            raise ValueError(
                "Carino and Menchero reporting keys differ."
            )

        paired = paired.drop(columns="_merge")

        for effect in effects:
            paired[f"{effect}_difference"] = (
                paired[f"{effect}_menchero"]
                - paired[f"{effect}_carino"]
            )

        return paired

    # --------------------------------------------------------
    # 1. Compare period-level results
    # --------------------------------------------------------

    period_metadata = [
        "window_type",
        "window_label",
        "coverage_status",
        "is_full_period",
        "period_start",
        "period_end",
        "observations",
    ]

    period_comparison = pair_methods(
        period_attribution,
        period_keys,
        period_metadata,
        effects + performance,
    )

    # Compounded performance must be method-independent.
    for metric in performance:
        maximum_difference = (
            period_comparison[f"{metric}_menchero"]
            - period_comparison[f"{metric}_carino"]
        ).abs().max()

        if maximum_difference > tolerance:
            raise ValueError(
                f"Compounded performance differs: {metric}"
            )

    # --------------------------------------------------------
    # 2. Compare sector-level results
    # --------------------------------------------------------

    sector_comparison = pair_methods(
        sector_attribution,
        sector_keys,
        [
            "window_type",
            "window_label",
            "coverage_status",
        ],
        effects,
    )

    # --------------------------------------------------------
    # 3. Reconcile sector-level differences to period totals
    # --------------------------------------------------------

    difference_columns = [
        f"{effect}_difference"
        for effect in effects
    ]

    sector_difference_totals = (
        sector_comparison.groupby(
            period_keys,
            as_index=False,
        )[difference_columns]
        .sum()
    )

    check = period_comparison.merge(
        sector_difference_totals,
        on=period_keys,
        how="left",
        validate="one_to_one",
        suffixes=("_period", "_sectors"),
    )

    for column in difference_columns:
        residual = (
            check[f"{column}_sectors"]
            - check[f"{column}_period"]
        )

        if (
            residual.isna().any()
            or residual.abs().max() > tolerance
        ):
            raise ValueError(
                f"Method sensitivity reconciliation failed: "
                f"{column}"
            )

    # The two methods must have the same total attribution.
    total_method_difference = (
        period_comparison[
            "linked_total_effect_difference"
        ]
        .abs()
        .max()
    )

    if total_method_difference > tolerance:
        raise ValueError(
            "Linking methods disagree on total attribution."
        )

    period_comparison = (
        period_comparison.sort_values(
            ["period_start", "window_id"]
        )
        .reset_index(drop=True)
    )

    sector_comparison = (
        sector_comparison.sort_values(
            sector_keys
        )
        .reset_index(drop=True)
    )

    return period_comparison, sector_comparison
