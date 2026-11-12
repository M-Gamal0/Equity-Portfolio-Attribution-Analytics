"""Performance summaries: validated research calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd



# Research reference: 07_performance_analytics.ipynb, code cell index 18.
def build_performance_summary(
    absolute_df: pd.DataFrame,
    relative_df: pd.DataFrame,
    capture_df: pd.DataFrame,
    return_basis_df: pd.DataFrame,
    *,
    portfolio_id: str,
    benchmark_portfolio_id: str,
    as_of_date: pd.Timestamp,
) -> pd.DataFrame:
    """Combine independently validated window-level results."""

    pair_keys = [
        "portfolio_id",
        "benchmark_portfolio_id",
        "as_of_date",
        "window_id",
    ]

    as_of = pd.Timestamp(as_of_date)

    report = relative_df.loc[
        relative_df["portfolio_id"].eq(portfolio_id)
        & relative_df["benchmark_portfolio_id"].eq(
            benchmark_portfolio_id
        )
        & relative_df["as_of_date"].eq(as_of),
        pair_keys
        + [
            "resolved_start_date",
            "observation_count",
            "calendar_days",
            "arithmetic_active_period_return",
            "geometric_relative_period_return",
            "annualized_cagr_spread",
            "annualized_geometric_relative_return",
            "calculation_status",
            "cagr_status",
        ],
    ].copy()

    report = report.rename(
        columns={
            "calculation_status": "relative_calculation_status",
            "cagr_status": "relative_cagr_status",
        }
    )

    if report.empty or report.duplicated(pair_keys).any():
        raise ValueError(
            "Missing or duplicate relative reporting windows."
        )

    # Add each portfolio's independently calculated
    # absolute return and CAGR.

    for selected_id, prefix in (
        (portfolio_id, "active"),
        (benchmark_portfolio_id, "benchmark"),
    ):

        absolute = absolute_df.loc[
            absolute_df["portfolio_id"].eq(selected_id)
            & absolute_df["as_of_date"].eq(as_of),
            [
                "portfolio_id",
                "as_of_date",
                "window_id",
                "period_return",
                "annualized_return_cagr",
                "cagr_status",
            ],
        ].copy()

        absolute = absolute.rename(
            columns={
                "period_return": f"{prefix}_period_return",
                "annualized_return_cagr": f"{prefix}_cagr",
                "cagr_status": f"{prefix}_cagr_status",
            }
        )

        if prefix == "benchmark":
            absolute = absolute.rename(
                columns={
                    "portfolio_id": "benchmark_portfolio_id"
                }
            )

        join_keys = [
            "portfolio_id"
            if prefix == "active"
            else "benchmark_portfolio_id",
            "as_of_date",
            "window_id",
        ]

        report = report.merge(
            absolute,
            on=join_keys,
            how="left",
            validate="one_to_one",
        )

        if report[f"{prefix}_cagr_status"].isna().any():
            raise ValueError(
                f"Missing {prefix} absolute-window results."
            )

    # Add independently validated market capture.

    capture = capture_df.loc[
        capture_df["as_of_date"].eq(as_of),
        pair_keys
        + [
            "up_arithmetic_capture_pct",
            "down_arithmetic_capture_pct",
            "up_observations",
            "down_observations",
            "flat_observations",
        ],
    ]

    report = report.merge(
        capture,
        on=pair_keys,
        how="left",
        validate="one_to_one",
    )

    if report["up_observations"].isna().any():
        raise ValueError("Missing market capture results.")

    # Add price-return results without confusing
    # the total-return premium with dividend yield.

    basis_columns = [
        "active_period_return_price",
        "benchmark_period_return_price",
        "active_total_return_premium_pp",
        "benchmark_total_return_premium_pp",
        "geometric_relative_return_price",
    ]

    basis = return_basis_df.loc[
        return_basis_df["as_of_date"].eq(as_of),
        pair_keys + basis_columns,
    ]

    report = report.merge(
        basis,
        on=pair_keys,
        how="left",
        validate="one_to_one",
    )

    if report[basis_columns].isna().any().any():
        raise ValueError("Missing price/total comparison results.")

    # Explicit percentage-point fields.

    report["arithmetic_active_period_return_pp"] = (
        100.0 * report["arithmetic_active_period_return"]
    )

    report["annualized_cagr_spread_pp"] = (
        100.0 * report["annualized_cagr_spread"]
    )

    # Independent identities, not repeated formula tests.

    np.testing.assert_allclose(
        (
            report["active_period_return"]
            - report["benchmark_period_return"]
        ).to_numpy(),
        report["arithmetic_active_period_return"].to_numpy(),
        rtol=5e-12,
        atol=5e-12,
        equal_nan=True,
    )

    np.testing.assert_allclose(
        (
            report["active_cagr"]
            - report["benchmark_cagr"]
        ).to_numpy(),
        report["annualized_cagr_spread"].to_numpy(),
        rtol=5e-12,
        atol=5e-12,
        equal_nan=True,
    )

    report["reporting_currency_assumption"] = "USD"
    report["currency_verification_status"] = (
        "ASSUMED_NOT_SOURCE_AUDITED"
    )

    report["publication_status"] = "NOT_APPROVED"

    report["session_coverage_status"] = (
        "NOT_ASSESSED_EXCHANGE_CALENDAR"
    )

    return (
        report
        .sort_values(["as_of_date", "window_id"])
        .reset_index(drop=True)
    )


# Research reference: 07_performance_analytics.ipynb, code cell index 18.
def build_risk_adjusted_summary(
    risk_metrics_df: pd.DataFrame,
) -> pd.DataFrame:
    """Retain metric values, cutoffs, and eligibility statuses."""

    keys = [
        "portfolio_id",
        "benchmark_portfolio_id",
        "as_of_date",
        "window_id",
    ]

    metric_keys = keys + ["metric_name"]

    if risk_metrics_df.duplicated(metric_keys).any():
        raise ValueError("Duplicate risk-adjusted metric keys.")

    values = risk_metrics_df.pivot(
        index=keys,
        columns="metric_name",
        values="metric_value",
    )

    statuses = risk_metrics_df.pivot(
        index=keys,
        columns="metric_name",
        values="calculation_status",
    ).add_suffix("_status")

    coverage = (
        risk_metrics_df
        .groupby(keys, as_index=True)
        .agg(
            return_observations=(
                "return_observations", "first"
            ),
            rf_available_observations=(
                "rf_available_observations", "first"
            ),
        )
    )

    summary = pd.concat(
        [coverage, values, statuses],
        axis=1,
    ).reset_index()

    summary.columns.name = None

    summary["rf_missing_observations"] = (
        summary["return_observations"]
        - summary["rf_available_observations"]
    )

    summary["reporting_currency_assumption"] = "USD"

    return (
        summary
        .sort_values(["as_of_date", "window_id"])
        .reset_index(drop=True)
    )
