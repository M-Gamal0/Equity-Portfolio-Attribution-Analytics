"""Attribution security: validated research calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd



# Research reference: 08_portfolio_attribution.ipynb, code cell index 2.
def calculate_security_active_contributions(
    portfolio_holdings: pd.DataFrame,
    aligned_returns: pd.DataFrame,
    *,
    portfolio_id: str,
    benchmark_id: str,
    reconciliation_tolerance: float = 1e-12,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Calculate and reconcile daily security-level active contributions.

    Required holdings columns:
        portfolio_id, date, security_id, sector,
        weight_bop, return_total.

    Optional holdings column:
        contribution_total, used for independent verification.

    Required aligned-return columns:
        date, portfolio_return_total,
        benchmark_return_total, arithmetic_active_return.

    All returns, weights and contributions use decimal units.

    Returns
    -------
    security_contributions_daily:
        One row per portfolio, date and held security.

    security_active_contributions_daily:
        One row per date and security held by either portfolio.
        Includes separate portfolio/benchmark contributions.

    security_active_reconciliation:
        One row per date, containing independently calculated
        returns, supplied returns, and reconciliation residuals.

    Raises
    ------
    ValueError:
        On invalid inputs, inconsistent overlapping security
        observations, or failed numerical reconciliation.
    """

    if portfolio_id == benchmark_id:
        raise ValueError(
            "Portfolio and benchmark identifiers must differ."
        )

    if (
        not np.isfinite(reconciliation_tolerance)
        or reconciliation_tolerance < 0
    ):
        raise ValueError(
            "Reconciliation tolerance must be finite "
            "and nonnegative."
        )

    # --------------------------------------------------------
    # 1. Prepare required security-level inputs
    # --------------------------------------------------------

    required_holdings_columns = {
        "portfolio_id",
        "date",
        "security_id",
        "sector",
        "weight_bop",
        "return_total",
    }

    required_return_columns = {
        "date",
        "portfolio_return_total",
        "benchmark_return_total",
        "arithmetic_active_return",
    }

    missing_holdings = (
        required_holdings_columns
        - set(portfolio_holdings.columns)
    )

    missing_returns = (
        required_return_columns
        - set(aligned_returns.columns)
    )

    if missing_holdings or missing_returns:
        raise ValueError(
            "Missing required attribution fields. "
            f"Holdings: {sorted(missing_holdings)}; "
            f"Returns: {sorted(missing_returns)}"
        )

    holding_columns = [
        "portfolio_id",
        "date",
        "security_id",
        "sector",
        "weight_bop",
        "return_total",
    ]

    saved_contributions_available = (
        "contribution_total" in portfolio_holdings.columns
    )

    if saved_contributions_available:
        holding_columns.append("contribution_total")

    security_contributions_daily = (
        portfolio_holdings.loc[
            portfolio_holdings["portfolio_id"].isin(
                [portfolio_id, benchmark_id]
            ),
            holding_columns,
        ]
        .copy()
    )

    if security_contributions_daily.empty:
        raise ValueError(
            "No holdings available for the selected portfolios."
        )

    security_contributions_daily["date"] = pd.to_datetime(
        security_contributions_daily["date"],
        errors="raise",
    )

    if security_contributions_daily[
        ["portfolio_id", "date", "security_id", "sector"]
    ].isna().any().any():
        raise ValueError(
            "Security holdings contain null identifiers or sectors."
        )

    security_keys = [
        "portfolio_id",
        "date",
        "security_id",
    ]

    if security_contributions_daily.duplicated(
        security_keys
    ).any():
        raise ValueError(
            "Duplicate security contribution keys detected."
        )

    numeric_columns = [
        "weight_bop",
        "return_total",
    ]

    if saved_contributions_available:
        numeric_columns.append("contribution_total")

    for column in numeric_columns:
        values = security_contributions_daily[column]

        if (
            not pd.api.types.is_numeric_dtype(values)
            or not np.isfinite(
                values.to_numpy(dtype=float)
            ).all()
        ):
            raise ValueError(
                f"Invalid numeric holdings field: {column}"
            )

    if security_contributions_daily["weight_bop"].lt(0).any():
        raise ValueError(
            "Negative security weights are unsupported."
        )

    # --------------------------------------------------------
    # 2. Independently calculate security return contributions
    # --------------------------------------------------------

    calculated_contribution = (
        security_contributions_daily["weight_bop"]
        * security_contributions_daily["return_total"]
    )

    if saved_contributions_available:

        saved_contribution = (
            security_contributions_daily["contribution_total"]
        )

        security_contributions_daily[
            "saved_contribution_error"
        ] = calculated_contribution - saved_contribution

        max_saved_error = float(
            security_contributions_daily[
                "saved_contribution_error"
            ].abs().max()
        )

        if max_saved_error > reconciliation_tolerance:
            raise ValueError(
                "Calculated security contributions disagree "
                "with saved contributions. "
                f"Maximum error: {max_saved_error:.3e}"
            )

        # Retain the independently calculated value.
        security_contributions_daily = (
            security_contributions_daily.drop(
                columns=["contribution_total"]
            )
        )

    security_contributions_daily[
        "contribution_total"
    ] = calculated_contribution

    # --------------------------------------------------------
    # 3. Prepare portfolio and benchmark security ledgers
    # --------------------------------------------------------

    comparison_columns = [
        "date",
        "security_id",
        "sector",
        "weight_bop",
        "return_total",
        "contribution_total",
    ]

    portfolio_security = (
        security_contributions_daily.loc[
            security_contributions_daily[
                "portfolio_id"
            ].eq(portfolio_id),
            comparison_columns,
        ]
        .rename(
            columns={
                column: f"portfolio_{column}"
                for column in comparison_columns
                if column not in ["date", "security_id"]
            }
        )
    )

    benchmark_security = (
        security_contributions_daily.loc[
            security_contributions_daily[
                "portfolio_id"
            ].eq(benchmark_id),
            comparison_columns,
        ]
        .rename(
            columns={
                column: f"benchmark_{column}"
                for column in comparison_columns
                if column not in ["date", "security_id"]
            }
        )
    )

    if portfolio_security.empty or benchmark_security.empty:
        raise ValueError(
            "Both portfolio and benchmark holdings are required."
        )

    # --------------------------------------------------------
    # 4. Full outer join across all held securities
    # --------------------------------------------------------

    security_active_contributions_daily = (
        portfolio_security.merge(
            benchmark_security,
            on=["date", "security_id"],
            how="outer",
            validate="one_to_one",
            indicator="security_presence",
            sort=False,
        )
    )

    active_security = security_active_contributions_daily

    overlapping = active_security[
        "security_presence"
    ].eq("both")

    # --------------------------------------------------------
    # 5. Validate overlapping-security total returns
    # --------------------------------------------------------

    overlapping_return_errors = (
        active_security.loc[
            overlapping, "portfolio_return_total"
        ]
        - active_security.loc[
            overlapping, "benchmark_return_total"
        ]
    ).abs()

    max_overlapping_return_error = (
        float(overlapping_return_errors.max())
        if len(overlapping_return_errors)
        else 0.0
    )

    if (
        max_overlapping_return_error
        > reconciliation_tolerance
    ):
        raise ValueError(
            "Common security has inconsistent portfolio "
            "and benchmark total returns. "
            f"Maximum difference: "
            f"{max_overlapping_return_error:.3e}"
        )

    conflicting_sectors = (
        active_security.loc[
            overlapping, "portfolio_sector"
        ]
        .ne(
            active_security.loc[
                overlapping, "benchmark_sector"
            ]
        )
    )

    if conflicting_sectors.any():
        raise ValueError(
            "A common security has conflicting sector labels."
        )

    # --------------------------------------------------------
    # 6. Construct the common security classification
    # --------------------------------------------------------

    active_security["sector"] = (
        active_security["portfolio_sector"]
        .combine_first(
            active_security["benchmark_sector"]
        )
    )

    presence_mapping = {
        "both": "BOTH_HELD",
        "left_only": "PORTFOLIO_ONLY",
        "right_only": "BENCHMARK_ONLY",
    }

    active_security["security_presence_status"] = (
        active_security["security_presence"]
        .astype(str)
        .map(presence_mapping)
    )

    # --------------------------------------------------------
    # 7. Handle absent positions, without imputing returns
    # --------------------------------------------------------

    position_columns = [
        "portfolio_weight_bop",
        "benchmark_weight_bop",
        "portfolio_contribution_total",
        "benchmark_contribution_total",
    ]

    # An absent position has zero weight and contribution.
    # Security returns remain NaN where no position exists.
    for column in position_columns:
        active_security[column] = (
            active_security[column].fillna(0.0)
        )

    active_security["active_contribution_total"] = (
        active_security["portfolio_contribution_total"]
        - active_security["benchmark_contribution_total"]
    )

    active_security.insert(
        0, "portfolio_id", portfolio_id
    )

    active_security.insert(
        1, "benchmark_id", benchmark_id
    )

    active_security = active_security.drop(
        columns=[
            "portfolio_sector",
            "benchmark_sector",
            "security_presence",
        ]
    )

    # --------------------------------------------------------
    # 8. Independently aggregate daily contributions
    # --------------------------------------------------------

    daily_calculated = (
        active_security.groupby(
            "date",
            as_index=False,
        )[
            [
                "portfolio_contribution_total",
                "benchmark_contribution_total",
                "active_contribution_total",
            ]
        ]
        .sum()
        .rename(
            columns={
                "portfolio_contribution_total":
                    "calculated_portfolio_return",
                "benchmark_contribution_total":
                    "calculated_benchmark_return",
                "active_contribution_total":
                    "calculated_active_return",
            }
        )
    )

    expected_returns = aligned_returns[
        [
            "date",
            "portfolio_return_total",
            "benchmark_return_total",
            "arithmetic_active_return",
        ]
    ].copy()

    expected_returns["date"] = pd.to_datetime(
        expected_returns["date"],
        errors="raise",
    )

    if expected_returns["date"].duplicated().any():
        raise ValueError(
            "Aligned returns contain duplicate dates."
        )

    for column in [
        "portfolio_return_total",
        "benchmark_return_total",
        "arithmetic_active_return",
    ]:
        if not np.isfinite(
            expected_returns[column].to_numpy(dtype=float)
        ).all():
            raise ValueError(
                f"Aligned returns contain non-finite {column}."
            )

    security_active_reconciliation = (
        expected_returns.merge(
            daily_calculated,
            on="date",
            how="outer",
            validate="one_to_one",
            indicator=True,
        )
    )

    if not security_active_reconciliation[
        "_merge"
    ].eq("both").all():
        raise ValueError(
            "Security contribution dates do not match "
            "aligned portfolio return dates."
        )

    security_active_reconciliation = (
        security_active_reconciliation.drop(
            columns="_merge"
        )
    )

    # --------------------------------------------------------
    # 9. Calculate reconciliation residuals
    # --------------------------------------------------------

    reconciliation_equations = {
        "portfolio_reconciliation_error": (
            "calculated_portfolio_return",
            "portfolio_return_total",
        ),
        "benchmark_reconciliation_error": (
            "calculated_benchmark_return",
            "benchmark_return_total",
        ),
        "active_reconciliation_error": (
            "calculated_active_return",
            "arithmetic_active_return",
        ),
    }

    for error_column, (
        calculated_column,
        expected_column,
    ) in reconciliation_equations.items():

        security_active_reconciliation[error_column] = (
            security_active_reconciliation[calculated_column]
            - security_active_reconciliation[expected_column]
        )

        maximum_error = float(
            security_active_reconciliation[
                error_column
            ].abs().max()
        )

        if maximum_error > reconciliation_tolerance:
            raise ValueError(
                f"{error_column} failed. "
                f"Maximum error: {maximum_error:.3e}"
            )

    # --------------------------------------------------------
    # 10. Return standardized analytical results
    # --------------------------------------------------------

    security_contributions_daily = (
        security_contributions_daily.sort_values(
            security_keys
        ).reset_index(drop=True)
    )

    active_security = (
        active_security.sort_values(
            ["date", "security_id"]
        ).reset_index(drop=True)
    )

    security_active_reconciliation.insert(
        0, "portfolio_id", portfolio_id
    )

    security_active_reconciliation.insert(
        1, "benchmark_id", benchmark_id
    )

    security_active_reconciliation = (
        security_active_reconciliation.sort_values(
            "date"
        ).reset_index(drop=True)
    )

    return (
        security_contributions_daily,
        active_security,
        security_active_reconciliation,
    )
