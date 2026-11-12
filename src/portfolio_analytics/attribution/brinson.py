"""Attribution brinson: validated research calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd



# Research reference: 08_portfolio_attribution.ipynb, code cell index 5.
def calculate_daily_brinson_fachler(
    sector_comparison: pd.DataFrame,
    *,
    return_tolerance: float = 1e-12,
    weight_tolerance: float = 1e-12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate three-effect daily Brinson-Fachler attribution.

    Parameters
    ----------
    sector_comparison:
        Aligned daily portfolio and benchmark sector data
        containing BOP weights, contributions, observed returns,
        attribution reference returns, and daily total returns.

    return_tolerance:
        Maximum absolute financial reconciliation error.

    weight_tolerance:
        Maximum absolute deviation of daily sector weight sums
        from one.

    Returns
    -------
    brinson_sector_daily:
        Grain: portfolio_id, benchmark_id, date, sector.
        Contains sector-level allocation, selection, interaction,
        total effects, and financial reconciliation information.

    brinson_total_daily:
        Grain: portfolio_id, benchmark_id, date.
        Contains aggregated daily attribution effects,
        arithmetic active returns, and reconciliation errors.

    Notes
    -----
    Missing sector positions have zero weights and contributions.

    Attribution reference returns must follow the explicitly
    defined zero-weight sector policy.

    The method attributes arithmetic daily active returns.
    Multi-period attribution requires a separate linking method.

    Raises
    ------
    ValueError:
        If input contracts, sector reference conventions,
        portfolio weights, returns, or attribution identities
        are inconsistent.
    """

    # --------------------------------------------------------
    # 1. Validate function settings
    # --------------------------------------------------------

    for name, tolerance in [
        ("return_tolerance", return_tolerance),
        ("weight_tolerance", weight_tolerance),
    ]:
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError(
                f"{name} must be finite and nonnegative."
            )

    required_columns = {
        "portfolio_id",
        "benchmark_id",
        "date",
        "sector",
        "portfolio_sector_weight_bop",
        "benchmark_sector_weight_bop",
        "portfolio_sector_contribution_total",
        "benchmark_sector_contribution_total",
        "portfolio_sector_return_observed",
        "benchmark_sector_return_observed",
        "portfolio_sector_return_for_brinson",
        "benchmark_sector_return_for_brinson",
        "portfolio_return_total",
        "benchmark_return_total",
        "arithmetic_active_return",
        "sector_presence_status",
        "attribution_reference_status",
    }

    missing_columns = (
        required_columns - set(sector_comparison.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing Brinson-Fachler input columns: "
            f"{sorted(missing_columns)}"
        )

    # Preserve inputs and prevent accidental mutation.
    sector = sector_comparison.loc[
        :, sorted(required_columns)
    ].copy()

    if sector.empty:
        raise ValueError(
            "Brinson-Fachler sector input is empty."
        )

    # --------------------------------------------------------
    # 2. Validate keys and daily observations
    # --------------------------------------------------------

    keys = [
        "portfolio_id",
        "benchmark_id",
        "date",
        "sector",
    ]

    daily_keys = [
        "portfolio_id",
        "benchmark_id",
        "date",
    ]

    sector["date"] = pd.to_datetime(
        sector["date"],
        errors="raise",
    )

    if sector[keys].isna().any().any():
        raise ValueError(
            "Brinson sector input contains null keys."
        )

    if sector.duplicated(keys).any():
        raise ValueError(
            "Duplicate Brinson sector keys detected."
        )

    if not sector["date"].eq(
        sector["date"].dt.normalize()
    ).all():
        raise ValueError(
            "Brinson input contains non-normalized dates."
        )

    if sector["portfolio_id"].eq(
        sector["benchmark_id"]
    ).any():
        raise ValueError(
            "Portfolio and benchmark identifiers must differ."
        )

    if sector["sector"].astype("string").str.strip().eq(
        ""
    ).fillna(False).any():
        raise ValueError(
            "Blank sector classification detected."
        )

    # Daily portfolio/benchmark returns must be identical
    # across all sectors belonging to the same date and pair.

    daily_return_columns = [
        "portfolio_return_total",
        "benchmark_return_total",
        "arithmetic_active_return",
    ]

    inconsistent_daily_returns = (
        sector.groupby(daily_keys)[daily_return_columns]
        .nunique(dropna=False)
        .gt(1)
        .any()
        .any()
    )

    if inconsistent_daily_returns:
        raise ValueError(
            "Inconsistent portfolio or benchmark returns "
            "across sectors on the same date."
        )

    # --------------------------------------------------------
    # 3. Validate numerical inputs
    # --------------------------------------------------------

    numeric_columns = [
        "portfolio_sector_weight_bop",
        "benchmark_sector_weight_bop",
        "portfolio_sector_contribution_total",
        "benchmark_sector_contribution_total",
        "portfolio_sector_return_for_brinson",
        "benchmark_sector_return_for_brinson",
        "portfolio_return_total",
        "benchmark_return_total",
        "arithmetic_active_return",
    ]

    for column in numeric_columns:
        if (
            not pd.api.types.is_numeric_dtype(sector[column])
            or not np.isfinite(
                sector[column].to_numpy(dtype=float)
            ).all()
        ):
            raise ValueError(
                f"Invalid Brinson numeric input: {column}"
            )

    weight_p = sector["portfolio_sector_weight_bop"]
    weight_b = sector["benchmark_sector_weight_bop"]

    if weight_p.lt(0).any() or weight_b.lt(0).any():
        raise ValueError(
            "Negative sector weights are unsupported."
        )

    portfolio_positive = weight_p.gt(0)
    benchmark_positive = weight_b.gt(0)

    # --------------------------------------------------------
    # 4. Verify zero-weight sector classification
    # --------------------------------------------------------

    expected_presence = np.select(
        [
            portfolio_positive & benchmark_positive,
            ~portfolio_positive & benchmark_positive,
            portfolio_positive & ~benchmark_positive,
        ],
        [
            "BOTH_INVESTED",
            "BENCHMARK_ONLY",
            "PORTFOLIO_ONLY",
        ],
        default="BOTH_ZERO",
    )

    if not np.array_equal(
        sector["sector_presence_status"].to_numpy(),
        expected_presence,
    ):
        raise ValueError(
            "Sector presence classifications disagree "
            "with observed BOP weights."
        )

    observed_p = sector["portfolio_sector_return_observed"]
    observed_b = sector["benchmark_sector_return_observed"]

    for label, observed, positive in [
        ("portfolio", observed_p, portfolio_positive),
        ("benchmark", observed_b, benchmark_positive),
    ]:
        if not np.isfinite(
            observed.loc[positive].to_numpy(dtype=float)
        ).all():
            raise ValueError(
                f"Invested {label} sectors require finite "
                "observed sector returns."
            )

        if observed.loc[~positive].notna().any():
            raise ValueError(
                f"Zero-weight {label} sector has an "
                "unexpected observed sector return."
            )

    # --------------------------------------------------------
    # 5. Verify the approved attribution reference policy
    # --------------------------------------------------------

    benchmark_total = sector["benchmark_return_total"]

    expected_portfolio_reference = np.where(
        portfolio_positive,
        observed_p,
        np.where(
            benchmark_positive,
            observed_b,
            benchmark_total,
        ),
    )

    expected_benchmark_reference = np.where(
        benchmark_positive,
        observed_b,
        benchmark_total,
    )

    reference_p = sector[
        "portfolio_sector_return_for_brinson"
    ]

    reference_b = sector[
        "benchmark_sector_return_for_brinson"
    ]

    if not np.allclose(
        reference_p.to_numpy(dtype=float),
        expected_portfolio_reference,
        atol=return_tolerance,
        rtol=0.0,
    ):
        raise ValueError(
            "Portfolio Brinson reference returns do not "
            "follow the approved sector-presence policy."
        )

    if not np.allclose(
        reference_b.to_numpy(dtype=float),
        expected_benchmark_reference,
        atol=return_tolerance,
        rtol=0.0,
    ):
        raise ValueError(
            "Benchmark Brinson reference returns do not "
            "follow the approved sector-presence policy."
        )

    expected_reference_status = np.select(
        [
            ~portfolio_positive & benchmark_positive,
            portfolio_positive & ~benchmark_positive,
            ~portfolio_positive & ~benchmark_positive,
        ],
        [
            "PORTFOLIO_REFERENCE_BENCHMARK_SECTOR",
            "BENCHMARK_REFERENCE_TOTAL_BENCHMARK",
            "BOTH_REFERENCE_TOTAL_BENCHMARK",
        ],
        default="BOTH_OBSERVED",
    )

    if not np.array_equal(
        sector["attribution_reference_status"].to_numpy(),
        expected_reference_status,
    ):
        raise ValueError(
            "Attribution reference status is inconsistent."
        )

    # --------------------------------------------------------
    # 6. Reconcile sector contributions to reference returns
    # --------------------------------------------------------

    portfolio_contribution_error = (
        weight_p * reference_p
        - sector["portfolio_sector_contribution_total"]
    )

    benchmark_contribution_error = (
        weight_b * reference_b
        - sector["benchmark_sector_contribution_total"]
    )

    for name, error in [
        ("portfolio", portfolio_contribution_error),
        ("benchmark", benchmark_contribution_error),
    ]:
        if error.abs().max() > return_tolerance:
            raise ValueError(
                f"{name} sector contributions disagree "
                "with sector weights and observed returns."
            )

    # --------------------------------------------------------
    # 7. Calculate Brinson-Fachler allocation effect
    # --------------------------------------------------------

    sector["active_sector_weight"] = weight_p - weight_b

    sector["allocation_effect"] = (
        sector["active_sector_weight"]
        * (
            reference_b
            - benchmark_total
        )
    )

    # --------------------------------------------------------
    # 8. Calculate selection and interaction effects
    # --------------------------------------------------------

    sector["sector_selection_return_difference"] = (
        reference_p - reference_b
    )

    sector["selection_effect"] = (
        weight_b
        * sector["sector_selection_return_difference"]
    )

    sector["interaction_effect"] = (
        sector["active_sector_weight"]
        * sector["sector_selection_return_difference"]
    )

    sector["total_brinson_effect"] = (
        sector["allocation_effect"]
        + sector["selection_effect"]
        + sector["interaction_effect"]
    )

    # --------------------------------------------------------
    # 9. Independently validate the sector-level bridge
    # --------------------------------------------------------

    sector["sector_active_contribution"] = (
        sector["portfolio_sector_contribution_total"]
        - sector["benchmark_sector_contribution_total"]
    )

    sector["active_contribution_bridge"] = (
        sector["total_brinson_effect"]
        + sector["active_sector_weight"] * benchmark_total
    )

    sector["sector_bridge_error"] = (
        sector["active_contribution_bridge"]
        - sector["sector_active_contribution"]
    )

    if (
        sector["sector_bridge_error"].abs().max()
        > return_tolerance
    ):
        raise ValueError(
            "Brinson effects do not reconcile to "
            "sector active contributions."
        )

    # --------------------------------------------------------
    # 10. Validate specific zero-weight attribution effects
    # --------------------------------------------------------

    benchmark_only = sector[
        "sector_presence_status"
    ].eq("BENCHMARK_ONLY")

    portfolio_only = sector[
        "sector_presence_status"
    ].eq("PORTFOLIO_ONLY")

    if benchmark_only.any():
        omitted_sector_effects = sector.loc[
            benchmark_only,
            ["selection_effect", "interaction_effect"],
        ]

        if omitted_sector_effects.abs().to_numpy().max() > (
            return_tolerance
        ):
            raise ValueError(
                "Benchmark-only sectors must have zero "
                "selection and interaction effects."
            )

    if portfolio_only.any():
        off_benchmark_effects = sector.loc[
            portfolio_only,
            ["allocation_effect", "selection_effect"],
        ]

        if off_benchmark_effects.abs().to_numpy().max() > (
            return_tolerance
        ):
            raise ValueError(
                "Portfolio-only sector reference convention "
                "requires zero allocation and selection effects."
            )

    # --------------------------------------------------------
    # 11. Aggregate daily Brinson effects
    # --------------------------------------------------------

    brinson_total_daily = (
        sector.groupby(
            daily_keys,
            as_index=False,
        )
        .agg(
            allocation_effect=("allocation_effect", "sum"),
            selection_effect=("selection_effect", "sum"),
            interaction_effect=("interaction_effect", "sum"),
            total_brinson_effect=("total_brinson_effect", "sum"),
            sector_active_contribution=(
                "sector_active_contribution",
                "sum",
            ),
            sector_bridge_error=("sector_bridge_error", "sum"),
            portfolio_sector_weight_sum=(
                "portfolio_sector_weight_bop",
                "sum",
            ),
            benchmark_sector_weight_sum=(
                "benchmark_sector_weight_bop",
                "sum",
            ),
            portfolio_sector_contribution_sum=(
                "portfolio_sector_contribution_total",
                "sum",
            ),
            benchmark_sector_contribution_sum=(
                "benchmark_sector_contribution_total",
                "sum",
            ),
            portfolio_return_total=(
                "portfolio_return_total",
                "first",
            ),
            benchmark_return_total=(
                "benchmark_return_total",
                "first",
            ),
            arithmetic_active_return=(
                "arithmetic_active_return",
                "first",
            ),
            sector_count=("sector", "size"),
        )
    )

    # --------------------------------------------------------
    # 12. Calculate daily financial reconciliation residuals
    # --------------------------------------------------------

    daily = brinson_total_daily

    daily["brinson_reconciliation_error"] = (
        daily["total_brinson_effect"]
        - daily["arithmetic_active_return"]
    )

    daily["active_contribution_reconciliation_error"] = (
        daily["sector_active_contribution"]
        - daily["arithmetic_active_return"]
    )

    daily["portfolio_return_reconciliation_error"] = (
        daily["portfolio_sector_contribution_sum"]
        - daily["portfolio_return_total"]
    )

    daily["benchmark_return_reconciliation_error"] = (
        daily["benchmark_sector_contribution_sum"]
        - daily["benchmark_return_total"]
    )

    daily["arithmetic_active_return_error"] = (
        daily["arithmetic_active_return"]
        - (
            daily["portfolio_return_total"]
            - daily["benchmark_return_total"]
        )
    )

    daily["portfolio_weight_error"] = (
        daily["portfolio_sector_weight_sum"] - 1.0
    )

    daily["benchmark_weight_error"] = (
        daily["benchmark_sector_weight_sum"] - 1.0
    )

    # --------------------------------------------------------
    # 13. Enforce financial reconciliation tolerances
    # --------------------------------------------------------

    return_error_columns = [
        "brinson_reconciliation_error",
        "active_contribution_reconciliation_error",
        "portfolio_return_reconciliation_error",
        "benchmark_return_reconciliation_error",
        "arithmetic_active_return_error",
        "sector_bridge_error",
    ]

    weight_error_columns = [
        "portfolio_weight_error",
        "benchmark_weight_error",
    ]

    for column in return_error_columns:
        maximum_error = float(daily[column].abs().max())

        if maximum_error > return_tolerance:
            raise ValueError(
                f"Daily Brinson reconciliation failed: "
                f"{column}; maximum error "
                f"{maximum_error:.12e}"
            )

    for column in weight_error_columns:
        maximum_error = float(daily[column].abs().max())

        if maximum_error > weight_tolerance:
            raise ValueError(
                f"Daily Brinson weight validation failed: "
                f"{column}; maximum error "
                f"{maximum_error:.12e}"
            )

    # --------------------------------------------------------
    # 14. Finalize reproducible output DataFrames
    # --------------------------------------------------------

    sector["attribution_method"] = (
        "brinson_fachler_three_effect"
    )

    daily["attribution_method"] = (
        "brinson_fachler_three_effect"
    )

    brinson_sector_daily = (
        sector.sort_values(keys)
        .reset_index(drop=True)
    )

    brinson_total_daily = (
        daily.sort_values(daily_keys)
        .reset_index(drop=True)
    )

    return (
        brinson_sector_daily,
        brinson_total_daily,
    )
