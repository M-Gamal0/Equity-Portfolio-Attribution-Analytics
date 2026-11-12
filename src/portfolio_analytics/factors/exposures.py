"""Factors exposures: validated research calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd



# Research reference: 11_fama_french_factor_analytics.ipynb, code cell index 11.
def aggregate_holdings_based_factor_exposures(
    holdings_df: pd.DataFrame,
    *,
    factor_columns: tuple[str, ...],
    model_id: str,
    weight_tolerance: float = 1e-10,
) -> pd.DataFrame:
    """
    Calculate holdings-weighted portfolio factor exposures.

    Input grain
    -----------
    One row per:
        portfolio_id, date, security_id, model_id.

    Required inputs
    ---------------
    portfolio_id
    date
    security_id
    model_id
    weight_bop
    model_available
    estimation_date
    estimated_alpha_daily
    beta_<factor>
    factor_available_on_return_date
    lagged_beta_status

    Output grain
    ------------
    One row per:
        portfolio_id, date, model_id.

    Factor exposure convention
    --------------------------
    For each factor k:

        exposure_k =
            sum(weight_bop_i * beta_i,k)

    Only securities with valid prior regressions contribute
    to this partial exposure estimate.

    No model is assigned a zero coefficient when unavailable.

    No normalization by modeled portfolio weight occurs.

    If no holdings have valid estimates, factor exposures
    and the estimated intercept contribution remain NaN.

    Calendar convention
    -------------------
    Security coefficient estimation_date must be strictly
    earlier than the portfolio return date.

    Factor return availability is recorded separately from
    estimated exposure availability.
    """

    # --------------------------------------------------------
    # A. Validate input configuration
    # --------------------------------------------------------

    if (
        not factor_columns
        or len(set(factor_columns)) != len(factor_columns)
    ):
        raise ValueError(
            "Invalid factor specification."
        )

    if not isinstance(model_id, str) or not model_id:
        raise ValueError(
            "model_id must be a nonempty string."
        )

    if (
        not np.isfinite(weight_tolerance)
        or weight_tolerance < 0
    ):
        raise ValueError(
            "Invalid weight reconciliation tolerance."
        )

    beta_columns = [
        f"beta_{factor}"
        for factor in factor_columns
    ]

    coefficient_columns = [
        "estimated_alpha_daily",
        *beta_columns,
    ]

    required_columns = {
        "portfolio_id",
        "security_id",
        "date",
        "model_id",
        "weight_bop",
        "model_available",
        "estimation_date",
        "factor_available_on_return_date",
        "lagged_beta_status",
        *coefficient_columns,
    }

    missing_columns = (
        required_columns - set(holdings_df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing holdings exposure columns: "
            f"{sorted(missing_columns)}"
        )

    # --------------------------------------------------------
    # B. Prepare independent working copy
    # --------------------------------------------------------

    working = holdings_df[
        list(required_columns)
    ].copy()

    working["date"] = pd.to_datetime(
        working["date"],
        errors="raise",
    ).astype("datetime64[ns]")

    working["estimation_date"] = pd.to_datetime(
        working["estimation_date"],
        errors="raise",
    ).astype("datetime64[ns]")

    key_columns = [
        "portfolio_id",
        "date",
        "security_id",
        "model_id",
    ]

    if working[
        key_columns
    ].isna().any().any():
        raise ValueError(
            "Holdings exposure keys contain missing values."
        )

    if working.duplicated(
        subset=key_columns
    ).any():
        raise ValueError(
            "Duplicate holdings exposure keys detected."
        )

    if not working["model_id"].eq(
        model_id
    ).all():
        raise ValueError(
            "Holdings exposure model IDs are inconsistent."
        )

    for date_column in (
        "date",
        "estimation_date",
    ):

        nonmissing_dates = (
            working[date_column].dropna()
        )

        if not nonmissing_dates.eq(
            nonmissing_dates.dt.normalize()
        ).all():
            raise ValueError(
                "Holdings exposure dates contain "
                "intraday timestamps."
            )

    # --------------------------------------------------------
    # C. Validate weights and exposure availability
    # --------------------------------------------------------

    if working[
        [
            "weight_bop",
            "model_available",
            "factor_available_on_return_date",
        ]
    ].isna().any().any():
        raise ValueError(
            "Weights or availability flags contain nulls."
        )

    weights = working[
        "weight_bop"
    ].to_numpy(dtype=np.float64)

    modeled = working[
        "model_available"
    ].to_numpy(dtype=bool)

    if not np.isfinite(weights).all():
        raise ValueError(
            "Beginning-of-period weights must be finite."
        )

    coefficient_values = working[
        coefficient_columns
    ].to_numpy(dtype=np.float64)

    if not np.isfinite(
        coefficient_values[modeled]
    ).all():
        raise ValueError(
            "A modeled security has missing "
            "or non-finite factor coefficients."
        )

    if not np.isnan(
        coefficient_values[~modeled]
    ).all():
        raise ValueError(
            "Unmodeled security coefficients must "
            "remain NaN."
        )

    # --------------------------------------------------------
    # D. Enforce strict historical estimation boundaries
    # --------------------------------------------------------

    modeled_rows = working.loc[
        modeled
    ]

    if not modeled_rows[
        "estimation_date"
    ].lt(
        modeled_rows["date"]
    ).all():
        raise ValueError(
            "LOOK-AHEAD VIOLATION: Factor coefficients "
            "were not estimated strictly before "
            "the portfolio return date."
        )

    if working.loc[
        ~modeled,
        "estimation_date",
    ].notna().any():
        raise ValueError(
            "Unmodeled security has a regression "
            "estimation date."
        )

    # --------------------------------------------------------
    # E. Validate factor availability consistency
    # --------------------------------------------------------

    group_keys = [
        "portfolio_id",
        "date",
        "model_id",
    ]

    factor_availability_counts = (
        working
        .groupby(
            group_keys,
            observed=True,
        )[
            "factor_available_on_return_date"
        ]
        .nunique()
    )

    if factor_availability_counts.gt(1).any():
        raise ValueError(
            "Factor source availability differs among "
            "holdings on the same portfolio return date."
        )

    # --------------------------------------------------------
    # F. Calculate modeled and unmodeled weight components
    # --------------------------------------------------------

    gross_weights = np.abs(
        weights
    )

    working[
        "gross_weight_bop"
    ] = gross_weights

    working[
        "modeled_net_weight_bop"
    ] = np.where(
        modeled,
        weights,
        0.0,
    )

    working[
        "unmodeled_net_weight_bop"
    ] = np.where(
        ~modeled,
        weights,
        0.0,
    )

    working[
        "modeled_gross_weight_bop"
    ] = np.where(
        modeled,
        gross_weights,
        0.0,
    )

    working[
        "unmodeled_gross_weight_bop"
    ] = np.where(
        ~modeled,
        gross_weights,
        0.0,
    )

    working[
        "stale_modeled_gross_weight_bop"
    ] = np.where(
        working[
            "lagged_beta_status"
        ].eq("MODELED_STALE"),
        gross_weights,
        0.0,
    )

    # --------------------------------------------------------
    # G. Calculate weighted factor contributions
    # --------------------------------------------------------

    weighted_column_map = {
        "estimated_alpha_daily": (
            "estimated_intercept_contribution_daily"
        ),
        **{
            f"beta_{factor}": f"exposure_{factor}"
            for factor in factor_columns
        },
    }

    for source_column, output_column in (
        weighted_column_map.items()
    ):

        # Unavailable exposures are NaN, not zero.
        # The group-level sum uses min_count=1 to
        # preserve NaN when no securities are modeled.

        working[output_column] = (
            working[source_column]
            * working["weight_bop"]
        ).where(
            working["model_available"]
        )

    # --------------------------------------------------------
    # H. Aggregate portfolio-date weight and coverage
    # --------------------------------------------------------

    grouped = working.groupby(
        group_keys,
        observed=True,
        sort=True,
    )

    coverage_df = (
        grouped
        .agg(
            holding_count=(
                "security_id",
                "size",
            ),
            modeled_holding_count=(
                "model_available",
                "sum",
            ),
            total_net_weight_bop=(
                "weight_bop",
                "sum",
            ),
            total_gross_weight_bop=(
                "gross_weight_bop",
                "sum",
            ),
            modeled_net_weight_bop=(
                "modeled_net_weight_bop",
                "sum",
            ),
            unmodeled_net_weight_bop=(
                "unmodeled_net_weight_bop",
                "sum",
            ),
            modeled_gross_weight_bop=(
                "modeled_gross_weight_bop",
                "sum",
            ),
            unmodeled_gross_weight_bop=(
                "unmodeled_gross_weight_bop",
                "sum",
            ),
            stale_modeled_gross_weight_bop=(
                "stale_modeled_gross_weight_bop",
                "sum",
            ),
            factor_available_on_return_date=(
                "factor_available_on_return_date",
                "first",
            ),
        )
        .reset_index()
    )

    weighted_exposure_df = (
        grouped[
            list(weighted_column_map.values())
        ]
        .sum(min_count=1)
        .reset_index()
    )

    result = coverage_df.merge(
        weighted_exposure_df,
        on=group_keys,
        how="inner",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # I. Calculate coverage ratios and statuses
    # --------------------------------------------------------

    result[
        "unmodeled_holding_count"
    ] = (
        result["holding_count"]
        - result["modeled_holding_count"]
    )

    if result[
        "total_gross_weight_bop"
    ].le(0).any():
        raise ValueError(
            "Portfolio-date gross weight must be positive."
        )

    result[
        "modeled_gross_weight_coverage"
    ] = (
        result["modeled_gross_weight_bop"]
        / result["total_gross_weight_bop"]
    )

    result[
        "exposure_coverage_status"
    ] = np.select(
        [
            result[
                "modeled_gross_weight_bop"
            ].le(weight_tolerance),
            result[
                "unmodeled_gross_weight_bop"
            ].le(weight_tolerance),
        ],
        [
            "NO_MODELED_HOLDINGS",
            "FULLY_MODELED",
        ],
        default="PARTIALLY_MODELED",
    )

    result[
        "factor_attribution_status"
    ] = np.select(
        [
            ~result[
                "factor_available_on_return_date"
            ],
            result[
                "exposure_coverage_status"
            ].eq("NO_MODELED_HOLDINGS"),
            result[
                "exposure_coverage_status"
            ].eq("FULLY_MODELED"),
        ],
        [
            "NO_CURRENT_FACTOR_RETURN",
            "NO_MODELED_EXPOSURE",
            "FULL_COVERAGE_FACTORS_AVAILABLE",
        ],
        default="PARTIAL_COVERAGE_FACTORS_AVAILABLE",
    )

    # --------------------------------------------------------
    # J. Validate weight reconciliation
    # --------------------------------------------------------

    np.testing.assert_allclose(
        (
            result["modeled_net_weight_bop"]
            + result["unmodeled_net_weight_bop"]
        ),
        result["total_net_weight_bop"],
        rtol=1e-10,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        (
            result["modeled_gross_weight_bop"]
            + result["unmodeled_gross_weight_bop"]
        ),
        result["total_gross_weight_bop"],
        rtol=1e-10,
        atol=1e-12,
    )

    if not result[
        "holding_count"
    ].eq(
        result["modeled_holding_count"]
        + result["unmodeled_holding_count"]
    ).all():
        raise ValueError(
            "Modeled/unmodeled holdings counts "
            "do not reconcile."
        )

    # --------------------------------------------------------
    # K. Validate incomplete exposure treatment
    # --------------------------------------------------------

    output_exposure_columns = list(
        weighted_column_map.values()
    )

    no_model_mask = result[
        "exposure_coverage_status"
    ].eq("NO_MODELED_HOLDINGS")

    if not result.loc[
        no_model_mask,
        output_exposure_columns,
    ].isna().all().all():
        raise ValueError(
            "Unmodeled portfolio exposures "
            "were incorrectly converted to zero."
        )

    if not np.isfinite(
        result.loc[
            ~no_model_mask,
            output_exposure_columns,
        ].to_numpy(dtype=np.float64)
    ).all():
        raise ValueError(
            "Modeled portfolio exposures contain "
            "non-finite values."
        )

    if result.duplicated(
        subset=group_keys
    ).any():
        raise ValueError(
            "Duplicate portfolio factor exposure keys."
        )

    return result


# Research reference: 11_fama_french_factor_analytics.ipynb, code cell index 11.
def calculate_active_holdings_factor_exposures(
    portfolio_factor_exposures_df: pd.DataFrame,
    *,
    active_portfolio_id: str,
    benchmark_portfolio_id: str,
    factor_columns: tuple[str, ...],
) -> pd.DataFrame:
    """
    Calculate active factor exposures from independently
    aggregated portfolio and benchmark exposures.

    Active exposure:
        portfolio exposure - benchmark exposure.

    Missing portfolio or benchmark exposure remains NaN.

    No missing-model weight is redistributed.

    Calendar and factor-source availability must match.
    """

    if active_portfolio_id == benchmark_portfolio_id:
        raise ValueError(
            "Active portfolio and benchmark must differ."
        )

    exposure_columns = [
        "estimated_intercept_contribution_daily",
        *[
            f"exposure_{factor}"
            for factor in factor_columns
        ],
    ]

    comparison_columns = [
        *exposure_columns,
        "modeled_gross_weight_coverage",
        "modeled_gross_weight_bop",
        "unmodeled_gross_weight_bop",
        "total_net_weight_bop",
        "factor_available_on_return_date",
        "exposure_coverage_status",
    ]

    # --------------------------------------------------------
    # A. Extract independently calculated portfolio exposures
    # --------------------------------------------------------

    portfolio_side = (
        portfolio_factor_exposures_df.loc[
            portfolio_factor_exposures_df[
                "portfolio_id"
            ].eq(active_portfolio_id),
            [
                "date",
                *comparison_columns,
            ],
        ]
        .copy()
        .rename(
            columns={
                column: f"portfolio_{column}"
                for column in comparison_columns
            }
        )
    )

    benchmark_side = (
        portfolio_factor_exposures_df.loc[
            portfolio_factor_exposures_df[
                "portfolio_id"
            ].eq(benchmark_portfolio_id),
            [
                "date",
                *comparison_columns,
            ],
        ]
        .copy()
        .rename(
            columns={
                column: f"benchmark_{column}"
                for column in comparison_columns
            }
        )
    )

    if (
        portfolio_side.empty
        or benchmark_side.empty
    ):
        raise ValueError(
            "Active portfolio or benchmark exposures "
            "are unavailable."
        )

    if (
        portfolio_side["date"].duplicated().any()
        or benchmark_side["date"].duplicated().any()
    ):
        raise ValueError(
            "Duplicate portfolio or benchmark dates."
        )

    # --------------------------------------------------------
    # B. Enforce identical return calendars
    # --------------------------------------------------------

    result = portfolio_side.merge(
        benchmark_side,
        on="date",
        how="outer",
        indicator=True,
        validate="one_to_one",
    )

    if not result[
        "_merge"
    ].eq("both").all():
        raise ValueError(
            "Active portfolio and benchmark "
            "return calendars do not match."
        )

    result = result.drop(
        columns="_merge"
    )

    if not result[
        "portfolio_factor_available_on_return_date"
    ].eq(
        result[
            "benchmark_factor_available_on_return_date"
        ]
    ).all():
        raise ValueError(
            "Portfolio and benchmark factor-source "
            "availability is inconsistent."
        )

    # --------------------------------------------------------
    # C. Calculate active-minus-benchmark exposures
    # --------------------------------------------------------

    for column in exposure_columns:

        if column == (
            "estimated_intercept_contribution_daily"
        ):
            output_column = (
                "active_estimated_intercept_contribution_daily"
            )

        else:
            output_column = (
                f"active_{column}"
            )

        result[output_column] = (
            result[f"portfolio_{column}"]
            - result[f"benchmark_{column}"]
        )

    result[
        "active_net_weight_difference"
    ] = (
        result[
            "portfolio_total_net_weight_bop"
        ]
        - result[
            "benchmark_total_net_weight_bop"
        ]
    )

    # --------------------------------------------------------
    # D. Attach model and portfolio identifiers
    # --------------------------------------------------------

    result["model_id"] = (
        portfolio_factor_exposures_df[
            "model_id"
        ].iloc[0]
    )

    result["portfolio_id"] = (
        active_portfolio_id
    )

    result["benchmark_portfolio_id"] = (
        benchmark_portfolio_id
    )

    result[
        "factor_available_on_return_date"
    ] = result[
        "portfolio_factor_available_on_return_date"
    ]

    # --------------------------------------------------------
    # E. Classify active exposure completeness
    # --------------------------------------------------------

    missing_side = (
        result[
            "portfolio_exposure_coverage_status"
        ].eq("NO_MODELED_HOLDINGS")
        | result[
            "benchmark_exposure_coverage_status"
        ].eq("NO_MODELED_HOLDINGS")
    )

    full_both = (
        result[
            "portfolio_exposure_coverage_status"
        ].eq("FULLY_MODELED")
        & result[
            "benchmark_exposure_coverage_status"
        ].eq("FULLY_MODELED")
    )

    result[
        "active_exposure_coverage_status"
    ] = np.select(
        [
            missing_side,
            full_both,
        ],
        [
            "MISSING_SIDE_EXPOSURE",
            "FULLY_MODELED_BOTH",
        ],
        default="PARTIALLY_MODELED_EITHER",
    )

    result[
        "active_factor_attribution_status"
    ] = np.select(
        [
            ~result[
                "factor_available_on_return_date"
            ],
            missing_side,
            full_both,
        ],
        [
            "NO_CURRENT_FACTOR_RETURN",
            "MISSING_SIDE_EXPOSURE",
            "FULL_COVERAGE_FACTORS_AVAILABLE",
        ],
        default="PARTIAL_COVERAGE_FACTORS_AVAILABLE",
    )

    return (
        result
        .sort_values("date")
        .reset_index(drop=True)
    )
