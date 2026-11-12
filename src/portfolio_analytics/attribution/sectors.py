"""Attribution sectors: validated research calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd



# Research reference: 08_portfolio_attribution.ipynb, code cell index 3.
def aggregate_sector_returns(
    portfolio_holdings: pd.DataFrame,
    portfolio_returns: pd.DataFrame,
    *,
    portfolio_ids: tuple[str, ...],
    return_tolerance: float = 1e-12,
    weight_tolerance: float = 1e-12,
    require_fully_invested: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate daily sector returns and reconcile sector contributions.

    Parameters
    ----------
    portfolio_holdings:
        Standardized daily holdings with portfolio_id, date,
        security_id, sector, weight_bop and return_total.

    portfolio_returns:
        Standardized portfolio returns with portfolio_id,
        date and return_total.

    portfolio_ids:
        Identifiers of portfolios to analyze.

    return_tolerance:
        Maximum permitted daily return reconciliation error.

    weight_tolerance:
        Maximum permitted fully-invested weight deviation.

    require_fully_invested:
        Require total security BOP weights to equal one.
        The current attribution contract excludes explicit
        cash-return contributions.

    Returns
    -------
    sector_returns_daily:
        Grain: portfolio_id, date, sector.
        Contains sector BOP weights, contributions, returns,
        security counts and return-availability status.

    sector_return_reconciliation:
        Grain: portfolio_id, date.
        Contains sector contribution sums, source portfolio
        returns, sector weight sums and reconciliation errors.

    Notes
    -----
    - Uses total returns and BOP weights.
    - No artificial sector rows are generated.
    - Zero-weight sector returns remain NaN.
    - Missing portfolio positions are handled in the
      subsequent portfolio/benchmark sector-alignment stage.
    - Numerical calculations retain full precision.

    Raises
    ------
    ValueError:
        On invalid inputs, mismatched dates, invalid weights,
        missing sectors or failed financial reconciliation.
    """

    # --------------------------------------------------------
    # 1. Validate settings and required columns
    # --------------------------------------------------------

    if (
        isinstance(portfolio_ids, str)
        or not portfolio_ids
        or len(set(portfolio_ids)) != len(portfolio_ids)
    ):
        raise ValueError(
            "portfolio_ids must contain distinct identifiers."
        )

    for name, tolerance in [
        ("return_tolerance", return_tolerance),
        ("weight_tolerance", weight_tolerance),
    ]:
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError(
                f"{name} must be finite and nonnegative."
            )

    required_holdings_columns = {
        "portfolio_id",
        "date",
        "security_id",
        "sector",
        "weight_bop",
        "return_total",
    }

    required_return_columns = {
        "portfolio_id",
        "date",
        "return_total",
    }

    missing_holdings = (
        required_holdings_columns
        - set(portfolio_holdings.columns)
    )

    missing_returns = (
        required_return_columns
        - set(portfolio_returns.columns)
    )

    if missing_holdings or missing_returns:
        raise ValueError(
            "Missing required sector-attribution columns. "
            f"Holdings: {sorted(missing_holdings)}; "
            f"Returns: {sorted(missing_returns)}"
        )

    # --------------------------------------------------------
    # 2. Select relevant holdings and portfolio returns
    # --------------------------------------------------------

    holdings = portfolio_holdings.loc[
        portfolio_holdings["portfolio_id"].isin(
            portfolio_ids
        ),
        [
            "portfolio_id",
            "date",
            "security_id",
            "sector",
            "weight_bop",
            "return_total",
        ],
    ].copy()

    returns = portfolio_returns.loc[
        portfolio_returns["portfolio_id"].isin(
            portfolio_ids
        ),
        [
            "portfolio_id",
            "date",
            "return_total",
        ],
    ].copy()

    if holdings.empty or returns.empty:
        raise ValueError(
            "Selected portfolio holdings or returns are empty."
        )

    for portfolio_id in portfolio_ids:
        if (
            portfolio_id not in set(holdings["portfolio_id"])
            or portfolio_id not in set(returns["portfolio_id"])
        ):
            raise ValueError(
                f"Missing holdings or returns for {portfolio_id}"
            )

    # --------------------------------------------------------
    # 3. Validate attribution-specific observations
    # --------------------------------------------------------

    holdings["date"] = pd.to_datetime(
        holdings["date"],
        errors="raise",
    )

    returns["date"] = pd.to_datetime(
        returns["date"],
        errors="raise",
    )

    for name, frame, keys in [
        (
            "holdings",
            holdings,
            ["portfolio_id", "date", "security_id"],
        ),
        (
            "returns",
            returns,
            ["portfolio_id", "date"],
        ),
    ]:
        if frame[keys].isna().any().any():
            raise ValueError(
                f"{name} contains null primary keys."
            )

        if frame.duplicated(keys).any():
            raise ValueError(
                f"{name} contains duplicate primary keys."
            )

        if not frame["date"].eq(
            frame["date"].dt.normalize()
        ).all():
            raise ValueError(
                f"{name} contains non-normalized dates."
            )

    invalid_sectors = (
        holdings["sector"].isna()
        | holdings["sector"]
            .astype("string")
            .str.strip()
            .eq("")
            .fillna(False)
    )

    if invalid_sectors.any():
        raise ValueError(
            "Holdings contain missing or blank sectors."
        )

    for name, frame, numeric_columns in [
        (
            "holdings",
            holdings,
            ["weight_bop", "return_total"],
        ),
        (
            "returns",
            returns,
            ["return_total"],
        ),
    ]:
        for column in numeric_columns:
            if (
                not pd.api.types.is_numeric_dtype(
                    frame[column]
                )
                or not np.isfinite(
                    frame[column].to_numpy(dtype=float)
                ).all()
            ):
                raise ValueError(
                    f"{name}.{column} contains invalid "
                    "numeric observations."
                )

    if holdings["weight_bop"].lt(0).any():
        raise ValueError(
            "Negative weights are unsupported for "
            "the current long-only methodology."
        )

    # --------------------------------------------------------
    # 4. Independently calculate security contributions
    # --------------------------------------------------------

    holdings["calculated_contribution_total"] = (
        holdings["weight_bop"]
        * holdings["return_total"]
    )

    if not np.isfinite(
        holdings["calculated_contribution_total"]
        .to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Calculated security contributions are non-finite."
        )

    holdings["positive_weight_security"] = (
        holdings["weight_bop"] > 0
    )

    # --------------------------------------------------------
    # 5. Aggregate security observations into daily sectors
    # --------------------------------------------------------

    sector_returns_daily = (
        holdings.groupby(
            ["portfolio_id", "date", "sector"],
            as_index=False,
            observed=True,
        )
        .agg(
            sector_weight_bop=("weight_bop", "sum"),
            sector_contribution_total=(
                "calculated_contribution_total",
                "sum",
            ),
            security_count=("security_id", "size"),
            positive_weight_security_count=(
                "positive_weight_security",
                "sum",
            ),
        )
    )

    # --------------------------------------------------------
    # 6. Calculate sector total returns
    # --------------------------------------------------------

    positive_sector_weight = (
        sector_returns_daily["sector_weight_bop"] > 0
    )

    sector_returns_daily["sector_return_total"] = np.nan

    sector_returns_daily.loc[
        positive_sector_weight,
        "sector_return_total",
    ] = (
        sector_returns_daily.loc[
            positive_sector_weight,
            "sector_contribution_total",
        ].to_numpy()
        /
        sector_returns_daily.loc[
            positive_sector_weight,
            "sector_weight_bop",
        ].to_numpy()
    )

    sector_returns_daily["sector_return_status"] = (
        np.where(
            positive_sector_weight,
            "AVAILABLE",
            "UNAVAILABLE_ZERO_WEIGHT",
        )
    )

    # Zero-weight sectors must have zero contributions.
    zero_weight_nonzero_contribution = (
        ~positive_sector_weight
        & sector_returns_daily[
            "sector_contribution_total"
        ].abs().gt(return_tolerance)
    )

    if zero_weight_nonzero_contribution.any():
        raise ValueError(
            "A zero-weight sector has a nonzero "
            "calculated return contribution."
        )

    # --------------------------------------------------------
    # 7. Aggregate sector contributions to portfolio level
    # --------------------------------------------------------

    sector_return_reconciliation = (
        sector_returns_daily.groupby(
            ["portfolio_id", "date"],
            as_index=False,
        )
        .agg(
            calculated_portfolio_return=(
                "sector_contribution_total",
                "sum",
            ),
            total_sector_weight_bop=(
                "sector_weight_bop",
                "sum",
            ),
            sector_count=("sector", "size"),
            available_sector_count=(
                "sector_return_status",
                lambda values: values.eq("AVAILABLE").sum(),
            ),
        )
    )

    # --------------------------------------------------------
    # 8. Align independently calculated and saved returns
    # --------------------------------------------------------

    expected_returns = returns.rename(
        columns={
            "return_total": "portfolio_return_total"
        }
    )

    sector_return_reconciliation = (
        expected_returns.merge(
            sector_return_reconciliation,
            on=["portfolio_id", "date"],
            how="outer",
            validate="one_to_one",
            indicator=True,
        )
    )

    unmatched = ~sector_return_reconciliation[
        "_merge"
    ].eq("both")

    if unmatched.any():
        raise ValueError(
            "Sector aggregation and portfolio returns "
            "have unmatched portfolio-date observations. "
            f"Unmatched rows: {int(unmatched.sum())}"
        )

    sector_return_reconciliation = (
        sector_return_reconciliation.drop(
            columns="_merge"
        )
    )

    # --------------------------------------------------------
    # 9. Calculate financial reconciliation residuals
    # --------------------------------------------------------

    sector_return_reconciliation[
        "return_reconciliation_error"
    ] = (
        sector_return_reconciliation[
            "calculated_portfolio_return"
        ]
        - sector_return_reconciliation[
            "portfolio_return_total"
        ]
    )

    sector_return_reconciliation[
        "weight_normalization_error"
    ] = (
        sector_return_reconciliation[
            "total_sector_weight_bop"
        ]
        - 1.0
    )

    max_return_error = float(
        sector_return_reconciliation[
            "return_reconciliation_error"
        ].abs().max()
    )

    max_weight_error = float(
        sector_return_reconciliation[
            "weight_normalization_error"
        ].abs().max()
    )

    if max_return_error > return_tolerance:
        raise ValueError(
            "Sector contributions do not reconcile to "
            "portfolio returns. "
            f"Maximum error: {max_return_error:.12e}"
        )

    if (
        require_fully_invested
        and max_weight_error > weight_tolerance
    ):
        raise ValueError(
            "Sector BOP weights do not sum to one. "
            f"Maximum error: {max_weight_error:.12e}"
        )

    # --------------------------------------------------------
    # 10. Return deterministic standardized DataFrames
    # --------------------------------------------------------

    sector_returns_daily = (
        sector_returns_daily.sort_values(
            ["portfolio_id", "date", "sector"]
        )
        .reset_index(drop=True)
    )

    sector_return_reconciliation = (
        sector_return_reconciliation.sort_values(
            ["portfolio_id", "date"]
        )
        .reset_index(drop=True)
    )

    return (
        sector_returns_daily,
        sector_return_reconciliation,
    )


# Research reference: 08_portfolio_attribution.ipynb, code cell index 4.
def build_brinson_sector_comparison(
    sector_returns: pd.DataFrame,
    aligned_returns: pd.DataFrame,
    *,
    portfolio_id: str,
    benchmark_id: str,
    weight_tolerance: float = 1e-12,
    return_tolerance: float = 1e-12,
) -> pd.DataFrame:
    """
    Prepare a daily sector comparison for Brinson-Fachler attribution.

    Parameters
    ----------
    sector_returns:
        Daily sector weights, contributions and observed returns.
        Grain: portfolio_id, date, sector.

    aligned_returns:
        Daily portfolio/benchmark total returns.
        Grain: portfolio_id, benchmark_id, date.

    portfolio_id:
        Active portfolio identifier.

    benchmark_id:
        Benchmark identifier.

    weight_tolerance:
        Absolute tolerance for fully invested sector weights.

    return_tolerance:
        Absolute tolerance for sector contribution reconciliation.

    Returns
    -------
    pd.DataFrame
        Daily aligned sector comparison.

        Grain:
            portfolio_id, benchmark_id, date, sector.

        Observed sector returns remain distinct from
        attribution-specific reference returns.

    Methodological conventions
    --------------------------
    BOTH_INVESTED:
        Use observed portfolio and benchmark sector returns.

    BENCHMARK_ONLY:
        Portfolio sector return is unavailable.
        Use benchmark sector return as the attribution reference
        for the absent portfolio sector.

    PORTFOLIO_ONLY:
        Benchmark sector return is unavailable.
        Use the daily total benchmark return as its attribution
        reference.

    BOTH_ZERO:
        Neither sector has invested weight.
        Use total benchmark return for both attribution references.
        Both contributions remain zero.

    These are decomposition conventions, not imputations
    of unavailable realized sector returns.
    """

    # --------------------------------------------------------
    # 1. Validate parameters
    # --------------------------------------------------------

    if portfolio_id == benchmark_id:
        raise ValueError(
            "Portfolio and benchmark identifiers must differ."
        )

    for name, tolerance in [
        ("weight_tolerance", weight_tolerance),
        ("return_tolerance", return_tolerance),
    ]:
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError(
                f"{name} must be finite and nonnegative."
            )

    required_sector_columns = {
        "portfolio_id",
        "date",
        "sector",
        "sector_weight_bop",
        "sector_contribution_total",
        "sector_return_total",
        "sector_return_status",
    }

    required_return_columns = {
        "portfolio_id",
        "benchmark_id",
        "date",
        "portfolio_return_total",
        "benchmark_return_total",
        "arithmetic_active_return",
    }

    missing_sector = (
        required_sector_columns - set(sector_returns.columns)
    )

    missing_returns = (
        required_return_columns - set(aligned_returns.columns)
    )

    if missing_sector or missing_returns:
        raise ValueError(
            "Missing required sector-alignment columns. "
            f"Sector: {sorted(missing_sector)}; "
            f"Returns: {sorted(missing_returns)}"
        )

    # --------------------------------------------------------
    # 2. Select and validate portfolio/benchmark observations
    # --------------------------------------------------------

    selected_ids = [portfolio_id, benchmark_id]

    sectors = sector_returns.loc[
        sector_returns["portfolio_id"].isin(selected_ids),
        sorted(required_sector_columns),
    ].copy()

    returns = aligned_returns[
        sorted(required_return_columns)
    ].copy()

    returns = returns.loc[
        returns["portfolio_id"].eq(portfolio_id)
        & returns["benchmark_id"].eq(benchmark_id)
    ].copy()

    if sectors.empty or returns.empty:
        raise ValueError(
            "Selected sector observations or returns are empty."
        )

    sectors["date"] = pd.to_datetime(
        sectors["date"],
        errors="raise",
    )

    returns["date"] = pd.to_datetime(
        returns["date"],
        errors="raise",
    )

    sector_keys = [
        "portfolio_id",
        "date",
        "sector",
    ]

    if sectors[sector_keys].isna().any().any():
        raise ValueError(
            "Sector observations contain null primary keys."
        )

    if sectors.duplicated(sector_keys).any():
        raise ValueError(
            "Duplicate sector observations detected."
        )

    if returns["date"].duplicated().any():
        raise ValueError(
            "Duplicate aligned daily return dates detected."
        )

    for name, frame in [
        ("sectors", sectors),
        ("returns", returns),
    ]:
        if frame["date"].isna().any():
            raise ValueError(f"{name} contains null dates.")

        if not frame["date"].eq(
            frame["date"].dt.normalize()
        ).all():
            raise ValueError(
                f"{name} contains non-normalized dates."
            )

    invalid_sectors = (
        sectors["sector"]
        .astype("string")
        .str.strip()
        .eq("")
        .fillna(False)
    )

    if invalid_sectors.any():
        raise ValueError(
            "Blank sector classifications detected."
        )

    # --------------------------------------------------------
    # 3. Validate observed sector weights, returns and status
    # --------------------------------------------------------

    numeric_sector_columns = [
        "sector_weight_bop",
        "sector_contribution_total",
    ]

    for column in numeric_sector_columns:
        if (
            not pd.api.types.is_numeric_dtype(sectors[column])
            or not np.isfinite(
                sectors[column].to_numpy(dtype=float)
            ).all()
        ):
            raise ValueError(
                f"Invalid sector numeric field: {column}"
            )

    for column in [
        "portfolio_return_total",
        "benchmark_return_total",
        "arithmetic_active_return",
    ]:
        if (
            not pd.api.types.is_numeric_dtype(returns[column])
            or not np.isfinite(
                returns[column].to_numpy(dtype=float)
            ).all()
        ):
            raise ValueError(
                f"Invalid aligned return field: {column}"
            )

    if sectors["sector_weight_bop"].lt(0).any():
        raise ValueError(
            "Negative sector weights are unsupported."
        )

    positive_weight = sectors["sector_weight_bop"].gt(0)

    available_return = (
        sectors["sector_return_total"].notna()
        & np.isfinite(
            pd.to_numeric(
                sectors["sector_return_total"],
                errors="coerce",
            ).to_numpy(dtype=float)
        )
    )

    invalid_positive = (
        positive_weight
        & (
            ~available_return
            | sectors["sector_return_status"].ne("AVAILABLE")
        )
    )

    invalid_zero = (
        ~positive_weight
        & (
            sectors["sector_return_total"].notna()
            | sectors["sector_return_status"].ne(
                "UNAVAILABLE_ZERO_WEIGHT"
            )
            | sectors["sector_contribution_total"].abs().gt(
                return_tolerance
            )
        )
    )

    if invalid_positive.any() or invalid_zero.any():
        raise ValueError(
            "Sector return availability, weights, contributions "
            "or status values are inconsistent."
        )

    # --------------------------------------------------------
    # 4. Verify both portfolio sector calendars
    # --------------------------------------------------------

    expected_dates = set(returns["date"])

    for selected_id in selected_ids:
        observed_dates = set(
            sectors.loc[
                sectors["portfolio_id"].eq(selected_id),
                "date",
            ]
        )

        if observed_dates != expected_dates:
            raise ValueError(
                f"Sector calendar mismatch for {selected_id}. "
                f"Missing dates: "
                f"{len(expected_dates - observed_dates)}; "
                f"unexpected dates: "
                f"{len(observed_dates - expected_dates)}."
            )

    # --------------------------------------------------------
    # 5. Prepare separately named portfolio/benchmark columns
    # --------------------------------------------------------

    comparison_fields = [
        "date",
        "sector",
        "sector_weight_bop",
        "sector_contribution_total",
        "sector_return_total",
    ]

    portfolio_sectors = (
        sectors.loc[
            sectors["portfolio_id"].eq(portfolio_id),
            comparison_fields,
        ]
        .rename(
            columns={
                "sector_weight_bop":
                    "portfolio_sector_weight_bop",
                "sector_contribution_total":
                    "portfolio_sector_contribution_total",
                "sector_return_total":
                    "portfolio_sector_return_observed",
            }
        )
    )

    benchmark_sectors = (
        sectors.loc[
            sectors["portfolio_id"].eq(benchmark_id),
            comparison_fields,
        ]
        .rename(
            columns={
                "sector_weight_bop":
                    "benchmark_sector_weight_bop",
                "sector_contribution_total":
                    "benchmark_sector_contribution_total",
                "sector_return_total":
                    "benchmark_sector_return_observed",
            }
        )
    )

    # --------------------------------------------------------
    # 6. Full outer join to preserve all portfolio/benchmark
    #    sectors, including structurally absent positions
    # --------------------------------------------------------

    comparison = portfolio_sectors.merge(
        benchmark_sectors,
        on=["date", "sector"],
        how="outer",
        validate="one_to_one",
        indicator="sector_join_status",
    )

    comparison = comparison.merge(
        returns[
            [
                "date",
                "portfolio_return_total",
                "benchmark_return_total",
                "arithmetic_active_return",
            ]
        ],
        on="date",
        how="left",
        validate="many_to_one",
    )

    if comparison[
        [
            "portfolio_return_total",
            "benchmark_return_total",
            "arithmetic_active_return",
        ]
    ].isna().any().any():
        raise ValueError(
            "Sector observations have unmatched daily returns."
        )

    # --------------------------------------------------------
    # 7. Assign zero weights/contributions only to absent
    #    positions, never to unavailable observed returns
    # --------------------------------------------------------

    absent_portfolio = comparison[
        "sector_join_status"
    ].eq("right_only")

    absent_benchmark = comparison[
        "sector_join_status"
    ].eq("left_only")

    comparison.loc[
        absent_portfolio,
        [
            "portfolio_sector_weight_bop",
            "portfolio_sector_contribution_total",
        ],
    ] = 0.0

    comparison.loc[
        absent_benchmark,
        [
            "benchmark_sector_weight_bop",
            "benchmark_sector_contribution_total",
        ],
    ] = 0.0

    # Original observed sector returns remain NaN
    # when the corresponding position does not exist.

    portfolio_positive = (
        comparison["portfolio_sector_weight_bop"] > 0
    )

    benchmark_positive = (
        comparison["benchmark_sector_weight_bop"] > 0
    )

    # --------------------------------------------------------
    # 8. Classify daily sector investment status
    # --------------------------------------------------------

    comparison["sector_presence_status"] = np.select(
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

    # --------------------------------------------------------
    # 9. Construct explicit Brinson reference-return columns
    # --------------------------------------------------------

    comparison[
        "portfolio_sector_return_for_brinson"
    ] = comparison[
        "portfolio_sector_return_observed"
    ].copy()

    comparison[
        "benchmark_sector_return_for_brinson"
    ] = comparison[
        "benchmark_sector_return_observed"
    ].copy()

    benchmark_only = comparison[
        "sector_presence_status"
    ].eq("BENCHMARK_ONLY")

    portfolio_only = comparison[
        "sector_presence_status"
    ].eq("PORTFOLIO_ONLY")

    both_zero = comparison[
        "sector_presence_status"
    ].eq("BOTH_ZERO")

    # Benchmark-only sectors:
    # Absent portfolio sector receives benchmark sector
    # return as an attribution reference, not an observed return.

    comparison.loc[
        benchmark_only,
        "portfolio_sector_return_for_brinson",
    ] = comparison.loc[
        benchmark_only,
        "benchmark_sector_return_observed",
    ]

    # Portfolio-only sectors:
    # Absent benchmark sector receives the total benchmark
    # return as its neutral attribution reference.

    comparison.loc[
        portfolio_only,
        "benchmark_sector_return_for_brinson",
    ] = comparison.loc[
        portfolio_only,
        "benchmark_return_total",
    ]

    # Both-zero sectors:
    # Both reference returns equal the total benchmark return.
    # No invested exposure or contribution is manufactured.

    comparison.loc[
        both_zero,
        "portfolio_sector_return_for_brinson",
    ] = comparison.loc[
        both_zero,
        "benchmark_return_total",
    ]

    comparison.loc[
        both_zero,
        "benchmark_sector_return_for_brinson",
    ] = comparison.loc[
        both_zero,
        "benchmark_return_total",
    ]

    comparison["attribution_reference_status"] = np.select(
        [
            benchmark_only,
            portfolio_only,
            both_zero,
        ],
        [
            "PORTFOLIO_REFERENCE_BENCHMARK_SECTOR",
            "BENCHMARK_REFERENCE_TOTAL_BENCHMARK",
            "BOTH_REFERENCE_TOTAL_BENCHMARK",
        ],
        default="BOTH_OBSERVED",
    )

    # --------------------------------------------------------
    # 10. Validate all attribution reference returns
    # --------------------------------------------------------

    reference_columns = [
        "portfolio_sector_return_for_brinson",
        "benchmark_sector_return_for_brinson",
    ]

    for column in reference_columns:
        if not np.isfinite(
            comparison[column].to_numpy(dtype=float)
        ).all():
            raise ValueError(
                f"Non-finite Brinson attribution references "
                f"detected in {column}."
            )

    # Confirm actual observed returns remain unavailable
    # for structurally absent positions.

    if comparison.loc[
        absent_portfolio,
        "portfolio_sector_return_observed",
    ].notna().any():
        raise ValueError(
            "Absent portfolio sectors have observed returns."
        )

    if comparison.loc[
        absent_benchmark,
        "benchmark_sector_return_observed",
    ].notna().any():
        raise ValueError(
            "Absent benchmark sectors have observed returns."
        )

    # --------------------------------------------------------
    # 11. Independently reconcile aligned sector aggregates
    # --------------------------------------------------------

    daily_sums = (
        comparison.groupby("date", as_index=False)
        .agg(
            portfolio_weight_sum=(
                "portfolio_sector_weight_bop",
                "sum",
            ),
            benchmark_weight_sum=(
                "benchmark_sector_weight_bop",
                "sum",
            ),
            portfolio_contribution_sum=(
                "portfolio_sector_contribution_total",
                "sum",
            ),
            benchmark_contribution_sum=(
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
        )
    )

    validation_errors = {
        "portfolio_weights": (
            daily_sums["portfolio_weight_sum"] - 1.0
        ),
        "benchmark_weights": (
            daily_sums["benchmark_weight_sum"] - 1.0
        ),
        "portfolio_contributions": (
            daily_sums["portfolio_contribution_sum"]
            - daily_sums["portfolio_return_total"]
        ),
        "benchmark_contributions": (
            daily_sums["benchmark_contribution_sum"]
            - daily_sums["benchmark_return_total"]
        ),
    }

    for name, residual in validation_errors.items():
        maximum_error = float(residual.abs().max())

        tolerance = (
            weight_tolerance
            if "weights" in name
            else return_tolerance
        )

        if maximum_error > tolerance:
            raise ValueError(
                f"Sector alignment reconciliation failed: "
                f"{name}; maximum error "
                f"{maximum_error:.12e}"
            )

    # --------------------------------------------------------
    # 12. Finalize standardized output
    # --------------------------------------------------------

    comparison.insert(
        0, "portfolio_id", portfolio_id
    )

    comparison.insert(
        1, "benchmark_id", benchmark_id
    )

    comparison = comparison.drop(
        columns="sector_join_status"
    )

    comparison = (
        comparison.sort_values(["date", "sector"])
        .reset_index(drop=True)
    )

    output_keys = [
        "portfolio_id",
        "benchmark_id",
        "date",
        "sector",
    ]

    if comparison.duplicated(output_keys).any():
        raise ValueError(
            "Duplicate Brinson sector comparison keys detected."
        )

    return comparison
