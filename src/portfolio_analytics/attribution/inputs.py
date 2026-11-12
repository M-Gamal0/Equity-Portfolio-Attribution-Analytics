"""Attribution inputs: validated research calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd



# Research reference: 08_portfolio_attribution.ipynb, code cell index 1.
def prepare_attribution_inputs(
    portfolio_returns: pd.DataFrame,
    portfolio_holdings: pd.DataFrame,
    *,
    portfolio_id: str,
    benchmark_id: str,
    weight_tolerance: float = 1e-12,
    require_fully_invested: bool = True,
    require_static_sectors: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Validate and align inputs for daily holdings-based attribution.

    Parameters
    ----------
    portfolio_returns:
        Standardized portfolio returns with portfolio_id, date,
        and return_total.

    portfolio_holdings:
        Standardized security holdings with portfolio_id, date,
        security_id, sector, weight_bop, and return_total.

    portfolio_id:
        Active portfolio identifier.

    benchmark_id:
        Benchmark portfolio identifier.

    weight_tolerance:
        Absolute tolerance for fully invested weight sums.

    require_fully_invested:
        Require BOP security weights to sum to one daily.

    require_static_sectors:
        Require each security to retain one sector classification
        throughout the supplied historical sample.

    Returns
    -------
    aligned_returns:
        One row per portfolio/benchmark pair and date.
        Includes arithmetic daily active return.

    aligned_holdings:
        Filtered, validated security-day holdings.
        Original security-level weights and returns preserved.

    sector_coverage:
        Daily sector weight-presence diagnostics for both
        portfolios. Missing positions have zero weight;
        missing sector returns are not imputed.

    Raises
    ------
    ValueError:
        If identifiers, calendars, security classifications,
        numeric inputs, or required weight identities are invalid.
    """

    if portfolio_id == benchmark_id:
        raise ValueError(
            "Portfolio and benchmark IDs must be different."
        )

    if (
        not np.isfinite(weight_tolerance)
        or weight_tolerance < 0
    ):
        raise ValueError(
            "weight_tolerance must be finite and nonnegative."
        )

    selected_ids = [portfolio_id, benchmark_id]

    # --------------------------------------------------------
    # 1. Select required standardized input columns
    # --------------------------------------------------------

    return_columns = [
        "portfolio_id",
        "date",
        "return_total",
    ]

    holding_columns = [
        "portfolio_id",
        "date",
        "security_id",
        "sector",
        "weight_bop",
        "return_total",
    ]

    for name, frame, columns in [
        ("portfolio_returns", portfolio_returns, return_columns),
        ("portfolio_holdings", portfolio_holdings, holding_columns),
    ]:
        missing = set(columns) - set(frame.columns)

        if missing:
            raise ValueError(
                f"{name} missing columns: {sorted(missing)}"
            )

    returns = portfolio_returns.loc[
        portfolio_returns["portfolio_id"].isin(selected_ids),
        return_columns,
    ].copy()

    holdings = portfolio_holdings.loc[
        portfolio_holdings["portfolio_id"].isin(selected_ids),
        holding_columns,
    ].copy()

    # --------------------------------------------------------
    # 2. Validate dates, identifiers, and numerical values
    # --------------------------------------------------------

    for name, frame, keys in [
        (
            "portfolio_returns",
            returns,
            ["portfolio_id", "date"],
        ),
        (
            "portfolio_holdings",
            holdings,
            ["portfolio_id", "date", "security_id"],
        ),
    ]:
        if frame.empty:
            raise ValueError(f"{name} contains no selected rows.")

        if frame[keys].isna().any().any():
            raise ValueError(
                f"{name} contains null primary-key values."
            )

        frame["date"] = pd.to_datetime(
            frame["date"],
            errors="raise",
        )

        if not frame["date"].eq(
            frame["date"].dt.normalize()
        ).all():
            raise ValueError(
                f"{name} contains non-midnight date timestamps."
            )

        if frame.duplicated(keys).any():
            raise ValueError(
                f"{name} contains duplicate attribution keys."
            )

    numeric_fields = [
        (returns, "return_total"),
        (holdings, "weight_bop"),
        (holdings, "return_total"),
    ]

    for frame, column in numeric_fields:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            raise ValueError(
                f"{column} must have a numeric dtype."
            )

        if not np.isfinite(
            frame[column].to_numpy(dtype=float)
        ).all():
            raise ValueError(
                f"{column} contains non-finite values."
            )

    if holdings["weight_bop"].lt(0).any():
        raise ValueError(
            "Negative BOP weights are unsupported for "
            "the current long-only attribution method."
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

    for portfolio_key in selected_ids:
        if portfolio_key not in set(returns["portfolio_id"]):
            raise ValueError(
                f"Missing portfolio returns: {portfolio_key}"
            )

        if portfolio_key not in set(holdings["portfolio_id"]):
            raise ValueError(
                f"Missing portfolio holdings: {portfolio_key}"
            )

    # --------------------------------------------------------
    # 3. Align active and benchmark return calendars
    # --------------------------------------------------------

    return_matrix = returns.pivot(
        index="date",
        columns="portfolio_id",
        values="return_total",
    ).reindex(columns=selected_ids)

    if return_matrix.isna().any().any():
        raise ValueError(
            "Portfolio and benchmark have unmatched daily "
            "return observations."
        )

    aligned_returns = (
        return_matrix
        .rename(
            columns={
                portfolio_id: "portfolio_return_total",
                benchmark_id: "benchmark_return_total",
            }
        )
        .reset_index()
        .sort_values("date")
        .reset_index(drop=True)
    )

    aligned_returns.insert(
        0, "portfolio_id", portfolio_id
    )

    aligned_returns.insert(
        1, "benchmark_id", benchmark_id
    )

    aligned_returns["arithmetic_active_return"] = (
        aligned_returns["portfolio_return_total"]
        - aligned_returns["benchmark_return_total"]
    )

    # --------------------------------------------------------
    # 4. Verify holdings and return-date correspondence
    # --------------------------------------------------------

    for portfolio_key in selected_ids:
        return_dates = set(
            returns.loc[
                returns["portfolio_id"].eq(portfolio_key),
                "date",
            ]
        )

        holding_dates = set(
            holdings.loc[
                holdings["portfolio_id"].eq(portfolio_key),
                "date",
            ]
        )

        if return_dates != holding_dates:
            raise ValueError(
                "Return/holding calendar mismatch for "
                f"{portfolio_key}. "
                f"Missing holdings dates: "
                f"{len(return_dates - holding_dates)}; "
                f"unexpected holdings dates: "
                f"{len(holding_dates - return_dates)}."
            )

    # --------------------------------------------------------
    # 5. Validate fully invested BOP weights
    # --------------------------------------------------------

    daily_weight_sums = (
        holdings.groupby(
            ["portfolio_id", "date"]
        )["weight_bop"]
        .sum()
    )

    daily_weight_deviation = (
        daily_weight_sums - 1.0
    ).abs()

    max_weight_deviation = float(
        daily_weight_deviation.max()
    )

    if (
        require_fully_invested
        and max_weight_deviation > weight_tolerance
    ):
        raise ValueError(
            "Daily BOP weights do not sum to one. "
            f"Maximum deviation: {max_weight_deviation:.3e}"
        )

    # --------------------------------------------------------
    # 6. Validate consistent sector classifications
    # --------------------------------------------------------

    daily_security_sector_counts = (
        holdings.groupby(
            ["date", "security_id"]
        )["sector"]
        .nunique(dropna=False)
    )

    if daily_security_sector_counts.gt(1).any():
        raise ValueError(
            "A security has conflicting portfolio/benchmark "
            "sector classifications on the same date."
        )

    if require_static_sectors:
        historical_sector_counts = (
            holdings.groupby("security_id")["sector"]
            .nunique(dropna=False)
        )

        if historical_sector_counts.gt(1).any():
            raise ValueError(
                "A security changes sectors across the "
                "historical sample, contrary to the selected "
                "fixed-for-history classification policy."
            )

    # --------------------------------------------------------
    # 7. Diagnose zero-weight and off-benchmark sectors
    # --------------------------------------------------------

    daily_sector_weights = (
        holdings.groupby(
            ["portfolio_id", "date", "sector"],
            as_index=False,
        )["weight_bop"]
        .sum()
    )

    sector_matrix = daily_sector_weights.pivot(
        index=["date", "sector"],
        columns="portfolio_id",
        values="weight_bop",
    ).reindex(columns=selected_ids)

    # Missing POSITION weights are zero by definition.
    # No sector RETURN values are filled or inferred.
    sector_matrix = sector_matrix.fillna(0.0)

    sector_coverage = (
        sector_matrix.rename(
            columns={
                portfolio_id: "portfolio_weight_bop",
                benchmark_id: "benchmark_weight_bop",
            }
        )
        .reset_index()
    )

    portfolio_positive = (
        sector_coverage["portfolio_weight_bop"] > 0
    )

    benchmark_positive = (
        sector_coverage["benchmark_weight_bop"] > 0
    )

    sector_coverage["sector_presence_status"] = np.select(
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
    # 8. Return standardized, independently usable results
    # --------------------------------------------------------

    aligned_holdings = (
        holdings.sort_values(
            ["portfolio_id", "date", "security_id"]
        )
        .reset_index(drop=True)
    )

    sector_coverage = (
        sector_coverage.sort_values(["date", "sector"])
        .reset_index(drop=True)
    )

    return (
        aligned_returns,
        aligned_holdings,
        sector_coverage,
    )
