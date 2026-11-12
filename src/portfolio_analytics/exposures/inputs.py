"""Exposures inputs: validated research calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd



# Research reference: 10_holdings_exposure_analytics.ipynb, code cell index 1.
def validate_exposure_inputs(
    holdings_df: pd.DataFrame,
    *,
    portfolio_ids: tuple[str, ...] | None = None,
    require_fully_invested: bool = False,
    require_aligned_calendars: bool = False,
    require_fixed_sector_history: bool = False,
    weight_tolerance: float = 1e-10,
) -> pd.DataFrame:
    """
    Validate standardized security holdings for exposure analytics.

    Required columns:
        portfolio_id, date, security_id,
        weight_bop, weight_eop, sector

    Grain:
        One row per portfolio_id, date, security_id.

    Units:
        Weights are decimal fractions.

    Conventions:
        weight_bop:
            Weight applied to the daily return interval.

        weight_eop:
            Final closing weight after portfolio events.

        sector:
            Classification supplied by the source contract.

    Policies:
        Fully invested:
            BOP and EOP weights must each sum to one.

        Aligned calendars:
            All selected portfolios share identical dates.

        Fixed sector history:
            Each security_id retains one sector across dates.

    Returns:
        One validation summary row per portfolio.

    Raises:
        TypeError or ValueError for invalid inputs.

    Notes:
        The function does not mutate the supplied DataFrame.
        Full-investment and fixed-sector requirements are
        optional to preserve portability across projects.
    """

    # --------------------------------------------------------
    # A. Validate parameters
    # --------------------------------------------------------

    if not isinstance(holdings_df, pd.DataFrame):
        raise TypeError(
            "holdings_df must be a pandas DataFrame."
        )

    if (
        not np.isfinite(weight_tolerance)
        or weight_tolerance <= 0
        or weight_tolerance >= 1
    ):
        raise ValueError(
            "weight_tolerance must be finite, positive, "
            "and smaller than one."
        )

    required_columns = {
        "portfolio_id",
        "date",
        "security_id",
        "weight_bop",
        "weight_eop",
        "sector",
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
    # B. Select requested portfolios
    # --------------------------------------------------------

    if portfolio_ids is None:

        selected_holdings = holdings_df.copy()

    else:

        portfolio_ids = tuple(portfolio_ids)

        if (
            not portfolio_ids
            or len(portfolio_ids) != len(set(portfolio_ids))
        ):
            raise ValueError(
                "portfolio_ids must contain distinct IDs."
            )

        observed_ids = set(
            holdings_df["portfolio_id"].dropna().unique()
        )

        missing_ids = set(portfolio_ids) - observed_ids

        if missing_ids:
            raise ValueError(
                "Requested portfolio IDs are missing: "
                f"{sorted(missing_ids)}"
            )

        selected_holdings = holdings_df.loc[
            holdings_df["portfolio_id"].isin(portfolio_ids)
        ].copy()

    if selected_holdings.empty:
        raise ValueError(
            "No holdings observations available."
        )

    # --------------------------------------------------------
    # C. Validate identifiers and dates
    # --------------------------------------------------------

    key_columns = [
        "portfolio_id",
        "date",
        "security_id",
    ]

    if selected_holdings[key_columns].isna().any().any():
        raise ValueError(
            "Null values found in holdings primary keys."
        )

    selected_holdings["date"] = pd.to_datetime(
        selected_holdings["date"],
        errors="raise",
    )

    if selected_holdings["date"].isna().any():
        raise ValueError(
            "Invalid or missing holdings dates."
        )

    duplicate_mask = selected_holdings.duplicated(
        subset=key_columns,
        keep=False,
    )

    if duplicate_mask.any():
        raise ValueError(
            "Duplicate holdings primary keys detected. "
            f"Affected rows: {int(duplicate_mask.sum()):,}"
        )

    # --------------------------------------------------------
    # D. Validate numeric weight fields
    # --------------------------------------------------------

    weight_columns = [
        "weight_bop",
        "weight_eop",
    ]

    for column in weight_columns:

        if not pd.api.types.is_numeric_dtype(
            selected_holdings[column]
        ):
            raise TypeError(
                f"{column} must have a numeric dtype."
            )

        values = selected_holdings[column].to_numpy(
            dtype=np.float64
        )

        if not np.isfinite(values).all():
            raise ValueError(
                f"{column} contains nonfinite values."
            )

        if (values < 0).any():
            raise ValueError(
                f"{column} contains negative weights."
            )

        if (values > 1.0 + weight_tolerance).any():
            raise ValueError(
                f"{column} contains weights above 100%."
            )

    # --------------------------------------------------------
    # E. Validate sector completeness
    # --------------------------------------------------------

    if selected_holdings["sector"].isna().any():
        raise ValueError(
            "Missing sector classifications detected."
        )

    blank_sector_mask = (
        selected_holdings["sector"]
        .astype("string")
        .str.strip()
        .eq("")
    )

    if blank_sector_mask.any():
        raise ValueError(
            "Blank sector classifications detected."
        )

    # Same security must not have conflicting sectors
    # across portfolios on the same observation date.

    daily_sector_counts = (
        selected_holdings
        .groupby(["date", "security_id"])["sector"]
        .nunique()
    )

    conflicting_daily_sectors = (
        daily_sector_counts > 1
    )

    if conflicting_daily_sectors.any():
        raise ValueError(
            "Conflicting same-day sector classifications "
            "detected for shared securities."
        )

    # Optional current-sector-fixed-through-history policy.

    if require_fixed_sector_history:

        historical_sector_counts = (
            selected_holdings
            .groupby("security_id")["sector"]
            .nunique()
        )

        inconsistent_securities = (
            historical_sector_counts > 1
        )

        if inconsistent_securities.any():
            raise ValueError(
                "Sector history violates the required "
                "fixed-for-history classification policy. "
                "Affected securities: "
                f"{int(inconsistent_securities.sum())}"
            )

    # --------------------------------------------------------
    # F. Calculate daily weight reconciliation errors
    # --------------------------------------------------------

    daily_weight_sums = (
        selected_holdings
        .groupby(["portfolio_id", "date"])[weight_columns]
        .sum()
    )

    daily_weight_errors = (
        daily_weight_sums.sub(1.0).abs()
    )

    max_daily_errors = (
        daily_weight_errors
        .groupby(level="portfolio_id")
        .max()
        .rename(
            columns={
                "weight_bop": "max_abs_bop_weight_error",
                "weight_eop": "max_abs_eop_weight_error",
            }
        )
    )

    if require_fully_invested:

        failed_weight_dates = (
            daily_weight_errors > weight_tolerance
        ).any(axis=1)

        if failed_weight_dates.any():

            failed_examples = (
                daily_weight_errors.loc[failed_weight_dates]
                .head(5)
            )

            raise ValueError(
                "Daily BOP/EOP weights do not reconcile "
                "to full investment.\n"
                f"{failed_examples}"
            )

    # --------------------------------------------------------
    # G. Validate portfolio calendar alignment
    # --------------------------------------------------------

    portfolio_calendars = {
        portfolio_id: pd.DatetimeIndex(
            group["date"].unique()
        ).sort_values()
        for portfolio_id, group in selected_holdings.groupby(
            "portfolio_id"
        )
    }

    calendar_aligned = all(
        dates.equals(
            next(iter(portfolio_calendars.values()))
        )
        for dates in portfolio_calendars.values()
    )

    if require_aligned_calendars and not calendar_aligned:
        raise ValueError(
            "Selected portfolio holdings calendars "
            "are not aligned."
        )

    # --------------------------------------------------------
    # H. Construct standardized validation summary
    # --------------------------------------------------------

    validation_summary = (
        selected_holdings
        .groupby("portfolio_id")
        .agg(
            holdings_rows=("security_id", "size"),
            trading_dates=("date", "nunique"),
            distinct_securities=("security_id", "nunique"),
            distinct_sectors=("sector", "nunique"),
            first_date=("date", "min"),
            last_date=("date", "max"),
            min_weight_bop=("weight_bop", "min"),
            max_weight_bop=("weight_bop", "max"),
            min_weight_eop=("weight_eop", "min"),
            max_weight_eop=("weight_eop", "max"),
        )
        .join(max_daily_errors)
        .reset_index()
    )

    validation_summary["full_investment_status"] = (
        "PASSED"
        if require_fully_invested
        else "NOT_ASSESSED"
    )

    validation_summary["calendar_alignment_status"] = (
        "PASSED"
        if require_aligned_calendars
        else "NOT_ASSESSED"
    )

    validation_summary["fixed_sector_history_status"] = (
        "PASSED"
        if require_fixed_sector_history
        else "NOT_ASSESSED"
    )

    validation_summary["validation_status"] = "PASSED"

    return validation_summary
