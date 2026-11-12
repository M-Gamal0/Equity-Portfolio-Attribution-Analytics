"""Exposures sectors: validated research calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd



# Research reference: 10_holdings_exposure_analytics.ipynb, code cell index 3.
def calculate_sector_exposures(
    security_exposures: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate standardized security exposures by sector.

    Parameters
    ----------
    security_exposures : pd.DataFrame
        Required columns:
            portfolio_id
            date
            security_id
            sector
            weight_basis
            security_weight
            is_held

    Returns
    -------
    pd.DataFrame
        Grain:
            portfolio_id, date, weight_basis, sector

        Fields:
            sector_weight:
                Sum of constituent security weights.

            ledger_security_count:
                Number of security rows, including zero weights.

            invested_security_count:
                Number of securities with positive weight.

    Financial conventions
    ---------------------
    All weights are decimal portfolio-capital fractions.

    BOP and EOP exposures are aggregated independently.

    Zero-weight sector observations are preserved when
    source security records exist for that sector.

    Sectors completely absent from a portfolio's holdings
    are not artificially introduced.

    The input DataFrame is not modified.
    """

    # --------------------------------------------------------
    # A. Validate input structure
    # --------------------------------------------------------

    if not isinstance(security_exposures, pd.DataFrame):
        raise TypeError(
            "security_exposures must be a pandas DataFrame."
        )

    required_columns = {
        "portfolio_id",
        "date",
        "security_id",
        "sector",
        "weight_basis",
        "security_weight",
        "is_held",
    }

    missing_columns = (
        required_columns - set(security_exposures.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing security exposure columns: "
            f"{sorted(missing_columns)}"
        )

    if security_exposures.empty:
        raise ValueError(
            "Security exposures cannot be empty."
        )

    exposure_keys = [
        "portfolio_id",
        "date",
        "security_id",
        "weight_basis",
    ]

    if security_exposures[exposure_keys].isna().any().any():
        raise ValueError(
            "Security exposure keys contain null values."
        )

    if security_exposures.duplicated(
        subset=exposure_keys
    ).any():
        raise ValueError(
            "Duplicate security exposure keys detected."
        )

    # --------------------------------------------------------
    # B. Validate sector classifications
    # --------------------------------------------------------

    sector_labels = (
        security_exposures["sector"]
        .astype("string")
        .str.strip()
    )

    if sector_labels.isna().any():
        raise ValueError(
            "Missing sector classifications detected."
        )

    if sector_labels.eq("").any():
        raise ValueError(
            "Blank sector classifications detected."
        )

    # --------------------------------------------------------
    # C. Validate weight basis and security weights
    # --------------------------------------------------------

    allowed_bases = {"bop", "eop"}

    observed_bases = set(
        security_exposures["weight_basis"].unique()
    )

    if not observed_bases.issubset(allowed_bases):
        raise ValueError(
            "Unsupported weight_basis values detected."
        )

    if not pd.api.types.is_numeric_dtype(
        security_exposures["security_weight"]
    ):
        raise TypeError(
            "security_weight must have a numeric dtype."
        )

    weights = security_exposures[
        "security_weight"
    ].to_numpy(
        dtype=np.float64,
        na_value=np.nan,
    )

    if not np.isfinite(weights).all():
        raise ValueError(
            "Security exposures contain nonfinite weights."
        )

    if (weights < 0).any() or (weights > 1).any():
        raise ValueError(
            "Security weights must be between zero and one."
        )

    # --------------------------------------------------------
    # D. Validate invested-position flags
    # --------------------------------------------------------

    if not pd.api.types.is_bool_dtype(
        security_exposures["is_held"]
    ):
        raise TypeError(
            "is_held must be Boolean."
        )

    if security_exposures["is_held"].isna().any():
        raise ValueError(
            "is_held contains missing values."
        )

    expected_is_held = (
        security_exposures["security_weight"] > 0
    )

    if not security_exposures["is_held"].eq(
        expected_is_held
    ).all():
        raise ValueError(
            "is_held flags do not match positive weights."
        )

    # --------------------------------------------------------
    # E. Aggregate security exposures by sector
    # --------------------------------------------------------

    sector_group_keys = [
        "portfolio_id",
        "date",
        "weight_basis",
        "sector",
    ]

    sector_exposures = (
        security_exposures
        .groupby(
            sector_group_keys,
            sort=False,
        )
        .agg(
            sector_weight=(
                "security_weight",
                "sum",
            ),
            ledger_security_count=(
                "security_id",
                "size",
            ),
            invested_security_count=(
                "is_held",
                "sum",
            ),
        )
        .reset_index()
        .sort_values(
            sector_group_keys,
            kind="stable",
        )
        .reset_index(drop=True)
    )

    return sector_exposures


# Research reference: 10_holdings_exposure_analytics.ipynb, code cell index 5.
def calculate_active_sector_exposures(
    sector_exposures: pd.DataFrame,
    *,
    portfolio_id: str,
    benchmark_portfolio_id: str,
    require_fully_invested: bool = False,
    weight_tolerance: float = 1e-10,
) -> pd.DataFrame:
    """
    Calculate benchmark-relative sector exposures.

    Required input columns:
        portfolio_id, date, weight_basis,
        sector, sector_weight

    Input grain:
        portfolio_id, date, weight_basis, sector

    Output grain:
        portfolio_id, benchmark_portfolio_id,
        date, weight_basis, sector

    Method:
        active_sector_weight =
            portfolio_sector_weight
            - benchmark_sector_weight

    Conventions:
        Weights are decimal portfolio-capital fractions.
        BOP and EOP are treated independently.

    Missing sectors:
        A sector absent from a date-complete portfolio
        ledger is assigned zero exposure.

        An explicitly recorded zero-weight sector remains
        distinguishable from an absent sector.

    The input DataFrame is not modified.
    """

    # --------------------------------------------------------
    # A. Validate parameters and required fields
    # --------------------------------------------------------

    if not isinstance(sector_exposures, pd.DataFrame):
        raise TypeError(
            "sector_exposures must be a pandas DataFrame."
        )

    if (
        not isinstance(portfolio_id, str)
        or not isinstance(benchmark_portfolio_id, str)
        or not portfolio_id
        or not benchmark_portfolio_id
        or portfolio_id == benchmark_portfolio_id
    ):
        raise ValueError(
            "Portfolio and benchmark must have distinct, "
            "nonempty string identifiers."
        )

    if (
        not np.isfinite(weight_tolerance)
        or not 0 < weight_tolerance < 1
    ):
        raise ValueError(
            "weight_tolerance must be finite and "
            "strictly between zero and one."
        )

    required_columns = {
        "portfolio_id",
        "date",
        "weight_basis",
        "sector",
        "sector_weight",
    }

    missing_columns = (
        required_columns - set(sector_exposures.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing sector exposure fields: "
            f"{sorted(missing_columns)}"
        )

    # --------------------------------------------------------
    # B. Select portfolio and benchmark observations
    # --------------------------------------------------------

    selected_ids = {
        portfolio_id,
        benchmark_portfolio_id,
    }

    selected = sector_exposures.loc[
        sector_exposures["portfolio_id"].isin(
            selected_ids
        ),
        [
            "portfolio_id",
            "date",
            "weight_basis",
            "sector",
            "sector_weight",
        ],
    ].copy()

    observed_ids = set(
        selected["portfolio_id"].dropna().unique()
    )

    if observed_ids != selected_ids:
        raise ValueError(
            "Portfolio or benchmark sector observations "
            "are missing."
        )

    # --------------------------------------------------------
    # C. Validate keys, dates, sectors, and weights
    # --------------------------------------------------------

    selected["date"] = pd.to_datetime(
        selected["date"],
        errors="raise",
    )

    input_keys = [
        "portfolio_id",
        "date",
        "weight_basis",
        "sector",
    ]

    if selected[input_keys].isna().any().any():
        raise ValueError(
            "Sector exposure keys contain null values."
        )

    if selected.duplicated(subset=input_keys).any():
        raise ValueError(
            "Duplicate sector exposure keys detected."
        )

    if not selected["weight_basis"].isin(
        ["bop", "eop"]
    ).all():
        raise ValueError(
            "Unsupported weight_basis detected."
        )

    sector_labels = (
        selected["sector"]
        .astype("string")
        .str.strip()
    )

    if (
        sector_labels.isna().any()
        or sector_labels.eq("").any()
    ):
        raise ValueError(
            "Missing or blank sector classifications."
        )

    if not pd.api.types.is_numeric_dtype(
        selected["sector_weight"]
    ):
        raise TypeError(
            "sector_weight must be numeric."
        )

    weights = selected["sector_weight"].to_numpy(
        dtype=np.float64,
        na_value=np.nan,
    )

    if not np.isfinite(weights).all():
        raise ValueError(
            "Nonfinite sector weights detected."
        )

    if (
        (weights < 0).any()
        or (weights > 1 + weight_tolerance).any()
    ):
        raise ValueError(
            "Invalid long-only sector weights detected."
        )

    # --------------------------------------------------------
    # D. Validate complete matching date/basis coverage
    # --------------------------------------------------------

    coverage_keys = [
        "date",
        "weight_basis",
    ]

    portfolio_coverage = (
        selected.loc[
            selected["portfolio_id"].eq(portfolio_id),
            coverage_keys,
        ]
        .drop_duplicates()
        .sort_values(coverage_keys)
        .reset_index(drop=True)
    )

    benchmark_coverage = (
        selected.loc[
            selected["portfolio_id"].eq(
                benchmark_portfolio_id
            ),
            coverage_keys,
        ]
        .drop_duplicates()
        .sort_values(coverage_keys)
        .reset_index(drop=True)
    )

    if not portfolio_coverage.equals(
        benchmark_coverage
    ):
        raise ValueError(
            "Portfolio and benchmark sector date/basis "
            "coverage is not aligned."
        )

    # --------------------------------------------------------
    # E. Prepare independently identified sector records
    # --------------------------------------------------------

    join_keys = [
        "date",
        "weight_basis",
        "sector",
    ]

    portfolio_sectors = (
        selected.loc[
            selected["portfolio_id"].eq(portfolio_id),
            join_keys + ["sector_weight"],
        ]
        .rename(
            columns={
                "sector_weight": "portfolio_sector_weight"
            }
        )
    )

    benchmark_sectors = (
        selected.loc[
            selected["portfolio_id"].eq(
                benchmark_portfolio_id
            ),
            join_keys + ["sector_weight"],
        ]
        .rename(
            columns={
                "sector_weight": "benchmark_sector_weight"
            }
        )
    )

    # --------------------------------------------------------
    # F. Full outer join across observed sector records
    # --------------------------------------------------------

    aligned = portfolio_sectors.merge(
        benchmark_sectors,
        on=join_keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    portfolio_sector_present = (
        aligned["_merge"].ne("right_only")
    )

    benchmark_sector_present = (
        aligned["_merge"].ne("left_only")
    )

    # Missing exposure is zero only after date/basis
    # coverage has been independently established.

    portfolio_sector_weight = (
        aligned["portfolio_sector_weight"]
        .where(portfolio_sector_present, 0.0)
    )

    benchmark_sector_weight = (
        aligned["benchmark_sector_weight"]
        .where(benchmark_sector_present, 0.0)
    )

    # --------------------------------------------------------
    # G. Construct standardized active-sector output
    # --------------------------------------------------------

    active_sectors = pd.DataFrame(
        {
            "portfolio_id": portfolio_id,
            "benchmark_portfolio_id": (
                benchmark_portfolio_id
            ),
            "date": aligned["date"],
            "weight_basis": aligned["weight_basis"],
            "sector": aligned["sector"],
            "portfolio_sector_weight": (
                portfolio_sector_weight
            ),
            "benchmark_sector_weight": (
                benchmark_sector_weight
            ),
            "active_sector_weight": (
                portfolio_sector_weight
                - benchmark_sector_weight
            ),
            "portfolio_sector_present": (
                portfolio_sector_present
            ),
            "benchmark_sector_present": (
                benchmark_sector_present
            ),
            "portfolio_sector_invested": (
                portfolio_sector_weight > 0
            ),
            "benchmark_sector_invested": (
                benchmark_sector_weight > 0
            ),
        }
    )

    # --------------------------------------------------------
    # H. Optional full-investment reconciliation
    # --------------------------------------------------------

    if require_fully_invested:

        daily_totals = (
            active_sectors
            .groupby(
                ["date", "weight_basis"]
            )[
                [
                    "portfolio_sector_weight",
                    "benchmark_sector_weight",
                    "active_sector_weight",
                ]
            ]
            .sum()
        )

        portfolio_error = (
            daily_totals["portfolio_sector_weight"]
            .sub(1.0)
            .abs()
        )

        benchmark_error = (
            daily_totals["benchmark_sector_weight"]
            .sub(1.0)
            .abs()
        )

        active_error = (
            daily_totals["active_sector_weight"]
            .abs()
        )

        maximum_error = max(
            portfolio_error.max(),
            benchmark_error.max(),
            active_error.max(),
        )

        if (
            not np.isfinite(maximum_error)
            or maximum_error > weight_tolerance
        ):
            raise ValueError(
                "Sector full-investment or active-weight "
                "neutrality reconciliation failed. "
                f"Maximum error: {maximum_error}"
            )

    # --------------------------------------------------------
    # I. Validate output keys and return
    # --------------------------------------------------------

    output_keys = [
        "portfolio_id",
        "benchmark_portfolio_id",
        "date",
        "weight_basis",
        "sector",
    ]

    if active_sectors[output_keys].isna().any().any():
        raise ValueError(
            "Active-sector output keys contain nulls."
        )

    if active_sectors.duplicated(
        subset=output_keys
    ).any():
        raise ValueError(
            "Duplicate active-sector output keys."
        )

    return (
        active_sectors
        .sort_values(
            ["date", "weight_basis", "sector"],
            kind="stable",
        )
        .reset_index(drop=True)
    )
