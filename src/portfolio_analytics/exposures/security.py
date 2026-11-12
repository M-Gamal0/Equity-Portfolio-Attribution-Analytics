"""Exposures security: validated research calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd



# Research reference: 10_holdings_exposure_analytics.ipynb, code cell index 2.
def prepare_security_exposures(
    holdings_df: pd.DataFrame,
    *,
    weight_bases: tuple[str, ...] = ("bop", "eop"),
) -> pd.DataFrame:
    """
    Transform standardized daily holdings into security exposures.

    Parameters
    ----------
    holdings_df:
        Standardized holdings with portfolio/date/security keys,
        sector classifications and selected weight columns.

    weight_bases:
        Tuple containing 'bop', 'eop', or both.

    Returns
    -------
    pd.DataFrame:
        One row per portfolio_id, date, security_id,
        and weight_basis.

    Financial conventions
    ---------------------
    bop:
        Weight applied to the daily close-to-close return interval.

    eop:
        Final closing weight after price drift and portfolio events.

    security_weight:
        Decimal fraction of portfolio capital.

    is_held:
        True if security_weight > 0.

    Notes
    -----
    Zero-weight observations are preserved.
    No missing security weights are inferred.
    The input DataFrame is not modified.
    """

    if not isinstance(holdings_df, pd.DataFrame):
        raise TypeError(
            "holdings_df must be a pandas DataFrame."
        )

    weight_column_map = {
        "bop": "weight_bop",
        "eop": "weight_eop",
    }

    if (
        not isinstance(weight_bases, tuple)
        or not weight_bases
        or len(weight_bases) != len(set(weight_bases))
        or any(
            basis not in weight_column_map
            for basis in weight_bases
        )
    ):
        raise ValueError(
            "weight_bases must be a nonempty tuple of "
            "distinct values selected from ('bop', 'eop')."
        )

    # --------------------------------------------------------
    # A. Validate required input columns
    # --------------------------------------------------------

    key_columns = [
        "portfolio_id",
        "date",
        "security_id",
    ]

    required_columns = set(key_columns) | {"sector"}

    selected_weight_columns = [
        weight_column_map[basis]
        for basis in weight_bases
    ]

    required_columns.update(selected_weight_columns)

    missing_columns = (
        required_columns - set(holdings_df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required exposure fields: "
            f"{sorted(missing_columns)}"
        )

    if holdings_df.empty:
        raise ValueError(
            "Holdings DataFrame cannot be empty."
        )

    if holdings_df[key_columns].isna().any().any():
        raise ValueError(
            "Holdings contain null primary keys."
        )

    if holdings_df.duplicated(
        subset=key_columns
    ).any():
        raise ValueError(
            "Holdings contain duplicate primary keys."
        )

    if holdings_df["sector"].isna().any():
        raise ValueError(
            "Holdings contain missing sector classifications."
        )

    # --------------------------------------------------------
    # B. Validate selected weight columns
    # --------------------------------------------------------

    for column in selected_weight_columns:

        if not pd.api.types.is_numeric_dtype(
            holdings_df[column]
        ):
            raise TypeError(
                f"{column} must be numeric."
            )

        values = holdings_df[column].to_numpy(
            dtype=np.float64,
            na_value=np.nan,
        )

        if not np.isfinite(values).all():
            raise ValueError(
                f"{column} contains nonfinite weights."
            )

        if (values < 0).any():
            raise ValueError(
                f"{column} contains negative weights."
            )

        if (values > 1).any():
            raise ValueError(
                f"{column} contains weights above 100%."
            )

    # --------------------------------------------------------
    # C. Select identifying and optional fields
    # --------------------------------------------------------

    identifier_columns = [
        "portfolio_id",
        "date",
        "security_id",
        "sector",
    ]

    if "ticker" in holdings_df.columns:
        identifier_columns.append("ticker")

    # --------------------------------------------------------
    # D. Transform BOP/EOP weights into long format
    # --------------------------------------------------------

    security_exposures = holdings_df[
        identifier_columns + selected_weight_columns
    ].melt(
        id_vars=identifier_columns,
        value_vars=selected_weight_columns,
        var_name="weight_basis",
        value_name="security_weight",
    )

    reverse_weight_map = {
        column: basis
        for basis, column in weight_column_map.items()
    }

    security_exposures["weight_basis"] = (
        security_exposures["weight_basis"]
        .map(reverse_weight_map)
    )

    security_exposures["is_held"] = (
        security_exposures["security_weight"] > 0
    )

    return security_exposures


# Research reference: 10_holdings_exposure_analytics.ipynb, code cell index 4.
def calculate_active_security_exposures(
    security_exposures: pd.DataFrame,
    *,
    portfolio_id: str,
    benchmark_portfolio_id: str,
    require_fully_invested: bool = False,
    weight_tolerance: float = 1e-10,
) -> pd.DataFrame:
    """
    Calculate benchmark-relative security exposures.

    Input grain:
        portfolio_id, date, security_id, weight_basis

    Output grain:
        portfolio_id, benchmark_portfolio_id,
        date, weight_basis, security_id

    Required input fields:
        portfolio_id, date, security_id, sector,
        weight_basis, security_weight, is_held

    Method:
        active_weight = portfolio_weight - benchmark_weight

    Financial conventions:
        Weights are decimal fractions.
        BOP and EOP exposures remain separate.

    Missing positions:
        Absence from a validated, date-complete portfolio
        ledger represents zero exposure for that security.

        Recorded zero-weight holdings remain distinguishable
        from securities absent from the ledger.

    Policies:
        The selected portfolios must have identical
        observation calendars for each weight basis.

        Full investment is optional for portability.

    Returns:
        A new DataFrame without modifying the input.
    """

    # --------------------------------------------------------
    # A. Validate parameters and required columns
    # --------------------------------------------------------

    if not isinstance(security_exposures, pd.DataFrame):
        raise TypeError(
            "security_exposures must be a pandas DataFrame."
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

    # --------------------------------------------------------
    # B. Select portfolio and benchmark observations
    # --------------------------------------------------------

    selected_ids = {
        portfolio_id,
        benchmark_portfolio_id,
    }

    selected_columns = [
        "portfolio_id",
        "date",
        "security_id",
        "sector",
        "weight_basis",
        "security_weight",
        "is_held",
    ]

    if "ticker" in security_exposures.columns:
        selected_columns.append("ticker")

    selected = security_exposures.loc[
        security_exposures["portfolio_id"].isin(
            selected_ids
        ),
        selected_columns,
    ].copy()

    observed_ids = set(
        selected["portfolio_id"].dropna().unique()
    )

    if observed_ids != selected_ids:
        raise ValueError(
            "Portfolio or benchmark observations are missing."
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
        "security_id",
        "weight_basis",
    ]

    if selected[input_keys].isna().any().any():
        raise ValueError(
            "Security exposure primary keys contain nulls."
        )

    if selected.duplicated(subset=input_keys).any():
        raise ValueError(
            "Duplicate security exposure keys detected."
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
        selected["security_weight"]
    ):
        raise TypeError(
            "security_weight must be numeric."
        )

    weights = selected["security_weight"].to_numpy(
        dtype=np.float64,
        na_value=np.nan,
    )

    if not np.isfinite(weights).all():
        raise ValueError(
            "Nonfinite security weights detected."
        )

    if (
        (weights < 0).any()
        or (weights > 1 + weight_tolerance).any()
    ):
        raise ValueError(
            "Invalid long-only security weights detected."
        )

    if (
        not pd.api.types.is_bool_dtype(selected["is_held"])
        or selected["is_held"].isna().any()
    ):
        raise TypeError(
            "is_held must contain nonmissing Boolean values."
        )

    if not selected["is_held"].eq(
        selected["security_weight"] > 0
    ).all():
        raise ValueError(
            "is_held flags do not match security weights."
        )

    # --------------------------------------------------------
    # D. Verify complete, aligned observation calendars
    # --------------------------------------------------------

    observed_bases = set(
        selected["weight_basis"].unique()
    )

    for basis in observed_bases:

        basis_data = selected.loc[
            selected["weight_basis"].eq(basis)
        ]

        calendars = {}

        for selected_id in (
            portfolio_id,
            benchmark_portfolio_id,
        ):

            dates = basis_data.loc[
                basis_data["portfolio_id"].eq(selected_id),
                "date",
            ].unique()

            calendars[selected_id] = (
                pd.DatetimeIndex(dates).sort_values()
            )

            if calendars[selected_id].empty:
                raise ValueError(
                    f"Missing {basis} observations "
                    f"for portfolio {selected_id}."
                )

        if not calendars[portfolio_id].equals(
            calendars[benchmark_portfolio_id]
        ):
            raise ValueError(
                "Portfolio and benchmark holdings calendars "
                f"are not aligned for weight_basis={basis}."
            )

    # --------------------------------------------------------
    # E. Prepare independently identified holdings
    # --------------------------------------------------------

    join_keys = [
        "date",
        "weight_basis",
        "security_id",
    ]

    value_columns = [
        "sector",
        "security_weight",
        "is_held",
    ]

    if "ticker" in selected.columns:
        value_columns.append("ticker")

    portfolio_positions = selected.loc[
        selected["portfolio_id"].eq(portfolio_id),
        join_keys + value_columns,
    ]

    benchmark_positions = selected.loc[
        selected["portfolio_id"].eq(
            benchmark_portfolio_id
        ),
        join_keys + value_columns,
    ]

    # --------------------------------------------------------
    # F. Full outer join by security, date, and basis
    # --------------------------------------------------------

    aligned = portfolio_positions.merge(
        benchmark_positions,
        on=join_keys,
        how="outer",
        suffixes=("_portfolio", "_benchmark"),
        validate="one_to_one",
        indicator=True,
    )

    portfolio_record_present = (
        aligned["_merge"].ne("right_only")
    )

    benchmark_record_present = (
        aligned["_merge"].ne("left_only")
    )

    # --------------------------------------------------------
    # G. Validate cross-portfolio sector consistency
    # --------------------------------------------------------

    shared_records = (
        portfolio_record_present
        & benchmark_record_present
    )

    sector_conflicts = (
        shared_records
        & aligned["sector_portfolio"].ne(
            aligned["sector_benchmark"]
        )
    )

    if sector_conflicts.any():
        raise ValueError(
            "Conflicting sector classifications for "
            "securities shared by portfolio and benchmark."
        )

    resolved_sector = (
        aligned["sector_portfolio"]
        .combine_first(aligned["sector_benchmark"])
    )

    # --------------------------------------------------------
    # H. Resolve legitimate absent-security exposures
    # --------------------------------------------------------

    # All dates/bases have already been validated.
    # A missing security-side record can therefore be
    # interpreted as an absent position with zero weight.

    portfolio_weight = (
        aligned["security_weight_portfolio"]
        .where(portfolio_record_present, 0.0)
    )

    benchmark_weight = (
        aligned["security_weight_benchmark"]
        .where(benchmark_record_present, 0.0)
    )

    portfolio_is_held = (
        portfolio_weight > 0
    )

    benchmark_is_held = (
        benchmark_weight > 0
    )

    # --------------------------------------------------------
    # I. Construct standardized active security output
    # --------------------------------------------------------

    active_exposures = pd.DataFrame(
        {
            "portfolio_id": portfolio_id,
            "benchmark_portfolio_id": (
                benchmark_portfolio_id
            ),
            "date": aligned["date"],
            "weight_basis": aligned["weight_basis"],
            "security_id": aligned["security_id"],
            "sector": resolved_sector,
            "portfolio_weight": portfolio_weight,
            "benchmark_weight": benchmark_weight,
            "active_weight": (
                portfolio_weight - benchmark_weight
            ),
            "portfolio_record_present": (
                portfolio_record_present
            ),
            "benchmark_record_present": (
                benchmark_record_present
            ),
            "portfolio_is_held": portfolio_is_held,
            "benchmark_is_held": benchmark_is_held,
        }
    )

    if "ticker" in selected.columns:

        active_exposures["ticker"] = (
            aligned["ticker_portfolio"]
            .combine_first(aligned["ticker_benchmark"])
        )

    active_exposures["shared_holding"] = (
        portfolio_is_held & benchmark_is_held
    )

    active_exposures["off_benchmark_holding"] = (
        portfolio_is_held & ~benchmark_is_held
    )

    active_exposures["benchmark_only_holding"] = (
        benchmark_is_held & ~portfolio_is_held
    )

    # --------------------------------------------------------
    # J. Optional full-investment reconciliation
    # --------------------------------------------------------

    if require_fully_invested:

        daily_totals = (
            active_exposures
            .groupby(
                ["date", "weight_basis"]
            )[
                [
                    "portfolio_weight",
                    "benchmark_weight",
                    "active_weight",
                ]
            ]
            .sum()
        )

        portfolio_error = (
            daily_totals["portfolio_weight"] - 1.0
        ).abs()

        benchmark_error = (
            daily_totals["benchmark_weight"] - 1.0
        ).abs()

        active_error = (
            daily_totals["active_weight"]
        ).abs()

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
                "Full-investment or active-weight "
                "neutrality reconciliation failed. "
                f"Maximum error: {maximum_error}"
            )

    # --------------------------------------------------------
    # K. Validate final output grain
    # --------------------------------------------------------

    output_keys = [
        "portfolio_id",
        "benchmark_portfolio_id",
        "date",
        "weight_basis",
        "security_id",
    ]

    if active_exposures[output_keys].isna().any().any():
        raise ValueError(
            "Active exposure output keys contain nulls."
        )

    if active_exposures.duplicated(
        subset=output_keys
    ).any():
        raise ValueError(
            "Duplicate active security exposure keys."
        )

    return (
        active_exposures
        .sort_values(
            ["date", "weight_basis"],
            kind="stable",
        )
        .reset_index(drop=True)
    )
