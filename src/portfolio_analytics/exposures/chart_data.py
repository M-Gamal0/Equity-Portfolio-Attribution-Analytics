"""Exposures chart data: validated research calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd



# Research reference: 10_holdings_exposure_analytics.ipynb, code cell index 15.
def prepare_historical_exposure_chart_data(
    active_share: pd.DataFrame,
    security_concentration: pd.DataFrame,
    sector_concentration: pd.DataFrame,
    *,
    portfolio_id: str,
    benchmark_portfolio_id: str,
    reporting_window: pd.Series,
    available_dates: pd.DatetimeIndex,
    weight_basis: str = "eop",
) -> pd.DataFrame:
    """
    Prepare aligned daily exposure metrics for visualization.

    Output grain:
        One row per valid trading date.

    Required conditions:
        Approved reporting window.
        Exact date coverage across all source outputs.
        Unique daily observations.
        Finite numerical measures.
        Valid Active Share and concentration bounds.

    No missing dates are interpolated or forward-filled.

    The function does not modify its inputs.
    """

    # --------------------------------------------------------
    # A. Validate reporting settings
    # --------------------------------------------------------

    if weight_basis not in ("bop", "eop"):
        raise ValueError(
            "Invalid chart weight_basis."
        )

    if (
        reporting_window["availability_status"]
        != "AVAILABLE"
    ):
        raise ValueError(
            "Requested reporting window is unavailable: "
            f"{reporting_window['report_window']}"
        )

    window_start = pd.Timestamp(
        reporting_window["window_start_date"]
    )

    window_end = pd.Timestamp(
        reporting_window["as_of_date"]
    )

    calendar = pd.DatetimeIndex(
        pd.to_datetime(available_dates)
    ).sort_values().unique()

    expected_dates = calendar[
        (calendar >= window_start)
        & (calendar <= window_end)
    ]

    if expected_dates.empty:
        raise ValueError(
            "Requested chart window contains no observations."
        )

    # --------------------------------------------------------
    # B. Define expected source observations
    # --------------------------------------------------------

    source_specs = {
        "active_share": {
            "frame": active_share,
            "portfolio_id": portfolio_id,
            "benchmark_portfolio_id": (
                benchmark_portfolio_id
            ),
            "columns": {
                "security_active_share": (
                    "security_active_share"
                ),
                "sector_active_share": (
                    "sector_active_share"
                ),
            },
        },
        "active_security_concentration": {
            "frame": security_concentration,
            "portfolio_id": portfolio_id,
            "columns": {
                "security_hhi": (
                    "active_security_hhi"
                ),
                "effective_holdings": (
                    "active_effective_holdings"
                ),
            },
        },
        "benchmark_security_concentration": {
            "frame": security_concentration,
            "portfolio_id": benchmark_portfolio_id,
            "columns": {
                "security_hhi": (
                    "benchmark_security_hhi"
                ),
                "effective_holdings": (
                    "benchmark_effective_holdings"
                ),
            },
        },
        "active_sector_concentration": {
            "frame": sector_concentration,
            "portfolio_id": portfolio_id,
            "columns": {
                "effective_sectors": (
                    "active_effective_sectors"
                ),
            },
        },
        "benchmark_sector_concentration": {
            "frame": sector_concentration,
            "portfolio_id": benchmark_portfolio_id,
            "columns": {
                "effective_sectors": (
                    "benchmark_effective_sectors"
                ),
            },
        },
    }

    # --------------------------------------------------------
    # C. Require complete source coverage
    # --------------------------------------------------------

    result = pd.DataFrame(
        {"date": expected_dates}
    )

    for source_name, spec in source_specs.items():

        source = spec["frame"]

        required_columns = {
            "portfolio_id",
            "date",
            "weight_basis",
            *spec["columns"].keys(),
        }

        if "benchmark_portfolio_id" in spec:
            required_columns.add(
                "benchmark_portfolio_id"
            )

        missing_columns = (
            required_columns - set(source.columns)
        )

        if missing_columns:
            raise ValueError(
                f"{source_name}: missing fields "
                f"{sorted(missing_columns)}."
            )

        selection = (
            source["portfolio_id"].eq(
                spec["portfolio_id"]
            )
            & source["weight_basis"].eq(
                weight_basis
            )
            & source["date"].between(
                window_start,
                window_end,
            )
        )

        if "benchmark_portfolio_id" in spec:

            selection &= source[
                "benchmark_portfolio_id"
            ].eq(
                spec["benchmark_portfolio_id"]
            )

        selected = source.loc[
            selection,
            ["date", *spec["columns"].keys()],
        ].copy()

        selected["date"] = pd.to_datetime(
            selected["date"],
            errors="raise",
        )

        if selected["date"].duplicated().any():
            raise ValueError(
                f"{source_name}: duplicate chart dates."
            )

        selected = selected.sort_values(
            "date"
        ).reset_index(drop=True)

        observed_dates = pd.DatetimeIndex(
            selected["date"]
        )

        if not observed_dates.equals(expected_dates):
            raise ValueError(
                f"{source_name}: chart calendar "
                "coverage mismatch."
            )

        selected = selected.rename(
            columns=spec["columns"]
        )

        result = result.merge(
            selected,
            on="date",
            how="left",
            validate="one_to_one",
        )

    # --------------------------------------------------------
    # D. Validate complete finite numerical measures
    # --------------------------------------------------------

    measure_columns = [
        column
        for column in result.columns
        if column != "date"
    ]

    values = result[measure_columns].to_numpy(
        dtype=np.float64,
        na_value=np.nan,
    )

    if not np.isfinite(values).all():
        raise ValueError(
            "Historical chart data contain missing "
            "or nonfinite metrics."
        )

    if (
        result["sector_active_share"]
        > result["security_active_share"] + 1e-10
    ).any():
        raise ValueError(
            "Sector Active Share exceeds security Active Share."
        )

    for column in (
        "security_active_share",
        "sector_active_share",
        "active_security_hhi",
        "benchmark_security_hhi",
    ):

        if (
            result[column].lt(-1e-10).any()
            or result[column].gt(1 + 1e-10).any()
        ):
            raise ValueError(
                f"{column}: metric outside [0, 1]."
            )

    for column in (
        "active_effective_holdings",
        "benchmark_effective_holdings",
        "active_effective_sectors",
        "benchmark_effective_sectors",
    ):

        if not result[column].gt(0).all():
            raise ValueError(
                f"{column}: expected positive values."
            )

    return result


# Research reference: 10_holdings_exposure_analytics.ipynb, code cell index 16.
def prepare_sector_positioning_chart_data(
    active_sector_exposures: pd.DataFrame,
    *,
    portfolio_id: str,
    benchmark_portfolio_id: str,
    reporting_window: pd.Series,
    available_dates: pd.DatetimeIndex,
    weight_basis: str = "eop",
    weight_tolerance: float = 1e-10,
) -> pd.DataFrame:
    """
    Prepare calendar-complete sector positioning data.

    Output grain:
        portfolio_id, benchmark_portfolio_id,
        date, weight_basis, sector

    Financial conventions:
        Sector weights are decimal capital fractions.

        Active sector weight equals portfolio weight
        minus benchmark weight.

        Missing sectors on validated portfolio dates
        are treated as zero.

        Missing entire dates are rejected.

    Sector classifications are preserved exactly as
    provided by the standardized input contract.

    The input DataFrame is not modified.
    """

    # --------------------------------------------------------
    # A. Validate input and reporting parameters
    # --------------------------------------------------------

    if not isinstance(
        active_sector_exposures, pd.DataFrame
    ):
        raise TypeError(
            "active_sector_exposures must be a DataFrame."
        )

    if weight_basis not in ("bop", "eop"):
        raise ValueError(
            "Invalid sector chart weight_basis."
        )

    if (
        not np.isfinite(weight_tolerance)
        or not 0 < weight_tolerance < 1
    ):
        raise ValueError(
            "Invalid sector chart weight_tolerance."
        )

    if (
        reporting_window["availability_status"]
        != "AVAILABLE"
    ):
        raise ValueError(
            "Requested sector chart reporting "
            "window is unavailable."
        )

    start_date = pd.Timestamp(
        reporting_window["window_start_date"]
    )

    end_date = pd.Timestamp(
        reporting_window["as_of_date"]
    )

    calendar = pd.DatetimeIndex(
        pd.to_datetime(available_dates)
    ).sort_values().unique()

    expected_dates = calendar[
        (calendar >= start_date)
        & (calendar <= end_date)
    ]

    if expected_dates.empty:
        raise ValueError(
            "Sector chart reporting window is empty."
        )

    # --------------------------------------------------------
    # B. Validate required source columns
    # --------------------------------------------------------

    required_columns = {
        "portfolio_id",
        "benchmark_portfolio_id",
        "date",
        "weight_basis",
        "sector",
        "portfolio_sector_weight",
        "benchmark_sector_weight",
        "active_sector_weight",
    }

    missing_columns = (
        required_columns
        - set(active_sector_exposures.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing sector chart fields: "
            f"{sorted(missing_columns)}"
        )

    # --------------------------------------------------------
    # C. Select the required portfolio pair and weight basis
    # --------------------------------------------------------

    pair_mask = (
        active_sector_exposures["portfolio_id"].eq(
            portfolio_id
        )
        & active_sector_exposures[
            "benchmark_portfolio_id"
        ].eq(
            benchmark_portfolio_id
        )
        & active_sector_exposures["weight_basis"].eq(
            weight_basis
        )
    )

    pair_history = active_sector_exposures.loc[
        pair_mask
    ].copy()

    if pair_history.empty:
        raise ValueError(
            "No sector observations for the requested pair."
        )

    pair_history["date"] = pd.to_datetime(
        pair_history["date"],
        errors="raise",
    )

    # Preserve the sector universe represented in the
    # approved historical data through the reporting cutoff.

    historical_sectors = sorted(
        pair_history.loc[
            pair_history["date"].le(end_date),
            "sector",
        ].dropna().unique()
    )

    if not historical_sectors:
        raise ValueError(
            "No historical sector classifications."
        )

    work = pair_history.loc[
        pair_history["date"].between(
            start_date,
            end_date,
        ),
        [
            "date",
            "sector",
            "portfolio_sector_weight",
            "benchmark_sector_weight",
            "active_sector_weight",
        ],
    ].copy()

    # --------------------------------------------------------
    # D. Validate keys and numerical integrity
    # --------------------------------------------------------

    if work[["date", "sector"]].isna().any().any():
        raise ValueError(
            "Sector chart observation keys contain nulls."
        )

    if work.duplicated(
        subset=["date", "sector"]
    ).any():
        raise ValueError(
            "Duplicate sector chart observation keys."
        )

    sector_labels = (
        work["sector"]
        .astype("string")
        .str.strip()
    )

    if sector_labels.eq("").any():
        raise ValueError(
            "Blank sector classifications detected."
        )

    numeric_columns = [
        "portfolio_sector_weight",
        "benchmark_sector_weight",
        "active_sector_weight",
    ]

    for column in numeric_columns:

        if not pd.api.types.is_numeric_dtype(
            work[column]
        ):
            raise TypeError(
                f"{column} must be numeric."
            )

        values = work[column].to_numpy(
            dtype=np.float64,
            na_value=np.nan,
        )

        if not np.isfinite(values).all():
            raise ValueError(
                f"{column} contains nonfinite observations."
            )

    for column in (
        "portfolio_sector_weight",
        "benchmark_sector_weight",
    ):

        if (
            (work[column] < 0).any()
            or (
                work[column]
                > 1 + weight_tolerance
            ).any()
        ):
            raise ValueError(
                f"{column} violates long-only bounds."
            )

    active_identity_error = (
        work["active_sector_weight"]
        - (
            work["portfolio_sector_weight"]
            - work["benchmark_sector_weight"]
        )
    ).abs().max()

    if active_identity_error > weight_tolerance:
        raise ValueError(
            "Sector chart active-weight identity failed."
        )

    # --------------------------------------------------------
    # E. Reject missing entire observation dates
    # --------------------------------------------------------

    observed_dates = pd.DatetimeIndex(
        work["date"].unique()
    ).sort_values()

    if not observed_dates.equals(expected_dates):
        raise ValueError(
            "Sector positioning calendar coverage mismatch."
        )

    # --------------------------------------------------------
    # F. Construct complete date-by-sector observation grid
    # --------------------------------------------------------

    sector_grid = (
        pd.MultiIndex.from_product(
            [
                expected_dates,
                historical_sectors,
            ],
            names=["date", "sector"],
        )
        .to_frame(index=False)
    )

    sector_grid = sector_grid.merge(
        work,
        on=["date", "sector"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    sector_grid["sector_record_present"] = (
        sector_grid["_merge"].eq("both")
    )

    # Absent sectors are zero only after complete
    # date coverage has been established.

    sector_grid[numeric_columns] = (
        sector_grid[numeric_columns].fillna(0.0)
    )

    sector_grid = sector_grid.drop(
        columns="_merge"
    )

    sector_grid["portfolio_id"] = portfolio_id

    sector_grid["benchmark_portfolio_id"] = (
        benchmark_portfolio_id
    )

    sector_grid["weight_basis"] = weight_basis

    # --------------------------------------------------------
    # G. Reconcile daily sector weights to full investment
    # --------------------------------------------------------

    daily_totals = (
        sector_grid
        .groupby("date")[numeric_columns]
        .sum()
    )

    portfolio_error = (
        daily_totals["portfolio_sector_weight"]
        .sub(1.0)
        .abs()
        .max()
    )

    benchmark_error = (
        daily_totals["benchmark_sector_weight"]
        .sub(1.0)
        .abs()
        .max()
    )

    active_neutrality_error = (
        daily_totals["active_sector_weight"]
        .abs()
        .max()
    )

    if max(
        portfolio_error,
        benchmark_error,
        active_neutrality_error,
    ) > weight_tolerance:
        raise ValueError(
            "Sector chart daily capital reconciliation failed."
        )

    # --------------------------------------------------------
    # H. Validate standardized chart-data output
    # --------------------------------------------------------

    output_keys = [
        "portfolio_id",
        "benchmark_portfolio_id",
        "date",
        "weight_basis",
        "sector",
    ]

    if sector_grid.duplicated(
        subset=output_keys
    ).any():
        raise ValueError(
            "Duplicate sector chart output keys."
        )

    expected_rows = (
        len(expected_dates) * len(historical_sectors)
    )

    if len(sector_grid) != expected_rows:
        raise ValueError(
            "Incomplete sector chart date/sector grid."
        )

    return (
        sector_grid
        .sort_values(
            ["date", "sector"],
            kind="stable",
        )
        .reset_index(drop=True)
    )
