"""Exposures windows: validated research calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd



# Research reference: 10_holdings_exposure_analytics.ipynb, code cell index 14.
def resolve_exposure_reporting_windows(
    available_dates: pd.DatetimeIndex,
    *,
    as_of_date: pd.Timestamp,
    trailing_years: tuple[int, ...] = (3, 5, 10),
) -> pd.DataFrame:
    """
    Resolve calendar-anniversary exposure reporting windows.

    A trailing window begins on the first observed trading
    date on or after its calendar anniversary.

    Complete historical coverage requires an observation
    on or before the anniversary date.

    If the first available observation occurs after the
    anniversary, the trailing window is unavailable.

    Since-inception reporting remains separately available.

    The function does not approximate calendar years
    using fixed trading-day counts.
    """

    dates = pd.DatetimeIndex(
        pd.to_datetime(available_dates)
    ).sort_values().unique()

    cutoff = pd.Timestamp(as_of_date)

    if (
        dates.empty
        or dates.hasnans
        or cutoff not in dates
    ):
        raise ValueError(
            "Reporting cutoff must exist in a valid "
            "observation calendar."
        )

    if (
        not isinstance(trailing_years, tuple)
        or not trailing_years
        or len(trailing_years) != len(set(trailing_years))
        or any(
            isinstance(years, (bool, np.bool_))
            or not isinstance(years, (int, np.integer))
            or years <= 0
            for years in trailing_years
        )
    ):
        raise ValueError(
            "trailing_years must contain distinct "
            "positive integers."
        )

    dates = dates[dates <= cutoff]

    records = [
        {
            "report_window": "INCEPTION",
            "years": np.nan,
            "as_of_date": cutoff,
            "required_start_date": pd.NaT,
            "window_start_date": dates.min(),
            "first_available_date": dates.min(),
            "available_observations": len(dates),
            "availability_status": "AVAILABLE",
        }
    ]

    for years in trailing_years:

        anniversary = (
            cutoff - pd.DateOffset(years=int(years))
        )

        eligible_dates = dates[dates >= anniversary]

        full_history_available = (
            dates.min() <= anniversary
        )

        status = (
            "AVAILABLE"
            if full_history_available
            else "UNAVAILABLE_INCOMPLETE_START"
        )

        records.append(
            {
                "report_window": f"{years}Y",
                "years": years,
                "as_of_date": cutoff,
                "required_start_date": anniversary,
                "window_start_date": (
                    eligible_dates.min()
                    if full_history_available
                    else pd.NaT
                ),
                "first_available_date": dates.min(),
                "available_observations": len(
                    eligible_dates
                ),
                "availability_status": status,
            }
        )

    return pd.DataFrame(records)


# Research reference: 10_holdings_exposure_analytics.ipynb, code cell index 14.
def build_exposure_window_summaries(
    security_concentration: pd.DataFrame,
    sector_concentration: pd.DataFrame,
    top_k_concentration: pd.DataFrame,
    active_share: pd.DataFrame,
    holdings_overlap: pd.DataFrame,
    active_sector_exposures: pd.DataFrame,
    reporting_windows: pd.DataFrame,
    *,
    weight_basis: str = "eop",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build three reporting tables from validated analytics.

    Returns
    -------
    portfolio_window_summary:
        Portfolio security and sector concentration.

    active_window_summary:
        Active Share and benchmark overlap.

    sector_tilt_window_summary:
        Historical active-sector weight statistics.

    All exposure measures remain decimal fractions.

    EOP is the recommended point-in-time holdings basis.

    Unavailable reporting windows retain explicit status
    but receive no fabricated numerical summary.

    Sector means include zero exposure on otherwise valid
    paired dates where neither portfolio holds that sector.
    """

    if weight_basis not in ("bop", "eop"):
        raise ValueError(
            "Reporting weight_basis must be 'bop' or 'eop'."
        )

    portfolio_keys = [
        "portfolio_id",
        "date",
        "weight_basis",
    ]

    pair_keys = [
        "portfolio_id",
        "benchmark_portfolio_id",
        "date",
        "weight_basis",
    ]

    # --------------------------------------------------------
    # A. Prepare portfolio-level daily measures
    # --------------------------------------------------------

    security_daily = security_concentration.loc[
        security_concentration["weight_basis"].eq(
            weight_basis
        ),
        portfolio_keys + [
            "invested_security_count",
            "security_hhi",
            "effective_holdings",
            "largest_position_weight",
        ],
    ].copy()

    sector_daily = sector_concentration.loc[
        sector_concentration["weight_basis"].eq(
            weight_basis
        ),
        portfolio_keys + [
            "invested_sector_count",
            "sector_hhi",
            "effective_sectors",
            "largest_sector_weight",
        ],
    ].copy()

    top_five_daily = (
        top_k_concentration.loc[
            top_k_concentration["weight_basis"].eq(
                weight_basis
            )
            & top_k_concentration["k"].eq(5),
            portfolio_keys + ["top_k_weight"],
        ]
        .rename(
            columns={
                "top_k_weight": "top_five_weight"
            }
        )
    )

    portfolio_daily = security_daily.merge(
        sector_daily,
        on=portfolio_keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    if not portfolio_daily["_merge"].eq("both").all():
        raise ValueError(
            "Security/sector reporting coverage mismatch."
        )

    portfolio_daily = (
        portfolio_daily
        .drop(columns="_merge")
        .merge(
            top_five_daily,
            on=portfolio_keys,
            how="left",
            validate="one_to_one",
        )
    )

    if portfolio_daily["top_five_weight"].isna().any():
        raise ValueError(
            "Missing top-five portfolio concentration."
        )

    # --------------------------------------------------------
    # B. Prepare benchmark-relative daily measures
    # --------------------------------------------------------

    share_daily = active_share.loc[
        active_share["weight_basis"].eq(weight_basis),
        pair_keys + [
            "security_active_share",
            "sector_active_share",
            "common_weight_overlap",
        ],
    ].copy()

    overlap_daily = holdings_overlap.loc[
        holdings_overlap["weight_basis"].eq(
            weight_basis
        ),
        pair_keys + [
            "jaccard_constituent_overlap",
            "excluded_benchmark_weight",
            "off_benchmark_weight",
        ],
    ].copy()

    pair_daily = share_daily.merge(
        overlap_daily,
        on=pair_keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    if not pair_daily["_merge"].eq("both").all():
        raise ValueError(
            "Active Share/overlap reporting coverage mismatch."
        )

    pair_daily = pair_daily.drop(
        columns="_merge"
    )

    tilt_daily = active_sector_exposures.loc[
        active_sector_exposures["weight_basis"].eq(
            weight_basis
        ),
        pair_keys + [
            "sector",
            "active_sector_weight",
        ],
    ].copy()

    # --------------------------------------------------------
    # C. Validate complete reporting calendar coverage
    # --------------------------------------------------------

    # The full approved calendar is reconstructed from
    # the pair-level Active Share observations.

    report_calendar = pd.DatetimeIndex(
        pair_daily["date"].unique()
    ).sort_values()

    if report_calendar.empty:
        raise ValueError(
            "No benchmark-relative reporting dates."
        )

    if (
        report_calendar.min()
        != reporting_windows["first_available_date"].iloc[0]
        or report_calendar.max()
        != reporting_windows["as_of_date"].iloc[0]
    ):
        raise ValueError(
            "Reporting calendar differs from the "
            "declared available observation period."
        )

    calendar_sources = [
        (
            "Portfolio concentration",
            portfolio_daily,
            ["portfolio_id"],
        ),
        (
            "Active positioning",
            pair_daily,
            [
                "portfolio_id",
                "benchmark_portfolio_id",
            ],
        ),
        (
            "Active sector tilts",
            tilt_daily,
            [
                "portfolio_id",
                "benchmark_portfolio_id",
            ],
        ),
    ]

    for name, frame, identity_columns in calendar_sources:

        if frame.empty:
            raise ValueError(
                f"{name}: no reporting observations."
            )

        for _, group in frame.groupby(
            identity_columns
        ):

            observed_dates = pd.DatetimeIndex(
                group["date"].unique()
            ).sort_values()

            if not observed_dates.equals(
                report_calendar
            ):
                raise ValueError(
                    f"{name}: incomplete observation calendar."
                )

    if not reporting_windows[
        "as_of_date"
    ].eq(report_calendar.max()).all():
        raise ValueError(
            "Reporting windows use incompatible cutoffs."
        )

    # --------------------------------------------------------
    # D. Initialize reporting outputs
    # --------------------------------------------------------

    portfolio_records = []
    pair_records = []
    tilt_records = []

    portfolio_ids = (
        portfolio_daily["portfolio_id"]
        .drop_duplicates()
        .tolist()
    )

    pair_identities = (
        pair_daily[
            [
                "portfolio_id",
                "benchmark_portfolio_id",
            ]
        ]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )

    pair_identities = list(pair_identities)

    # --------------------------------------------------------
    # E. Calculate statistics for each reporting window
    # --------------------------------------------------------

    for window in reporting_windows.itertuples(index=False):

        available = (
            window.availability_status == "AVAILABLE"
        )

        period_start = window.window_start_date
        period_end = window.as_of_date

        # Preserve unavailable windows in both high-level
        # summary contracts, without presenting partial
        # observations as complete-period metrics.

        for portfolio_id in portfolio_ids:

            result = {
                "report_window": window.report_window,
                "availability_status": (
                    window.availability_status
                ),
                "weight_basis": weight_basis,
                "portfolio_id": portfolio_id,
                "as_of_date": period_end,
                "window_start_date": period_start,
                "observations": 0,
                "median_effective_holdings": np.nan,
                "median_effective_sectors": np.nan,
                "median_top_five_weight": np.nan,
                "latest_invested_security_count": np.nan,
                "latest_invested_sector_count": np.nan,
                "latest_effective_holdings": np.nan,
                "latest_effective_sectors": np.nan,
                "latest_top_five_weight": np.nan,
            }

            if available:

                sample = portfolio_daily.loc[
                    portfolio_daily["portfolio_id"].eq(
                        portfolio_id
                    )
                    & portfolio_daily["date"].between(
                        period_start,
                        period_end,
                    )
                ].sort_values("date")

                if sample.empty:
                    raise ValueError(
                        "Available portfolio window is empty."
                    )

                latest = sample.iloc[-1]

                result.update(
                    {
                        "observations": len(sample),
                        "median_effective_holdings": (
                            sample["effective_holdings"].median()
                        ),
                        "median_effective_sectors": (
                            sample["effective_sectors"].median()
                        ),
                        "median_top_five_weight": (
                            sample["top_five_weight"].median()
                        ),
                        "latest_invested_security_count": (
                            latest["invested_security_count"]
                        ),
                        "latest_invested_sector_count": (
                            latest["invested_sector_count"]
                        ),
                        "latest_effective_holdings": (
                            latest["effective_holdings"]
                        ),
                        "latest_effective_sectors": (
                            latest["effective_sectors"]
                        ),
                        "latest_top_five_weight": (
                            latest["top_five_weight"]
                        ),
                    }
                )

            portfolio_records.append(result)

        for portfolio_id, benchmark_id in pair_identities:

            result = {
                "report_window": window.report_window,
                "availability_status": (
                    window.availability_status
                ),
                "weight_basis": weight_basis,
                "portfolio_id": portfolio_id,
                "benchmark_portfolio_id": benchmark_id,
                "as_of_date": period_end,
                "window_start_date": period_start,
                "observations": 0,
                "median_security_active_share": np.nan,
                "median_sector_active_share": np.nan,
                "median_common_weight_overlap": np.nan,
                "median_jaccard_overlap": np.nan,
                "median_excluded_benchmark_weight": np.nan,
                "latest_security_active_share": np.nan,
                "latest_sector_active_share": np.nan,
                "latest_common_weight_overlap": np.nan,
            }

            if available:

                sample = pair_daily.loc[
                    pair_daily["portfolio_id"].eq(
                        portfolio_id
                    )
                    & pair_daily[
                        "benchmark_portfolio_id"
                    ].eq(benchmark_id)
                    & pair_daily["date"].between(
                        period_start,
                        period_end,
                    )
                ].sort_values("date")

                if sample.empty:
                    raise ValueError(
                        "Available active-pair window is empty."
                    )

                latest = sample.iloc[-1]

                result.update(
                    {
                        "observations": len(sample),
                        "median_security_active_share": (
                            sample[
                                "security_active_share"
                            ].median()
                        ),
                        "median_sector_active_share": (
                            sample[
                                "sector_active_share"
                            ].median()
                        ),
                        "median_common_weight_overlap": (
                            sample[
                                "common_weight_overlap"
                            ].median()
                        ),
                        "median_jaccard_overlap": (
                            sample[
                                "jaccard_constituent_overlap"
                            ].median()
                        ),
                        "median_excluded_benchmark_weight": (
                            sample[
                                "excluded_benchmark_weight"
                            ].median()
                        ),
                        "latest_security_active_share": (
                            latest["security_active_share"]
                        ),
                        "latest_sector_active_share": (
                            latest["sector_active_share"]
                        ),
                        "latest_common_weight_overlap": (
                            latest["common_weight_overlap"]
                        ),
                    }
                )

            pair_records.append(result)

            if not available:
                continue

            # ------------------------------------------------
            # F. Build calendar-complete sector exposure grid
            # ------------------------------------------------

            sector_sample = tilt_daily.loc[
                tilt_daily["portfolio_id"].eq(
                    portfolio_id
                )
                & tilt_daily[
                    "benchmark_portfolio_id"
                ].eq(benchmark_id)
                & tilt_daily["date"].between(
                    period_start,
                    period_end,
                )
            ].copy()

            period_dates = report_calendar[
                (report_calendar >= period_start)
                & (report_calendar <= period_end)
            ]

            sectors = sorted(
                sector_sample["sector"].unique()
            )

            sector_grid = (
                pd.MultiIndex.from_product(
                    [period_dates, sectors],
                    names=["date", "sector"],
                )
                .to_frame(index=False)
            )

            sector_grid = sector_grid.merge(
                sector_sample[
                    [
                        "date",
                        "sector",
                        "active_sector_weight",
                    ]
                ],
                on=["date", "sector"],
                how="left",
                validate="one_to_one",
                indicator=True,
            )

            sector_grid["absent_in_both_ledgers"] = (
                sector_grid["_merge"].eq(
                    "left_only"
                )
            )

            # A missing sector row is zero only because
            # the paired portfolio date was validated
            # as fully observed.

            sector_grid["active_sector_weight"] = (
                sector_grid["active_sector_weight"]
                .fillna(0.0)
            )

            # ------------------------------------------------
            # G. Summarize complete daily sector tilts
            # ------------------------------------------------

            sector_statistics = (
                sector_grid
                .assign(
                    positive_tilt=lambda df: (
                        df["active_sector_weight"] > 0
                    )
                )
                .groupby("sector", as_index=False)
                .agg(
                    observations=(
                        "date", "size"
                    ),
                    mean_active_sector_weight=(
                        "active_sector_weight", "mean"
                    ),
                    median_active_sector_weight=(
                        "active_sector_weight", "median"
                    ),
                    minimum_active_sector_weight=(
                        "active_sector_weight", "min"
                    ),
                    maximum_active_sector_weight=(
                        "active_sector_weight", "max"
                    ),
                    fraction_dates_overweight=(
                        "positive_tilt", "mean"
                    ),
                    dates_absent_in_both=(
                        "absent_in_both_ledgers", "sum"
                    ),
                )
            )

            end_tilts = sector_grid.loc[
                sector_grid["date"].eq(period_end),
                [
                    "sector",
                    "active_sector_weight",
                ],
            ].rename(
                columns={
                    "active_sector_weight": (
                        "latest_active_sector_weight"
                    )
                }
            )

            sector_statistics = sector_statistics.merge(
                end_tilts,
                on="sector",
                how="left",
                validate="one_to_one",
            )

            sector_statistics["report_window"] = (
                window.report_window
            )

            sector_statistics["weight_basis"] = (
                weight_basis
            )

            sector_statistics["portfolio_id"] = (
                portfolio_id
            )

            sector_statistics[
                "benchmark_portfolio_id"
            ] = benchmark_id

            sector_statistics["as_of_date"] = period_end

            sector_statistics[
                "window_start_date"
            ] = period_start

            tilt_records.append(sector_statistics)

    # --------------------------------------------------------
    # H. Construct standardized reporting outputs
    # --------------------------------------------------------

    portfolio_report = pd.DataFrame(
        portfolio_records
    )

    pair_report = pd.DataFrame(
        pair_records
    )

    tilt_report = pd.concat(
        tilt_records,
        ignore_index=True,
    ) if tilt_records else pd.DataFrame()

    return portfolio_report, pair_report, tilt_report
