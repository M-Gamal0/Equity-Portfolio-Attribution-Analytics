"""Exposures active share: validated research calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd



# Research reference: 10_holdings_exposure_analytics.ipynb, code cell index 6.
def calculate_active_share(
    active_security_exposures: pd.DataFrame,
    *,
    require_fully_invested: bool = False,
    weight_tolerance: float = 1e-10,
) -> pd.DataFrame:
    """
    Calculate security-level and sector-level Active Share.

    Required input columns:
        portfolio_id, benchmark_portfolio_id,
        date, weight_basis, security_id, sector,
        portfolio_weight, benchmark_weight, active_weight

    Input grain:
        portfolio_id, benchmark_portfolio_id,
        date, weight_basis, security_id

    Output grain:
        portfolio_id, benchmark_portfolio_id,
        date, weight_basis

    Financial methodology:
        Security Active Share:
            0.5 * sum(abs(security active weights))

        Sector Active Share:
            0.5 * sum(abs(sector active weights))

        Common-weight overlap:
            sum(min(portfolio_weight, benchmark_weight))

    For fully invested long-only portfolios:
        0 <= sector Active Share <= security Active Share <= 1

        Security Active Share = positive active-weight sum
                              = absolute negative-weight sum
                              = 1 - common-weight overlap

    All weights and Active Share values use decimal fractions.

    Missing positions must already be explicitly aligned
    in the supplied active-security exposure DataFrame.

    Does not modify the input DataFrame.
    """

    # --------------------------------------------------------
    # A. Validate parameters and input structure
    # --------------------------------------------------------

    if not isinstance(
        active_security_exposures, pd.DataFrame
    ):
        raise TypeError(
            "active_security_exposures must be a DataFrame."
        )

    if (
        not np.isfinite(weight_tolerance)
        or not 0 < weight_tolerance < 1
    ):
        raise ValueError(
            "weight_tolerance must be finite and "
            "strictly between zero and one."
        )

    group_keys = [
        "portfolio_id",
        "benchmark_portfolio_id",
        "date",
        "weight_basis",
    ]

    security_keys = group_keys + ["security_id"]

    required_columns = set(security_keys) | {
        "sector",
        "portfolio_weight",
        "benchmark_weight",
        "active_weight",
    }

    missing_columns = (
        required_columns
        - set(active_security_exposures.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing Active Share input columns: "
            f"{sorted(missing_columns)}"
        )

    if active_security_exposures.empty:
        raise ValueError(
            "Active security exposures cannot be empty."
        )

    # Work exclusively on a local projection.

    work = active_security_exposures.loc[
        :,
        list(security_keys) + [
            "sector",
            "portfolio_weight",
            "benchmark_weight",
            "active_weight",
        ],
    ].copy()

    # --------------------------------------------------------
    # B. Validate keys, dates, and classifications
    # --------------------------------------------------------

    work["date"] = pd.to_datetime(
        work["date"],
        errors="raise",
    )

    if work[security_keys].isna().any().any():
        raise ValueError(
            "Active Share input keys contain null values."
        )

    if work.duplicated(subset=security_keys).any():
        raise ValueError(
            "Duplicate Active Share security keys detected."
        )

    if not work["weight_basis"].isin(
        ["bop", "eop"]
    ).all():
        raise ValueError(
            "Unsupported weight_basis in Active Share input."
        )

    sector_labels = (
        work["sector"]
        .astype("string")
        .str.strip()
    )

    if (
        sector_labels.isna().any()
        or sector_labels.eq("").any()
    ):
        raise ValueError(
            "Missing or blank Active Share sector labels."
        )

    # --------------------------------------------------------
    # C. Validate financial weight relationships
    # --------------------------------------------------------

    numeric_columns = [
        "portfolio_weight",
        "benchmark_weight",
        "active_weight",
    ]

    for column in numeric_columns:

        if not pd.api.types.is_numeric_dtype(
            work[column]
        ):
            raise TypeError(
                f"{column} must have a numeric dtype."
            )

        values = work[column].to_numpy(
            dtype=np.float64,
            na_value=np.nan,
        )

        if not np.isfinite(values).all():
            raise ValueError(
                f"{column} contains nonfinite values."
            )

    for column in (
        "portfolio_weight",
        "benchmark_weight",
    ):

        values = work[column].to_numpy(
            dtype=np.float64
        )

        if (
            (values < 0).any()
            or (values > 1 + weight_tolerance).any()
        ):
            raise ValueError(
                f"{column} violates long-only weight bounds."
            )

    expected_active_weights = (
        work["portfolio_weight"]
        - work["benchmark_weight"]
    )

    active_weight_errors = (
        work["active_weight"]
        .sub(expected_active_weights)
        .abs()
    )

    if active_weight_errors.max() > weight_tolerance:
        raise ValueError(
            "active_weight does not equal portfolio_weight "
            "minus benchmark_weight."
        )

    # --------------------------------------------------------
    # D. Calculate security-level active-weight components
    # --------------------------------------------------------

    work["abs_active_weight"] = (
        work["active_weight"].abs()
    )

    work["positive_active_weight"] = (
        work["active_weight"].clip(lower=0)
    )

    work["negative_active_weight_abs"] = (
        (-work["active_weight"]).clip(lower=0)
    )

    work["common_weight"] = np.minimum(
        work["portfolio_weight"],
        work["benchmark_weight"],
    )

    security_stats = (
        work
        .groupby(group_keys, as_index=False)
        .agg(
            portfolio_weight_sum=(
                "portfolio_weight", "sum"
            ),
            benchmark_weight_sum=(
                "benchmark_weight", "sum"
            ),
            net_active_weight=(
                "active_weight", "sum"
            ),
            abs_active_weight_sum=(
                "abs_active_weight", "sum"
            ),
            security_positive_active_weight=(
                "positive_active_weight", "sum"
            ),
            security_negative_active_weight_abs=(
                "negative_active_weight_abs", "sum"
            ),
            common_weight_overlap=(
                "common_weight", "sum"
            ),
        )
    )

    security_stats["security_active_share"] = (
        0.5 * security_stats["abs_active_weight_sum"]
    )

    security_stats = security_stats.drop(
        columns="abs_active_weight_sum"
    )

    # --------------------------------------------------------
    # E. Calculate sector-level active-weight components
    #    directly from aligned security exposures
    # --------------------------------------------------------

    sector_weights = (
        work
        .groupby(
            group_keys + ["sector"],
            as_index=False,
        )["active_weight"]
        .sum()
    )

    sector_weights["abs_sector_active_weight"] = (
        sector_weights["active_weight"].abs()
    )

    sector_weights["positive_sector_active_weight"] = (
        sector_weights["active_weight"].clip(lower=0)
    )

    sector_weights["negative_sector_active_weight_abs"] = (
        (-sector_weights["active_weight"]).clip(lower=0)
    )

    sector_stats = (
        sector_weights
        .groupby(group_keys, as_index=False)
        .agg(
            abs_sector_active_weight_sum=(
                "abs_sector_active_weight", "sum"
            ),
            sector_positive_active_weight=(
                "positive_sector_active_weight", "sum"
            ),
            sector_negative_active_weight_abs=(
                "negative_sector_active_weight_abs", "sum"
            ),
        )
    )

    sector_stats["sector_active_share"] = (
        0.5 * sector_stats[
            "abs_sector_active_weight_sum"
        ]
    )

    sector_stats = sector_stats.drop(
        columns="abs_sector_active_weight_sum"
    )

    # --------------------------------------------------------
    # F. Construct standardized Active Share output
    # --------------------------------------------------------

    active_share = security_stats.merge(
        sector_stats,
        on=group_keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    if not active_share["_merge"].eq("both").all():
        raise ValueError(
            "Security and sector Active Share groups "
            "do not match."
        )

    active_share = active_share.drop(
        columns="_merge"
    )

    active_share["security_minus_sector_active_share"] = (
        active_share["security_active_share"]
        - active_share["sector_active_share"]
    )

    # --------------------------------------------------------
    # G. Validate general Active Share identities
    # --------------------------------------------------------

    hierarchy_error = (
        active_share["sector_active_share"]
        - active_share["security_active_share"]
    )

    if (hierarchy_error > weight_tolerance).any():
        raise ValueError(
            "Sector Active Share exceeds security Active "
            "Share beyond the permitted tolerance."
        )

    for label in ("security", "sector"):

        share_column = f"{label}_active_share"

        positive_column = (
            f"{label}_positive_active_weight"
        )

        negative_column = (
            f"{label}_negative_active_weight_abs"
        )

        expected_half_l1 = (
            0.5 * (
                active_share[positive_column]
                + active_share[negative_column]
            )
        )

        if (
            active_share[share_column]
            .sub(expected_half_l1)
            .abs()
            .max()
            > weight_tolerance
        ):
            raise ValueError(
                f"{label} Active Share half-L1 identity failed."
            )

    # --------------------------------------------------------
    # H. Validate fully invested portfolio identities
    # --------------------------------------------------------

    if require_fully_invested:

        portfolio_errors = (
            active_share["portfolio_weight_sum"]
            .sub(1.0)
            .abs()
        )

        benchmark_errors = (
            active_share["benchmark_weight_sum"]
            .sub(1.0)
            .abs()
        )

        net_active_errors = (
            active_share["net_active_weight"].abs()
        )

        max_investment_error = max(
            portfolio_errors.max(),
            benchmark_errors.max(),
            net_active_errors.max(),
        )

        if max_investment_error > weight_tolerance:
            raise ValueError(
                "Active Share requires fully invested "
                "portfolio and benchmark weights. "
                f"Maximum error: {max_investment_error}"
            )

        for label in ("security", "sector"):

            share = active_share[
                f"{label}_active_share"
            ]

            positive = active_share[
                f"{label}_positive_active_weight"
            ]

            negative = active_share[
                f"{label}_negative_active_weight_abs"
            ]

            if (
                share.sub(positive).abs().max()
                > weight_tolerance
            ):
                raise ValueError(
                    f"{label} positive active-weight "
                    "identity failed."
                )

            if (
                share.sub(negative).abs().max()
                > weight_tolerance
            ):
                raise ValueError(
                    f"{label} negative active-weight "
                    "identity failed."
                )

            if (
                (share < -weight_tolerance).any()
                or (share > 1 + weight_tolerance).any()
            ):
                raise ValueError(
                    f"{label} Active Share outside [0, 1]."
                )

        # For fully invested long-only portfolios:
        #
        # AS = 1 - sum(min(portfolio_weight,
        #                  benchmark_weight))

        overlap_identity_error = (
            active_share["security_active_share"]
            + active_share["common_weight_overlap"]
            - 1.0
        ).abs()

        if overlap_identity_error.max() > weight_tolerance:
            raise ValueError(
                "Security Active Share / common-weight "
                "overlap identity failed."
            )

    # --------------------------------------------------------
    # I. Validate output keys
    # --------------------------------------------------------

    if active_share[group_keys].isna().any().any():
        raise ValueError(
            "Active Share output keys contain null values."
        )

    if active_share.duplicated(
        subset=group_keys
    ).any():
        raise ValueError(
            "Duplicate Active Share output keys detected."
        )

    return (
        active_share
        .sort_values(group_keys, kind="stable")
        .reset_index(drop=True)
    )
