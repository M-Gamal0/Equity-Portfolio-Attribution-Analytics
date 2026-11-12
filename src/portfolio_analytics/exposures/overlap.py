"""Exposures overlap: validated research calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd



# Research reference: 10_holdings_exposure_analytics.ipynb, code cell index 9.
def calculate_holdings_overlap(
    active_security_exposures: pd.DataFrame,
    *,
    require_fully_invested: bool = False,
    weight_tolerance: float = 1e-10,
) -> pd.DataFrame:
    """
    Calculate constituent and capital-weight overlap.

    Required input grain:
        portfolio_id, benchmark_portfolio_id,
        date, weight_basis, security_id

    Output grain:
        portfolio_id, benchmark_portfolio_id,
        date, weight_basis

    Methodology
    -----------
    Invested constituents:
        Securities with strictly positive selected weights.

    Shared holdings:
        Positive portfolio and benchmark weights.

    Off-benchmark holdings:
        Positive portfolio weight and zero benchmark weight.

    Benchmark-only holdings:
        Positive benchmark weight and zero portfolio weight.

    Jaccard overlap:
        shared_security_count / union_security_count

    Common-weight overlap:
        sum(min(portfolio_weight, benchmark_weight))

    For fully invested portfolios:
        common_weight_overlap = 1 - security_active_share

    Notes
    -----
    Missing security weights must already have been
    resolved by a validated security-level outer join.

    Recorded zero-weight positions do not count as
    invested constituents.

    Count-based ratios are unavailable (NaN) when their
    denominator is zero.

    The input DataFrame is not modified.
    """

    # --------------------------------------------------------
    # A. Validate input structure and parameters
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
            "Invalid weight_tolerance."
        )

    group_keys = [
        "portfolio_id",
        "benchmark_portfolio_id",
        "date",
        "weight_basis",
    ]

    security_keys = group_keys + ["security_id"]

    required_columns = set(security_keys) | {
        "portfolio_weight",
        "benchmark_weight",
        "active_weight",
        "portfolio_record_present",
        "benchmark_record_present",
        "portfolio_is_held",
        "benchmark_is_held",
    }

    missing_columns = (
        required_columns
        - set(active_security_exposures.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing holdings overlap input columns: "
            f"{sorted(missing_columns)}"
        )

    if active_security_exposures.empty:
        raise ValueError(
            "Active security exposures cannot be empty."
        )

    optional_flags = [
        "shared_holding",
        "off_benchmark_holding",
        "benchmark_only_holding",
    ]

    selected_columns = (
        security_keys
        + [
            "portfolio_weight",
            "benchmark_weight",
            "active_weight",
            "portfolio_record_present",
            "benchmark_record_present",
            "portfolio_is_held",
            "benchmark_is_held",
        ]
        + [
            column
            for column in optional_flags
            if column in active_security_exposures.columns
        ]
    )

    work = active_security_exposures.loc[
        :, selected_columns
    ].copy()

    # --------------------------------------------------------
    # B. Validate input keys and observation conventions
    # --------------------------------------------------------

    work["date"] = pd.to_datetime(
        work["date"],
        errors="raise",
    )

    if work[security_keys].isna().any().any():
        raise ValueError(
            "Holdings overlap keys contain null values."
        )

    if work.duplicated(subset=security_keys).any():
        raise ValueError(
            "Duplicate holdings overlap security keys."
        )

    if not work["weight_basis"].isin(
        ["bop", "eop"]
    ).all():
        raise ValueError(
            "Unsupported holdings overlap weight_basis."
        )

    if work[
        "portfolio_id"
    ].eq(work["benchmark_portfolio_id"]).any():
        raise ValueError(
            "Portfolio and benchmark identities must differ."
        )

    # --------------------------------------------------------
    # C. Validate security weights
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

    active_identity_error = (
        work["active_weight"]
        .sub(
            work["portfolio_weight"]
            - work["benchmark_weight"]
        )
        .abs()
        .max()
    )

    if active_identity_error > weight_tolerance:
        raise ValueError(
            "Active weights do not reconcile to "
            "portfolio minus benchmark weights."
        )

    # --------------------------------------------------------
    # D. Validate ledger presence and invested flags
    # --------------------------------------------------------

    boolean_columns = [
        "portfolio_record_present",
        "benchmark_record_present",
        "portfolio_is_held",
        "benchmark_is_held",
    ]

    for column in boolean_columns:

        if (
            not pd.api.types.is_bool_dtype(work[column])
            or work[column].isna().any()
        ):
            raise TypeError(
                f"{column} must contain nonmissing "
                "Boolean values."
            )

    if (
        ~work["portfolio_record_present"]
        & ~work["benchmark_record_present"]
    ).any():
        raise ValueError(
            "An aligned security row cannot be absent "
            "from both underlying ledgers."
        )

    for role in ("portfolio", "benchmark"):

        weight_column = f"{role}_weight"
        held_column = f"{role}_is_held"
        record_column = f"{role}_record_present"

        expected_held = (
            work[weight_column] > 0
        )

        if not work[held_column].eq(
            expected_held
        ).all():
            raise ValueError(
                f"{held_column} flags do not match "
                "positive security weights."
            )

        invalid_absence = (
            ~work[record_column]
            & work[weight_column].ne(0)
        )

        if invalid_absence.any():
            raise ValueError(
                f"Positive or nonzero {role} weight "
                "without a ledger record."
            )

    # --------------------------------------------------------
    # E. Derive mutually exclusive position classifications
    # --------------------------------------------------------

    derived_flags = {
        "shared_holding": (
            work["portfolio_is_held"]
            & work["benchmark_is_held"]
        ),
        "off_benchmark_holding": (
            work["portfolio_is_held"]
            & ~work["benchmark_is_held"]
        ),
        "benchmark_only_holding": (
            work["benchmark_is_held"]
            & ~work["portfolio_is_held"]
        ),
    }

    for column, expected in derived_flags.items():

        if column in work.columns:

            if (
                not pd.api.types.is_bool_dtype(work[column])
                or work[column].isna().any()
            ):
                raise TypeError(
                    f"{column} must contain nonmissing "
                    "Boolean values."
                )

            if not work[column].eq(expected).all():
                raise ValueError(
                    f"{column} classification is inconsistent "
                    "with positive security weights."
                )

        work[column] = expected

    work["union_holding"] = (
        work["portfolio_is_held"]
        | work["benchmark_is_held"]
    )

    work["portfolio_recorded_zero"] = (
        work["portfolio_record_present"]
        & ~work["portfolio_is_held"]
    )

    work["benchmark_recorded_zero"] = (
        work["benchmark_record_present"]
        & ~work["benchmark_is_held"]
    )

    # --------------------------------------------------------
    # F. Calculate capital-weight overlap components
    # --------------------------------------------------------

    work["common_security_weight"] = np.minimum(
        work["portfolio_weight"],
        work["benchmark_weight"],
    )

    work["portfolio_shared_weight"] = (
        work["portfolio_weight"].where(
            work["shared_holding"], 0.0
        )
    )

    work["benchmark_shared_weight"] = (
        work["benchmark_weight"].where(
            work["shared_holding"], 0.0
        )
    )

    work["off_benchmark_weight"] = (
        work["portfolio_weight"].where(
            work["off_benchmark_holding"], 0.0
        )
    )

    work["excluded_benchmark_weight"] = (
        work["benchmark_weight"].where(
            work["benchmark_only_holding"], 0.0
        )
    )

    work["absolute_active_weight"] = (
        work["active_weight"].abs()
    )

    # --------------------------------------------------------
    # G. Aggregate daily constituent and weight measures
    # --------------------------------------------------------

    overlap = (
        work
        .groupby(group_keys, as_index=False)
        .agg(
            portfolio_security_count=(
                "portfolio_is_held", "sum"
            ),
            benchmark_security_count=(
                "benchmark_is_held", "sum"
            ),
            shared_security_count=(
                "shared_holding", "sum"
            ),
            off_benchmark_security_count=(
                "off_benchmark_holding", "sum"
            ),
            benchmark_only_security_count=(
                "benchmark_only_holding", "sum"
            ),
            union_security_count=(
                "union_holding", "sum"
            ),
            portfolio_recorded_zero_count=(
                "portfolio_recorded_zero", "sum"
            ),
            benchmark_recorded_zero_count=(
                "benchmark_recorded_zero", "sum"
            ),
            portfolio_weight_sum=(
                "portfolio_weight", "sum"
            ),
            benchmark_weight_sum=(
                "benchmark_weight", "sum"
            ),
            portfolio_shared_weight=(
                "portfolio_shared_weight", "sum"
            ),
            benchmark_shared_weight=(
                "benchmark_shared_weight", "sum"
            ),
            off_benchmark_weight=(
                "off_benchmark_weight", "sum"
            ),
            excluded_benchmark_weight=(
                "excluded_benchmark_weight", "sum"
            ),
            common_weight_overlap=(
                "common_security_weight", "sum"
            ),
            absolute_active_weight_sum=(
                "absolute_active_weight", "sum"
            ),
        )
    )

    # --------------------------------------------------------
    # H. Calculate count-based overlap ratios
    # --------------------------------------------------------

    overlap["jaccard_constituent_overlap"] = np.nan

    overlap["active_constituent_coverage"] = np.nan

    overlap["benchmark_constituent_coverage"] = np.nan

    has_union = (
        overlap["union_security_count"] > 0
    )

    has_active = (
        overlap["portfolio_security_count"] > 0
    )

    has_benchmark = (
        overlap["benchmark_security_count"] > 0
    )

    overlap.loc[
        has_union, "jaccard_constituent_overlap"
    ] = (
        overlap.loc[
            has_union, "shared_security_count"
        ]
        / overlap.loc[
            has_union, "union_security_count"
        ]
    )

    overlap.loc[
        has_active, "active_constituent_coverage"
    ] = (
        overlap.loc[
            has_active, "shared_security_count"
        ]
        / overlap.loc[
            has_active, "portfolio_security_count"
        ]
    )

    overlap.loc[
        has_benchmark, "benchmark_constituent_coverage"
    ] = (
        overlap.loc[
            has_benchmark, "shared_security_count"
        ]
        / overlap.loc[
            has_benchmark, "benchmark_security_count"
        ]
    )

    # --------------------------------------------------------
    # I. Calculate independent security Active Share
    # --------------------------------------------------------

    overlap["security_active_share_from_weights"] = (
        0.5 * overlap["absolute_active_weight_sum"]
    )

    overlap = overlap.drop(
        columns="absolute_active_weight_sum"
    )

    # This identity holds for long-only portfolios
    # regardless of whether they are fully invested.

    overlap_identity_error = (
        overlap["security_active_share_from_weights"]
        + overlap["common_weight_overlap"]
        - 0.5 * (
            overlap["portfolio_weight_sum"]
            + overlap["benchmark_weight_sum"]
        )
    ).abs()

    if overlap_identity_error.max() > weight_tolerance:
        raise ValueError(
            "General capital-weight overlap identity failed."
        )

    # --------------------------------------------------------
    # J. Validate constituent partition identities
    # --------------------------------------------------------

    count_identities = [
        (
            overlap["portfolio_security_count"],
            overlap["shared_security_count"]
            + overlap["off_benchmark_security_count"],
        ),
        (
            overlap["benchmark_security_count"],
            overlap["shared_security_count"]
            + overlap["benchmark_only_security_count"],
        ),
        (
            overlap["union_security_count"],
            overlap["shared_security_count"]
            + overlap["off_benchmark_security_count"]
            + overlap["benchmark_only_security_count"],
        ),
    ]

    for observed, expected in count_identities:

        if not observed.eq(expected).all():
            raise ValueError(
                "Constituent overlap count identity failed."
            )

    # --------------------------------------------------------
    # K. Validate capital-weight partition identities
    # --------------------------------------------------------

    portfolio_partition_error = (
        overlap["portfolio_shared_weight"]
        + overlap["off_benchmark_weight"]
        - overlap["portfolio_weight_sum"]
    ).abs()

    benchmark_partition_error = (
        overlap["benchmark_shared_weight"]
        + overlap["excluded_benchmark_weight"]
        - overlap["benchmark_weight_sum"]
    ).abs()

    if max(
        portfolio_partition_error.max(),
        benchmark_partition_error.max(),
    ) > weight_tolerance:
        raise ValueError(
            "Shared/exclusive capital-weight partition failed."
        )

    for column in (
        "portfolio_weight_sum",
        "benchmark_weight_sum",
    ):

        if (
            overlap[column] > 1 + weight_tolerance
        ).any():
            raise ValueError(
                f"{column} exceeds full portfolio capital."
            )

    # --------------------------------------------------------
    # L. Optional full-investment identities
    # --------------------------------------------------------

    if require_fully_invested:

        investment_error = max(
            overlap["portfolio_weight_sum"]
            .sub(1.0)
            .abs()
            .max(),
            overlap["benchmark_weight_sum"]
            .sub(1.0)
            .abs()
            .max(),
        )

        if investment_error > weight_tolerance:
            raise ValueError(
                "Portfolio or benchmark is not fully invested."
            )

        fully_invested_overlap_error = (
            overlap["common_weight_overlap"]
            + overlap["security_active_share_from_weights"]
            - 1.0
        ).abs()

        if (
            fully_invested_overlap_error.max()
            > weight_tolerance
        ):
            raise ValueError(
                "Full-investment Active Share/overlap "
                "identity failed."
            )

    # --------------------------------------------------------
    # M. Validate output grain
    # --------------------------------------------------------

    if overlap[group_keys].isna().any().any():
        raise ValueError(
            "Holdings overlap output keys contain nulls."
        )

    if overlap.duplicated(subset=group_keys).any():
        raise ValueError(
            "Duplicate holdings overlap output keys."
        )

    return (
        overlap
        .sort_values(group_keys, kind="stable")
        .reset_index(drop=True)
    )
