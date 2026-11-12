"""Exposures changes: validated research calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd



# Research reference: 10_holdings_exposure_analytics.ipynb, code cell index 10.
def calculate_exposure_changes(
    security_exposures: pd.DataFrame,
    *,
    require_fully_invested: bool = False,
    require_fixed_sector_history: bool = False,
    weight_tolerance: float = 1e-10,
) -> pd.DataFrame:
    """
    Calculate changes between consecutive observed trading dates.

    Required input grain:
        portfolio_id, date, weight_basis, security_id

    Output grain:
        portfolio_id, date, weight_basis, security_id

    Method:
        weight_change = current_weight - previous_weight

    Previous date:
        Immediately preceding observed trading date for
        the same portfolio and weight basis.

    Position transitions:
        continued:
            Positive weight on both dates.

        entered:
            Zero previous weight, positive current weight.

        exited:
            Positive previous weight, zero current weight.

        uninvested_both:
            Zero weight on both dates, with at least one
            underlying ledger record.

    Missing positions:
        An absent security on an otherwise valid observed
        portfolio date is assigned zero weight.

    Important:
        Entered/exited classifications describe invested
        position status, not verified trading activity.

        The first available date has no previous observation
        and is excluded from the output.

        BOP and EOP are compared independently.

    The input DataFrame is not modified.
    """

    # --------------------------------------------------------
    # A. Validate parameters and required columns
    # --------------------------------------------------------

    if not isinstance(security_exposures, pd.DataFrame):
        raise TypeError(
            "security_exposures must be a DataFrame."
        )

    if (
        not np.isfinite(weight_tolerance)
        or not 0 < weight_tolerance < 1
    ):
        raise ValueError(
            "Invalid weight_tolerance."
        )

    key_columns = [
        "portfolio_id",
        "date",
        "weight_basis",
        "security_id",
    ]

    required_columns = set(key_columns) | {
        "sector",
        "security_weight",
        "is_held",
    }

    missing_columns = (
        required_columns - set(security_exposures.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing exposure change fields: "
            f"{sorted(missing_columns)}"
        )

    if security_exposures.empty:
        raise ValueError(
            "Security exposures cannot be empty."
        )

    selected_columns = key_columns + [
        "sector",
        "security_weight",
        "is_held",
    ]

    if "ticker" in security_exposures.columns:
        selected_columns.append("ticker")

    work = security_exposures.loc[
        :, selected_columns
    ].copy()

    # --------------------------------------------------------
    # B. Validate keys, dates, and weights
    # --------------------------------------------------------

    work["date"] = pd.to_datetime(
        work["date"],
        errors="raise",
    )

    if work[key_columns].isna().any().any():
        raise ValueError(
            "Exposure change keys contain null values."
        )

    if work.duplicated(subset=key_columns).any():
        raise ValueError(
            "Duplicate exposure change security keys."
        )

    if not work["weight_basis"].isin(
        ["bop", "eop"]
    ).all():
        raise ValueError(
            "Unsupported exposure weight_basis."
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
            "Missing or blank sector classifications."
        )

    if not pd.api.types.is_numeric_dtype(
        work["security_weight"]
    ):
        raise TypeError(
            "security_weight must be numeric."
        )

    weights = work["security_weight"].to_numpy(
        dtype=np.float64,
        na_value=np.nan,
    )

    if (
        not np.isfinite(weights).all()
        or (weights < 0).any()
        or (weights > 1 + weight_tolerance).any()
    ):
        raise ValueError(
            "Nonfinite or invalid long-only weights."
        )

    if (
        not pd.api.types.is_bool_dtype(work["is_held"])
        or work["is_held"].isna().any()
    ):
        raise TypeError(
            "is_held must be a nonmissing Boolean."
        )

    if not work["is_held"].eq(
        work["security_weight"] > 0
    ).all():
        raise ValueError(
            "is_held flags disagree with positive weights."
        )

    # --------------------------------------------------------
    # C. Validate fixed-sector-history policy
    # --------------------------------------------------------

    if require_fixed_sector_history:

        sector_counts = (
            work.groupby("security_id")["sector"].nunique()
        )

        if (sector_counts > 1).any():
            raise ValueError(
                "Inconsistent fixed-for-history "
                "sector classifications."
            )

    # --------------------------------------------------------
    # D. Identify consecutive observed trading dates
    # --------------------------------------------------------

    calendar_keys = [
        "portfolio_id",
        "weight_basis",
    ]

    calendar = (
        work[calendar_keys + ["date"]]
        .drop_duplicates()
        .sort_values(
            calendar_keys + ["date"],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    if (
        calendar
        .groupby(calendar_keys)
        .size()
        .lt(2)
        .any()
    ):
        raise ValueError(
            "Each portfolio/basis requires at least "
            "two observed dates."
        )

    calendar["previous_date"] = (
        calendar
        .groupby(calendar_keys)["date"]
        .shift(1)
    )

    date_pairs = (
        calendar
        .dropna(subset=["previous_date"])
        .copy()
    )

    # --------------------------------------------------------
    # E. Align current and previous observed security records
    # --------------------------------------------------------

    current_rows = work.merge(
        date_pairs,
        on=calendar_keys + ["date"],
        how="inner",
        validate="many_to_one",
    )

    previous_rows = (
        work
        .rename(columns={"date": "previous_date"})
        .merge(
            date_pairs,
            on=calendar_keys + ["previous_date"],
            how="inner",
            validate="many_to_one",
        )
    )

    join_keys = calendar_keys + [
        "date",
        "previous_date",
        "security_id",
    ]

    joined = previous_rows.merge(
        current_rows,
        on=join_keys,
        how="outer",
        suffixes=("_previous", "_current"),
        validate="one_to_one",
        indicator=True,
    )

    previous_present = (
        joined["_merge"].ne("right_only")
    )

    current_present = (
        joined["_merge"].ne("left_only")
    )

    # --------------------------------------------------------
    # F. Preserve sector classification and ticker lineage
    # --------------------------------------------------------

    shared_records = (
        previous_present & current_present
    )

    if require_fixed_sector_history:

        inconsistent_sectors = (
            shared_records
            & joined["sector_previous"].ne(
                joined["sector_current"]
            )
        )

        if inconsistent_sectors.any():
            raise ValueError(
                "Historical sector mapping changed "
                "across dates."
            )

    # Current classification takes precedence only when
    # a non-fixed-sector policy has been explicitly allowed.

    # No sector mapping is inferred or externally replaced.

    resolved_sector = (
        joined["sector_current"]
        .combine_first(joined["sector_previous"])
    )

    # --------------------------------------------------------
    # G. Resolve genuine security absence to zero exposure
    # --------------------------------------------------------

    previous_weight = (
        joined["security_weight_previous"]
        .where(previous_present, 0.0)
    )

    current_weight = (
        joined["security_weight_current"]
        .where(current_present, 0.0)
    )

    previous_held = (
        previous_weight > 0
    )

    current_held = (
        current_weight > 0
    )

    # --------------------------------------------------------
    # H. Classify invested-position transitions
    # --------------------------------------------------------

    transition_conditions = [
        previous_held & current_held,
        ~previous_held & current_held,
        previous_held & ~current_held,
    ]

    transition_labels = [
        "continued",
        "entered",
        "exited",
    ]

    position_transition = np.select(
        transition_conditions,
        transition_labels,
        default="uninvested_both",
    )

    # --------------------------------------------------------
    # I. Construct standardized exposure-change output
    # --------------------------------------------------------

    changes = joined[join_keys].copy()

    changes["sector"] = resolved_sector

    if "ticker" in work.columns:

        changes["ticker"] = (
            joined["ticker_current"]
            .combine_first(
                joined["ticker_previous"]
            )
        )

    changes["previous_weight"] = previous_weight

    changes["current_weight"] = current_weight

    changes["weight_change"] = (
        current_weight - previous_weight
    )

    changes["previous_record_present"] = (
        previous_present
    )

    changes["current_record_present"] = (
        current_present
    )

    changes["previous_is_held"] = (
        previous_held
    )

    changes["current_is_held"] = (
        current_held
    )

    changes["position_transition"] = (
        position_transition
    )

    changes["weight_change_direction"] = np.select(
        [
            changes["weight_change"].gt(
                weight_tolerance
            ),
            changes["weight_change"].lt(
                -weight_tolerance
            ),
        ],
        [
            "increase",
            "decrease",
        ],
        default="unchanged",
    )

    changes["absolute_weight_change"] = (
        changes["weight_change"].abs()
    )

    # --------------------------------------------------------
    # J. Validate daily weight-change reconciliation
    # --------------------------------------------------------

    daily_keys = calendar_keys + [
        "date",
        "previous_date",
    ]

    daily_totals = (
        changes
        .groupby(daily_keys, as_index=False)
        .agg(
            previous_weight_sum=(
                "previous_weight", "sum"
            ),
            current_weight_sum=(
                "current_weight", "sum"
            ),
            net_weight_change=(
                "weight_change", "sum"
            ),
        )
    )

    delta_error = (
        daily_totals["net_weight_change"]
        - (
            daily_totals["current_weight_sum"]
            - daily_totals["previous_weight_sum"]
        )
    ).abs().max()

    if (
        not np.isfinite(delta_error)
        or delta_error > weight_tolerance
    ):
        raise ValueError(
            "Daily exposure changes do not reconcile."
        )

    # --------------------------------------------------------
    # K. Optional full-investment reconciliation
    # --------------------------------------------------------

    if require_fully_invested:

        maximum_error = max(
            daily_totals["previous_weight_sum"]
            .sub(1.0)
            .abs()
            .max(),
            daily_totals["current_weight_sum"]
            .sub(1.0)
            .abs()
            .max(),
            daily_totals["net_weight_change"]
            .abs()
            .max(),
        )

        if (
            not np.isfinite(maximum_error)
            or maximum_error > weight_tolerance
        ):
            raise ValueError(
                "Full-investment daily exposure-change "
                "reconciliation failed."
            )

    # --------------------------------------------------------
    # L. Validate output grain
    # --------------------------------------------------------

    output_keys = [
        "portfolio_id",
        "date",
        "weight_basis",
        "security_id",
    ]

    if changes[output_keys].isna().any().any():
        raise ValueError(
            "Exposure change output keys contain nulls."
        )

    if changes.duplicated(
        subset=output_keys
    ).any():
        raise ValueError(
            "Duplicate exposure change output keys."
        )

    return (
        changes
        .sort_values(
            output_keys,
            kind="stable",
        )
        .reset_index(drop=True)
    )
