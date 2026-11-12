"""Turnover implementation: validated research calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd



# Research reference: 12_turnover_rebalancing_analytics.ipynb, code cell index 4.
def reconcile_rebalance_target_implementation(
    holdings_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    events_df: pd.DataFrame,
    *,
    weight_tolerance: float = 1e-10,
    require_sector_match: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reconcile scheduled targets against effective-date BOP weights.

    Parameters
    ----------
    holdings_df : pd.DataFrame
        Complete portfolio security-day holdings ledger with
        portfolio_id, date, security_id, and weight_bop.

    targets_df : pd.DataFrame
        Selection-date security target weights.

    events_df : pd.DataFrame
        Rebalance event schedule containing event classifications
        and effective dates.

    weight_tolerance : float
        Absolute tolerance for weight reconciliation.

    require_sector_match : bool
        Require saved target and actual sector labels to match.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        security_df:
            One row per security in the union of target and actual
            effective-date positions.

            Key: (portfolio_id, selection_date, security_id).

        event_df:
            One row per portfolio selection event.

            Key: (portfolio_id, selection_date).

    Notes
    -----
    Weights are decimal fractions.

    Absent positions are interpreted as zero exposure only after
    joining complete security-level portfolio ledgers.

    Missing recorded weights are never replaced with zero.

    Pending events remain unassessed.

    This is weight implementation reconciliation, not evidence
    of executed orders, fills, or transaction costs.
    """

    if not all(
        isinstance(frame, pd.DataFrame)
        for frame in (holdings_df, targets_df, events_df)
    ):
        raise TypeError(
            "All reconciliation inputs must be pandas DataFrames."
        )

    if (
        not np.isfinite(weight_tolerance)
        or weight_tolerance < 0
    ):
        raise ValueError(
            "weight_tolerance must be finite and nonnegative."
        )

    required_columns = {
        "holdings": {
            "portfolio_id",
            "date",
            "security_id",
            "weight_bop",
        },
        "targets": {
            "portfolio_id",
            "selection_date",
            "security_id",
            "weight_target",
        },
        "events": {
            "portfolio_id",
            "selection_date",
            "effective_rebalance_date",
            "event_status",
        },
    }

    input_frames = {
        "holdings": holdings_df,
        "targets": targets_df,
        "events": events_df,
    }

    for name, frame in input_frames.items():

        missing = required_columns[name].difference(
            frame.columns
        )

        if missing:
            raise ValueError(
                f"{name}: missing required fields "
                f"{sorted(missing)}"
            )

    if require_sector_match:

        for name, frame in (
            ("holdings", holdings_df),
            ("targets", targets_df),
        ):

            if "sector" not in frame.columns:
                raise ValueError(
                    f"{name}: sector required for reconciliation."
                )

    # --------------------------------------------------------
    # 2. Verify unique keys and required numeric values
    # --------------------------------------------------------

    key_definitions = {
        "holdings": [
            "portfolio_id",
            "date",
            "security_id",
        ],
        "targets": [
            "portfolio_id",
            "selection_date",
            "security_id",
        ],
        "events": [
            "portfolio_id",
            "selection_date",
        ],
    }

    for name, keys in key_definitions.items():

        frame = input_frames[name]

        if frame[keys].isna().any().any():
            raise ValueError(
                f"{name}: null reconciliation keys."
            )

        if frame.duplicated(keys).any():
            raise ValueError(
                f"{name}: duplicate reconciliation keys."
            )

    for name, column in (
        ("holdings", "weight_bop"),
        ("targets", "weight_target"),
    ):

        values = pd.to_numeric(
            input_frames[name][column],
            errors="raise",
        ).to_numpy(
            dtype=float,
            na_value=np.nan,
        )

        if not np.isfinite(values).all():
            raise ValueError(
                f"{name}.{column}: nonfinite weights."
            )

        if (
            (values < -weight_tolerance)
            | (values > 1.0 + weight_tolerance)
        ).any():
            raise ValueError(
                f"{name}.{column}: invalid long-only weights."
            )

    # --------------------------------------------------------
    # 3. Identify selection events eligible for reconciliation
    # --------------------------------------------------------

    effective_statuses = {
        "INITIAL_ESTABLISHMENT",
        "EFFECTIVE_REBALANCE",
    }

    effective_events = (
        events_df.loc[
            events_df["event_status"].isin(
                effective_statuses
            ),
            [
                "portfolio_id",
                "selection_date",
                "effective_rebalance_date",
            ],
        ]
        .copy()
    )

    if effective_events[
        "effective_rebalance_date"
    ].isna().any():
        raise ValueError(
            "Implemented events cannot have missing effective dates."
        )

    if effective_events.duplicated(
        ["portfolio_id", "effective_rebalance_date"]
    ).any():
        raise ValueError(
            "Multiple implemented events share an effective date."
        )

    # Confirm that the event schedule describes the supplied
    # target-selection universe.

    target_event_keys = pd.MultiIndex.from_frame(
        targets_df[
            ["portfolio_id", "selection_date"]
        ].drop_duplicates()
    )

    schedule_event_keys = pd.MultiIndex.from_frame(
        events_df[
            ["portfolio_id", "selection_date"]
        ]
    )

    if not target_event_keys.isin(
        schedule_event_keys
    ).all():
        raise ValueError(
            "Targets contain selections absent from the event schedule."
        )

    if not schedule_event_keys.isin(
        target_event_keys
    ).all():
        raise ValueError(
            "Event schedule contains selections absent from targets."
        )

    # --------------------------------------------------------
    # 4. Attach effective dates to target security positions
    # --------------------------------------------------------

    target_columns = [
        "portfolio_id",
        "selection_date",
        "security_id",
        "weight_target",
    ]

    if "sector" in targets_df.columns:
        target_columns.append("sector")

    effective_targets = (
        targets_df[target_columns]
        .merge(
            effective_events,
            on=["portfolio_id", "selection_date"],
            how="inner",
            validate="many_to_one",
        )
        .rename(
            columns={"sector": "target_sector"}
        )
    )

    # --------------------------------------------------------
    # 5. Attach event identities to actual BOP holdings
    # --------------------------------------------------------

    actual_columns = [
        "portfolio_id",
        "date",
        "security_id",
        "weight_bop",
    ]

    if "sector" in holdings_df.columns:
        actual_columns.append("sector")

    effective_holdings = (
        holdings_df[actual_columns]
        .rename(
            columns={
                "date": "effective_rebalance_date",
                "weight_bop": "actual_weight_bop",
                "sector": "actual_sector",
            }
        )
        .merge(
            effective_events,
            on=[
                "portfolio_id",
                "effective_rebalance_date",
            ],
            how="inner",
            validate="many_to_one",
        )
    )

    # --------------------------------------------------------
    # 6. Outer-join all target and actual security positions
    # --------------------------------------------------------

    security_keys = [
        "portfolio_id",
        "selection_date",
        "effective_rebalance_date",
        "security_id",
    ]

    security_df = effective_targets.merge(
        effective_holdings,
        on=security_keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    security_df["position_match_status"] = (
        security_df["_merge"]
        .map(
            {
                "both": "MATCHED",
                "left_only": "TARGET_ONLY",
                "right_only": "ACTUAL_ONLY",
            }
        )
        .astype("string")
    )

    # Zero here represents an absent position, not a missing
    # numeric observation in an existing source record.

    security_df["target_weight"] = (
        security_df["weight_target"].fillna(0.0)
    )

    security_df["actual_weight"] = (
        security_df["actual_weight_bop"].fillna(0.0)
    )

    security_df["weight_error"] = (
        security_df["actual_weight"]
        - security_df["target_weight"]
    )

    security_df["abs_weight_error"] = (
        security_df["weight_error"].abs()
    )

    security_df["weight_reconciled"] = (
        security_df["abs_weight_error"]
        <= weight_tolerance
    )

    security_df["missing_positive_target"] = (
        security_df["position_match_status"].eq(
            "TARGET_ONLY"
        )
        & security_df["target_weight"].gt(
            weight_tolerance
        )
    )

    security_df["unexpected_positive_actual"] = (
        security_df["position_match_status"].eq(
            "ACTUAL_ONLY"
        )
        & security_df["actual_weight"].gt(
            weight_tolerance
        )
    )

    # --------------------------------------------------------
    # 7. Reconcile sector labels when explicitly requested
    # --------------------------------------------------------

    if require_sector_match:

        matched_positions = (
            security_df["position_match_status"]
            .eq("MATCHED")
        )

        security_df["sector_mismatch"] = (
            matched_positions
            & security_df["target_sector"]
                .fillna("__MISSING_TARGET_SECTOR__")
                .ne(
                    security_df["actual_sector"]
                    .fillna("__MISSING_ACTUAL_SECTOR__")
                )
        )

        security_df["sector"] = (
            security_df["target_sector"]
            .combine_first(
                security_df["actual_sector"]
            )
        )

    else:

        security_df["sector_mismatch"] = False

        if "target_sector" in security_df.columns:
            security_df["sector"] = (
                security_df["target_sector"]
            )

        elif "actual_sector" in security_df.columns:
            security_df["sector"] = (
                security_df["actual_sector"]
            )

        else:
            security_df["sector"] = pd.NA

    # --------------------------------------------------------
    # 8. Aggregate security discrepancies by selection event
    # --------------------------------------------------------

    event_keys = [
        "portfolio_id",
        "selection_date",
    ]

    event_metrics = (
        security_df
        .groupby(event_keys, as_index=False)
        .agg(
            compared_security_rows=(
                "security_id",
                "size",
            ),
            target_weight_sum=(
                "target_weight",
                "sum",
            ),
            actual_weight_sum=(
                "actual_weight",
                "sum",
            ),
            max_abs_weight_error=(
                "abs_weight_error",
                "max",
            ),
            total_abs_weight_error=(
                "abs_weight_error",
                "sum",
            ),
            missing_positive_targets=(
                "missing_positive_target",
                "sum",
            ),
            unexpected_positive_actuals=(
                "unexpected_positive_actual",
                "sum",
            ),
            sector_mismatches=(
                "sector_mismatch",
                "sum",
            ),
        )
    )

    # --------------------------------------------------------
    # 9. Preserve all scheduled events, including pending
    # --------------------------------------------------------

    event_df = (
        events_df[
            [
                "portfolio_id",
                "selection_date",
                "effective_rebalance_date",
                "event_status",
            ]
        ]
        .merge(
            event_metrics,
            on=event_keys,
            how="left",
            validate="one_to_one",
        )
    )

    is_effective = event_df[
        "event_status"
    ].isin(effective_statuses)

    target_sum_passed = (
        event_df["target_weight_sum"]
        .sub(1.0)
        .abs()
        .le(weight_tolerance)
    )

    actual_sum_passed = (
        event_df["actual_weight_sum"]
        .sub(1.0)
        .abs()
        .le(weight_tolerance)
    )

    security_weights_passed = (
        event_df["max_abs_weight_error"]
        .le(weight_tolerance)
    )

    position_coverage_passed = (
        event_df["missing_positive_targets"].eq(0)
        & event_df["unexpected_positive_actuals"].eq(0)
    )

    if require_sector_match:
        sector_passed = event_df[
            "sector_mismatches"
        ].eq(0)

    else:
        sector_passed = pd.Series(
            True,
            index=event_df.index,
        )

    reconciled = (
        target_sum_passed
        & actual_sum_passed
        & security_weights_passed
        & position_coverage_passed
        & sector_passed
    ).fillna(False)

    event_df["reconciliation_status"] = (
        "NOT_ASSESSED"
    )

    event_df.loc[
        event_df["event_status"].eq(
            "PENDING_AFTER_CUTOFF"
        ),
        "reconciliation_status",
    ] = "NOT_ASSESSED_PENDING_SELECTION"

    event_df.loc[
        is_effective,
        "reconciliation_status",
    ] = np.where(
        reconciled.loc[is_effective],
        "PASSED",
        "FAILED",
    )

    if not require_sector_match:
        event_df["sector_alignment_status"] = (
            "NOT_ASSESSED"
        )
    else:
        event_df["sector_alignment_status"] = (
            "NOT_ASSESSED"
        )
        event_df.loc[
            is_effective,
            "sector_alignment_status",
        ] = np.where(
            sector_passed.loc[is_effective],
            "PASSED",
            "FAILED",
        )

    # --------------------------------------------------------
    # 10. Return stable, sorted analytical output contracts
    # --------------------------------------------------------

    security_df = (
        security_df
        .drop(
            columns=[
                "_merge",
                "weight_target",
                "actual_weight_bop",
            ]
        )
        .sort_values(
            [
                "portfolio_id",
                "selection_date",
                "security_id",
            ]
        )
        .reset_index(drop=True)
    )

    event_df = (
        event_df
        .sort_values(
            ["portfolio_id", "selection_date"]
        )
        .reset_index(drop=True)
    )

    return security_df, event_df
