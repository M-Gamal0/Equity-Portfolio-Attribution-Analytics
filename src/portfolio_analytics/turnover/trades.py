"""Turnover trades: validated research calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd



# Research reference: 12_turnover_rebalancing_analytics.ipynb, code cell index 5.
def calculate_scheduled_rebalance_turnover(
    holdings_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    events_df: pd.DataFrame,
    *,
    weight_tolerance: float = 1e-10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate security and portfolio scheduled-rebalance turnover.

    Parameters
    ----------
    holdings_df : pd.DataFrame
        Complete security-day holdings containing:
        portfolio_id, date, security_id, weight_bop, weight_eop.

    targets_df : pd.DataFrame
        Security-level rebalance targets containing:
        portfolio_id, selection_date, security_id, weight_target.

    events_df : pd.DataFrame
        Rebalance schedule containing:
        portfolio_id, selection_date, effective_rebalance_date,
        preceding_holdings_date, event_status.

    weight_tolerance : float
        Absolute tolerance for weight and reconciliation identities.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]

        security_df:
            One row per security in the union of previous
            holdings and target positions at each subsequent
            effective rebalance.

        event_df:
            One row per scheduled selection event, preserving
            unavailable initial and pending turnover observations.

    Notes
    -----
    Turnover is hypothetical, weight-implied rebalancing activity.

    A position absent from one complete security ledger receives
    zero exposure on that side of the comparison.

    Missing recorded numeric weights are errors, not zero weights.

    Target implementation is independently verified against
    effective-date BOP holdings.

    Initial establishment is excluded from ongoing turnover.
    """

    # --------------------------------------------------------
    # 2. Validate schemas and calculation settings
    # --------------------------------------------------------

    frames = {
        "holdings": holdings_df,
        "targets": targets_df,
        "events": events_df,
    }

    required_columns = {
        "holdings": {
            "portfolio_id",
            "date",
            "security_id",
            "weight_bop",
            "weight_eop",
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
            "preceding_holdings_date",
            "event_status",
        },
    }

    if (
        not np.isfinite(weight_tolerance)
        or weight_tolerance < 0
    ):
        raise ValueError(
            "weight_tolerance must be finite and nonnegative."
        )

    for name, frame in frames.items():

        if not isinstance(frame, pd.DataFrame):
            raise TypeError(
                f"{name} must be a pandas DataFrame."
            )

        if frame.empty:
            raise ValueError(
                f"{name} must not be empty."
            )

        missing = required_columns[name].difference(
            frame.columns
        )

        if missing:
            raise ValueError(
                f"{name}: missing columns {sorted(missing)}"
            )

    holdings = holdings_df[
        [
            "portfolio_id",
            "date",
            "security_id",
            "weight_bop",
            "weight_eop",
        ]
    ].copy()

    targets = targets_df[
        [
            "portfolio_id",
            "selection_date",
            "security_id",
            "weight_target",
        ]
    ].copy()

    events = events_df[
        [
            "portfolio_id",
            "selection_date",
            "effective_rebalance_date",
            "preceding_holdings_date",
            "event_status",
        ]
    ].copy()

    # --------------------------------------------------------
    # 3. Normalize dates and validate unique keys
    # --------------------------------------------------------

    holdings["date"] = pd.to_datetime(
        holdings["date"],
        errors="raise",
    )

    targets["selection_date"] = pd.to_datetime(
        targets["selection_date"],
        errors="raise",
    )

    for column in (
        "selection_date",
        "effective_rebalance_date",
        "preceding_holdings_date",
    ):

        events[column] = pd.to_datetime(
            events[column],
            errors="raise",
        )

    keys = {
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

    for name, frame in (
        ("holdings", holdings),
        ("targets", targets),
        ("events", events),
    ):

        if frame[keys[name]].isna().any().any():
            raise ValueError(
                f"{name}: null primary-key values."
            )

        if frame.duplicated(keys[name]).any():
            raise ValueError(
                f"{name}: duplicate primary keys."
            )

    # Every target selection must have exactly one event.

    target_event_keys = pd.MultiIndex.from_frame(
        targets[
            ["portfolio_id", "selection_date"]
        ].drop_duplicates()
    )

    schedule_event_keys = pd.MultiIndex.from_frame(
        events[
            ["portfolio_id", "selection_date"]
        ]
    )

    if (
        not target_event_keys.isin(schedule_event_keys).all()
        or not schedule_event_keys.isin(target_event_keys).all()
    ):
        raise ValueError(
            "Target selections and event schedule differ."
        )

    # --------------------------------------------------------
    # 4. Validate recorded numeric weights
    # --------------------------------------------------------

    for name, frame, columns in (
        (
            "holdings",
            holdings,
            ["weight_bop", "weight_eop"],
        ),
        (
            "targets",
            targets,
            ["weight_target"],
        ),
    ):

        for column in columns:

            if not pd.api.types.is_numeric_dtype(
                frame[column]
            ):
                raise ValueError(
                    f"{name}.{column} must be numeric."
                )

            values = frame[column].to_numpy(
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
                    f"{name}.{column}: invalid weight bounds."
                )

    # --------------------------------------------------------
    # 5. Select subsequent implemented rebalances
    # --------------------------------------------------------

    event_keys = [
        "portfolio_id",
        "selection_date",
    ]

    eligible_events = events.loc[
        events["event_status"].eq(
            "EFFECTIVE_REBALANCE"
        )
    ].copy()

    if eligible_events.empty:
        raise ValueError(
            "No subsequent effective rebalances available "
            "for ongoing turnover calculation."
        )

    if eligible_events[
        [
            "effective_rebalance_date",
            "preceding_holdings_date",
        ]
    ].isna().any().any():
        raise ValueError(
            "Effective rebalances require both effective "
            "and preceding holdings dates."
        )

    if (
        eligible_events["preceding_holdings_date"]
        >= eligible_events["effective_rebalance_date"]
    ).any():
        raise ValueError(
            "Pre-rebalance holdings dates must precede "
            "effective return dates."
        )

    if eligible_events.duplicated(
        ["portfolio_id", "effective_rebalance_date"]
    ).any():
        raise ValueError(
            "Duplicate effective rebalance dates."
        )

    # --------------------------------------------------------
    # 6. Verify preceding dates against observed calendars
    # --------------------------------------------------------

    for portfolio_id, portfolio_events in (
        eligible_events.groupby("portfolio_id")
    ):

        calendar = pd.DatetimeIndex(
            holdings.loc[
                holdings["portfolio_id"].eq(portfolio_id),
                "date",
            ].unique()
        ).sort_values()

        if calendar.empty:
            raise ValueError(
                f"Missing holdings calendar: {portfolio_id}"
            )

        for event in portfolio_events.itertuples(index=False):

            effective_date = pd.Timestamp(
                event.effective_rebalance_date
            )

            previous_date = pd.Timestamp(
                event.preceding_holdings_date
            )

            position = int(
                calendar.searchsorted(effective_date)
            )

            if (
                position == 0
                or position == len(calendar)
                or calendar[position] != effective_date
                or calendar[position - 1] != previous_date
            ):
                raise ValueError(
                    f"{portfolio_id}: invalid preceding "
                    f"holdings date for {effective_date.date()}."
                )

    # --------------------------------------------------------
    # 7. Attach event identities to target securities
    # --------------------------------------------------------

    effective_targets = targets.merge(
        eligible_events,
        on=event_keys,
        how="inner",
        validate="many_to_one",
    )

    # --------------------------------------------------------
    # 8. Retrieve previous closing holdings
    # --------------------------------------------------------

    previous_holdings = (
        holdings[
            [
                "portfolio_id",
                "date",
                "security_id",
                "weight_eop",
            ]
        ]
        .rename(
            columns={
                "date": "preceding_holdings_date",
                "weight_eop": "weight_pre_rebalance",
            }
        )
        .merge(
            eligible_events,
            on=[
                "portfolio_id",
                "preceding_holdings_date",
            ],
            how="inner",
            validate="many_to_one",
        )
    )

    # --------------------------------------------------------
    # 9. Retrieve actual effective-date BOP holdings
    # --------------------------------------------------------

    actual_holdings = (
        holdings[
            [
                "portfolio_id",
                "date",
                "security_id",
                "weight_bop",
            ]
        ]
        .rename(
            columns={
                "date": "effective_rebalance_date",
                "weight_bop": "weight_implemented",
            }
        )
        .merge(
            eligible_events,
            on=[
                "portfolio_id",
                "effective_rebalance_date",
            ],
            how="inner",
            validate="many_to_one",
        )
    )

    # --------------------------------------------------------
    # 10. Verify complete portfolio weights at every event
    # --------------------------------------------------------

    weight_totals = eligible_events[
        event_keys
    ].copy()

    weight_sources = [
        (
            effective_targets,
            "weight_target",
            "target_weight_sum",
        ),
        (
            previous_holdings,
            "weight_pre_rebalance",
            "previous_weight_sum",
        ),
        (
            actual_holdings,
            "weight_implemented",
            "implemented_weight_sum",
        ),
    ]

    for frame, source_column, output_column in weight_sources:

        totals = (
            frame
            .groupby(event_keys, as_index=False)
            .agg(**{
                output_column: (
                    source_column,
                    "sum",
                )
            })
        )

        weight_totals = weight_totals.merge(
            totals,
            on=event_keys,
            how="left",
            validate="one_to_one",
        )

        if weight_totals[output_column].isna().any():
            raise ValueError(
                f"Missing {output_column} for an eligible event."
            )

        maximum_error = (
            weight_totals[output_column]
            .sub(1.0)
            .abs()
            .max()
        )

        if maximum_error > weight_tolerance:
            raise ValueError(
                f"{output_column}: full-investment "
                f"error {maximum_error:.12g} exceeds tolerance."
            )

    # --------------------------------------------------------
    # 11. Independently verify actual target implementation
    # --------------------------------------------------------

    security_keys = [
        "portfolio_id",
        "selection_date",
        "security_id",
    ]

    implementation_check = effective_targets[
        security_keys + ["weight_target"]
    ].merge(
        actual_holdings[
            security_keys + ["weight_implemented"]
        ],
        on=security_keys,
        how="outer",
        validate="one_to_one",
        indicator="implementation_presence",
    )

    # Missing positions mean zero exposure in a complete
    # ledger. All recorded numeric weights were validated above.

    implementation_check["target_weight"] = (
        implementation_check["weight_target"].fillna(0.0)
    )

    implementation_check["implemented_weight"] = (
        implementation_check["weight_implemented"].fillna(0.0)
    )

    implementation_check["implementation_error"] = (
        implementation_check["implemented_weight"]
        - implementation_check["target_weight"]
    )

    max_implementation_error = float(
        implementation_check[
            "implementation_error"
        ].abs().max()
    )

    if max_implementation_error > weight_tolerance:
        raise ValueError(
            "Effective-date BOP holdings do not reconcile "
            "to scheduled target weights. "
            f"Maximum error: {max_implementation_error:.12g}"
        )

    # --------------------------------------------------------
    # 12. Outer-join previous and target security universes
    # --------------------------------------------------------

    security_df = effective_targets[
        security_keys + ["weight_target"]
    ].merge(
        previous_holdings[
            security_keys + ["weight_pre_rebalance"]
        ],
        on=security_keys,
        how="outer",
        validate="one_to_one",
        indicator="position_presence",
    )

    security_df["position_presence"] = (
        security_df["position_presence"]
        .map(
            {
                "both": "PREVIOUS_AND_TARGET",
                "left_only": "TARGET_ONLY",
                "right_only": "PREVIOUS_ONLY",
            }
        )
        .astype("string")
    )

    security_df["weight_pre_rebalance"] = (
        security_df["weight_pre_rebalance"].fillna(0.0)
    )

    security_df["weight_target"] = (
        security_df["weight_target"].fillna(0.0)
    )

    # --------------------------------------------------------
    # 13. Calculate signed weight changes and implied trades
    # --------------------------------------------------------

    security_df["weight_change"] = (
        security_df["weight_target"]
        - security_df["weight_pre_rebalance"]
    )

    security_df["implied_buy_weight"] = (
        security_df["weight_change"].clip(lower=0.0)
    )

    security_df["implied_sell_weight"] = (
        (-security_df["weight_change"]).clip(lower=0.0)
    )

    security_df["absolute_weight_change"] = (
        security_df["weight_change"].abs()
    )

    security_df["one_way_turnover_contribution"] = (
        0.5 * security_df["absolute_weight_change"]
    )

    # --------------------------------------------------------
    # 14. Classify security-level portfolio changes
    # --------------------------------------------------------

    previous_weight = security_df[
        "weight_pre_rebalance"
    ]

    target_weight = security_df[
        "weight_target"
    ]

    weight_change = security_df[
        "weight_change"
    ]

    is_new = (
        previous_weight.le(weight_tolerance)
        & target_weight.gt(weight_tolerance)
    )

    is_exit = (
        previous_weight.gt(weight_tolerance)
        & target_weight.le(weight_tolerance)
    )

    is_increase = (
        previous_weight.gt(weight_tolerance)
        & target_weight.gt(weight_tolerance)
        & weight_change.gt(weight_tolerance)
    )

    is_decrease = (
        previous_weight.gt(weight_tolerance)
        & target_weight.gt(weight_tolerance)
        & weight_change.lt(-weight_tolerance)
    )

    security_df["position_change_type"] = np.select(
        [
            is_new,
            is_exit,
            is_increase,
            is_decrease,
        ],
        [
            "NEW_POSITION",
            "EXIT_POSITION",
            "WEIGHT_INCREASE",
            "WEIGHT_DECREASE",
        ],
        default="UNCHANGED",
    )

    # --------------------------------------------------------
    # 15. Aggregate security changes to portfolio events
    # --------------------------------------------------------

    event_metrics = (
        security_df
        .groupby(event_keys, as_index=False)
        .agg(
            securities_compared=("security_id", "size"),
            total_purchases=(
                "implied_buy_weight",
                "sum",
            ),
            total_sales=(
                "implied_sell_weight",
                "sum",
            ),
            gross_two_way_turnover=(
                "absolute_weight_change",
                "sum",
            ),
            one_way_turnover=(
                "one_way_turnover_contribution",
                "sum",
            ),
            net_weight_change=(
                "weight_change",
                "sum",
            ),
            new_positions=(
                "position_change_type",
                lambda values: (
                    values == "NEW_POSITION"
                ).sum(),
            ),
            exited_positions=(
                "position_change_type",
                lambda values: (
                    values == "EXIT_POSITION"
                ).sum(),
            ),
            increased_positions=(
                "position_change_type",
                lambda values: (
                    values == "WEIGHT_INCREASE"
                ).sum(),
            ),
            decreased_positions=(
                "position_change_type",
                lambda values: (
                    values == "WEIGHT_DECREASE"
                ).sum(),
            ),
        )
    )

    # --------------------------------------------------------
    # 16. Reconcile portfolio-level turnover identities
    # --------------------------------------------------------

    event_metrics["buy_sell_balance_error"] = (
        event_metrics["total_purchases"]
        - event_metrics["total_sales"]
    ).abs()

    event_metrics["turnover_identity_error"] = (
        event_metrics["one_way_turnover"]
        - 0.5 * event_metrics["gross_two_way_turnover"]
    ).abs()

    event_metrics["gross_flow_identity_error"] = (
        event_metrics["gross_two_way_turnover"]
        - event_metrics["total_purchases"]
        - event_metrics["total_sales"]
    ).abs()

    # Since both previous and target portfolios are fully
    # invested, net implied purchases and sales must balance.

    if (
        event_metrics["buy_sell_balance_error"]
        > 2.0 * weight_tolerance
    ).any():
        raise ValueError(
            "Implied purchases and sales do not balance."
        )

    if (
        event_metrics["turnover_identity_error"]
        > weight_tolerance
    ).any():
        raise ValueError(
            "One-way turnover identity failed."
        )

    if (
        event_metrics["gross_flow_identity_error"]
        > weight_tolerance
    ).any():
        raise ValueError(
            "Gross trading-flow reconciliation failed."
        )

    event_metrics = event_metrics.merge(
        weight_totals,
        on=event_keys,
        how="left",
        validate="one_to_one",
    )

    event_metrics[
        "max_abs_implementation_weight_error"
    ] = max_implementation_error

    # --------------------------------------------------------
    # 17. Preserve initial and pending event statuses
    # --------------------------------------------------------

    event_df = events.merge(
        event_metrics,
        on=event_keys,
        how="left",
        validate="one_to_one",
    )

    event_df["turnover_status"] = (
        "NOT_ASSESSED_OTHER_EVENT"
    )

    event_df.loc[
        event_df["event_status"].eq(
            "INITIAL_ESTABLISHMENT"
        ),
        "turnover_status",
    ] = "NOT_ASSESSED_INITIAL_ESTABLISHMENT"

    event_df.loc[
        event_df["event_status"].eq(
            "PENDING_AFTER_CUTOFF"
        ),
        "turnover_status",
    ] = "NOT_ASSESSED_PENDING_SELECTION"

    event_df.loc[
        event_df["event_status"].eq(
            "EFFECTIVE_REBALANCE"
        ),
        "turnover_status",
    ] = "PASSED"

    # Attach event timing to security-level analytical output.

    security_df = security_df.merge(
        eligible_events,
        on=event_keys,
        how="left",
        validate="many_to_one",
    )

    # --------------------------------------------------------
    # 18. Sort and return standardized analytical outputs
    # --------------------------------------------------------

    security_df = (
        security_df
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

    if security_df.duplicated(security_keys).any():
        raise ValueError(
            "Duplicate security-turnover output keys."
        )

    if event_df.duplicated(event_keys).any():
        raise ValueError(
            "Duplicate event-turnover output keys."
        )

    return security_df, event_df


# Research reference: 12_turnover_rebalancing_analytics.ipynb, code cell index 15.
def calculate_scheduled_rebalance_turnover_portable(
    holdings_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    events_df: pd.DataFrame,
    *,
    weight_tolerance: float = 1e-10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Portable scheduled-turnover entry point.

    Delegate ordinary turnover calculations to the independently
    validated function. Support the valid no-subsequent-rebalance
    edge case with explicit unavailable event-level observations.

    A no-rebalance portfolio must have a valid initial allocation
    and no unexplained subsequent beginning-of-period weight changes.
    """

    frames = {
        "holdings": holdings_df,
        "targets": targets_df,
        "events": events_df,
    }

    required = {
        "holdings": {
            "portfolio_id", "date", "security_id",
            "weight_bop", "weight_eop",
        },
        "targets": {
            "portfolio_id", "selection_date",
            "security_id", "weight_target",
        },
        "events": {
            "portfolio_id", "selection_date",
            "effective_rebalance_date",
            "preceding_holdings_date", "event_status",
        },
    }

    if not np.isfinite(weight_tolerance) or weight_tolerance < 0:
        raise ValueError("Invalid weight tolerance.")

    for name, frame in frames.items():
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise ValueError(f"{name} must be a nonempty DataFrame.")
        missing = required[name].difference(frame.columns)
        if missing:
            raise ValueError(
                f"{name}: missing fields {sorted(missing)}"
            )

    # Preserve the exact validated calculation path when
    # at least one subsequent rebalance exists.
    if events_df["event_status"].eq("EFFECTIVE_REBALANCE").any():
        return calculate_scheduled_rebalance_turnover(
            holdings_df,
            targets_df,
            events_df,
            weight_tolerance=weight_tolerance,
        )

    h = holdings_df[
        ["portfolio_id", "date", "security_id",
         "weight_bop", "weight_eop"]
    ].copy()

    t = targets_df[
        ["portfolio_id", "selection_date",
         "security_id", "weight_target"]
    ].copy()

    e = events_df[
        ["portfolio_id", "selection_date",
         "effective_rebalance_date",
         "preceding_holdings_date", "event_status"]
    ].copy()

    h["date"] = pd.to_datetime(h["date"], errors="raise")
    t["selection_date"] = pd.to_datetime(
        t["selection_date"], errors="raise"
    )

    for column in (
        "selection_date",
        "effective_rebalance_date",
        "preceding_holdings_date",
    ):
        e[column] = pd.to_datetime(e[column], errors="raise")

    key_specs = (
        (
            "holdings", h,
            ["portfolio_id", "date", "security_id"],
        ),
        (
            "targets", t,
            ["portfolio_id", "selection_date", "security_id"],
        ),
        (
            "events", e,
            ["portfolio_id", "selection_date"],
        ),
    )

    for name, frame, keys in key_specs:
        if frame[keys].isna().any().any() or frame.duplicated(keys).any():
            raise ValueError(f"{name}: invalid primary keys.")

    portfolio_sets = [
        set(frame["portfolio_id"])
        for frame in (h, t, e)
    ]
    if not all(ids == portfolio_sets[0] for ids in portfolio_sets):
        raise ValueError("Portfolio identity sets differ.")

    permitted_statuses = {
        "INITIAL_ESTABLISHMENT",
        "PENDING_AFTER_CUTOFF",
    }
    if not set(e["event_status"]).issubset(permitted_statuses):
        raise ValueError(
            "Unrecognized event status without a subsequent rebalance."
        )

    # Verify numeric values and complete long-only portfolios.
    for name, frame, weight_columns in (
        ("holdings", h, ["weight_bop", "weight_eop"]),
        ("targets", t, ["weight_target"]),
    ):
        for column in weight_columns:
            values = pd.to_numeric(
                frame[column], errors="raise"
            ).to_numpy(dtype=float, na_value=np.nan)

            if not np.isfinite(values).all():
                raise ValueError(f"{name}: nonfinite weights.")

            if (
                (values < -weight_tolerance)
                | (values > 1.0 + weight_tolerance)
            ).any():
                raise ValueError(f"{name}: invalid long-only weights.")

    for frame, keys, column in (
        (h, ["portfolio_id", "date"], "weight_bop"),
        (h, ["portfolio_id", "date"], "weight_eop"),
        (t, ["portfolio_id", "selection_date"], "weight_target"),
    ):
        errors = (
            frame.groupby(keys)[column].sum() - 1.0
        ).abs()

        if errors.gt(weight_tolerance).any():
            raise ValueError(
                f"Full-investment identity failed: {column}."
            )

    # Target selections and the event schedule must correspond.
    target_keys = pd.MultiIndex.from_frame(
        t[["portfolio_id", "selection_date"]].drop_duplicates()
    )
    event_keys = pd.MultiIndex.from_frame(
        e[["portfolio_id", "selection_date"]]
    )

    if (
        not target_keys.isin(event_keys).all()
        or not event_keys.isin(target_keys).all()
    ):
        raise ValueError("Target/event selection coverage differs.")

    # Independently validate each portfolio's initial allocation
    # and its subsequent passive EOP-to-BOP continuity.
    for portfolio_id, portfolio_holdings in h.groupby("portfolio_id"):

        portfolio_events = e.loc[
            e["portfolio_id"].eq(portfolio_id)
        ]

        initial = portfolio_events.loc[
            portfolio_events["event_status"].eq(
                "INITIAL_ESTABLISHMENT"
            )
        ]

        if len(initial) != 1:
            raise ValueError(
                "Exactly one initial establishment is required "
                "for a no-rebalance portfolio."
            )

        dates = pd.DatetimeIndex(
            portfolio_holdings["date"].unique()
        ).sort_values()

        first_date = dates[0]
        last_date = dates[-1]

        if initial["effective_rebalance_date"].iloc[0] != first_date:
            raise ValueError(
                "Initial effective date differs from first holdings date."
            )

        first_bop = (
            portfolio_holdings.loc[
                portfolio_holdings["date"].eq(first_date)
            ]
            .set_index("security_id")["weight_bop"]
        )

        initial_target = (
            t.loc[
                t["portfolio_id"].eq(portfolio_id)
                & t["selection_date"].eq(
                    initial["selection_date"].iloc[0]
                )
            ]
            .set_index("security_id")["weight_target"]
        )

        all_ids = first_bop.index.union(initial_target.index)

        implementation_error = (
            first_bop.reindex(all_ids, fill_value=0.0)
            - initial_target.reindex(all_ids, fill_value=0.0)
        ).abs().max()

        if implementation_error > weight_tolerance:
            raise ValueError(
                "Initial target mismatch with first BOP holdings."
            )

        # No subsequent rebalance implies continuity of the
        # final EOP holdings into every following BOP observation.
        for previous_date, current_date in zip(
            dates[:-1], dates[1:]
        ):
            previous = (
                portfolio_holdings.loc[
                    portfolio_holdings["date"].eq(previous_date)
                ]
                .set_index("security_id")["weight_eop"]
            )

            current = (
                portfolio_holdings.loc[
                    portfolio_holdings["date"].eq(current_date)
                ]
                .set_index("security_id")["weight_bop"]
            )

            all_ids = previous.index.union(current.index)

            continuity_error = (
                previous.reindex(all_ids, fill_value=0.0)
                - current.reindex(all_ids, fill_value=0.0)
            ).abs().max()

            if continuity_error > weight_tolerance:
                raise ValueError(
                    "Unscheduled BOP weight change in "
                    "no-rebalance portfolio."
                )

        pending = portfolio_events.loc[
            portfolio_events["event_status"].eq(
                "PENDING_AFTER_CUTOFF"
            )
        ]

        if (
            pending["effective_rebalance_date"].notna().any()
            or pending["selection_date"].lt(last_date).any()
        ):
            raise ValueError(
                "Pending selection is inconsistent with "
                "the observed holdings cutoff."
            )

    # No security turnover observations are available because
    # no recurring rebalance event exists.
    security_column_types = {
        "portfolio_id": "object",
        "selection_date": "datetime64[ns]",
        "security_id": "object",
        "weight_target": "float64",
        "weight_pre_rebalance": "float64",
        "position_presence": "string",
        "weight_change": "float64",
        "implied_buy_weight": "float64",
        "implied_sell_weight": "float64",
        "absolute_weight_change": "float64",
        "one_way_turnover_contribution": "float64",
        "position_change_type": "object",
        "effective_rebalance_date": "datetime64[ns]",
        "preceding_holdings_date": "datetime64[ns]",
        "event_status": "object",
    }

    empty_security_df = pd.DataFrame({
        column: pd.Series(dtype=dtype)
        for column, dtype in security_column_types.items()
    })

    event_df = e.copy()

    numerical_event_fields = [
        "securities_compared",
        "total_purchases",
        "total_sales",
        "gross_two_way_turnover",
        "one_way_turnover",
        "net_weight_change",
        "new_positions",
        "exited_positions",
        "increased_positions",
        "decreased_positions",
        "buy_sell_balance_error",
        "turnover_identity_error",
        "gross_flow_identity_error",
        "target_weight_sum",
        "previous_weight_sum",
        "implemented_weight_sum",
        "max_abs_implementation_weight_error",
    ]

    for column in numerical_event_fields:
        event_df[column] = np.nan

    event_df["turnover_status"] = np.where(
        event_df["event_status"].eq("INITIAL_ESTABLISHMENT"),
        "NOT_ASSESSED_INITIAL_ESTABLISHMENT",
        "NOT_ASSESSED_PENDING_SELECTION",
    )

    event_df = (
        event_df
        .sort_values(["portfolio_id", "selection_date"])
        .reset_index(drop=True)
    )

    return empty_security_df, event_df
