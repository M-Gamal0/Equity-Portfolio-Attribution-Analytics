"""Turnover transitions: validated research calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd



# Research reference: 12_turnover_rebalancing_analytics.ipynb, code cell index 11.
def reconcile_daily_weight_transitions(
    holdings_df: pd.DataFrame,
    turnover_events_df: pd.DataFrame,
    security_changes_df: pd.DataFrame,
    drift_summary_df: pd.DataFrame,
    *,
    tolerance: float = 1e-10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reconcile previous final closing weights against current BOP.

    Returns
    -------
    detail_df:
        One row per security in the union of previous EOP
        and current BOP holdings.

        Key: (portfolio_id, date, security_id).

    daily_df:
        One row per (portfolio_id, date), including transition
        classifications, modeled turnover, and drift diagnostics.

    Notes
    -----
    Missing positions in complete ledgers represent zero
    exposure. Missing recorded numeric values are rejected.

    Scheduled turnover is model-implied, not execution-based.

    Price drift and terminal redistribution remain separate
    from scheduled trading.
    """

    # --------------------------------------------------------
    # 2. Validate schemas and calculation settings
    # --------------------------------------------------------

    required = {
        "holdings": {
            "portfolio_id", "date", "security_id",
            "weight_bop", "weight_eop",
        },
        "events": {
            "portfolio_id", "selection_date",
            "effective_rebalance_date", "event_status",
            "turnover_status", "one_way_turnover",
            "total_purchases", "total_sales",
        },
        "security changes": {
            "portfolio_id", "selection_date", "security_id",
            "weight_pre_rebalance", "weight_target",
            "weight_change",
        },
        "daily drift": {
            "portfolio_id", "date",
            "price_drift_half_l1_magnitude",
            "terminal_model_half_l1_adjustment",
            "terminal_zeroed_securities",
            "drift_reconciliation_status",
        },
    }

    frames = {
        "holdings": holdings_df,
        "events": turnover_events_df,
        "security changes": security_changes_df,
        "daily drift": drift_summary_df,
    }

    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError(
            "tolerance must be finite and nonnegative."
        )

    for name, frame in frames.items():

        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise ValueError(
                f"{name} must be a nonempty DataFrame."
            )

        missing = required[name] - set(frame.columns)

        if missing:
            raise ValueError(
                f"{name}: missing required fields "
                f"{sorted(missing)}"
            )

    h = holdings_df[
        [
            "portfolio_id", "date", "security_id",
            "weight_bop", "weight_eop",
        ]
    ].copy()

    e = turnover_events_df[
        [
            "portfolio_id", "selection_date",
            "effective_rebalance_date", "event_status",
            "turnover_status", "one_way_turnover",
            "total_purchases", "total_sales",
        ]
    ].copy()

    s = security_changes_df[
        [
            "portfolio_id", "selection_date",
            "security_id", "weight_pre_rebalance",
            "weight_target", "weight_change",
        ]
    ].copy()

    d = drift_summary_df[
        [
            "portfolio_id", "date",
            "price_drift_half_l1_magnitude",
            "terminal_model_half_l1_adjustment",
            "terminal_zeroed_securities",
            "drift_reconciliation_status",
        ]
    ].copy()

    # --------------------------------------------------------
    # 3. Normalize dates and validate primary keys
    # --------------------------------------------------------

    for frame, date_columns in (
        (h, ["date"]),
        (e, ["selection_date", "effective_rebalance_date"]),
        (s, ["selection_date"]),
        (d, ["date"]),
    ):

        for column in date_columns:

            frame[column] = pd.to_datetime(
                frame[column],
                errors="raise",
            )

    daily_key = ["portfolio_id", "date"]
    event_key = ["portfolio_id", "selection_date"]
    security_key = ["portfolio_id", "date", "security_id"]

    key_checks = [
        ("holdings", h, security_key),
        ("events", e, event_key),
        (
            "security changes",
            s,
            ["portfolio_id", "selection_date", "security_id"],
        ),
        ("daily drift", d, daily_key),
    ]

    for name, frame, keys in key_checks:

        if (
            frame[keys].isna().any().any()
            or frame.duplicated(keys).any()
        ):
            raise ValueError(
                f"{name}: null or duplicate primary keys."
            )

    # --------------------------------------------------------
    # 4. Validate weights and diagnostic numeric values
    # --------------------------------------------------------

    numerical_checks = [
        ("holdings", h, ["weight_bop", "weight_eop"]),
        (
            "security changes",
            s,
            [
                "weight_pre_rebalance",
                "weight_target",
                "weight_change",
            ],
        ),
        (
            "daily drift",
            d,
            [
                "price_drift_half_l1_magnitude",
                "terminal_model_half_l1_adjustment",
                "terminal_zeroed_securities",
            ],
        ),
    ]

    weight_fields = {
        "weight_bop",
        "weight_eop",
        "weight_pre_rebalance",
        "weight_target",
    }

    for name, frame, columns in numerical_checks:

        for column in columns:

            values = pd.to_numeric(
                frame[column],
                errors="raise",
            ).to_numpy(
                dtype=float,
                na_value=np.nan,
            )

            if not np.isfinite(values).all():
                raise ValueError(
                    f"{name}.{column}: nonfinite values."
                )

            if column in weight_fields:

                if (
                    (values < -tolerance)
                    | (values > 1.0 + tolerance)
                ).any():
                    raise ValueError(
                        f"{name}.{column}: invalid weight bounds."
                    )

            elif column != "weight_change":

                if (values < -tolerance).any():
                    raise ValueError(
                        f"{name}.{column}: invalid negative value."
                    )

    if not d["drift_reconciliation_status"].eq(
        "PASSED"
    ).all():
        raise ValueError(
            "Daily price-drift diagnostics must be validated."
        )

    eligible_numeric = e.loc[
        e["turnover_status"].eq("PASSED"),
        [
            "one_way_turnover",
            "total_purchases",
            "total_sales",
        ],
    ]

    if not np.isfinite(
        eligible_numeric.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Validated rebalance events contain nonfinite turnover."
        )

    # --------------------------------------------------------
    # 5. Verify full-investment identities
    # --------------------------------------------------------

    for column in ("weight_bop", "weight_eop"):

        daily_sums = (
            h.groupby(daily_key)[column].sum()
        )

        if (daily_sums - 1.0).abs().gt(
            tolerance
        ).any():
            raise ValueError(
                f"Daily {column} weights do not sum to one."
            )

    # --------------------------------------------------------
    # 6. Construct each portfolio's observed trading calendar
    # --------------------------------------------------------

    calendar = (
        h[daily_key]
        .drop_duplicates()
        .sort_values(daily_key)
        .reset_index(drop=True)
    )

    calendar["previous_date"] = (
        calendar
        .groupby("portfolio_id")["date"]
        .shift(1)
    )

    # --------------------------------------------------------
    # 7. Attach implemented rebalance-event classifications
    # --------------------------------------------------------

    dated_events = e.loc[
        e["event_status"].isin(
            ["INITIAL_ESTABLISHMENT", "EFFECTIVE_REBALANCE"]
        )
    ].copy()

    if (
        dated_events["effective_rebalance_date"].isna().any()
        or dated_events.duplicated(
            ["portfolio_id", "effective_rebalance_date"]
        ).any()
    ):
        raise ValueError(
            "Invalid implemented rebalance-event dates."
        )

    subsequent = dated_events["event_status"].eq(
        "EFFECTIVE_REBALANCE"
    )

    initial = dated_events["event_status"].eq(
        "INITIAL_ESTABLISHMENT"
    )

    if not dated_events.loc[
        subsequent, "turnover_status"
    ].eq("PASSED").all():
        raise ValueError(
            "Subsequent rebalances lack validated turnover."
        )

    if not dated_events.loc[
        initial, "turnover_status"
    ].eq("NOT_ASSESSED_INITIAL_ESTABLISHMENT").all():
        raise ValueError(
            "Initial establishment turnover status is invalid."
        )

    dated_events = dated_events.rename(
        columns={"effective_rebalance_date": "date"}
    )

    calendar = calendar.merge(
        dated_events,
        on=daily_key,
        how="left",
        validate="one_to_one",
    )

    first_date_mask = calendar["previous_date"].isna()

    if not calendar.loc[
        first_date_mask, "event_status"
    ].eq("INITIAL_ESTABLISHMENT").all():
        raise ValueError(
            "First holdings dates do not match initial events."
        )

    if calendar.loc[
        ~first_date_mask, "event_status"
    ].eq("INITIAL_ESTABLISHMENT").any():
        raise ValueError(
            "Initial establishment occurs after holdings inception."
        )

    if (
        calendar["event_status"].notna().sum()
        != len(dated_events)
    ):
        raise ValueError(
            "Implemented events are missing from the holdings calendar."
        )

    calendar["transition_type"] = np.select(
        [
            first_date_mask,
            calendar["event_status"].eq(
                "EFFECTIVE_REBALANCE"
            ),
        ],
        [
            "INITIAL_ESTABLISHMENT",
            "SCHEDULED_REBALANCE",
        ],
        default="PASSIVE_CARRYOVER",
    )

    # --------------------------------------------------------
    # 8. Retrieve prior trading session's final EOP weights
    # --------------------------------------------------------

    prior_weights = (
        h[
            [
                "portfolio_id", "date",
                "security_id", "weight_eop",
            ]
        ]
        .rename(
            columns={
                "date": "previous_date",
                "weight_eop": "previous_eop_weight",
            }
        )
    )

    prior_weights = (
        calendar.loc[
            calendar["previous_date"].notna(),
            ["portfolio_id", "date", "previous_date"],
        ]
        .merge(
            prior_weights,
            on=["portfolio_id", "previous_date"],
            how="left",
            validate="one_to_many",
        )
    )

    if prior_weights["security_id"].isna().any():
        raise ValueError(
            "Missing preceding holdings observations."
        )

    # --------------------------------------------------------
    # 9. Outer-join previous EOP and current BOP security sets
    # --------------------------------------------------------

    detail = (
        h[
            [
                "portfolio_id", "date",
                "security_id", "weight_bop",
            ]
        ]
        .rename(
            columns={"weight_bop": "current_bop_weight"}
        )
        .merge(
            prior_weights[
                [
                    "portfolio_id", "date",
                    "security_id", "previous_eop_weight",
                ]
            ],
            on=security_key,
            how="outer",
            validate="one_to_one",
            indicator="position_presence",
        )
    )

    detail = detail.merge(
        calendar[
            daily_key + ["previous_date", "transition_type"]
        ],
        on=daily_key,
        how="left",
        validate="many_to_one",
    )

    if detail["transition_type"].isna().any():
        raise ValueError(
            "Security transitions contain unknown dates."
        )

    initial_security_rows = detail[
        "transition_type"
    ].eq("INITIAL_ESTABLISHMENT")

    # A missing current/previous security implies zero exposure
    # only when a preceding holdings ledger actually exists.

    detail["current_bop_weight"] = (
        detail["current_bop_weight"].fillna(0.0)
    )

    detail.loc[
        ~initial_security_rows,
        "previous_eop_weight",
    ] = detail.loc[
        ~initial_security_rows,
        "previous_eop_weight",
    ].fillna(0.0)

    # --------------------------------------------------------
    # 10. Calculate daily beginning-of-period weight changes
    # --------------------------------------------------------

    detail["bridge_weight_change"] = (
        detail["current_bop_weight"]
        - detail["previous_eop_weight"]
    )

    detail["bridge_abs_change"] = (
        detail["bridge_weight_change"].abs()
    )

    detail["bridge_buy_weight"] = (
        detail["bridge_weight_change"].clip(lower=0.0)
    )

    detail["bridge_sell_weight"] = (
        (-detail["bridge_weight_change"]).clip(lower=0.0)
    )

    # --------------------------------------------------------
    # 11. Reconcile rebalance security changes against Cell 06
    # --------------------------------------------------------

    rebalance_calendar = calendar.loc[
        calendar["transition_type"].eq(
            "SCHEDULED_REBALANCE"
        ),
        daily_key + ["selection_date"],
    ]

    expected_trades = s.merge(
        rebalance_calendar,
        on=event_key,
        how="outer",
        validate="many_to_one",
        indicator="event_coverage",
    )

    if not expected_trades[
        "event_coverage"
    ].eq("both").all():
        raise ValueError(
            "Security trade events differ from the rebalance calendar."
        )

    actual_trades = detail.loc[
        detail["transition_type"].eq(
            "SCHEDULED_REBALANCE"
        ),
        security_key + [
            "previous_eop_weight",
            "current_bop_weight",
            "bridge_weight_change",
        ],
    ]

    security_check = actual_trades.merge(
        expected_trades[
            security_key + [
                "weight_pre_rebalance",
                "weight_target",
                "weight_change",
            ]
        ],
        on=security_key,
        how="outer",
        validate="one_to_one",
        indicator="security_coverage",
    )

    if not security_check[
        "security_coverage"
    ].eq("both").all():
        raise ValueError(
            "Rebalance security universes do not match."
        )

    security_comparisons = [
        ("previous_eop_weight", "weight_pre_rebalance"),
        ("current_bop_weight", "weight_target"),
        ("bridge_weight_change", "weight_change"),
    ]

    for observed, expected in security_comparisons:

        errors = (
            security_check[observed]
            - security_check[expected]
        ).abs()

        if errors.gt(tolerance).any():
            raise ValueError(
                f"Security transition mismatch: "
                f"{observed} vs. {expected}."
            )

    # --------------------------------------------------------
    # 12. Aggregate daily security-transition diagnostics
    # --------------------------------------------------------

    bridge_totals = (
        detail
        .groupby(daily_key, as_index=False)
        .agg(
            bridge_security_rows=("security_id", "size"),
            prior_close_weight_sum=(
                "previous_eop_weight", "sum"
            ),
            current_bop_weight_sum=(
                "current_bop_weight", "sum"
            ),
            net_bridge_change=(
                "bridge_weight_change", "sum"
            ),
            maximum_abs_bridge_change=(
                "bridge_abs_change", "max"
            ),
            bridge_gross_abs_change=(
                "bridge_abs_change", "sum"
            ),
            bridge_buy_total=(
                "bridge_buy_weight", "sum"
            ),
            bridge_sell_total=(
                "bridge_sell_weight", "sum"
            ),
        )
    )

    bridge_totals["bridge_half_l1_change"] = (
        0.5 * bridge_totals["bridge_gross_abs_change"]
    )

    daily = calendar.merge(
        bridge_totals,
        on=daily_key,
        how="left",
        validate="one_to_one",
    )

    first = daily["transition_type"].eq(
        "INITIAL_ESTABLISHMENT"
    )

    passive = daily["transition_type"].eq(
        "PASSIVE_CARRYOVER"
    )

    rebalance = daily["transition_type"].eq(
        "SCHEDULED_REBALANCE"
    )

    # Do not reinterpret an unavailable initial preceding
    # portfolio as a portfolio with zero previous holdings.

    unavailable_columns = [
        "prior_close_weight_sum",
        "net_bridge_change",
        "maximum_abs_bridge_change",
        "bridge_gross_abs_change",
        "bridge_buy_total",
        "bridge_sell_total",
        "bridge_half_l1_change",
    ]

    daily.loc[first, unavailable_columns] = np.nan

    # --------------------------------------------------------
    # 13. Validate passive carryover and investment identities
    # --------------------------------------------------------

    noninitial = ~first

    if (
        (daily.loc[
            noninitial, "prior_close_weight_sum"
        ] - 1.0).abs().gt(tolerance).any()
        or (daily[
            "current_bop_weight_sum"
        ] - 1.0).abs().gt(tolerance).any()
    ):
        raise ValueError(
            "Daily transition investment identity failed."
        )

    if daily.loc[
        noninitial, "net_bridge_change"
    ].abs().gt(tolerance).any():
        raise ValueError(
            "Daily security weight changes do not net to zero."
        )

    if daily.loc[
        passive, "maximum_abs_bridge_change"
    ].gt(tolerance).any():
        raise ValueError(
            "Unexpected holdings change without a scheduled rebalance."
        )

    # --------------------------------------------------------
    # 14. Reconcile daily bridge turnover to Cell 06
    # --------------------------------------------------------

    portfolio_comparisons = [
        ("bridge_half_l1_change", "one_way_turnover"),
        ("bridge_buy_total", "total_purchases"),
        ("bridge_sell_total", "total_sales"),
    ]

    for observed, expected in portfolio_comparisons:

        errors = (
            daily.loc[rebalance, observed]
            - daily.loc[rebalance, expected]
        ).abs()

        if errors.gt(tolerance).any():
            raise ValueError(
                f"Rebalance turnover reconciliation failed: {observed}."
            )

    daily["scheduled_one_way_turnover"] = np.where(
        first,
        np.nan,
        np.where(
            rebalance,
            daily["one_way_turnover"],
            0.0,
        ),
    )

    daily["transition_validation_status"] = np.select(
        [first, rebalance],
        [
            "NOT_ASSESSED_INITIAL_ESTABLISHMENT",
            "PASSED_SCHEDULED_REBALANCE",
        ],
        default="PASSED_PASSIVE_CARRYOVER",
    )

    # --------------------------------------------------------
    # 15. Attach independently validated daily drift metrics
    # --------------------------------------------------------

    daily = daily.merge(
        d,
        on=daily_key,
        how="left",
        validate="one_to_one",
    )

    if (
        daily["drift_reconciliation_status"].isna().any()
        or len(daily) != len(d)
    ):
        raise ValueError(
            "Transition and price-drift calendars differ."
        )

    # A terminal event affects the current day's final EOP.
    # Its continuity is assessed on the next portfolio date.

    prior_terminal = (
        d[
            [
                "portfolio_id", "date",
                "terminal_zeroed_securities",
            ]
        ]
        .rename(
            columns={
                "date": "previous_date",
                "terminal_zeroed_securities":
                    "previous_close_terminal_zeroed_securities",
            }
        )
    )

    daily = daily.merge(
        prior_terminal,
        on=["portfolio_id", "previous_date"],
        how="left",
        validate="many_to_one",
    )

    if daily.loc[
        noninitial,
        "previous_close_terminal_zeroed_securities",
    ].isna().any():
        raise ValueError(
            "Preceding close terminal-event coverage is missing."
        )

    # --------------------------------------------------------
    # 16. Return stable analytical output contracts
    # --------------------------------------------------------

    detail = (
        detail
        .sort_values(security_key)
        .reset_index(drop=True)
    )

    daily = (
        daily
        .sort_values(daily_key)
        .reset_index(drop=True)
    )

    if (
        detail.duplicated(security_key).any()
        or daily.duplicated(daily_key).any()
    ):
        raise ValueError(
            "Duplicate daily transition output keys."
        )

    return detail, daily
