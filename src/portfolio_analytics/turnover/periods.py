"""Turnover periods: validated research calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd



# Research reference: 12_turnover_rebalancing_analytics.ipynb, code cell index 13.
def summarize_rebalance_turnover_periods(
    turnover_events_df: pd.DataFrame,
    daily_diagnostics_df: pd.DataFrame,
    *,
    tolerance: float = 1e-10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build quarterly, annual, and full-sample turnover summaries.

    Reporting period membership is determined exclusively by
    the effective rebalance date.

    Returns
    -------
    quarterly_df:
        One row per (portfolio_id, calendar quarter).

    annual_df:
        One row per (portfolio_id, calendar year).

    portfolio_df:
        One row per portfolio across its observed daily history.

    Notes
    -----
    Inception-only reporting periods retain unavailable
    ongoing turnover rather than recording zero turnover.

    Periods with observed portfolio returns but no scheduled
    rebalance may report zero scheduled turnover.

    Price drift and terminal adjustments remain distinct
    diagnostics and are never added to turnover.

    Full-sample arithmetic turnover per observed year divides
    cumulative turnover by calendar-time coverage. It is not
    a compounded statistic or a forecast.
    """

    # --------------------------------------------------------
    # 2. Validate schemas, types, and input keys
    # --------------------------------------------------------

    if (
        not isinstance(turnover_events_df, pd.DataFrame)
        or not isinstance(daily_diagnostics_df, pd.DataFrame)
    ):
        raise TypeError(
            "Turnover events and daily diagnostics must be DataFrames."
        )

    if turnover_events_df.empty or daily_diagnostics_df.empty:
        raise ValueError("Reporting inputs must be nonempty.")

    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Invalid numerical tolerance.")

    required_events = {
        "portfolio_id", "selection_date",
        "effective_rebalance_date", "event_status",
        "turnover_status", "one_way_turnover",
        "gross_two_way_turnover", "total_purchases",
        "total_sales", "new_positions", "exited_positions",
        "increased_positions", "decreased_positions",
    }

    required_daily = {
        "portfolio_id", "date", "transition_type",
        "scheduled_one_way_turnover",
        "price_drift_half_l1_magnitude",
        "terminal_model_half_l1_adjustment",
        "terminal_zeroed_securities",
        "drift_reconciliation_status",
    }

    for name, frame, required in (
        ("events", turnover_events_df, required_events),
        ("daily diagnostics", daily_diagnostics_df, required_daily),
    ):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(
                f"{name}: missing fields {sorted(missing)}"
            )

    events = turnover_events_df.copy(deep=True)
    daily = daily_diagnostics_df.copy(deep=True)

    events["selection_date"] = pd.to_datetime(
        events["selection_date"], errors="raise"
    )
    events["effective_rebalance_date"] = pd.to_datetime(
        events["effective_rebalance_date"], errors="raise"
    )
    daily["date"] = pd.to_datetime(
        daily["date"], errors="raise"
    )

    event_key = ["portfolio_id", "selection_date"]
    daily_key = ["portfolio_id", "date"]

    for name, frame, keys in (
        ("events", events, event_key),
        ("daily", daily, daily_key),
    ):
        if frame[keys].isna().any().any():
            raise ValueError(f"{name}: null primary keys.")
        if frame.duplicated(keys).any():
            raise ValueError(f"{name}: duplicate primary keys.")

    if set(events["portfolio_id"]) != set(daily["portfolio_id"]):
        raise ValueError(
            "Turnover and daily diagnostic portfolio sets differ."
        )

    # --------------------------------------------------------
    # 3. Validate historical event availability conventions
    # --------------------------------------------------------

    measured = events["turnover_status"].eq("PASSED")
    unmeasured = ~measured

    if not events.loc[
        measured, "event_status"
    ].eq("EFFECTIVE_REBALANCE").all():
        raise ValueError(
            "Measured turnover includes a non-rebalance event."
        )

    if events.loc[
        measured, "effective_rebalance_date"
    ].isna().any():
        raise ValueError(
            "Measured turnover events lack effective dates."
        )

    trade_columns = [
        "one_way_turnover",
        "gross_two_way_turnover",
        "total_purchases",
        "total_sales",
    ]

    if events.loc[unmeasured, trade_columns].notna().any().any():
        raise ValueError(
            "Unassessed events contain numeric turnover."
        )

    measured_events = events.loc[measured].copy()

    for column in trade_columns + [
        "new_positions", "exited_positions",
        "increased_positions", "decreased_positions",
    ]:
        values = pd.to_numeric(
            measured_events[column], errors="raise"
        ).to_numpy(dtype=float, na_value=np.nan)

        if not np.isfinite(values).all():
            raise ValueError(
                f"Nonfinite measured event metric: {column}"
            )
        if (values < -tolerance).any():
            raise ValueError(
                f"Negative measured event metric: {column}"
            )

    for label, left, right in (
        (
            "gross turnover",
            measured_events["gross_two_way_turnover"],
            2.0 * measured_events["one_way_turnover"],
        ),
        (
            "purchase and sale balance",
            measured_events["total_purchases"],
            measured_events["total_sales"],
        ),
    ):
        if (left - right).abs().gt(tolerance).any():
            raise ValueError(f"Event identity failed: {label}")

    if measured_events["one_way_turnover"].gt(
        1.0 + tolerance
    ).any():
        raise ValueError(
            "One-way turnover exceeds 100% for a long-only event."
        )

    if measured_events.duplicated(
        ["portfolio_id", "effective_rebalance_date"]
    ).any():
        raise ValueError("Duplicate effective rebalance dates.")

    # --------------------------------------------------------
    # 4. Reconcile eligible events to actual daily transitions
    # --------------------------------------------------------

    valid_transition_types = {
        "INITIAL_ESTABLISHMENT",
        "SCHEDULED_REBALANCE",
        "PASSIVE_CARRYOVER",
    }

    if not set(daily["transition_type"]).issubset(
        valid_transition_types
    ):
        raise ValueError(
            "Unexpected daily transition classification."
        )

    if not daily["drift_reconciliation_status"].eq(
        "PASSED"
    ).all():
        raise ValueError(
            "Reporting requires validated daily drift results."
        )

    daily_trade_fields = [
        "price_drift_half_l1_magnitude",
        "terminal_model_half_l1_adjustment",
        "terminal_zeroed_securities",
    ]

    for column in daily_trade_fields:
        values = pd.to_numeric(
            daily[column], errors="raise"
        ).to_numpy(dtype=float, na_value=np.nan)

        if (
            not np.isfinite(values).all()
            or (values < -tolerance).any()
        ):
            raise ValueError(
                f"Invalid daily diagnostic values: {column}"
            )

    daily_rebalances = daily.loc[
        daily["transition_type"].eq("SCHEDULED_REBALANCE"),
        ["portfolio_id", "date", "scheduled_one_way_turnover"],
    ]

    event_daily_check = measured_events[
        ["portfolio_id", "effective_rebalance_date", "one_way_turnover"]
    ].rename(
        columns={"effective_rebalance_date": "date"}
    ).merge(
        daily_rebalances,
        on=["portfolio_id", "date"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    if not event_daily_check["_merge"].eq("both").all():
        raise ValueError(
            "Scheduled event dates and daily transitions differ."
        )

    if (
        event_daily_check["one_way_turnover"]
        - event_daily_check["scheduled_one_way_turnover"]
    ).abs().gt(tolerance).any():
        raise ValueError(
            "Scheduled event and daily turnover totals differ."
        )

    initial_mask = daily["transition_type"].eq(
        "INITIAL_ESTABLISHMENT"
    )
    passive_mask = daily["transition_type"].eq(
        "PASSIVE_CARRYOVER"
    )

    if daily.loc[
        initial_mask, "scheduled_one_way_turnover"
    ].notna().any():
        raise ValueError(
            "Initial establishment must not report turnover."
        )

    if daily.loc[
        passive_mask, "scheduled_one_way_turnover"
    ].sub(0.0).abs().gt(tolerance).any():
        raise ValueError(
            "Passive dates contain scheduled turnover."
        )

    # Confirm initial establishment events are represented
    # in the daily portfolio calendar.

    initial_events = events.loc[
        events["event_status"].eq("INITIAL_ESTABLISHMENT"),
        ["portfolio_id", "effective_rebalance_date"],
    ].rename(
        columns={"effective_rebalance_date": "date"}
    )

    actual_initial = daily.loc[
        initial_mask, ["portfolio_id", "date"]
    ]

    initial_check = initial_events.merge(
        actual_initial,
        on=["portfolio_id", "date"],
        how="outer",
        indicator=True,
        validate="one_to_one",
    )

    if not initial_check["_merge"].eq("both").all():
        raise ValueError(
            "Initial establishment dates and daily history differ."
        )

    # --------------------------------------------------------
    # 5. Define period-level aggregation
    # --------------------------------------------------------

    daily["is_initial_day"] = initial_mask.astype(int)
    daily["is_rebalance_day"] = daily[
        "transition_type"
    ].eq("SCHEDULED_REBALANCE").astype(int)

    daily["is_terminal_adjustment_day"] = (
        daily["terminal_zeroed_securities"].gt(0).astype(int)
    )

    def aggregate_period(frequency: str) -> pd.DataFrame:

        observed = daily.copy()

        observed["period"] = observed[
            "date"
        ].dt.to_period(frequency)

        coverage = (
            observed
            .groupby(["portfolio_id", "period"], as_index=False)
            .agg(
                observed_trading_days=("date", "size"),
                first_observed_date=("date", "min"),
                last_observed_date=("date", "max"),
                initial_establishment_days=(
                    "is_initial_day", "sum"
                ),
                scheduled_rebalance_days=(
                    "is_rebalance_day", "sum"
                ),
                cumulative_price_drift_magnitude=(
                    "price_drift_half_l1_magnitude", "sum"
                ),
                terminal_adjustment_days=(
                    "is_terminal_adjustment_day", "sum"
                ),
                terminal_zeroed_securities=(
                    "terminal_zeroed_securities", "sum"
                ),
                cumulative_terminal_adjustment_magnitude=(
                    "terminal_model_half_l1_adjustment", "sum"
                ),
            )
        )

        eligible = measured_events.copy()

        eligible["period"] = eligible[
            "effective_rebalance_date"
        ].dt.to_period(frequency)

        period_events = (
            eligible
            .groupby(["portfolio_id", "period"], as_index=False)
            .agg(
                measured_rebalances=(
                    "selection_date", "size"
                ),
                cumulative_one_way_turnover=(
                    "one_way_turnover", "sum"
                ),
                cumulative_gross_two_way_turnover=(
                    "gross_two_way_turnover", "sum"
                ),
                total_purchases=(
                    "total_purchases", "sum"
                ),
                total_sales=(
                    "total_sales", "sum"
                ),
                average_rebalance_turnover=(
                    "one_way_turnover", "mean"
                ),
                maximum_rebalance_turnover=(
                    "one_way_turnover", "max"
                ),
                new_positions=("new_positions", "sum"),
                exited_positions=("exited_positions", "sum"),
                increased_positions=("increased_positions", "sum"),
                decreased_positions=("decreased_positions", "sum"),
            )
        )

        result = coverage.merge(
            period_events,
            on=["portfolio_id", "period"],
            how="left",
            validate="one_to_one",
        )

        result["measured_rebalances"] = (
            result["measured_rebalances"]
            .fillna(0)
            .astype(int)
        )

        if not result[
            "measured_rebalances"
        ].eq(result["scheduled_rebalance_days"]).all():
            raise ValueError(
                f"{frequency}: event counts differ from daily history."
            )

        initial_only = (
            result["measured_rebalances"].eq(0)
            & result["initial_establishment_days"].gt(0)
        )

        no_scheduled_event = (
            result["measured_rebalances"].eq(0)
            & ~initial_only
        )

        result["turnover_measurement_status"] = np.select(
            [initial_only, no_scheduled_event],
            [
                "NOT_ASSESSED_INITIAL_ONLY",
                "NO_SCHEDULED_REBALANCE",
            ],
            default="MEASURED",
        )

        # Zero is valid only when a fully observed reporting
        # period has no scheduled rebalance and is not inception-only.

        additive_fields = [
            "cumulative_one_way_turnover",
            "cumulative_gross_two_way_turnover",
            "total_purchases",
            "total_sales",
            "new_positions",
            "exited_positions",
            "increased_positions",
            "decreased_positions",
        ]

        result.loc[
            no_scheduled_event, additive_fields
        ] = 0.0

        # Averages and maxima are undefined with zero
        # measured rebalance events and remain NaN.

        result["period_label"] = (
            result["period"].astype(str)
        )

        result["period_start"] = [
            value.start_time.normalize()
            for value in result["period"]
        ]

        result["period_end"] = [
            value.end_time.normalize()
            for value in result["period"]
        ]

        # Explicitly indicate first/last observed sample periods.
        sample_boundaries = daily.groupby(
            "portfolio_id"
        )["date"].agg(["min", "max"])

        result["sample_start_period"] = [
            pd.Period(
                sample_boundaries.loc[pid, "min"],
                freq=frequency,
            ) == period
            for pid, period in zip(
                result["portfolio_id"], result["period"]
            )
        ]

        result["sample_cutoff_period"] = [
            pd.Period(
                sample_boundaries.loc[pid, "max"],
                freq=frequency,
            ) == period
            for pid, period in zip(
                result["portfolio_id"], result["period"]
            )
        ]

        result["sample_boundary_status"] = np.select(
            [
                (
                    result["sample_start_period"]
                    & result["sample_cutoff_period"]
                ),
                result["sample_start_period"],
                result["sample_cutoff_period"],
            ],
            [
                "INCEPTION_AND_CUTOFF",
                "INCEPTION_PERIOD",
                "SAMPLE_CUTOFF_PERIOD",
            ],
            default="INTERIOR_PERIOD",
        )

        # Check event-level arithmetic identities.

        measured_periods = result[
            "measured_rebalances"
        ].gt(0)

        if (
            result.loc[
                measured_periods,
                "cumulative_gross_two_way_turnover",
            ]
            - 2.0 * result.loc[
                measured_periods,
                "cumulative_one_way_turnover",
            ]
        ).abs().gt(tolerance).any():
            raise ValueError(
                f"{frequency}: gross turnover identity failed."
            )

        if (
            result.loc[
                measured_periods, "total_purchases"
            ]
            - result.loc[
                measured_periods, "total_sales"
            ]
        ).abs().gt(tolerance).any():
            raise ValueError(
                f"{frequency}: purchase/sale identity failed."
            )

        return (
            result
            .sort_values(["portfolio_id", "period"])
            .reset_index(drop=True)
        )

    # --------------------------------------------------------
    # 6. Construct quarterly and annual summaries
    # --------------------------------------------------------

    quarterly_df = aggregate_period("Q")
    annual_df = aggregate_period("Y")

    # --------------------------------------------------------
    # 7. Construct full-sample portfolio summaries
    # --------------------------------------------------------

    sample_coverage = (
        daily.groupby("portfolio_id", as_index=False)
        .agg(
            first_return_date=("date", "min"),
            last_return_date=("date", "max"),
            observed_trading_days=("date", "size"),
            cumulative_price_drift_magnitude=(
                "price_drift_half_l1_magnitude", "sum"
            ),
            terminal_adjustment_days=(
                "is_terminal_adjustment_day", "sum"
            ),
            terminal_zeroed_securities=(
                "terminal_zeroed_securities", "sum"
            ),
            cumulative_terminal_adjustment_magnitude=(
                "terminal_model_half_l1_adjustment", "sum"
            ),
        )
    )

    sample_events = (
        measured_events
        .groupby("portfolio_id", as_index=False)
        .agg(
            measured_rebalances=("selection_date", "size"),
            cumulative_one_way_turnover=(
                "one_way_turnover", "sum"
            ),
            cumulative_gross_two_way_turnover=(
                "gross_two_way_turnover", "sum"
            ),
            cumulative_purchases=("total_purchases", "sum"),
            cumulative_sales=("total_sales", "sum"),
            average_rebalance_turnover=(
                "one_way_turnover", "mean"
            ),
            median_rebalance_turnover=(
                "one_way_turnover", "median"
            ),
            maximum_rebalance_turnover=(
                "one_way_turnover", "max"
            ),
            new_positions=("new_positions", "sum"),
            exited_positions=("exited_positions", "sum"),
        )
    )

    portfolio_df = sample_coverage.merge(
        sample_events,
        on="portfolio_id",
        how="left",
        validate="one_to_one",
    )

    portfolio_df["measured_rebalances"] = (
        portfolio_df["measured_rebalances"]
        .fillna(0)
        .astype(int)
    )

    portfolio_df["observed_calendar_years"] = (
        (
            portfolio_df["last_return_date"]
            - portfolio_df["first_return_date"]
        ).dt.days
        / 365.2425
    )

    portfolio_df["arithmetic_turnover_per_observed_year"] = (
        portfolio_df["cumulative_one_way_turnover"]
        / portfolio_df["observed_calendar_years"]
    )

    portfolio_df["turnover_measurement_status"] = np.where(
        portfolio_df["measured_rebalances"].gt(0),
        "MEASURED",
        "NOT_ASSESSED_NO_RECURRING_REBALANCE",
    )

    # --------------------------------------------------------
    # 8. Reconcile both reporting frequencies to source events
    # --------------------------------------------------------

    for frequency, result in (
        ("quarterly", quarterly_df),
        ("annual", annual_df),
    ):

        totals = (
            result
            .groupby("portfolio_id")
            .agg(
                measured_rebalances=("measured_rebalances", "sum"),
                cumulative_one_way_turnover=(
                    "cumulative_one_way_turnover", "sum"
                ),
                total_purchases=("total_purchases", "sum"),
                total_sales=("total_sales", "sum"),
            )
        )

        for row in portfolio_df.itertuples(index=False):

            observed = totals.loc[row.portfolio_id]

            if (
                int(observed["measured_rebalances"])
                != int(row.measured_rebalances)
            ):
                raise ValueError(
                    f"{frequency}: rebalance counts do not reconcile."
                )

            for field, source_value in (
                (
                    "cumulative_one_way_turnover",
                    row.cumulative_one_way_turnover,
                ),
                ("total_purchases", row.cumulative_purchases),
                ("total_sales", row.cumulative_sales),
            ):
                if not np.isclose(
                    observed[field],
                    source_value,
                    atol=tolerance,
                    rtol=0,
                ):
                    raise ValueError(
                        f"{frequency}: {field} does not reconcile."
                    )

    # Independently reconcile the daily drift magnitude
    # across the two calendar reporting frequencies.

    for frequency, result in (
        ("quarterly", quarterly_df),
        ("annual", annual_df),
    ):

        drift_totals = result.groupby(
            "portfolio_id"
        )["cumulative_price_drift_magnitude"].sum()

        for row in portfolio_df.itertuples(index=False):

            if not np.isclose(
                drift_totals.loc[row.portfolio_id],
                row.cumulative_price_drift_magnitude,
                atol=tolerance,
                rtol=0,
            ):
                raise ValueError(
                    f"{frequency}: daily drift does not reconcile."
                )

    return quarterly_df, annual_df, portfolio_df


# Research reference: 12_turnover_rebalancing_analytics.ipynb, code cell index 16.
def summarize_rebalance_turnover_periods_portable(
    turnover_events_df: pd.DataFrame,
    daily_diagnostics_df: pd.DataFrame,
    *,
    tolerance: float = 1e-10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Support mixed populations with and without recurring rebalances.

    Portfolios containing measured subsequent rebalances use
    the validated Cell 14 calculation unchanged.

    Portfolios without subsequent rebalances use explicit
    inception and passive-only availability conventions.
    """

    if (
        not isinstance(turnover_events_df, pd.DataFrame)
        or not isinstance(daily_diagnostics_df, pd.DataFrame)
        or turnover_events_df.empty
        or daily_diagnostics_df.empty
    ):
        raise ValueError(
            "Reporting inputs must be nonempty DataFrames."
        )

    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Invalid numerical tolerance.")

    required_events = {
        "portfolio_id", "selection_date",
        "effective_rebalance_date", "event_status",
        "turnover_status", "one_way_turnover",
    }

    required_daily = {
        "portfolio_id", "date", "transition_type",
        "scheduled_one_way_turnover",
        "price_drift_half_l1_magnitude",
        "terminal_model_half_l1_adjustment",
        "terminal_zeroed_securities",
        "drift_reconciliation_status",
    }

    if required_events - set(turnover_events_df.columns):
        raise ValueError("Missing required event fields.")

    if required_daily - set(daily_diagnostics_df.columns):
        raise ValueError("Missing required daily fields.")

    events = turnover_events_df.copy(deep=True)
    daily = daily_diagnostics_df.copy(deep=True)

    events["selection_date"] = pd.to_datetime(
        events["selection_date"], errors="raise"
    )

    events["effective_rebalance_date"] = pd.to_datetime(
        events["effective_rebalance_date"], errors="raise"
    )

    daily["date"] = pd.to_datetime(
        daily["date"], errors="raise"
    )

    event_key = ["portfolio_id", "selection_date"]
    daily_key = ["portfolio_id", "date"]

    for name, frame, keys in (
        ("events", events, event_key),
        ("daily", daily, daily_key),
    ):
        if (
            frame[keys].isna().any().any()
            or frame.duplicated(keys).any()
        ):
            raise ValueError(
                f"{name}: null or duplicate primary keys."
            )

    event_ids = set(events["portfolio_id"])
    daily_ids = set(daily["portfolio_id"])

    if event_ids != daily_ids:
        raise ValueError(
            "Event and daily portfolio populations differ."
        )

    # --------------------------------------------------------
    # 2. Separate portfolios by measured rebalance availability
    # --------------------------------------------------------

    measured_ids = set(
        events.loc[
            events["turnover_status"].eq("PASSED"),
            "portfolio_id",
        ]
    )

    no_rebalance_ids = event_ids.difference(measured_ids)

    quarterly_parts = []
    annual_parts = []
    portfolio_parts = []

    # --------------------------------------------------------
    # 3. Preserve the original validated calculation path
    # --------------------------------------------------------

    if measured_ids:

        measured_events = events.loc[
            events["portfolio_id"].isin(measured_ids)
        ]

        measured_daily = daily.loc[
            daily["portfolio_id"].isin(measured_ids)
        ]

        q, y, p = summarize_rebalance_turnover_periods(
            measured_events,
            measured_daily,
            tolerance=tolerance,
        )

        quarterly_parts.append(q)
        annual_parts.append(y)
        portfolio_parts.append(p)

    # --------------------------------------------------------
    # 4. Validate no-rebalance portfolio event histories
    # --------------------------------------------------------

    for portfolio_id in sorted(no_rebalance_ids):

        portfolio_events = events.loc[
            events["portfolio_id"].eq(portfolio_id)
        ].copy()

        portfolio_daily = (
            daily.loc[
                daily["portfolio_id"].eq(portfolio_id)
            ]
            .sort_values("date")
            .reset_index(drop=True)
        )

        initial = portfolio_events.loc[
            portfolio_events["event_status"].eq(
                "INITIAL_ESTABLISHMENT"
            )
        ]

        if (
            len(initial) != 1
            or not portfolio_events["event_status"].isin(
                [
                    "INITIAL_ESTABLISHMENT",
                    "PENDING_AFTER_CUTOFF",
                ]
            ).all()
        ):
            raise ValueError(
                "No-rebalance portfolio requires exactly "
                "one initial event and optional pending selections."
            )

        first_date = portfolio_daily["date"].iloc[0]
        last_date = portfolio_daily["date"].iloc[-1]

        if (
            initial["effective_rebalance_date"].iloc[0]
            != first_date
        ):
            raise ValueError(
                "Initial establishment date differs "
                "from the first observed return date."
            )

        if not initial["turnover_status"].eq(
            "NOT_ASSESSED_INITIAL_ESTABLISHMENT"
        ).all():
            raise ValueError(
                "Initial turnover availability status is invalid."
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
                "Pending selection chronology is invalid."
            )

        if not pending["turnover_status"].eq(
            "NOT_ASSESSED_PENDING_SELECTION"
        ).all():
            raise ValueError(
                "Pending turnover availability status is invalid."
            )

        if portfolio_events["one_way_turnover"].notna().any():
            raise ValueError(
                "No-rebalance events contain numeric turnover."
            )

        # ----------------------------------------------------
        # 5. Validate daily inception and passive classifications
        # ----------------------------------------------------

        first_day = portfolio_daily.iloc[0]

        if first_day["transition_type"] != (
            "INITIAL_ESTABLISHMENT"
        ):
            raise ValueError(
                "No-rebalance history must begin "
                "with initial establishment."
            )

        if pd.notna(
            first_day["scheduled_one_way_turnover"]
        ):
            raise ValueError(
                "Initial establishment must not report turnover."
            )

        passive_days = portfolio_daily.iloc[1:]

        if not passive_days["transition_type"].eq(
            "PASSIVE_CARRYOVER"
        ).all():
            raise ValueError(
                "Unexpected scheduled transition "
                "in no-rebalance portfolio."
            )

        passive_turnover = pd.to_numeric(
            passive_days["scheduled_one_way_turnover"],
            errors="raise",
        ).to_numpy(dtype=float, na_value=np.nan)

        if (
            not np.isfinite(passive_turnover).all()
            or (
                np.abs(passive_turnover) > tolerance
            ).any()
        ):
            raise ValueError(
                "Passive-only dates contain scheduled turnover."
            )

        if not portfolio_daily[
            "drift_reconciliation_status"
        ].eq("PASSED").all():
            raise ValueError(
                "No-rebalance portfolio requires "
                "validated daily price-drift diagnostics."
            )

        diagnostic_columns = [
            "price_drift_half_l1_magnitude",
            "terminal_model_half_l1_adjustment",
            "terminal_zeroed_securities",
        ]

        for column in diagnostic_columns:

            values = pd.to_numeric(
                portfolio_daily[column],
                errors="raise",
            ).to_numpy(dtype=float, na_value=np.nan)

            if (
                not np.isfinite(values).all()
                or (values < -tolerance).any()
            ):
                raise ValueError(
                    f"Invalid no-rebalance diagnostic: {column}"
                )

        # ----------------------------------------------------
        # 6. Construct quarterly and annual passive-only periods
        # ----------------------------------------------------

        for frequency, destination in (
            ("Q", quarterly_parts),
            ("Y", annual_parts),
        ):

            period_data = portfolio_daily.copy()

            period_data["period"] = (
                period_data["date"].dt.to_period(frequency)
            )

            period_records = []

            for period, rows in period_data.groupby(
                "period", sort=True
            ):

                initial_count = int(
                    rows["transition_type"]
                    .eq("INITIAL_ESTABLISHMENT")
                    .sum()
                )

                inception_only = initial_count > 0

                period_is_start = (
                    first_date.to_period(frequency)
                    == period
                )

                period_is_cutoff = (
                    last_date.to_period(frequency)
                    == period
                )

                if period_is_start and period_is_cutoff:
                    boundary_status = "INCEPTION_AND_CUTOFF"
                elif period_is_start:
                    boundary_status = "INCEPTION_PERIOD"
                elif period_is_cutoff:
                    boundary_status = "SAMPLE_CUTOFF_PERIOD"
                else:
                    boundary_status = "INTERIOR_PERIOD"

                no_activity_value = (
                    np.nan if inception_only else 0.0
                )

                terminal_counts = rows[
                    "terminal_zeroed_securities"
                ]

                period_records.append(
                    {
                        "portfolio_id": portfolio_id,
                        "period": period,
                        "observed_trading_days": len(rows),
                        "first_observed_date": rows["date"].min(),
                        "last_observed_date": rows["date"].max(),
                        "initial_establishment_days": initial_count,
                        "scheduled_rebalance_days": 0,
                        "cumulative_price_drift_magnitude": (
                            rows[
                                "price_drift_half_l1_magnitude"
                            ].sum()
                        ),
                        "terminal_adjustment_days": int(
                            terminal_counts.gt(0).sum()
                        ),
                        "terminal_zeroed_securities": int(
                            terminal_counts.sum()
                        ),
                        "cumulative_terminal_adjustment_magnitude": (
                            rows[
                                "terminal_model_half_l1_adjustment"
                            ].sum()
                        ),
                        "measured_rebalances": 0,
                        "cumulative_one_way_turnover": (
                            no_activity_value
                        ),
                        "cumulative_gross_two_way_turnover": (
                            no_activity_value
                        ),
                        "total_purchases": no_activity_value,
                        "total_sales": no_activity_value,
                        "average_rebalance_turnover": np.nan,
                        "maximum_rebalance_turnover": np.nan,
                        "new_positions": no_activity_value,
                        "exited_positions": no_activity_value,
                        "increased_positions": no_activity_value,
                        "decreased_positions": no_activity_value,
                        "turnover_measurement_status": (
                            "NOT_ASSESSED_INITIAL_ONLY"
                            if inception_only
                            else "NO_SCHEDULED_REBALANCE"
                        ),
                        "period_label": str(period),
                        "period_start": (
                            period.start_time.normalize()
                        ),
                        "period_end": (
                            period.end_time.normalize()
                        ),
                        "sample_start_period": period_is_start,
                        "sample_cutoff_period": period_is_cutoff,
                        "sample_boundary_status": boundary_status,
                    }
                )

            destination.append(
                pd.DataFrame.from_records(period_records)
            )

        # ----------------------------------------------------
        # 7. Construct the full-sample unavailable-turnover record
        # ----------------------------------------------------

        observed_years = (
            (last_date - first_date).days / 365.2425
        )

        portfolio_parts.append(
            pd.DataFrame.from_records(
                [
                    {
                        "portfolio_id": portfolio_id,
                        "first_return_date": first_date,
                        "last_return_date": last_date,
                        "observed_trading_days": len(
                            portfolio_daily
                        ),
                        "cumulative_price_drift_magnitude": (
                            portfolio_daily[
                                "price_drift_half_l1_magnitude"
                            ].sum()
                        ),
                        "terminal_adjustment_days": int(
                            portfolio_daily[
                                "terminal_zeroed_securities"
                            ].gt(0).sum()
                        ),
                        "terminal_zeroed_securities": int(
                            portfolio_daily[
                                "terminal_zeroed_securities"
                            ].sum()
                        ),
                        "cumulative_terminal_adjustment_magnitude": (
                            portfolio_daily[
                                "terminal_model_half_l1_adjustment"
                            ].sum()
                        ),
                        "measured_rebalances": 0,
                        "cumulative_one_way_turnover": np.nan,
                        "cumulative_gross_two_way_turnover": np.nan,
                        "cumulative_purchases": np.nan,
                        "cumulative_sales": np.nan,
                        "average_rebalance_turnover": np.nan,
                        "median_rebalance_turnover": np.nan,
                        "maximum_rebalance_turnover": np.nan,
                        "new_positions": np.nan,
                        "exited_positions": np.nan,
                        "observed_calendar_years": observed_years,
                        "arithmetic_turnover_per_observed_year": np.nan,
                        "turnover_measurement_status": (
                            "NOT_ASSESSED_NO_RECURRING_REBALANCE"
                        ),
                    }
                ]
            )
        )

    # --------------------------------------------------------
    # 8. Return stable, complete reporting output contracts
    # --------------------------------------------------------

    quarterly = (
        pd.concat(
            quarterly_parts,
            ignore_index=True,
            sort=False,
        )
        .sort_values(["portfolio_id", "period"])
        .reset_index(drop=True)
    )

    annual = (
        pd.concat(
            annual_parts,
            ignore_index=True,
            sort=False,
        )
        .sort_values(["portfolio_id", "period"])
        .reset_index(drop=True)
    )

    portfolio = (
        pd.concat(
            portfolio_parts,
            ignore_index=True,
            sort=False,
        )
        .sort_values("portfolio_id")
        .reset_index(drop=True)
    )

    for label, frame, keys in (
        (
            "quarterly",
            quarterly,
            ["portfolio_id", "period"],
        ),
        (
            "annual",
            annual,
            ["portfolio_id", "period"],
        ),
        (
            "portfolio",
            portfolio,
            ["portfolio_id"],
        ),
    ):

        if (
            frame[keys].isna().any().any()
            or frame.duplicated(keys).any()
        ):
            raise ValueError(
                f"Invalid {label} reporting output keys."
            )

    return quarterly, annual, portfolio
