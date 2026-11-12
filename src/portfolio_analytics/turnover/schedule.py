"""Turnover schedule: validated research calculations."""

from __future__ import annotations

import pandas as pd



# Research reference: 12_turnover_rebalancing_analytics.ipynb, code cell index 3.
def build_rebalance_event_schedule(
    holdings_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    *,
    enforce_next_return_date: bool = True,
    enforce_return_base_alignment: bool = True,
) -> pd.DataFrame:
    """
    Construct a portfolio-level rebalance-event schedule.

    Parameters
    ----------
    holdings_df : pd.DataFrame
        Standardized daily holdings with portfolio_id and date.
        Dates identify close-to-close return interval ends.

    targets_df : pd.DataFrame
        Standardized target weights with portfolio_id,
        selection_date, security_id, and weight_target.
        Effective and return-base dates are optional.

    enforce_next_return_date : bool
        If True, an observed effective date must equal the
        first available portfolio return date after selection.

    enforce_return_base_alignment : bool
        If True, a recorded return-base date must match the
        preceding observed return date when available.

    Returns
    -------
    pd.DataFrame
        One row per (portfolio_id, selection_date).

        Includes target characteristics, event classification,
        effective dates, calendar diagnostics, and alignment
        statuses.

    Notes
    -----
    The function does not infer executed transactions.

    An undated selection is classified as pending beyond the
    observed return window only when its timing supports that
    classification.

    Effective-date calendar validation does not independently
    verify target-versus-actual security weights.

    Raises
    ------
    TypeError
        An input is not a pandas DataFrame.

    ValueError
        Required schemas, keys, dates, event consistency,
        or enforced timing relationships are invalid.
    """

    # --------------------------------------------------------
    # 2. Validate required structural inputs
    # --------------------------------------------------------

    if not isinstance(holdings_df, pd.DataFrame):
        raise TypeError(
            "holdings_df must be a pandas DataFrame."
        )

    if not isinstance(targets_df, pd.DataFrame):
        raise TypeError(
            "targets_df must be a pandas DataFrame."
        )

    required_holdings_columns = {
        "portfolio_id",
        "date",
    }

    required_target_columns = {
        "portfolio_id",
        "selection_date",
        "security_id",
        "weight_target",
    }

    missing_holdings = required_holdings_columns.difference(
        holdings_df.columns
    )

    missing_targets = required_target_columns.difference(
        targets_df.columns
    )

    if missing_holdings:
        raise ValueError(
            f"Holdings missing columns: "
            f"{sorted(missing_holdings)}"
        )

    if missing_targets:
        raise ValueError(
            f"Targets missing columns: "
            f"{sorted(missing_targets)}"
        )

    if holdings_df.empty or targets_df.empty:
        raise ValueError(
            "Holdings and targets must both be nonempty."
        )

    # Work on independent copies.

    holdings = holdings_df[
        ["portfolio_id", "date"]
    ].copy()

    target_columns = [
        "portfolio_id",
        "selection_date",
        "security_id",
        "weight_target",
    ]

    optional_columns = [
        "effective_rebalance_date",
        "return_base_date",
        "sector",
    ]

    target_columns += [
        column
        for column in optional_columns
        if column in targets_df.columns
    ]

    targets = targets_df[target_columns].copy()

    effective_field_available = (
        "effective_rebalance_date" in targets.columns
    )

    return_base_field_available = (
        "return_base_date" in targets.columns
    )

    if not effective_field_available:
        targets["effective_rebalance_date"] = pd.NaT

    if not return_base_field_available:
        targets["return_base_date"] = pd.NaT

    # --------------------------------------------------------
    # 3. Normalize and validate date identifiers
    # --------------------------------------------------------

    holdings["date"] = pd.to_datetime(
        holdings["date"],
        errors="raise",
    )

    for column in (
        "selection_date",
        "effective_rebalance_date",
        "return_base_date",
    ):

        targets[column] = pd.to_datetime(
            targets[column],
            errors="raise",
        )

    if holdings[
        ["portfolio_id", "date"]
    ].isna().any().any():
        raise ValueError(
            "Holdings contain null portfolio IDs or dates."
        )

    if targets[
        ["portfolio_id", "selection_date", "security_id"]
    ].isna().any().any():
        raise ValueError(
            "Targets contain null primary-key values."
        )

    if targets.duplicated(
        ["portfolio_id", "selection_date", "security_id"]
    ).any():
        raise ValueError(
            "Duplicate security-level target keys."
        )

    holdings_ids = set(holdings["portfolio_id"])
    target_ids = set(targets["portfolio_id"])

    if holdings_ids != target_ids:
        raise ValueError(
            "Holdings and targets portfolio IDs differ."
        )

    # --------------------------------------------------------
    # 4. Aggregate security targets into selection events
    # --------------------------------------------------------

    events = (
        targets
        .groupby(
            ["portfolio_id", "selection_date"],
            as_index=False,
        )
        .agg(
            target_securities=("security_id", "size"),
            target_weight_sum=("weight_target", "sum"),
            dated_target_rows=(
                "effective_rebalance_date",
                "count",
            ),
            distinct_effective_dates=(
                "effective_rebalance_date",
                "nunique",
            ),
            effective_rebalance_date=(
                "effective_rebalance_date",
                "min",
            ),
            base_dated_rows=(
                "return_base_date",
                "count",
            ),
            distinct_return_base_dates=(
                "return_base_date",
                "nunique",
            ),
            return_base_date=(
                "return_base_date",
                "min",
            ),
        )
    )

    if "sector" in targets.columns:

        sector_counts = (
            targets
            .groupby(
                ["portfolio_id", "selection_date"]
            )["sector"]
            .nunique()
            .rename("target_sectors")
            .reset_index()
        )

        events = events.merge(
            sector_counts,
            on=["portfolio_id", "selection_date"],
            how="left",
            validate="one_to_one",
        )

    else:

        events["target_sectors"] = pd.NA

    # --------------------------------------------------------
    # 5. Validate within-event date consistency
    # --------------------------------------------------------

    mixed_effective_dates = (
        events["dated_target_rows"].gt(0)
        & events["dated_target_rows"].ne(
            events["target_securities"]
        )
    )

    multiple_effective_dates = (
        events["distinct_effective_dates"].gt(1)
    )

    if (
        mixed_effective_dates.any()
        or multiple_effective_dates.any()
    ):
        raise ValueError(
            "Inconsistent effective dates within a "
            "rebalance selection event."
        )

    mixed_base_dates = (
        events["base_dated_rows"].gt(0)
        & events["base_dated_rows"].ne(
            events["target_securities"]
        )
    )

    multiple_base_dates = (
        events["distinct_return_base_dates"].gt(1)
    )

    if (
        mixed_base_dates.any()
        or multiple_base_dates.any()
    ):
        raise ValueError(
            "Inconsistent return-base dates within a "
            "rebalance selection event."
        )

    dated_events = events[
        "effective_rebalance_date"
    ].notna()

    if (
        events.loc[
            dated_events,
            "effective_rebalance_date",
        ].array
        <= events.loc[
            dated_events,
            "selection_date",
        ].array
    ).any():
        raise ValueError(
            "Effective dates must follow selection dates."
        )

    base_dated_events = events[
        "return_base_date"
    ].notna()

    if (
        events.loc[
            base_dated_events,
            "return_base_date",
        ].array
        > events.loc[
            base_dated_events,
            "selection_date",
        ].array
    ).any():
        raise ValueError(
            "Return-base dates cannot follow selection dates."
        )

    both_dates_available = (
        dated_events & base_dated_events
    )

    if (
        events.loc[
            both_dates_available,
            "return_base_date",
        ].array
        >= events.loc[
            both_dates_available,
            "effective_rebalance_date",
        ].array
    ).any():
        raise ValueError(
            "Return-base dates must precede effective dates."
        )

    duplicate_effective_events = events.loc[
        dated_events
    ].duplicated(
        ["portfolio_id", "effective_rebalance_date"]
    )

    if duplicate_effective_events.any():
        raise ValueError(
            "Multiple selection events have the same "
            "effective date within a portfolio."
        )

    # --------------------------------------------------------
    # 6. Build independent observed portfolio calendars
    # --------------------------------------------------------

    portfolio_calendars = {}

    for portfolio_id, portfolio_rows in holdings.groupby(
        "portfolio_id"
    ):

        calendar = pd.DatetimeIndex(
            portfolio_rows["date"].unique()
        ).sort_values()

        if len(calendar) == 0:
            raise ValueError(
                f"Empty holdings calendar: {portfolio_id}"
            )

        portfolio_calendars[portfolio_id] = calendar

    # --------------------------------------------------------
    # 7. Classify and validate individual selection events
    # --------------------------------------------------------

    event_records = []

    for event in events.sort_values(
        ["portfolio_id", "selection_date"]
    ).itertuples(index=False):

        portfolio_id = event.portfolio_id

        calendar = portfolio_calendars[portfolio_id]

        selection_date = pd.Timestamp(
            event.selection_date
        )

        effective_date = event.effective_rebalance_date

        return_base_date = event.return_base_date

        first_holdings_date = calendar[0]
        last_holdings_date = calendar[-1]

        # First observed return interval strictly after
        # the target selection date.

        next_position = int(
            calendar.searchsorted(
                selection_date,
                side="right",
            )
        )

        first_eligible_return_date = (
            calendar[next_position]
            if next_position < len(calendar)
            else pd.NaT
        )

        preceding_holdings_date = pd.NaT

        timing_status = "NOT_ASSESSED"

        base_alignment_status = "NOT_ASSESSED"

        implementation_status = "NOT_ASSESSED"

        # ----------------------------------------------------
        # 7A. Handle unavailable effective-date information
        # ----------------------------------------------------

        if not effective_field_available:

            event_status = (
                "NOT_ASSESSED_EFFECTIVE_DATE_ABSENT"
            )

            implementation_status = "NOT_ASSESSED"

        # ----------------------------------------------------
        # 7B. Handle explicitly undated selection events
        # ----------------------------------------------------

        elif pd.isna(effective_date):

            if (
                selection_date >= last_holdings_date
                and pd.isna(first_eligible_return_date)
            ):

                event_status = "PENDING_AFTER_CUTOFF"

            else:

                event_status = "UNDATED_UNRESOLVED"

            timing_status = "NOT_APPLICABLE"

            base_alignment_status = "NOT_APPLICABLE"

            implementation_status = "NOT_APPLICABLE"

        # ----------------------------------------------------
        # 7C. Handle dated selection events
        # ----------------------------------------------------

        else:

            effective_date = pd.Timestamp(
                effective_date
            )

            effective_position = int(
                calendar.searchsorted(
                    effective_date,
                    side="left",
                )
            )

            effective_in_calendar = (
                effective_position < len(calendar)
                and calendar[effective_position] == effective_date
            )

            if not effective_in_calendar:

                if (
                    first_holdings_date
                    <= effective_date
                    <= last_holdings_date
                ):
                    raise ValueError(
                        f"Effective date {effective_date.date()} "
                        f"for {portfolio_id} is missing from "
                        "the observed holdings calendar."
                    )

                event_status = (
                    "DATED_OUTSIDE_OBSERVED_WINDOW"
                )

                timing_status = (
                    "NOT_ASSESSED_OUTSIDE_WINDOW"
                )

                base_alignment_status = (
                    "NOT_ASSESSED_OUTSIDE_WINDOW"
                )

                implementation_status = "NOT_ASSESSED"

            else:

                # Validate the approved next-return-date
                # convention without using fixed calendars.

                if effective_date == first_eligible_return_date:

                    timing_status = (
                        "ALIGNED_FIRST_RETURN_AFTER_SELECTION"
                    )

                else:

                    timing_status = (
                        "LATER_THAN_FIRST_ELIGIBLE_RETURN"
                    )

                    if enforce_next_return_date:

                        raise ValueError(
                            f"{portfolio_id}: effective date "
                            f"{effective_date.date()} is not "
                            "the first observed return date "
                            "after selection "
                            f"{selection_date.date()}."
                        )

                # Determine the preceding recorded close.

                if effective_position > 0:

                    preceding_holdings_date = (
                        calendar[effective_position - 1]
                    )

                # Distinguish historical establishment from
                # later changes to an existing portfolio.

                if effective_date == first_holdings_date:

                    event_status = (
                        "INITIAL_ESTABLISHMENT"
                    )

                else:

                    event_status = (
                        "EFFECTIVE_REBALANCE"
                    )

                # Calendar presence is not equivalent to
                # verified target implementation.

                implementation_status = "NOT_ASSESSED"

                # ------------------------------------------------
                # 7D. Validate return-base date when assessable
                # ------------------------------------------------

                if not return_base_field_available:

                    base_alignment_status = (
                        "NOT_ASSESSED_BASE_FIELD_ABSENT"
                    )

                elif pd.isna(return_base_date):

                    base_alignment_status = (
                        "NOT_ASSESSED_BASE_DATE_MISSING"
                    )

                elif pd.isna(preceding_holdings_date):

                    # Inception base date predates the first
                    # observed portfolio holdings return.
                    #
                    # Its chronology was checked above, but
                    # prior-holdings equality is unavailable.

                    base_alignment_status = (
                        "NOT_ASSESSED_BASE_OUTSIDE_HISTORY"
                    )

                elif (
                    pd.Timestamp(return_base_date)
                    == preceding_holdings_date
                ):

                    base_alignment_status = (
                        "MATCHED_PREVIOUS_OBSERVED_CLOSE"
                    )

                else:

                    base_alignment_status = (
                        "MISMATCH_PREVIOUS_OBSERVED_CLOSE"
                    )

                    if enforce_return_base_alignment:

                        raise ValueError(
                            f"{portfolio_id}: return-base "
                            f"date mismatch for selection "
                            f"{selection_date.date()}."
                        )

        # ----------------------------------------------------
        # 7E. Save one standardized event record
        # ----------------------------------------------------

        event_records.append(
            {
                "portfolio_id": portfolio_id,
                "selection_date": selection_date,
                "effective_rebalance_date": effective_date,
                "return_base_date": return_base_date,
                "target_securities": event.target_securities,
                "target_sectors": event.target_sectors,
                "target_weight_sum": event.target_weight_sum,
                "first_holdings_date": first_holdings_date,
                "last_holdings_date": last_holdings_date,
                "first_eligible_return_date": (
                    first_eligible_return_date
                ),
                "preceding_holdings_date": (
                    preceding_holdings_date
                ),
                "event_status": event_status,
                "timing_status": timing_status,
                "return_base_alignment_status": (
                    base_alignment_status
                ),
                "implementation_status": (
                    implementation_status
                ),
            }
        )

    # --------------------------------------------------------
    # 8. Produce stable event-level output
    # --------------------------------------------------------

    result = (
        pd.DataFrame.from_records(event_records)
        .sort_values(
            ["portfolio_id", "selection_date"]
        )
        .reset_index(drop=True)
    )

    result["effective_event_number"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="Int64",
    )

    effective_mask = result["event_status"].isin(
        [
            "INITIAL_ESTABLISHMENT",
            "EFFECTIVE_REBALANCE",
        ]
    )

    effective_rows = (
        result.loc[effective_mask]
        .sort_values(
            ["portfolio_id", "effective_rebalance_date"]
        )
    )

    if not effective_rows.empty:

        event_numbers = (
            effective_rows
            .groupby("portfolio_id")
            .cumcount()
            .add(1)
        )

        result.loc[
            effective_rows.index,
            "effective_event_number",
        ] = event_numbers.to_numpy()

    if result.duplicated(
        ["portfolio_id", "selection_date"]
    ).any():
        raise ValueError(
            "Duplicate event-level output keys."
        )

    return result
