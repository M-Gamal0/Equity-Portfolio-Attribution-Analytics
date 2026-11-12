"""Turnover costs: validated research calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd



# Research reference: 12_turnover_rebalancing_analytics.ipynb, code cell index 20.
def calculate_turnover_unit_cost_sensitivity(
    turnover_events_df: pd.DataFrame,
    quarterly_df: pd.DataFrame,
    annual_df: pd.DataFrame,
    portfolio_df: pd.DataFrame,
    *,
    tolerance: float = 1e-10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Calculate the NAV-basis-point sensitivity to an additional
    1 bp cost charged on each dollar of purchases and sales.

    Returns event, quarterly, annual, and portfolio summaries.

    The resulting coefficient is hypothetical and linear:
        NAV cost impact in bp
        = sensitivity coefficient * assumed trading cost in bp.

    Multi-event coefficients are arithmetic sums of individually
    NAV-normalized exposures, not realized cumulative performance.
    """

    sources = {
        "events": turnover_events_df,
        "quarterly": quarterly_df,
        "annual": annual_df,
        "portfolio": portfolio_df,
    }

    required = {
        "events": {
            "portfolio_id", "selection_date",
            "effective_rebalance_date", "turnover_status",
            "one_way_turnover", "gross_two_way_turnover",
            "total_purchases", "total_sales",
        },
        "quarterly": {
            "portfolio_id", "period",
            "turnover_measurement_status",
            "measured_rebalances",
            "cumulative_one_way_turnover",
            "cumulative_gross_two_way_turnover",
            "total_purchases", "total_sales",
        },
        "annual": {
            "portfolio_id", "period",
            "turnover_measurement_status",
            "measured_rebalances",
            "cumulative_one_way_turnover",
            "cumulative_gross_two_way_turnover",
            "total_purchases", "total_sales",
        },
        "portfolio": {
            "portfolio_id",
            "turnover_measurement_status",
            "measured_rebalances",
            "cumulative_one_way_turnover",
            "cumulative_gross_two_way_turnover",
            "cumulative_purchases", "cumulative_sales",
            "arithmetic_turnover_per_observed_year",
        },
    }

    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Invalid unit-cost sensitivity tolerance.")

    for name, frame in sources.items():

        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise ValueError(f"{name}: nonempty DataFrame required.")

        missing = required[name] - set(frame.columns)

        if missing:
            raise ValueError(
                f"{name}: missing required columns {sorted(missing)}"
            )

    events = turnover_events_df.copy(deep=True)
    quarterly = quarterly_df.copy(deep=True)
    annual = annual_df.copy(deep=True)
    portfolio = portfolio_df.copy(deep=True)

    events["selection_date"] = pd.to_datetime(
        events["selection_date"], errors="raise"
    )

    events["effective_rebalance_date"] = pd.to_datetime(
        events["effective_rebalance_date"], errors="raise"
    )

    # --------------------------------------------------------
    # 2. Validate keys and portfolio populations
    # --------------------------------------------------------

    for label, frame, keys in (
        (
            "events", events,
            ["portfolio_id", "selection_date"],
        ),
        (
            "quarterly", quarterly,
            ["portfolio_id", "period"],
        ),
        (
            "annual", annual,
            ["portfolio_id", "period"],
        ),
        (
            "portfolio", portfolio,
            ["portfolio_id"],
        ),
    ):

        if (
            frame[keys].isna().any().any()
            or frame.duplicated(keys).any()
        ):
            raise ValueError(
                f"{label}: invalid primary keys."
            )

    portfolio_ids = set(portfolio["portfolio_id"])

    if any(
        set(frame["portfolio_id"]) != portfolio_ids
        for frame in (events, quarterly, annual)
    ):
        raise ValueError(
            "Cost-sensitivity portfolio populations differ."
        )

    # --------------------------------------------------------
    # 3. Validate measured event notional and trading identities
    # --------------------------------------------------------

    measured_mask = events["turnover_status"].eq("PASSED")

    measured = events.loc[measured_mask].copy()
    unavailable = events.loc[~measured_mask]

    traded_fields = [
        "one_way_turnover",
        "gross_two_way_turnover",
        "total_purchases",
        "total_sales",
    ]

    if unavailable[traded_fields].notna().any().any():
        raise ValueError(
            "Unavailable events must not contain trading notional."
        )

    for column in traded_fields:

        values = pd.to_numeric(
            measured[column], errors="raise"
        ).to_numpy(dtype=float, na_value=np.nan)

        if (
            not np.isfinite(values).all()
            or (values < -tolerance).any()
        ):
            raise ValueError(
                f"Invalid measured event amount: {column}."
            )

    event_identities = {
        "two-sided turnover": (
            measured["gross_two_way_turnover"]
            - 2.0 * measured["one_way_turnover"]
        ),
        "gross purchases plus sales": (
            measured["gross_two_way_turnover"]
            - measured["total_purchases"]
            - measured["total_sales"]
        ),
        "balanced purchases and sales": (
            measured["total_purchases"]
            - measured["total_sales"]
        ),
    }

    for label, residuals in event_identities.items():

        if residuals.abs().gt(tolerance).any():
            raise ValueError(
                f"Event trading-notional identity failed: {label}."
            )

    if measured[
        "effective_rebalance_date"
    ].isna().any():
        raise ValueError(
            "Measured rebalance lacks an effective trading date."
        )

    # --------------------------------------------------------
    # 4. Calculate event-level unit-cost sensitivity
    # --------------------------------------------------------

    event_output = events[
        [
            "portfolio_id", "selection_date",
            "effective_rebalance_date", "turnover_status",
            "one_way_turnover", "total_purchases",
            "total_sales", "gross_two_way_turnover",
        ]
    ].copy()

    # This is the coefficient mapping a 1 bp charge on
    # each traded dollar to basis points of event reference NAV.
    #
    # Example:
    #   purchases = 0.30 NAV
    #   sales     = 0.30 NAV
    #   unit sensitivity = 0.60 bp NAV per 1 bp trading charge.

    event_output["buy_side_sensitivity"] = (
        event_output["total_purchases"]
    )

    event_output["sell_side_sensitivity"] = (
        event_output["total_sales"]
    )

    event_output["nav_bp_per_1bp_traded"] = (
        event_output["gross_two_way_turnover"]
    )

    event_output["cost_sensitivity_status"] = np.where(
        event_output["turnover_status"].eq("PASSED"),
        "UNIT_RATE_SENSITIVITY_ONLY",
        "NOT_ASSESSED",
    )

    # --------------------------------------------------------
    # 5. Define period-level reconciliation and sensitivity
    # --------------------------------------------------------

    def build_period_sensitivity(
        report: pd.DataFrame,
        frequency: str,
    ) -> pd.DataFrame:

        result = report.copy(deep=True)

        eligible = measured.copy()

        eligible["period"] = (
            eligible["effective_rebalance_date"]
            .dt.to_period(frequency)
        )

        event_periods = (
            eligible.groupby(
                ["portfolio_id", "period"],
                as_index=False,
            )
            .agg(
                expected_rebalance_count=(
                    "selection_date", "size"
                ),
                expected_one_way=(
                    "one_way_turnover", "sum"
                ),
                expected_gross=(
                    "gross_two_way_turnover", "sum"
                ),
                expected_buys=(
                    "total_purchases", "sum"
                ),
                expected_sells=(
                    "total_sales", "sum"
                ),
            )
        )

        result = result.merge(
            event_periods,
            on=["portfolio_id", "period"],
            how="outer",
            validate="one_to_one",
            indicator="_event_period_coverage",
        )

        if result["_event_period_coverage"].eq(
            "right_only"
        ).any():
            raise ValueError(
                f"{frequency}: measured event outside "
                "reporting-period coverage."
            )

        expected_count = (
            result["expected_rebalance_count"]
            .fillna(0)
            .astype(int)
        )

        if not result[
            "measured_rebalances"
        ].eq(expected_count).all():
            raise ValueError(
                f"{frequency}: measured rebalance counts differ."
            )

        has_events = expected_count.gt(0)

        if not result.loc[
            has_events, "turnover_measurement_status"
        ].eq("MEASURED").all():
            raise ValueError(
                f"{frequency}: measured-event status mismatch."
            )

        expected_metrics = {
            "cumulative_one_way_turnover": "expected_one_way",
            "cumulative_gross_two_way_turnover": "expected_gross",
            "total_purchases": "expected_buys",
            "total_sales": "expected_sells",
        }

        for observed_column, expected_column in (
            expected_metrics.items()
        ):

            errors = (
                result.loc[has_events, observed_column]
                - result.loc[has_events, expected_column]
            ).abs()

            if (
                errors.isna().any()
                or errors.gt(tolerance).any()
            ):
                raise ValueError(
                    f"{frequency}: {observed_column} "
                    "does not reconcile with rebalance events."
                )

        no_events = ~has_events

        initial_only = (
            no_events
            & result["turnover_measurement_status"].eq(
                "NOT_ASSESSED_INITIAL_ONLY"
            )
        )

        passive_only = (
            no_events
            & result["turnover_measurement_status"].eq(
                "NO_SCHEDULED_REBALANCE"
            )
        )

        if not (
            initial_only | passive_only | has_events
        ).all():
            raise ValueError(
                f"{frequency}: invalid reporting availability."
            )

        if result.loc[
            initial_only,
            list(expected_metrics),
        ].notna().any().any():
            raise ValueError(
                f"{frequency}: inception-only values "
                "must be unavailable."
            )

        if (
            result.loc[
                passive_only,
                list(expected_metrics),
            ].fillna(np.inf).abs().gt(tolerance).any().any()
        ):
            raise ValueError(
                f"{frequency}: passive-only "
                "trading notional must be zero."
            )

        # Check gross two-sided notional and individual legs.
        assessable = has_events | passive_only

        gross_error = (
            result.loc[
                assessable, "cumulative_gross_two_way_turnover"
            ]
            - result.loc[
                assessable, "total_purchases"
            ]
            - result.loc[
                assessable, "total_sales"
            ]
        ).abs()

        if (
            gross_error.isna().any()
            or gross_error.gt(tolerance).any()
        ):
            raise ValueError(
                f"{frequency}: two-sided notional identity failed."
            )

        result["nav_bp_per_1bp_traded"] = (
            result["cumulative_gross_two_way_turnover"]
        )

        result["buy_side_sensitivity"] = (
            result["total_purchases"]
        )

        result["sell_side_sensitivity"] = (
            result["total_sales"]
        )

        result["cost_sensitivity_status"] = np.select(
            [
                initial_only,
                passive_only,
            ],
            [
                "NOT_ASSESSED_INITIAL_ONLY",
                "NO_SCHEDULED_REBALANCE",
            ],
            default="UNIT_RATE_SENSITIVITY_ONLY",
        )

        result = result.drop(
            columns=[
                "_event_period_coverage",
                "expected_rebalance_count",
                "expected_one_way",
                "expected_gross",
                "expected_buys",
                "expected_sells",
            ]
        )

        return (
            result
            .sort_values(["portfolio_id", "period"])
            .reset_index(drop=True)
        )

    # --------------------------------------------------------
    # 6. Calculate quarterly and calendar-year sensitivities
    # --------------------------------------------------------

    quarterly_output = build_period_sensitivity(
        quarterly, "Q"
    )

    annual_output = build_period_sensitivity(
        annual, "Y"
    )

    # --------------------------------------------------------
    # 7. Calculate full-sample and annualized coefficients
    # --------------------------------------------------------

    portfolio_output = portfolio.copy(deep=True)

    portfolio_output["nav_bp_per_1bp_traded"] = (
        portfolio_output["cumulative_gross_two_way_turnover"]
    )

    portfolio_output["arithmetic_nav_bp_per_year_per_1bp_traded"] = (
        2.0
        * portfolio_output[
            "arithmetic_turnover_per_observed_year"
        ]
    )

    portfolio_output["cost_sensitivity_status"] = np.where(
        portfolio_output["measured_rebalances"].gt(0),
        "UNIT_RATE_SENSITIVITY_ONLY",
        "NOT_ASSESSED_NO_RECURRING_REBALANCE",
    )

    # --------------------------------------------------------
    # 8. Reconcile all aggregation levels to measured events
    # --------------------------------------------------------

    event_totals = measured.groupby(
        "portfolio_id"
    )["gross_two_way_turnover"].sum()

    event_counts = measured.groupby(
        "portfolio_id"
    ).size()

    for row in portfolio_output.itertuples(index=False):

        portfolio_id = row.portfolio_id

        count = int(
            event_counts.get(portfolio_id, 0)
        )

        if int(row.measured_rebalances) != count:
            raise ValueError(
                "Full-sample rebalance counts do not reconcile."
            )

        if count == 0:

            if (
                pd.notna(row.nav_bp_per_1bp_traded)
                or pd.notna(
                    row.arithmetic_nav_bp_per_year_per_1bp_traded
                )
            ):
                raise ValueError(
                    "No-rebalance full-sample sensitivity "
                    "must remain unavailable."
                )

            continue

        expected = float(
            event_totals.loc[portfolio_id]
        )

        if not np.isclose(
            row.nav_bp_per_1bp_traded,
            expected,
            atol=tolerance,
            rtol=0,
        ):
            raise ValueError(
                "Full-sample sensitivity does not match events."
            )

        for label, frame in (
            ("quarterly", quarterly_output),
            ("annual", annual_output),
        ):

            observed = (
                frame.loc[
                    frame["portfolio_id"].eq(portfolio_id),
                    "nav_bp_per_1bp_traded",
                ].sum(min_count=1)
            )

            if not np.isclose(
                observed,
                expected,
                atol=tolerance,
                rtol=0,
            ):
                raise ValueError(
                    f"{label}: unit-cost sensitivity "
                    "does not match full-sample trading."
                )

    return (
        event_output.sort_values(
            ["portfolio_id", "selection_date"]
        ).reset_index(drop=True),
        quarterly_output,
        annual_output,
        portfolio_output.sort_values(
            "portfolio_id"
        ).reset_index(drop=True),
    )
