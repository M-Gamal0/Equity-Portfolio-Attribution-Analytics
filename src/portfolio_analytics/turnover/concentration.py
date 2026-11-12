"""Turnover concentration: validated research calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd



# Research reference: 12_turnover_rebalancing_analytics.ipynb, code cell index 17.
def analyze_scheduled_turnover_concentration(
    security_changes_df: pd.DataFrame,
    sector_turnover_df: pd.DataFrame,
    turnover_events_df: pd.DataFrame,
    *,
    tolerance: float = 1e-10,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Calculate event-level and cumulative turnover concentration.

    Returns
    -------
    security_event_df:
        Security turnover contributions and event-level shares.

    sector_event_df:
        Sector turnover contributions and event-level shares.

    event_concentration_df:
        Event-level top-N concentration and HHI metrics,
        including initial and pending events.

    cumulative_security_df:
        Historical security turnover totals, shares, and ranks.

    cumulative_sector_df:
        Historical sector turnover totals, shares, and ranks.

    portfolio_concentration_df:
        Full-sample concentration statistics per portfolio.

    Notes
    -----
    Zero-turnover events have undefined concentration ratios.

    Portfolios with no measured recurring rebalance retain an
    explicit NOT_ASSESSED_NO_REBALANCES status.
    """

    # --------------------------------------------------------
    # 2. Validate required input schemas
    # --------------------------------------------------------

    required_columns = {
        "security": {
            "portfolio_id",
            "selection_date",
            "security_id",
            "weight_pre_rebalance",
            "weight_target",
            "weight_change",
            "implied_buy_weight",
            "implied_sell_weight",
            "one_way_turnover_contribution",
        },
        "sector": {
            "portfolio_id",
            "selection_date",
            "sector",
            "security_rows",
            "buy_weight",
            "sell_weight",
            "one_way_turnover_contribution",
        },
        "events": {
            "portfolio_id",
            "selection_date",
            "turnover_status",
            "one_way_turnover",
            "total_purchases",
            "total_sales",
        },
    }

    source_frames = {
        "security": security_changes_df,
        "sector": sector_turnover_df,
        "events": turnover_events_df,
    }

    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Invalid concentration tolerance.")

    for name, frame in source_frames.items():

        if not isinstance(frame, pd.DataFrame):
            raise TypeError(
                f"{name} input must be a DataFrame."
            )

        missing = required_columns[name] - set(frame.columns)

        if missing:
            raise ValueError(
                f"{name}: missing required columns "
                f"{sorted(missing)}"
            )

    security = security_changes_df.copy(deep=True)
    sector = sector_turnover_df.copy(deep=True)
    events = turnover_events_df.copy(deep=True)

    if events.empty:
        raise ValueError(
            "Turnover-event schedule must be nonempty."
        )

    # --------------------------------------------------------
    # 3. Normalize dates and verify unique analytical keys
    # --------------------------------------------------------

    event_key = ["portfolio_id", "selection_date"]

    for name, frame, key in (
        (
            "security",
            security,
            event_key + ["security_id"],
        ),
        (
            "sector",
            sector,
            event_key + ["sector"],
        ),
        (
            "events",
            events,
            event_key,
        ),
    ):

        frame["selection_date"] = pd.to_datetime(
            frame["selection_date"],
            errors="raise",
        )

        if (
            frame[key].isna().any().any()
            or frame.duplicated(key).any()
        ):
            raise ValueError(
                f"{name}: null or duplicate primary keys."
            )

    sector["sector"] = (
        sector["sector"]
        .astype("string")
        .str.strip()
    )

    if sector["sector"].eq("").any():
        raise ValueError(
            "Empty saved sector classifications."
        )

    if sector.duplicated(
        event_key + ["sector"]
    ).any():
        raise ValueError(
            "Duplicate sector keys after label normalization."
        )

    # --------------------------------------------------------
    # 4. Validate security and sector turnover measurements
    # --------------------------------------------------------

    security_numeric = [
        "weight_pre_rebalance",
        "weight_target",
        "weight_change",
        "implied_buy_weight",
        "implied_sell_weight",
        "one_way_turnover_contribution",
    ]

    sector_numeric = [
        "security_rows",
        "buy_weight",
        "sell_weight",
        "one_way_turnover_contribution",
    ]

    for name, frame, columns in (
        ("security", security, security_numeric),
        ("sector", sector, sector_numeric),
    ):

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
                    f"{name}: nonfinite {column}."
                )

            if column != "weight_change":

                if (values < -tolerance).any():
                    raise ValueError(
                        f"{name}: negative {column}."
                    )

            if column in (
                "weight_pre_rebalance",
                "weight_target",
            ):

                if (values > 1.0 + tolerance).any():
                    raise ValueError(
                        f"{name}: weight exceeds unity."
                    )

    if (
        sector["security_rows"].lt(1).any()
        or (
            sector["security_rows"]
            - sector["security_rows"].round()
        ).abs().gt(tolerance).any()
    ):
        raise ValueError(
            "Invalid sector security-row counts."
        )

    # --------------------------------------------------------
    # 5. Independently verify security-level turnover identities
    # --------------------------------------------------------

    delta = (
        security["weight_target"]
        - security["weight_pre_rebalance"]
    )

    expected_security_values = {
        "weight_change": delta,
        "implied_buy_weight": delta.clip(lower=0.0),
        "implied_sell_weight": (-delta).clip(lower=0.0),
        "one_way_turnover_contribution": (
            0.5 * delta.abs()
        ),
    }

    for column, expected in expected_security_values.items():

        errors = (
            security[column] - expected
        ).abs()

        if errors.gt(tolerance).any():
            raise ValueError(
                f"Security turnover identity failed: {column}."
            )

    sector_identity_error = (
        sector["one_way_turnover_contribution"]
        - 0.5 * (
            sector["buy_weight"]
            + sector["sell_weight"]
        )
    ).abs()

    if sector_identity_error.gt(tolerance).any():
        raise ValueError(
            "Sector buy/sell turnover identity failed."
        )

    # --------------------------------------------------------
    # 6. Validate event coverage and event-level trade values
    # --------------------------------------------------------

    measured_events = events.loc[
        events["turnover_status"].eq("PASSED")
    ].copy()

    unavailable_events = events.loc[
        ~events["turnover_status"].eq("PASSED")
    ]

    if unavailable_events[
        "one_way_turnover"
    ].notna().any():
        raise ValueError(
            "Unavailable events contain numeric turnover."
        )

    expected_event_index = pd.MultiIndex.from_frame(
        measured_events[event_key]
    )

    for name, frame in (
        ("security", security),
        ("sector", sector),
    ):

        observed_event_index = pd.MultiIndex.from_frame(
            frame[event_key].drop_duplicates()
        )

        if (
            len(observed_event_index)
            != len(expected_event_index)
            or not observed_event_index.isin(
                expected_event_index
            ).all()
        ):
            raise ValueError(
                f"{name}: measured event coverage mismatch."
            )

    for column in (
        "one_way_turnover",
        "total_purchases",
        "total_sales",
    ):

        values = pd.to_numeric(
            measured_events[column],
            errors="raise",
        ).to_numpy(
            dtype=float,
            na_value=np.nan,
        )

        if (
            not np.isfinite(values).all()
            or (values < -tolerance).any()
        ):
            raise ValueError(
                f"Invalid measured event metric: {column}."
            )

    if (
        measured_events["total_purchases"]
        - measured_events["total_sales"]
    ).abs().gt(tolerance).any():
        raise ValueError(
            "Event-level purchase/sale balance failed."
        )

    # --------------------------------------------------------
    # 7. Independently reconcile security and sector aggregates
    # --------------------------------------------------------

    security_totals = (
        security.groupby(event_key, as_index=False)
        .agg(
            security_trade_rows=("security_id", "size"),
            security_one_way=(
                "one_way_turnover_contribution", "sum"
            ),
            security_buys=("implied_buy_weight", "sum"),
            security_sells=("implied_sell_weight", "sum"),
        )
    )

    sector_totals = (
        sector.groupby(event_key, as_index=False)
        .agg(
            sector_rows=("sector", "size"),
            sector_security_rows=("security_rows", "sum"),
            sector_one_way=(
                "one_way_turnover_contribution", "sum"
            ),
            sector_buys=("buy_weight", "sum"),
            sector_sells=("sell_weight", "sum"),
        )
    )

    audit = (
        measured_events[
            event_key + [
                "one_way_turnover",
                "total_purchases",
                "total_sales",
            ]
        ]
        .merge(
            security_totals,
            on=event_key,
            how="left",
            validate="one_to_one",
        )
        .merge(
            sector_totals,
            on=event_key,
            how="left",
            validate="one_to_one",
        )
    )

    if audit[
        ["security_trade_rows", "sector_rows"]
    ].isna().any().any():
        raise ValueError(
            "Missing measured event aggregation."
        )

    source_reconciliations = [
        ("security_one_way", "one_way_turnover"),
        ("sector_one_way", "one_way_turnover"),
        ("security_buys", "total_purchases"),
        ("sector_buys", "total_purchases"),
        ("security_sells", "total_sales"),
        ("sector_sells", "total_sales"),
        ("security_trade_rows", "sector_security_rows"),
    ]

    for observed, expected in source_reconciliations:

        error = (
            audit[observed] - audit[expected]
        ).abs()

        if error.gt(tolerance).any():
            raise ValueError(
                f"Concentration reconciliation failed: {observed}."
            )

    audit["security_turnover_error"] = (
        audit["security_one_way"]
        - audit["one_way_turnover"]
    ).abs()

    audit["sector_turnover_error"] = (
        audit["sector_one_way"]
        - audit["one_way_turnover"]
    ).abs()

    audit["sector_security_count_error"] = (
        audit["sector_security_rows"]
        - audit["security_trade_rows"]
    ).abs()

    # --------------------------------------------------------
    # 8. Calculate event-level contributor shares and rankings
    # --------------------------------------------------------

    eligible_turnover = measured_events[
        event_key + ["one_way_turnover"]
    ]

    security = security.merge(
        eligible_turnover,
        on=event_key,
        how="left",
        validate="many_to_one",
    )

    sector = sector.merge(
        eligible_turnover,
        on=event_key,
        how="left",
        validate="many_to_one",
    )

    for frame, identifier, top_counts in (
        (security, "security_id", (1, 5, 10)),
        (sector, "sector", (1, 3)),
    ):

        frame["turnover_share"] = np.where(
            frame["one_way_turnover"].gt(tolerance),
            (
                frame["one_way_turnover_contribution"]
                / frame["one_way_turnover"]
            ),
            np.nan,
        )

        frame.sort_values(
            event_key + [
                "one_way_turnover_contribution",
                identifier,
            ],
            ascending=[True, True, False, True],
            kind="mergesort",
            inplace=True,
        )

        frame["event_turnover_rank"] = (
            frame.groupby(event_key).cumcount() + 1
        )

        frame["share_hhi_term"] = (
            frame["turnover_share"].pow(2)
        )

        frame["nonzero_trade_indicator"] = (
            frame["one_way_turnover_contribution"]
            .gt(tolerance)
            .astype(int)
        )

        for top_count in top_counts:

            frame[f"top_{top_count}_share_term"] = (
                np.where(
                    frame["event_turnover_rank"].le(
                        top_count
                    ),
                    frame["turnover_share"],
                    0.0,
                )
            )

        frame.reset_index(
            drop=True,
            inplace=True,
        )

    # --------------------------------------------------------
    # 9. Aggregate event-level turnover concentration
    # --------------------------------------------------------

    def aggregate_event_concentration(
        frame: pd.DataFrame,
        label: str,
        top_counts: tuple[int, ...],
    ) -> pd.DataFrame:

        aggregation = {
            f"{label}_rows": (
                "turnover_share", "size"
            ),
            f"{label}_nonzero_rows": (
                "nonzero_trade_indicator", "sum"
            ),
            f"{label}_hhi": (
                "share_hhi_term", "sum"
            ),
        }

        for top_count in top_counts:

            aggregation[
                f"top_{top_count}_{label}_share"
            ] = (
                f"top_{top_count}_share_term",
                "sum",
            )

        return (
            frame.groupby(
                event_key,
                as_index=False,
            )
            .agg(**aggregation)
        )

    event_security = aggregate_event_concentration(
        security,
        "security",
        (1, 5, 10),
    )

    event_sector = aggregate_event_concentration(
        sector,
        "sector",
        (1, 3),
    )

    event_concentration = (
        events[
            event_key + [
                "turnover_status",
                "one_way_turnover",
            ]
        ]
        .merge(
            event_security,
            on=event_key,
            how="left",
            validate="one_to_one",
        )
        .merge(
            event_sector,
            on=event_key,
            how="left",
            validate="one_to_one",
        )
        .merge(
            audit[
                event_key + [
                    "security_turnover_error",
                    "sector_turnover_error",
                    "sector_security_count_error",
                ]
            ],
            on=event_key,
            how="left",
            validate="one_to_one",
        )
    )

    measured_mask = event_concentration[
        "turnover_status"
    ].eq("PASSED")

    positive_mask = (
        measured_mask
        & event_concentration[
            "one_way_turnover"
        ].gt(tolerance)
    )

    zero_turnover_mask = (
        measured_mask & ~positive_mask
    )

    concentration_columns = [
        "security_hhi",
        "sector_hhi",
        "top_1_security_share",
        "top_5_security_share",
        "top_10_security_share",
        "top_1_sector_share",
        "top_3_sector_share",
    ]

    event_concentration.loc[
        zero_turnover_mask,
        concentration_columns,
    ] = np.nan

    event_concentration[
        "security_effective_contributors"
    ] = (
        1.0
        / event_concentration["security_hhi"]
    )

    event_concentration[
        "sector_effective_contributors"
    ] = (
        1.0
        / event_concentration["sector_hhi"]
    )

    event_concentration[
        "concentration_status"
    ] = np.select(
        [
            positive_mask,
            zero_turnover_mask,
        ],
        [
            "PASSED",
            "NOT_APPLICABLE_ZERO_TURNOVER",
        ],
        default="NOT_ASSESSED",
    )

    for column in (
        "security_hhi",
        "sector_hhi",
    ):

        values = event_concentration.loc[
            positive_mask, column
        ]

        if (
            values.isna().any()
            or values.le(0.0).any()
            or values.gt(1.0 + tolerance).any()
        ):
            raise ValueError(
                f"Invalid event-level turnover HHI: {column}."
            )

    # --------------------------------------------------------
    # 10. Verify contributor shares sum to one per eligible event
    # --------------------------------------------------------

    for label, frame in (
        ("security", security),
        ("sector", sector),
    ):

        event_share_sums = (
            frame.groupby(
                event_key
            )["turnover_share"]
            .sum(min_count=1)
            .rename("sum_share")
            .reset_index()
            .merge(
                eligible_turnover,
                on=event_key,
                validate="one_to_one",
            )
        )

        positive_shares = event_share_sums[
            "one_way_turnover"
        ].gt(tolerance)

        share_error = (
            event_share_sums.loc[
                positive_shares, "sum_share"
            ] - 1.0
        ).abs()

        if share_error.gt(tolerance).any():
            raise ValueError(
                f"{label}: event turnover shares "
                "do not sum to one."
            )

    # --------------------------------------------------------
    # 11. Aggregate cumulative security and sector trading
    # --------------------------------------------------------

    cumulative_security = (
        security.groupby(
            ["portfolio_id", "security_id"],
            as_index=False,
        )
        .agg(
            event_appearances=("selection_date", "nunique"),
            cumulative_one_way=(
                "one_way_turnover_contribution", "sum"
            ),
            cumulative_buys=("implied_buy_weight", "sum"),
            cumulative_sells=("implied_sell_weight", "sum"),
        )
    )

    cumulative_sector = (
        sector.groupby(
            ["portfolio_id", "sector"],
            as_index=False,
        )
        .agg(
            event_appearances=("selection_date", "nunique"),
            cumulative_one_way=(
                "one_way_turnover_contribution", "sum"
            ),
            cumulative_buys=("buy_weight", "sum"),
            cumulative_sells=("sell_weight", "sum"),
        )
    )

    portfolio_totals = (
        measured_events.groupby(
            "portfolio_id",
            as_index=False,
        )
        .agg(
            measured_rebalances=("selection_date", "size"),
            cumulative_one_way_turnover=(
                "one_way_turnover", "sum"
            ),
        )
    )

    for frame, identifier in (
        (cumulative_security, "security_id"),
        (cumulative_sector, "sector"),
    ):

        denominator = frame[
            ["portfolio_id", "cumulative_one_way"]
        ].merge(
            portfolio_totals[
                [
                    "portfolio_id",
                    "cumulative_one_way_turnover",
                ]
            ],
            on="portfolio_id",
            validate="many_to_one",
        )

        frame["portfolio_turnover_share"] = (
            denominator["cumulative_one_way"]
            .div(
                denominator["cumulative_one_way_turnover"]
            )
            .where(
                denominator[
                    "cumulative_one_way_turnover"
                ].gt(tolerance)
            )
        )

        frame.sort_values(
            [
                "portfolio_id",
                "cumulative_one_way",
                identifier,
            ],
            ascending=[True, False, True],
            kind="mergesort",
            inplace=True,
        )

        frame["cumulative_rank"] = (
            frame.groupby("portfolio_id").cumcount() + 1
        )

        frame.reset_index(
            drop=True,
            inplace=True,
        )

    # --------------------------------------------------------
    # 12. Construct full-sample portfolio concentration metrics
    # --------------------------------------------------------

    portfolio_concentration = pd.DataFrame(
        {
            "portfolio_id": sorted(
                events["portfolio_id"].unique()
            )
        }
    )

    portfolio_concentration = (
        portfolio_concentration.merge(
            portfolio_totals,
            on="portfolio_id",
            how="left",
            validate="one_to_one",
        )
    )

    portfolio_concentration[
        "measured_rebalances"
    ] = (
        portfolio_concentration[
            "measured_rebalances"
        ]
        .fillna(0)
        .astype(int)
    )

    for label, frame, top_n in (
        ("security", cumulative_security, 10),
        ("sector", cumulative_sector, 3),
    ):

        concentration_inputs = frame.copy()

        concentration_inputs["hhi_term"] = (
            concentration_inputs[
                "portfolio_turnover_share"
            ].pow(2)
        )

        concentration_inputs["top_share_term"] = (
            np.where(
                concentration_inputs[
                    "cumulative_rank"
                ].le(top_n),
                concentration_inputs[
                    "portfolio_turnover_share"
                ],
                0.0,
            )
        )

        count_name = (
            "distinct_securities"
            if label == "security"
            else "distinct_sectors"
        )

        cumulative_metrics = (
            concentration_inputs.groupby(
                "portfolio_id",
                as_index=False,
            )
            .agg(
                **{
                    count_name: (
                        "cumulative_rank", "size"
                    ),
                    f"cumulative_{label}_hhi": (
                        "hhi_term", "sum"
                    ),
                    f"top_{top_n}_cumulative_{label}_share": (
                        "top_share_term", "sum"
                    ),
                }
            )
        )

        portfolio_concentration = (
            portfolio_concentration.merge(
                cumulative_metrics,
                on="portfolio_id",
                how="left",
                validate="one_to_one",
            )
        )

    valid_events = event_concentration.loc[
        event_concentration[
            "concentration_status"
        ].eq("PASSED")
    ]

    event_statistics = (
        valid_events.groupby(
            "portfolio_id",
            as_index=False,
        )
        .agg(
            average_event_top5_security_share=(
                "top_5_security_share", "mean"
            ),
            median_event_top5_security_share=(
                "top_5_security_share", "median"
            ),
            maximum_event_top5_security_share=(
                "top_5_security_share", "max"
            ),
            average_event_top3_sector_share=(
                "top_3_sector_share", "mean"
            ),
        )
    )

    portfolio_concentration = (
        portfolio_concentration.merge(
            event_statistics,
            on="portfolio_id",
            how="left",
            validate="one_to_one",
        )
    )

    portfolio_concentration[
        "concentration_status"
    ] = np.select(
        [
            portfolio_concentration[
                "measured_rebalances"
            ].eq(0),
            portfolio_concentration[
                "cumulative_one_way_turnover"
            ].le(tolerance),
        ],
        [
            "NOT_ASSESSED_NO_REBALANCES",
            "NOT_APPLICABLE_ZERO_TOTAL_TURNOVER",
        ],
        default="PASSED",
    )

    # Concentration is undefined without positive turnover.
    unavailable_concentration = (
        portfolio_concentration["concentration_status"].ne("PASSED")
    )

    undefined_concentration_columns = [
        "cumulative_security_hhi",
        "cumulative_sector_hhi",
        "top_10_cumulative_security_share",
        "top_3_cumulative_sector_share",
    ]

    portfolio_concentration.loc[
        unavailable_concentration,
        undefined_concentration_columns,
    ] = np.nan

    # --------------------------------------------------------
    # 13. Verify cumulative security and sector shares
    # --------------------------------------------------------

    for frame in (
        cumulative_security,
        cumulative_sector,
    ):

        share_sums = (
            frame.groupby("portfolio_id")[
                "portfolio_turnover_share"
            ]
            .sum(min_count=1)
        )

        for portfolio_id, observed_share in (
            share_sums.items()
        ):

            total_turnover = (
                portfolio_totals.set_index(
                    "portfolio_id"
                ).loc[
                    portfolio_id,
                    "cumulative_one_way_turnover",
                ]
            )

            if total_turnover > tolerance:

                if not np.isclose(
                    observed_share,
                    1.0,
                    atol=tolerance,
                    rtol=0,
                ):
                    raise ValueError(
                        "Cumulative concentration shares "
                        "do not reconcile."
                    )

    # --------------------------------------------------------
    # 14. Return stable analytical output contracts
    # --------------------------------------------------------

    return (
        security,
        sector,
        event_concentration.sort_values(
            event_key
        ).reset_index(drop=True),
        cumulative_security,
        cumulative_sector,
        portfolio_concentration,
    )
