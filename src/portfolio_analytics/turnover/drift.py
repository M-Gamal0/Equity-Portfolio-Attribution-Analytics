"""Turnover drift: validated research calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd



# Research reference: 12_turnover_rebalancing_analytics.ipynb, code cell index 9.
def calculate_daily_price_weight_drift(
    holdings_df: pd.DataFrame,
    *,
    weight_tolerance: float = 1e-10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reconstruct price-driven security weight drift and separately
    reconcile hypothetical post-close terminal adjustments.

    Parameters
    ----------
    holdings_df : pd.DataFrame
        Complete security-day holdings ledger.

    weight_tolerance : float
        Absolute numerical tolerance for financial identities.

    Returns
    -------
    detail_df : pd.DataFrame
        One row per (portfolio_id, date, security_id).

    summary_df : pd.DataFrame
        One row per (portfolio_id, date).

    Notes
    -----
    The pre-liquidation field is optional for source-independent
    portability. If unavailable, modeled price drift remains
    calculable, but observed drift and terminal-event reconciliation
    are explicitly not assessed.

    The half-L1 magnitude of price drift is a diagnostic statistic,
    not turnover or executed portfolio trading.
    """

    # --------------------------------------------------------
    # 2. Validate required inputs
    # --------------------------------------------------------

    if not isinstance(holdings_df, pd.DataFrame):
        raise TypeError(
            "holdings_df must be a pandas DataFrame."
        )

    if (
        not np.isfinite(weight_tolerance)
        or weight_tolerance < 0
    ):
        raise ValueError(
            "weight_tolerance must be finite and nonnegative."
        )

    required_columns = {
        "portfolio_id",
        "date",
        "security_id",
        "weight_bop",
        "weight_eop",
        "return_price",
    }

    missing_columns = required_columns.difference(
        holdings_df.columns
    )

    if missing_columns:
        raise ValueError(
            f"Missing holdings columns: {sorted(missing_columns)}"
        )

    if holdings_df.empty:
        raise ValueError(
            "Holdings must be nonempty."
        )

    pre_available = (
        "weight_eop_pre_liquidation"
        in holdings_df.columns
    )

    columns = [
        "portfolio_id",
        "date",
        "security_id",
        "weight_bop",
        "weight_eop",
        "return_price",
    ]

    if pre_available:
        columns.append(
            "weight_eop_pre_liquidation"
        )

    if "sector" in holdings_df.columns:
        columns.append("sector")

    detail = holdings_df[columns].copy()

    detail["date"] = pd.to_datetime(
        detail["date"],
        errors="raise",
    )

    security_key = [
        "portfolio_id",
        "date",
        "security_id",
    ]

    daily_key = [
        "portfolio_id",
        "date",
    ]

    if (
        detail[security_key].isna().any().any()
        or detail.duplicated(security_key).any()
    ):
        raise ValueError(
            "Null or duplicated security-day holdings keys."
        )

    # --------------------------------------------------------
    # 3. Validate security weights and price returns
    # --------------------------------------------------------

    weight_columns = [
        "weight_bop",
        "weight_eop",
    ]

    if pre_available:
        weight_columns.append(
            "weight_eop_pre_liquidation"
        )

    for column in weight_columns + ["return_price"]:

        if not pd.api.types.is_numeric_dtype(
            detail[column]
        ):
            raise ValueError(
                f"{column} must be numeric."
            )

        values = detail[column].to_numpy(
            dtype=float,
            na_value=np.nan,
        )

        if not np.isfinite(values).all():
            raise ValueError(
                f"{column} must be finite and nonmissing."
            )

        if column in weight_columns:

            invalid_weights = (
                (values < -weight_tolerance)
                | (values > 1.0 + weight_tolerance)
            )

            if invalid_weights.any():
                raise ValueError(
                    f"{column} violates long-only bounds."
                )

        elif (values < -1.0).any():

            raise ValueError(
                "Price returns cannot be below -100%."
            )

    # --------------------------------------------------------
    # 4. Reconstruct daily portfolio price returns
    # --------------------------------------------------------

    detail["weighted_price_return"] = (
        detail["weight_bop"]
        * detail["return_price"]
    )

    portfolio_price_returns = (
        detail
        .groupby(daily_key, as_index=False)[
            "weighted_price_return"
        ]
        .sum()
        .rename(
            columns={
                "weighted_price_return":
                    "portfolio_price_return"
            }
        )
    )

    portfolio_price_returns[
        "portfolio_price_growth"
    ] = (
        1.0
        + portfolio_price_returns[
            "portfolio_price_return"
        ]
    )

    growth = portfolio_price_returns[
        "portfolio_price_growth"
    ]

    if (
        ~np.isfinite(growth)
        | growth.le(0)
    ).any():
        raise ValueError(
            "Price growth denominator must be finite "
            "and strictly positive."
        )

    detail = detail.merge(
        portfolio_price_returns,
        on=daily_key,
        how="left",
        validate="many_to_one",
    )

    # --------------------------------------------------------
    # 5. Independently reconstruct price-drifted EOP weights
    # --------------------------------------------------------

    detail[
        "modeled_pre_liquidation_weight"
    ] = (
        detail["weight_bop"]
        * (1.0 + detail["return_price"])
        / detail["portfolio_price_growth"]
    )

    detail[
        "price_drift_weight_change"
    ] = (
        detail["modeled_pre_liquidation_weight"]
        - detail["weight_bop"]
    )

    detail[
        "abs_price_drift_weight_change"
    ] = (
        detail["price_drift_weight_change"].abs()
    )

    # --------------------------------------------------------
    # 6. Compare modeled and observed pre-liquidation weights
    # --------------------------------------------------------

    if pre_available:

        detail[
            "pre_liquidation_abs_error"
        ] = (
            detail["weight_eop_pre_liquidation"]
            - detail["modeled_pre_liquidation_weight"]
        ).abs()

        detail[
            "post_close_weight_adjustment"
        ] = (
            detail["weight_eop"]
            - detail["weight_eop_pre_liquidation"]
        )

        detail[
            "is_terminal_zeroed"
        ] = (
            detail[
                "weight_eop_pre_liquidation"
            ].gt(weight_tolerance)
            & detail[
                "weight_eop"
            ].le(weight_tolerance)
        )

        detail[
            "zeroed_pre_liquidation_weight"
        ] = np.where(
            detail["is_terminal_zeroed"],
            detail["weight_eop_pre_liquidation"],
            0.0,
        )

    else:

        detail[
            "weight_eop_pre_liquidation"
        ] = np.nan

        detail[
            "pre_liquidation_abs_error"
        ] = np.nan

        detail[
            "post_close_weight_adjustment"
        ] = np.nan

        detail[
            "is_terminal_zeroed"
        ] = pd.NA

        detail[
            "zeroed_pre_liquidation_weight"
        ] = np.nan

    # --------------------------------------------------------
    # 7. Identify dates with modeled terminal-security zeroing
    # --------------------------------------------------------

    if pre_available:

        terminal_day_totals = (
            detail
            .groupby(
                daily_key,
                as_index=False,
            )
            .agg(
                terminal_zeroed_securities=(
                    "is_terminal_zeroed",
                    "sum",
                ),
                terminal_zeroed_pre_weight=(
                    "zeroed_pre_liquidation_weight",
                    "sum",
                ),
            )
        )

        detail = detail.merge(
            terminal_day_totals,
            on=daily_key,
            how="left",
            validate="many_to_one",
        )

        terminal_day_mask = (
            detail[
                "terminal_zeroed_securities"
            ].gt(0)
        )

        if (
            detail.loc[
                terminal_day_mask,
                "terminal_zeroed_pre_weight",
            ].ge(1.0 - weight_tolerance)
        ).any():
            raise ValueError(
                "Terminal model has no positive "
                "surviving portfolio weight."
            )

        # ----------------------------------------------------
        # 8. Reconstruct proportional survivor redistribution
        # ----------------------------------------------------

        detail[
            "expected_post_close_weight"
        ] = detail[
            "weight_eop_pre_liquidation"
        ].copy()

        survivor_mask = (
            terminal_day_mask
            & ~detail["is_terminal_zeroed"]
        )

        detail.loc[
            survivor_mask,
            "expected_post_close_weight",
        ] = (
            detail.loc[
                survivor_mask,
                "weight_eop_pre_liquidation",
            ]
            / (
                1.0
                - detail.loc[
                    survivor_mask,
                    "terminal_zeroed_pre_weight",
                ]
            )
        )

        detail.loc[
            detail["is_terminal_zeroed"],
            "expected_post_close_weight",
        ] = 0.0

        detail[
            "post_close_model_abs_error"
        ] = (
            detail["weight_eop"]
            - detail["expected_post_close_weight"]
        ).abs()

    else:

        detail[
            "terminal_zeroed_securities"
        ] = pd.NA

        detail[
            "terminal_zeroed_pre_weight"
        ] = np.nan

        detail[
            "expected_post_close_weight"
        ] = np.nan

        detail[
            "post_close_model_abs_error"
        ] = np.nan

    detail[
        "abs_post_close_adjustment"
    ] = (
        detail[
            "post_close_weight_adjustment"
        ].abs()
    )

    # --------------------------------------------------------
    # 9. Aggregate security diagnostics to portfolio-day grain
    # --------------------------------------------------------

    summary = (
        detail
        .groupby(
            daily_key,
            as_index=False,
        )
        .agg(
            security_rows=(
                "security_id",
                "size",
            ),
            portfolio_price_return=(
                "portfolio_price_return",
                "first",
            ),
            bop_weight_sum=(
                "weight_bop",
                "sum",
            ),
            modeled_pre_liquidation_weight_sum=(
                "modeled_pre_liquidation_weight",
                "sum",
            ),
            eop_weight_sum=(
                "weight_eop",
                "sum",
            ),
            price_drift_abs_weight_sum=(
                "abs_price_drift_weight_change",
                "sum",
            ),
            price_drift_signed_sum=(
                "price_drift_weight_change",
                "sum",
            ),
            max_pre_liquidation_error=(
                "pre_liquidation_abs_error",
                "max",
            ),
            max_post_close_model_error=(
                "post_close_model_abs_error",
                "max",
            ),
            terminal_zeroed_securities=(
                "is_terminal_zeroed",
                "sum",
            ),
            terminal_zeroed_pre_weight=(
                "zeroed_pre_liquidation_weight",
                "sum",
            ),
            post_close_abs_adjustment=(
                "abs_post_close_adjustment",
                "sum",
            ),
            post_close_signed_adjustment=(
                "post_close_weight_adjustment",
                "sum",
            ),
        )
    )

    # Half-L1 is a magnitude statistic, not turnover.

    summary[
        "price_drift_half_l1_magnitude"
    ] = (
        0.5
        * summary[
            "price_drift_abs_weight_sum"
        ]
    )

    summary[
        "terminal_model_half_l1_adjustment"
    ] = (
        0.5
        * summary[
            "post_close_abs_adjustment"
        ]
    )

    # --------------------------------------------------------
    # 10. Validate daily investment and drift identities
    # --------------------------------------------------------

    for column in (
        "bop_weight_sum",
        "modeled_pre_liquidation_weight_sum",
        "eop_weight_sum",
    ):

        error = (
            summary[column] - 1.0
        ).abs()

        if error.gt(weight_tolerance).any():
            raise ValueError(
                f"Daily full-investment reconciliation "
                f"failed: {column}"
            )

    if summary[
        "price_drift_signed_sum"
    ].abs().gt(weight_tolerance).any():
        raise ValueError(
            "Market-induced weight drift "
            "does not net to zero."
        )

    # --------------------------------------------------------
    # 11. Validate observed pre-liquidation and terminal weights
    # --------------------------------------------------------

    if pre_available:

        observed_pre_totals = (
            detail
            .groupby(
                daily_key,
                as_index=False,
            )[
                "weight_eop_pre_liquidation"
            ]
            .sum()
            .rename(
                columns={
                    "weight_eop_pre_liquidation":
                        "observed_pre_weight_sum"
                }
            )
        )

        summary = summary.merge(
            observed_pre_totals,
            on=daily_key,
            how="left",
            validate="one_to_one",
        )

        invalid_pre_weights = (
            summary[
                "observed_pre_weight_sum"
            ].sub(1.0).abs().gt(weight_tolerance)
        )

        invalid_pre_model = (
            summary[
                "max_pre_liquidation_error"
            ].gt(weight_tolerance)
        )

        if (
            invalid_pre_weights.any()
            or invalid_pre_model.any()
        ):
            raise ValueError(
                "Price-only EOP pre-liquidation "
                "reconciliation failed."
            )

        invalid_post_model = (
            summary[
                "max_post_close_model_error"
            ].gt(weight_tolerance)
        )

        invalid_post_net = (
            summary[
                "post_close_signed_adjustment"
            ].abs().gt(weight_tolerance)
        )

        terminal_identity_error = (
            summary[
                "terminal_model_half_l1_adjustment"
            ]
            - summary[
                "terminal_zeroed_pre_weight"
            ]
        ).abs()

        if (
            invalid_post_model.any()
            or invalid_post_net.any()
            or terminal_identity_error.gt(
                weight_tolerance
            ).any()
        ):
            raise ValueError(
                "Hypothetical terminal redistribution "
                "reconciliation failed."
            )

        summary[
            "drift_reconciliation_status"
        ] = "PASSED"

        summary[
            "post_close_adjustment_status"
        ] = np.where(
            summary[
                "terminal_zeroed_securities"
            ].gt(0),
            "PROPORTIONAL_TERMINAL_MODEL_VALIDATED",
            "NO_POST_CLOSE_MODEL_ADJUSTMENT",
        )

    else:

        summary[
            "observed_pre_weight_sum"
        ] = np.nan

        for field in (
            "terminal_zeroed_securities",
            "terminal_zeroed_pre_weight",
            "post_close_abs_adjustment",
            "post_close_signed_adjustment",
            "terminal_model_half_l1_adjustment",
        ):
            summary[field] = np.nan

        summary[
            "drift_reconciliation_status"
        ] = "NOT_ASSESSED_PRE_LIQUIDATION_UNAVAILABLE"

        summary[
            "post_close_adjustment_status"
        ] = "NOT_ASSESSED_PRE_LIQUIDATION_UNAVAILABLE"

    # --------------------------------------------------------
    # 12. Return stable, unique analytical output contracts
    # --------------------------------------------------------

    detail = (
        detail
        .sort_values(security_key)
        .reset_index(drop=True)
    )

    summary = (
        summary
        .sort_values(daily_key)
        .reset_index(drop=True)
    )

    return detail, summary
