"""Factors linking: validated research calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd



# Research reference: 11_fama_french_factor_analytics.ipynb, code cell index 14.
def link_portfolio_factor_attribution_periods(
    daily_attribution_df: pd.DataFrame,
    period_definitions_df: pd.DataFrame,
    *,
    component_columns: tuple[str, ...],
    modeled_component_columns: tuple[str, ...],
    tolerance: float = 1e-11,
) -> pd.DataFrame:
    """
    Link daily factor attribution to compounded
    portfolio returns over explicitly defined periods.

    Period definitions:
        period_id
        period_type
        period_start_date
        period_end_date

    Output grain:
        portfolio_id, model_id, period_id

    The realized compounded return is calculated
    from all observed portfolio returns in each period.

    Linked components are unavailable if:
        - Any observed return date lacks factor data.
        - A trailing requested period begins before
          the available source return history.

    A calendar period touching a source boundary
    is explicitly labeled as observed-only.

    The method does not redistribute missing exposures.
    """

    # --------------------------------------------------------
    # A. Validate input schema
    # --------------------------------------------------------

    daily_required = {
        "portfolio_id",
        "model_id",
        "date",
        "realized_total_return",
        "factor_available_on_return_date",
        "modeled_holding_count",
        "unmodeled_holding_count",
        "modeled_gross_weight_coverage",
        *component_columns,
    }

    period_required = {
        "period_id",
        "period_type",
        "period_start_date",
        "period_end_date",
    }

    if not daily_required.issubset(
        daily_attribution_df.columns
    ):
        raise ValueError(
            "Daily attribution input columns are incomplete."
        )

    if not period_required.issubset(
        period_definitions_df.columns
    ):
        raise ValueError(
            "Period definition columns are incomplete."
        )

    if (
        not component_columns
        or len(set(component_columns)) != len(component_columns)
    ):
        raise ValueError(
            "Invalid attribution component specification."
        )

    if not set(modeled_component_columns).issubset(
        set(component_columns)
    ):
        raise ValueError(
            "Modeled components are not a subset "
            "of the attribution components."
        )

    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError(
            "Invalid linking tolerance."
        )

    # --------------------------------------------------------
    # B. Prepare chronological daily source data
    # --------------------------------------------------------

    daily = daily_attribution_df[
        list(daily_required)
    ].copy()

    periods = period_definitions_df[
        list(period_required)
    ].copy()

    daily["date"] = pd.to_datetime(
        daily["date"],
        errors="raise",
    ).astype("datetime64[ns]")

    for column in (
        "period_start_date",
        "period_end_date",
    ):

        periods[column] = pd.to_datetime(
            periods[column],
            errors="raise",
        ).astype("datetime64[ns]")

    if daily[
        ["portfolio_id", "model_id", "date"]
    ].isna().any().any():
        raise ValueError(
            "Daily attribution keys contain missing values."
        )

    if daily.duplicated(
        subset=["portfolio_id", "model_id", "date"]
    ).any():
        raise ValueError(
            "Duplicate daily portfolio attribution keys."
        )

    if periods[
        list(period_required)
    ].isna().any().any():
        raise ValueError(
            "Period definitions contain missing values."
        )

    if periods["period_id"].duplicated().any():
        raise ValueError(
            "Duplicate reporting period identifiers."
        )

    if (
        periods["period_start_date"]
        > periods["period_end_date"]
    ).any():
        raise ValueError(
            "Reporting period start exceeds end."
        )

    if not pd.api.types.is_bool_dtype(
        daily["factor_available_on_return_date"]
    ):
        raise TypeError(
            "Factor availability must be Boolean."
        )

    if daily[
        "factor_available_on_return_date"
    ].isna().any():
        raise ValueError(
            "Factor availability contains missing values."
        )

    returns = daily[
        "realized_total_return"
    ].to_numpy(dtype=float)

    if not np.isfinite(returns).all():
        raise ValueError(
            "Daily realized returns must be finite."
        )

    if (returns <= -1.0).any():
        raise ValueError(
            "Beginning-wealth linking requires daily "
            "portfolio returns greater than -100%."
        )

    if not np.isfinite(
        daily["modeled_gross_weight_coverage"]
        .to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Modeled weight coverage must be finite."
        )

    if not daily[
        "modeled_gross_weight_coverage"
    ].between(0, 1 + tolerance).all():
        raise ValueError(
            "Modeled gross weight coverage is invalid."
        )

    for column in (
        "modeled_holding_count",
        "unmodeled_holding_count",
    ):

        if (
            daily[column].isna().any()
            or daily[column].lt(0).any()
        ):
            raise ValueError(
                f"Invalid {column}."
            )

    daily = (
        daily
        .sort_values(["portfolio_id", "date"])
        .reset_index(drop=True)
    )

    periods = (
        periods
        .sort_values(
            [
                "period_start_date",
                "period_end_date",
                "period_id",
            ]
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # C. Establish source date boundaries
    # --------------------------------------------------------

    portfolio_groups = {
        key: group.reset_index(drop=True)
        for key, group in daily.groupby(
            ["portfolio_id", "model_id"],
            sort=False,
            observed=True,
        )
    }

    output_records = []

    # --------------------------------------------------------
    # D. Process each portfolio and requested period
    # --------------------------------------------------------

    for (
        portfolio_id,
        model_id,
    ), portfolio_daily in portfolio_groups.items():

        source_start = portfolio_daily[
            "date"
        ].min()

        source_end = portfolio_daily[
            "date"
        ].max()

        for period in periods.itertuples(index=False):

            period_start = period.period_start_date

            period_end = period.period_end_date

            sample = portfolio_daily.loc[
                portfolio_daily["date"].between(
                    period_start,
                    period_end,
                    inclusive="both",
                )
            ].copy()

            if sample.empty:
                raise ValueError(
                    f"No portfolio observations for "
                    f"period {period.period_id}."
                )

            sample = (
                sample
                .sort_values("date")
                .reset_index(drop=True)
            )

            period_returns = sample[
                "realized_total_return"
            ].to_numpy(dtype=float)

            # ------------------------------------------------
            # E. Calculate independently compounded return
            # ------------------------------------------------

            compounded_return = float(
                np.prod(
                    1.0 + period_returns
                ) - 1.0
            )

            factor_flags = sample[
                "factor_available_on_return_date"
            ].to_numpy(dtype=bool)

            available_days = int(
                factor_flags.sum()
            )

            missing_days = int(
                (~factor_flags).sum()
            )

            source_start_shortfall = (
                period_start < source_start
            )

            source_end_shortfall = (
                period_end > source_end
            )

            if (
                source_start_shortfall
                or source_end_shortfall
            ):

                period_source_status = (
                    "OBSERVED_ONLY_SOURCE_BOUNDARY"
                )

            else:

                period_source_status = (
                    "WITHIN_SOURCE_BOUNDS"
                )

            # Trailing 3Y/5Y/10Y requests require
            # the full requested calendar lookback.

            insufficient_requested_history = (
                period.period_type == "TRAILING"
                and (
                    source_start_shortfall
                    or source_end_shortfall
                )
            )

            modeled_days = sample[
                "modeled_holding_count"
            ].gt(0).to_numpy(dtype=bool)

            if (
                sample[
                    "unmodeled_holding_count"
                ].gt(0).any()
            ):

                model_coverage_status = (
                    "PARTIAL_MODEL_COVERAGE"
                )

            elif not modeled_days.any():

                model_coverage_status = (
                    "NO_MODELED_HOLDINGS"
                )

            else:

                model_coverage_status = (
                    "FULL_MODEL_COVERAGE"
                )

            record = {
                "portfolio_id": portfolio_id,
                "model_id": model_id,
                "period_id": period.period_id,
                "period_type": period.period_type,
                "period_start_date": period_start,
                "period_end_date": period_end,
                "observed_start_date": sample["date"].min(),
                "observed_end_date": sample["date"].max(),
                "observed_return_days": len(sample),
                "factor_available_days": available_days,
                "factor_missing_days": missing_days,
                "factor_coverage_ratio": (
                    available_days / len(sample)
                ),
                "minimum_modeled_gross_weight_coverage": (
                    sample[
                        "modeled_gross_weight_coverage"
                    ].min()
                ),
                "model_coverage_status": (
                    model_coverage_status
                ),
                "period_source_status": (
                    period_source_status
                ),
                "realized_compounded_total_return": (
                    compounded_return
                ),
                "linked_attributed_total_return": np.nan,
                "period_reconciliation_error": np.nan,
            }

            for component in component_columns:

                record[f"linked_{component}"] = np.nan

            # ------------------------------------------------
            # F. Explicit unavailable-period classifications
            # ------------------------------------------------

            if insufficient_requested_history:

                record["link_status"] = (
                    "UNAVAILABLE_INSUFFICIENT_SOURCE_HISTORY"
                )

                output_records.append(record)

                continue

            if missing_days > 0:

                record["link_status"] = (
                    "UNAVAILABLE_MISSING_FACTOR_DATES"
                )

                output_records.append(record)

                continue

            # ------------------------------------------------
            # G. Prepare daily factor attribution components
            # ------------------------------------------------

            component_frame = sample[
                list(component_columns)
            ].copy()

            # A portfolio with no modeled holdings on a
            # particular day contributes a structural zero
            # from modeled effects for THAT day only.
            #
            # Missing security factor coefficients themselves
            # are never assigned zero or estimated here.

            for component in modeled_component_columns:

                no_model_today = (
                    ~modeled_days
                )

                if component_frame.loc[
                    no_model_today,
                    component,
                ].notna().any():
                    raise ValueError(
                        "Nonmissing modeled contribution "
                        "without modeled holdings."
                    )

                component_frame.loc[
                    no_model_today,
                    component,
                ] = 0.0

            component_matrix = component_frame.to_numpy(
                dtype=float
            )

            if not np.isfinite(
                component_matrix
            ).all():
                raise ValueError(
                    "Complete factor-history period "
                    "contains incomplete attribution components."
                )

            # ------------------------------------------------
            # H. Independently reconcile every daily return
            # ------------------------------------------------

            daily_reconstructed = (
                component_matrix.sum(axis=1)
            )

            np.testing.assert_allclose(
                daily_reconstructed,
                period_returns,
                rtol=1e-10,
                atol=tolerance,
                err_msg=(
                    f"Daily attribution mismatch in "
                    f"{period.period_id}."
                ),
            )

            # ------------------------------------------------
            # I. Calculate beginning-of-day accumulated wealth
            # ------------------------------------------------

            beginning_wealth = np.concatenate(
                [
                    np.array([1.0]),
                    np.cumprod(
                        1.0 + period_returns
                    )[:-1],
                ]
            )

            if not np.isfinite(
                beginning_wealth
            ).all():
                raise ValueError(
                    "Beginning-period wealth is non-finite."
                )

            # ------------------------------------------------
            # J. Link all attribution components
            # ------------------------------------------------

            linked_values = (
                beginning_wealth[:, None]
                * component_matrix
            ).sum(axis=0)

            for index, component in enumerate(
                component_columns
            ):

                linked_value = float(
                    linked_values[index]
                )

                if (
                    component in modeled_component_columns
                    and not modeled_days.any()
                ):

                    # Preserve unavailable aggregate modeled
                    # exposure when no security was modeled.
                    record[f"linked_{component}"] = np.nan

                else:

                    record[f"linked_{component}"] = (
                        linked_value
                    )

            linked_total = float(
                linked_values.sum()
            )

            # ------------------------------------------------
            # K. Validate exact compounded-return identity
            # ------------------------------------------------

            np.testing.assert_allclose(
                linked_total,
                compounded_return,
                rtol=1e-10,
                atol=tolerance,
                err_msg=(
                    f"Compounded factor attribution "
                    f"failed for {period.period_id}."
                ),
            )

            record[
                "linked_attributed_total_return"
            ] = linked_total

            record[
                "period_reconciliation_error"
            ] = (
                compounded_return - linked_total
            )

            if not modeled_days.any():

                record["link_status"] = (
                    "LINKED_WITH_NO_MODELED_HOLDINGS"
                )

            elif model_coverage_status == (
                "FULL_MODEL_COVERAGE"
            ):

                record["link_status"] = (
                    "LINKED_FULL_MODEL_COVERAGE"
                )

            else:

                record["link_status"] = (
                    "LINKED_PARTIAL_MODEL_COVERAGE"
                )

            output_records.append(record)

    result = pd.DataFrame(output_records)

    if result.duplicated(
        subset=[
            "portfolio_id",
            "model_id",
            "period_id",
        ]
    ).any():
        raise ValueError(
            "Duplicate linked portfolio-period keys."
        )

    return (
        result
        .sort_values(
            [
                "period_type",
                "period_start_date",
                "portfolio_id",
            ]
        )
        .reset_index(drop=True)
    )


# Research reference: 11_fama_french_factor_analytics.ipynb, code cell index 14.
def calculate_linked_active_factor_attribution(
    linked_portfolio_df: pd.DataFrame,
    *,
    active_portfolio_id: str,
    benchmark_portfolio_id: str,
    component_columns: tuple[str, ...],
    tolerance: float = 1e-11,
) -> pd.DataFrame:
    """
    Calculate active multi-period attribution by subtracting
    independently wealth-linked portfolio and benchmark
    contributions.

    Two separate performance measures are retained:

    1. Arithmetic compounded-return spread:
        cumulative portfolio return - cumulative benchmark

    2. Relative compounded wealth:
        (1 + cumulative portfolio return)
        / (1 + cumulative benchmark return) - 1

    The linked active factor effects reconcile to measure 1,
    not measure 2.
    """

    keys = [
        "model_id",
        "period_id",
        "period_type",
        "period_start_date",
        "period_end_date",
    ]

    comparison_columns = [
        "realized_compounded_total_return",
        "linked_attributed_total_return",
        "observed_return_days",
        "factor_available_days",
        "factor_missing_days",
        "link_status",
        *[
            f"linked_{component}"
            for component in component_columns
        ],
    ]

    portfolio_side = (
        linked_portfolio_df.loc[
            linked_portfolio_df[
                "portfolio_id"
            ].eq(active_portfolio_id),
            [
                *keys,
                *comparison_columns,
            ],
        ]
        .rename(
            columns={
                column: f"portfolio_{column}"
                for column in comparison_columns
            }
        )
    )

    benchmark_side = (
        linked_portfolio_df.loc[
            linked_portfolio_df[
                "portfolio_id"
            ].eq(benchmark_portfolio_id),
            [
                *keys,
                *comparison_columns,
            ],
        ]
        .rename(
            columns={
                column: f"benchmark_{column}"
                for column in comparison_columns
            }
        )
    )

    result = portfolio_side.merge(
        benchmark_side,
        on=keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    if not result["_merge"].eq("both").all():
        raise ValueError(
            "Portfolio and benchmark period sets differ."
        )

    result = result.drop(columns="_merge")

    for field in (
        "observed_return_days",
        "factor_available_days",
        "factor_missing_days",
    ):

        if not result[
            f"portfolio_{field}"
        ].eq(
            result[
                f"benchmark_{field}"
            ]
        ).all():
            raise ValueError(
                f"Portfolio/benchmark {field} mismatch."
            )

    result["portfolio_id"] = active_portfolio_id

    result["benchmark_portfolio_id"] = (
        benchmark_portfolio_id
    )

    portfolio_returns = result[
        "portfolio_realized_compounded_total_return"
    ]

    benchmark_returns = result[
        "benchmark_realized_compounded_total_return"
    ]

    result[
        "active_compounded_return_spread"
    ] = (
        portfolio_returns - benchmark_returns
    )

    result[
        "active_relative_compounded_wealth_return"
    ] = (
        (1.0 + portfolio_returns)
        / (1.0 + benchmark_returns)
        - 1.0
    )

    for component in component_columns:

        result[f"active_linked_{component}"] = (
            result[
                f"portfolio_linked_{component}"
            ]
            - result[
                f"benchmark_linked_{component}"
            ]
        )

    result[
        "active_linked_attributed_total_return"
    ] = (
        result[
            "portfolio_linked_attributed_total_return"
        ]
        - result[
            "benchmark_linked_attributed_total_return"
        ]
    )

    result[
        "active_period_reconciliation_error"
    ] = (
        result["active_compounded_return_spread"]
        - result[
            "active_linked_attributed_total_return"
        ]
    )

    portfolio_linked = result[
        "portfolio_link_status"
    ].str.startswith("LINKED_")

    benchmark_linked = result[
        "benchmark_link_status"
    ].str.startswith("LINKED_")

    valid_link = (
        portfolio_linked & benchmark_linked
    )

    missing_factors = (
        result[
            "portfolio_factor_missing_days"
        ].gt(0)
        | result[
            "benchmark_factor_missing_days"
        ].gt(0)
    )

    result["active_link_status"] = np.select(
        [
            missing_factors,
            ~valid_link,
            (
                result[
                    "portfolio_link_status"
                ].eq("LINKED_FULL_MODEL_COVERAGE")
                & result[
                    "benchmark_link_status"
                ].eq("LINKED_FULL_MODEL_COVERAGE")
            ),
        ],
        [
            "UNAVAILABLE_MISSING_FACTOR_DATES",
            "UNAVAILABLE_INCOMPLETE_LINKED_SIDE",
            "LINKED_FULL_MODEL_COVERAGE",
        ],
        default="LINKED_PARTIAL_MODEL_COVERAGE",
    )

    np.testing.assert_allclose(
        result.loc[
            valid_link,
            "active_linked_attributed_total_return",
        ].to_numpy(dtype=float),
        result.loc[
            valid_link,
            "active_compounded_return_spread",
        ].to_numpy(dtype=float),
        rtol=1e-10,
        atol=tolerance,
    )

    if result.loc[
        ~valid_link,
        "active_linked_attributed_total_return",
    ].notna().any():
        raise ValueError(
            "Unavailable periods received linked "
            "active factor attribution."
        )

    return (
        result
        .sort_values(
            [
                "period_type",
                "period_start_date",
            ]
        )
        .reset_index(drop=True)
    )
