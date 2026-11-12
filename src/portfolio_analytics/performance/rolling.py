"""Performance rolling: validated research calculations."""

from __future__ import annotations

from .inputs import validate_performance_returns
from collections.abc import Mapping
import numpy as np
import pandas as pd



# Research reference: 07_performance_analytics.ipynb, code cell index 9.
def compute_rolling_performance(
    returns_df: pd.DataFrame,
    *,
    horizons_months: Mapping[str, int] | None = None,
    inception_baselines: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """
    Calculate calendar-month rolling returns at each
    observed portfolio return date.

    The function requires only daily portfolio returns
    and explicitly supplied inception baselines.

    Returns remain unavailable when the requested
    calendar horizon exceeds available history.
    """

    horizons = (
        {"1M": 1, "3M": 3, "12M": 12}
        if horizons_months is None
        else dict(horizons_months)
    )

    baselines = (
        {}
        if inception_baselines is None
        else dict(inception_baselines)
    )

    if not horizons or any(
        not isinstance(horizon_id, str)
        or not horizon_id
        or not isinstance(months, int)
        or isinstance(months, bool)
        or months <= 0
        for horizon_id, months in horizons.items()
    ):
        raise ValueError(
            "Each horizon requires a nonempty ID "
            "and a positive integer month count."
        )

    # Reuse the established input contract rather
    # than repeat the earlier validation implementation.

    data = validate_performance_returns(
        returns_df[
            ["portfolio_id", "date", "return_total"]
        ]
    )

    results = []

    for portfolio_id, group in data.groupby(
        "portfolio_id",
        sort=True,
    ):

        dates = pd.DatetimeIndex(group["date"])

        date_values = dates.to_numpy(
            dtype="datetime64[ns]"
        )

        daily_returns = group[
            "return_total"
        ].to_numpy(dtype=np.float64)

        n = len(dates)
        end_positions = np.arange(n)

        # Log compounding avoids repeated products
        # over overlapping rolling windows.
        #
        # Complete capital losses are tracked
        # separately to avoid log(0).

        complete_loss = daily_returns == -1.0

        log_gross = np.zeros(
            n,
            dtype=np.float64,
        )

        log_gross[~complete_loss] = np.log1p(
            daily_returns[~complete_loss]
        )

        cumulative_log = np.concatenate(
            ([0.0], np.cumsum(log_gross))
        )

        cumulative_losses = np.concatenate(
            (
                [0],
                np.cumsum(
                    complete_loss.astype(np.int64)
                ),
            )
        )

        baseline = (
            pd.Timestamp(baselines[portfolio_id])
            if portfolio_id in baselines
            else None
        )

        if baseline is not None:

            if (
                baseline.tzinfo is not None
                or baseline != baseline.normalize()
                or baseline >= dates[0]
            ):
                raise ValueError(
                    "Invalid inception baseline for "
                    f"{portfolio_id}."
                )

        for horizon_id, months in horizons.items():

            target_dates = (
                dates - pd.DateOffset(months=months)
            )

            target_values = target_dates.to_numpy(
                dtype="datetime64[ns]"
            )

            # Latest observed return date on or
            # before the calendar anniversary.

            anchor_positions = (
                dates.searchsorted(
                    target_dates,
                    side="right",
                )
                - 1
            )

            observed_anchor = (
                anchor_positions >= 0
            )

            baseline_anchor = np.zeros(
                n,
                dtype=bool,
            )

            if baseline is not None:

                baseline_anchor = (
                    ~observed_anchor
                    & (target_dates >= baseline)
                )

            has_anchor = (
                observed_anchor | baseline_anchor
            )

            anchor_dates = np.full(
                n,
                np.datetime64("NaT"),
                dtype="datetime64[ns]",
            )

            anchor_dates[observed_anchor] = (
                date_values[
                    anchor_positions[observed_anchor]
                ]
            )

            if baseline is not None:

                anchor_dates[baseline_anchor] = (
                    baseline.to_datetime64()
                )

            anchor_sources = np.full(
                n,
                "UNAVAILABLE",
                dtype=object,
            )

            anchor_sources[observed_anchor] = (
                "OBSERVED_RETURN_DATE"
            )

            anchor_sources[baseline_anchor] = (
                "EXPLICIT_INCEPTION_BASELINE"
            )

            # Wealth already extinguished before
            # the start anchor is not a valid
            # self-financing rolling investment.

            previous_loss = (
                cumulative_losses[
                    anchor_positions + 1
                ]
                > 0
            )

            available = (
                has_anchor & ~previous_loss
            )

            statuses = np.full(
                n,
                "UNAVAILABLE_INSUFFICIENT_HISTORY",
                dtype=object,
            )

            statuses[
                has_anchor & previous_loss
            ] = "UNAVAILABLE_ZERO_START_WEALTH"

            statuses[available] = "AVAILABLE"

            observation_counts = np.zeros(
                n,
                dtype=np.int64,
            )

            observation_counts[has_anchor] = (
                end_positions[has_anchor]
                - anchor_positions[has_anchor]
            )

            first_return_dates = np.full(
                n,
                np.datetime64("NaT"),
                dtype="datetime64[ns]",
            )

            first_return_dates[has_anchor] = (
                date_values[
                    anchor_positions[has_anchor] + 1
                ]
            )

            calendar_days = np.full(
                n,
                np.nan,
            )

            adjustment_days = np.full(
                n,
                np.nan,
            )

            calendar_days[has_anchor] = (
                (
                    date_values[has_anchor]
                    - anchor_dates[has_anchor]
                )
                / np.timedelta64(1, "D")
            )

            adjustment_days[has_anchor] = (
                (
                    target_values[has_anchor]
                    - anchor_dates[has_anchor]
                )
                / np.timedelta64(1, "D")
            )

            # Efficient geometric compounding:
            #
            # exp(sum(log(1 + r_t))) - 1
            #
            # The cumulative difference contains
            # only returns after the start anchor.

            log_change = (
                cumulative_log[end_positions + 1]
                - cumulative_log[
                    anchor_positions + 1
                ]
            )

            loss_inside_window = (
                cumulative_losses[end_positions + 1]
                - cumulative_losses[
                    anchor_positions + 1
                ]
            ) > 0

            rolling_returns = np.full(
                n,
                np.nan,
            )

            rolling_returns[available] = (
                np.expm1(log_change[available])
            )

            rolling_returns[
                available & loss_inside_window
            ] = -1.0

            if not np.isfinite(
                rolling_returns[available]
            ).all():
                raise ValueError(
                    "Non-finite rolling returns for "
                    f"{portfolio_id}."
                )

            results.append(
                pd.DataFrame(
                    {
                        "portfolio_id": portfolio_id,
                        "date": dates,
                        "horizon_id": horizon_id,
                        "horizon_months": months,
                        "target_start_date": target_dates,
                        "resolved_start_date": (
                            pd.to_datetime(anchor_dates)
                        ),
                        "first_return_date": (
                            pd.to_datetime(
                                first_return_dates
                            )
                        ),
                        "observation_count": (
                            observation_counts
                        ),
                        "calendar_days": calendar_days,
                        "start_adjustment_days": (
                            adjustment_days
                        ),
                        "period_return": rolling_returns,
                        "start_anchor_source": (
                            anchor_sources
                        ),
                        "window_status": statuses,
                        "session_coverage_status": (
                            "NOT_ASSESSED_EXCHANGE_CALENDAR"
                        ),
                    }
                )
            )

    result_df = (
        pd.concat(results, ignore_index=True)
        .sort_values(
            ["portfolio_id", "date", "horizon_months"]
        )
        .reset_index(drop=True)
    )

    return result_df


# Research reference: 07_performance_analytics.ipynb, code cell index 10.
def compute_rolling_relative_performance(
    rolling_df: pd.DataFrame,
    *,
    portfolio_id: str,
    benchmark_portfolio_id: str,
) -> pd.DataFrame:
    """Compare matched rolling returns for two portfolios."""

    required = {
        "portfolio_id", "date", "horizon_id",
        "horizon_months", "resolved_start_date",
        "observation_count", "period_return",
        "window_status",
    }

    if not required.issubset(rolling_df.columns):
        raise ValueError(
            f"Missing columns: {sorted(required - set(rolling_df.columns))}"
        )

    if portfolio_id == benchmark_portfolio_id:
        raise ValueError("Portfolio and benchmark must differ.")

    keys = ["date", "horizon_id"]

    columns = keys + [
        "horizon_months", "resolved_start_date",
        "observation_count", "period_return",
        "window_status",
    ]

    def select_portfolio(pid, prefix):
        selected = rolling_df.loc[
            rolling_df["portfolio_id"].eq(pid),
            columns,
        ].copy()

        if selected.empty or selected.duplicated(keys).any():
            raise ValueError(
                f"Missing or duplicate rolling observations: {pid}"
            )

        return selected.rename(
            columns={
                column: f"{prefix}_{column}"
                for column in columns
                if column not in keys
            }
        )

    portfolio = select_portfolio(portfolio_id, "portfolio")
    benchmark = select_portfolio(
        benchmark_portfolio_id, "benchmark"
    )

    result = portfolio.merge(
        benchmark,
        on=keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    if not result["_merge"].eq("both").all():
        raise ValueError(
            "Portfolio and benchmark rolling calendars differ."
        )

    result = result.drop(columns="_merge")

    available = (
        result["portfolio_window_status"].eq("AVAILABLE")
        & result["benchmark_window_status"].eq("AVAILABLE")
    )

    # Matched windows must represent the same investment
    # period, not merely have the same ending date.

    for field in (
        "horizon_months",
        "resolved_start_date",
        "observation_count",
    ):
        mismatch = (
            available
            & result[f"portfolio_{field}"].ne(
                result[f"benchmark_{field}"]
            )
        )

        if mismatch.any():
            raise ValueError(
                f"Rolling window alignment mismatch: {field}"
            )

    rp = result["portfolio_period_return"].to_numpy(
        dtype=np.float64
    )

    rb = result["benchmark_period_return"].to_numpy(
        dtype=np.float64
    )

    eligible = available.to_numpy()

    if (
        not np.isfinite(rp[eligible]).all()
        or not np.isfinite(rb[eligible]).all()
        or (rp[eligible] < -1.0).any()
        or (rb[eligible] < -1.0).any()
    ):
        raise ValueError(
            "Available rolling returns must be finite "
            "simple returns >= -100%."
        )

    valid_relative = eligible & (rb > -1.0)

    result["calculation_status"] = np.where(
        valid_relative,
        "AVAILABLE",
        "UNAVAILABLE_INPUT_WINDOW",
    )

    result.loc[
        eligible & ~valid_relative,
        "calculation_status",
    ] = "UNDEFINED_BENCHMARK_WEALTH"

    result["arithmetic_active_return"] = np.nan
    result["geometric_relative_return"] = np.nan

    result.loc[
        eligible,
        "arithmetic_active_return",
    ] = rp[eligible] - rb[eligible]

    result.loc[
        valid_relative,
        "geometric_relative_return",
    ] = (
        (1.0 + rp[valid_relative])
        / (1.0 + rb[valid_relative])
        - 1.0
    )

    result.insert(0, "portfolio_id", portfolio_id)
    result.insert(
        1, "benchmark_portfolio_id", benchmark_portfolio_id
    )

    result = result.rename(
        columns={
            "portfolio_horizon_months": "horizon_months",
            "portfolio_resolved_start_date": (
                "resolved_start_date"
            ),
            "portfolio_observation_count": (
                "observation_count"
            ),
        }
    )

    return (
        result
        .sort_values(["date", "horizon_months"])
        .reset_index(drop=True)
    )


# Research reference: 07_performance_analytics.ipynb, code cell index 10.
def summarize_rolling_outperformance(
    relative_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate descriptive win rates across rolling windows.

    Win/loss/tie counts are based on the arithmetic
    active return. Overlapping observations are not
    independent statistical samples.
    """

    keys = [
        "portfolio_id",
        "benchmark_portfolio_id",
        "horizon_id",
        "horizon_months",
    ]

    records = []

    for group_key, group in relative_df.groupby(
        keys, sort=True
    ):
        valid = group.loc[
            group["calculation_status"].eq("AVAILABLE")
        ]

        active = valid[
            "arithmetic_active_return"
        ].to_numpy(dtype=np.float64)

        geometric = valid[
            "geometric_relative_return"
        ].to_numpy(dtype=np.float64)

        n = len(valid)
        wins = int((active > 0.0).sum())
        losses = int((active < 0.0).sum())
        ties = int((active == 0.0).sum())

        records.append(
            {
                **dict(zip(keys, group_key)),
                "as_of_date": group["date"].max(),
                "total_windows": len(group),
                "available_windows": n,
                "unavailable_windows": len(group) - n,
                "outperforming_windows": wins,
                "underperforming_windows": losses,
                "tied_windows": ties,
                "outperformance_frequency": (
                    wins / n if n else np.nan
                ),
                "mean_arithmetic_active_return": (
                    float(np.mean(active)) if n else np.nan
                ),
                "median_geometric_relative_return": (
                    float(np.median(geometric))
                    if n else np.nan
                ),
            }
        )

    return pd.DataFrame(records)
