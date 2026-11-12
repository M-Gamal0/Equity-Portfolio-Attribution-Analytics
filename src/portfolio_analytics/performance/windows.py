"""Performance windows: validated research calculations."""

from __future__ import annotations

from .inputs import validate_performance_returns
from collections.abc import Mapping
import numpy as np
import pandas as pd



# Research reference: 07_performance_analytics.ipynb, code cell index 6.
def _performance_calendar_date(value, label: str) -> pd.Timestamp:
    """Require one valid, timezone-naive, date-only value."""

    if value is None or pd.isna(value):
        raise ValueError(f"{label} cannot be missing.")

    date = pd.Timestamp(value)

    if date.tzinfo is not None:
        raise ValueError(f"{label} must be timezone-naive.")

    if date != date.normalize():
        raise ValueError(f"{label} must be date-only.")

    return date


# Research reference: 07_performance_analytics.ipynb, code cell index 6.
def _performance_window_start(
    window_id: str,
    end_date: pd.Timestamp,
    inception_baseline: pd.Timestamp | None,
) -> pd.Timestamp | None:
    """Determine the unadjusted calendar start boundary."""

    if window_id == "SI":
        return inception_baseline

    if window_id == "YTD":
        return pd.Timestamp(
            year=end_date.year - 1,
            month=12,
            day=31,
        )

    if (
        window_id.endswith("Y")
        and window_id[:-1].isdigit()
        and int(window_id[:-1]) > 0
    ):
        return end_date - pd.DateOffset(
            years=int(window_id[:-1])
        )

    raise ValueError(
        f"Unsupported reporting window: {window_id}"
    )


# Research reference: 07_performance_analytics.ipynb, code cell index 6.
def resolve_performance_windows(
    returns_df: pd.DataFrame,
    *,
    as_of_date: str | pd.Timestamp,
    inception_baseline_dates: Mapping | None = None,
    window_ids: tuple[str, ...] = (
        "SI", "YTD", "1Y", "3Y", "5Y", "10Y"
    ),
) -> pd.DataFrame:
    """
    Resolve reporting windows from portfolio return dates.

    Parameters
    ----------
    returns_df:
        Daily return data with columns:
            portfolio_id, date, return_total

    as_of_date:
        Requested reporting cutoff.

    inception_baseline_dates:
        Optional mapping:
            portfolio_id -> inception baseline date

        The baseline is the valuation date immediately
        preceding the first observed daily return interval.

    window_ids:
        SI, YTD, or positive integer calendar-year windows.

    Returns
    -------
    pd.DataFrame:
        One row per portfolio and requested window.

    Window eligibility
    ------------------
    AVAILABLE:
        A valid start anchor and at least one subsequent
        return observation exist.

    UNAVAILABLE_INSUFFICIENT_HISTORY:
        Requested start predates the supplied history.

    UNAVAILABLE_NO_INCEPTION_BASELINE:
        Inception window requires an explicit baseline.

    No exchange-calendar completeness is inferred.
    """

    if not isinstance(returns_df, pd.DataFrame):
        raise TypeError("returns_df must be a DataFrame.")

    required = {"portfolio_id", "date", "return_total"}

    if returns_df.empty or not required.issubset(
        returns_df.columns
    ):
        raise ValueError(
            "Nonempty returns with portfolio_id, date, "
            "and return_total are required."
        )

    if not returns_df.columns.is_unique:
        raise ValueError("Duplicate source column names.")

    if (
        not isinstance(window_ids, tuple)
        or not window_ids
        or len(window_ids) != len(set(window_ids))
    ):
        raise ValueError(
            "window_ids must be a nonempty unique tuple."
        )

    if not all(
        isinstance(window_id, str)
        for window_id in window_ids
    ):
        raise TypeError("Window identifiers must be strings.")

    requested_as_of = _performance_calendar_date(
        as_of_date,
        "as_of_date",
    )

    baselines = (
        {}
        if inception_baseline_dates is None
        else inception_baseline_dates
    )

    if not isinstance(baselines, Mapping):
        raise TypeError(
            "inception_baseline_dates must be a mapping."
        )

    data = returns_df[
        ["portfolio_id", "date", "return_total"]
    ].copy()

    if data.isna().any().any():
        raise ValueError("Null required return fields.")

    if not data["portfolio_id"].map(
        lambda x: isinstance(x, str) and bool(x.strip())
    ).all():
        raise ValueError("Invalid portfolio identifiers.")

    data["date"] = pd.to_datetime(
        data["date"],
        errors="raise",
    )

    if data["date"].dt.tz is not None:
        raise ValueError("Dates must be timezone-naive.")

    if not data["date"].eq(
        data["date"].dt.normalize()
    ).all():
        raise ValueError("Dates must be date-only.")

    if data.duplicated(["portfolio_id", "date"]).any():
        raise ValueError("Duplicate portfolio-date keys.")

    if not pd.api.types.is_numeric_dtype(
        data["return_total"]
    ):
        raise TypeError("Daily returns must be numeric.")

    values = data["return_total"].to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(values).all() or (values < -1).any():
        raise ValueError(
            "Daily returns must be finite and >= -100%."
        )

    # Sorting is independent of source row order.

    data = data.sort_values(
        ["portfolio_id", "date"]
    )

    records = []

    for portfolio_id, group in data.groupby(
        "portfolio_id",
        sort=True,
    ):

        dates = pd.DatetimeIndex(group["date"])

        baseline = None

        if portfolio_id in baselines:
            baseline = _performance_calendar_date(
                baselines[portfolio_id],
                f"Inception baseline for {portfolio_id}",
            )

            if baseline >= dates.min():
                raise ValueError(
                    "Inception baseline must precede the "
                    f"first return date: {portfolio_id}"
                )

        # Resolve the latest actual observation not later
        # than the requested reporting cutoff.

        eligible_end_dates = dates[
            dates <= requested_as_of
        ]

        if len(eligible_end_dates) == 0:
            raise ValueError(
                "No return observation on or before "
                f"{requested_as_of.date()} for {portfolio_id}."
            )

        resolved_end = eligible_end_dates[-1]

        cutoff_status = (
            "EXACT_REQUESTED_DATE"
            if resolved_end == requested_as_of
            else "ROLLED_BACK_TO_OBSERVED_DATE"
        )

        for window_id in window_ids:

            target_start = _performance_window_start(
                window_id,
                resolved_end,
                baseline,
            )

            start_anchor = pd.NaT
            first_return_date = pd.NaT
            observations = 0
            calendar_days = pd.NA
            adjustment_days = pd.NA
            anchor_source = "UNAVAILABLE"

            if target_start is None:

                status = (
                    "UNAVAILABLE_NO_INCEPTION_BASELINE"
                )

            elif (
                baseline is not None
                and target_start < baseline
            ):

                status = (
                    "UNAVAILABLE_INSUFFICIENT_HISTORY"
                )

            else:

                # Use the last observed return date
                # on/before the requested start boundary.
                #
                # Excluding the anchor's own return gives
                # a close-to-close investment window.

                prior_dates = dates[
                    dates <= target_start
                ]

                if len(prior_dates):

                    start_anchor = prior_dates[-1]
                    anchor_source = (
                        "OBSERVED_RETURN_DATE"
                    )

                elif (
                    baseline is not None
                    and baseline <= target_start
                ):

                    start_anchor = baseline
                    anchor_source = (
                        "EXPLICIT_INCEPTION_BASELINE"
                    )

                if pd.isna(start_anchor):

                    status = (
                        "UNAVAILABLE_INSUFFICIENT_HISTORY"
                    )

                else:

                    included = dates[
                        (dates > start_anchor)
                        & (dates <= resolved_end)
                    ]

                    observations = len(included)

                    if observations == 0:

                        status = (
                            "UNAVAILABLE_NO_RETURN_OBSERVATIONS"
                        )

                    else:

                        status = "AVAILABLE"

                        first_return_date = included[0]

                        calendar_days = (
                            resolved_end - start_anchor
                        ).days

                        adjustment_days = (
                            target_start - start_anchor
                        ).days

            records.append(
                {
                    "portfolio_id": portfolio_id,
                    "window_id": window_id,
                    "requested_as_of_date": requested_as_of,
                    "as_of_date": resolved_end,
                    "requested_start_date": (
                        pd.NaT
                        if target_start is None
                        else target_start
                    ),
                    "resolved_start_date": start_anchor,
                    "first_return_date": first_return_date,
                    "last_return_date": resolved_end,
                    "observation_count": observations,
                    "calendar_days": calendar_days,
                    "start_adjustment_days": adjustment_days,
                    "start_anchor_source": anchor_source,
                    "cutoff_status": cutoff_status,
                    "window_status": status,
                    "session_coverage_status": (
                        "NOT_ASSESSED_EXCHANGE_CALENDAR"
                    ),
                }
            )

    result = pd.DataFrame.from_records(records)

    # Explicit date normalization preserves the
    # schema when some windows are unavailable.

    date_fields = [
        "requested_as_of_date",
        "as_of_date",
        "requested_start_date",
        "resolved_start_date",
        "first_return_date",
        "last_return_date",
    ]

    for column in date_fields:
        result[column] = pd.to_datetime(
            result[column]
        )

    result = result.sort_values(
        ["portfolio_id", "as_of_date", "window_id"]
    ).reset_index(drop=True)

    if result.duplicated(
        ["portfolio_id", "as_of_date", "window_id"]
    ).any():
        raise ValueError(
            "Duplicate reporting-window output keys."
        )

    return result


# Research reference: 07_performance_analytics.ipynb, code cell index 7.
def compute_absolute_performance(
    returns_df: pd.DataFrame,
    windows_df: pd.DataFrame,
    *,
    return_column: str = "return_total",
    annualization_day_basis: float = 365.25,
    arithmetic_trading_days: int = 252,
    minimum_cagr_calendar_days: int = 365,
) -> pd.DataFrame:
    """
    Calculate absolute returns over explicitly resolved windows.

    All calculation inputs are passed directly.

    The function does not require:
        - A benchmark.
        - Risk-free returns.
        - A previously calculated wealth index.
        - Another analytical module's results.

    Annualized arithmetic returns are descriptive daily
    mean estimates, not realized compounded annual returns.

    Unavailable windows remain in the output with missing
    performance metrics and explicit status indicators.
    """

    # --------------------------------------------------------
    # Validate configuration.
    # --------------------------------------------------------

    if (
        isinstance(annualization_day_basis, (bool, np.bool_))
        or not np.isscalar(annualization_day_basis)
    ):
        raise TypeError(
            "annualization_day_basis must be numeric."
        )

    try:
        day_basis = float(annualization_day_basis)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(
            "annualization_day_basis must be numeric."
        ) from exc

    if not np.isfinite(day_basis) or day_basis <= 0:
        raise ValueError(
            "annualization_day_basis must be positive and finite."
        )

    if (
        isinstance(arithmetic_trading_days, (bool, np.bool_))
        or not isinstance(
            arithmetic_trading_days, (int, np.integer)
        )
        or arithmetic_trading_days <= 0
    ):
        raise ValueError(
            "arithmetic_trading_days must be a positive integer."
        )

    if (
        isinstance(minimum_cagr_calendar_days, (bool, np.bool_))
        or not isinstance(
            minimum_cagr_calendar_days, (int, np.integer)
        )
        or minimum_cagr_calendar_days <= 0
    ):
        raise ValueError(
            "minimum_cagr_calendar_days must be positive."
        )

    if not isinstance(returns_df, pd.DataFrame):
        raise TypeError("returns_df must be a DataFrame.")

    if not isinstance(windows_df, pd.DataFrame):
        raise TypeError("windows_df must be a DataFrame.")

    if returns_df.empty or windows_df.empty:
        raise ValueError(
            "Returns and window definitions must be nonempty."
        )

    if (
        not returns_df.columns.is_unique
        or not windows_df.columns.is_unique
    ):
        raise ValueError("Duplicate source column names.")

    if (
        return_column in {"portfolio_id", "date"}
        or return_column not in returns_df.columns
    ):
        raise ValueError("Invalid return_column selection.")

    # Reuse the performance input validation contract.
    # A temporary canonical name supports alternative
    # simple-return series without changing the mathematics.

    canonical_returns = returns_df[
        ["portfolio_id", "date", return_column]
    ].rename(
        columns={return_column: "return_total"}
    )

    validated_returns = validate_performance_returns(
        canonical_returns
    )

    required_window_columns = {
        "portfolio_id",
        "window_id",
        "as_of_date",
        "resolved_start_date",
        "first_return_date",
        "last_return_date",
        "observation_count",
        "calendar_days",
        "start_anchor_source",
        "window_status",
    }

    missing = required_window_columns.difference(
        windows_df.columns
    )

    if missing:
        raise ValueError(
            f"Missing window fields: {sorted(missing)}"
        )

    windows = windows_df.copy(deep=True)

    for column in [
        "as_of_date",
        "resolved_start_date",
        "first_return_date",
        "last_return_date",
    ]:
        windows[column] = pd.to_datetime(
            windows[column],
            errors="raise",
        )

        if windows[column].dt.tz is not None:
            raise ValueError(
                f"{column} must be timezone-naive."
            )

        nonnull_dates = windows[column].dropna()

        if not nonnull_dates.eq(
            nonnull_dates.dt.normalize()
        ).all():
            raise ValueError(
                f"{column} must contain date-only values."
            )

    if windows[
        ["portfolio_id", "window_id", "as_of_date"]
    ].isna().any().any():
        raise ValueError(
            "Null reporting-window identifiers detected."
        )

    for identifier in ["portfolio_id", "window_id"]:
        if not windows[identifier].map(
            lambda x: isinstance(x, str) and bool(x.strip())
        ).all():
            raise ValueError(
                f"{identifier} must contain nonempty strings."
            )

    output_keys = [
        "portfolio_id",
        "as_of_date",
        "window_id",
    ]

    if windows.duplicated(output_keys).any():
        raise ValueError(
            "Duplicate performance-window keys detected."
        )

    known_portfolios = set(
        validated_returns["portfolio_id"]
    )

    unknown_portfolios = (
        set(windows["portfolio_id"]) - known_portfolios
    )

    if unknown_portfolios:
        raise ValueError(
            "Missing source returns for portfolio IDs: "
            f"{sorted(unknown_portfolios)}"
        )

    # --------------------------------------------------------
    # Calculate each requested window independently.
    # --------------------------------------------------------

    results = []

    for portfolio_id, group in validated_returns.groupby(
        "portfolio_id",
        sort=False,
    ):

        portfolio_windows = windows.loc[
            windows["portfolio_id"].eq(portfolio_id)
        ]

        if portfolio_windows.empty:
            continue

        group = group.sort_values("date")

        observed_dates = pd.DatetimeIndex(
            group["date"]
        )

        earliest_date = observed_dates.min()

        for _, window in portfolio_windows.iterrows():

            record = window.to_dict()

            # Explicit missing metrics for unavailable windows.

            record.update(
                {
                    "wealth_start_normalized": np.nan,
                    "wealth_end_normalized": np.nan,
                    "period_return": np.nan,
                    "annualized_return_cagr": np.nan,
                    "annualized_arithmetic_mean": np.nan,
                    "calculation_status": "NOT_CALCULATED",
                    "cagr_status": "NOT_CALCULATED",
                    "annualization_day_basis": day_basis,
                    "arithmetic_trading_days": (
                        arithmetic_trading_days
                    ),
                    "minimum_cagr_calendar_days": (
                        minimum_cagr_calendar_days
                    ),
                    "return_column": return_column,
                    "return_units": "simple_decimal",
                }
            )

            if window["window_status"] != "AVAILABLE":

                record["calculation_status"] = (
                    "WINDOW_UNAVAILABLE"
                )

                record["cagr_status"] = (
                    "WINDOW_UNAVAILABLE"
                )

                results.append(record)
                continue

            start = window["resolved_start_date"]
            end = window["as_of_date"]

            if pd.isna(start) or pd.isna(end):
                raise ValueError(
                    "Available window has missing boundaries."
                )

            if start >= end:
                raise ValueError(
                    "Window start must precede its end."
                )

            if window["last_return_date"] != end:
                raise ValueError(
                    "Window end must match its last return date."
                )

            if end not in observed_dates:
                raise ValueError(
                    "Window end is not an observed return date."
                )

            # The start anchor may be:
            #   1. An observed return-interval end date.
            #   2. An explicit pre-first-return baseline.

            anchor_source = window[
                "start_anchor_source"
            ]

            if anchor_source == "OBSERVED_RETURN_DATE":

                if start not in observed_dates:
                    raise ValueError(
                        "Start anchor is not an observed date."
                    )

            elif anchor_source == "EXPLICIT_INCEPTION_BASELINE":

                if start >= earliest_date:
                    raise ValueError(
                        "Inception baseline must precede "
                        "the first observed return date."
                    )

            else:
                raise ValueError(
                    f"Unsupported anchor source: {anchor_source}"
                )

            actual_days = (end - start).days

            if actual_days != int(
                window["calendar_days"]
            ):
                raise ValueError(
                    "Window calendar-day count mismatch."
                )

            # Exclude the return earned up to the start
            # anchor. Include returns after that anchor.

            included = group.loc[
                group["date"].gt(start)
                & group["date"].le(end)
            ]

            actual_count = len(included)

            if actual_count == 0:
                raise ValueError(
                    "Available window has no return observations."
                )

            if actual_count != int(
                window["observation_count"]
            ):
                raise ValueError(
                    "Window observation-count mismatch."
                )

            if (
                included["date"].iloc[0]
                != window["first_return_date"]
            ):
                raise ValueError(
                    "Window first-return date mismatch."
                )

            if included["date"].iloc[-1] != end:
                raise ValueError(
                    "Window last-return date mismatch."
                )

            # A portfolio already extinguished by a
            # complete capital loss cannot earn subsequent
            # self-financing investment returns from zero.

            prior_returns = group.loc[
                group["date"].le(start),
                "return_total",
            ]

            if prior_returns.eq(-1.0).any():

                record["calculation_status"] = (
                    "UNAVAILABLE_ZERO_START_WEALTH"
                )

                record["cagr_status"] = (
                    "UNAVAILABLE_ZERO_START_WEALTH"
                )

                results.append(record)
                continue

            daily_returns = included[
                "return_total"
            ].to_numpy(dtype=np.float64)

            gross_returns = 1.0 + daily_returns

            gross_wealth = float(
                np.prod(
                    gross_returns,
                    dtype=np.float64,
                )
            )

            if not np.isfinite(gross_wealth):
                raise ValueError(
                    "Non-finite compounded window wealth."
                )

            period_return = gross_wealth - 1.0

            arithmetic_mean_annualized = (
                float(np.mean(daily_returns))
                * arithmetic_trading_days
            )

            record["wealth_start_normalized"] = 1.0
            record["wealth_end_normalized"] = gross_wealth

            record["period_return"] = period_return

            record["annualized_arithmetic_mean"] = (
                arithmetic_mean_annualized
            )

            record["calculation_status"] = "CALCULATED"

            # CAGR is not reported for a window shorter
            # than the configured calendar-day threshold.

            if actual_days < minimum_cagr_calendar_days:

                record["cagr_status"] = (
                    "NOT_ANNUALIZED_SHORT_WINDOW"
                )

            else:

                annualized_cagr = (
                    gross_wealth ** (day_basis / actual_days)
                    - 1.0
                )

                if not np.isfinite(annualized_cagr):
                    raise ValueError(
                        "Non-finite annualized CAGR."
                    )

                record["annualized_return_cagr"] = (
                    annualized_cagr
                )

                record["cagr_status"] = "CALCULATED"

            results.append(record)

    result_df = (
        pd.DataFrame.from_records(results)
        .sort_values(output_keys)
        .reset_index(drop=True)
    )

    if result_df.duplicated(output_keys).any():
        raise ValueError(
            "Duplicate absolute-performance output keys."
        )

    return result_df


# Research reference: 07_performance_analytics.ipynb, code cell index 8.
def compute_relative_window_performance(
    aligned_returns_df: pd.DataFrame,
    window_definitions_df: pd.DataFrame,
    *,
    annualization_day_basis: float = 365.25,
    arithmetic_trading_days: int = 252,
    minimum_cagr_calendar_days: int = 365,
) -> pd.DataFrame:
    """
    Independently calculate matched-window relative performance.

    Required daily columns
    ----------------------
    portfolio_id
    benchmark_portfolio_id
    date
    portfolio_return_total
    benchmark_return_total

    Required window columns
    -----------------------
    portfolio_id
    window_id
    as_of_date
    resolved_start_date
    first_return_date
    last_return_date
    observation_count
    calendar_days
    start_anchor_source
    window_status

    The portfolio and benchmark must have identical
    reporting-window boundaries and observations.

    Unavailable or mismatched windows retain explicit
    statuses rather than silently using shorter windows.

    Calculation does not require another analytics
    function or any previously calculated return series.
    """

    # --------------------------------------------------------
    # Validate methodology configuration.
    # --------------------------------------------------------

    if isinstance(
        annualization_day_basis, (bool, np.bool_)
    ):
        raise TypeError(
            "annualization_day_basis cannot be Boolean."
        )

    try:
        day_basis = float(annualization_day_basis)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(
            "annualization_day_basis must be numeric."
        ) from exc

    if not np.isfinite(day_basis) or day_basis <= 0:
        raise ValueError(
            "annualization_day_basis must be positive and finite."
        )

    for name, value in (
        ("arithmetic_trading_days", arithmetic_trading_days),
        ("minimum_cagr_calendar_days", minimum_cagr_calendar_days),
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value <= 0
        ):
            raise ValueError(
                f"{name} must be a positive integer."
            )

    # --------------------------------------------------------
    # Validate input contracts.
    # --------------------------------------------------------

    if not isinstance(
        aligned_returns_df, pd.DataFrame
    ):
        raise TypeError(
            "aligned_returns_df must be a DataFrame."
        )

    if not isinstance(
        window_definitions_df, pd.DataFrame
    ):
        raise TypeError(
            "window_definitions_df must be a DataFrame."
        )

    if (
        aligned_returns_df.empty
        or window_definitions_df.empty
    ):
        raise ValueError(
            "Aligned returns and windows must be nonempty."
        )

    if (
        not aligned_returns_df.columns.is_unique
        or not window_definitions_df.columns.is_unique
    ):
        raise ValueError(
            "Duplicate DataFrame column names."
        )

    pair_keys = [
        "portfolio_id",
        "benchmark_portfolio_id",
    ]

    return_columns = [
        "portfolio_return_total",
        "benchmark_return_total",
    ]

    aligned_required = (
        pair_keys + ["date"] + return_columns
    )

    window_required = [
        "portfolio_id",
        "window_id",
        "as_of_date",
        "resolved_start_date",
        "first_return_date",
        "last_return_date",
        "observation_count",
        "calendar_days",
        "start_anchor_source",
        "window_status",
    ]

    for frame, required, name in (
        (
            aligned_returns_df,
            aligned_required,
            "aligned_returns_df",
        ),
        (
            window_definitions_df,
            window_required,
            "window_definitions_df",
        ),
    ):
        missing = set(required).difference(
            frame.columns
        )

        if missing:
            raise ValueError(
                f"{name}: missing {sorted(missing)}"
            )

    aligned = aligned_returns_df[
        aligned_required
    ].copy()

    windows = window_definitions_df.copy(
        deep=True
    )

    if aligned.isna().any().any():
        raise ValueError(
            "Null fields detected in aligned returns."
        )

    if windows[
        ["portfolio_id", "window_id", "as_of_date"]
    ].isna().any().any():
        raise ValueError(
            "Null reporting-window identifiers."
        )

    for frame, fields in (
        (aligned, pair_keys),
        (windows, ["portfolio_id", "window_id"]),
    ):
        for field in fields:
            if not frame[field].map(
                lambda value: isinstance(value, str)
                and bool(value.strip())
            ).all():
                raise ValueError(
                    f"Invalid {field} identifiers."
                )

    if aligned["portfolio_id"].eq(
        aligned["benchmark_portfolio_id"]
    ).any():
        raise ValueError(
            "Portfolio and benchmark IDs must differ."
        )

    aligned["date"] = pd.to_datetime(
        aligned["date"],
        errors="raise",
    )

    for column in [
        "as_of_date",
        "resolved_start_date",
        "first_return_date",
        "last_return_date",
    ]:
        windows[column] = pd.to_datetime(
            windows[column],
            errors="raise",
        )

    if windows["as_of_date"].isna().any():
        raise ValueError(
            "Null or invalid as-of dates."
        )

    for frame, fields in (
        (aligned, ["date"]),
        (
            windows,
            [
                "as_of_date",
                "resolved_start_date",
                "first_return_date",
                "last_return_date",
            ],
        ),
    ):
        for field in fields:
            if frame[field].dt.tz is not None:
                raise ValueError(
                    f"{field} must be timezone-naive."
                )

            nonnull_dates = frame[field].dropna()

            if not nonnull_dates.eq(
                nonnull_dates.dt.normalize()
            ).all():
                raise ValueError(
                    f"{field} must contain date-only values."
                )

    if aligned.duplicated(
        pair_keys + ["date"]
    ).any():
        raise ValueError(
            "Duplicate aligned portfolio/benchmark/date keys."
        )

    if windows.duplicated(
        ["portfolio_id", "as_of_date", "window_id"]
    ).any():
        raise ValueError(
            "Duplicate window-definition keys."
        )

    for column in return_columns:

        if not pd.api.types.is_numeric_dtype(
            aligned[column]
        ):
            raise TypeError(
                f"{column} must be numeric."
            )

        values = aligned[column].to_numpy(
            dtype=np.float64
        )

        if (
            not np.isfinite(values).all()
            or (values < -1.0).any()
        ):
            raise ValueError(
                f"{column} must be finite and >= -100%."
            )

    aligned = (
        aligned
        .sort_values(pair_keys + ["date"])
        .reset_index(drop=True)
    )

    windows_indexed = windows.set_index(
        ["portfolio_id", "as_of_date", "window_id"],
        verify_integrity=True,
    )

    # --------------------------------------------------------
    # Define standardized metric fields.
    # --------------------------------------------------------

    metric_columns = [
        "portfolio_period_return",
        "benchmark_period_return",
        "arithmetic_active_period_return",
        "geometric_relative_period_return",
        "sum_daily_arithmetic_active_return",
        "annualized_arithmetic_active_mean",
        "portfolio_annualized_return_cagr",
        "benchmark_annualized_return_cagr",
        "annualized_cagr_spread",
        "annualized_geometric_relative_return",
    ]

    records = []

    # --------------------------------------------------------
    # Process portfolio/benchmark pairs independently.
    # --------------------------------------------------------

    for (portfolio_id, benchmark_id), group in (
        aligned.groupby(pair_keys, sort=True)
    ):

        group = group.sort_values("date")

        available_dates = pd.DatetimeIndex(
            group["date"]
        )

        portfolio_windows = windows.loc[
            windows["portfolio_id"].eq(
                portfolio_id
            )
        ]

        benchmark_windows = windows.loc[
            windows["portfolio_id"].eq(
                benchmark_id
            )
        ]

        portfolio_window_keys = set(
            zip(
                portfolio_windows["as_of_date"],
                portfolio_windows["window_id"],
            )
        )

        benchmark_window_keys = set(
            zip(
                benchmark_windows["as_of_date"],
                benchmark_windows["window_id"],
            )
        )

        if (
            not portfolio_window_keys
            or portfolio_window_keys
            != benchmark_window_keys
        ):
            raise ValueError(
                "Incomplete matched reporting-window "
                f"definitions for {portfolio_id} "
                f"versus {benchmark_id}."
            )

        for as_of, window_id in sorted(
            portfolio_window_keys
        ):

            portfolio_window = windows_indexed.loc[
                (portfolio_id, as_of, window_id)
            ]

            benchmark_window = windows_indexed.loc[
                (benchmark_id, as_of, window_id)
            ]

            # Preserve one output record even when a
            # particular window is unavailable.

            record = {
                "portfolio_id": portfolio_id,
                "benchmark_portfolio_id": benchmark_id,
                "as_of_date": as_of,
                "window_id": window_id,
                "resolved_start_date": (
                    portfolio_window["resolved_start_date"]
                ),
                "first_return_date": (
                    portfolio_window["first_return_date"]
                ),
                "last_return_date": (
                    portfolio_window["last_return_date"]
                ),
                "observation_count": (
                    portfolio_window["observation_count"]
                ),
                "calendar_days": (
                    portfolio_window["calendar_days"]
                ),
                "start_anchor_source": (
                    portfolio_window["start_anchor_source"]
                ),
                "portfolio_window_status": (
                    portfolio_window["window_status"]
                ),
                "benchmark_window_status": (
                    benchmark_window["window_status"]
                ),
                "annualization_day_basis": day_basis,
                "arithmetic_trading_days": (
                    arithmetic_trading_days
                ),
                "minimum_cagr_calendar_days": (
                    minimum_cagr_calendar_days
                ),
                "return_units": "simple_decimal",
                "calculation_status": "NOT_CALCULATED",
                "cagr_status": "NOT_CALCULATED",
            }

            record.update(
                {
                    column: np.nan
                    for column in metric_columns
                }
            )

            # ------------------------------------------------
            # Preserve unavailable-window status.
            # ------------------------------------------------

            if (
                portfolio_window["window_status"]
                != "AVAILABLE"
                or benchmark_window["window_status"]
                != "AVAILABLE"
            ):

                record["calculation_status"] = (
                    "WINDOW_UNAVAILABLE"
                )

                record["cagr_status"] = (
                    "WINDOW_UNAVAILABLE"
                )

                records.append(record)
                continue

            # ------------------------------------------------
            # Enforce exact matched boundaries.
            # ------------------------------------------------

            boundary_fields = [
                "resolved_start_date",
                "first_return_date",
                "last_return_date",
                "observation_count",
                "calendar_days",
                "start_anchor_source",
            ]

            if any(
                portfolio_window[field]
                != benchmark_window[field]
                for field in boundary_fields
            ):

                record["calculation_status"] = (
                    "UNAVAILABLE_MISMATCHED_BOUNDARIES"
                )

                record["cagr_status"] = (
                    "UNAVAILABLE_MISMATCHED_BOUNDARIES"
                )

                records.append(record)
                continue

            start = portfolio_window[
                "resolved_start_date"
            ]

            end = portfolio_window[
                "last_return_date"
            ]

            if (
                pd.isna(start)
                or pd.isna(end)
                or not start < end
            ):
                raise ValueError(
                    f"Invalid {window_id} boundaries."
                )

            if (
                end != as_of
                or end not in available_dates
            ):
                raise ValueError(
                    f"Invalid {window_id} end date."
                )

            days = (end - start).days

            if days != int(
                portfolio_window["calendar_days"]
            ):
                raise ValueError(
                    f"Invalid {window_id} day count."
                )

            anchor_source = portfolio_window[
                "start_anchor_source"
            ]

            if anchor_source == "OBSERVED_RETURN_DATE":

                if start not in available_dates:
                    raise ValueError(
                        f"Invalid {window_id} start anchor."
                    )

            elif (
                anchor_source
                == "EXPLICIT_INCEPTION_BASELINE"
            ):

                if start >= available_dates.min():
                    raise ValueError(
                        f"Invalid {window_id} inception baseline."
                    )

            else:
                raise ValueError(
                    f"Unsupported {window_id} anchor source."
                )

            # ------------------------------------------------
            # Include only returns after the start anchor.
            # ------------------------------------------------

            sample = group.loc[
                group["date"].gt(start)
                & group["date"].le(end)
            ]

            if (
                sample.empty
                or len(sample)
                != int(portfolio_window["observation_count"])
                or sample["date"].iloc[0]
                != portfolio_window["first_return_date"]
                or sample["date"].iloc[-1] != end
            ):
                raise ValueError(
                    f"Invalid {window_id} return-date coverage."
                )

            # A prior total loss extinguishes the starting
            # capital of a self-financing investment.

            prior = group.loc[
                group["date"].le(start)
            ]

            if prior[
                return_columns
            ].eq(-1.0).any().any():

                record["calculation_status"] = (
                    "UNAVAILABLE_ZERO_START_WEALTH"
                )

                record["cagr_status"] = (
                    "UNAVAILABLE_ZERO_START_WEALTH"
                )

                records.append(record)
                continue

            # ------------------------------------------------
            # Independently compound matched daily returns.
            # ------------------------------------------------

            portfolio_returns = sample[
                "portfolio_return_total"
            ].to_numpy(dtype=np.float64)

            benchmark_returns = sample[
                "benchmark_return_total"
            ].to_numpy(dtype=np.float64)

            if (benchmark_returns == -1.0).any():
                raise ValueError(
                    "Complete benchmark loss makes "
                    "relative wealth undefined."
                )

            portfolio_gross = float(
                np.prod(
                    1.0 + portfolio_returns,
                    dtype=np.float64,
                )
            )

            benchmark_gross = float(
                np.prod(
                    1.0 + benchmark_returns,
                    dtype=np.float64,
                )
            )

            if (
                not np.isfinite(portfolio_gross)
                or not np.isfinite(benchmark_gross)
                or benchmark_gross <= 0.0
            ):
                raise ValueError(
                    "Non-finite or undefined relative wealth."
                )

            portfolio_period_return = (
                portfolio_gross - 1.0
            )

            benchmark_period_return = (
                benchmark_gross - 1.0
            )

            relative_gross = (
                portfolio_gross / benchmark_gross
            )

            record.update(
                {
                    "portfolio_period_return": (
                        portfolio_period_return
                    ),
                    "benchmark_period_return": (
                        benchmark_period_return
                    ),
                    "arithmetic_active_period_return": (
                        portfolio_period_return
                        - benchmark_period_return
                    ),
                    "geometric_relative_period_return": (
                        relative_gross - 1.0
                    ),
                    "sum_daily_arithmetic_active_return": (
                        float(
                            np.sum(
                                portfolio_returns
                                - benchmark_returns
                            )
                        )
                    ),
                    "annualized_arithmetic_active_mean": (
                        float(
                            np.mean(
                                portfolio_returns
                                - benchmark_returns
                            )
                            * arithmetic_trading_days
                        )
                    ),
                    "calculation_status": "CALCULATED",
                }
            )

            # ------------------------------------------------
            # Apply explicit CAGR eligibility.
            # ------------------------------------------------

            if days < minimum_cagr_calendar_days:

                record["cagr_status"] = (
                    "NOT_ANNUALIZED_SHORT_WINDOW"
                )

            else:

                exponent = day_basis / days

                portfolio_cagr = (
                    portfolio_gross ** exponent - 1.0
                )

                benchmark_cagr = (
                    benchmark_gross ** exponent - 1.0
                )

                geometric_annualized_relative = (
                    relative_gross ** exponent - 1.0
                )

                if not np.isfinite(
                    [
                        portfolio_cagr,
                        benchmark_cagr,
                        geometric_annualized_relative,
                    ]
                ).all():
                    raise ValueError(
                        "Non-finite annualized relative performance."
                    )

                record.update(
                    {
                        "portfolio_annualized_return_cagr": (
                            portfolio_cagr
                        ),
                        "benchmark_annualized_return_cagr": (
                            benchmark_cagr
                        ),
                        "annualized_cagr_spread": (
                            portfolio_cagr - benchmark_cagr
                        ),
                        "annualized_geometric_relative_return": (
                            geometric_annualized_relative
                        ),
                        "cagr_status": "CALCULATED",
                    }
                )

            records.append(record)

    if not records:
        raise ValueError(
            "No matched performance windows available."
        )

    result_df = (
        pd.DataFrame.from_records(records)
        .sort_values(
            pair_keys + ["as_of_date", "window_id"]
        )
        .reset_index(drop=True)
    )

    result_keys = (
        pair_keys + ["as_of_date", "window_id"]
    )

    if result_df.duplicated(result_keys).any():
        raise ValueError(
            "Duplicate relative-window output keys."
        )

    observed_metrics = result_df[
        metric_columns
    ].to_numpy(dtype=np.float64)

    if not np.isfinite(
        observed_metrics[~np.isnan(observed_metrics)]
    ).all():
        raise ValueError(
            "Non-finite nonmissing relative metrics."
        )

    return result_df
