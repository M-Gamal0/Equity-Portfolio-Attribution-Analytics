"""Performance capture: validated research calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd



# Research reference: 07_performance_analytics.ipynb, code cell index 11.
def compute_market_capture(
    aligned_returns_df: pd.DataFrame,
    windows_df: pd.DataFrame,
    *,
    portfolio_id: str,
    benchmark_portfolio_id: str,
) -> pd.DataFrame:
    """
    Calculate benchmark-conditioned market capture
    across matched reporting windows.

    Requires daily paired returns and independently
    resolved portfolio/benchmark window definitions.

    Arithmetic capture:
        100 * mean(portfolio returns in regime)
              / mean(benchmark returns in regime)

    Geometric capture:
        100 * geometric_mean(portfolio gross returns) - 100
        --------------------------------------------------
           geometric_mean(benchmark gross returns) - 1

    Equivalently, the geometric capture ratio is 100
    times the portfolio's conditional geometric mean
    simple return divided by the benchmark equivalent.

    The geometric mean is evaluated per selected day;
    no assumption of contiguous regime observations
    or annualized conditional holding periods is made.
    """

    required_daily = {
        "portfolio_id",
        "benchmark_portfolio_id",
        "date",
        "portfolio_return_total",
        "benchmark_return_total",
    }

    required_window = {
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

    for frame, required in (
        (aligned_returns_df, required_daily),
        (windows_df, required_window),
    ):
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("Expected pandas DataFrame inputs.")

        missing = required.difference(frame.columns)

        if missing:
            raise ValueError(
                f"Missing input columns: {sorted(missing)}"
            )

    if portfolio_id == benchmark_portfolio_id:
        raise ValueError("Portfolio and benchmark must differ.")

    daily = aligned_returns_df.loc[
        aligned_returns_df["portfolio_id"].eq(portfolio_id)
        & aligned_returns_df[
            "benchmark_portfolio_id"
        ].eq(benchmark_portfolio_id),
        [
            "date",
            "portfolio_return_total",
            "benchmark_return_total",
        ],
    ].copy()

    if daily.empty:
        raise ValueError("No aligned return observations.")

    daily["date"] = pd.to_datetime(daily["date"])

    daily = daily.sort_values("date")

    if daily["date"].isna().any() or (
        not daily["date"].is_unique
    ):
        raise ValueError("Invalid aligned return dates.")

    return_columns = [
        "portfolio_return_total",
        "benchmark_return_total",
    ]

    if any(
        not pd.api.types.is_numeric_dtype(daily[column])
        for column in return_columns
    ):
        raise TypeError("Daily returns must be numeric.")

    daily_values = daily[
        return_columns
    ].to_numpy(dtype=np.float64)

    if (
        not np.isfinite(daily_values).all()
        or (daily_values < -1.0).any()
    ):
        raise ValueError(
            "Daily returns must be finite and >= -100%."
        )

    # Benchmark complete loss makes geometric
    # relative performance undefined.

    if daily["benchmark_return_total"].eq(-1.0).any():
        raise ValueError(
            "Complete benchmark loss makes capture undefined."
        )

    selected_windows = windows_df.loc[
        windows_df["portfolio_id"].isin(
            [portfolio_id, benchmark_portfolio_id]
        ),
        list(required_window),
    ].copy()

    for column in (
        "as_of_date",
        "resolved_start_date",
        "first_return_date",
        "last_return_date",
    ):
        selected_windows[column] = pd.to_datetime(
            selected_windows[column]
        )

    window_index = selected_windows.set_index(
        ["portfolio_id", "as_of_date", "window_id"]
    )

    if not window_index.index.is_unique:
        raise ValueError("Duplicate reporting-window keys.")

    portfolio_keys = {
        (as_of, window_id)
        for pid, as_of, window_id in window_index.index
        if pid == portfolio_id
    }

    benchmark_keys = {
        (as_of, window_id)
        for pid, as_of, window_id in window_index.index
        if pid == benchmark_portfolio_id
    }

    if not portfolio_keys or portfolio_keys != benchmark_keys:
        raise ValueError(
            "Portfolio and benchmark window definitions differ."
        )

    def conditional_statistics(
        portfolio_returns: np.ndarray,
        benchmark_returns: np.ndarray,
    ) -> dict:
        """Calculate daily-equivalent conditional statistics."""

        if len(portfolio_returns) == 0:
            return {
                "portfolio_arithmetic_mean": np.nan,
                "benchmark_arithmetic_mean": np.nan,
                "portfolio_geometric_mean": np.nan,
                "benchmark_geometric_mean": np.nan,
                "arithmetic_capture_pct": np.nan,
                "geometric_capture_pct": np.nan,
                "capture_status": "NO_REGIME_OBSERVATIONS",
            }

        arithmetic_p = float(np.mean(portfolio_returns))
        arithmetic_b = float(np.mean(benchmark_returns))

        geometric_p = (
            -1.0
            if np.any(portfolio_returns == -1.0)
            else float(
                np.expm1(
                    np.mean(np.log1p(portfolio_returns))
                )
            )
        )

        geometric_b = float(
            np.expm1(
                np.mean(np.log1p(benchmark_returns))
            )
        )

        arithmetic_capture = (
            100.0 * arithmetic_p / arithmetic_b
            if arithmetic_b != 0.0
            else np.nan
        )

        geometric_capture = (
            100.0 * geometric_p / geometric_b
            if geometric_b != 0.0
            else np.nan
        )

        return {
            "portfolio_arithmetic_mean": arithmetic_p,
            "benchmark_arithmetic_mean": arithmetic_b,
            "portfolio_geometric_mean": geometric_p,
            "benchmark_geometric_mean": geometric_b,
            "arithmetic_capture_pct": arithmetic_capture,
            "geometric_capture_pct": geometric_capture,
            "capture_status": (
                "CALCULATED"
                if np.isfinite(
                    [arithmetic_capture, geometric_capture]
                ).all()
                else "UNDEFINED_CAPTURE_DENOMINATOR"
            ),
        }

    records = []

    for as_of, window_id in sorted(portfolio_keys):

        portfolio_window = window_index.loc[
            (portfolio_id, as_of, window_id)
        ]

        benchmark_window = window_index.loc[
            (benchmark_portfolio_id, as_of, window_id)
        ]

        record = {
            "portfolio_id": portfolio_id,
            "benchmark_portfolio_id": benchmark_portfolio_id,
            "as_of_date": as_of,
            "window_id": window_id,
            "resolved_start_date": (
                portfolio_window["resolved_start_date"]
            ),
            "observation_count": (
                portfolio_window["observation_count"]
            ),
            "calculation_status": "NOT_CALCULATED",
        }

        for regime in ("up", "down"):
            record[f"{regime}_observations"] = 0

            for metric in (
                "portfolio_arithmetic_mean",
                "benchmark_arithmetic_mean",
                "portfolio_geometric_mean",
                "benchmark_geometric_mean",
                "arithmetic_capture_pct",
                "geometric_capture_pct",
            ):
                record[f"{regime}_{metric}"] = np.nan

            record[f"{regime}_capture_status"] = (
                "NOT_CALCULATED"
            )

        record["flat_observations"] = 0

        if (
            portfolio_window["window_status"] != "AVAILABLE"
            or benchmark_window["window_status"] != "AVAILABLE"
        ):
            record["calculation_status"] = "WINDOW_UNAVAILABLE"
            records.append(record)
            continue

        boundary_fields = [
            "resolved_start_date",
            "first_return_date",
            "last_return_date",
            "observation_count",
            "calendar_days",
            "start_anchor_source",
        ]

        if any(
            portfolio_window[field] != benchmark_window[field]
            for field in boundary_fields
        ):
            record["calculation_status"] = (
                "UNAVAILABLE_MISMATCHED_BOUNDARIES"
            )
            records.append(record)
            continue

        start = portfolio_window["resolved_start_date"]
        end = portfolio_window["last_return_date"]

        if pd.isna(start) or pd.isna(end) or not start < end:
            raise ValueError(f"Invalid {window_id} boundaries.")

        if end != as_of:
            raise ValueError(f"Invalid {window_id} cutoff.")

        anchor_source = portfolio_window[
            "start_anchor_source"
        ]

        if anchor_source == "OBSERVED_RETURN_DATE":
            if not daily["date"].eq(start).any():
                raise ValueError(
                    f"Missing {window_id} start anchor."
                )

        elif anchor_source == "EXPLICIT_INCEPTION_BASELINE":
            if start >= daily["date"].min():
                raise ValueError(
                    f"Invalid {window_id} inception baseline."
                )

        else:
            raise ValueError(
                f"Unsupported {window_id} anchor source."
            )

        sample = daily.loc[
            daily["date"].gt(start)
            & daily["date"].le(end)
        ]

        if (
            sample.empty
            or len(sample) != int(
                portfolio_window["observation_count"]
            )
            or sample["date"].iloc[0]
            != portfolio_window["first_return_date"]
            or sample["date"].iloc[-1] != end
            or (end - start).days != int(
                portfolio_window["calendar_days"]
            )
        ):
            raise ValueError(
                f"Inconsistent {window_id} return-date coverage."
            )

        # A return window cannot restart an extinguished
        # self-financing portfolio without new capital.

        prior = daily.loc[daily["date"].le(start)]

        if prior[return_columns].eq(-1.0).any().any():
            record["calculation_status"] = (
                "UNAVAILABLE_ZERO_START_WEALTH"
            )
            records.append(record)
            continue

        rp = sample["portfolio_return_total"].to_numpy(
            dtype=np.float64
        )

        rb = sample["benchmark_return_total"].to_numpy(
            dtype=np.float64
        )

        regimes = {
            "up": rb > 0.0,
            "down": rb < 0.0,
        }

        record["flat_observations"] = int(
            np.count_nonzero(rb == 0.0)
        )

        regime_statuses = []

        for regime, mask in regimes.items():

            record[f"{regime}_observations"] = int(
                np.count_nonzero(mask)
            )

            stats = conditional_statistics(
                rp[mask], rb[mask]
            )

            for metric, value in stats.items():
                record[f"{regime}_{metric}"] = value

            regime_statuses.append(
                stats["capture_status"]
            )

        if (
            record["up_observations"]
            + record["down_observations"]
            + record["flat_observations"]
            != len(sample)
        ):
            raise AssertionError(
                "Benchmark regimes do not partition the sample."
            )

        record["calculation_status"] = (
            "CALCULATED"
            if all(
                status == "CALCULATED"
                for status in regime_statuses
            )
            else "PARTIAL_REGIME_COVERAGE"
        )

        records.append(record)

    return (
        pd.DataFrame.from_records(records)
        .sort_values(["as_of_date", "window_id"])
        .reset_index(drop=True)
    )
