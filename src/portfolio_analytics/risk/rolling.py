"""Risk rolling: validated research calculations."""

from __future__ import annotations

from numpy.lib.stride_tricks import sliding_window_view
import numpy as np
import pandas as pd



# Research reference: 09_risk_analytics.ipynb, code cell index 5.
def calculate_rolling_risk(
    aligned_returns: pd.DataFrame,
    portfolio_id: str,
    benchmark_portfolio_id: str,
    lookback_sessions: tuple[int, ...],
    annualization_sessions: int,
    variance_ddof: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate rolling absolute and benchmark-relative risk.

    Required aligned_returns columns:
        date
        portfolio_return_total
        benchmark_return_total

    Returns:
        risk_rolling_df:
            Portfolio/benchmark x date x lookback.

        active_risk_rolling_df:
            Portfolio x benchmark x date x lookback.

    All returns use daily simple decimal units.

    Complete trailing observation windows are required.
    Initial unavailable estimates remain NaN with explicit
    statuses. Zero-variance beta/correlation is unavailable.

    Calculations use centered observations to improve
    numerical stability and preserve full precision.

    Inputs are not modified.
    """

    required = {
        "date",
        "portfolio_return_total",
        "benchmark_return_total",
    }

    missing = required.difference(aligned_returns.columns)

    if missing:
        raise ValueError(
            f"Missing rolling-risk columns: {sorted(missing)}"
        )

    if portfolio_id == benchmark_portfolio_id:
        raise ValueError(
            "Portfolio and benchmark IDs must differ."
        )

    for name, value in (
        ("annualization_sessions", annualization_sessions),
        ("variance_ddof", variance_ddof),
    ):
        if (
            not isinstance(value, (int, np.integer))
            or isinstance(value, (bool, np.bool_))
            or value < 1
        ):
            raise ValueError(
                f"{name} must be a positive integer."
            )

    if (
        not lookback_sessions
        or any(
            not isinstance(n, (int, np.integer))
            or isinstance(n, (bool, np.bool_))
            or n <= variance_ddof
            for n in lookback_sessions
        )
        or len(set(lookback_sessions)) != len(lookback_sessions)
    ):
        raise ValueError(
            "Lookbacks must be unique integers greater "
            "than variance_ddof."
        )

    dates = pd.DatetimeIndex(
        pd.to_datetime(
            aligned_returns["date"],
            errors="raise",
        )
    )

    if (
        dates.hasnans
        or not dates.is_unique
        or not dates.is_monotonic_increasing
    ):
        raise ValueError(
            "Return dates must be non-null, unique, "
            "and chronologically increasing."
        )

    return_columns = [
        "portfolio_return_total",
        "benchmark_return_total",
    ]

    for column in return_columns:
        values = aligned_returns[column]

        if (
            not pd.api.types.is_numeric_dtype(values)
            or pd.api.types.is_bool_dtype(values)
            or pd.api.types.is_complex_dtype(values)
        ):
            raise TypeError(
                f"{column} must contain real numeric returns."
            )

    values = aligned_returns[
        return_columns
    ].to_numpy(dtype=np.float64)

    if not np.isfinite(values).all():
        raise ValueError(
            "Returns contain non-finite observations."
        )

    if (values < -1.0).any():
        raise ValueError(
            "Simple returns cannot be below -100%."
        )

    portfolio = values[:, 0]
    benchmark = values[:, 1]
    active = portfolio - benchmark

    observation_count = len(dates)

    absolute_records = []
    active_records = []

    # --------------------------------------------------------
    # Estimate each requested rolling window independently
    # --------------------------------------------------------

    for window in sorted(lookback_sessions):

        observations = np.minimum(
            np.arange(1, observation_count + 1),
            window,
        )

        eligible = observations.eq(window) if isinstance(
            observations, pd.Series
        ) else observations == window

        eligible_count = max(
            observation_count - window + 1,
            0,
        )

        portfolio_vol = np.full(
            observation_count, np.nan
        )
        benchmark_vol = np.full(
            observation_count, np.nan
        )
        tracking_error = np.full(
            observation_count, np.nan
        )
        beta = np.full(
            observation_count, np.nan
        )
        correlation = np.full(
            observation_count, np.nan
        )

        portfolio_variance = np.full(
            observation_count, np.nan
        )
        benchmark_variance = np.full(
            observation_count, np.nan
        )
        active_variance = np.full(
            observation_count, np.nan
        )
        covariance = np.full(
            observation_count, np.nan
        )

        beta_status = np.full(
            observation_count,
            "UNAVAILABLE_INSUFFICIENT_HISTORY",
            dtype=object,
        )

        correlation_status = beta_status.copy()

        if eligible_count:

            # Each row contains one complete trailing window.

            p_windows = sliding_window_view(
                portfolio, window
            )
            b_windows = sliding_window_view(
                benchmark, window
            )
            a_windows = sliding_window_view(
                active, window
            )

            # Center using the first observation as an
            # offset to avoid artificial nonzero variance
            # for exactly constant return sequences.

            def center_windows(matrix):
                differences = matrix - matrix[:, :1]
                return (
                    differences
                    - differences.mean(axis=1, keepdims=True)
                )

            pc = center_windows(p_windows)
            bc = center_windows(b_windows)
            ac = center_windows(a_windows)

            denominator = window - variance_ddof

            pv = np.einsum(
                "ij,ij->i", pc, pc
            ) / denominator

            bv = np.einsum(
                "ij,ij->i", bc, bc
            ) / denominator

            av = np.einsum(
                "ij,ij->i", ac, ac
            ) / denominator

            cov = np.einsum(
                "ij,ij->i", pc, bc
            ) / denominator

            valid_positions = slice(window - 1, None)

            portfolio_variance[valid_positions] = pv
            benchmark_variance[valid_positions] = bv
            active_variance[valid_positions] = av
            covariance[valid_positions] = cov

            portfolio_vol[valid_positions] = np.sqrt(
                pv * annualization_sessions
            )

            benchmark_vol[valid_positions] = np.sqrt(
                bv * annualization_sessions
            )

            tracking_error[valid_positions] = np.sqrt(
                av * annualization_sessions
            )

            # Beta requires positive benchmark variance.

            beta_values = np.full(
                eligible_count, np.nan
            )

            beta_available = bv > 0.0

            np.divide(
                cov,
                bv,
                out=beta_values,
                where=beta_available,
            )

            beta[valid_positions] = beta_values

            beta_status[valid_positions] = np.where(
                beta_available,
                "PASS",
                "UNAVAILABLE_ZERO_BENCHMARK_VARIANCE",
            )

            # Correlation requires both variances > 0.

            correlation_values = np.full(
                eligible_count, np.nan
            )

            correlation_available = (
                (pv > 0.0) & (bv > 0.0)
            )

            correlation_denominator = np.sqrt(
                pv * bv
            )

            np.divide(
                cov,
                correlation_denominator,
                out=correlation_values,
                where=correlation_available,
            )

            correlation[valid_positions] = (
                correlation_values
            )

            correlation_status[valid_positions] = np.where(
                correlation_available,
                "PASS",
                "UNAVAILABLE_ZERO_RETURN_VARIANCE",
            )

            # Independent rolling active-variance identity.

            np.testing.assert_allclose(
                av,
                pv + bv - 2.0 * cov,
                rtol=1e-10,
                atol=1e-14,
                err_msg=(
                    f"Rolling active-variance identity: {window}"
                ),
            )

            if np.any(
                np.abs(
                    correlation_values[correlation_available]
                ) > 1.0 + 1e-10
            ):
                raise ArithmeticError(
                    "Rolling correlation exceeds valid bounds."
                )

        window_status = np.where(
            eligible,
            "PASS",
            "UNAVAILABLE_INSUFFICIENT_HISTORY",
        )

        # ----------------------------------------------------
        # Absolute rolling risk: active and benchmark
        # ----------------------------------------------------

        for current_id, volatility, daily_variance in (
            (
                portfolio_id,
                portfolio_vol,
                portfolio_variance,
            ),
            (
                benchmark_portfolio_id,
                benchmark_vol,
                benchmark_variance,
            ),
        ):

            absolute_records.append(
                pd.DataFrame(
                    {
                        "portfolio_id": current_id,
                        "date": dates,
                        "lookback_sessions": window,
                        "observations": observations,
                        "daily_variance": daily_variance,
                        "annualized_volatility": volatility,
                        "status": window_status,
                    }
                )
            )

        # ----------------------------------------------------
        # Benchmark-relative rolling risk
        # ----------------------------------------------------

        active_records.append(
            pd.DataFrame(
                {
                    "portfolio_id": portfolio_id,
                    "benchmark_portfolio_id": (
                        benchmark_portfolio_id
                    ),
                    "date": dates,
                    "lookback_sessions": window,
                    "observations": observations,
                    "portfolio_variance_daily": (
                        portfolio_variance
                    ),
                    "benchmark_variance_daily": (
                        benchmark_variance
                    ),
                    "active_variance_daily": (
                        active_variance
                    ),
                    "covariance_daily": covariance,
                    "tracking_error_annualized": (
                        tracking_error
                    ),
                    "benchmark_beta": beta,
                    "benchmark_correlation": correlation,
                    "tracking_error_status": window_status,
                    "beta_status": beta_status,
                    "correlation_status": correlation_status,
                }
            )
        )

    absolute_result = pd.concat(
        absolute_records,
        ignore_index=True,
    )

    active_result = pd.concat(
        active_records,
        ignore_index=True,
    )

    return absolute_result, active_result
