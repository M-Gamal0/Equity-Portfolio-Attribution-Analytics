"""Risk active: validated research calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd



# Research reference: 09_risk_analytics.ipynb, code cell index 4.
def calculate_active_risk(
    portfolio_returns: pd.Series | np.ndarray,
    benchmark_returns: pd.Series | np.ndarray,
    annualization_sessions: int,
    variance_ddof: int = 1,
    downside_target_daily: float = 0.0,
) -> dict:
    """
    Calculate historical benchmark-relative risk statistics.

    Parameters
    ----------
    portfolio_returns:
        Daily simple decimal portfolio total returns.

    benchmark_returns:
        Daily simple decimal benchmark total returns,
        already aligned to portfolio return intervals.

    annualization_sessions:
        Trading sessions per annualization year.

    variance_ddof:
        Degrees-of-freedom adjustment shared by all
        variance and covariance estimators.

    downside_target_daily:
        Daily arithmetic active-return threshold.

    Returns
    -------
    dict
        Daily covariance, variance, active-return
        statistics, annualized tracking error, beta,
        correlation, downside deviation, and statuses.

    Notes
    -----
    Both arrays must describe identical return intervals
    in identical order. Date alignment is the caller's
    responsibility.

    No observations are filled, truncated, or dropped.

    Benchmark beta is undefined when benchmark variance
    equals zero.

    Correlation is undefined when either return series
    has zero variance.

    Undefined statistics are returned as NaN with
    explicit availability statuses.

    Raises
    ------
    ValueError
        Invalid numerical values, incompatible array
        dimensions, or insufficient observations.

    TypeError
        Non-numeric returns.
    """

    if (
        not isinstance(
            annualization_sessions,
            (int, np.integer),
        )
        or isinstance(
            annualization_sessions,
            (bool, np.bool_),
        )
        or annualization_sessions <= 0
    ):
        raise ValueError(
            "annualization_sessions must be "
            "a positive integer."
        )

    if (
        not isinstance(
            variance_ddof,
            (int, np.integer),
        )
        or isinstance(variance_ddof, (bool, np.bool_))
        or variance_ddof < 1
    ):
        raise ValueError(
            "variance_ddof must be a positive integer."
        )

    if (
        not isinstance(
            downside_target_daily,
            (int, float, np.integer, np.floating),
        )
        or not np.isfinite(downside_target_daily)
    ):
        raise ValueError(
            "downside_target_daily must be finite."
        )

    portfolio = np.asarray(portfolio_returns)
    benchmark = np.asarray(benchmark_returns)

    if portfolio.ndim != 1 or benchmark.ndim != 1:
        raise ValueError(
            "Both return arrays must be one-dimensional."
        )

    if portfolio.shape != benchmark.shape:
        raise ValueError(
            "Portfolio and benchmark observation counts "
            "must match exactly."
        )

    for label, values in (
        ("Portfolio", portfolio),
        ("Benchmark", benchmark),
    ):
        if (
            not pd.api.types.is_numeric_dtype(values.dtype)
            or pd.api.types.is_bool_dtype(values.dtype)
            or pd.api.types.is_complex_dtype(values.dtype)
        ):
            raise TypeError(
                f"{label} returns must be real numeric values."
            )

    portfolio = portfolio.astype(
        np.float64,
        copy=False,
    )

    benchmark = benchmark.astype(
        np.float64,
        copy=False,
    )

    observations = len(portfolio)

    if observations <= variance_ddof:
        raise ValueError(
            "Insufficient matched observations for "
            "the requested variance estimator."
        )

    if (
        not np.isfinite(portfolio).all()
        or not np.isfinite(benchmark).all()
    ):
        raise ValueError(
            "Returns contain missing or non-finite values."
        )

    if (
        np.any(portfolio < -1.0)
        or np.any(benchmark < -1.0)
    ):
        raise ValueError(
            "Daily simple returns cannot fall below -100%."
        )

    # --------------------------------------------------------
    # Calculate arithmetic active returns
    # --------------------------------------------------------

    active_returns = portfolio - benchmark

    mean_active_return = float(
        np.mean(active_returns)
    )

    # --------------------------------------------------------
    # Sample variances and covariance
    # --------------------------------------------------------

    portfolio_variance = float(
        np.var(
            portfolio,
            ddof=variance_ddof,
        )
    )

    benchmark_variance = float(
        np.var(
            benchmark,
            ddof=variance_ddof,
        )
    )

    active_variance = float(
        np.var(
            active_returns,
            ddof=variance_ddof,
        )
    )

    portfolio_centered = (
        portfolio - np.mean(portfolio)
    )

    benchmark_centered = (
        benchmark - np.mean(benchmark)
    )

    covariance_daily = float(
        np.dot(
            portfolio_centered,
            benchmark_centered,
        )
        / (observations - variance_ddof)
    )

    # --------------------------------------------------------
    # Calculate annualized tracking error
    # --------------------------------------------------------

    tracking_error_annualized = float(
        np.sqrt(
            active_variance * annualization_sessions
        )
    )

    # --------------------------------------------------------
    # Calculate benchmark beta
    # --------------------------------------------------------

    if benchmark_variance > 0.0:

        benchmark_beta = float(
            covariance_daily / benchmark_variance
        )

        beta_status = "PASS"

    else:

        benchmark_beta = np.nan

        beta_status = (
            "UNAVAILABLE_ZERO_BENCHMARK_VARIANCE"
        )

    # --------------------------------------------------------
    # Calculate portfolio-benchmark correlation
    # --------------------------------------------------------

    if (
        portfolio_variance > 0.0
        and benchmark_variance > 0.0
    ):

        benchmark_correlation = float(
            covariance_daily
            / np.sqrt(
                portfolio_variance * benchmark_variance
            )
        )

        correlation_status = "PASS"

    else:

        benchmark_correlation = np.nan

        correlation_status = (
            "UNAVAILABLE_ZERO_RETURN_VARIANCE"
        )

    # --------------------------------------------------------
    # Calculate downside risk of arithmetic active returns
    # --------------------------------------------------------

    downside_differences = np.minimum(
        active_returns - downside_target_daily,
        0.0,
    )

    downside_observations = int(
        np.count_nonzero(downside_differences)
    )

    active_downside_lpm2_daily = float(
        np.mean(downside_differences**2)
    )

    active_downside_deviation_annualized = float(
        np.sqrt(
            active_downside_lpm2_daily
            * annualization_sessions
        )
    )

    # --------------------------------------------------------
    # Calculate independent active-variance identity
    # --------------------------------------------------------

    active_variance_from_components = float(
        portfolio_variance
        + benchmark_variance
        - 2.0 * covariance_daily
    )

    return {
        "observations": observations,
        "mean_active_return_daily": mean_active_return,
        "portfolio_variance_daily": portfolio_variance,
        "benchmark_variance_daily": benchmark_variance,
        "active_variance_daily": active_variance,
        "covariance_daily": covariance_daily,
        "active_variance_from_components": (
            active_variance_from_components
        ),
        "tracking_error_annualized": (
            tracking_error_annualized
        ),
        "benchmark_beta": benchmark_beta,
        "benchmark_correlation": benchmark_correlation,
        "active_downside_observations": (
            downside_observations
        ),
        "active_downside_lpm2_daily": (
            active_downside_lpm2_daily
        ),
        "active_downside_deviation_annualized": (
            active_downside_deviation_annualized
        ),
        "tracking_error_status": "PASS",
        "beta_status": beta_status,
        "correlation_status": correlation_status,
    }
