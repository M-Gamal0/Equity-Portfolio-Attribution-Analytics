"""Risk security: validated research calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd



# Research reference: 09_risk_analytics.ipynb, code cell index 10.
def calculate_security_risk_statistics(
    security_returns: pd.DataFrame,
    reference_returns: pd.DataFrame,
    security_ids,
    trading_dates: pd.DatetimeIndex,
    as_of_date: pd.Timestamp,
    lookback_sessions: int,
    target_return_column: str,
    annualization_sessions: int = 252,
    variance_ddof: int = 1,
    downside_target_daily: float = 0.0,
) -> pd.DataFrame:
    """
    Calculate historical risk statistics for securities.

    Parameters
    ----------
    security_returns:
        Security-level daily total returns.

    reference_returns:
        Aligned daily portfolio/benchmark returns with:
            date
            portfolio_return_total
            benchmark_return_total

    security_ids:
        Explicit security universe for the estimation.

    trading_dates:
        Approved ascending portfolio trading calendar.

    as_of_date:
        Observed estimation cutoff. Future returns excluded.

    lookback_sessions:
        Number of trailing common return observations.

    target_return_column:
        Reference series for security-to-portfolio
        statistics. Either portfolio_return_total or
        benchmark_return_total.

    annualization_sessions:
        Number of trading sessions per annualization year.

    variance_ddof:
        Sample variance/covariance degrees of freedom.

    downside_target_daily:
        Daily minimum acceptable security return.

    Returns
    -------
    pd.DataFrame
        One observation per security, containing:

        - Historical daily mean and sample variance.
        - Annualized variance and volatility.
        - Annualized downside deviation.
        - Negative-return observation count.
        - Covariance, beta, and correlation to portfolio.
        - Covariance, beta, and correlation to benchmark.
        - Average correlation with other universe securities.
        - Explicit availability statuses.

    Notes
    -----
    All statistics use identical return dates.

    No missing values are filled or silently dropped.

    Zero reference variance makes beta unavailable.
    Zero variance in either series makes correlation
    unavailable.

    Average peer correlation excludes self-correlation.

    The function does not calculate portfolio risk
    contributions or modify its inputs.
    """

    required_security_columns = {
        "security_id", "date", "return_total"
    }

    required_reference_columns = {
        "date",
        "portfolio_return_total",
        "benchmark_return_total",
    }

    if not required_security_columns.issubset(
        security_returns.columns
    ):
        raise ValueError(
            "Missing security-return input columns."
        )

    if not required_reference_columns.issubset(
        reference_returns.columns
    ):
        raise ValueError(
            "Missing portfolio reference-return columns."
        )

    if target_return_column not in (
        "portfolio_return_total",
        "benchmark_return_total",
    ):
        raise ValueError(
            "Unsupported target portfolio return column."
        )

    for name, value in (
        ("lookback_sessions", lookback_sessions),
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

    if lookback_sessions <= variance_ddof:
        raise ValueError(
            "Lookback must exceed variance_ddof."
        )

    if (
        not np.isscalar(downside_target_daily)
        or not isinstance(
            downside_target_daily,
            (int, float, np.integer, np.floating),
        )
        or not np.isfinite(downside_target_daily)
    ):
        raise ValueError(
            "Downside target must be a finite number."
        )

    securities = pd.Index(security_ids)

    if (
        securities.empty
        or securities.hasnans
        or securities.has_duplicates
    ):
        raise ValueError(
            "Security universe must be nonempty, "
            "non-null, and unique."
        )

    # --------------------------------------------------------
    # Validate trading calendar and cutoff
    # --------------------------------------------------------

    calendar = pd.DatetimeIndex(
        pd.to_datetime(
            trading_dates,
            errors="raise",
        )
    )

    if (
        calendar.hasnans
        or not calendar.is_unique
        or not calendar.is_monotonic_increasing
    ):
        raise ValueError(
            "Trading calendar must be non-null, "
            "unique, and increasing."
        )

    cutoff = pd.Timestamp(as_of_date)

    if pd.isna(cutoff) or cutoff not in calendar:
        raise ValueError(
            "As-of date must exist in the trading calendar."
        )

    eligible_dates = calendar[calendar <= cutoff]

    if len(eligible_dates) < lookback_sessions:
        raise ValueError(
            "Insufficient historical trading observations."
        )

    window_dates = eligible_dates[-lookback_sessions:]

    # --------------------------------------------------------
    # Assemble complete security-return panel
    # --------------------------------------------------------

    selected_returns = security_returns.loc[
        security_returns["security_id"].isin(securities)
        & security_returns["date"].isin(window_dates),
        ["security_id", "date", "return_total"],
    ].copy()

    selected_returns["date"] = pd.to_datetime(
        selected_returns["date"],
        errors="raise",
    )

    if selected_returns.duplicated(
        ["security_id", "date"]
    ).any():
        raise ValueError(
            "Duplicate security/date observations."
        )

    if (
        not pd.api.types.is_numeric_dtype(
            selected_returns["return_total"]
        )
        or pd.api.types.is_bool_dtype(
            selected_returns["return_total"]
        )
        or pd.api.types.is_complex_dtype(
            selected_returns["return_total"]
        )
    ):
        raise TypeError(
            "Security returns must be real numeric values."
        )

    security_panel = (
        selected_returns
        .pivot(
            index="date",
            columns="security_id",
            values="return_total",
        )
        .reindex(
            index=window_dates,
            columns=securities,
        )
    )

    if security_panel.isna().any().any():
        raise ValueError(
            "Incomplete security-return estimation panel."
        )

    security_matrix = security_panel.to_numpy(
        dtype=np.float64
    )

    # --------------------------------------------------------
    # Assemble date-matched portfolio reference returns
    # --------------------------------------------------------

    reference = reference_returns[
        [
            "date",
            "portfolio_return_total",
            "benchmark_return_total",
        ]
    ].copy()

    reference["date"] = pd.to_datetime(
        reference["date"],
        errors="raise",
    )

    if (
        reference["date"].isna().any()
        or reference["date"].duplicated().any()
    ):
        raise ValueError(
            "Reference return dates must be unique and non-null."
        )

    for column in (
        "portfolio_return_total",
        "benchmark_return_total",
    ):
        if (
            not pd.api.types.is_numeric_dtype(reference[column])
            or pd.api.types.is_bool_dtype(reference[column])
            or pd.api.types.is_complex_dtype(reference[column])
        ):
            raise TypeError(
                f"{column} must contain real numeric returns."
            )

    reference_window = (
        reference
        .set_index("date")
        .reindex(window_dates)
    )

    if reference_window.isna().any().any():
        raise ValueError(
            "Incomplete date-matched reference returns."
        )

    target_returns = reference_window[
        target_return_column
    ].to_numpy(dtype=np.float64)

    benchmark_returns = reference_window[
        "benchmark_return_total"
    ].to_numpy(dtype=np.float64)

    # --------------------------------------------------------
    # Validate all selected observations
    # --------------------------------------------------------

    for name, values in (
        ("Security returns", security_matrix),
        ("Portfolio returns", target_returns),
        ("Benchmark returns", benchmark_returns),
    ):
        if not np.isfinite(values).all():
            raise ValueError(
                f"{name} contain non-finite observations."
            )

        if np.any(values < -1.0):
            raise ValueError(
                f"{name} contain returns below -100%."
            )

    observations, security_count = (
        security_matrix.shape
    )

    denominator = observations - variance_ddof

    # --------------------------------------------------------
    # Calculate centered return observations
    # --------------------------------------------------------

    security_differences = (
        security_matrix - security_matrix[:1, :]
    )

    security_centered = (
        security_differences
        - security_differences.mean(
            axis=0,
            keepdims=True,
        )
    )

    target_differences = (
        target_returns - target_returns[0]
    )

    target_centered = (
        target_differences
        - target_differences.mean()
    )

    benchmark_differences = (
        benchmark_returns - benchmark_returns[0]
    )

    benchmark_centered = (
        benchmark_differences
        - benchmark_differences.mean()
    )

    # --------------------------------------------------------
    # Calculate individual security variance and volatility
    # --------------------------------------------------------

    security_variance = (
        np.einsum(
            "ij,ij->j",
            security_centered,
            security_centered,
        )
        / denominator
    )

    security_std = np.sqrt(security_variance)

    annualized_variance = (
        security_variance * annualization_sessions
    )

    annualized_volatility = np.sqrt(
        annualized_variance
    )

    mean_daily_return = security_matrix.mean(
        axis=0
    )

    # --------------------------------------------------------
    # Calculate downside risk
    # --------------------------------------------------------

    downside_differences = np.minimum(
        security_matrix - downside_target_daily,
        0.0,
    )

    downside_lpm2_daily = np.mean(
        downside_differences**2,
        axis=0,
    )

    downside_deviation_annualized = np.sqrt(
        downside_lpm2_daily * annualization_sessions
    )

    negative_observations = np.count_nonzero(
        security_matrix < 0.0,
        axis=0,
    )

    # --------------------------------------------------------
    # Calculate reference variances and covariances
    # --------------------------------------------------------

    target_variance = float(
        np.dot(
            target_centered,
            target_centered,
        ) / denominator
    )

    benchmark_variance = float(
        np.dot(
            benchmark_centered,
            benchmark_centered,
        ) / denominator
    )

    covariance_to_portfolio = (
        security_centered.T @ target_centered
    ) / denominator

    covariance_to_benchmark = (
        security_centered.T @ benchmark_centered
    ) / denominator

    # --------------------------------------------------------
    # Calculate beta and correlation to portfolio
    # --------------------------------------------------------

    beta_to_portfolio = np.full(
        security_count,
        np.nan,
    )

    correlation_to_portfolio = np.full(
        security_count,
        np.nan,
    )

    if target_variance > 0.0:

        beta_to_portfolio = (
            covariance_to_portfolio / target_variance
        )

        valid = security_variance > 0.0

        np.divide(
            covariance_to_portfolio,
            security_std * np.sqrt(target_variance),
            out=correlation_to_portfolio,
            where=valid,
        )

    # --------------------------------------------------------
    # Calculate beta and correlation to benchmark
    # --------------------------------------------------------

    beta_to_benchmark = np.full(
        security_count,
        np.nan,
    )

    correlation_to_benchmark = np.full(
        security_count,
        np.nan,
    )

    if benchmark_variance > 0.0:

        beta_to_benchmark = (
            covariance_to_benchmark / benchmark_variance
        )

        valid = security_variance > 0.0

        np.divide(
            covariance_to_benchmark,
            security_std * np.sqrt(benchmark_variance),
            out=correlation_to_benchmark,
            where=valid,
        )

    # --------------------------------------------------------
    # Calculate average correlation with other securities
    # --------------------------------------------------------

    covariance_matrix = (
        security_centered.T @ security_centered
    ) / denominator

    correlation_denominator = np.outer(
        security_std,
        security_std,
    )

    peer_correlation_matrix = np.full(
        (security_count, security_count),
        np.nan,
    )

    np.divide(
        covariance_matrix,
        correlation_denominator,
        out=peer_correlation_matrix,
        where=correlation_denominator > 0.0,
    )

    # A security is not its own diversification peer.

    np.fill_diagonal(
        peer_correlation_matrix,
        np.nan,
    )

    valid_peer_counts = np.isfinite(
        peer_correlation_matrix
    ).sum(axis=1)

    mean_peer_correlation = np.full(
        security_count,
        np.nan,
    )

    np.divide(
        np.nansum(
            peer_correlation_matrix,
            axis=1,
        ),
        valid_peer_counts,
        out=mean_peer_correlation,
        where=valid_peer_counts > 0,
    )

    # --------------------------------------------------------
    # Validate numerical correlation boundaries
    # --------------------------------------------------------

    for label, correlations in (
        ("Portfolio", correlation_to_portfolio),
        ("Benchmark", correlation_to_benchmark),
        ("Peer", mean_peer_correlation),
    ):
        finite = correlations[np.isfinite(correlations)]

        if np.any(np.abs(finite) > 1.0 + 1e-10):
            raise ArithmeticError(
                f"{label} correlation exceeds valid bounds."
            )

    # --------------------------------------------------------
    # Assemble security-level statistics and statuses
    # --------------------------------------------------------

    portfolio_beta_available = (
        target_variance > 0.0
    )

    benchmark_beta_available = (
        benchmark_variance > 0.0
    )

    return pd.DataFrame(
        {
            "security_id": securities,
            "as_of_date": cutoff,
            "sample_start": window_dates.min(),
            "sample_end": window_dates.max(),
            "lookback_sessions": lookback_sessions,
            "observations": observations,
            "annualization_sessions": annualization_sessions,
            "variance_ddof": variance_ddof,
            "downside_target_daily": (
                downside_target_daily
            ),
            "mean_return_daily": mean_daily_return,
            "variance_daily": security_variance,
            "variance_annualized": annualized_variance,
            "volatility_annualized": annualized_volatility,
            "downside_lpm2_daily": downside_lpm2_daily,
            "downside_deviation_annualized": (
                downside_deviation_annualized
            ),
            "negative_return_observations": (
                negative_observations
            ),
            "covariance_to_portfolio_daily": (
                covariance_to_portfolio
            ),
            "beta_to_portfolio": beta_to_portfolio,
            "correlation_to_portfolio": (
                correlation_to_portfolio
            ),
            "covariance_to_benchmark_daily": (
                covariance_to_benchmark
            ),
            "beta_to_benchmark": beta_to_benchmark,
            "correlation_to_benchmark": (
                correlation_to_benchmark
            ),
            "mean_peer_correlation": (
                mean_peer_correlation
            ),
            "valid_peer_count": valid_peer_counts,
            "portfolio_beta_status": (
                "PASS"
                if portfolio_beta_available
                else "UNAVAILABLE_ZERO_PORTFOLIO_VARIANCE"
            ),
            "benchmark_beta_status": (
                "PASS"
                if benchmark_beta_available
                else "UNAVAILABLE_ZERO_BENCHMARK_VARIANCE"
            ),
            "portfolio_correlation_status": np.where(
                np.isfinite(correlation_to_portfolio),
                "PASS",
                "UNAVAILABLE_ZERO_VARIANCE",
            ),
            "benchmark_correlation_status": np.where(
                np.isfinite(correlation_to_benchmark),
                "PASS",
                "UNAVAILABLE_ZERO_VARIANCE",
            ),
            "peer_correlation_status": np.where(
                valid_peer_counts > 0,
                "PASS",
                "UNAVAILABLE_NO_VALID_PEERS",
            ),
            "status": "PASS",
        }
    )
