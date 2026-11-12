"""Risk covariance: validated research calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd

RISK_COVARIANCE_CONDITION_THRESHOLD = 10000000000.0

# Research reference: 09_risk_analytics.ipynb, code cell index 9.
def estimate_security_covariance(
    security_returns: pd.DataFrame,
    security_ids,
    trading_dates: pd.DatetimeIndex,
    as_of_date: pd.Timestamp,
    lookback_sessions: int,
    annualization_sessions: int = 252,
    variance_ddof: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Estimate historical security covariance and correlation.

    Parameters
    ----------
    security_returns:
        Standardized security total-return observations.
        Required: security_id, date, return_total.

    security_ids:
        Explicit estimation security universe.

    trading_dates:
        Approved, ascending portfolio trading calendar.

    as_of_date:
        Required observed estimation cutoff date.

    lookback_sessions:
        Number of trailing daily return observations.

    annualization_sessions:
        Sessions per annualization year.

    variance_ddof:
        Degrees-of-freedom adjustment for sample covariance.

    Returns
    -------
    covariance_daily_df:
        Square security covariance matrix in daily units.

    correlation_df:
        Square Pearson correlation matrix.

    diagnostics:
        Estimation coverage, annualization settings,
        spectral diagnostics, and availability statuses.

    Notes
    -----
    The covariance estimator uses a common complete
    observation panel. Pairwise deletion is prohibited.

    Covariance matrices are not inverted or modified
    to enforce positive definiteness.

    Historical covariance is not a validated prediction
    of future covariance.

    Inputs remain unchanged.
    """

    # --------------------------------------------------------
    # Validate required input structure
    # --------------------------------------------------------

    required_columns = {
        "security_id",
        "date",
        "return_total",
    }

    missing = required_columns.difference(
        security_returns.columns
    )

    if missing:
        raise ValueError(
            "Missing security-return columns: "
            f"{sorted(missing)}"
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
    # Validate as-of trading calendar
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

    eligible_dates = calendar[
        calendar <= cutoff
    ]

    if len(eligible_dates) < lookback_sessions:
        raise ValueError(
            "Insufficient as-of portfolio trading history."
        )

    window_dates = eligible_dates[
        -lookback_sessions:
    ]

    # --------------------------------------------------------
    # Construct an as-of eligible security-return panel
    # --------------------------------------------------------

    selected = security_returns.loc[
        security_returns["security_id"].isin(securities)
        & security_returns["date"].isin(window_dates),
        ["security_id", "date", "return_total"],
    ].copy()

    selected["date"] = pd.to_datetime(
        selected["date"],
        errors="raise",
    )

    if selected.duplicated(
        subset=["security_id", "date"]
    ).any():
        raise ValueError(
            "Duplicate security/date estimation observations."
        )

    if not pd.api.types.is_numeric_dtype(
        selected["return_total"]
    ):
        raise TypeError(
            "Security returns must be numeric."
        )

    if (
        pd.api.types.is_bool_dtype(
            selected["return_total"]
        )
        or pd.api.types.is_complex_dtype(
            selected["return_total"]
        )
    ):
        raise TypeError(
            "Security returns must be real numeric values."
        )

    panel = (
        selected
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

    if panel.isna().any().any():
        missing_count = int(
            panel.isna().sum().sum()
        )

        raise ValueError(
            "Incomplete security-return estimation panel: "
            f"{missing_count:,} missing observations."
        )

    matrix = panel.to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(matrix).all():
        raise ValueError(
            "Non-finite returns in covariance sample."
        )

    if np.any(matrix < -1.0):
        raise ValueError(
            "Security simple returns cannot be below -100%."
        )

    observations, security_count = matrix.shape

    # --------------------------------------------------------
    # Estimate sample covariance using common observations
    # --------------------------------------------------------

    # Subtract a per-security reference observation before
    # demeaning for improved numerical stability.

    differences = matrix - matrix[:1, :]

    centered = (
        differences
        - differences.mean(
            axis=0,
            keepdims=True,
        )
    )

    denominator = observations - variance_ddof

    covariance_daily = (
        centered.T @ centered
    ) / denominator

    covariance_annualized = (
        covariance_daily * annualization_sessions
    )

    # --------------------------------------------------------
    # Validate matrix symmetry and covariance diagonal
    # --------------------------------------------------------

    np.testing.assert_allclose(
        covariance_daily,
        covariance_daily.T,
        rtol=1e-12,
        atol=1e-14,
        err_msg="Covariance symmetry failure.",
    )

    diagonal_variances = np.diag(
        covariance_daily
    )

    if np.any(diagonal_variances < 0.0):
        raise ArithmeticError(
            "Covariance diagonal contains negative variance."
        )

    np.testing.assert_allclose(
        diagonal_variances,
        np.var(
            centered,
            axis=0,
            ddof=variance_ddof,
        ),
        rtol=1e-12,
        atol=1e-14,
    )

    # --------------------------------------------------------
    # Calculate Pearson correlation
    # --------------------------------------------------------

    standard_deviations = np.sqrt(
        diagonal_variances
    )

    correlation_denominator = np.outer(
        standard_deviations,
        standard_deviations,
    )

    correlation = np.full(
        covariance_daily.shape,
        np.nan,
        dtype=np.float64,
    )

    np.divide(
        covariance_daily,
        correlation_denominator,
        out=correlation,
        where=correlation_denominator > 0.0,
    )

    valid_correlation = np.isfinite(correlation)

    if np.any(
        np.abs(correlation[valid_correlation])
        > 1.0 + 1e-10
    ):
        raise ArithmeticError(
            "Correlation matrix exceeds valid boundaries."
        )

    # --------------------------------------------------------
    # Assess covariance positive semidefiniteness
    # --------------------------------------------------------

    eigenvalues = np.linalg.eigvalsh(
        covariance_annualized
    )

    minimum_eigenvalue = float(
        eigenvalues[0]
    )

    maximum_eigenvalue = float(
        eigenvalues[-1]
    )

    psd_tolerance = max(
        1e-14,
        1e-10 * max(maximum_eigenvalue, 0.0),
    )

    if minimum_eigenvalue < -psd_tolerance:
        raise ArithmeticError(
            "Sample covariance matrix is not "
            "positive semidefinite within tolerance."
        )

    # --------------------------------------------------------
    # Assess numerical rank and conditioning
    # --------------------------------------------------------

    rank_tolerance = (
        max(observations, security_count)
        * np.finfo(np.float64).eps
        * max(maximum_eigenvalue, 0.0)
    )

    numerical_rank = int(
        np.count_nonzero(
            eigenvalues > rank_tolerance
        )
    )

    full_rank = numerical_rank == security_count

    if full_rank:

        condition_number = float(
            maximum_eigenvalue / minimum_eigenvalue
        )

        conditioning_status = (
            "ILL_CONDITIONED"
            if condition_number
            > RISK_COVARIANCE_CONDITION_THRESHOLD
            else "FULL_RANK"
        )

    else:

        condition_number = np.inf

        conditioning_status = "SINGULAR"

    # A sample covariance matrix from demeaned returns
    # has rank at most observations - 1.

    maximum_theoretical_rank = min(
        security_count,
        observations - 1,
    )

    if numerical_rank > maximum_theoretical_rank:
        raise ArithmeticError(
            "Numerical rank exceeds sample covariance "
            "theoretical rank bound."
        )

    # --------------------------------------------------------
    # Package explicitly labeled outputs
    # --------------------------------------------------------

    covariance_daily_df = pd.DataFrame(
        covariance_daily,
        index=securities,
        columns=securities,
    )

    correlation_df = pd.DataFrame(
        correlation,
        index=securities,
        columns=securities,
    )

    diagnostics = {
        "as_of_date": cutoff,
        "sample_start": window_dates.min(),
        "sample_end": window_dates.max(),
        "lookback_sessions": lookback_sessions,
        "observations": observations,
        "security_count": security_count,
        "annualization_sessions": (
            annualization_sessions
        ),
        "variance_ddof": variance_ddof,
        "numerical_rank": numerical_rank,
        "maximum_theoretical_rank": (
            maximum_theoretical_rank
        ),
        "full_rank": bool(full_rank),
        "minimum_eigenvalue_annualized": (
            minimum_eigenvalue
        ),
        "maximum_eigenvalue_annualized": (
            maximum_eigenvalue
        ),
        "condition_number": condition_number,
        "conditioning_status": conditioning_status,
        "zero_variance_security_count": int(
            np.count_nonzero(
                diagonal_variances == 0.0
            )
        ),
        "unavailable_correlation_pairs": int(
            np.count_nonzero(
                ~valid_correlation
            )
        ),
        "psd_status": "PASS",
        "coverage_status": "FULL_PANEL_COVERAGE",
    }

    return (
        covariance_daily_df,
        correlation_df,
        diagnostics,
    )
