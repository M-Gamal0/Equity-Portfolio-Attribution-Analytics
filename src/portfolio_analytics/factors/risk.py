"""Factors risk: validated research calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd



# Research reference: 11_fama_french_factor_analytics.ipynb, code cell index 17.
def decompose_direct_regression_factor_risk(
    factor_matrix: np.ndarray,
    response: np.ndarray,
    coefficients: np.ndarray,
    residuals: np.ndarray,
    *,
    factor_names: tuple[str, ...],
    annualization_days: int = 252,
    tolerance: float = 1e-11,
) -> tuple[dict, pd.DataFrame, np.ndarray]:
    """
    Decompose the sample variance of an OLS regression
    response into systematic factor and residual risk.

    Inputs
    ------
    factor_matrix:
        n x K realized factor returns.

    response:
        n contemporaneous regression response observations.

    coefficients:
        K+1 OLS coefficients, intercept first.

    residuals:
        n fitted OLS residual observations.

    Returns
    -------
    summary:
        Regression response variance, systematic variance,
        residual variance, annualized volatilities, R²,
        and exact reconciliation errors.

    component_df:
        Individual factor contributions to sample variance
        and annualized response volatility.

    factor_covariance:
        K x K sample covariance matrix.

    Interpretation
    --------------
    For the active-return regression, annualized response
    volatility is historical tracking error.

    For portfolio and benchmark regressions, it is
    the annualized volatility of the EXCESS-return response,
    not necessarily total-return volatility.

    This is an in-sample, conditional regression
    decomposition, not an ex-ante equity risk model.
    """

    F = np.asarray(
        factor_matrix,
        dtype=np.float64,
    )

    y = np.asarray(
        response,
        dtype=np.float64,
    ).reshape(-1)

    beta_full = np.asarray(
        coefficients,
        dtype=np.float64,
    ).reshape(-1)

    e = np.asarray(
        residuals,
        dtype=np.float64,
    ).reshape(-1)

    if F.ndim != 2:
        raise ValueError(
            "Factor matrix must be two-dimensional."
        )

    n, k = F.shape

    if (
        n <= k + 1
        or y.shape != (n,)
        or e.shape != (n,)
        or beta_full.shape != (k + 1,)
        or len(factor_names) != k
    ):
        raise ValueError(
            "Invalid factor risk input dimensions."
        )

    if annualization_days <= 0:
        raise ValueError(
            "Annualization days must be positive."
        )

    if (
        not np.isfinite(F).all()
        or not np.isfinite(y).all()
        or not np.isfinite(e).all()
        or not np.isfinite(beta_full).all()
    ):
        raise ValueError(
            "Factor risk inputs contain non-finite values."
        )

    # --------------------------------------------------------
    # A. Verify the response and supplied residual identity
    # --------------------------------------------------------

    X = np.column_stack(
        [
            np.ones(n),
            F,
        ]
    )

    expected_residuals = (
        y - X @ beta_full
    )

    np.testing.assert_allclose(
        e,
        expected_residuals,
        rtol=1e-9,
        atol=tolerance,
        err_msg=(
            "Supplied residuals are inconsistent "
            "with the fitted regression response."
        ),
    )

    # --------------------------------------------------------
    # B. Confirm OLS orthogonality
    # --------------------------------------------------------

    regression_scores = (
        X.T @ e
    )

    if np.max(
        np.abs(regression_scores)
    ) > 1e-8:
        raise ValueError(
            "Regression residuals are not orthogonal "
            "to the fitted factor design."
        )

    # --------------------------------------------------------
    # C. Estimate sample factor covariance
    # --------------------------------------------------------

    factor_covariance = np.cov(
        F,
        rowvar=False,
        ddof=1,
    )

    factor_covariance = np.asarray(
        factor_covariance,
        dtype=np.float64,
    )

    if factor_covariance.shape != (k, k):
        raise ValueError(
            "Factor covariance matrix dimensions are invalid."
        )

    factor_covariance = 0.5 * (
        factor_covariance
        + factor_covariance.T
    )

    if not np.isfinite(
        factor_covariance
    ).all():
        raise ValueError(
            "Factor covariance matrix is non-finite."
        )

    covariance_eigenvalues = np.linalg.eigvalsh(
        factor_covariance
    )

    if covariance_eigenvalues.min() < -tolerance:
        raise ValueError(
            "Factor sample covariance is materially "
            "non-positive-semidefinite."
        )

    # --------------------------------------------------------
    # D. Calculate the systematic factor response
    # --------------------------------------------------------

    factor_betas = beta_full[1:]

    systematic_response = (
        F @ factor_betas
    )

    total_variance = float(
        np.var(
            y,
            ddof=1,
        )
    )

    systematic_variance = float(
        factor_betas
        @ factor_covariance
        @ factor_betas
    )

    residual_variance = float(
        np.var(
            e,
            ddof=1,
        )
    )

    if (
        total_variance <= 0
        or systematic_variance < -tolerance
        or residual_variance < -tolerance
    ):
        raise ValueError(
            "Invalid regression risk variances."
        )

    systematic_variance = max(
        systematic_variance,
        0.0,
    )

    residual_variance = max(
        residual_variance,
        0.0,
    )

    # --------------------------------------------------------
    # E. Independently validate the systematic variance
    # --------------------------------------------------------

    np.testing.assert_allclose(
        systematic_variance,
        np.var(
            systematic_response,
            ddof=1,
        ),
        rtol=1e-10,
        atol=tolerance,
    )

    cross_covariance = float(
        np.cov(
            systematic_response,
            e,
            ddof=1,
        )[0, 1]
    )

    if abs(cross_covariance) > tolerance:
        raise ValueError(
            "Systematic and residual responses "
            "are not sample-orthogonal."
        )

    # --------------------------------------------------------
    # F. Validate the complete variance identity
    # --------------------------------------------------------

    reconstructed_variance = (
        systematic_variance
        + residual_variance
    )

    variance_error = (
        total_variance
        - reconstructed_variance
    )

    np.testing.assert_allclose(
        reconstructed_variance,
        total_variance,
        rtol=1e-10,
        atol=tolerance,
    )

    systematic_variance_share = (
        systematic_variance / total_variance
    )

    residual_variance_share = (
        residual_variance / total_variance
    )

    np.testing.assert_allclose(
        (
            systematic_variance_share
            + residual_variance_share
        ),
        1.0,
        rtol=1e-10,
        atol=1e-10,
    )

    # --------------------------------------------------------
    # G. Calculate annualized response volatility
    # --------------------------------------------------------

    annual_total_volatility = float(
        np.sqrt(
            annualization_days
            * total_variance
        )
    )

    annual_systematic_volatility = float(
        np.sqrt(
            annualization_days
            * systematic_variance
        )
    )

    annual_residual_volatility = float(
        np.sqrt(
            annualization_days
            * residual_variance
        )
    )

    # --------------------------------------------------------
    # H. Calculate individual factor variance contributions
    # --------------------------------------------------------

    marginal_variance = (
        factor_covariance @ factor_betas
    )

    factor_variance_contributions = (
        factor_betas * marginal_variance
    )

    np.testing.assert_allclose(
        factor_variance_contributions.sum(),
        systematic_variance,
        rtol=1e-10,
        atol=tolerance,
    )

    # --------------------------------------------------------
    # I. Calculate Euler contributions to total volatility
    # --------------------------------------------------------

    factor_annual_volatility_contributions = (
        annualization_days
        * factor_variance_contributions
        / annual_total_volatility
    )

    residual_annual_volatility_contribution = (
        annualization_days
        * residual_variance
        / annual_total_volatility
    )

    np.testing.assert_allclose(
        (
            factor_annual_volatility_contributions.sum()
            + residual_annual_volatility_contribution
        ),
        annual_total_volatility,
        rtol=1e-10,
        atol=tolerance,
    )

    # --------------------------------------------------------
    # J. Construct factor contribution output
    # --------------------------------------------------------

    component_df = pd.DataFrame(
        {
            "factor": list(factor_names),
            "beta": factor_betas,
            "factor_variance_contribution_daily": (
                factor_variance_contributions
            ),
            "factor_variance_share_total": (
                factor_variance_contributions
                / total_variance
            ),
            "factor_variance_share_systematic": np.where(
                systematic_variance > 0,
                (
                    factor_variance_contributions
                    / systematic_variance
                ),
                np.nan,
            ),
            "factor_annual_volatility_contribution": (
                factor_annual_volatility_contributions
            ),
        }
    )

    # --------------------------------------------------------
    # K. Construct risk summary
    # --------------------------------------------------------

    summary = {
        "n_observations": n,
        "n_factors": k,
        "annualization_days": annualization_days,
        "response_variance_daily": total_variance,
        "systematic_variance_daily": systematic_variance,
        "residual_variance_daily": residual_variance,
        "response_volatility_annual": (
            annual_total_volatility
        ),
        "systematic_volatility_annual": (
            annual_systematic_volatility
        ),
        "residual_volatility_annual": (
            annual_residual_volatility
        ),
        "systematic_variance_share": (
            systematic_variance_share
        ),
        "residual_variance_share": (
            residual_variance_share
        ),
        "residual_annual_volatility_contribution": (
            residual_annual_volatility_contribution
        ),
        "factor_annual_volatility_contribution_total": float(
            factor_annual_volatility_contributions.sum()
        ),
        "systematic_residual_covariance_daily": (
            cross_covariance
        ),
        "variance_reconciliation_error": (
            variance_error
        ),
    }

    return (
        summary,
        component_df,
        factor_covariance,
    )
