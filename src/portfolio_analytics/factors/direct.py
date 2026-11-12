"""Factors direct: validated research calculations."""

from __future__ import annotations

from scipy.stats import chi2
from scipy.stats import norm
from scipy.stats import t as student_t
import numpy as np
import pandas as pd



# Research reference: 11_fama_french_factor_analytics.ipynb, code cell index 16.
def select_direct_hac_lag(
    n_observations: int,
) -> int:
    """
    Automatic Bartlett-kernel lag selection:

        floor(4 * (n / 100) ** (2 / 9))

    The lag is capped below the sample size.
    """

    if n_observations <= 1:
        raise ValueError(
            "HAC lag selection requires at least "
            "two observations."
        )

    selected_lag = int(
        np.floor(
            4.0
            * (
                n_observations / 100.0
            ) ** (2.0 / 9.0)
        )
    )

    return min(
        selected_lag,
        n_observations - 1,
    )


# Research reference: 11_fama_french_factor_analytics.ipynb, code cell index 16.
def calculate_direct_ols_hac_inference(
    X: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    *,
    terms: tuple[str, ...],
    hac_lags: int,
    confidence_level: float = 0.95,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Calculate classical and Newey-West coefficient inference.

    Uses a QR factorization for the regression bread matrix.

    Classical covariance:
        sigma² * inverse(X'X)

    HAC covariance:
        [n/(n-p)] * inverse(X'X)
        * long_run_score_covariance
        * inverse(X'X)

    Newey-West kernel:
        Bartlett weights, 1 - lag/(L+1).

    HAC p-values and intervals:
        Asymptotic normal approximation.

    Returns
    -------
    inference_df
    classical_covariance
    hac_covariance
    """

    design = np.asarray(
        X,
        dtype=np.float64,
    )

    response = np.asarray(
        y,
        dtype=np.float64,
    ).reshape(-1)

    coefficients = np.asarray(
        beta,
        dtype=np.float64,
    ).reshape(-1)

    if design.ndim != 2:
        raise ValueError(
            "Regression design matrix must be two-dimensional."
        )

    n, p = design.shape

    if response.shape != (n,):
        raise ValueError(
            "Regression response shape does not match design."
        )

    if coefficients.shape != (p,):
        raise ValueError(
            "Coefficient vector shape does not match design."
        )

    if len(terms) != p:
        raise ValueError(
            "Regression term count does not match design."
        )

    if n <= p:
        raise ValueError(
            "Positive residual degrees of freedom required."
        )

    if not isinstance(hac_lags, (int, np.integer)):
        raise TypeError(
            "HAC lag must be an integer."
        )

    if not 0 <= hac_lags < n:
        raise ValueError(
            "HAC lag must be between zero and n-1."
        )

    if not 0 < confidence_level < 1:
        raise ValueError(
            "Invalid confidence level."
        )

    if not (
        np.isfinite(design).all()
        and np.isfinite(response).all()
        and np.isfinite(coefficients).all()
    ):
        raise ValueError(
            "OLS inference inputs must be finite."
        )

    # --------------------------------------------------------
    # A. Validate the supplied OLS coefficients using QR
    # --------------------------------------------------------

    Q, R = np.linalg.qr(
        design,
        mode="reduced",
    )

    if np.linalg.matrix_rank(R) != p:
        raise ValueError(
            "Regression design is not full rank."
        )

    qr_beta = np.linalg.solve(
        R,
        Q.T @ response,
    )

    np.testing.assert_allclose(
        coefficients,
        qr_beta,
        rtol=1e-9,
        atol=1e-10,
        err_msg=(
            "Supplied regression coefficients "
            "do not reconcile to QR OLS."
        ),
    )

    residuals = (
        response - design @ coefficients
    )

    rss = float(
        residuals @ residuals
    )

    df_residual = n - p

    # --------------------------------------------------------
    # B. Calculate numerically stable OLS bread
    # --------------------------------------------------------

    inverse_R = np.linalg.solve(
        R,
        np.eye(p),
    )

    inverse_xtx = (
        inverse_R @ inverse_R.T
    )

    inverse_xtx = 0.5 * (
        inverse_xtx + inverse_xtx.T
    )

    # --------------------------------------------------------
    # C. Classical OLS covariance
    # --------------------------------------------------------

    residual_variance = (
        rss / df_residual
    )

    classical_covariance = (
        residual_variance * inverse_xtx
    )

    # --------------------------------------------------------
    # D. Newey-West long-run score covariance
    # --------------------------------------------------------

    scores = (
        design * residuals[:, None]
    )

    meat = (
        scores.T @ scores
    )

    for lag in range(
        1,
        hac_lags + 1,
    ):

        bartlett_weight = (
            1.0
            - lag / (hac_lags + 1.0)
        )

        lagged_cross_product = (
            scores[lag:].T
            @ scores[:-lag]
        )

        meat += bartlett_weight * (
            lagged_cross_product
            + lagged_cross_product.T
        )

    # --------------------------------------------------------
    # E. Apply the finite-sample HAC adjustment
    # --------------------------------------------------------

    small_sample_adjustment = (
        n / df_residual
    )

    hac_covariance = (
        small_sample_adjustment
        * inverse_xtx
        @ meat
        @ inverse_xtx
    )

    hac_covariance = 0.5 * (
        hac_covariance
        + hac_covariance.T
    )

    # --------------------------------------------------------
    # F. Validate covariance matrices
    # --------------------------------------------------------

    for covariance_name, covariance in (
        (
            "classical",
            classical_covariance,
        ),
        (
            "HAC",
            hac_covariance,
        ),
    ):

        if not np.isfinite(
            covariance
        ).all():
            raise ValueError(
                f"{covariance_name} covariance "
                "contains non-finite values."
            )

        diagonal = np.diag(
            covariance
        )

        if (diagonal <= 0).any():
            raise ValueError(
                f"{covariance_name} covariance "
                "has nonpositive coefficient variances."
            )

        eigenvalues = np.linalg.eigvalsh(
            covariance
        )

        if eigenvalues.min() < (
            -1e-10
            * max(
                1.0,
                float(
                    np.max(
                        np.abs(eigenvalues)
                    )
                ),
            )
        ):
            raise ValueError(
                f"{covariance_name} covariance "
                "is materially non-positive-semidefinite."
            )

    # --------------------------------------------------------
    # G. Standard errors
    # --------------------------------------------------------

    classical_standard_errors = np.sqrt(
        np.diag(
            classical_covariance
        )
    )

    hac_standard_errors = np.sqrt(
        np.diag(
            hac_covariance
        )
    )

    # --------------------------------------------------------
    # H. Classical t-inference
    # --------------------------------------------------------

    classical_statistics = (
        coefficients
        / classical_standard_errors
    )

    classical_p_values = (
        2.0
        * student_t.sf(
            np.abs(
                classical_statistics
            ),
            df=df_residual,
        )
    )

    classical_critical_value = (
        student_t.ppf(
            (1.0 + confidence_level) / 2.0,
            df=df_residual,
        )
    )

    classical_ci_low = (
        coefficients
        - classical_critical_value
        * classical_standard_errors
    )

    classical_ci_high = (
        coefficients
        + classical_critical_value
        * classical_standard_errors
    )

    # --------------------------------------------------------
    # I. HAC asymptotic normal inference
    # --------------------------------------------------------

    hac_statistics = (
        coefficients
        / hac_standard_errors
    )

    hac_p_values = (
        2.0
        * norm.sf(
            np.abs(
                hac_statistics
            )
        )
    )

    hac_critical_value = (
        norm.ppf(
            (1.0 + confidence_level) / 2.0
        )
    )

    hac_ci_low = (
        coefficients
        - hac_critical_value
        * hac_standard_errors
    )

    hac_ci_high = (
        coefficients
        + hac_critical_value
        * hac_standard_errors
    )

    # --------------------------------------------------------
    # J. Construct coefficient inference output
    # --------------------------------------------------------

    result = pd.DataFrame(
        {
            "term": list(terms),
            "coefficient": coefficients,
            "n_observations": n,
            "n_parameters": p,
            "residual_degrees_of_freedom": df_residual,
            "hac_lags": hac_lags,
            "classical_standard_error": (
                classical_standard_errors
            ),
            "classical_t_statistic": (
                classical_statistics
            ),
            "classical_p_value": (
                classical_p_values
            ),
            "classical_ci_lower": (
                classical_ci_low
            ),
            "classical_ci_upper": (
                classical_ci_high
            ),
            "hac_standard_error": (
                hac_standard_errors
            ),
            "hac_z_statistic": (
                hac_statistics
            ),
            "hac_p_value": (
                hac_p_values
            ),
            "hac_ci_lower": (
                hac_ci_low
            ),
            "hac_ci_upper": (
                hac_ci_high
            ),
            "hac_small_sample_adjustment": (
                small_sample_adjustment
            ),
        }
    )

    return (
        result,
        classical_covariance,
        hac_covariance,
    )


# Research reference: 11_fama_french_factor_analytics.ipynb, code cell index 16.
def calculate_direct_regression_residual_diagnostics(
    X: np.ndarray,
    residuals: np.ndarray,
    *,
    diagnostic_lags: tuple[int, ...] = (
        1,
        5,
        10,
        20,
    ),
) -> dict:
    """
    Calculate regression residual diagnostics.

    Durbin-Watson:
        sum((e_t - e_(t-1))²) / sum(e_t²)

    Ljung-Box:
        Q = n(n+2) * sum[rho_k²/(n-k)]

    Koenker-Breusch-Pagan:
        n * auxiliary regression R²,
        with squared OLS residuals as the response.

    Jarque-Bera:
        n/6 * [skewness² + excess_kurtosis²/4]

    Tests are diagnostic, not proof of correct
    specification or independent residuals.
    """

    design = np.asarray(
        X,
        dtype=np.float64,
    )

    e = np.asarray(
        residuals,
        dtype=np.float64,
    ).reshape(-1)

    n, p = design.shape

    if len(e) != n:
        raise ValueError(
            "Residual and design lengths differ."
        )

    if not np.isfinite(e).all():
        raise ValueError(
            "Residual diagnostic inputs must be finite."
        )

    if n <= max(diagnostic_lags):
        raise ValueError(
            "Insufficient observations for diagnostic lags."
        )

    squared_sum = float(
        e @ e
    )

    if squared_sum <= 0:
        raise ValueError(
            "Residual diagnostics require "
            "positive residual variation."
        )

    # --------------------------------------------------------
    # A. Durbin-Watson
    # --------------------------------------------------------

    durbin_watson = float(
        np.sum(
            np.diff(e) ** 2
        )
        / squared_sum
    )

    # --------------------------------------------------------
    # B. Ljung-Box
    # --------------------------------------------------------

    centered_e = (
        e - e.mean()
    )

    autocorrelation_denominator = float(
        centered_e @ centered_e
    )

    if autocorrelation_denominator <= 0:
        raise ValueError(
            "Residual autocorrelation is undefined."
        )

    maximum_lag = max(
        diagnostic_lags
    )

    autocorrelations = np.array(
        [
            (
                centered_e[lag:]
                @ centered_e[:-lag]
            )
            / autocorrelation_denominator
            for lag in range(
                1,
                maximum_lag + 1,
            )
        ]
    )

    output = {
        "n_observations": n,
        "n_parameters": p,
        "durbin_watson": durbin_watson,
        "residual_mean": float(
            e.mean()
        ),
        "residual_standard_deviation": float(
            np.std(
                e,
                ddof=1,
            )
        ),
    }

    for lag in diagnostic_lags:

        q_statistic = (
            n * (n + 2)
            * np.sum(
                (
                    autocorrelations[:lag] ** 2
                )
                / (
                    n
                    - np.arange(
                        1,
                        lag + 1,
                    )
                )
            )
        )

        output[
            f"ljung_box_q_{lag}"
        ] = float(
            q_statistic
        )

        output[
            f"ljung_box_p_{lag}"
        ] = float(
            chi2.sf(
                q_statistic,
                df=lag,
            )
        )

    # --------------------------------------------------------
    # C. Koenker-Breusch-Pagan
    # --------------------------------------------------------

    auxiliary_y = (
        e ** 2
    )

    auxiliary_beta = np.linalg.lstsq(
        design,
        auxiliary_y,
        rcond=None,
    )[0]

    auxiliary_residual = (
        auxiliary_y
        - design @ auxiliary_beta
    )

    auxiliary_tss = float(
        np.sum(
            (
                auxiliary_y
                - auxiliary_y.mean()
            ) ** 2
        )
    )

    if auxiliary_tss <= 0:
        raise ValueError(
            "Heteroskedasticity diagnostic "
            "has zero auxiliary variation."
        )

    auxiliary_r2 = (
        1.0
        - float(
            auxiliary_residual
            @ auxiliary_residual
        )
        / auxiliary_tss
    )

    auxiliary_r2 = float(
        np.clip(
            auxiliary_r2,
            0.0,
            1.0,
        )
    )

    bp_statistic = (
        n * auxiliary_r2
    )

    output[
        "koenker_bp_statistic"
    ] = float(
        bp_statistic
    )

    output[
        "koenker_bp_p_value"
    ] = float(
        chi2.sf(
            bp_statistic,
            df=p - 1,
        )
    )

    # --------------------------------------------------------
    # D. Jarque-Bera
    # --------------------------------------------------------

    centered = (
        e - e.mean()
    )

    m2 = float(
        np.mean(
            centered ** 2
        )
    )

    m3 = float(
        np.mean(
            centered ** 3
        )
    )

    m4 = float(
        np.mean(
            centered ** 4
        )
    )

    if m2 <= 0:
        raise ValueError(
            "Residual normality diagnostic is undefined."
        )

    skewness = (
        m3 / m2 ** 1.5
    )

    excess_kurtosis = (
        m4 / m2 ** 2 - 3.0
    )

    jb_statistic = (
        n / 6.0
        * (
            skewness ** 2
            + excess_kurtosis ** 2 / 4.0
        )
    )

    output[
        "jarque_bera_statistic"
    ] = float(
        jb_statistic
    )

    output[
        "jarque_bera_p_value"
    ] = float(
        chi2.sf(
            jb_statistic,
            df=2,
        )
    )

    output["residual_skewness"] = float(
        skewness
    )

    output["residual_excess_kurtosis"] = float(
        excess_kurtosis
    )

    return output
