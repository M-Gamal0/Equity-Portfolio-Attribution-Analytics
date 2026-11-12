"""Risk fixed weights: validated research calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd



# Research reference: 09_risk_analytics.ipynb, code cell index 14.
def select_asof_closing_weights(
    holdings_df: pd.DataFrame,
    portfolio_ids: tuple[str, ...],
    as_of_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Select positive EOP security weights on an exact as-of date.

    The returned weights are not renormalized.

    Nonunit weight sums are reported later, rather than
    silently assuming an unobserved cash allocation.
    """

    required = {
        "portfolio_id",
        "date",
        "security_id",
        "weight_eop",
        "sector",
    }

    if not required.issubset(holdings_df.columns):
        raise ValueError(
            "Missing required as-of holdings columns."
        )

    if not portfolio_ids or len(set(portfolio_ids)) != len(
        portfolio_ids
    ):
        raise ValueError(
            "Portfolio identifiers must be nonempty and unique."
        )

    cutoff = pd.Timestamp(as_of_date)

    if pd.isna(cutoff):
        raise ValueError("As-of date cannot be missing.")

    selected = holdings_df.loc[
        holdings_df["portfolio_id"].isin(portfolio_ids)
        & holdings_df["date"].eq(cutoff),
        [
            "portfolio_id",
            "date",
            "security_id",
            "weight_eop",
            "sector",
        ],
    ].copy()

    if (
        selected.empty
        or selected["security_id"].isna().any()
        or selected.duplicated(
            ["portfolio_id", "security_id"]
        ).any()
    ):
        raise ValueError(
            "Missing or duplicate as-of security holdings."
        )

    if (
        not pd.api.types.is_numeric_dtype(
            selected["weight_eop"]
        )
        or pd.api.types.is_bool_dtype(
            selected["weight_eop"]
        )
        or pd.api.types.is_complex_dtype(
            selected["weight_eop"]
        )
    ):
        raise TypeError(
            "Closing weights must contain real numeric values."
        )

    weights = selected[
        "weight_eop"
    ].to_numpy(dtype=np.float64)

    if not np.isfinite(weights).all():
        raise ValueError(
            "Closing weights contain non-finite observations."
        )

    if np.any(weights < 0.0):
        raise ValueError(
            "Negative closing holdings are outside the "
            "long-only source portfolio specification."
        )

    selected = selected.loc[
        selected["weight_eop"].gt(0.0)
    ].copy()

    if set(selected["portfolio_id"]) != set(portfolio_ids):
        raise ValueError(
            "At least one portfolio has no positive closing holdings."
        )

    return (
        selected.sort_values(
            ["portfolio_id", "security_id"]
        ).reset_index(drop=True)
    )


# Research reference: 09_risk_analytics.ipynb, code cell index 14.
def calculate_fixed_weight_risk_components(
    covariance_daily: pd.DataFrame,
    weights: pd.Series,
    annualization_sessions: int = 252,
) -> tuple[pd.DataFrame, dict]:
    """
    Attribute covariance-based risk for a fixed weight vector.

    Supports long-only portfolio weights and signed active
    weight differences.

    No matrix inversion or covariance adjustment is performed.

    Component risk shares sum to one when total modeled
    volatility is positive.

    Zero-variance portfolios have unavailable marginal
    risk, component volatility, and risk-share statistics.

    Diversification ratio:
        Sum(|weight_i| * standalone_volatility_i)
        / modeled_portfolio_volatility

    For signed active weights this is explicitly a
    gross-exposure diversification measure, not the
    conventional long-only portfolio diversification ratio.
    """

    if (
        not isinstance(
            annualization_sessions, (int, np.integer)
        )
        or isinstance(
            annualization_sessions, (bool, np.bool_)
        )
        or annualization_sessions <= 0
    ):
        raise ValueError(
            "Annualization sessions must be a positive integer."
        )

    if not isinstance(covariance_daily, pd.DataFrame):
        raise TypeError(
            "Covariance must be a pandas DataFrame."
        )

    if not isinstance(weights, pd.Series):
        raise TypeError(
            "Weights must be a labeled pandas Series."
        )

    securities = covariance_daily.index

    if (
        len(securities) == 0
        or securities.has_duplicates
        or securities.hasnans
        or not securities.equals(covariance_daily.columns)
    ):
        raise ValueError(
            "Covariance must have matching unique row "
            "and column security identifiers."
        )

    if (
        weights.index.has_duplicates
        or weights.index.hasnans
        or set(weights.index) != set(securities)
    ):
        raise ValueError(
            "Weight identifiers must exactly match "
            "covariance security identifiers."
        )

    matrix = covariance_daily.to_numpy(
        dtype=np.float64
    )

    aligned_weights = weights.reindex(
        securities
    ).to_numpy(dtype=np.float64)

    if (
        not np.isfinite(matrix).all()
        or not np.isfinite(aligned_weights).all()
    ):
        raise ValueError(
            "Covariance and weights must be finite."
        )

    np.testing.assert_allclose(
        matrix,
        matrix.T,
        rtol=1e-12,
        atol=1e-14,
        err_msg="Covariance matrix is not symmetric.",
    )

    # --------------------------------------------------------
    # Validate positive semidefiniteness
    # --------------------------------------------------------

    eigenvalues = np.linalg.eigvalsh(matrix)

    maximum_eigenvalue = float(
        max(eigenvalues[-1], 0.0)
    )

    psd_tolerance = max(
        1e-14,
        1e-10 * maximum_eigenvalue,
    )

    if eigenvalues[0] < -psd_tolerance:
        raise ValueError(
            "Covariance is not positive semidefinite."
        )

    annual_covariance = (
        matrix * annualization_sessions
    )

    marginal_variance_vector = (
        annual_covariance @ aligned_weights
    )

    component_variance = (
        aligned_weights * marginal_variance_vector
    )

    variance_annualized_raw = float(
        aligned_weights @ marginal_variance_vector
    )

    variance_tolerance = (
        1e-12
        * max(
            1.0,
            float(
                np.max(
                    np.abs(np.diag(annual_covariance))
                )
            )
            * float(np.abs(aligned_weights).sum()) ** 2,
        )
    )

    if variance_annualized_raw < -variance_tolerance:
        raise ArithmeticError(
            "Negative modeled variance outside tolerance."
        )

    variance_annualized = max(
        variance_annualized_raw,
        0.0,
    )

    modeled_volatility = float(
        np.sqrt(variance_annualized)
    )

    standalone_variance = (
        np.diag(annual_covariance)
    )

    if np.any(standalone_variance < -1e-12):
        raise ArithmeticError(
            "Negative standalone security variance."
        )

    standalone_volatility = np.sqrt(
        np.maximum(standalone_variance, 0.0)
    )

    # --------------------------------------------------------
    # Component risk, marginal risk and risk shares
    # --------------------------------------------------------

    count = len(securities)

    if modeled_volatility > 0.0:

        marginal_volatility = (
            marginal_variance_vector / modeled_volatility
        )

        component_volatility = (
            component_variance / modeled_volatility
        )

        risk_share = (
            component_volatility / modeled_volatility
        )

        status = "PASS"

    else:

        marginal_volatility = np.full(
            count,
            np.nan,
        )

        component_volatility = np.full(
            count,
            np.nan,
        )

        risk_share = np.full(
            count,
            np.nan,
        )

        status = "UNAVAILABLE_ZERO_MODELED_VOLATILITY"

    # --------------------------------------------------------
    # Diversification and concentration diagnostics
    # --------------------------------------------------------

    gross_weight = float(
        np.abs(aligned_weights).sum()
    )

    net_weight = float(
        aligned_weights.sum()
    )

    sum_squared_weights = float(
        np.dot(
            aligned_weights,
            aligned_weights,
        )
    )

    effective_exposure_count = (
        gross_weight**2 / sum_squared_weights
        if sum_squared_weights > 0.0
        else np.nan
    )

    gross_weighted_standalone_volatility = float(
        np.dot(
            np.abs(aligned_weights),
            standalone_volatility,
        )
    )

    diversification_ratio = (
        gross_weighted_standalone_volatility
        / modeled_volatility
        if modeled_volatility > 0.0
        else np.nan
    )

    if status == "PASS":

        absolute_components = np.abs(
            component_volatility
        )

        absolute_component_sum = float(
            absolute_components.sum()
        )

        squared_component_sum = float(
            np.dot(
                component_volatility,
                component_volatility,
            )
        )

        effective_risk_contributor_count = (
            absolute_component_sum**2
            / squared_component_sum
            if squared_component_sum > 0.0
            else np.nan
        )

        largest_absolute_component_fraction = (
            float(
                absolute_components.max()
                / absolute_component_sum
            )
            if absolute_component_sum > 0.0
            else np.nan
        )

    else:

        effective_risk_contributor_count = np.nan
        largest_absolute_component_fraction = np.nan

    # --------------------------------------------------------
    # Validate Euler risk decomposition
    # --------------------------------------------------------

    np.testing.assert_allclose(
        component_variance.sum(),
        variance_annualized,
        rtol=1e-10,
        atol=1e-12,
    )

    if status == "PASS":

        np.testing.assert_allclose(
            component_volatility.sum(),
            modeled_volatility,
            rtol=1e-10,
            atol=1e-12,
        )

        np.testing.assert_allclose(
            risk_share.sum(),
            1.0,
            rtol=1e-10,
            atol=1e-12,
        )

        np.testing.assert_allclose(
            component_volatility,
            aligned_weights * marginal_volatility,
            rtol=1e-10,
            atol=1e-12,
        )

    # --------------------------------------------------------
    # Construct outputs
    # --------------------------------------------------------

    detail = pd.DataFrame(
        {
            "security_id": securities,
            "weight": aligned_weights,
            "standalone_volatility_annualized": (
                standalone_volatility
            ),
            "marginal_variance_annualized": (
                marginal_variance_vector
            ),
            "marginal_volatility_annualized": (
                marginal_volatility
            ),
            "component_variance_annualized": (
                component_variance
            ),
            "component_volatility_annualized": (
                component_volatility
            ),
            "risk_share": risk_share,
            "status": status,
        }
    )

    diagnostics = {
        "security_count": count,
        "annualization_sessions": annualization_sessions,
        "gross_weight": gross_weight,
        "net_weight": net_weight,
        "variance_annualized": variance_annualized,
        "volatility_annualized": modeled_volatility,
        "sum_component_variance": float(
            component_variance.sum()
        ),
        "sum_component_volatility": (
            float(component_volatility.sum())
            if status == "PASS"
            else np.nan
        ),
        "sum_risk_shares": (
            float(risk_share.sum())
            if status == "PASS"
            else np.nan
        ),
        "gross_weighted_standalone_volatility": (
            gross_weighted_standalone_volatility
        ),
        "diversification_ratio": diversification_ratio,
        "effective_exposure_count": (
            effective_exposure_count
        ),
        "effective_risk_contributor_count": (
            effective_risk_contributor_count
        ),
        "largest_absolute_component_fraction": (
            largest_absolute_component_fraction
        ),
        "negative_risk_contributors": int(
            np.count_nonzero(
                component_variance < -1e-14
            )
        ),
        "status": status,
    }

    return detail, diagnostics
