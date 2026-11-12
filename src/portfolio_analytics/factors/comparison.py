"""Factors comparison: validated research calculations."""

from __future__ import annotations

from scipy.stats import f as fisher_f
import numpy as np



# Research reference: 11_fama_french_factor_analytics.ipynb, code cell index 18.
def calculate_nested_factor_model_f_test(
    *,
    restricted_rss: float,
    unrestricted_rss: float,
    restricted_parameters: int,
    unrestricted_parameters: int,
    observations: int,
) -> dict:
    """
    Classical partial F test for two nested OLS models
    estimated on exactly the same response and dates.

    F =
        ((RSS_R - RSS_U) / (p_U - p_R))
        / (RSS_U / (n - p_U))

    This is not HAC-robust inference.
    """

    if not (
        0 < restricted_parameters
        < unrestricted_parameters
        < observations
    ):
        raise ValueError(
            "Invalid nested regression dimensions."
        )

    if not (
        np.isfinite(restricted_rss)
        and np.isfinite(unrestricted_rss)
    ):
        raise ValueError(
            "Nested regression RSS must be finite."
        )

    if (
        restricted_rss <= 0
        or unrestricted_rss <= 0
    ):
        raise ValueError(
            "Nested regression requires positive RSS."
        )

    improvement = (
        restricted_rss - unrestricted_rss
    )

    numerical_tolerance = (
        1e-9
        * max(
            abs(restricted_rss),
            abs(unrestricted_rss),
        )
    )

    if improvement < -numerical_tolerance:
        raise ValueError(
            "Nested unrestricted regression has "
            "materially higher residual RSS."
        )

    improvement = max(
        improvement,
        0.0,
    )

    numerator_df = (
        unrestricted_parameters
        - restricted_parameters
    )

    denominator_df = (
        observations - unrestricted_parameters
    )

    f_statistic = (
        (improvement / numerator_df)
        / (
            unrestricted_rss / denominator_df
        )
    )

    p_value = float(
        fisher_f.sf(
            f_statistic,
            numerator_df,
            denominator_df,
        )
    )

    return {
        "rss_improvement": improvement,
        "numerator_df": numerator_df,
        "denominator_df": denominator_df,
        "f_statistic": float(f_statistic),
        "classical_f_p_value": p_value,
        "classical_f_significant_5pct": (
            p_value < 0.05
        ),
    }
