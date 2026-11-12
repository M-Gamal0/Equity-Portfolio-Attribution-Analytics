"""Risk absolute: validated research calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd



# Research reference: 09_risk_analytics.ipynb, code cell index 3.
def calculate_absolute_risk(
    returns: pd.Series | np.ndarray,
    annualization_sessions: int,
    variance_ddof: int = 1,
    downside_target_daily: float = 0.0,
) -> dict:
    """
    Calculate absolute risk from daily simple returns.

    Parameters
    ----------
    returns:
        One-dimensional daily simple returns in decimals.

    annualization_sessions:
        Annualization factor in trading sessions per year.

    variance_ddof:
        Degrees-of-freedom adjustment for return variance.
        The approved working sample estimator uses ddof=1.

    downside_target_daily:
        Daily minimum acceptable return in decimal units.

    Returns
    -------
    dict
        Observation count, daily return mean, sample variance,
        annualized variance, daily/annualized volatility,
        downside semivariance, downside deviation, and
        downside observation count.

    Notes
    -----
    Downside semivariance is defined as the lower partial
    second moment relative to the supplied target.

    Its denominator is the complete observation count,
    not the number of below-target observations.

    No missing observations are filled or discarded.

    Raises
    ------
    ValueError
        Invalid inputs or insufficient observations.

    TypeError
        Non-numeric returns or invalid parameter types.
    """

    if (
        not isinstance(
            annualization_sessions,
            (int, np.integer),
        )
        or isinstance(annualization_sessions, (bool, np.bool_))
        or annualization_sessions <= 0
    ):
        raise ValueError(
            "annualization_sessions must be a positive integer."
        )

    if (
        not isinstance(variance_ddof, (int, np.integer))
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
            "downside_target_daily must be a finite number."
        )

    return_values = np.asarray(returns)

    if return_values.ndim != 1:
        raise ValueError(
            "Returns must be a one-dimensional series."
        )

    if not pd.api.types.is_numeric_dtype(
        return_values.dtype
    ):
        raise TypeError(
            "Returns must have a numeric dtype."
        )

    return_values = return_values.astype(
        np.float64,
        copy=False,
    )

    observations = len(return_values)

    if observations <= variance_ddof:
        raise ValueError(
            "Insufficient observations for the selected "
            "sample variance estimator."
        )

    if not np.isfinite(return_values).all():
        raise ValueError(
            "Returns contain missing or non-finite values."
        )

    if np.any(return_values < -1.0):
        raise ValueError(
            "Daily simple returns cannot be below -100%."
        )

    # --------------------------------------------------------
    # Arithmetic mean and sample variance
    # --------------------------------------------------------

    mean_daily_return = float(
        np.mean(return_values)
    )

    daily_variance = float(
        np.var(
            return_values,
            ddof=variance_ddof,
        )
    )

    daily_volatility = float(
        np.sqrt(daily_variance)
    )

    annualized_variance = float(
        daily_variance * annualization_sessions
    )

    annualized_volatility = float(
        np.sqrt(annualized_variance)
    )

    # --------------------------------------------------------
    # Downside lower partial second moment
    # --------------------------------------------------------

    downside_differences = np.minimum(
        return_values - downside_target_daily,
        0.0,
    )

    downside_observations = int(
        np.count_nonzero(downside_differences)
    )

    downside_lpm2_daily = float(
        np.mean(downside_differences**2)
    )

    annualized_downside_semivariance = float(
        downside_lpm2_daily * annualization_sessions
    )

    annualized_downside_deviation = float(
        np.sqrt(annualized_downside_semivariance)
    )

    return {
        "observations": observations,
        "mean_daily_return": mean_daily_return,
        "daily_variance": daily_variance,
        "daily_volatility": daily_volatility,
        "annualized_variance": annualized_variance,
        "annualized_volatility": annualized_volatility,
        "downside_observations": downside_observations,
        "downside_lpm2_daily": downside_lpm2_daily,
        "annualized_downside_semivariance": (
            annualized_downside_semivariance
        ),
        "annualized_downside_deviation": (
            annualized_downside_deviation
        ),
    }
