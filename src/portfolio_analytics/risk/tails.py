"""Risk tails: validated research calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd



# Research reference: 09_risk_analytics.ipynb, code cell index 8.
def calculate_historical_tail_risk(
    returns: pd.Series | np.ndarray,
    dates: pd.Series | pd.DatetimeIndex,
    as_of_date: pd.Timestamp,
    confidence_levels: tuple[float, ...],
    minimum_tail_observation_mass: float = 20.0,
    return_kind: str = "SIMPLE_TOTAL",
) -> pd.DataFrame:
    """
    Calculate historical one-session Value at Risk and ES.

    Parameters
    ----------
    returns:
        Daily returns in decimal units.

    dates:
        Matching, ascending return-interval end dates.

    as_of_date:
        Required observed cutoff date. Later observations
        are excluded before estimating the distribution.

    confidence_levels:
        Unique confidence levels strictly between 0 and 1.

    minimum_tail_observation_mass:
        Provisional minimum effective tail-sample threshold.
        Insufficient tail mass produces an explicit warning
        status without inventing additional observations.

    return_kind:
        SIMPLE_TOTAL:
            Portfolio total returns, bounded below by -1.

        ARITHMETIC_ACTIVE:
            Portfolio minus benchmark returns.
            These spreads can fall below -1.

    Returns
    -------
    pd.DataFrame
        One row per confidence level.

    Empirical VaR
    -------------
    Sort losses in ascending order:

        L(1) <= L(2) <= ... <= L(N)

    VaR(alpha) = L(ceil(alpha * N))

    Empirical Expected Shortfall
    ---------------------------
    Let m = N * (1 - alpha).

    Take the worst floor(m) losses, together with
    fractional weight m - floor(m) on the next loss.

    Divide the weighted sum by m.

    This is a fractional-order-statistic tail estimator.
    It is not the unweighted average of observations
    strictly exceeding VaR.

    Positive VaR/ES denotes loss; negative denotes gain.

    No observations are filled or silently discarded.
    """

    # --------------------------------------------------------
    # Validate methodology parameters
    # --------------------------------------------------------

    if return_kind not in (
        "SIMPLE_TOTAL",
        "ARITHMETIC_ACTIVE",
    ):
        raise ValueError(
            "Unsupported return_kind."
        )

    if (
        not isinstance(
            minimum_tail_observation_mass,
            (int, float, np.integer, np.floating),
        )
        or isinstance(
            minimum_tail_observation_mass,
            (bool, np.bool_),
        )
        or not np.isfinite(
            minimum_tail_observation_mass
        )
        or minimum_tail_observation_mass <= 0
    ):
        raise ValueError(
            "Minimum tail-observation mass must be "
            "a positive finite number."
        )

    if not confidence_levels:
        raise ValueError(
            "At least one confidence level is required."
        )

    for confidence in confidence_levels:

        if (
            not isinstance(
                confidence,
                (int, float, np.integer, np.floating),
            )
            or isinstance(confidence, (bool, np.bool_))
            or not np.isfinite(confidence)
            or not 0.0 < confidence < 1.0
        ):
            raise ValueError(
                "Confidence levels must be finite "
                "numbers strictly between 0 and 1."
            )

    if len(set(confidence_levels)) != len(
        confidence_levels
    ):
        raise ValueError(
            "Duplicate confidence levels are not permitted."
        )

    # --------------------------------------------------------
    # Validate input dimensions, types, and dates
    # --------------------------------------------------------

    values = np.asarray(returns)

    if values.ndim != 1:
        raise ValueError(
            "Returns must be one-dimensional."
        )

    if (
        not pd.api.types.is_numeric_dtype(values.dtype)
        or pd.api.types.is_bool_dtype(values.dtype)
        or pd.api.types.is_complex_dtype(values.dtype)
    ):
        raise TypeError(
            "Returns must contain real numeric values."
        )

    values = values.astype(
        np.float64,
        copy=False,
    )

    observed_dates = pd.DatetimeIndex(
        pd.to_datetime(dates, errors="raise")
    )

    if len(values) != len(observed_dates):
        raise ValueError(
            "Return and date observation counts must match."
        )

    if (
        observed_dates.hasnans
        or not observed_dates.is_unique
        or not observed_dates.is_monotonic_increasing
    ):
        raise ValueError(
            "Dates must be non-null, unique, and increasing."
        )

    cutoff = pd.Timestamp(as_of_date)

    if pd.isna(cutoff):
        raise ValueError(
            "As-of date cannot be missing."
        )

    if cutoff not in observed_dates:
        raise ValueError(
            "As-of date must exist in the return calendar."
        )

    # --------------------------------------------------------
    # Apply as-of cutoff before examining return values
    # --------------------------------------------------------

    eligible = observed_dates <= cutoff

    sample = values[eligible]
    sample_dates = observed_dates[eligible]

    observations = len(sample)

    if observations < 2:
        raise ValueError(
            "Historical tail risk requires at least "
            "two observed returns."
        )

    if not np.isfinite(sample).all():
        raise ValueError(
            "Historical estimation sample contains "
            "non-finite returns."
        )

    if (
        return_kind == "SIMPLE_TOTAL"
        and np.any(sample < -1.0)
    ):
        raise ValueError(
            "Portfolio simple returns cannot be below -100%."
        )

    # --------------------------------------------------------
    # Sort observed daily losses
    # --------------------------------------------------------

    losses_ascending = np.sort(
        -sample
    )

    losses_descending = losses_ascending[::-1]

    # --------------------------------------------------------
    # Calculate each confidence level independently
    # --------------------------------------------------------

    records = []

    for confidence in sorted(confidence_levels):

        # Nearest-rank empirical VaR.

        var_rank = int(
            np.ceil(confidence * observations)
        )

        var_loss = float(
            losses_ascending[var_rank - 1]
        )

        # Exact fractional mass of the worst sample tail.

        tail_mass = float(
            observations * (1.0 - confidence)
        )

        fully_weighted_count = int(
            np.floor(tail_mass)
        )

        fractional_weight = float(
            tail_mass - fully_weighted_count
        )

        tail_loss_sum = float(
            np.sum(
                losses_descending[
                    :fully_weighted_count
                ],
                dtype=np.float64,
            )
        )

        if fractional_weight > 0.0:

            tail_loss_sum += (
                fractional_weight
                * float(
                    losses_descending[
                        fully_weighted_count
                    ]
                )
            )

        expected_shortfall = float(
            tail_loss_sum / tail_mass
        )

        if not np.isfinite(expected_shortfall):
            raise ArithmeticError(
                "Expected Shortfall is non-finite."
            )

        if expected_shortfall < var_loss - 1e-12:
            raise ArithmeticError(
                "Expected Shortfall is below empirical VaR."
            )

        # Evaluate tail-sample coverage.

        coverage_status = (
            "PASS"
            if tail_mass >= minimum_tail_observation_mass
            else "LOW_EFFECTIVE_TAIL_SAMPLE"
        )

        records.append(
            {
                "as_of_date": cutoff,
                "sample_start": sample_dates.min(),
                "sample_end": sample_dates.max(),
                "observations": observations,
                "confidence_level": float(confidence),
                "var_loss_daily": var_loss,
                "expected_shortfall_loss_daily": (
                    expected_shortfall
                ),
                "effective_tail_observation_mass": (
                    tail_mass
                ),
                "fully_weighted_tail_observations": (
                    fully_weighted_count
                ),
                "fractional_boundary_weight": (
                    fractional_weight
                ),
                "minimum_tail_observation_mass": (
                    float(minimum_tail_observation_mass)
                ),
                "return_kind": return_kind,
                "coverage_status": coverage_status,
            }
        )

    return pd.DataFrame(records)
