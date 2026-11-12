"""Factors lagging: validated research calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd



# Research reference: 11_fama_french_factor_analytics.ipynb, code cell index 10.
def align_prior_security_factor_coefficients(
    target_security_dates_df: pd.DataFrame,
    estimated_coefficients_df: pd.DataFrame,
    *,
    coefficient_columns: tuple[str, ...],
    model_id: str,
) -> pd.DataFrame:
    """
    Align the latest strictly prior factor coefficients.

    Inputs
    ------
    target_security_dates_df:
        Unique security_id, date observations.

    estimated_coefficients_df:
        Unique security_id, estimation_date observations
        with estimated regression coefficients.

    coefficient_columns:
        Ordered regression coefficient fields.

    model_id:
        Factor model identifier.

    Returns
    -------
    pd.DataFrame

        Key:
            security_id, date, model_id.

        Contains:
            estimation_date,
            coefficient_columns,
            model_available.

    Temporal rule
    -------------
    The selected estimation_date must be strictly
    earlier than the target return date.

    Same-day estimates are excluded.

    Missing coefficients remain NaN.

    No forward-return information may be used.
    """

    # --------------------------------------------------------
    # A. Validate required fields
    # --------------------------------------------------------

    required_target_columns = {
        "security_id",
        "date",
    }

    required_estimate_columns = {
        "security_id",
        "estimation_date",
        *coefficient_columns,
    }

    if not required_target_columns.issubset(
        target_security_dates_df.columns
    ):
        raise ValueError(
            "Target security-date columns are incomplete."
        )

    if not required_estimate_columns.issubset(
        estimated_coefficients_df.columns
    ):
        raise ValueError(
            "Estimated coefficient columns are incomplete."
        )

    # --------------------------------------------------------
    # B. Prepare independent working copies
    # --------------------------------------------------------

    targets = (
        target_security_dates_df[
            ["security_id", "date"]
        ]
        .copy()
    )

    estimates = (
        estimated_coefficients_df[
            [
                "security_id",
                "estimation_date",
                *coefficient_columns,
            ]
        ]
        .copy()
    )

    if (
        targets["security_id"].isna().any()
        or estimates["security_id"].isna().any()
    ):
        raise ValueError(
            "Missing security identifiers detected."
        )

    targets["security_id"] = (
        targets["security_id"].astype(str)
    )

    estimates["security_id"] = (
        estimates["security_id"].astype(str)
    )

    targets["date"] = pd.to_datetime(
        targets["date"],
        errors="raise",
    ).astype("datetime64[ns]")

    estimates["estimation_date"] = pd.to_datetime(
        estimates["estimation_date"],
        errors="raise",
    ).astype("datetime64[ns]")

    if (
        targets["date"].isna().any()
        or estimates["estimation_date"].isna().any()
    ):
        raise ValueError(
            "Lagged exposure dates contain missing values."
        )

    if targets["date"].dt.tz is not None:
        raise ValueError(
            "Target dates must be timezone-naive."
        )

    if estimates["estimation_date"].dt.tz is not None:
        raise ValueError(
            "Estimation dates must be timezone-naive."
        )

    if not targets["date"].eq(
        targets["date"].dt.normalize()
    ).all():
        raise ValueError(
            "Target dates contain intraday timestamps."
        )

    if not estimates["estimation_date"].eq(
        estimates["estimation_date"].dt.normalize()
    ).all():
        raise ValueError(
            "Estimation dates contain intraday timestamps."
        )

    # --------------------------------------------------------
    # C. Validate unique source and target keys
    # --------------------------------------------------------

    if targets.duplicated(
        subset=["security_id", "date"]
    ).any():
        raise ValueError(
            "Duplicate target security-date keys."
        )

    if estimates.duplicated(
        subset=["security_id", "estimation_date"]
    ).any():
        raise ValueError(
            "Duplicate security estimation-date keys."
        )

    estimate_values = estimates[
        list(coefficient_columns)
    ].to_numpy(dtype=np.float64)

    if not np.isfinite(estimate_values).all():
        raise ValueError(
            "Lagged alignment requires complete, "
            "finite estimated coefficients."
        )

    # --------------------------------------------------------
    # D. Sort globally by the as-of join date
    # --------------------------------------------------------

    # pandas.merge_asof requires the primary as-of
    # keys to be sorted globally, not only within
    # each security group.

    targets = (
        targets
        .sort_values(["date", "security_id"])
        .reset_index(drop=True)
    )

    estimates = (
        estimates
        .sort_values(
            ["estimation_date", "security_id"]
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # E. Perform strictly backward coefficient alignment
    # --------------------------------------------------------

    aligned = pd.merge_asof(
        targets,
        estimates,
        left_on="date",
        right_on="estimation_date",
        by="security_id",
        direction="backward",
        allow_exact_matches=False,
    )

    # --------------------------------------------------------
    # F. Validate temporal integrity
    # --------------------------------------------------------

    if len(aligned) != len(targets):
        raise ValueError(
            "Lagged alignment changed target row count."
        )

    modeled_mask = (
        aligned["estimation_date"].notna()
    )

    if not aligned.loc[
        modeled_mask,
        "estimation_date",
    ].lt(
        aligned.loc[modeled_mask, "date"]
    ).all():
        raise ValueError(
            "LOOK-AHEAD VIOLATION: Same-day or future "
            "regression coefficients were assigned."
        )

    available_coefficients = (
        aligned[
            list(coefficient_columns)
        ].notna()
    )

    complete_coefficients = (
        available_coefficients.all(axis=1)
    )

    partially_available = (
        available_coefficients.any(axis=1)
        & ~complete_coefficients
    )

    if partially_available.any():
        raise ValueError(
            "Lagged exposure alignment contains "
            "partially available coefficient vectors."
        )

    if not (
        modeled_mask == complete_coefficients
    ).all():
        raise ValueError(
            "Coefficient availability does not reconcile "
            "to matched estimation dates."
        )

    aligned["model_id"] = model_id

    aligned["model_available"] = modeled_mask

    return (
        aligned
        .sort_values(["security_id", "date"])
        .reset_index(drop=True)
    )
