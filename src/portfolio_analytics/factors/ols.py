"""Factors ols: validated research calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd

OLS_RECONCILIATION_TOLERANCE = 1e-12

# Research reference: 11_fama_french_factor_analytics.ipynb, code cell index 3.
def fit_ols_factor_regression(
    observations_df: pd.DataFrame,
    *,
    factor_columns: tuple[str, ...],
    model_id: str,
    response_column: str = "excess_return",
    date_column: str = "date",
    max_condition_number: float = 1e10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Estimate one historical factor regression using OLS.

    Parameters
    ----------
    observations_df:
        Complete, single-entity regression observations.

        Required columns:
            date_column,
            response_column,
            factor_columns.

        The dependent variable must be daily excess return.

    factor_columns:
        Selected independent factor columns.

    model_id:
        Identifier of the selected regression model.

    response_column:
        Daily excess return column.

    date_column:
        Daily return-interval end date.

    max_condition_number:
        Maximum permitted condition number of the
        unscaled regression design matrix.

    Returns
    -------
    tuple of three pandas DataFrames:

        1. fit_summary_df
            One row per regression estimation.

        2. coefficients_df
            One row per intercept or factor coefficient.

        3. residuals_df
            One row per regression observation.

    Conventions
    -----------
    - Intercept is included.
    - Returns and factors use daily decimal units.
    - Factor betas are dimensionless sensitivities.
    - No annualization is performed.
    - No observations are silently dropped.
    - No coefficient inference is calculated.
    - Regression estimation uses only supplied rows.

    Raises
    ------
    ValueError:
        Invalid sample, missing observations, insufficient
        residual degrees of freedom, rank deficiency,
        or excessive matrix conditioning.

    TypeError:
        Non-numeric regression variables.
    """

    # --------------------------------------------------------
    # A. Validate regression specification
    # --------------------------------------------------------

    if not factor_columns:
        raise ValueError(
            "At least one regression factor is required."
        )

    if len(set(factor_columns)) != len(factor_columns):
        raise ValueError(
            "Regression factor names must be unique."
        )

    if response_column in factor_columns:
        raise ValueError(
            "Dependent variable cannot also be a factor."
        )

    if date_column in factor_columns:
        raise ValueError(
            "Date cannot be a regression factor."
        )

    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError(
            "model_id must be a nonempty string."
        )

    if (
        not np.isfinite(max_condition_number)
        or max_condition_number <= 1
    ):
        raise ValueError(
            "max_condition_number must be finite "
            "and greater than one."
        )

    required_columns = [
        date_column,
        response_column,
        *factor_columns,
    ]

    missing_columns = (
        set(required_columns)
        - set(observations_df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Regression input missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # --------------------------------------------------------
    # B. Prepare and validate sample dates
    # --------------------------------------------------------

    sample = observations_df[
        required_columns
    ].copy()

    sample[date_column] = pd.to_datetime(
        sample[date_column],
        errors="raise",
    )

    if sample[date_column].isna().any():
        raise ValueError(
            "Regression sample contains missing dates."
        )

    if sample[date_column].dt.tz is not None:
        raise ValueError(
            "Regression dates must be timezone-naive."
        )

    if not sample[date_column].eq(
        sample[date_column].dt.normalize()
    ).all():
        raise ValueError(
            "Regression contains non-daily timestamps."
        )

    if sample[date_column].duplicated().any():
        raise ValueError(
            "Duplicate regression dates detected."
        )

    sample = (
        sample
        .sort_values(date_column)
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # C. Validate numeric regression variables
    # --------------------------------------------------------

    numeric_columns = [
        response_column,
        *factor_columns,
    ]

    for column in numeric_columns:

        if (
            not pd.api.types.is_numeric_dtype(
                sample[column]
            )
            or pd.api.types.is_bool_dtype(
                sample[column]
            )
        ):
            raise TypeError(
                f"Regression column {column} "
                "must be numeric and non-Boolean."
            )

    numeric_values = sample[
        numeric_columns
    ].to_numpy(
        dtype=np.float64,
        na_value=np.nan,
    )

    if not np.isfinite(numeric_values).all():
        raise ValueError(
            "Regression sample contains missing "
            "or non-finite numeric observations."
        )

    # --------------------------------------------------------
    # D. Construct regression response and design matrix
    # --------------------------------------------------------

    y = sample[
        response_column
    ].to_numpy(
        dtype=np.float64
    )

    factor_matrix = sample[
        list(factor_columns)
    ].to_numpy(
        dtype=np.float64
    )

    n_observations = len(y)

    n_factors = len(factor_columns)

    n_parameters = n_factors + 1

    residual_degrees_of_freedom = (
        n_observations - n_parameters
    )

    if residual_degrees_of_freedom <= 0:
        raise ValueError(
            "Insufficient observations for positive "
            "regression residual degrees of freedom."
        )

    X = np.column_stack(
        [
            np.ones(
                n_observations,
                dtype=np.float64,
            ),
            factor_matrix,
        ]
    )

    # --------------------------------------------------------
    # E. Validate regression matrix rank and conditioning
    # --------------------------------------------------------

    matrix_rank = int(
        np.linalg.matrix_rank(X)
    )

    if matrix_rank != n_parameters:
        raise ValueError(
            "Regression design matrix is rank-deficient."
        )

    condition_number = float(
        np.linalg.cond(X)
    )

    if not np.isfinite(condition_number):
        raise ValueError(
            "Regression design matrix has "
            "a non-finite condition number."
        )

    if condition_number > max_condition_number:
        raise ValueError(
            "Regression design matrix exceeds "
            "the permitted condition-number threshold. "
            f"Observed: {condition_number:.6g}"
        )

    # --------------------------------------------------------
    # F. Estimate regression coefficients
    # --------------------------------------------------------

    coefficients, _, estimated_rank, _ = (
        np.linalg.lstsq(
            X,
            y,
            rcond=None,
        )
    )

    if estimated_rank != n_parameters:
        raise ValueError(
            "OLS solver reported a rank-deficient matrix."
        )

    if not np.isfinite(coefficients).all():
        raise ValueError(
            "OLS returned non-finite coefficients."
        )

    # --------------------------------------------------------
    # G. Calculate fitted returns and residuals
    # --------------------------------------------------------

    fitted_returns = X @ coefficients

    residual_returns = (
        y - fitted_returns
    )

    sum_squared_residuals = float(
        residual_returns @ residual_returns
    )

    centered_y = (
        y - y.mean()
    )

    total_sum_squares = float(
        centered_y @ centered_y
    )

    # --------------------------------------------------------
    # H. Calculate regression fit statistics
    # --------------------------------------------------------

    constant_response = bool(
        np.all(y == y[0])
    )

    if constant_response or total_sum_squares == 0:

        r_squared = np.nan

        adjusted_r_squared = np.nan

        fit_status = (
            "FIT_COMPLETED_UNDEFINED_R2"
        )

    else:

        r_squared = (
            1.0
            - sum_squared_residuals / total_sum_squares
        )

        adjusted_r_squared = (
            1.0
            - (
                (1.0 - r_squared)
                * (n_observations - 1)
                / residual_degrees_of_freedom
            )
        )

        fit_status = "FIT_COMPLETED"

    residual_variance = (
        sum_squared_residuals
        / residual_degrees_of_freedom
    )

    residual_standard_error = float(
        np.sqrt(residual_variance)
    )

    residual_mean = float(
        residual_returns.mean()
    )

    # --------------------------------------------------------
    # I. Validate residual accounting and OLS orthogonality
    # --------------------------------------------------------

    reconstruction_error = float(
        np.max(
            np.abs(
                y - (fitted_returns + residual_returns)
            )
        )
    )

    if reconstruction_error > OLS_RECONCILIATION_TOLERANCE:
        raise ValueError(
            "Regression fitted-return/residual "
            "reconciliation failed."
        )

    normal_equation_residual = (
        X.T @ residual_returns
    )

    max_normal_equation_error = float(
        np.max(
            np.abs(normal_equation_residual)
        )
    )

    if max_normal_equation_error > (
        OLS_RECONCILIATION_TOLERANCE
        * max(1.0, float(np.linalg.norm(y, ord=1)))
    ):
        raise ValueError(
            "OLS residual orthogonality validation failed."
        )

    # --------------------------------------------------------
    # J. Construct standardized regression summary
    # --------------------------------------------------------

    estimation_date = sample[
        date_column
    ].max()

    sample_start_date = sample[
        date_column
    ].min()

    fit_summary_df = pd.DataFrame(
        [
            {
                "model_id": model_id,
                "estimation_date": estimation_date,
                "sample_start_date": sample_start_date,
                "sample_end_date": estimation_date,
                "n_observations": n_observations,
                "n_factors": n_factors,
                "n_parameters": n_parameters,
                "residual_degrees_of_freedom": (
                    residual_degrees_of_freedom
                ),
                "matrix_rank": matrix_rank,
                "condition_number": condition_number,
                "sum_squared_residuals": (
                    sum_squared_residuals
                ),
                "total_sum_squares": total_sum_squares,
                "r_squared": r_squared,
                "adjusted_r_squared": adjusted_r_squared,
                "residual_variance": residual_variance,
                "residual_standard_error": (
                    residual_standard_error
                ),
                "residual_mean": residual_mean,
                "reconstruction_error": (
                    reconstruction_error
                ),
                "max_normal_equation_error": (
                    max_normal_equation_error
                ),
                "fit_status": fit_status,
            }
        ]
    )

    # --------------------------------------------------------
    # K. Construct long-format coefficient DataFrame
    # --------------------------------------------------------

    coefficient_terms = [
        "intercept",
        *factor_columns,
    ]

    coefficient_units = [
        "daily_decimal",
        *(
            ["dimensionless"]
            * n_factors
        ),
    ]

    coefficients_df = pd.DataFrame(
        {
            "model_id": model_id,
            "estimation_date": estimation_date,
            "term": coefficient_terms,
            "coefficient": coefficients,
            "units": coefficient_units,
        }
    )

    # --------------------------------------------------------
    # L. Construct daily fitted-return/residual DataFrame
    # --------------------------------------------------------

    residuals_df = pd.DataFrame(
        {
            "model_id": model_id,
            "estimation_date": estimation_date,
            "date": sample[
                date_column
            ].to_numpy(),
            "actual_excess_return": y,
            "fitted_excess_return": fitted_returns,
            "residual_excess_return": residual_returns,
        }
    )

    return (
        fit_summary_df,
        coefficients_df,
        residuals_df,
    )
