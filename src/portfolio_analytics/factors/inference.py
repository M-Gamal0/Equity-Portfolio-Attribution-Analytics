"""Factors inference: validated research calculations."""

from __future__ import annotations

from scipy import stats
import numpy as np
import pandas as pd

OLS_INFERENCE_METHOD = 'OLS_CLASSICAL'

# Research reference: 11_fama_french_factor_analytics.ipynb, code cell index 4.
def calculate_classical_ols_inference(
    observations_df: pd.DataFrame,
    fit_summary_df: pd.DataFrame,
    coefficients_df: pd.DataFrame,
    residuals_df: pd.DataFrame,
    *,
    factor_columns: tuple[str, ...],
    response_column: str = "excess_return",
    date_column: str = "date",
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """
    Calculate classical OLS coefficient inference.

    Parameters
    ----------
    observations_df:
        Complete regression observations containing
        the response, factor columns, and dates.

    fit_summary_df:
        Single-regression summary returned by
        fit_ols_factor_regression().

    coefficients_df:
        Estimated intercept and factor coefficients
        returned by fit_ols_factor_regression().

    residuals_df:
        Observation-level fitted returns and residuals
        returned by fit_ols_factor_regression().

    factor_columns:
        Ordered independent regression factor names.

    response_column:
        Daily excess-return dependent variable.

    date_column:
        Daily return-interval end date.

    confidence_level:
        Confidence level strictly between zero and one.

    Returns
    -------
    pd.DataFrame
        One row per coefficient.

        Key:
            model_id,
            estimation_date,
            inference_method,
            term.

        Includes:
            coefficient,
            standard_error,
            t_statistic,
            p_value,
            confidence_interval_lower,
            confidence_interval_upper,
            confidence_level,
            residual_degrees_of_freedom,
            inference_status.

    Assumptions
    -----------
    Classical covariance:
        sigma_squared * inverse(X'X)

    Conventional OLS covariance and Student-t inference
    assume appropriately specified regression errors.

    This function does not establish the validity of
    those statistical assumptions.

    Raises
    ------
    ValueError:
        Invalid model specification, inconsistent
        regression outputs, singular design matrices,
        missing observations, or numerically
        unidentified coefficient uncertainty.

    TypeError:
        Non-numeric regression variables.
    """

    # --------------------------------------------------------
    # A. Validate function parameters
    # --------------------------------------------------------

    if not (
        np.isfinite(confidence_level)
        and 0.0 < confidence_level < 1.0
    ):
        raise ValueError(
            "confidence_level must be between zero and one."
        )

    if not factor_columns:
        raise ValueError(
            "At least one factor is required."
        )

    if len(set(factor_columns)) != len(factor_columns):
        raise ValueError(
            "Factor names must be unique."
        )

    if len(fit_summary_df) != 1:
        raise ValueError(
            "Expected exactly one regression summary row."
        )

    required_observation_columns = {
        date_column,
        response_column,
        *factor_columns,
    }

    required_fit_columns = {
        "model_id",
        "estimation_date",
        "n_observations",
        "n_parameters",
        "residual_degrees_of_freedom",
        "residual_standard_error",
    }

    required_coefficient_columns = {
        "model_id",
        "estimation_date",
        "term",
        "coefficient",
    }

    required_residual_columns = {
        "model_id",
        "estimation_date",
        "date",
        "actual_excess_return",
        "fitted_excess_return",
        "residual_excess_return",
    }

    validation_groups = (
        (
            "observations",
            observations_df,
            required_observation_columns,
        ),
        (
            "fit_summary",
            fit_summary_df,
            required_fit_columns,
        ),
        (
            "coefficients",
            coefficients_df,
            required_coefficient_columns,
        ),
        (
            "residuals",
            residuals_df,
            required_residual_columns,
        ),
    )

    for frame_name, frame, required in validation_groups:

        missing = required - set(frame.columns)

        if missing:
            raise ValueError(
                f"{frame_name} missing required columns: "
                f"{sorted(missing)}"
            )

    # --------------------------------------------------------
    # B. Prepare independent regression sample
    # --------------------------------------------------------

    sample = observations_df[
        [
            date_column,
            response_column,
            *factor_columns,
        ]
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
            "Regression sample dates must be timezone-naive."
        )

    if not sample[date_column].eq(
        sample[date_column].dt.normalize()
    ).all():
        raise ValueError(
            "Regression sample contains intraday timestamps."
        )

    if sample[date_column].duplicated().any():
        raise ValueError(
            "Duplicate regression sample dates detected."
        )

    sample = (
        sample
        .sort_values(date_column)
        .reset_index(drop=True)
    )

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
            "or non-finite values."
        )

    # --------------------------------------------------------
    # C. Construct design matrix and dependent variable
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

    n_parameters = len(factor_columns) + 1

    residual_df = (
        n_observations - n_parameters
    )

    if residual_df <= 0:
        raise ValueError(
            "Inference requires positive residual "
            "degrees of freedom."
        )

    X = np.column_stack(
        [
            np.ones(n_observations),
            factor_matrix,
        ]
    )

    if np.linalg.matrix_rank(X) != n_parameters:
        raise ValueError(
            "Regression matrix is rank-deficient."
        )

    # --------------------------------------------------------
    # D. Validate regression output identities
    # --------------------------------------------------------

    fit_row = fit_summary_df.iloc[0]

    model_id = fit_row["model_id"]

    estimation_date = pd.Timestamp(
        fit_row["estimation_date"]
    )

    if estimation_date != sample[date_column].max():
        raise ValueError(
            "Regression estimation date does not match "
            "the supplied sample."
        )

    if int(fit_row["n_observations"]) != n_observations:
        raise ValueError(
            "Stored regression observation count "
            "does not match the supplied sample."
        )

    if int(fit_row["n_parameters"]) != n_parameters:
        raise ValueError(
            "Stored regression parameter count "
            "does not match the factor specification."
        )

    if int(
        fit_row["residual_degrees_of_freedom"]
    ) != residual_df:
        raise ValueError(
            "Stored residual degrees of freedom "
            "do not match the supplied sample."
        )

    expected_terms = [
        "intercept",
        *factor_columns,
    ]

    if coefficients_df["term"].duplicated().any():
        raise ValueError(
            "Duplicate regression coefficient terms."
        )

    if (
        len(coefficients_df) != n_parameters
        or set(coefficients_df["term"]) != set(expected_terms)
    ):
        raise ValueError(
            "Regression coefficient terms do not match "
            "the specified model."
        )

    if not coefficients_df["model_id"].eq(
        model_id
    ).all():
        raise ValueError(
            "Coefficient model IDs are inconsistent."
        )

    if not pd.to_datetime(
        coefficients_df["estimation_date"],
        errors="raise",
    ).eq(estimation_date).all():
        raise ValueError(
            "Coefficient estimation dates are inconsistent."
        )

    ordered_coefficients = (
        coefficients_df
        .set_index("term")
        .loc[expected_terms]
    )

    coefficient_values = ordered_coefficients[
        "coefficient"
    ].to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(coefficient_values).all():
        raise ValueError(
            "Stored regression coefficients are non-finite."
        )

    # --------------------------------------------------------
    # E. Validate fitted observations and residuals
    # --------------------------------------------------------

    stored_residuals = residuals_df.copy()

    stored_residuals["date"] = pd.to_datetime(
        stored_residuals["date"],
        errors="raise",
    )

    if stored_residuals["date"].duplicated().any():
        raise ValueError(
            "Duplicate stored regression residual dates."
        )

    stored_residuals = (
        stored_residuals
        .sort_values("date")
        .reset_index(drop=True)
    )

    if len(stored_residuals) != n_observations:
        raise ValueError(
            "Stored residual observation count differs "
            "from the regression sample."
        )

    if not np.array_equal(
        sample[date_column].to_numpy(),
        stored_residuals["date"].to_numpy(),
    ):
        raise ValueError(
            "Regression and residual dates do not align."
        )

    if not stored_residuals["model_id"].eq(
        model_id
    ).all():
        raise ValueError(
            "Residual model IDs are inconsistent."
        )

    if not pd.to_datetime(
        stored_residuals["estimation_date"],
        errors="raise",
    ).eq(estimation_date).all():
        raise ValueError(
            "Residual estimation dates are inconsistent."
        )

    expected_fitted = (
        X @ coefficient_values
    )

    expected_residuals = (
        y - expected_fitted
    )

    residual_checks = (
        (
            "actual_excess_return",
            y,
        ),
        (
            "fitted_excess_return",
            expected_fitted,
        ),
        (
            "residual_excess_return",
            expected_residuals,
        ),
    )

    for column, expected in residual_checks:

        observed = stored_residuals[
            column
        ].to_numpy(
            dtype=np.float64
        )

        if not np.isfinite(observed).all():
            raise ValueError(
                f"Stored {column} contains non-finite values."
            )

        if not np.allclose(
            observed,
            expected,
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError(
                f"Stored {column} does not reconcile "
                "to the regression design."
            )

    # --------------------------------------------------------
    # F. Calculate residual variance
    # --------------------------------------------------------

    residual_norm = float(
        np.linalg.norm(expected_residuals)
    )

    response_scale = max(
        float(np.linalg.norm(y)),
        float(
            np.linalg.norm(X)
            * np.linalg.norm(coefficient_values)
        ),
    )

    numerical_zero_tolerance = (
        100.0
        * np.finfo(np.float64).eps
        * response_scale
    )

    if residual_norm <= numerical_zero_tolerance:
        raise ValueError(
            "Residual variation is numerically "
            "indistinguishable from zero; "
            "coefficient uncertainty is unavailable."
        )

    residual_sum_squares = float(
        expected_residuals @ expected_residuals
    )

    residual_variance = (
        residual_sum_squares / residual_df
    )

    if (
        not np.isfinite(residual_variance)
        or residual_variance <= 0
    ):
        raise ValueError(
            "Regression residual variance is invalid."
        )

    residual_standard_error = float(
        np.sqrt(residual_variance)
    )

    if not np.isclose(
        residual_standard_error,
        float(fit_row["residual_standard_error"]),
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError(
            "Residual standard error does not reconcile "
            "to the previously estimated regression."
        )

    # --------------------------------------------------------
    # G. Calculate OLS covariance using QR decomposition
    # --------------------------------------------------------

    _, R = np.linalg.qr(
        X,
        mode="reduced",
    )

    inverse_R = np.linalg.solve(
        R,
        np.eye(n_parameters),
    )

    # Because X = Q R and Q'Q = I:
    #
    # inverse(X'X) = inverse(R) @ inverse(R).T

    inverse_xtx = (
        inverse_R @ inverse_R.T
    )

    covariance_matrix = (
        residual_variance * inverse_xtx
    )

    covariance_diagonal = np.diag(
        covariance_matrix
    )

    if not np.isfinite(covariance_matrix).all():
        raise ValueError(
            "Coefficient covariance contains "
            "non-finite values."
        )

    if (covariance_diagonal <= 0).any():
        raise ValueError(
            "Coefficient covariance contains "
            "nonpositive diagonal elements."
        )

    standard_errors = np.sqrt(
        covariance_diagonal
    )

    # --------------------------------------------------------
    # H. Calculate coefficient t-statistics and p-values
    # --------------------------------------------------------

    t_statistics = (
        coefficient_values / standard_errors
    )

    p_values = (
        2.0
        * stats.t.sf(
            np.abs(t_statistics),
            df=residual_df,
        )
    )

    # --------------------------------------------------------
    # I. Calculate two-sided confidence intervals
    # --------------------------------------------------------

    significance_level = (
        1.0 - confidence_level
    )

    critical_value = float(
        stats.t.ppf(
            1.0 - significance_level / 2.0,
            df=residual_df,
        )
    )

    confidence_lower = (
        coefficient_values
        - critical_value * standard_errors
    )

    confidence_upper = (
        coefficient_values
        + critical_value * standard_errors
    )

    # --------------------------------------------------------
    # J. Validate inference outputs
    # --------------------------------------------------------

    inference_arrays = (
        standard_errors,
        t_statistics,
        p_values,
        confidence_lower,
        confidence_upper,
    )

    for values in inference_arrays:

        if not np.isfinite(values).all():
            raise ValueError(
                "Coefficient inference produced "
                "non-finite results."
            )

    if (
        (p_values < 0).any()
        or (p_values > 1).any()
    ):
        raise ValueError(
            "Invalid coefficient p-values."
        )

    if not np.all(
        (confidence_lower <= coefficient_values)
        & (coefficient_values <= confidence_upper)
    ):
        raise ValueError(
            "Coefficient confidence intervals "
            "do not contain the estimates."
        )

    # --------------------------------------------------------
    # K. Preserve explicitly defined coefficient units
    # --------------------------------------------------------

    if "units" in ordered_coefficients.columns:

        coefficient_units = (
            ordered_coefficients["units"]
            .to_numpy()
        )

    else:

        coefficient_units = np.array(
            [
                "daily_decimal",
                *(
                    ["dimensionless"]
                    * len(factor_columns)
                ),
            ],
            dtype=object,
        )

    # --------------------------------------------------------
    # L. Construct standardized inference DataFrame
    # --------------------------------------------------------

    inference_df = pd.DataFrame(
        {
            "model_id": model_id,
            "estimation_date": estimation_date,
            "inference_method": OLS_INFERENCE_METHOD,
            "term": expected_terms,
            "coefficient": coefficient_values,
            "standard_error": standard_errors,
            "t_statistic": t_statistics,
            "p_value": p_values,
            "confidence_interval_lower": confidence_lower,
            "confidence_interval_upper": confidence_upper,
            "confidence_level": confidence_level,
            "residual_degrees_of_freedom": residual_df,
            "n_observations": n_observations,
            "units": coefficient_units,
            "inference_status": (
                "ESTIMATED_CLASSICAL_ASSUMPTIONS_UNTESTED"
            ),
        }
    )

    inference_key = [
        "model_id",
        "estimation_date",
        "inference_method",
        "term",
    ]

    if inference_df.duplicated(
        subset=inference_key
    ).any():
        raise ValueError(
            "Duplicate coefficient inference keys."
        )

    return inference_df


# Research reference: 11_fama_french_factor_analytics.ipynb, code cell index 5.
def calculate_newey_west_inference(
    observations_df: pd.DataFrame,
    coefficients_df: pd.DataFrame,
    residuals_df: pd.DataFrame,
    *,
    factor_columns: tuple[str, ...],
    model_id: str,
    response_column: str = "excess_return",
    date_column: str = "date",
    max_lags: int | None = None,
    small_sample_correction: bool = True,
    confidence_level: float = 0.95,
    max_condition_number: float = 1e10,
) -> pd.DataFrame:
    """
    Calculate Newey-West HAC inference for one OLS model.

    Required observations:
        Daily excess returns, factor returns, and dates.

    Required coefficients:
        model_id, estimation_date, term, coefficient.

    Required residuals:
        model_id, estimation_date, date,
        actual_excess_return, fitted_excess_return,
        residual_excess_return.

    Returns:
        One DataFrame row per regression coefficient.

    Units:
        Intercept: daily decimal return.
        Factor coefficients: dimensionless.
        Standard errors: same units as their coefficients.

    Conventions:
        OLS point estimates remain unchanged.
        HAC accounts for heteroskedasticity and
        autocorrelation up to the selected lag length.

        Bartlett weights:
            w_lag = 1 - lag / (L + 1)

        Automatic lag rule:
            floor(4 * (n / 100) ** (2 / 9))

        Optional covariance correction:
            n / (n - p)

        Inference uses asymptotic normal statistics.

    The input observations are sorted chronologically.
    No observations are dropped or imputed.
    """

    # --------------------------------------------------------
    # A. Validate configuration
    # --------------------------------------------------------

    if not factor_columns:
        raise ValueError(
            "At least one regression factor is required."
        )

    if len(set(factor_columns)) != len(factor_columns):
        raise ValueError(
            "Regression factor names must be unique."
        )

    if (
        response_column in factor_columns
        or date_column in factor_columns
    ):
        raise ValueError(
            "Response/date columns cannot be regressors."
        )

    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError(
            "model_id must be a nonempty string."
        )

    if (
        not np.isfinite(confidence_level)
        or not 0 < confidence_level < 1
    ):
        raise ValueError(
            "confidence_level must be strictly "
            "between zero and one."
        )

    if not isinstance(small_sample_correction, bool):
        raise TypeError(
            "small_sample_correction must be Boolean."
        )

    if (
        not np.isfinite(max_condition_number)
        or max_condition_number <= 1
    ):
        raise ValueError(
            "max_condition_number must be finite "
            "and greater than one."
        )

    # --------------------------------------------------------
    # B. Validate required DataFrame columns
    # --------------------------------------------------------

    required_observation_columns = {
        date_column,
        response_column,
        *factor_columns,
    }

    required_coefficient_columns = {
        "model_id",
        "estimation_date",
        "term",
        "coefficient",
    }

    required_residual_columns = {
        "model_id",
        "estimation_date",
        "date",
        "actual_excess_return",
        "fitted_excess_return",
        "residual_excess_return",
    }

    validation_groups = (
        (
            "observations",
            observations_df,
            required_observation_columns,
        ),
        (
            "coefficients",
            coefficients_df,
            required_coefficient_columns,
        ),
        (
            "residuals",
            residuals_df,
            required_residual_columns,
        ),
    )

    for name, frame, required in validation_groups:

        missing = required - set(frame.columns)

        if missing:
            raise ValueError(
                f"{name} missing required columns: "
                f"{sorted(missing)}"
            )

    # --------------------------------------------------------
    # C. Prepare chronological regression observations
    # --------------------------------------------------------

    sample = observations_df[
        [
            date_column,
            response_column,
            *factor_columns,
        ]
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
            "Regression dates contain intraday timestamps."
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
                f"{column} must be numeric and non-Boolean."
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
            "or non-finite numeric values."
        )

    # --------------------------------------------------------
    # D. Construct regression design matrix
    # --------------------------------------------------------

    y = sample[
        response_column
    ].to_numpy(dtype=np.float64)

    factor_matrix = sample[
        list(factor_columns)
    ].to_numpy(dtype=np.float64)

    n_observations = len(y)

    n_parameters = len(factor_columns) + 1

    residual_df = n_observations - n_parameters

    if residual_df <= 0:
        raise ValueError(
            "HAC inference requires positive residual "
            "degrees of freedom."
        )

    X = np.column_stack(
        [
            np.ones(n_observations),
            factor_matrix,
        ]
    )

    if np.linalg.matrix_rank(X) != n_parameters:
        raise ValueError(
            "Regression design matrix is rank-deficient."
        )

    condition_number = float(
        np.linalg.cond(X)
    )

    if (
        not np.isfinite(condition_number)
        or condition_number > max_condition_number
    ):
        raise ValueError(
            "Regression design matrix fails "
            "the conditioning requirement."
        )

    # --------------------------------------------------------
    # E. Resolve HAC lag length
    # --------------------------------------------------------

    if max_lags is None:

        effective_lags = int(
            np.floor(
                4.0
                * (n_observations / 100.0) ** (2.0 / 9.0)
            )
        )

        effective_lags = min(
            effective_lags,
            n_observations - 1,
        )

        lag_selection_method = (
            "NW_SAMPLE_SIZE_RULE"
        )

    else:

        if (
            not isinstance(max_lags, (int, np.integer))
            or isinstance(max_lags, (bool, np.bool_))
        ):
            raise TypeError(
                "max_lags must be an integer or None."
            )

        effective_lags = int(max_lags)

        if not 0 <= effective_lags < n_observations:
            raise ValueError(
                "max_lags must satisfy "
                "0 <= max_lags < n_observations."
            )

        lag_selection_method = "EXPLICIT"

    # --------------------------------------------------------
    # F. Validate supplied regression coefficients
    # --------------------------------------------------------

    expected_terms = [
        "intercept",
        *factor_columns,
    ]

    if coefficients_df["term"].duplicated().any():
        raise ValueError(
            "Duplicate coefficient terms detected."
        )

    if (
        len(coefficients_df) != n_parameters
        or set(coefficients_df["term"]) != set(expected_terms)
    ):
        raise ValueError(
            "Coefficient terms do not match "
            "the regression specification."
        )

    if not coefficients_df["model_id"].eq(
        model_id
    ).all():
        raise ValueError(
            "Coefficient model IDs are inconsistent."
        )

    estimation_date = sample[date_column].max()

    coefficient_dates = pd.to_datetime(
        coefficients_df["estimation_date"],
        errors="raise",
    )

    if not coefficient_dates.eq(
        estimation_date
    ).all():
        raise ValueError(
            "Coefficient estimation dates are inconsistent."
        )

    ordered_coefficients = (
        coefficients_df
        .set_index("term")
        .loc[expected_terms]
    )

    beta = ordered_coefficients[
        "coefficient"
    ].to_numpy(dtype=np.float64)

    if not np.isfinite(beta).all():
        raise ValueError(
            "Regression coefficients must be finite."
        )

    # Verify that the coefficients are the OLS estimates
    # of the supplied regression sample.

    reference_beta, _, reference_rank, _ = (
        np.linalg.lstsq(
            X,
            y,
            rcond=None,
        )
    )

    if reference_rank != n_parameters:
        raise ValueError(
            "OLS reference estimation is rank-deficient."
        )

    if not np.allclose(
        beta,
        reference_beta,
        rtol=1e-9,
        atol=1e-12,
    ):
        raise ValueError(
            "Supplied coefficients do not reconcile "
            "to the regression sample."
        )

    # --------------------------------------------------------
    # G. Validate stored fitted returns and residuals
    # --------------------------------------------------------

    stored_residuals = residuals_df.copy()

    stored_residuals["date"] = pd.to_datetime(
        stored_residuals["date"],
        errors="raise",
    )

    if (
        stored_residuals["date"].isna().any()
        or stored_residuals["date"].duplicated().any()
    ):
        raise ValueError(
            "Stored residual dates are missing "
            "or duplicated."
        )

    stored_residuals = (
        stored_residuals
        .sort_values("date")
        .reset_index(drop=True)
    )

    if len(stored_residuals) != n_observations:
        raise ValueError(
            "Stored residual observation count "
            "does not match the regression sample."
        )

    if not np.array_equal(
        sample[date_column].to_numpy(),
        stored_residuals["date"].to_numpy(),
    ):
        raise ValueError(
            "Stored residual dates do not match "
            "the regression sample."
        )

    if not stored_residuals["model_id"].eq(
        model_id
    ).all():
        raise ValueError(
            "Stored residual model IDs are inconsistent."
        )

    stored_estimation_dates = pd.to_datetime(
        stored_residuals["estimation_date"],
        errors="raise",
    )

    if not stored_estimation_dates.eq(
        estimation_date
    ).all():
        raise ValueError(
            "Stored residual estimation dates "
            "are inconsistent."
        )

    fitted = X @ beta

    residuals = y - fitted

    residual_checks = (
        ("actual_excess_return", y),
        ("fitted_excess_return", fitted),
        ("residual_excess_return", residuals),
    )

    for column, expected_values in residual_checks:

        observed_values = stored_residuals[
            column
        ].to_numpy(
            dtype=np.float64
        )

        if not np.isfinite(observed_values).all():
            raise ValueError(
                f"{column} contains non-finite values."
            )

        if not np.allclose(
            observed_values,
            expected_values,
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError(
                f"Stored {column} does not reconcile "
                "to the regression sample."
            )

    # --------------------------------------------------------
    # H. Reject numerically unidentified residual variation
    # --------------------------------------------------------

    residual_norm = float(
        np.linalg.norm(residuals)
    )

    response_scale = max(
        float(np.linalg.norm(y)),
        float(
            np.linalg.norm(X)
            * np.linalg.norm(beta)
        ),
    )

    zero_tolerance = (
        100.0
        * np.finfo(np.float64).eps
        * response_scale
    )

    if residual_norm <= zero_tolerance:
        raise ValueError(
            "Residual variation is numerically "
            "indistinguishable from zero; "
            "robust coefficient uncertainty is unavailable."
        )

    # --------------------------------------------------------
    # I. Calculate inverse cross-product using QR
    # --------------------------------------------------------

    _, R = np.linalg.qr(
        X,
        mode="reduced",
    )

    inverse_R = np.linalg.solve(
        R,
        np.eye(n_parameters),
    )

    inverse_xtx = (
        inverse_R @ inverse_R.T
    )

    # --------------------------------------------------------
    # J. Construct Newey-West Bartlett covariance
    # --------------------------------------------------------

    # Each row is the regression score:
    #
    #     score_t = x_t * residual_t

    scores = (
        X * residuals[:, None]
    )

    # Zero-lag score covariance.

    covariance_meat = (
        scores.T @ scores
    )

    # Add positive- and negative-lag autocovariances
    # using Bartlett-kernel weights.

    for lag in range(1, effective_lags + 1):

        lag_cross_product = (
            scores[lag:].T
            @ scores[:-lag]
        )

        bartlett_weight = (
            1.0
            - lag / (effective_lags + 1.0)
        )

        covariance_meat += (
            bartlett_weight
            * (
                lag_cross_product
                + lag_cross_product.T
            )
        )

    covariance_matrix = (
        inverse_xtx
        @ covariance_meat
        @ inverse_xtx
    )

    correction_factor = 1.0

    if small_sample_correction:

        correction_factor = (
            n_observations / residual_df
        )

        covariance_matrix *= correction_factor

    # Explicitly remove negligible floating-point
    # asymmetry in the estimated covariance matrix.

    covariance_matrix = (
        0.5
        * (
            covariance_matrix
            + covariance_matrix.T
        )
    )

    if not np.isfinite(covariance_matrix).all():
        raise ValueError(
            "HAC covariance contains non-finite values."
        )

    covariance_diagonal = np.diag(
        covariance_matrix
    )

    if (covariance_diagonal <= 0).any():
        raise ValueError(
            "HAC covariance contains nonpositive "
            "coefficient variances."
        )

    # --------------------------------------------------------
    # K. Calculate robust coefficient inference
    # --------------------------------------------------------

    standard_errors = np.sqrt(
        covariance_diagonal
    )

    z_statistics = (
        beta / standard_errors
    )

    p_values = (
        2.0
        * stats.norm.sf(
            np.abs(z_statistics)
        )
    )

    significance_level = (
        1.0 - confidence_level
    )

    critical_value = float(
        stats.norm.ppf(
            1.0 - significance_level / 2.0
        )
    )

    confidence_lower = (
        beta - critical_value * standard_errors
    )

    confidence_upper = (
        beta + critical_value * standard_errors
    )

    # --------------------------------------------------------
    # L. Validate numerical inference
    # --------------------------------------------------------

    inference_arrays = (
        standard_errors,
        z_statistics,
        p_values,
        confidence_lower,
        confidence_upper,
    )

    for values in inference_arrays:

        if not np.isfinite(values).all():
            raise ValueError(
                "HAC inference produced non-finite values."
            )

    if (
        (p_values < 0).any()
        or (p_values > 1).any()
    ):
        raise ValueError(
            "HAC coefficient p-values are invalid."
        )

    if not np.all(
        (confidence_lower <= beta)
        & (beta <= confidence_upper)
    ):
        raise ValueError(
            "HAC confidence intervals do not contain "
            "the estimated coefficients."
        )

    # --------------------------------------------------------
    # M. Preserve coefficient units
    # --------------------------------------------------------

    if "units" in ordered_coefficients.columns:

        coefficient_units = (
            ordered_coefficients["units"]
            .to_numpy()
        )

    else:

        coefficient_units = np.array(
            [
                "daily_decimal",
                *(
                    ["dimensionless"]
                    * len(factor_columns)
                ),
            ],
            dtype=object,
        )

    # --------------------------------------------------------
    # N. Construct standardized inference output
    # --------------------------------------------------------

    inference_df = pd.DataFrame(
        {
            "model_id": model_id,
            "estimation_date": estimation_date,
            "inference_method": "OLS_NEWEY_WEST_HAC",
            "term": expected_terms,
            "coefficient": beta,
            "standard_error": standard_errors,
            "z_statistic": z_statistics,
            "p_value": p_values,
            "confidence_interval_lower": confidence_lower,
            "confidence_interval_upper": confidence_upper,
            "confidence_level": confidence_level,
            "n_observations": n_observations,
            "n_parameters": n_parameters,
            "residual_degrees_of_freedom": residual_df,
            "hac_max_lags": effective_lags,
            "hac_lag_selection_method": lag_selection_method,
            "hac_kernel": "bartlett",
            "small_sample_correction": (
                small_sample_correction
            ),
            "covariance_correction_factor": (
                correction_factor
            ),
            "statistic_distribution": (
                "asymptotic_normal"
            ),
            "units": coefficient_units,
            "inference_status": (
                "ESTIMATED_HAC_ASYMPTOTIC"
            ),
        }
    )

    inference_key = [
        "model_id",
        "estimation_date",
        "inference_method",
        "term",
    ]

    if inference_df.duplicated(
        subset=inference_key
    ).any():
        raise ValueError(
            "Duplicate HAC inference keys detected."
        )

    return inference_df


# Research reference: 11_fama_french_factor_analytics.ipynb, code cell index 6.
def calculate_ols_residual_diagnostics(
    observations_df: pd.DataFrame,
    residuals_df: pd.DataFrame,
    *,
    factor_columns: tuple[str, ...],
    response_column: str = "excess_return",
    date_column: str = "date",
    ljung_box_lags: tuple[int, ...] = (1, 5, 10, 20),
    min_normality_observations: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate residual diagnostics for one OLS regression.

    Parameters
    ----------
    observations_df:
        Regression observations containing dates,
        daily excess returns and factor regressors.

    residuals_df:
        Regression residual output containing:
            model_id,
            estimation_date,
            date,
            actual_excess_return,
            fitted_excess_return,
            residual_excess_return.

    factor_columns:
        Independent factor names used in the regression.

    response_column:
        Daily excess-return dependent variable.

    date_column:
        Daily return-interval end date.

    ljung_box_lags:
        Positive integer lags for serial-correlation tests.

    min_normality_observations:
        Minimum observations required before reporting
        asymptotic Jarque-Bera inference.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]

        residual_diagnostics_df:
            One row per regression.

            Key:
                model_id, estimation_date.

        residual_ljung_box_df:
            One row per requested diagnostic lag.

            Key:
                model_id, estimation_date, lag.

    Conventions
    -----------
    - Residuals are ordered chronologically.
    - Durbin-Watson uses original OLS residuals.
    - Ljung-Box uses mean-centered residuals.
    - Koenker-BP uses squared OLS residuals.
    - Jarque-Bera uses sample residual moments.
    - Unassessable statistics receive explicit statuses.
    - No observations are dropped or imputed.

    Notes
    -----
    The p-values are asymptotic diagnostics.

    Their interpretation depends on the underlying
    regression specification and sampling assumptions.

    Diagnostic significance does not establish
    causality, model stability or forecasting accuracy.
    """

    # --------------------------------------------------------
    # A. Validate parameters
    # --------------------------------------------------------

    if not factor_columns:
        raise ValueError(
            "At least one regression factor is required."
        )

    if len(set(factor_columns)) != len(factor_columns):
        raise ValueError(
            "Regression factor names must be unique."
        )

    if (
        response_column in factor_columns
        or date_column in factor_columns
    ):
        raise ValueError(
            "Response and date columns cannot be factors."
        )

    if not ljung_box_lags:
        raise ValueError(
            "At least one Ljung-Box lag is required."
        )

    for lag in ljung_box_lags:

        if (
            not isinstance(lag, (int, np.integer))
            or isinstance(lag, (bool, np.bool_))
            or lag < 1
        ):
            raise ValueError(
                "Ljung-Box lags must be positive integers."
            )

    if len(set(ljung_box_lags)) != len(ljung_box_lags):
        raise ValueError(
            "Duplicate Ljung-Box lags are not permitted."
        )

    if (
        not isinstance(min_normality_observations, int)
        or isinstance(min_normality_observations, bool)
        or min_normality_observations < 3
    ):
        raise ValueError(
            "min_normality_observations must be "
            "an integer of at least three."
        )

    # --------------------------------------------------------
    # B. Validate required input columns
    # --------------------------------------------------------

    required_observation_columns = {
        date_column,
        response_column,
        *factor_columns,
    }

    required_residual_columns = {
        "model_id",
        "estimation_date",
        "date",
        "actual_excess_return",
        "fitted_excess_return",
        "residual_excess_return",
    }

    missing_observation_columns = (
        required_observation_columns
        - set(observations_df.columns)
    )

    missing_residual_columns = (
        required_residual_columns
        - set(residuals_df.columns)
    )

    if missing_observation_columns:
        raise ValueError(
            "Regression observations missing columns: "
            f"{sorted(missing_observation_columns)}"
        )

    if missing_residual_columns:
        raise ValueError(
            "Regression residuals missing columns: "
            f"{sorted(missing_residual_columns)}"
        )

    # --------------------------------------------------------
    # C. Prepare independent working copies
    # --------------------------------------------------------

    observations = observations_df[
        [
            date_column,
            response_column,
            *factor_columns,
        ]
    ].copy()

    residual_data = residuals_df[
        [
            "model_id",
            "estimation_date",
            "date",
            "actual_excess_return",
            "fitted_excess_return",
            "residual_excess_return",
        ]
    ].copy()

    observations[date_column] = pd.to_datetime(
        observations[date_column],
        errors="raise",
    )

    for column in ("date", "estimation_date"):

        residual_data[column] = pd.to_datetime(
            residual_data[column],
            errors="raise",
        )

    for frame_name, frame, date_field in (
        ("observations", observations, date_column),
        ("residuals", residual_data, "date"),
    ):

        if frame[date_field].isna().any():
            raise ValueError(
                f"{frame_name} contain missing dates."
            )

        if frame[date_field].dt.tz is not None:
            raise ValueError(
                f"{frame_name} dates must be timezone-naive."
            )

        if not frame[date_field].eq(
            frame[date_field].dt.normalize()
        ).all():
            raise ValueError(
                f"{frame_name} contain intraday timestamps."
            )

        if frame[date_field].duplicated().any():
            raise ValueError(
                f"Duplicate {frame_name} dates detected."
            )

    if residual_data["estimation_date"].isna().any():
        raise ValueError(
            "Residual estimation dates contain nulls."
        )

    if residual_data["model_id"].isna().any():
        raise ValueError(
            "Residual model IDs contain nulls."
        )

    if residual_data["model_id"].nunique() != 1:
        raise ValueError(
            "Expected residuals from exactly one model."
        )

    if residual_data["estimation_date"].nunique() != 1:
        raise ValueError(
            "Expected one regression estimation date."
        )

    observations = (
        observations
        .sort_values(date_column)
        .reset_index(drop=True)
    )

    residual_data = (
        residual_data
        .sort_values("date")
        .reset_index(drop=True)
    )

    n_observations = len(observations)

    n_parameters = len(factor_columns) + 1

    if n_observations <= n_parameters:
        raise ValueError(
            "Insufficient observations for "
            "regression residual diagnostics."
        )

    if len(residual_data) != n_observations:
        raise ValueError(
            "Regression and residual observation "
            "counts do not match."
        )

    if not np.array_equal(
        observations[date_column].to_numpy(),
        residual_data["date"].to_numpy(),
    ):
        raise ValueError(
            "Regression and residual dates do not align."
        )

    model_id = residual_data["model_id"].iloc[0]

    estimation_date = (
        residual_data["estimation_date"].iloc[0]
    )

    if estimation_date != observations[date_column].max():
        raise ValueError(
            "Residual estimation date does not match "
            "the supplied regression sample."
        )

    # --------------------------------------------------------
    # D. Validate numerical observations
    # --------------------------------------------------------

    numeric_observation_columns = [
        response_column,
        *factor_columns,
    ]

    numeric_residual_columns = [
        "actual_excess_return",
        "fitted_excess_return",
        "residual_excess_return",
    ]

    for frame, columns in (
        (observations, numeric_observation_columns),
        (residual_data, numeric_residual_columns),
    ):

        for column in columns:

            if (
                not pd.api.types.is_numeric_dtype(
                    frame[column]
                )
                or pd.api.types.is_bool_dtype(
                    frame[column]
                )
            ):
                raise TypeError(
                    f"{column} must be numeric and non-Boolean."
                )

        values = frame[columns].to_numpy(
            dtype=np.float64,
            na_value=np.nan,
        )

        if not np.isfinite(values).all():
            raise ValueError(
                "Residual diagnostics require finite "
                "numeric observations."
            )

    # --------------------------------------------------------
    # E. Validate residual accounting
    # --------------------------------------------------------

    y = observations[
        response_column
    ].to_numpy(dtype=np.float64)

    actual_y = residual_data[
        "actual_excess_return"
    ].to_numpy(dtype=np.float64)

    fitted_y = residual_data[
        "fitted_excess_return"
    ].to_numpy(dtype=np.float64)

    residuals = residual_data[
        "residual_excess_return"
    ].to_numpy(dtype=np.float64)

    if not np.allclose(
        actual_y,
        y,
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError(
            "Stored actual returns do not reconcile "
            "to regression observations."
        )

    if not np.allclose(
        fitted_y + residuals,
        actual_y,
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError(
            "Stored residual decomposition does not "
            "reconcile to actual returns."
        )

    residual_sum_squares = float(
        residuals @ residuals
    )

    if (
        not np.isfinite(residual_sum_squares)
        or residual_sum_squares <= 0
    ):
        raise ValueError(
            "Residual diagnostics require positive "
            "finite residual variation."
        )

    # --------------------------------------------------------
    # F. Calculate Durbin-Watson statistic
    # --------------------------------------------------------

    residual_differences = np.diff(
        residuals
    )

    durbin_watson = float(
        (
            residual_differences
            @ residual_differences
        )
        / residual_sum_squares
    )

    # --------------------------------------------------------
    # G. Calculate residual autocorrelations
    # --------------------------------------------------------

    centered_residuals = (
        residuals - residuals.mean()
    )

    centered_sum_squares = float(
        centered_residuals
        @ centered_residuals
    )

    autocorrelation_available = (
        np.isfinite(centered_sum_squares)
        and centered_sum_squares > 0
    )

    valid_max_lag = min(
        max(ljung_box_lags),
        n_observations - 1,
    )

    autocorrelation_by_lag = {}

    if autocorrelation_available:

        for lag in range(1, valid_max_lag + 1):

            autocorrelation_by_lag[lag] = float(
                (
                    centered_residuals[lag:]
                    @ centered_residuals[:-lag]
                )
                / centered_sum_squares
            )

    # --------------------------------------------------------
    # H. Calculate Ljung-Box statistics
    # --------------------------------------------------------

    ljung_box_records = []

    for lag in sorted(ljung_box_lags):

        if lag >= n_observations:

            ljung_box_records.append(
                {
                    "model_id": model_id,
                    "estimation_date": estimation_date,
                    "lag": lag,
                    "autocorrelation_at_lag": np.nan,
                    "ljung_box_statistic": np.nan,
                    "ljung_box_p_value": np.nan,
                    "diagnostic_status": (
                        "NOT_ASSESSED_INSUFFICIENT_HISTORY"
                    ),
                }
            )

            continue

        if not autocorrelation_available:

            ljung_box_records.append(
                {
                    "model_id": model_id,
                    "estimation_date": estimation_date,
                    "lag": lag,
                    "autocorrelation_at_lag": np.nan,
                    "ljung_box_statistic": np.nan,
                    "ljung_box_p_value": np.nan,
                    "diagnostic_status": (
                        "NOT_ASSESSED_ZERO_CENTERED_VARIANCE"
                    ),
                }
            )

            continue

        ljung_box_sum = sum(
            autocorrelation_by_lag[k] ** 2
            / (n_observations - k)
            for k in range(1, lag + 1)
        )

        ljung_box_statistic = float(
            n_observations
            * (n_observations + 2)
            * ljung_box_sum
        )

        ljung_box_p_value = float(
            stats.chi2.sf(
                ljung_box_statistic,
                df=lag,
            )
        )

        ljung_box_records.append(
            {
                "model_id": model_id,
                "estimation_date": estimation_date,
                "lag": lag,
                "autocorrelation_at_lag": (
                    autocorrelation_by_lag[lag]
                ),
                "ljung_box_statistic": (
                    ljung_box_statistic
                ),
                "ljung_box_p_value": (
                    ljung_box_p_value
                ),
                "diagnostic_status": "ASSESSED",
            }
        )

    residual_ljung_box_df = pd.DataFrame(
        ljung_box_records
    )

    # --------------------------------------------------------
    # I. Calculate Koenker-Breusch-Pagan LM statistic
    # --------------------------------------------------------

    factor_matrix = observations[
        list(factor_columns)
    ].to_numpy(
        dtype=np.float64
    )

    X = np.column_stack(
        [
            np.ones(n_observations),
            factor_matrix,
        ]
    )

    if np.linalg.matrix_rank(X) != n_parameters:
        raise ValueError(
            "Heteroskedasticity diagnostic design "
            "matrix is rank-deficient."
        )

    squared_residuals = (
        residuals ** 2
    )

    if not np.isfinite(squared_residuals).all():
        raise ValueError(
            "Squared residuals contain non-finite values."
        )

    auxiliary_mean = float(
        squared_residuals.mean()
    )

    auxiliary_centered = (
        squared_residuals - auxiliary_mean
    )

    auxiliary_total_ss = float(
        auxiliary_centered @ auxiliary_centered
    )

    if auxiliary_total_ss == 0:

        bp_r_squared = np.nan

        bp_lm_statistic = np.nan

        bp_p_value = np.nan

        bp_status = (
            "NOT_ASSESSED_CONSTANT_SQUARED_RESIDUAL"
        )

    else:

        auxiliary_beta, _, auxiliary_rank, _ = (
            np.linalg.lstsq(
                X,
                squared_residuals,
                rcond=None,
            )
        )

        if auxiliary_rank != n_parameters:
            raise ValueError(
                "Auxiliary heteroskedasticity regression "
                "is rank-deficient."
            )

        auxiliary_fitted = (
            X @ auxiliary_beta
        )

        auxiliary_explained_ss = float(
            np.sum(
                (
                    auxiliary_fitted
                    - auxiliary_mean
                ) ** 2
            )
        )

        bp_r_squared = (
            auxiliary_explained_ss
            / auxiliary_total_ss
        )

        if (
            not np.isfinite(bp_r_squared)
            or bp_r_squared < 0
            or bp_r_squared > 1 + 1e-10
        ):
            raise ValueError(
                "Invalid auxiliary regression R-squared."
            )

        bp_lm_statistic = float(
            n_observations * bp_r_squared
        )

        bp_p_value = float(
            stats.chi2.sf(
                bp_lm_statistic,
                df=len(factor_columns),
            )
        )

        bp_status = "ASSESSED"

    # --------------------------------------------------------
    # J. Calculate Jarque-Bera normality diagnostic
    # --------------------------------------------------------

    if n_observations < min_normality_observations:

        jb_statistic = np.nan

        jb_p_value = np.nan

        jb_status = (
            "NOT_ASSESSED_SMALL_SAMPLE"
        )

    elif not autocorrelation_available:

        jb_statistic = np.nan

        jb_p_value = np.nan

        jb_status = (
            "NOT_ASSESSED_ZERO_CENTERED_VARIANCE"
        )

    else:

        jb_result = stats.jarque_bera(
            residuals
        )

        jb_statistic = float(
            jb_result.statistic
        )

        jb_p_value = float(
            jb_result.pvalue
        )

        if not np.isfinite(
            [jb_statistic, jb_p_value]
        ).all():
            raise ValueError(
                "Jarque-Bera inference is non-finite."
            )

        jb_status = "ASSESSED"

    # --------------------------------------------------------
    # K. Construct standardized diagnostic output
    # --------------------------------------------------------

    residual_diagnostics_df = pd.DataFrame(
        [
            {
                "model_id": model_id,
                "estimation_date": estimation_date,
                "sample_start_date": (
                    observations[date_column].min()
                ),
                "sample_end_date": estimation_date,
                "n_observations": n_observations,
                "n_factors": len(factor_columns),
                "durbin_watson": durbin_watson,
                "residual_autocorrelation_lag1": (
                    autocorrelation_by_lag.get(
                        1,
                        np.nan,
                    )
                ),
                "koenker_bp_r_squared": bp_r_squared,
                "koenker_bp_lm_statistic": (
                    bp_lm_statistic
                ),
                "koenker_bp_p_value": bp_p_value,
                "koenker_bp_status": bp_status,
                "jarque_bera_statistic": jb_statistic,
                "jarque_bera_p_value": jb_p_value,
                "jarque_bera_status": jb_status,
                "residual_mean": float(
                    residuals.mean()
                ),
                "residual_sum_squares": (
                    residual_sum_squares
                ),
                "diagnostic_method_version": "1.0",
            }
        ]
    )

    # --------------------------------------------------------
    # L. Validate output contracts
    # --------------------------------------------------------

    if residual_diagnostics_df.duplicated(
        subset=[
            "model_id",
            "estimation_date",
        ]
    ).any():
        raise ValueError(
            "Duplicate regression diagnostic keys."
        )

    if residual_ljung_box_df.duplicated(
        subset=[
            "model_id",
            "estimation_date",
            "lag",
        ]
    ).any():
        raise ValueError(
            "Duplicate Ljung-Box diagnostic keys."
        )

    if not np.isfinite(durbin_watson):
        raise ValueError(
            "Durbin-Watson statistic is non-finite."
        )

    return (
        residual_diagnostics_df,
        residual_ljung_box_df,
    )


# Research reference: 11_fama_french_factor_analytics.ipynb, code cell index 7.
def evaluate_hac_lag_sensitivity(
    observations_df: pd.DataFrame,
    coefficients_df: pd.DataFrame,
    residuals_df: pd.DataFrame,
    *,
    factor_columns: tuple[str, ...],
    model_id: str,
    lag_grid: tuple[int, ...],
    baseline_lags: int,
    confidence_level: float = 0.95,
    significance_level: float = 0.05,
    small_sample_correction: bool = True,
    max_condition_number: float = 1e10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate Newey-West coefficient inference sensitivity.

    Parameters
    ----------
    observations_df:
        Complete single-entity regression observations.

    coefficients_df:
        Previously estimated OLS coefficients.

    residuals_df:
        Previously estimated OLS residual observations.

    factor_columns:
        Ordered model factor names.

    model_id:
        Regression model identifier.

    lag_grid:
        Distinct, nonnegative HAC lag specifications.

    baseline_lags:
        Selected reference HAC lag length.
        Must be included in lag_grid.

    confidence_level:
        Two-sided confidence interval level.

    significance_level:
        Threshold for descriptive significance flags.

    small_sample_correction:
        Whether to use covariance correction n / (n-p).

    max_condition_number:
        Numerical conditioning threshold.

    Returns
    -------
    hac_sensitivity_df:
        One row per model, estimation date, lag, term.

        Key:
            model_id,
            estimation_date,
            hac_max_lags,
            term.

    hac_stability_df:
        One row per model, estimation date, term.

        Key:
            model_id,
            estimation_date,
            term.

    Methodology
    -----------
    Recalculate HAC covariance for each specified lag.

    Preserve OLS coefficient estimates exactly.

    Compare:
        standard errors,
        p-values,
        confidence intervals,
        significance classifications.

    All significance-stability conclusions apply only
    to the explicitly tested lag specifications.
    """

    # --------------------------------------------------------
    # A. Validate lag specifications
    # --------------------------------------------------------

    if not lag_grid:
        raise ValueError(
            "At least one HAC lag specification is required."
        )

    if len(set(lag_grid)) != len(lag_grid):
        raise ValueError(
            "Duplicate HAC sensitivity lags detected."
        )

    n_observations = len(observations_df)

    for lag in lag_grid:

        if (
            not isinstance(lag, (int, np.integer))
            or isinstance(lag, (bool, np.bool_))
        ):
            raise TypeError(
                "HAC lag specifications must be integers."
            )

        if not 0 <= lag < n_observations:
            raise ValueError(
                "Each HAC lag must satisfy "
                "0 <= lag < n_observations."
            )

    if baseline_lags not in lag_grid:
        raise ValueError(
            "Baseline HAC lag must be included "
            "in the sensitivity grid."
        )

    if not (
        np.isfinite(significance_level)
        and 0 < significance_level < 1
    ):
        raise ValueError(
            "significance_level must lie between "
            "zero and one."
        )

    # --------------------------------------------------------
    # B. Establish coefficient ordering
    # --------------------------------------------------------

    expected_terms = [
        "intercept",
        *factor_columns,
    ]

    n_terms = len(expected_terms)

    # --------------------------------------------------------
    # C. Estimate inference across all HAC lag choices
    # --------------------------------------------------------

    inference_frames = []

    for lag in lag_grid:

        lag_inference_df = (
            calculate_newey_west_inference(
                observations_df,
                coefficients_df,
                residuals_df,
                factor_columns=factor_columns,
                model_id=model_id,
                max_lags=lag,
                small_sample_correction=(
                    small_sample_correction
                ),
                confidence_level=confidence_level,
                max_condition_number=(
                    max_condition_number
                ),
            )
        )

        if len(lag_inference_df) != n_terms:
            raise ValueError(
                "HAC inference returned an unexpected "
                "coefficient count."
            )

        if lag_inference_df["term"].tolist() != (
            expected_terms
        ):
            raise ValueError(
                "HAC coefficient term ordering changed."
            )

        if not lag_inference_df[
            "hac_max_lags"
        ].eq(lag).all():
            raise ValueError(
                "HAC inference returned an incorrect "
                "lag specification."
            )

        lag_inference_df[
            "is_baseline_lag"
        ] = (
            lag == baseline_lags
        )

        lag_inference_df[
            "significant_at_5pct"
        ] = (
            lag_inference_df["p_value"]
            < significance_level
        )

        inference_frames.append(
            lag_inference_df
        )

    # --------------------------------------------------------
    # D. Construct long-format sensitivity output
    # --------------------------------------------------------

    hac_sensitivity_df = pd.concat(
        inference_frames,
        ignore_index=True,
    )

    sensitivity_key = [
        "model_id",
        "estimation_date",
        "hac_max_lags",
        "term",
    ]

    if hac_sensitivity_df.duplicated(
        subset=sensitivity_key
    ).any():
        raise ValueError(
            "Duplicate HAC sensitivity keys detected."
        )

    expected_rows = (
        len(lag_grid) * n_terms
    )

    if len(hac_sensitivity_df) != expected_rows:
        raise ValueError(
            "HAC sensitivity output row count "
            "does not reconcile."
        )

    # --------------------------------------------------------
    # E. Validate coefficient invariance
    # --------------------------------------------------------

    reference_coefficients = (
        inference_frames[0]["coefficient"]
        .to_numpy(dtype=np.float64)
    )

    for frame in inference_frames[1:]:

        np.testing.assert_allclose(
            frame["coefficient"].to_numpy(
                dtype=np.float64
            ),
            reference_coefficients,
            rtol=0,
            atol=1e-12,
            err_msg=(
                "OLS coefficient estimates changed "
                "across HAC lag specifications."
            ),
        )

    # --------------------------------------------------------
    # F. Extract reference-lag inference
    # --------------------------------------------------------

    baseline_df = (
        hac_sensitivity_df.loc[
            hac_sensitivity_df[
                "hac_max_lags"
            ].eq(baseline_lags),
            [
                "term",
                "standard_error",
                "p_value",
                "significant_at_5pct",
            ],
        ]
        .rename(
            columns={
                "standard_error": "baseline_standard_error",
                "p_value": "baseline_p_value",
                "significant_at_5pct": (
                    "baseline_significant_5pct"
                ),
            }
        )
    )

    if len(baseline_df) != n_terms:
        raise ValueError(
            "Baseline inference does not contain "
            "all regression terms."
        )

    # --------------------------------------------------------
    # G. Aggregate coefficient stability metrics
    # --------------------------------------------------------

    lag_stability_summary = (
        hac_sensitivity_df
        .groupby(
            "term",
            as_index=False,
        )
        .agg(
            min_standard_error=(
                "standard_error",
                "min",
            ),
            max_standard_error=(
                "standard_error",
                "max",
            ),
            min_p_value=(
                "p_value",
                "min",
            ),
            max_p_value=(
                "p_value",
                "max",
            ),
            significant_lag_count=(
                "significant_at_5pct",
                "sum",
            ),
            evaluated_lag_count=(
                "hac_max_lags",
                "nunique",
            ),
        )
    )

    lag_stability_summary = (
        lag_stability_summary
        .merge(
            baseline_df,
            on="term",
            how="left",
            validate="one_to_one",
        )
    )

    # --------------------------------------------------------
    # H. Calculate significance stability
    # --------------------------------------------------------

    lag_stability_summary[
        "significance_switch_count"
    ] = np.where(
        lag_stability_summary[
            "baseline_significant_5pct"
        ],
        lag_stability_summary[
            "evaluated_lag_count"
        ]
        - lag_stability_summary[
            "significant_lag_count"
        ],
        lag_stability_summary[
            "significant_lag_count"
        ],
    )

    lag_stability_summary[
        "all_lags_agree_with_baseline"
    ] = (
        lag_stability_summary[
            "significance_switch_count"
        ].eq(0)
    )

    lag_stability_summary[
        "significant_at_all_tested_lags"
    ] = (
        lag_stability_summary[
            "significant_lag_count"
        ]
        == lag_stability_summary[
            "evaluated_lag_count"
        ]
    )

    lag_stability_summary[
        "insignificant_at_all_tested_lags"
    ] = (
        lag_stability_summary[
            "significant_lag_count"
        ].eq(0)
    )

    lag_stability_summary[
        "standard_error_max_to_min"
    ] = (
        lag_stability_summary[
            "max_standard_error"
        ]
        / lag_stability_summary[
            "min_standard_error"
        ]
    )

    # --------------------------------------------------------
    # I. Construct standardized term-level output
    # --------------------------------------------------------

    reference_model = (
        hac_sensitivity_df[
            "model_id"
        ].iloc[0]
    )

    reference_date = (
        hac_sensitivity_df[
            "estimation_date"
        ].iloc[0]
    )

    lag_stability_summary.insert(
        0,
        "model_id",
        reference_model,
    )

    lag_stability_summary.insert(
        1,
        "estimation_date",
        reference_date,
    )

    lag_stability_summary.insert(
        2,
        "baseline_hac_lags",
        baseline_lags,
    )

    hac_stability_df = (
        lag_stability_summary
        .set_index("term")
        .loc[expected_terms]
        .reset_index()
    )

    stability_key = [
        "model_id",
        "estimation_date",
        "term",
    ]

    if hac_stability_df.duplicated(
        subset=stability_key
    ).any():
        raise ValueError(
            "Duplicate HAC stability keys detected."
        )

    if len(hac_stability_df) != n_terms:
        raise ValueError(
            "HAC stability output row count "
            "does not reconcile."
        )

    if not hac_stability_df[
        "evaluated_lag_count"
    ].eq(len(lag_grid)).all():
        raise ValueError(
            "Not all HAC lag specifications "
            "were evaluated for every coefficient."
        )

    if not np.isfinite(
        hac_stability_df[
            "standard_error_max_to_min"
        ].to_numpy(dtype=np.float64)
    ).all():
        raise ValueError(
            "HAC standard-error sensitivity "
            "contains non-finite ratios."
        )

    return (
        hac_sensitivity_df,
        hac_stability_df,
    )
