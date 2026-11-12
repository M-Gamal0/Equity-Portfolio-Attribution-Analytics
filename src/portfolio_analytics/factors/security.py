"""Factors security: validated research calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd

SUCCESSFUL_FACTOR_FIT_STATUSES = ('FIT_COMPLETED', 'FIT_COMPLETED_UNDEFINED_R2')

# Research reference: 11_fama_french_factor_analytics.ipynb, code cell index 8.
def estimate_rolling_security_ols(
    security_observations_df: pd.DataFrame,
    factor_reference_df: pd.DataFrame,
    security_eligibility_df: pd.DataFrame,
    *,
    security_id,
    factor_columns: tuple[str, ...],
    model_id: str,
    max_condition_number: float = 1e10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Estimate complete-history rolling security regressions.

    Parameters
    ----------
    security_observations_df:
        Aligned security observations from Cell 02.

        Required:
            security_id, date, excess_return,
            model_input_status.

    factor_reference_df:
        Daily factor observations from the approved source.

        Required:
            date and selected factor columns.

        This implementation requires complete numerical
        factor observations on the reference calendar.

    security_eligibility_df:
        Security eligibility results from Cell 03.

        Required:
            security_id, estimation_date, model_id,
            window_start_date, window_end_date,
            required_observations, eligibility_status.

    security_id:
        Security to estimate.

    factor_columns:
        Ordered regression factor names.

    model_id:
        Regression model specification identifier.

    max_condition_number:
        Maximum permitted design-matrix condition number.

    Returns
    -------
    rolling_fit_df:
        One row per eligible estimation window.

        Key:
            security_id, estimation_date, model_id.

        Includes sample size, fit statistics, numerical
        diagnostics, and fit status.

    rolling_coefficients_df:
        One row per estimated coefficient and window.

        Key:
            security_id, estimation_date, model_id, term.

    Methodology
    -----------
    For each eligible window:

        beta_hat = inverse(X'X) X'y

    The actual implementation solves the cross-product
    system directly without explicitly inverting X'X.

    Cross-products are calculated using cumulative sums,
    allowing rolling-window statistics to be obtained
    through two indexed lookups.

    Full-history availability is independently checked
    before any regression is estimated.

    Numerical failures are recorded, not zero-filled.

    No future return observation enters a prior estimate.
    """

    # --------------------------------------------------------
    # A. Validate model specification
    # --------------------------------------------------------

    if not factor_columns:
        raise ValueError(
            "At least one regression factor is required."
        )

    if len(set(factor_columns)) != len(factor_columns):
        raise ValueError(
            "Regression factor names must be unique."
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
            "Invalid regression condition-number threshold."
        )

    required_security_columns = {
        "security_id",
        "date",
        "excess_return",
        "model_input_status",
    }

    required_factor_columns = {
        "date",
        *factor_columns,
    }

    required_eligibility_columns = {
        "security_id",
        "estimation_date",
        "model_id",
        "window_start_date",
        "window_end_date",
        "required_observations",
        "eligibility_status",
    }

    for name, frame, required in (
        (
            "security observations",
            security_observations_df,
            required_security_columns,
        ),
        (
            "factor reference",
            factor_reference_df,
            required_factor_columns,
        ),
        (
            "security eligibility",
            security_eligibility_df,
            required_eligibility_columns,
        ),
    ):

        missing = required - set(frame.columns)

        if missing:
            raise ValueError(
                f"{name} missing columns: {sorted(missing)}"
            )

    # --------------------------------------------------------
    # B. Select the requested security and eligible windows
    # --------------------------------------------------------

    security_data = (
        security_observations_df.loc[
            security_observations_df[
                "security_id"
            ].eq(security_id)
        ]
        .copy()
    )

    eligible_windows = (
        security_eligibility_df.loc[
            security_eligibility_df[
                "security_id"
            ].eq(security_id)
            & security_eligibility_df[
                "model_id"
            ].eq(model_id)
            & security_eligibility_df[
                "eligibility_status"
            ].eq("ELIGIBLE_FULL_HISTORY")
        ]
        .copy()
    )

    if security_data.empty:
        raise ValueError(
            "Requested security has no return observations."
        )

    if eligible_windows.empty:
        raise ValueError(
            "Requested security has no eligible "
            "full-history regression windows."
        )

    if security_data["date"].duplicated().any():
        raise ValueError(
            "Duplicate security observation dates."
        )

    if eligible_windows[
        "estimation_date"
    ].duplicated().any():
        raise ValueError(
            "Duplicate eligible estimation dates."
        )

    # --------------------------------------------------------
    # C. Prepare reference factor calendar
    # --------------------------------------------------------

    reference = (
        factor_reference_df[
            [
                "date",
                *factor_columns,
            ]
        ]
        .copy()
    )

    reference["date"] = pd.to_datetime(
        reference["date"],
        errors="raise",
    )

    if (
        reference["date"].isna().any()
        or reference["date"].duplicated().any()
    ):
        raise ValueError(
            "Factor reference dates are missing or duplicated."
        )

    reference = (
        reference
        .sort_values("date")
        .reset_index(drop=True)
    )

    factor_values = reference[
        list(factor_columns)
    ].to_numpy(
        dtype=np.float64,
        na_value=np.nan,
    )

    if not np.isfinite(factor_values).all():
        raise ValueError(
            "This rolling engine requires complete "
            "finite reference factor observations."
        )

    factor_dates = pd.DatetimeIndex(
        reference["date"]
    )

    reference_dates_np = factor_dates.to_numpy(
        dtype="datetime64[ns]"
    )

    n_reference_dates = len(reference)

    n_parameters = len(factor_columns) + 1

    X = np.column_stack(
        [
            np.ones(n_reference_dates),
            factor_values,
        ]
    )

    # --------------------------------------------------------
    # D. Align security excess returns to factor dates
    # --------------------------------------------------------

    security_data["date"] = pd.to_datetime(
        security_data["date"],
        errors="raise",
    )

    eligible_input_data = (
        security_data.loc[
            security_data[
                "model_input_status"
            ].eq("ELIGIBLE"),
            [
                "date",
                "excess_return",
            ],
        ]
        .copy()
    )

    if not np.isfinite(
        eligible_input_data[
            "excess_return"
        ].to_numpy(
            dtype=np.float64,
            na_value=np.nan,
        )
    ).all():
        raise ValueError(
            "Eligible security excess returns "
            "contain non-finite observations."
        )

    factor_positions = factor_dates.get_indexer(
        eligible_input_data["date"]
    )

    if (factor_positions < 0).any():
        raise ValueError(
            "Eligible security observations "
            "are missing from the factor calendar."
        )

    # Missing observations have neutral accumulator values.
    # They are NEVER treated as actual zero returns.
    # A rolling window may be estimated only when
    # its eligible count equals the full required count.

    aligned_y = np.zeros(
        n_reference_dates,
        dtype=np.float64,
    )

    availability = np.zeros(
        n_reference_dates,
        dtype=np.int64,
    )

    aligned_y[factor_positions] = (
        eligible_input_data[
            "excess_return"
        ].to_numpy(
            dtype=np.float64
        )
    )

    availability[factor_positions] = 1

    # --------------------------------------------------------
    # E. Resolve strictly historical window boundaries
    # --------------------------------------------------------

    for column in (
        "estimation_date",
        "window_start_date",
        "window_end_date",
    ):

        eligible_windows[column] = pd.to_datetime(
            eligible_windows[column],
            errors="raise",
        )

    eligible_windows = (
        eligible_windows
        .sort_values("estimation_date")
        .reset_index(drop=True)
    )

    estimation_dates = eligible_windows[
        "estimation_date"
    ].to_numpy(
        dtype="datetime64[ns]"
    )

    start_dates = eligible_windows[
        "window_start_date"
    ].to_numpy(
        dtype="datetime64[ns]"
    )

    end_dates = eligible_windows[
        "window_end_date"
    ].to_numpy(
        dtype="datetime64[ns]"
    )

    if not np.array_equal(
        estimation_dates,
        end_dates,
    ):
        raise ValueError(
            "Estimation dates and window end dates differ."
        )

    if (
        (start_dates < reference_dates_np[0]).any()
        or (end_dates > reference_dates_np[-1]).any()
        or (start_dates > end_dates).any()
    ):
        raise ValueError(
            "Eligible estimation windows exceed "
            "the available factor history."
        )

    left = np.searchsorted(
        reference_dates_np,
        start_dates,
        side="left",
    )

    right = np.searchsorted(
        reference_dates_np,
        end_dates,
        side="right",
    )

    required_counts = (
        right - left
    )

    if not np.array_equal(
        required_counts,
        eligible_windows[
            "required_observations"
        ].to_numpy(dtype=np.int64),
    ):
        raise ValueError(
            "Rolling window counts do not reconcile "
            "to Cell 03 eligibility."
        )

    # --------------------------------------------------------
    # F. Independently enforce complete security history
    # --------------------------------------------------------

    availability_prefix = np.concatenate(
        [
            np.array([0], dtype=np.int64),
            np.cumsum(availability),
        ]
    )

    actual_counts = (
        availability_prefix[right]
        - availability_prefix[left]
    )

    if not np.array_equal(
        actual_counts,
        required_counts,
    ):
        raise ValueError(
            "An eligible regression window contains "
            "missing security observations. "
            "Estimation stopped without imputation."
        )

    if (required_counts <= n_parameters).any():
        raise ValueError(
            "A rolling window has insufficient "
            "residual degrees of freedom."
        )

    # --------------------------------------------------------
    # G. Construct cumulative regression cross-products
    # --------------------------------------------------------

    # Cumulative X'X matrices.

    xx_rows = np.einsum(
        "ni,nj->nij",
        X,
        X,
    )

    xx_prefix = np.concatenate(
        [
            np.zeros(
                (1, n_parameters, n_parameters)
            ),
            np.cumsum(
                xx_rows,
                axis=0,
            ),
        ],
        axis=0,
    )

    # Cumulative X'y vectors.

    xy_prefix = np.concatenate(
        [
            np.zeros(
                (1, n_parameters)
            ),
            np.cumsum(
                X * aligned_y[:, None],
                axis=0,
            ),
        ],
        axis=0,
    )

    # Cumulative y'y scalars.

    yy_prefix = np.concatenate(
        [
            np.array([0.0]),
            np.cumsum(
                aligned_y ** 2
            ),
        ]
    )

    # --------------------------------------------------------
    # H. Extract cross-products for all eligible windows
    # --------------------------------------------------------

    gram_matrices = (
        xx_prefix[right]
        - xx_prefix[left]
    )

    gram_matrices = (
        0.5
        * (
            gram_matrices
            + np.swapaxes(
                gram_matrices,
                -1,
                -2,
            )
        )
    )

    xy_vectors = (
        xy_prefix[right]
        - xy_prefix[left]
    )

    yy_values = (
        yy_prefix[right]
        - yy_prefix[left]
    )

    # --------------------------------------------------------
    # I. Validate numerical conditioning
    # --------------------------------------------------------

    gram_eigenvalues = np.linalg.eigvalsh(
        gram_matrices
    )

    min_eigenvalues = gram_eigenvalues[:, 0]

    max_eigenvalues = gram_eigenvalues[:, -1]

    condition_numbers = np.full(
        len(eligible_windows),
        np.inf,
    )

    positive_definite = (
        (min_eigenvalues > 0)
        & np.isfinite(min_eigenvalues)
        & np.isfinite(max_eigenvalues)
    )

    condition_numbers[positive_definite] = np.sqrt(
        max_eigenvalues[positive_definite]
        / min_eigenvalues[positive_definite]
    )

    numerically_eligible = (
        positive_definite
        & np.isfinite(condition_numbers)
        & (
            condition_numbers
            <= max_condition_number
        )
    )

    # --------------------------------------------------------
    # J. Solve eligible regression systems in batch
    # --------------------------------------------------------

    n_windows = len(eligible_windows)

    beta_matrix = np.full(
        (n_windows, n_parameters),
        np.nan,
    )

    if numerically_eligible.any():

        try:

            beta_matrix[
                numerically_eligible
            ] = np.linalg.solve(
                gram_matrices[numerically_eligible],
                xy_vectors[numerically_eligible][..., None],
            )[..., 0]

        except np.linalg.LinAlgError as error:

            raise ValueError(
                "Numerical failure in batched rolling "
                "OLS solution."
            ) from error

    # --------------------------------------------------------
    # K. Calculate fit diagnostics
    # --------------------------------------------------------

    rss = np.full(
        n_windows,
        np.nan,
    )

    tss = np.full(
        n_windows,
        np.nan,
    )

    r_squared = np.full(
        n_windows,
        np.nan,
    )

    adjusted_r_squared = np.full(
        n_windows,
        np.nan,
    )

    residual_standard_error = np.full(
        n_windows,
        np.nan,
    )

    residual_df = (
        required_counts - n_parameters
    )

    fit_status = np.full(
        n_windows,
        "NUMERICAL_REGRESSION_REJECTED",
        dtype=object,
    )

    if numerically_eligible.any():

        fitted_cross_product = np.einsum(
            "ij,ij->i",
            beta_matrix[numerically_eligible],
            xy_vectors[numerically_eligible],
        )

        raw_rss = (
            yy_values[numerically_eligible]
            - fitted_cross_product
        )

        raw_tss = (
            yy_values[numerically_eligible]
            - (
                xy_vectors[
                    numerically_eligible,
                    0,
                ] ** 2
                / required_counts[
                    numerically_eligible
                ]
            )
        )

        # Reject materially negative sums of squares.
        # Only tiny accumulated floating-point errors
        # may be rounded to zero.

        if (
            raw_rss
            < (
                -1e-9
                * np.maximum(
                    1.0,
                    yy_values[numerically_eligible],
                )
            )
        ).any():
            raise ValueError(
                "Rolling regression produced materially "
                "negative residual sums of squares."
            )

        if (
            raw_tss
            < (
                -1e-9
                * np.maximum(
                    1.0,
                    yy_values[numerically_eligible],
                )
            )
        ).any():
            raise ValueError(
                "Rolling regression produced materially "
                "negative total sums of squares."
            )

        rss_values = np.maximum(
            raw_rss,
            0.0,
        )

        tss_values = np.maximum(
            raw_tss,
            0.0,
        )

        rss[numerically_eligible] = rss_values

        tss[numerically_eligible] = tss_values

        residual_standard_error[
            numerically_eligible
        ] = np.sqrt(
            rss_values
            / residual_df[
                numerically_eligible
            ]
        )

        valid_tss = (
            tss_values > 0
        )

        successful_positions = np.flatnonzero(
            numerically_eligible
        )

        r_squared[
            successful_positions[valid_tss]
        ] = (
            1.0
            - (
                rss_values[valid_tss]
                / tss_values[valid_tss]
            )
        )

        adjusted_r_squared[
            successful_positions[valid_tss]
        ] = (
            1.0
            - (
                1.0
                - r_squared[
                    successful_positions[valid_tss]
                ]
            )
            * (
                required_counts[
                    successful_positions[valid_tss]
                ] - 1
            )
            / residual_df[
                successful_positions[valid_tss]
            ]
        )

        fit_status[
            successful_positions
        ] = "FIT_COMPLETED"

        fit_status[
            successful_positions[~valid_tss]
        ] = "FIT_COMPLETED_UNDEFINED_R2"

    # --------------------------------------------------------
    # L. Construct rolling regression fit output
    # --------------------------------------------------------

    rolling_fit_df = pd.DataFrame(
        {
            "security_id": security_id,
            "model_id": model_id,
            "estimation_date": estimation_dates,
            "window_start_date": start_dates,
            "window_end_date": end_dates,
            "sample_start_date": (
                reference_dates_np[left]
            ),
            "n_observations": required_counts,
            "n_parameters": n_parameters,
            "residual_degrees_of_freedom": residual_df,
            "condition_number": condition_numbers,
            "sum_squared_residuals": rss,
            "total_sum_squares": tss,
            "r_squared": r_squared,
            "adjusted_r_squared": adjusted_r_squared,
            "residual_standard_error": (
                residual_standard_error
            ),
            "fit_status": fit_status,
        }
    )

    # --------------------------------------------------------
    # M. Construct long-format rolling coefficients
    # --------------------------------------------------------

    coefficient_terms = [
        "intercept",
        *factor_columns,
    ]

    coefficient_wide_df = pd.DataFrame(
        beta_matrix,
        columns=coefficient_terms,
    )

    coefficient_wide_df.insert(
        0,
        "security_id",
        security_id,
    )

    coefficient_wide_df.insert(
        1,
        "model_id",
        model_id,
    )

    coefficient_wide_df.insert(
        2,
        "estimation_date",
        estimation_dates,
    )

    rolling_coefficients_df = (
        coefficient_wide_df
        .melt(
            id_vars=[
                "security_id",
                "model_id",
                "estimation_date",
            ],
            value_vars=coefficient_terms,
            var_name="term",
            value_name="coefficient",
        )
        .sort_values(
            [
                "estimation_date",
                "term",
            ]
        )
        .reset_index(drop=True)
    )

    rolling_coefficients_df["units"] = np.where(
        rolling_coefficients_df[
            "term"
        ].eq("intercept"),
        "daily_decimal",
        "dimensionless",
    )

    # --------------------------------------------------------
    # N. Validate output integrity
    # --------------------------------------------------------

    if rolling_fit_df.duplicated(
        subset=[
            "security_id",
            "model_id",
            "estimation_date",
        ]
    ).any():
        raise ValueError(
            "Duplicate rolling regression fit keys."
        )

    if rolling_coefficients_df.duplicated(
        subset=[
            "security_id",
            "model_id",
            "estimation_date",
            "term",
        ]
    ).any():
        raise ValueError(
            "Duplicate rolling coefficient keys."
        )

    if len(rolling_fit_df) != n_windows:
        raise ValueError(
            "Rolling regression output lost "
            "eligible estimation windows."
        )

    if len(rolling_coefficients_df) != (
        n_windows * n_parameters
    ):
        raise ValueError(
            "Rolling coefficient output row count "
            "does not reconcile."
        )

    rejected_dates = set(
        rolling_fit_df.loc[
            rolling_fit_df[
                "fit_status"
            ].eq("NUMERICAL_REGRESSION_REJECTED"),
            "estimation_date",
        ]
    )

    if rejected_dates:

        rejected_coefficients = (
            rolling_coefficients_df.loc[
                rolling_coefficients_df[
                    "estimation_date"
                ].isin(rejected_dates),
                "coefficient",
            ]
        )

        if rejected_coefficients.notna().any():
            raise ValueError(
                "Numerically rejected regressions "
                "contain nonmissing coefficients."
            )

    return (
        rolling_fit_df,
        rolling_coefficients_df,
    )


# Research reference: 11_fama_french_factor_analytics.ipynb, code cell index 9.
def estimate_full_universe_security_factors(
    aligned_observations_df: pd.DataFrame,
    factor_reference_df: pd.DataFrame,
    eligibility_df: pd.DataFrame,
    *,
    factor_columns: tuple[str, ...],
    model_id: str,
    max_condition_number: float = 1e10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Estimate rolling factor regressions for all securities.

    Parameters
    ----------
    aligned_observations_df:
        Security returns aligned to factor observations.

    factor_reference_df:
        Complete daily factor reference dataset.

    eligibility_df:
        Three-year security history eligibility results.

    factor_columns:
        Ordered regression factor columns.

    model_id:
        Factor model specification.

    max_condition_number:
        Maximum permitted OLS design condition number.

    Returns
    -------
    security_factor_fit_df:
        One row per eligible security estimation window.

        Key:
            security_id, estimation_date, model_id.

    security_factor_coefficients_df:
        One row per coefficient per eligible window.

        Key:
            security_id, estimation_date, model_id, term.

    security_factor_coverage_df:
        One row per security and model.

        Key:
            security_id, model_id.

    Methodology
    -----------
    - Securities are processed independently.
    - The Cell 09 rolling OLS engine is reused.
    - Eligibility is inherited from Cell 03.
    - No ineligible observations are modeled.
    - Numerical failures remain explicit.
    - Unmodeled securities are retained in coverage.
    - No data are written to disk.
    """

    # --------------------------------------------------------
    # A. Validate security universes and model eligibility
    # --------------------------------------------------------

    observation_ids = set(
        aligned_observations_df[
            "security_id"
        ].dropna().unique()
    )

    eligibility_model_df = eligibility_df.loc[
        eligibility_df[
            "model_id"
        ].eq(model_id)
    ]

    eligibility_ids = set(
        eligibility_model_df[
            "security_id"
        ].dropna().unique()
    )

    if observation_ids != eligibility_ids:
        raise ValueError(
            "Security universes do not match between "
            "aligned observations and eligibility."
        )

    if not observation_ids:
        raise ValueError(
            "Full-universe security dataset is empty."
        )

    security_ids = sorted(
        observation_ids,
        key=lambda value: str(value),
    )

    # --------------------------------------------------------
    # B. Prepare grouped security inputs
    # --------------------------------------------------------

    observation_groups = (
        aligned_observations_df.groupby(
            "security_id",
            sort=False,
        )
    )

    eligibility_groups = (
        eligibility_model_df.groupby(
            "security_id",
            sort=False,
        )
    )

    fit_frames = []

    coefficient_frames = []

    coverage_records = []

    # --------------------------------------------------------
    # C. Estimate independent security regressions
    # --------------------------------------------------------

    for security_number, security_id in enumerate(
        security_ids,
        start=1,
    ):

        security_observations = (
            observation_groups.get_group(
                security_id
            )
        )

        security_eligibility = (
            eligibility_groups.get_group(
                security_id
            )
        )

        eligible_rows = (
            security_eligibility.loc[
                security_eligibility[
                    "eligibility_status"
                ].eq("ELIGIBLE_FULL_HISTORY")
            ]
        )

        eligible_count = len(eligible_rows)

        # ----------------------------------------------------
        # D. Preserve securities without eligible windows
        # ----------------------------------------------------

        if eligible_count == 0:

            coverage_records.append(
                {
                    "security_id": security_id,
                    "model_id": model_id,
                    "security_observations": (
                        len(security_observations)
                    ),
                    "eligible_windows": 0,
                    "successful_fits": 0,
                    "numerically_rejected_fits": 0,
                    "first_eligible_date": pd.NaT,
                    "last_eligible_date": pd.NaT,
                    "model_coverage_status": (
                        "NO_ELIGIBLE_THREE_YEAR_WINDOW"
                    ),
                }
            )

            continue

        # ----------------------------------------------------
        # E. Run the validated rolling regression engine
        # ----------------------------------------------------

        (
            security_fit,
            security_coefficients,
        ) = estimate_rolling_security_ols(
            security_observations,
            factor_reference_df,
            security_eligibility,
            security_id=security_id,
            factor_columns=factor_columns,
            model_id=model_id,
            max_condition_number=(
                max_condition_number
            ),
        )

        if len(security_fit) != eligible_count:
            raise ValueError(
                f"Security {security_id}: rolling output "
                "does not reconcile to eligibility."
            )

        expected_coefficient_count = (
            eligible_count
            * (len(factor_columns) + 1)
        )

        if len(security_coefficients) != (
            expected_coefficient_count
        ):
            raise ValueError(
                f"Security {security_id}: coefficient "
                "observation count does not reconcile."
            )

        # ----------------------------------------------------
        # F. Calculate security-level regression coverage
        # ----------------------------------------------------

        successful_mask = (
            security_fit[
                "fit_status"
            ].isin(
                SUCCESSFUL_FACTOR_FIT_STATUSES
            )
        )

        successful_count = int(
            successful_mask.sum()
        )

        rejected_count = (
            eligible_count - successful_count
        )

        unexpected_statuses = (
            set(
                security_fit[
                    "fit_status"
                ].unique()
            )
            - set(SUCCESSFUL_FACTOR_FIT_STATUSES)
            - {"NUMERICAL_REGRESSION_REJECTED"}
        )

        if unexpected_statuses:
            raise ValueError(
                f"Unexpected regression statuses for "
                f"security {security_id}: "
                f"{sorted(unexpected_statuses)}"
            )

        if successful_count == eligible_count:

            coverage_status = "FULLY_ESTIMATED"

        elif successful_count == 0:

            coverage_status = (
                "ALL_NUMERICALLY_REJECTED"
            )

        else:

            coverage_status = (
                "PARTIALLY_ESTIMATED"
            )

        coverage_records.append(
            {
                "security_id": security_id,
                "model_id": model_id,
                "security_observations": (
                    len(security_observations)
                ),
                "eligible_windows": eligible_count,
                "successful_fits": successful_count,
                "numerically_rejected_fits": (
                    rejected_count
                ),
                "first_eligible_date": (
                    eligible_rows[
                        "estimation_date"
                    ].min()
                ),
                "last_eligible_date": (
                    eligible_rows[
                        "estimation_date"
                    ].max()
                ),
                "model_coverage_status": (
                    coverage_status
                ),
            }
        )

        fit_frames.append(
            security_fit
        )

        coefficient_frames.append(
            security_coefficients
        )

        # ----------------------------------------------------
        # G. Display bounded execution progress
        # ----------------------------------------------------

        if (
            security_number % 30 == 0
            or security_number == len(security_ids)
        ):

            print(
                f"Processed "
                f"{security_number:,}/"
                f"{len(security_ids):,} securities"
            )

    # --------------------------------------------------------
    # H. Combine security-level regression outputs
    # --------------------------------------------------------

    if not fit_frames:
        raise ValueError(
            "No securities produced eligible "
            "factor regression windows."
        )

    full_fit_df = pd.concat(
        fit_frames,
        ignore_index=True,
    )

    full_coefficients_df = pd.concat(
        coefficient_frames,
        ignore_index=True,
    )

    full_coverage_df = pd.DataFrame(
        coverage_records
    )

    return (
        full_fit_df,
        full_coefficients_df,
        full_coverage_df,
    )
