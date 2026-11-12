"""Factors inputs: validated research calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd



# Research reference: 11_fama_french_factor_analytics.ipynb, code cell index 1.
def align_factor_regression_observations(
    returns_df: pd.DataFrame,
    factors_df: pd.DataFrame,
    *,
    entity_columns: tuple[str, ...],
    factor_columns: tuple[str, ...],
    risk_free_column: str = "rf",
    return_column: str = "return_total",
) -> pd.DataFrame:
    """
    Align daily entity returns with contemporaneous factor data.

    Parameters
    ----------
    returns_df:
        Entity-level daily simple total returns.

        Required columns:
            entity_columns, date, return_column.

    factors_df:
        Daily factor and risk-free observations.

        Required columns:
            date, factor_columns, risk_free_column.

    entity_columns:
        Unique identifiers for each modeled entity.

    factor_columns:
        Selected regression factor names.

    risk_free_column:
        Daily simple-decimal risk-free return column.

    return_column:
        Daily simple-decimal total return column.

    Returns
    -------
    pd.DataFrame
        One row per original entity-date observation.

        Includes:
            entity identifiers,
            date,
            total return,
            factor returns,
            risk-free return,
            excess_return,
            model_input_status.

    Conventions
    -----------
    - Exact date alignment.
    - No forward filling or imputation.
    - No security-history truncation.
    - Excess returns are available only for complete
      regression observations.
    - Factor columns are configurable.
    - Inputs are not modified.

    Raises
    ------
    ValueError:
        Invalid keys, dates, duplicated observations,
        factor specifications, or infinite numeric values.

    TypeError:
        Non-numeric regression input columns.
    """

    # --------------------------------------------------------
    # A. Validate model specification
    # --------------------------------------------------------

    if not entity_columns:
        raise ValueError(
            "At least one entity identifier is required."
        )

    if not factor_columns:
        raise ValueError(
            "At least one regression factor is required."
        )

    if len(set(entity_columns)) != len(entity_columns):
        raise ValueError(
            "Duplicate entity identifiers specified."
        )

    if len(set(factor_columns)) != len(factor_columns):
        raise ValueError(
            "Duplicate factor names specified."
        )

    if risk_free_column in factor_columns:
        raise ValueError(
            "Risk-free return cannot also be an "
            "independent regression factor."
        )

    if "date" in entity_columns:
        raise ValueError(
            "Date must not be included in entity_columns."
        )

    # --------------------------------------------------------
    # B. Validate required columns
    # --------------------------------------------------------

    return_fields = [
        *entity_columns,
        "date",
        return_column,
    ]

    factor_fields = [
        "date",
        *factor_columns,
        risk_free_column,
    ]

    missing_return_fields = (
        set(return_fields) - set(returns_df.columns)
    )

    missing_factor_fields = (
        set(factor_fields) - set(factors_df.columns)
    )

    if missing_return_fields:
        raise ValueError(
            "Missing required return columns: "
            f"{sorted(missing_return_fields)}"
        )

    if missing_factor_fields:
        raise ValueError(
            "Missing required factor columns: "
            f"{sorted(missing_factor_fields)}"
        )

    # --------------------------------------------------------
    # C. Extract independent working copies
    # --------------------------------------------------------

    entity_returns = returns_df[
        return_fields
    ].copy()

    model_factors = factors_df[
        factor_fields
    ].copy()

    # --------------------------------------------------------
    # D. Validate and normalize date representations
    # --------------------------------------------------------

    for frame_name, frame in (
        ("returns", entity_returns),
        ("factors", model_factors),
    ):

        frame["date"] = pd.to_datetime(
            frame["date"],
            errors="raise",
        )

        if frame["date"].isna().any():
            raise ValueError(
                f"{frame_name} contains missing dates."
            )

        if frame["date"].dt.tz is not None:
            raise ValueError(
                f"{frame_name} dates must be timezone-naive."
            )

        if not frame["date"].eq(
            frame["date"].dt.normalize()
        ).all():
            raise ValueError(
                f"{frame_name} contains non-daily timestamps."
            )

    # --------------------------------------------------------
    # E. Validate entity identifiers and observation keys
    # --------------------------------------------------------

    if entity_returns[
        list(entity_columns)
    ].isna().any().any():
        raise ValueError(
            "Entity identifiers contain missing values."
        )

    entity_key = [
        *entity_columns,
        "date",
    ]

    if entity_returns.duplicated(
        subset=entity_key
    ).any():
        raise ValueError(
            "Duplicate entity-date observations detected."
        )

    if model_factors["date"].duplicated().any():
        raise ValueError(
            "Duplicate factor dates detected."
        )

    # --------------------------------------------------------
    # F. Validate regression input numeric types
    # --------------------------------------------------------

    numeric_groups = (
        (
            "returns",
            entity_returns,
            [return_column],
        ),
        (
            "factors",
            model_factors,
            [
                *factor_columns,
                risk_free_column,
            ],
        ),
    )

    for group_name, frame, columns in numeric_groups:

        for column in columns:

            if not pd.api.types.is_numeric_dtype(
                frame[column]
            ):
                raise TypeError(
                    f"{group_name}.{column} must be numeric."
                )

        numeric_values = frame[
            columns
        ].to_numpy(
            dtype=float,
            na_value=np.nan,
        )

        if np.isinf(numeric_values).any():
            raise ValueError(
                f"{group_name} contains infinite "
                "regression inputs."
            )

    # --------------------------------------------------------
    # G. Align contemporaneous observations
    # --------------------------------------------------------

    aligned_df = entity_returns.merge(
        model_factors,
        on="date",
        how="left",
        validate="many_to_one",
        indicator=True,
        sort=False,
    )

    if len(aligned_df) != len(entity_returns):
        raise ValueError(
            "Factor alignment changed the number "
            "of return observations."
        )

    # --------------------------------------------------------
    # H. Identify observation availability
    # --------------------------------------------------------

    missing_return = aligned_df[
        return_column
    ].isna()

    missing_factor_date = aligned_df[
        "_merge"
    ].eq("left_only")

    incomplete_factor = aligned_df[
        [
            *factor_columns,
            risk_free_column,
        ]
    ].isna().any(axis=1)

    aligned_df["model_input_status"] = np.select(
        [
            missing_return,
            missing_factor_date,
            incomplete_factor,
        ],
        [
            "MISSING_DEPENDENT_RETURN",
            "MISSING_FACTOR_DATE",
            "INCOMPLETE_FACTOR_OBSERVATION",
        ],
        default="ELIGIBLE",
    )

    # --------------------------------------------------------
    # I. Calculate eligible daily excess returns
    # --------------------------------------------------------

    aligned_df["excess_return"] = (
        aligned_df[return_column]
        - aligned_df[risk_free_column]
    )

    ineligible_mask = (
        aligned_df["model_input_status"]
        != "ELIGIBLE"
    )

    aligned_df.loc[
        ineligible_mask,
        "excess_return",
    ] = np.nan

    # --------------------------------------------------------
    # J. Remove merge diagnostics and restore stable ordering
    # --------------------------------------------------------

    aligned_df = aligned_df.drop(
        columns="_merge"
    )

    aligned_df = (
        aligned_df
        .sort_values(entity_key)
        .reset_index(drop=True)
    )

    return aligned_df


# Research reference: 11_fama_french_factor_analytics.ipynb, code cell index 2.
def evaluate_factor_history_eligibility(
    aligned_returns_df: pd.DataFrame,
    factors_df: pd.DataFrame,
    *,
    entity_column: str,
    factor_columns: tuple[str, ...],
    model_id: str,
    lookback_years: int,
    risk_free_column: str = "rf",
) -> pd.DataFrame:
    """
    Evaluate complete calendar-based regression history.

    Parameters
    ----------
    aligned_returns_df:
        Daily entity returns previously aligned to factors.

        Required columns:
            entity_column,
            date,
            model_input_status.

        model_input_status == "ELIGIBLE" indicates an
        individually complete regression observation.

    factors_df:
        Reference factor observations.

        Required columns:
            date,
            factor_columns,
            risk_free_column.

    entity_column:
        Stable identifier of the modeled entity.

    factor_columns:
        Factor names used by the regression model.

    model_id:
        Identifier of the factor model specification.

    lookback_years:
        Number of trailing calendar years required.

    risk_free_column:
        Daily risk-free return column.

    Returns
    -------
    pd.DataFrame
        One row per original entity-date observation.

        Key:
            entity_column, estimation_date, model_id.

        Principal fields:
            window_start_date,
            window_end_date,
            required_observations,
            available_observations,
            missing_observations,
            eligibility_status.

    Methodology
    -----------
    For estimation date t:

        window_start = t - lookback_years
        window_end = t

    Complete history requires:
        1. The reference factor source covers the window.
        2. Every reference factor date is complete.
        3. The entity has an eligible return on every
           reference factor date in the window.
        4. The estimation-date observation is eligible.

    The factor-source calendar is the reference set of
    expected observations. It is not an independently
    certified exchange calendar.

    No future observations are used to determine
    eligibility at an earlier estimation date.

    Notes
    -----
    This function assesses historical data completeness,
    not regression matrix rank or coefficient quality.

    Missing estimates must not be represented as
    zero-valued factor exposures.
    """

    # --------------------------------------------------------
    # A. Validate parameters and required fields
    # --------------------------------------------------------

    if (
        not isinstance(lookback_years, int)
        or isinstance(lookback_years, bool)
        or lookback_years < 1
    ):
        raise ValueError(
            "lookback_years must be a positive integer."
        )

    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError(
            "model_id must be a nonempty string."
        )

    if not factor_columns:
        raise ValueError(
            "At least one factor column is required."
        )

    if len(set(factor_columns)) != len(factor_columns):
        raise ValueError(
            "Factor names must be unique."
        )

    if risk_free_column in factor_columns:
        raise ValueError(
            "Risk-free return cannot also be a factor."
        )

    required_entity_fields = {
        entity_column,
        "date",
        "model_input_status",
    }

    required_factor_fields = {
        "date",
        *factor_columns,
        risk_free_column,
    }

    missing_entity_fields = (
        required_entity_fields
        - set(aligned_returns_df.columns)
    )

    missing_factor_fields = (
        required_factor_fields
        - set(factors_df.columns)
    )

    if missing_entity_fields:
        raise ValueError(
            "Aligned returns missing required fields: "
            f"{sorted(missing_entity_fields)}"
        )

    if missing_factor_fields:
        raise ValueError(
            "Factor source missing required fields: "
            f"{sorted(missing_factor_fields)}"
        )

    # --------------------------------------------------------
    # B. Prepare independent working DataFrames
    # --------------------------------------------------------

    observations = aligned_returns_df[
        [
            entity_column,
            "date",
            "model_input_status",
        ]
    ].copy()

    reference_factors = factors_df[
        [
            "date",
            *factor_columns,
            risk_free_column,
        ]
    ].copy()

    for frame in (
        observations,
        reference_factors,
    ):

        frame["date"] = pd.to_datetime(
            frame["date"],
            errors="raise",
        )

        if frame["date"].isna().any():
            raise ValueError(
                "Regression history contains missing dates."
            )

        if frame["date"].dt.tz is not None:
            raise ValueError(
                "Regression history requires timezone-naive "
                "daily dates."
            )

        if not frame["date"].eq(
            frame["date"].dt.normalize()
        ).all():
            raise ValueError(
                "Regression history contains non-daily "
                "timestamps."
            )

    if observations[entity_column].isna().any():
        raise ValueError(
            "Entity identifiers contain missing values."
        )

    if observations["model_input_status"].isna().any():
        raise ValueError(
            "Model input statuses contain missing values."
        )

    if observations.duplicated(
        subset=[entity_column, "date"]
    ).any():
        raise ValueError(
            "Duplicate entity-date observations detected."
        )

    if reference_factors["date"].duplicated().any():
        raise ValueError(
            "Duplicate reference factor dates detected."
        )

    if reference_factors.empty:
        raise ValueError(
            "Reference factor history is empty."
        )

    # --------------------------------------------------------
    # C. Validate reference factor numerical observations
    # --------------------------------------------------------

    factor_value_columns = [
        *factor_columns,
        risk_free_column,
    ]

    for column in factor_value_columns:

        if not pd.api.types.is_numeric_dtype(
            reference_factors[column]
        ):
            raise TypeError(
                f"Factor column {column} must be numeric."
            )

    factor_values = reference_factors[
        factor_value_columns
    ].to_numpy(
        dtype=float,
        na_value=np.nan,
    )

    if np.isinf(factor_values).any():
        raise ValueError(
            "Reference factors contain infinite values."
        )

    # An incomplete source factor date remains part of
    # the required calendar, but makes its window ineligible.

    reference_factors = (
        reference_factors
        .sort_values("date")
        .reset_index(drop=True)
    )

    complete_reference_mask = (
        reference_factors[factor_value_columns]
        .notna()
        .all(axis=1)
        .to_numpy(dtype=bool)
    )

    reference_dates = (
        reference_factors["date"]
        .to_numpy(dtype="datetime64[ns]")
    )

    # --------------------------------------------------------
    # D. Establish individual observation eligibility
    # --------------------------------------------------------

    observations = (
        observations
        .sort_values([entity_column, "date"])
        .reset_index(drop=True)
    )

    eligible_input_mask = (
        observations["model_input_status"]
        .eq("ELIGIBLE")
    )

    # Eligible input rows must correspond to actual
    # reference factor dates.

    eligible_dates = observations.loc[
        eligible_input_mask,
        "date",
    ]

    if not eligible_dates.isin(
        reference_factors["date"]
    ).all():
        raise ValueError(
            "An eligible input observation has no "
            "corresponding reference factor date."
        )

    if not observations.loc[
        eligible_input_mask,
        "date",
    ].empty:

        eligible_reference_check = (
            observations.loc[
                eligible_input_mask,
                ["date"],
            ]
            .merge(
                reference_factors[
                    ["date"]
                ].assign(
                    factor_complete=complete_reference_mask
                ),
                on="date",
                how="left",
                validate="many_to_one",
            )
        )

        if not eligible_reference_check[
            "factor_complete"
        ].all():
            raise ValueError(
                "An ELIGIBLE input observation has "
                "incomplete reference factors."
            )

    # --------------------------------------------------------
    # E. Calculate calendar-based estimation boundaries
    # --------------------------------------------------------

    estimation_dates = observations[
        "date"
    ]

    window_start_dates = (
        estimation_dates
        - pd.DateOffset(years=lookback_years)
    )

    estimation_dates_np = estimation_dates.to_numpy(
        dtype="datetime64[ns]"
    )

    window_start_np = window_start_dates.to_numpy(
        dtype="datetime64[ns]"
    )

    # --------------------------------------------------------
    # F. Count required reference-calendar observations
    # --------------------------------------------------------

    # The left endpoint is inclusive.
    # The estimation date is also inclusive.

    left_positions = np.searchsorted(
        reference_dates,
        window_start_np,
        side="left",
    )

    right_positions = np.searchsorted(
        reference_dates,
        estimation_dates_np,
        side="right",
    )

    required_observations = (
        right_positions - left_positions
    )

    # Prefix counts allow complete-window factor checks
    # without repeatedly slicing the three-year history.

    complete_factor_prefix = np.concatenate(
        (
            np.array([0], dtype=np.int64),
            np.cumsum(
                complete_reference_mask.astype(np.int64)
            ),
        )
    )

    complete_factor_observations = (
        complete_factor_prefix[right_positions]
        - complete_factor_prefix[left_positions]
    )

    # --------------------------------------------------------
    # G. Count eligible security observations per window
    # --------------------------------------------------------

    available_observations = np.zeros(
        len(observations),
        dtype=np.int64,
    )

    for _, entity_group in observations.groupby(
        entity_column,
        sort=False,
    ):

        group_positions = (
            entity_group.index.to_numpy()
        )

        valid_entity_dates = (
            entity_group.loc[
                entity_group[
                    "model_input_status"
                ].eq("ELIGIBLE"),
                "date",
            ]
            .to_numpy(dtype="datetime64[ns]")
        )

        available_left = np.searchsorted(
            valid_entity_dates,
            window_start_np[group_positions],
            side="left",
        )

        available_right = np.searchsorted(
            valid_entity_dates,
            estimation_dates_np[group_positions],
            side="right",
        )

        available_observations[group_positions] = (
            available_right - available_left
        )

    missing_observations = (
        required_observations
        - available_observations
    )

    if (missing_observations < 0).any():
        raise ValueError(
            "Entity observation counts exceed the "
            "required reference-calendar counts."
        )

    # --------------------------------------------------------
    # H. Classify three-year regression eligibility
    # --------------------------------------------------------

    insufficient_factor_warmup = (
        window_start_np < reference_dates[0]
    )

    incomplete_factor_window = (
        complete_factor_observations
        < required_observations
    )

    incomplete_security_history = (
        missing_observations > 0
    )

    eligibility_status = np.select(
        [
            insufficient_factor_warmup,
            incomplete_factor_window,
            incomplete_security_history,
        ],
        [
            "INSUFFICIENT_FACTOR_WARMUP",
            "INCOMPLETE_FACTOR_WINDOW",
            "INCOMPLETE_SECURITY_HISTORY",
        ],
        default="ELIGIBLE_FULL_HISTORY",
    ).astype(object)

    # The estimation-date observation must also be
    # eligible. Preserve its original failure reason.

    invalid_current_observation = (
        ~eligible_input_mask.to_numpy()
    )

    eligibility_status[
        invalid_current_observation
    ] = observations.loc[
        invalid_current_observation,
        "model_input_status",
    ].to_numpy()

    # --------------------------------------------------------
    # I. Construct standardized eligibility output
    # --------------------------------------------------------

    result_df = pd.DataFrame(
        {
            entity_column: observations[
                entity_column
            ].to_numpy(),
            "estimation_date": (
                estimation_dates_np
            ),
            "model_id": model_id,
            "window_start_date": (
                window_start_np
            ),
            "window_end_date": (
                estimation_dates_np
            ),
            "lookback_calendar_years": (
                lookback_years
            ),
            "required_observations": (
                required_observations
            ),
            "complete_factor_observations": (
                complete_factor_observations
            ),
            "available_observations": (
                available_observations
            ),
            "missing_observations": (
                missing_observations
            ),
            "eligibility_status": (
                eligibility_status
            ),
        }
    )

    # --------------------------------------------------------
    # J. Validate result integrity
    # --------------------------------------------------------

    result_key = [
        entity_column,
        "estimation_date",
        "model_id",
    ]

    if result_df.duplicated(
        subset=result_key
    ).any():
        raise ValueError(
            "Duplicate regression eligibility keys detected."
        )

    if len(result_df) != len(aligned_returns_df):
        raise ValueError(
            "Regression eligibility changed the "
            "number of original entity observations."
        )

    eligible_results = result_df.loc[
        result_df["eligibility_status"].eq(
            "ELIGIBLE_FULL_HISTORY"
        )
    ]

    if not eligible_results.empty:

        if not eligible_results[
            "missing_observations"
        ].eq(0).all():
            raise ValueError(
                "Eligible regression windows contain "
                "missing security observations."
            )

        if not (
            eligible_results[
                "complete_factor_observations"
            ]
            == eligible_results[
                "required_observations"
            ]
        ).all():
            raise ValueError(
                "Eligible regression windows contain "
                "incomplete factor observations."
            )

        expected_start = (
            eligible_results[
                "estimation_date"
            ]
            - pd.DateOffset(years=lookback_years)
        )

        if not expected_start.ge(
            reference_factors["date"].min()
        ).all():
            raise ValueError(
                "Eligible windows do not have complete "
                "factor-source warm-up coverage."
            )

    return result_df
