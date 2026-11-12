"""Factors attribution: validated research calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd



# Research reference: 11_fama_french_factor_analytics.ipynb, code cell index 12.
def calculate_holding_factor_return_decomposition(
    lagged_holdings_df: pd.DataFrame,
    realized_holdings_df: pd.DataFrame,
    factor_reference_df: pd.DataFrame,
    *,
    factor_columns: tuple[str, ...],
    model_id: str,
    tolerance: float = 1e-12,
) -> pd.DataFrame:
    """
    Calculate security-level daily factor return attribution.

    Input grain:
        portfolio_id, date, security_id

    Output grain:
        portfolio_id, date, security_id, model_id

    Model:
        Security excess return is decomposed using
        strictly prior estimated OLS coefficients
        and contemporaneous realized factor returns.

    Unmodeled securities:
        Preserve realized excess contribution
        without assigning estimated factor effects.

    Missing factor dates:
        Preserve actual holding returns and contributions.
        Leave all factor-dependent attribution fields NaN.

    The resulting residual is an ex-post model residual,
    not a direct estimate of investment skill.
    """

    # --------------------------------------------------------
    # A. Validate configuration and required columns
    # --------------------------------------------------------

    if (
        not factor_columns
        or len(set(factor_columns)) != len(factor_columns)
    ):
        raise ValueError(
            "Factor names must be nonempty and distinct."
        )

    if not isinstance(model_id, str) or not model_id:
        raise ValueError(
            "Invalid model ID."
        )

    if (
        not np.isfinite(tolerance)
        or tolerance < 0
    ):
        raise ValueError(
            "Invalid reconciliation tolerance."
        )

    keys = [
        "portfolio_id",
        "date",
        "security_id",
    ]

    beta_columns = [
        f"beta_{factor}"
        for factor in factor_columns
    ]

    lagged_required = {
        *keys,
        "model_id",
        "weight_bop",
        "model_available",
        "estimation_date",
        "factor_available_on_return_date",
        "estimated_alpha_daily",
        *beta_columns,
    }

    realized_required = {
        *keys,
        "weight_bop",
        "return_total",
    }

    factor_required = {
        "date",
        "rf",
        *factor_columns,
    }

    for name, frame, required in (
        (
            "lagged holdings",
            lagged_holdings_df,
            lagged_required,
        ),
        (
            "realized holdings",
            realized_holdings_df,
            realized_required,
        ),
        (
            "factor source",
            factor_reference_df,
            factor_required,
        ),
    ):

        missing = required - set(frame.columns)

        if missing:
            raise ValueError(
                f"{name} missing columns: {sorted(missing)}"
            )

    # --------------------------------------------------------
    # B. Prepare independent working copies
    # --------------------------------------------------------

    lagged = lagged_holdings_df[
        list(lagged_required)
    ].copy()

    source_columns = [
        *keys,
        "weight_bop",
        "return_total",
    ]

    if "contribution_total" in realized_holdings_df.columns:
        source_columns.append("contribution_total")

    realized = realized_holdings_df[
        source_columns
    ].copy()

    factor_data = factor_reference_df[
        [
            "date",
            "rf",
            *factor_columns,
        ]
    ].copy()

    for frame in (
        lagged,
        realized,
        factor_data,
    ):

        frame["date"] = pd.to_datetime(
            frame["date"],
            errors="raise",
        ).astype("datetime64[ns]")

        if (
            frame["date"].isna().any()
            or not frame["date"].eq(
                frame["date"].dt.normalize()
            ).all()
        ):
            raise ValueError(
                "Dates must be complete and normalized."
            )

    lagged["estimation_date"] = pd.to_datetime(
        lagged["estimation_date"],
        errors="raise",
    ).astype("datetime64[ns]")

    # --------------------------------------------------------
    # C. Validate identifiers and uniqueness
    # --------------------------------------------------------

    for frame in (
        lagged,
        realized,
    ):

        if frame[
            [
                "portfolio_id",
                "security_id",
            ]
        ].isna().any().any():
            raise ValueError(
                "Missing holding identifiers."
            )

        frame["portfolio_id"] = (
            frame["portfolio_id"].astype(str)
        )

        frame["security_id"] = (
            frame["security_id"].astype(str)
        )

        if frame.duplicated(
            subset=keys
        ).any():
            raise ValueError(
                "Duplicate holding return keys."
            )

    if factor_data[
        "date"
    ].duplicated().any():
        raise ValueError(
            "Duplicate factor-source dates."
        )

    if not lagged[
        "model_id"
    ].eq(model_id).all():
        raise ValueError(
            "Unexpected factor model identifier."
        )

    # --------------------------------------------------------
    # D. Validate numerical inputs
    # --------------------------------------------------------

    for column in (
        "weight_bop",
        "estimated_alpha_daily",
        *beta_columns,
    ):

        if not pd.api.types.is_numeric_dtype(
            lagged[column]
        ):
            raise TypeError(
                f"{column} must be numeric."
            )

    for column in (
        "weight_bop",
        "return_total",
        *(
            ["contribution_total"]
            if "contribution_total" in realized.columns
            else []
        ),
    ):

        if not pd.api.types.is_numeric_dtype(
            realized[column]
        ):
            raise TypeError(
                f"{column} must be numeric."
            )

    for column in (
        "rf",
        *factor_columns,
    ):

        if not pd.api.types.is_numeric_dtype(
            factor_data[column]
        ):
            raise TypeError(
                f"{column} must be numeric."
            )

    if not np.isfinite(
        factor_data[
            ["rf", *factor_columns]
        ].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Factor source contains incomplete "
            "numeric observations."
        )

    availability_columns = [
        "model_available",
        "factor_available_on_return_date",
    ]

    if lagged[
        availability_columns
    ].isna().any().any():
        raise ValueError(
            "Missing model or factor availability flags."
        )

    for column in availability_columns:

        if not pd.api.types.is_bool_dtype(
            lagged[column]
        ):
            raise TypeError(
                f"{column} must be Boolean."
            )

    # --------------------------------------------------------
    # E. Join realized holding returns
    # --------------------------------------------------------

    merged = lagged.merge(
        realized,
        on=keys,
        how="outer",
        validate="one_to_one",
        indicator="_holding_merge",
        suffixes=("", "_source"),
    )

    if not merged[
        "_holding_merge"
    ].eq("both").all():
        raise ValueError(
            "Lagged and realized holding keys differ."
        )

    merged = merged.drop(
        columns="_holding_merge"
    )

    # --------------------------------------------------------
    # F. Join contemporaneous factor observations
    # --------------------------------------------------------

    merged = merged.merge(
        factor_data,
        on="date",
        how="left",
        validate="many_to_one",
        indicator="_factor_merge",
    )

    factor_available = (
        merged[
            "_factor_merge"
        ].eq("both").to_numpy(dtype=bool)
    )

    expected_factor_available = (
        merged[
            "factor_available_on_return_date"
        ].to_numpy(dtype=bool)
    )

    if not np.array_equal(
        factor_available,
        expected_factor_available,
    ):
        raise ValueError(
            "Factor availability disagrees with "
            "Cell 11 exposure alignment."
        )

    # --------------------------------------------------------
    # G. Validate model availability and historical timing
    # --------------------------------------------------------

    modeled = merged[
        "model_available"
    ].to_numpy(dtype=bool)

    if not merged.loc[
        modeled,
        "estimation_date",
    ].lt(
        merged.loc[modeled, "date"]
    ).all():
        raise ValueError(
            "LOOK-AHEAD VIOLATION: Security factor "
            "coefficients are not strictly historical."
        )

    if merged.loc[
        ~modeled,
        "estimation_date",
    ].notna().any():
        raise ValueError(
            "Unmodeled holding unexpectedly has "
            "a regression estimation date."
        )

    betas = merged[
        beta_columns
    ].to_numpy(dtype=float)

    alpha = merged[
        "estimated_alpha_daily"
    ].to_numpy(dtype=float)

    if (
        not np.isfinite(betas[modeled]).all()
        or not np.isfinite(alpha[modeled]).all()
    ):
        raise ValueError(
            "Modeled security coefficients "
            "must be complete and finite."
        )

    if (
        not np.isnan(betas[~modeled]).all()
        or not np.isnan(alpha[~modeled]).all()
    ):
        raise ValueError(
            "Unmodeled security coefficients "
            "must remain NaN."
        )

    # --------------------------------------------------------
    # H. Reconcile source weights and realized contributions
    # --------------------------------------------------------

    weights = merged[
        "weight_bop"
    ].to_numpy(dtype=float)

    source_weights = merged[
        "weight_bop_source"
    ].to_numpy(dtype=float)

    returns = merged[
        "return_total"
    ].to_numpy(dtype=float)

    if (
        not np.isfinite(weights).all()
        or not np.isfinite(source_weights).all()
        or not np.isfinite(returns).all()
    ):
        raise ValueError(
            "Holdings weights and realized returns "
            "must be finite."
        )

    np.testing.assert_allclose(
        weights,
        source_weights,
        rtol=1e-10,
        atol=tolerance,
        err_msg=(
            "Lagged holdings weights differ "
            "from standardized source weights."
        ),
    )

    merged[
        "actual_total_contribution"
    ] = weights * returns

    if "contribution_total" in merged.columns:

        np.testing.assert_allclose(
            merged[
                "contribution_total"
            ].to_numpy(dtype=float),
            merged[
                "actual_total_contribution"
            ].to_numpy(dtype=float),
            rtol=1e-10,
            atol=tolerance,
            err_msg=(
                "Recorded security return contribution "
                "does not reconcile to weight times return."
            ),
        )

    # --------------------------------------------------------
    # I. Prepare attribution masks and arrays
    # --------------------------------------------------------

    n_rows = len(merged)

    rf = merged[
        "rf"
    ].to_numpy(dtype=float)

    factors = merged[
        list(factor_columns)
    ].to_numpy(dtype=float)

    valid_model = (
        factor_available & modeled
    )

    valid_unmodeled = (
        factor_available & ~modeled
    )

    # --------------------------------------------------------
    # J. Calculate realized excess returns
    # --------------------------------------------------------

    realized_excess = np.full(
        n_rows,
        np.nan,
    )

    realized_excess[factor_available] = (
        returns[factor_available]
        - rf[factor_available]
    )

    merged[
        "security_realized_excess_return"
    ] = realized_excess

    # --------------------------------------------------------
    # K. Calculate the mechanical risk-free component
    # --------------------------------------------------------

    risk_free_contribution = np.full(
        n_rows,
        np.nan,
    )

    risk_free_contribution[factor_available] = (
        weights[factor_available]
        * rf[factor_available]
    )

    merged[
        "risk_free_contribution"
    ] = risk_free_contribution

    # --------------------------------------------------------
    # L. Calculate factor return contributions
    # --------------------------------------------------------

    factor_contributions = np.full(
        (
            n_rows,
            len(factor_columns),
        ),
        np.nan,
    )

    factor_contributions[valid_model] = (
        weights[valid_model, None]
        * betas[valid_model]
        * factors[valid_model]
    )

    for position, factor in enumerate(
        factor_columns
    ):

        merged[
            f"factor_contribution_{factor}"
        ] = factor_contributions[:, position]

    # --------------------------------------------------------
    # M. Calculate estimated intercept contribution
    # --------------------------------------------------------

    estimated_intercept = np.full(
        n_rows,
        np.nan,
    )

    estimated_intercept[valid_model] = (
        weights[valid_model]
        * alpha[valid_model]
    )

    merged[
        "estimated_intercept_contribution"
    ] = estimated_intercept

    # --------------------------------------------------------
    # N. Calculate modeled excess returns
    # --------------------------------------------------------

    modeled_excess = np.full(
        n_rows,
        np.nan,
    )

    modeled_excess[valid_model] = (
        alpha[valid_model]
        + np.einsum(
            "ij,ij->i",
            betas[valid_model],
            factors[valid_model],
        )
    )

    merged[
        "modeled_security_excess_return"
    ] = modeled_excess

    # --------------------------------------------------------
    # O. Calculate realized ex-post model residuals
    # --------------------------------------------------------

    residual_excess = np.full(
        n_rows,
        np.nan,
    )

    residual_excess[valid_model] = (
        realized_excess[valid_model]
        - modeled_excess[valid_model]
    )

    merged[
        "security_residual_excess_return"
    ] = residual_excess

    residual_contribution = np.full(
        n_rows,
        np.nan,
    )

    residual_contribution[valid_model] = (
        weights[valid_model]
        * residual_excess[valid_model]
    )

    merged[
        "modeled_residual_contribution"
    ] = residual_contribution

    # --------------------------------------------------------
    # P. Preserve unmodeled security excess contributions
    # --------------------------------------------------------

    unmodeled_contribution = np.full(
        n_rows,
        np.nan,
    )

    unmodeled_contribution[valid_unmodeled] = (
        weights[valid_unmodeled]
        * realized_excess[valid_unmodeled]
    )

    merged[
        "unmodeled_excess_contribution"
    ] = unmodeled_contribution

    # --------------------------------------------------------
    # Q. Reconstruct actual security total contributions
    # --------------------------------------------------------

    reconstructed = np.full(
        n_rows,
        np.nan,
    )

    reconstructed[valid_model] = (
        risk_free_contribution[valid_model]
        + estimated_intercept[valid_model]
        + factor_contributions[
            valid_model
        ].sum(axis=1)
        + residual_contribution[valid_model]
    )

    reconstructed[valid_unmodeled] = (
        risk_free_contribution[valid_unmodeled]
        + unmodeled_contribution[valid_unmodeled]
    )

    merged[
        "attribution_reconstructed_total_contribution"
    ] = reconstructed

    merged[
        "attribution_reconciliation_error"
    ] = (
        merged["actual_total_contribution"]
        - merged[
            "attribution_reconstructed_total_contribution"
        ]
    )

    # --------------------------------------------------------
    # R. Validate security-level return reconciliation
    # --------------------------------------------------------

    errors = merged[
        "attribution_reconciliation_error"
    ].to_numpy(dtype=float)

    if not np.isfinite(
        errors[factor_available]
    ).all():
        raise ValueError(
            "Security attribution reconciliation "
            "contains non-finite errors."
        )

    maximum_error = np.max(
        np.abs(
            errors[factor_available]
        ),
        initial=0.0,
    )

    if maximum_error > tolerance:
        raise ValueError(
            "Security-level factor return attribution "
            "does not reconcile to realized total returns."
        )

    # --------------------------------------------------------
    # S. Enforce missing-factor-date treatment
    # --------------------------------------------------------

    factor_dependent_columns = [
        "risk_free_contribution",
        "security_realized_excess_return",
        "estimated_intercept_contribution",
        "modeled_security_excess_return",
        "security_residual_excess_return",
        "modeled_residual_contribution",
        "unmodeled_excess_contribution",
        "attribution_reconstructed_total_contribution",
        "attribution_reconciliation_error",
        *[
            f"factor_contribution_{factor}"
            for factor in factor_columns
        ],
    ]

    if merged.loc[
        ~factor_available,
        factor_dependent_columns,
    ].notna().any().any():
        raise ValueError(
            "Missing factor dates received "
            "factor-dependent attribution values."
        )

    # --------------------------------------------------------
    # T. Assign transparent attribution statuses
    # --------------------------------------------------------

    merged[
        "attribution_status"
    ] = np.select(
        [
            ~factor_available,
            modeled,
        ],
        [
            "NO_CURRENT_FACTOR_RETURN",
            "MODELED_FACTOR_ATTRIBUTION",
        ],
        default="UNMODELED_SECURITY_EXCESS",
    )

    merged = merged.drop(
        columns=[
            "_factor_merge",
            "weight_bop_source",
        ]
    )

    return (
        merged
        .sort_values(keys)
        .reset_index(drop=True)
    )


# Research reference: 11_fama_french_factor_analytics.ipynb, code cell index 13.
def aggregate_daily_security_factor_attribution(
    security_attribution_df: pd.DataFrame,
    *,
    factor_columns: tuple[str, ...],
    model_id: str,
    tolerance: float = 1e-12,
) -> pd.DataFrame:
    """
    Aggregate security-level factor attribution to
    independent daily portfolio return decompositions.

    Output grain:
        portfolio_id, date, model_id

    All monetary-independent return contributions
    use daily simple-decimal return units.

    Missing factor-dependent components remain NaN.

    An absent holding population contributes an
    additive structural zero to the return identity,
    without changing missing security coefficients.
    """

    keys = [
        "portfolio_id",
        "date",
        "model_id",
    ]

    holding_keys = [
        *keys,
        "security_id",
    ]

    factor_components = [
        f"factor_contribution_{factor}"
        for factor in factor_columns
    ]

    component_columns = [
        "risk_free_contribution",
        "estimated_intercept_contribution",
        *factor_components,
        "modeled_residual_contribution",
        "unmodeled_excess_contribution",
    ]

    required = {
        *holding_keys,
        "weight_bop",
        "model_available",
        "factor_available_on_return_date",
        "attribution_status",
        "actual_total_contribution",
        "attribution_reconstructed_total_contribution",
        *component_columns,
    }

    missing = required - set(
        security_attribution_df.columns
    )

    if missing:
        raise ValueError(
            f"Missing security attribution columns: "
            f"{sorted(missing)}"
        )

    if not factor_columns:
        raise ValueError(
            "At least one factor is required."
        )

    if len(set(factor_columns)) != len(factor_columns):
        raise ValueError(
            "Duplicate factor names."
        )

    working = security_attribution_df[
        list(required)
    ].copy()

    working["date"] = pd.to_datetime(
        working["date"],
        errors="raise",
    ).astype("datetime64[ns]")

    if (
        working["date"].isna().any()
        or not working["date"].eq(
            working["date"].dt.normalize()
        ).all()
    ):
        raise ValueError(
            "Invalid security attribution dates."
        )

    if working[holding_keys].isna().any().any():
        raise ValueError(
            "Missing holding attribution keys."
        )

    if working.duplicated(
        subset=holding_keys
    ).any():
        raise ValueError(
            "Duplicate holding attribution keys."
        )

    if not working["model_id"].eq(
        model_id
    ).all():
        raise ValueError(
            "Inconsistent attribution model IDs."
        )

    if (
        working[
            [
                "model_available",
                "factor_available_on_return_date",
            ]
        ].isna().any().any()
    ):
        raise ValueError(
            "Missing attribution availability flags."
        )

    if not all(
        pd.api.types.is_bool_dtype(working[column])
        for column in (
            "model_available",
            "factor_available_on_return_date",
        )
    ):
        raise TypeError(
            "Attribution availability flags "
            "must be Boolean."
        )

    weights = working[
        "weight_bop"
    ].to_numpy(dtype=float)

    actual = working[
        "actual_total_contribution"
    ].to_numpy(dtype=float)

    modeled = working[
        "model_available"
    ].to_numpy(dtype=bool)

    factor_available = working[
        "factor_available_on_return_date"
    ].to_numpy(dtype=bool)

    if (
        not np.isfinite(weights).all()
        or not np.isfinite(actual).all()
    ):
        raise ValueError(
            "Holding weights and realized "
            "contributions must be finite."
        )

    # --------------------------------------------------------
    # A. Independently verify security components
    # --------------------------------------------------------

    expected_status = np.select(
        [
            ~factor_available,
            modeled,
        ],
        [
            "NO_CURRENT_FACTOR_RETURN",
            "MODELED_FACTOR_ATTRIBUTION",
        ],
        default="UNMODELED_SECURITY_EXCESS",
    )

    if not np.array_equal(
        expected_status,
        working["attribution_status"].to_numpy(),
    ):
        raise ValueError(
            "Security attribution statuses "
            "are inconsistent with availability."
        )

    component_values = working[
        component_columns
    ].to_numpy(dtype=float)

    reconstructed = working[
        "attribution_reconstructed_total_contribution"
    ].to_numpy(dtype=float)

    if working.loc[
        ~factor_available,
        [
            *component_columns,
            "attribution_reconstructed_total_contribution",
        ],
    ].notna().any().any():
        raise ValueError(
            "Factor-unavailable dates contain "
            "factor-dependent contributions."
        )

    modeled_factor_available = (
        factor_available & modeled
    )

    unmodeled_factor_available = (
        factor_available & ~modeled
    )

    modeled_components = [
        "risk_free_contribution",
        "estimated_intercept_contribution",
        *factor_components,
        "modeled_residual_contribution",
    ]

    if not np.isfinite(
        working.loc[
            modeled_factor_available,
            modeled_components,
        ].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Modeled holding attribution "
            "contains unavailable components."
        )

    modeled_reconstructed = (
        working.loc[
            modeled_factor_available,
            modeled_components,
        ]
        .sum(axis=1)
        .to_numpy(dtype=float)
    )

    np.testing.assert_allclose(
        modeled_reconstructed,
        actual[modeled_factor_available],
        rtol=1e-10,
        atol=tolerance,
    )

    if not np.isfinite(
        working.loc[
            unmodeled_factor_available,
            [
                "risk_free_contribution",
                "unmodeled_excess_contribution",
            ],
        ].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Unmodeled holding return components "
            "are incomplete."
        )

    unmodeled_reconstructed = (
        working.loc[
            unmodeled_factor_available,
            [
                "risk_free_contribution",
                "unmodeled_excess_contribution",
            ],
        ]
        .sum(axis=1)
        .to_numpy(dtype=float)
    )

    np.testing.assert_allclose(
        unmodeled_reconstructed,
        actual[unmodeled_factor_available],
        rtol=1e-10,
        atol=tolerance,
    )

    if working.loc[
        unmodeled_factor_available,
        [
            "estimated_intercept_contribution",
            *factor_components,
            "modeled_residual_contribution",
        ],
    ].notna().any().any():
        raise ValueError(
            "Unmodeled securities contain "
            "estimated factor components."
        )

    np.testing.assert_allclose(
        reconstructed[factor_available],
        actual[factor_available],
        rtol=1e-10,
        atol=tolerance,
    )

    # --------------------------------------------------------
    # B. Calculate weight coverage
    # --------------------------------------------------------

    working["gross_weight_bop"] = np.abs(weights)

    working["modeled_gross_weight_bop"] = np.where(
        modeled,
        np.abs(weights),
        0.0,
    )

    working["unmodeled_gross_weight_bop"] = np.where(
        ~modeled,
        np.abs(weights),
        0.0,
    )

    # --------------------------------------------------------
    # C. Aggregate daily holdings and returns
    # --------------------------------------------------------

    grouped = working.groupby(
        keys,
        observed=True,
        sort=True,
    )

    if grouped[
        "factor_available_on_return_date"
    ].nunique().gt(1).any():
        raise ValueError(
            "Factor availability differs within "
            "a portfolio return date."
        )

    daily_summary = (
        grouped
        .agg(
            holding_count=(
                "security_id",
                "size",
            ),
            modeled_holding_count=(
                "model_available",
                "sum",
            ),
            total_net_weight_bop=(
                "weight_bop",
                "sum",
            ),
            total_gross_weight_bop=(
                "gross_weight_bop",
                "sum",
            ),
            modeled_gross_weight_bop=(
                "modeled_gross_weight_bop",
                "sum",
            ),
            unmodeled_gross_weight_bop=(
                "unmodeled_gross_weight_bop",
                "sum",
            ),
            realized_total_return=(
                "actual_total_contribution",
                "sum",
            ),
            factor_available_on_return_date=(
                "factor_available_on_return_date",
                "first",
            ),
        )
        .reset_index()
    )

    daily_components = (
        grouped[component_columns]
        .sum(min_count=1)
        .reset_index()
    )

    result = daily_summary.merge(
        daily_components,
        on=keys,
        how="inner",
        validate="one_to_one",
    )

    result["unmodeled_holding_count"] = (
        result["holding_count"]
        - result["modeled_holding_count"]
    )

    # --------------------------------------------------------
    # D. Preserve legitimate structural-zero components
    # --------------------------------------------------------

    available_daily = result[
        "factor_available_on_return_date"
    ].to_numpy(dtype=bool)

    has_modeled = result[
        "modeled_holding_count"
    ].gt(0).to_numpy()

    has_unmodeled = result[
        "unmodeled_holding_count"
    ].gt(0).to_numpy()

    # Zero unmodeled holdings imply zero unmodeled
    # excess-return contribution, not zero beta.

    result.loc[
        available_daily & ~has_unmodeled,
        "unmodeled_excess_contribution",
    ] = 0.0

    # --------------------------------------------------------
    # E. Calculate modeled contribution totals
    # --------------------------------------------------------

    result["modeled_factor_contribution_total"] = (
        result[factor_components]
        .sum(
            axis=1,
            min_count=len(factor_components),
        )
    )

    result[
        "modeled_estimated_excess_contribution"
    ] = (
        result["estimated_intercept_contribution"]
        + result["modeled_factor_contribution_total"]
    )

    modeled_identity_columns = [
        "estimated_intercept_contribution",
        "modeled_factor_contribution_total",
        "modeled_residual_contribution",
    ]

    modeled_excess_total = (
        result[modeled_identity_columns]
        .sum(
            axis=1,
            min_count=len(modeled_identity_columns),
        )
        .to_numpy(dtype=float)
    )

    # No modeled holdings means no modeled return component.
    # This structural zero is used only for reconciliation.
    # The modeled exposure columns themselves remain NaN.

    modeled_excess_for_identity = np.where(
        has_modeled,
        modeled_excess_total,
        0.0,
    )

    unmodeled_excess = result[
        "unmodeled_excess_contribution"
    ].to_numpy(dtype=float)

    risk_free = result[
        "risk_free_contribution"
    ].to_numpy(dtype=float)

    result["attributed_total_return"] = np.where(
        available_daily,
        (
            risk_free
            + modeled_excess_for_identity
            + unmodeled_excess
        ),
        np.nan,
    )

    result["attribution_reconciliation_error"] = (
        result["realized_total_return"]
        - result["attributed_total_return"]
    )

    # --------------------------------------------------------
    # F. Validate daily attribution identities
    # --------------------------------------------------------

    if not np.isfinite(
        result.loc[
            available_daily,
            "attributed_total_return",
        ].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Factor-available daily attribution "
            "contains non-finite results."
        )

    np.testing.assert_allclose(
        result.loc[
            available_daily,
            "attributed_total_return",
        ].to_numpy(dtype=float),
        result.loc[
            available_daily,
            "realized_total_return",
        ].to_numpy(dtype=float),
        rtol=1e-10,
        atol=tolerance,
    )

    if result.loc[
        ~available_daily,
        [
            *component_columns,
            "modeled_factor_contribution_total",
            "modeled_estimated_excess_contribution",
            "attributed_total_return",
        ],
    ].notna().any().any():
        raise ValueError(
            "Missing factor dates received attribution."
        )

    if result.loc[
        available_daily & ~has_modeled,
        [
            "estimated_intercept_contribution",
            *factor_components,
            "modeled_residual_contribution",
            "modeled_factor_contribution_total",
        ],
    ].notna().any().any():
        raise ValueError(
            "Unavailable modeled contributions "
            "were incorrectly zero-filled."
        )

    # --------------------------------------------------------
    # G. Classify exposure and attribution completeness
    # --------------------------------------------------------

    if result[
        "total_gross_weight_bop"
    ].le(0).any():
        raise ValueError(
            "Invalid portfolio gross weight."
        )

    result["modeled_gross_weight_coverage"] = (
        result["modeled_gross_weight_bop"]
        / result["total_gross_weight_bop"]
    )

    result["exposure_coverage_status"] = np.select(
        [
            result[
                "modeled_gross_weight_bop"
            ].le(tolerance),
            result[
                "unmodeled_gross_weight_bop"
            ].le(tolerance),
        ],
        [
            "NO_MODELED_HOLDINGS",
            "FULLY_MODELED",
        ],
        default="PARTIALLY_MODELED",
    )

    result["factor_attribution_status"] = np.select(
        [
            ~available_daily,
            result[
                "exposure_coverage_status"
            ].eq("NO_MODELED_HOLDINGS"),
            result[
                "exposure_coverage_status"
            ].eq("FULLY_MODELED"),
        ],
        [
            "NO_CURRENT_FACTOR_RETURN",
            "NO_MODELED_EXPOSURE",
            "FULL_COVERAGE_FACTORS_AVAILABLE",
        ],
        default="PARTIAL_COVERAGE_FACTORS_AVAILABLE",
    )

    if result.duplicated(subset=keys).any():
        raise ValueError(
            "Duplicate daily portfolio attribution keys."
        )

    return (
        result
        .sort_values(keys)
        .reset_index(drop=True)
    )


# Research reference: 11_fama_french_factor_analytics.ipynb, code cell index 13.
def calculate_daily_active_factor_attribution(
    portfolio_attribution_df: pd.DataFrame,
    *,
    active_portfolio_id: str,
    benchmark_portfolio_id: str,
    factor_columns: tuple[str, ...],
    tolerance: float = 1e-12,
) -> pd.DataFrame:
    """
    Calculate daily active-minus-benchmark attribution.

    Outputs retain separate contributions from:
        risk-free,
        estimated intercept,
        individual factors,
        modeled residual,
        unmodeled security excess.

    No missing factor-date contribution is calculated.
    """

    if active_portfolio_id == benchmark_portfolio_id:
        raise ValueError(
            "Active and benchmark IDs must differ."
        )

    component_columns = [
        "risk_free_contribution",
        "estimated_intercept_contribution",
        *[
            f"factor_contribution_{factor}"
            for factor in factor_columns
        ],
        "modeled_factor_contribution_total",
        "modeled_residual_contribution",
        "unmodeled_excess_contribution",
    ]

    numeric_columns = [
        "realized_total_return",
        "attributed_total_return",
        "total_net_weight_bop",
        "modeled_gross_weight_coverage",
        *component_columns,
    ]

    other_columns = [
        "factor_available_on_return_date",
        "exposure_coverage_status",
    ]

    selected_columns = [
        "date",
        "model_id",
        *numeric_columns,
        *other_columns,
    ]

    portfolio_side = (
        portfolio_attribution_df.loc[
            portfolio_attribution_df[
                "portfolio_id"
            ].eq(active_portfolio_id),
            selected_columns,
        ]
        .copy()
        .rename(
            columns={
                column: f"portfolio_{column}"
                for column in [
                    *numeric_columns,
                    *other_columns,
                ]
            }
        )
    )

    benchmark_side = (
        portfolio_attribution_df.loc[
            portfolio_attribution_df[
                "portfolio_id"
            ].eq(benchmark_portfolio_id),
            selected_columns,
        ]
        .copy()
        .rename(
            columns={
                column: f"benchmark_{column}"
                for column in [
                    *numeric_columns,
                    *other_columns,
                ]
            }
        )
    )

    if (
        portfolio_side.empty
        or benchmark_side.empty
    ):
        raise ValueError(
            "Active or benchmark attribution is unavailable."
        )

    keys = [
        "date",
        "model_id",
    ]

    result = portfolio_side.merge(
        benchmark_side,
        on=keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    if not result["_merge"].eq("both").all():
        raise ValueError(
            "Active and benchmark attribution "
            "calendars differ."
        )

    result = result.drop(columns="_merge")

    if not result[
        "portfolio_factor_available_on_return_date"
    ].eq(
        result[
            "benchmark_factor_available_on_return_date"
        ]
    ).all():
        raise ValueError(
            "Inconsistent active/benchmark factor dates."
        )

    result["portfolio_id"] = active_portfolio_id

    result["benchmark_portfolio_id"] = (
        benchmark_portfolio_id
    )

    result["factor_available_on_return_date"] = (
        result[
            "portfolio_factor_available_on_return_date"
        ]
    )

    for component in [
        "realized_total_return",
        "attributed_total_return",
        *component_columns,
    ]:

        result[f"active_{component}"] = (
            result[f"portfolio_{component}"]
            - result[f"benchmark_{component}"]
        )

    result["active_net_weight_difference"] = (
        result["portfolio_total_net_weight_bop"]
        - result["benchmark_total_net_weight_bop"]
    )

    result["active_attribution_reconciliation_error"] = (
        result["active_realized_total_return"]
        - result["active_attributed_total_return"]
    )

    available = result[
        "factor_available_on_return_date"
    ].to_numpy(dtype=bool)

    np.testing.assert_allclose(
        result.loc[
            available,
            "active_attributed_total_return",
        ].to_numpy(dtype=float),
        result.loc[
            available,
            "active_realized_total_return",
        ].to_numpy(dtype=float),
        rtol=1e-10,
        atol=tolerance,
    )

    if result.loc[
        ~available,
        "active_attributed_total_return",
    ].notna().any():
        raise ValueError(
            "Active attribution was calculated "
            "without contemporaneous factor data."
        )

    missing_side = (
        result[
            "portfolio_exposure_coverage_status"
        ].eq("NO_MODELED_HOLDINGS")
        | result[
            "benchmark_exposure_coverage_status"
        ].eq("NO_MODELED_HOLDINGS")
    )

    full_coverage = (
        result[
            "portfolio_exposure_coverage_status"
        ].eq("FULLY_MODELED")
        & result[
            "benchmark_exposure_coverage_status"
        ].eq("FULLY_MODELED")
    )

    result[
        "active_factor_attribution_status"
    ] = np.select(
        [
            ~available,
            missing_side,
            full_coverage,
        ],
        [
            "NO_CURRENT_FACTOR_RETURN",
            "MISSING_SIDE_EXPOSURE",
            "FULL_COVERAGE_FACTORS_AVAILABLE",
        ],
        default="PARTIAL_COVERAGE_FACTORS_AVAILABLE",
    )

    return (
        result
        .sort_values("date")
        .reset_index(drop=True)
    )
