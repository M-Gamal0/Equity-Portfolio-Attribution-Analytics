"""Factors sectors: validated research calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd



# Research reference: 11_fama_french_factor_analytics.ipynb, code cell index 21.
def aggregate_sector_modeled_factor_exposures(
    holdings_df: pd.DataFrame,
    *,
    factor_columns: tuple[str, ...],
    tolerance: float = 1e-10,
) -> pd.DataFrame:
    """
    Aggregate holdings-weighted modeled factor exposures
    by portfolio, return date and sector.

    Grain:
        portfolio_id, date, sector

    The modeled-only sector contribution is:

        sum(weight_bop * beta)

    calculated exclusively for modeled securities.

    Unmodeled securities retain their gross weights
    and coverage classification.

    A numerical zero with no modeled holdings denotes
    an empty modeled-only subtotal, not a zero beta.
    """

    keys = [
        "portfolio_id",
        "date",
        "sector",
    ]

    beta_columns = [
        f"beta_{factor}"
        for factor in factor_columns
    ]

    required = {
        *keys,
        "security_id",
        "weight_bop",
        "model_available",
        *beta_columns,
    }

    missing = required - set(
        holdings_df.columns
    )

    if missing:
        raise ValueError(
            f"Missing sector inputs: {sorted(missing)}"
        )

    if (
        not factor_columns
        or len(set(factor_columns)) != len(factor_columns)
    ):
        raise ValueError(
            "Invalid sector factor specification."
        )

    working = holdings_df[
        list(required)
    ].copy()

    working["date"] = pd.to_datetime(
        working["date"],
        errors="raise",
    ).astype("datetime64[ns]")

    working["security_id"] = (
        working["security_id"].astype(str)
    )

    if working[
        [
            *keys,
            "security_id",
        ]
    ].isna().any().any():
        raise ValueError(
            "Sector holding keys contain missing values."
        )

    if working.duplicated(
        subset=[
            *keys,
            "security_id",
        ]
    ).any():
        raise ValueError(
            "Duplicate sector holdings detected."
        )

    if working["sector"].astype(str).str.strip().eq("").any():
        raise ValueError(
            "Empty sector classifications detected."
        )

    if (
        working["model_available"].isna().any()
        or not pd.api.types.is_bool_dtype(
            working["model_available"]
        )
    ):
        raise ValueError(
            "Model availability must be complete Boolean."
        )

    modeled = working[
        "model_available"
    ].to_numpy(dtype=bool)

    weights = working[
        "weight_bop"
    ].to_numpy(dtype=float)

    betas = working[
        beta_columns
    ].to_numpy(dtype=float)

    if not np.isfinite(weights).all():
        raise ValueError(
            "Invalid beginning-of-period sector weights."
        )

    if not np.isfinite(
        betas[modeled]
    ).all():
        raise ValueError(
            "Modeled sector coefficients are incomplete."
        )

    if not np.isnan(
        betas[~modeled]
    ).all():
        raise ValueError(
            "Unmodeled security coefficients "
            "were unexpectedly populated."
        )

    # --------------------------------------------------------
    # A. Calculate modeled and unmodeled gross weights
    # --------------------------------------------------------

    gross_weights = np.abs(weights)

    working["sector_gross_weight_bop"] = (
        gross_weights
    )

    working["modeled_gross_weight_bop"] = np.where(
        modeled,
        gross_weights,
        0.0,
    )

    working["unmodeled_gross_weight_bop"] = np.where(
        ~modeled,
        gross_weights,
        0.0,
    )

    # --------------------------------------------------------
    # B. Calculate modeled-only factor exposure subtotals
    # --------------------------------------------------------

    contribution_columns = []

    for factor in factor_columns:

        output_column = (
            f"modeled_exposure_{factor}"
        )

        source_column = (
            f"beta_{factor}"
        )

        working[output_column] = (
            working["weight_bop"]
            * working[source_column]
        ).where(
            working["model_available"],
            0.0,
        )

        contribution_columns.append(
            output_column
        )

    # --------------------------------------------------------
    # C. Aggregate by portfolio, date and sector
    # --------------------------------------------------------

    grouped = working.groupby(
        keys,
        observed=True,
        sort=True,
    )

    summary = (
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
            sector_net_weight_bop=(
                "weight_bop",
                "sum",
            ),
            sector_gross_weight_bop=(
                "sector_gross_weight_bop",
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
        )
        .reset_index()
    )

    components = (
        grouped[contribution_columns]
        .sum()
        .reset_index()
    )

    result = summary.merge(
        components,
        on=keys,
        how="inner",
        validate="one_to_one",
    )

    result["unmodeled_holding_count"] = (
        result["holding_count"]
        - result["modeled_holding_count"]
    )

    # --------------------------------------------------------
    # D. Explicit sector model coverage
    # --------------------------------------------------------

    result[
        "sector_modeled_gross_weight_coverage"
    ] = np.where(
        result["sector_gross_weight_bop"].gt(0),
        (
            result["modeled_gross_weight_bop"]
            / result["sector_gross_weight_bop"]
        ),
        np.nan,
    )

    result["sector_model_status"] = np.select(
        [
            result["modeled_holding_count"].eq(0),
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

    # --------------------------------------------------------
    # E. Weight reconciliation
    # --------------------------------------------------------

    np.testing.assert_allclose(
        (
            result["modeled_gross_weight_bop"]
            + result["unmodeled_gross_weight_bop"]
        ),
        result["sector_gross_weight_bop"],
        rtol=1e-10,
        atol=1e-12,
    )

    if result.duplicated(
        subset=keys
    ).any():
        raise ValueError(
            "Duplicate sector exposure output keys."
        )

    return result
