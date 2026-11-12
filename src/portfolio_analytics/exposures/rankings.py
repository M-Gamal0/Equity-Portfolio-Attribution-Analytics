"""Exposures rankings: validated research calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd



# Research reference: 10_holdings_exposure_analytics.ipynb, code cell index 11.
def rank_portfolio_holdings(
    security_exposures: pd.DataFrame,
    *,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Rank invested securities by descending portfolio weight.

    Input grain:
        portfolio_id, date, weight_basis, security_id

    Output grain:
        portfolio_id, date, weight_basis, rank

    Rank:
        Sequential integer position, starting at one.

    Tie breaking:
        Ascending string representation of security_id.

    Zero-weight ledger records are excluded.

    BOP and EOP exposures are ranked independently.

    Only the first top_n invested securities are returned.

    The input DataFrame is not modified.
    """

    if (
        isinstance(top_n, (bool, np.bool_))
        or not isinstance(top_n, (int, np.integer))
        or top_n <= 0
    ):
        raise ValueError(
            "top_n must be a positive integer."
        )

    if not isinstance(security_exposures, pd.DataFrame):
        raise TypeError(
            "security_exposures must be a DataFrame."
        )

    group_keys = [
        "portfolio_id",
        "date",
        "weight_basis",
    ]

    input_keys = group_keys + ["security_id"]

    required_columns = set(input_keys) | {
        "sector",
        "security_weight",
        "is_held",
    }

    missing_columns = (
        required_columns - set(security_exposures.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing top-holdings input fields: "
            f"{sorted(missing_columns)}"
        )

    if security_exposures.empty:
        raise ValueError(
            "Security exposures cannot be empty."
        )

    columns = input_keys + [
        "sector",
        "security_weight",
        "is_held",
    ]

    if "ticker" in security_exposures.columns:
        columns.append("ticker")

    work = security_exposures[columns].copy()

    work["date"] = pd.to_datetime(
        work["date"],
        errors="raise",
    )

    if work[input_keys].isna().any().any():
        raise ValueError(
            "Top-holdings input keys contain nulls."
        )

    if work.duplicated(subset=input_keys).any():
        raise ValueError(
            "Duplicate top-holdings input keys."
        )

    if not work["weight_basis"].isin(
        ["bop", "eop"]
    ).all():
        raise ValueError(
            "Unsupported top-holdings weight_basis."
        )

    sector_labels = (
        work["sector"]
        .astype("string")
        .str.strip()
    )

    if (
        sector_labels.isna().any()
        or sector_labels.eq("").any()
    ):
        raise ValueError(
            "Missing or blank sector classifications."
        )

    if not pd.api.types.is_numeric_dtype(
        work["security_weight"]
    ):
        raise TypeError(
            "security_weight must be numeric."
        )

    weights = work["security_weight"].to_numpy(
        dtype=np.float64,
        na_value=np.nan,
    )

    if (
        not np.isfinite(weights).all()
        or (weights < 0).any()
        or (weights > 1).any()
    ):
        raise ValueError(
            "Invalid long-only holdings weights."
        )

    if (
        not pd.api.types.is_bool_dtype(work["is_held"])
        or work["is_held"].isna().any()
    ):
        raise TypeError(
            "is_held must contain nonmissing Booleans."
        )

    if not work["is_held"].eq(
        work["security_weight"] > 0
    ).all():
        raise ValueError(
            "Invested-position flags disagree with weights."
        )

    # Rank only securities with positive invested weight.

    ranked = work.loc[work["is_held"]].copy()

    ranked["_security_sort_key"] = (
        ranked["security_id"].astype("string")
    )

    ranked = ranked.sort_values(
        group_keys + [
            "security_weight",
            "_security_sort_key",
        ],
        ascending=[
            True, True, True, False, True,
        ],
        kind="stable",
    )

    ranked["rank"] = (
        ranked.groupby(group_keys, sort=False)
        .cumcount()
        .add(1)
    )

    ranked = ranked.loc[
        ranked["rank"].le(top_n)
    ]

    output_columns = group_keys + [
        "rank",
        "security_id",
        "sector",
        "security_weight",
    ]

    if "ticker" in ranked.columns:
        output_columns.append("ticker")

    result = (
        ranked[output_columns]
        .reset_index(drop=True)
    )

    if result.duplicated(
        subset=group_keys + ["rank"]
    ).any():
        raise ValueError(
            "Duplicate top-holdings ranking keys."
        )

    return result


# Research reference: 10_holdings_exposure_analytics.ipynb, code cell index 11.
def rank_active_security_positions(
    active_exposures: pd.DataFrame,
    *,
    top_n: int = 10,
    weight_tolerance: float = 1e-10,
) -> pd.DataFrame:
    """
    Rank security overweights and underweights separately.

    Input grain:
        portfolio_id, benchmark_portfolio_id,
        date, weight_basis, security_id

    Output grain:
        portfolio_id, benchmark_portfolio_id,
        date, weight_basis, position_side, rank

    Position sides:
        overweight:
            active_weight > 0

        underweight:
            active_weight < 0

    Rankings:
        Descending absolute active-weight magnitude.

    Exact zero active weights are excluded.

    The weight_tolerance parameter validates the
    active-weight identity; it does not suppress
    legitimate small active positions.

    The input DataFrame is not modified.
    """

    if (
        isinstance(top_n, (bool, np.bool_))
        or not isinstance(top_n, (int, np.integer))
        or top_n <= 0
    ):
        raise ValueError(
            "top_n must be a positive integer."
        )

    if (
        not np.isfinite(weight_tolerance)
        or not 0 < weight_tolerance < 1
    ):
        raise ValueError(
            "Invalid weight_tolerance."
        )

    if not isinstance(active_exposures, pd.DataFrame):
        raise TypeError(
            "active_exposures must be a DataFrame."
        )

    group_keys = [
        "portfolio_id",
        "benchmark_portfolio_id",
        "date",
        "weight_basis",
    ]

    input_keys = group_keys + ["security_id"]

    required_columns = set(input_keys) | {
        "sector",
        "portfolio_weight",
        "benchmark_weight",
        "active_weight",
    }

    missing_columns = (
        required_columns - set(active_exposures.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing active-ranking input fields: "
            f"{sorted(missing_columns)}"
        )

    if active_exposures.empty:
        raise ValueError(
            "Active exposures cannot be empty."
        )

    columns = input_keys + [
        "sector",
        "portfolio_weight",
        "benchmark_weight",
        "active_weight",
    ]

    if "ticker" in active_exposures.columns:
        columns.append("ticker")

    work = active_exposures[columns].copy()

    work["date"] = pd.to_datetime(
        work["date"],
        errors="raise",
    )

    if work[input_keys].isna().any().any():
        raise ValueError(
            "Active-ranking input keys contain nulls."
        )

    if work.duplicated(subset=input_keys).any():
        raise ValueError(
            "Duplicate active-ranking input keys."
        )

    if not work["weight_basis"].isin(
        ["bop", "eop"]
    ).all():
        raise ValueError(
            "Unsupported active-ranking weight_basis."
        )

    sector_labels = (
        work["sector"]
        .astype("string")
        .str.strip()
    )

    if (
        sector_labels.isna().any()
        or sector_labels.eq("").any()
    ):
        raise ValueError(
            "Missing or blank sector classifications."
        )

    for column in (
        "portfolio_weight",
        "benchmark_weight",
        "active_weight",
    ):

        if not pd.api.types.is_numeric_dtype(
            work[column]
        ):
            raise TypeError(
                f"{column} must be numeric."
            )

        values = work[column].to_numpy(
            dtype=np.float64,
            na_value=np.nan,
        )

        if not np.isfinite(values).all():
            raise ValueError(
                f"{column} contains nonfinite values."
            )

    for column in (
        "portfolio_weight",
        "benchmark_weight",
    ):

        values = work[column].to_numpy(
            dtype=np.float64
        )

        if (
            (values < 0).any()
            or (values > 1 + weight_tolerance).any()
        ):
            raise ValueError(
                f"{column} violates long-only weight bounds."
            )

    identity_error = (
        work["active_weight"]
        .sub(
            work["portfolio_weight"]
            - work["benchmark_weight"]
        )
        .abs()
        .max()
    )

    if identity_error > weight_tolerance:
        raise ValueError(
            "Active-ranking weights do not reconcile."
        )

    # Exact zero active weights have no directional rank.

    ranked = work.loc[
        work["active_weight"].ne(0)
    ].copy()

    ranked["position_side"] = np.where(
        ranked["active_weight"].gt(0),
        "overweight",
        "underweight",
    )

    ranked["absolute_active_weight"] = (
        ranked["active_weight"].abs()
    )

    ranked["_security_sort_key"] = (
        ranked["security_id"].astype("string")
    )

    ranking_keys = group_keys + ["position_side"]

    ranked = ranked.sort_values(
        ranking_keys + [
            "absolute_active_weight",
            "_security_sort_key",
        ],
        ascending=[
            True, True, True, True, True,
            False, True,
        ],
        kind="stable",
    )

    ranked["rank"] = (
        ranked.groupby(ranking_keys, sort=False)
        .cumcount()
        .add(1)
    )

    ranked = ranked.loc[
        ranked["rank"].le(top_n)
    ]

    output_columns = ranking_keys + [
        "rank",
        "security_id",
        "sector",
        "portfolio_weight",
        "benchmark_weight",
        "active_weight",
        "absolute_active_weight",
    ]

    if "ticker" in ranked.columns:
        output_columns.append("ticker")

    result = (
        ranked[output_columns]
        .reset_index(drop=True)
    )

    if result.duplicated(
        subset=ranking_keys + ["rank"]
    ).any():
        raise ValueError(
            "Duplicate active-position ranking keys."
        )

    return result
