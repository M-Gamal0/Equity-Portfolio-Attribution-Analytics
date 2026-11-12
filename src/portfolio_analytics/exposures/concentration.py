"""Exposures concentration: validated research calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd



# Research reference: 10_holdings_exposure_analytics.ipynb, code cell index 7.
def calculate_security_concentration(
    security_exposures: pd.DataFrame,
    *,
    top_k_values: tuple[int, ...] = (1, 5, 10, 20),
    require_fully_invested: bool = False,
    weight_tolerance: float = 1e-10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate security concentration and top-K exposures.

    Required columns:
        portfolio_id, date, security_id, weight_basis,
        security_weight, is_held

    Input grain:
        portfolio_id, date, security_id, weight_basis

    Summary output grain:
        portfolio_id, date, weight_basis

    Top-K output grain:
        portfolio_id, date, weight_basis, k

    Methodology:
        Capital HHI:
            sum(w_i ** 2)

        Equity-sleeve HHI:
            sum(w_i ** 2) / sum(w_i) ** 2

        Effective holdings:
            1 / equity-sleeve HHI

        Top-K capital concentration:
            sum of the K largest positive security weights.

    All weights use decimal fractions.

    For fully invested portfolios, capital HHI equals
    equity-sleeve HHI.

    For a zero-investment equity sleeve, normalized HHI
    and effective holdings are unavailable (NaN).

    Zero-weight ledger records are preserved for
    observation counts but excluded from invested counts.

    The input DataFrame is not modified.
    """

    # --------------------------------------------------------
    # A. Validate parameters
    # --------------------------------------------------------

    if not isinstance(security_exposures, pd.DataFrame):
        raise TypeError(
            "security_exposures must be a pandas DataFrame."
        )

    if (
        not isinstance(top_k_values, tuple)
        or not top_k_values
        or any(
            isinstance(k, (bool, np.bool_))
            or not isinstance(k, (int, np.integer))
            or k <= 0
            for k in top_k_values
        )
        or len(top_k_values) != len(set(top_k_values))
    ):
        raise ValueError(
            "top_k_values must contain distinct "
            "positive integer values."
        )

    if (
        not np.isfinite(weight_tolerance)
        or not 0 < weight_tolerance < 1
    ):
        raise ValueError(
            "Invalid weight_tolerance."
        )

    # --------------------------------------------------------
    # B. Validate required fields and keys
    # --------------------------------------------------------

    group_keys = [
        "portfolio_id",
        "date",
        "weight_basis",
    ]

    security_keys = group_keys + ["security_id"]

    required_columns = set(security_keys) | {
        "security_weight",
        "is_held",
    }

    missing_columns = (
        required_columns - set(security_exposures.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing security concentration fields: "
            f"{sorted(missing_columns)}"
        )

    if security_exposures.empty:
        raise ValueError(
            "Security exposures cannot be empty."
        )

    work = security_exposures.loc[
        :,
        security_keys + ["security_weight", "is_held"],
    ].copy()

    work["date"] = pd.to_datetime(
        work["date"],
        errors="raise",
    )

    if work[security_keys].isna().any().any():
        raise ValueError(
            "Security concentration keys contain nulls."
        )

    if work.duplicated(subset=security_keys).any():
        raise ValueError(
            "Duplicate security concentration keys."
        )

    if not work["weight_basis"].isin(
        ["bop", "eop"]
    ).all():
        raise ValueError(
            "Unsupported weight_basis."
        )

    # --------------------------------------------------------
    # C. Validate weights and invested-position flags
    # --------------------------------------------------------

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

    if not np.isfinite(weights).all():
        raise ValueError(
            "Nonfinite security weights detected."
        )

    if (
        (weights < 0).any()
        or (weights > 1 + weight_tolerance).any()
    ):
        raise ValueError(
            "Invalid long-only security weights."
        )

    if (
        not pd.api.types.is_bool_dtype(work["is_held"])
        or work["is_held"].isna().any()
    ):
        raise TypeError(
            "is_held must contain nonmissing Boolean values."
        )

    if not work["is_held"].eq(
        work["security_weight"] > 0
    ).all():
        raise ValueError(
            "is_held flags do not match positive weights."
        )

    # --------------------------------------------------------
    # D. Calculate security concentration components
    # --------------------------------------------------------

    work["weight_squared"] = (
        work["security_weight"] ** 2
    )

    concentration = (
        work
        .groupby(group_keys, as_index=False)
        .agg(
            ledger_security_count=(
                "security_id", "size"
            ),
            invested_security_count=(
                "is_held", "sum"
            ),
            invested_weight_sum=(
                "security_weight", "sum"
            ),
            capital_hhi=(
                "weight_squared", "sum"
            ),
            largest_position_weight=(
                "security_weight", "max"
            ),
        )
    )

    if (
        concentration["invested_weight_sum"]
        > 1 + weight_tolerance
    ).any():
        raise ValueError(
            "Total security exposure exceeds 100%."
        )

    if require_fully_invested:

        investment_errors = (
            concentration["invested_weight_sum"]
            .sub(1.0)
            .abs()
        )

        if investment_errors.max() > weight_tolerance:
            raise ValueError(
                "Portfolio is not fully invested."
            )

    # --------------------------------------------------------
    # E. Calculate equity-sleeve-normalized concentration
    # --------------------------------------------------------

    positive_equity = (
        concentration["invested_weight_sum"] > 0
    )

    concentration["security_hhi"] = np.nan
    concentration["effective_holdings"] = np.nan
    concentration["largest_equity_sleeve_weight"] = np.nan

    concentration.loc[
        positive_equity, "security_hhi"
    ] = (
        concentration.loc[
            positive_equity, "capital_hhi"
        ]
        / concentration.loc[
            positive_equity, "invested_weight_sum"
        ] ** 2
    )

    concentration.loc[
        positive_equity, "effective_holdings"
    ] = (
        1.0
        / concentration.loc[
            positive_equity, "security_hhi"
        ]
    )

    concentration.loc[
        positive_equity, "largest_equity_sleeve_weight"
    ] = (
        concentration.loc[
            positive_equity, "largest_position_weight"
        ]
        / concentration.loc[
            positive_equity, "invested_weight_sum"
        ]
    )

    concentration["concentration_status"] = np.where(
        positive_equity,
        "AVAILABLE",
        "UNAVAILABLE_ZERO_EQUITY",
    )

    # --------------------------------------------------------
    # F. Validate normalized HHI and effective holdings
    # --------------------------------------------------------

    valid = concentration.loc[positive_equity]

    minimum_hhi = (
        1.0 / valid["invested_security_count"]
    )

    if (
        (
            valid["security_hhi"]
            < minimum_hhi - weight_tolerance
        ).any()
        or (
            valid["security_hhi"]
            > 1 + weight_tolerance
        ).any()
    ):
        raise ValueError(
            "Security HHI violates concentration bounds."
        )

    if (
        (
            valid["effective_holdings"]
            < 1 - weight_tolerance
        ).any()
        or (
            valid["effective_holdings"]
            > valid["invested_security_count"]
            + weight_tolerance
        ).any()
    ):
        raise ValueError(
            "Effective holdings violate mathematical bounds."
        )

    # --------------------------------------------------------
    # G. Rank positive security weights
    # --------------------------------------------------------

    ranked = (
        work.loc[
            work["is_held"],
            group_keys + ["security_weight"],
        ]
        .sort_values(
            group_keys + ["security_weight"],
            ascending=[True, True, True, False],
            kind="stable",
        )
        .copy()
    )

    ranked["security_rank"] = (
        ranked
        .groupby(group_keys)
        .cumcount()
        .add(1)
    )

    # --------------------------------------------------------
    # H. Calculate top-K concentration
    # --------------------------------------------------------

    top_k_frames = []

    for k in sorted(top_k_values):

        selected_top_k = (
            ranked.loc[
                ranked["security_rank"].le(k)
            ]
            .groupby(group_keys, as_index=False)[
                "security_weight"
            ]
            .sum()
            .rename(
                columns={
                    "security_weight": "top_k_weight"
                }
            )
        )

        top_k = concentration[
            group_keys + [
                "invested_security_count",
                "invested_weight_sum",
            ]
        ].merge(
            selected_top_k,
            on=group_keys,
            how="left",
            validate="one_to_one",
        )

        # Zero equity is an explicitly permitted state.
        # Top-K capital exposure is zero in that state.

        top_k["top_k_weight"] = (
            top_k["top_k_weight"].fillna(0.0)
        )

        top_k["k"] = k

        top_k["selected_count"] = np.minimum(
            k,
            top_k["invested_security_count"],
        )

        top_k["top_k_equity_weight"] = np.nan

        has_equity = (
            top_k["invested_weight_sum"] > 0
        )

        top_k.loc[
            has_equity, "top_k_equity_weight"
        ] = (
            top_k.loc[
                has_equity, "top_k_weight"
            ]
            / top_k.loc[
                has_equity, "invested_weight_sum"
            ]
        )

        top_k["concentration_status"] = np.where(
            has_equity,
            "AVAILABLE",
            "UNAVAILABLE_ZERO_EQUITY",
        )

        top_k_frames.append(top_k)

    top_k_concentration = pd.concat(
        top_k_frames,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # I. Validate top-K financial identities
    # --------------------------------------------------------

    top_k_concentration = (
        top_k_concentration
        .sort_values(
            group_keys + ["k"],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    if (
        top_k_concentration["top_k_weight"]
        > top_k_concentration["invested_weight_sum"]
        + weight_tolerance
    ).any():
        raise ValueError(
            "Top-K concentration exceeds invested weight."
        )

    top_k_changes = (
        top_k_concentration
        .groupby(group_keys)["top_k_weight"]
        .diff()
        .dropna()
    )

    if (top_k_changes < -weight_tolerance).any():
        raise ValueError(
            "Top-K concentration is not monotonic."
        )

    # --------------------------------------------------------
    # J. Validate standardized output keys
    # --------------------------------------------------------

    top_k_keys = group_keys + ["k"]

    if concentration.duplicated(group_keys).any():
        raise ValueError(
            "Duplicate security concentration output keys."
        )

    if top_k_concentration.duplicated(top_k_keys).any():
        raise ValueError(
            "Duplicate top-K concentration output keys."
        )

    return (
        concentration.sort_values(
            group_keys,
            kind="stable",
        ).reset_index(drop=True),
        top_k_concentration,
    )


# Research reference: 10_holdings_exposure_analytics.ipynb, code cell index 8.
def calculate_sector_concentration(
    sector_exposures: pd.DataFrame,
    *,
    top_k_values: tuple[int, ...] = (1, 3, 5),
    require_fully_invested: bool = False,
    weight_tolerance: float = 1e-10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate sector concentration and top-K sector exposure.

    Required input columns:
        portfolio_id, date, weight_basis,
        sector, sector_weight

    Input grain:
        portfolio_id, date, weight_basis, sector

    Summary output grain:
        portfolio_id, date, weight_basis

    Top-K output grain:
        portfolio_id, date, weight_basis, k

    Methodology
    -----------
    Capital sector HHI:
        sum(sector_weight ** 2)

    Equity-normalized sector HHI:
        capital_sector_hhi / invested_weight_sum ** 2

    Effective sectors:
        1 / normalized_sector_hhi

    Top-K sector weight:
        Sum of the K largest positive sector weights.

    Conventions
    -----------
    Weights are decimal portfolio-capital fractions.

    Zero-weight sector records contribute to the ledger
    count but not the invested sector count.

    Completely absent sectors are not manufactured.

    For zero invested equity, normalized HHI and effective
    sectors are unavailable (NaN).

    The input DataFrame is not modified.
    """

    # --------------------------------------------------------
    # A. Validate parameters
    # --------------------------------------------------------

    if not isinstance(sector_exposures, pd.DataFrame):
        raise TypeError(
            "sector_exposures must be a pandas DataFrame."
        )

    if (
        not isinstance(top_k_values, tuple)
        or not top_k_values
        or any(
            isinstance(k, (bool, np.bool_))
            or not isinstance(k, (int, np.integer))
            or k <= 0
            for k in top_k_values
        )
        or len(top_k_values) != len(set(top_k_values))
    ):
        raise ValueError(
            "top_k_values must contain distinct "
            "positive integers."
        )

    if (
        not np.isfinite(weight_tolerance)
        or not 0 < weight_tolerance < 1
    ):
        raise ValueError(
            "Invalid weight_tolerance."
        )

    # --------------------------------------------------------
    # B. Validate required columns and input grain
    # --------------------------------------------------------

    group_keys = [
        "portfolio_id",
        "date",
        "weight_basis",
    ]

    sector_keys = group_keys + ["sector"]

    required_columns = set(sector_keys) | {
        "sector_weight",
    }

    missing_columns = (
        required_columns - set(sector_exposures.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing sector concentration fields: "
            f"{sorted(missing_columns)}"
        )

    if sector_exposures.empty:
        raise ValueError(
            "Sector exposures cannot be empty."
        )

    work = sector_exposures.loc[
        :,
        sector_keys + ["sector_weight"],
    ].copy()

    work["date"] = pd.to_datetime(
        work["date"],
        errors="raise",
    )

    if work[sector_keys].isna().any().any():
        raise ValueError(
            "Sector concentration keys contain nulls."
        )

    if work.duplicated(subset=sector_keys).any():
        raise ValueError(
            "Duplicate sector concentration keys."
        )

    if not work["weight_basis"].isin(
        ["bop", "eop"]
    ).all():
        raise ValueError(
            "Unsupported weight_basis."
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

    # --------------------------------------------------------
    # C. Validate sector weights
    # --------------------------------------------------------

    if not pd.api.types.is_numeric_dtype(
        work["sector_weight"]
    ):
        raise TypeError(
            "sector_weight must be numeric."
        )

    weights = work["sector_weight"].to_numpy(
        dtype=np.float64,
        na_value=np.nan,
    )

    if not np.isfinite(weights).all():
        raise ValueError(
            "Nonfinite sector weights detected."
        )

    if (
        (weights < 0).any()
        or (weights > 1 + weight_tolerance).any()
    ):
        raise ValueError(
            "Invalid long-only sector weights."
        )

    # --------------------------------------------------------
    # D. Calculate sector concentration components
    # --------------------------------------------------------

    work["is_invested_sector"] = (
        work["sector_weight"] > 0
    )

    work["sector_weight_squared"] = (
        work["sector_weight"] ** 2
    )

    concentration = (
        work
        .groupby(group_keys, as_index=False)
        .agg(
            ledger_sector_count=(
                "sector", "size"
            ),
            invested_sector_count=(
                "is_invested_sector", "sum"
            ),
            invested_weight_sum=(
                "sector_weight", "sum"
            ),
            capital_sector_hhi=(
                "sector_weight_squared", "sum"
            ),
            largest_sector_weight=(
                "sector_weight", "max"
            ),
        )
    )

    if (
        concentration["invested_weight_sum"]
        > 1 + weight_tolerance
    ).any():
        raise ValueError(
            "Total sector exposure exceeds 100%."
        )

    if require_fully_invested:

        investment_errors = (
            concentration["invested_weight_sum"]
            .sub(1.0)
            .abs()
        )

        if investment_errors.max() > weight_tolerance:
            raise ValueError(
                "Portfolio is not fully invested."
            )

    # --------------------------------------------------------
    # E. Calculate normalized HHI and effective sectors
    # --------------------------------------------------------

    positive_equity = (
        concentration["invested_weight_sum"] > 0
    )

    concentration["sector_hhi"] = np.nan
    concentration["effective_sectors"] = np.nan

    concentration["largest_equity_sector_weight"] = (
        np.nan
    )

    concentration.loc[
        positive_equity, "sector_hhi"
    ] = (
        concentration.loc[
            positive_equity, "capital_sector_hhi"
        ]
        / concentration.loc[
            positive_equity, "invested_weight_sum"
        ] ** 2
    )

    concentration.loc[
        positive_equity, "effective_sectors"
    ] = (
        1.0
        / concentration.loc[
            positive_equity, "sector_hhi"
        ]
    )

    concentration.loc[
        positive_equity,
        "largest_equity_sector_weight",
    ] = (
        concentration.loc[
            positive_equity, "largest_sector_weight"
        ]
        / concentration.loc[
            positive_equity, "invested_weight_sum"
        ]
    )

    concentration["concentration_status"] = np.where(
        positive_equity,
        "AVAILABLE",
        "UNAVAILABLE_ZERO_EQUITY",
    )

    # --------------------------------------------------------
    # F. Validate mathematical sector concentration bounds
    # --------------------------------------------------------

    valid = concentration.loc[positive_equity]

    minimum_hhi = (
        1.0 / valid["invested_sector_count"]
    )

    if (
        (
            valid["sector_hhi"]
            < minimum_hhi - weight_tolerance
        ).any()
        or (
            valid["sector_hhi"]
            > 1 + weight_tolerance
        ).any()
    ):
        raise ValueError(
            "Sector HHI violates concentration bounds."
        )

    if (
        (
            valid["effective_sectors"]
            < 1 - weight_tolerance
        ).any()
        or (
            valid["effective_sectors"]
            > valid["invested_sector_count"]
            + weight_tolerance
        ).any()
    ):
        raise ValueError(
            "Effective sectors violate mathematical bounds."
        )

    # --------------------------------------------------------
    # G. Rank positive sector exposures
    # --------------------------------------------------------

    ranked = (
        work.loc[
            work["is_invested_sector"],
            sector_keys + ["sector_weight"],
        ]
        .sort_values(
            group_keys + ["sector_weight", "sector"],
            ascending=[True, True, True, False, True],
            kind="stable",
        )
        .copy()
    )

    ranked["sector_rank"] = (
        ranked
        .groupby(group_keys)
        .cumcount()
        .add(1)
    )

    # --------------------------------------------------------
    # H. Calculate top-K sector concentration
    # --------------------------------------------------------

    top_k_frames = []

    for k in sorted(top_k_values):

        top_k_weights = (
            ranked.loc[
                ranked["sector_rank"].le(k)
            ]
            .groupby(group_keys, as_index=False)[
                "sector_weight"
            ]
            .sum()
            .rename(
                columns={
                    "sector_weight": "top_k_sector_weight"
                }
            )
        )

        top_k = concentration[
            group_keys + [
                "invested_sector_count",
                "invested_weight_sum",
            ]
        ].merge(
            top_k_weights,
            on=group_keys,
            how="left",
            validate="one_to_one",
        )

        top_k["top_k_sector_weight"] = (
            top_k["top_k_sector_weight"].fillna(0.0)
        )

        top_k["k"] = k

        top_k["selected_count"] = np.minimum(
            k,
            top_k["invested_sector_count"],
        )

        top_k["top_k_equity_sector_weight"] = np.nan

        has_equity = (
            top_k["invested_weight_sum"] > 0
        )

        top_k.loc[
            has_equity,
            "top_k_equity_sector_weight",
        ] = (
            top_k.loc[
                has_equity, "top_k_sector_weight"
            ]
            / top_k.loc[
                has_equity, "invested_weight_sum"
            ]
        )

        top_k["concentration_status"] = np.where(
            has_equity,
            "AVAILABLE",
            "UNAVAILABLE_ZERO_EQUITY",
        )

        top_k_frames.append(top_k)

    top_k_concentration = pd.concat(
        top_k_frames,
        ignore_index=True,
    )

    top_k_concentration = (
        top_k_concentration
        .sort_values(
            group_keys + ["k"],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # I. Validate top-K financial identities
    # --------------------------------------------------------

    if (
        top_k_concentration["top_k_sector_weight"]
        > top_k_concentration["invested_weight_sum"]
        + weight_tolerance
    ).any():
        raise ValueError(
            "Top-K sector weight exceeds invested capital."
        )

    top_k_changes = (
        top_k_concentration
        .groupby(group_keys)["top_k_sector_weight"]
        .diff()
        .dropna()
    )

    if (top_k_changes < -weight_tolerance).any():
        raise ValueError(
            "Top-K sector concentration is not monotonic."
        )

    # --------------------------------------------------------
    # J. Validate standardized output keys
    # --------------------------------------------------------

    top_k_keys = group_keys + ["k"]

    if concentration.duplicated(group_keys).any():
        raise ValueError(
            "Duplicate sector concentration output keys."
        )

    if top_k_concentration.duplicated(top_k_keys).any():
        raise ValueError(
            "Duplicate top-K sector concentration keys."
        )

    return (
        concentration.sort_values(
            group_keys,
            kind="stable",
        ).reset_index(drop=True),
        top_k_concentration,
    )
