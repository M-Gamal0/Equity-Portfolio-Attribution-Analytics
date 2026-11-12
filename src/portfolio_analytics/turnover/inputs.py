"""Turnover inputs: validated research calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd



# Research reference: 12_turnover_rebalancing_analytics.ipynb, code cell index 1.
def validate_turnover_inputs(
    holdings_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    *,
    require_sector: bool = False,
    require_full_investment: bool = False,
    weight_tolerance: float = 1e-10,
) -> pd.DataFrame:
    """
    Validate standardized inputs for turnover/rebalancing analytics.

    Parameters
    ----------
    holdings_df : pd.DataFrame
        Security-day holdings containing portfolio_id, date,
        security_id, weight_bop, weight_eop, and return_price.

    targets_df : pd.DataFrame
        Security-selection targets containing portfolio_id,
        selection_date, security_id, and weight_target.

    require_sector : bool
        Require complete sector classifications in both datasets.

    require_full_investment : bool
        Require BOP, EOP, and target weights to sum to one.

    weight_tolerance : float
        Absolute numerical tolerance for weight identities.

    Returns
    -------
    pd.DataFrame
        Validation results with one row per check.

    Notes
    -----
    Weights are decimal fractions.
    Dates represent daily return intervals or selection dates.
    Effective-date availability does not establish execution.
    Undated selections are not automatically classified as pending.

    Raises
    ------
    TypeError
        Inputs are not pandas DataFrames.

    ValueError
        Required fields, keys, dates, numeric values, or
        applicable financial identities are invalid.
    """

    if not isinstance(holdings_df, pd.DataFrame):
        raise TypeError("holdings_df must be a pandas DataFrame.")

    if not isinstance(targets_df, pd.DataFrame):
        raise TypeError("targets_df must be a pandas DataFrame.")

    if (
        not np.isfinite(weight_tolerance)
        or weight_tolerance < 0
    ):
        raise ValueError(
            "weight_tolerance must be finite and nonnegative."
        )

    required_columns = {
        "holdings": {
            "portfolio_id",
            "date",
            "security_id",
            "weight_bop",
            "weight_eop",
            "return_price",
        },
        "targets": {
            "portfolio_id",
            "selection_date",
            "security_id",
            "weight_target",
        },
    }

    datasets = {
        "holdings": holdings_df,
        "targets": targets_df,
    }

    validation_records = []

    def record_check(
        check_id,
        status,
        observations,
        max_abs_error=np.nan,
        detail="",
    ):
        validation_records.append(
            {
                "check_id": check_id,
                "status": status,
                "observations": int(observations),
                "max_abs_error": max_abs_error,
                "detail": detail,
            }
        )

    # --------------------------------------------------------
    # 2. Validate required fields, identifiers, and unique keys
    # --------------------------------------------------------

    key_definitions = {
        "holdings": [
            "portfolio_id",
            "date",
            "security_id",
        ],
        "targets": [
            "portfolio_id",
            "selection_date",
            "security_id",
        ],
    }

    for dataset_name, frame in datasets.items():

        if frame.empty:
            raise ValueError(
                f"{dataset_name}: input DataFrame is empty."
            )

        missing_columns = required_columns[
            dataset_name
        ].difference(frame.columns)

        if missing_columns:
            raise ValueError(
                f"{dataset_name}: missing required columns "
                f"{sorted(missing_columns)}"
            )

        keys = key_definitions[dataset_name]

        if frame[keys].isna().any().any():
            raise ValueError(
                f"{dataset_name}: null primary-key values."
            )

        for identifier in ("portfolio_id", "security_id"):

            blank_identifiers = (
                frame[identifier]
                .astype("string")
                .str.strip()
                .eq("")
                .fillna(False)
            )

            if blank_identifiers.any():
                raise ValueError(
                    f"{dataset_name}: blank {identifier} values."
                )

        duplicates = frame.duplicated(
            subset=keys,
            keep=False,
        )

        if duplicates.any():
            raise ValueError(
                f"{dataset_name}: "
                f"{int(duplicates.sum())} duplicate-key rows."
            )

        record_check(
            f"{dataset_name}_primary_keys",
            "PASSED",
            len(frame),
        )

    holdings_ids = set(holdings_df["portfolio_id"])
    target_ids = set(targets_df["portfolio_id"])

    if holdings_ids != target_ids:
        raise ValueError(
            "Holdings and targets contain different "
            "portfolio ID sets."
        )

    record_check(
        "portfolio_identity_consistency",
        "PASSED",
        len(holdings_ids),
    )

    # --------------------------------------------------------
    # 3. Validate date fields
    # --------------------------------------------------------

    for dataset_name, date_column in (
        ("holdings", "date"),
        ("targets", "selection_date"),
    ):

        parsed_dates = pd.to_datetime(
            datasets[dataset_name][date_column],
            errors="raise",
        )

        if parsed_dates.isna().any():
            raise ValueError(
                f"{dataset_name}: invalid or missing dates."
            )

        record_check(
            f"{dataset_name}_dates",
            "PASSED",
            len(parsed_dates),
        )

    # --------------------------------------------------------
    # 4. Validate numeric values and long-only weights
    # --------------------------------------------------------

    weight_columns = {
        "holdings": [
            "weight_bop",
            "weight_eop",
        ],
        "targets": [
            "weight_target",
        ],
    }

    if "weight_eop_pre_liquidation" in holdings_df.columns:
        weight_columns["holdings"].append(
            "weight_eop_pre_liquidation"
        )

    for dataset_name, columns in weight_columns.items():

        frame = datasets[dataset_name]

        for column in columns:

            if not pd.api.types.is_numeric_dtype(
                frame[column]
            ):
                raise ValueError(
                    f"{dataset_name}.{column} must be numeric."
                )

            values = frame[column].to_numpy(
                dtype=float,
                na_value=np.nan,
            )

            if not np.isfinite(values).all():
                raise ValueError(
                    f"{dataset_name}.{column}: "
                    "nonfinite or missing weights."
                )

            invalid_bounds = (
                (values < -weight_tolerance)
                | (values > 1.0 + weight_tolerance)
            )

            if invalid_bounds.any():
                raise ValueError(
                    f"{dataset_name}.{column}: "
                    "weights outside long-only bounds."
                )

            record_check(
                f"{dataset_name}_{column}_bounds",
                "PASSED",
                len(values),
            )

    price_returns = pd.to_numeric(
        holdings_df["return_price"],
        errors="raise",
    ).to_numpy(
        dtype=float,
        na_value=np.nan,
    )

    if not np.isfinite(price_returns).all():
        raise ValueError(
            "Holdings contain missing or nonfinite price returns."
        )

    if (price_returns < -1.0).any():
        raise ValueError(
            "Holdings contain price returns below -100%."
        )

    record_check(
        "holdings_price_returns",
        "PASSED",
        len(price_returns),
    )

    # --------------------------------------------------------
    # 5. Check optional full-investment identities
    # --------------------------------------------------------

    investment_checks = [
        (
            "holdings_bop_full_investment",
            holdings_df,
            ["portfolio_id", "date"],
            "weight_bop",
        ),
        (
            "holdings_eop_full_investment",
            holdings_df,
            ["portfolio_id", "date"],
            "weight_eop",
        ),
        (
            "targets_full_investment",
            targets_df,
            ["portfolio_id", "selection_date"],
            "weight_target",
        ),
    ]

    for (
        check_id,
        frame,
        group_columns,
        weight_column,
    ) in investment_checks:

        if not require_full_investment:

            record_check(
                check_id,
                "NOT_ASSESSED",
                0,
                detail="Full investment was not required.",
            )

            continue

        grouped_weights = (
            frame
            .groupby(group_columns)[weight_column]
            .sum()
        )

        absolute_errors = (
            grouped_weights - 1.0
        ).abs()

        max_error = float(
            absolute_errors.max()
        )

        if max_error > weight_tolerance:
            raise ValueError(
                f"{check_id}: maximum deviation "
                f"{max_error:.12g} exceeds "
                f"tolerance {weight_tolerance:.12g}."
            )

        record_check(
            check_id,
            "PASSED",
            len(grouped_weights),
            max_abs_error=max_error,
        )

    # --------------------------------------------------------
    # 6. Check effective-date consistency by selection event
    # --------------------------------------------------------

    if "effective_rebalance_date" in targets_df.columns:

        effective_dates = pd.to_datetime(
            targets_df["effective_rebalance_date"],
            errors="raise",
        )

        selection_dates = pd.to_datetime(
            targets_df["selection_date"],
            errors="raise",
        )

        dated_mask = effective_dates.notna()

        invalid_chronology = (
            dated_mask
            & (effective_dates <= selection_dates)
        )

        if invalid_chronology.any():
            raise ValueError(
                "Effective rebalance date must be strictly "
                "later than its selection date."
            )

        event_groups = (
            targets_df
            .groupby(
                ["portfolio_id", "selection_date"]
            )["effective_rebalance_date"]
            .agg(
                rows="size",
                dated_rows="count",
                distinct_dates="nunique",
            )
        )

        mixed_dating = (
            event_groups["dated_rows"].gt(0)
            & event_groups["dated_rows"].ne(
                event_groups["rows"]
            )
        )

        multiple_dates = (
            event_groups["distinct_dates"] > 1
        )

        if mixed_dating.any() or multiple_dates.any():
            raise ValueError(
                "Inconsistent effective dates within "
                "one rebalance selection event."
            )

        dated_events = int(
            event_groups["dated_rows"].gt(0).sum()
        )

        undated_events = int(
            event_groups["dated_rows"].eq(0).sum()
        )

        record_check(
            "rebalance_effective_date_consistency",
            "PASSED",
            len(event_groups),
            detail=(
                f"{dated_events} dated events; "
                f"{undated_events} undated events."
            ),
        )

    else:

        record_check(
            "rebalance_effective_date_consistency",
            "NOT_ASSESSED",
            0,
            detail="Effective rebalance date field absent.",
        )

    # --------------------------------------------------------
    # 7. Check sector availability when required
    # --------------------------------------------------------

    if require_sector:

        for dataset_name, frame in datasets.items():

            if "sector" not in frame.columns:
                raise ValueError(
                    f"{dataset_name}: sector field required."
                )

            invalid_sectors = (
                frame["sector"].isna()
                | frame["sector"]
                    .astype("string")
                    .str.strip()
                    .eq("")
                    .fillna(False)
            )

            if invalid_sectors.any():
                raise ValueError(
                    f"{dataset_name}: missing/blank sectors."
                )

            record_check(
                f"{dataset_name}_sector_coverage",
                "PASSED",
                len(frame),
            )

    else:

        record_check(
            "sector_coverage",
            "NOT_ASSESSED",
            0,
            detail="Sector coverage was not required.",
        )

    # --------------------------------------------------------
    # 8. Record availability of optional event diagnostics
    # --------------------------------------------------------

    if "weight_eop_pre_liquidation" in holdings_df.columns:

        record_check(
            "pre_liquidation_weight_availability",
            "PASSED",
            len(holdings_df),
            detail="Field present; weight bounds validated.",
        )

    else:

        record_check(
            "pre_liquidation_weight_availability",
            "NOT_ASSESSED",
            0,
            detail="Pre-liquidation EOP weights unavailable.",
        )

    if "return_base_date" in targets_df.columns:

        record_check(
            "return_base_date_availability",
            "AVAILABLE_NOT_YET_RECONCILED",
            int(targets_df["return_base_date"].notna().sum()),
            detail="Field present; event alignment is not yet tested.",
        )

    else:

        record_check(
            "return_base_date_availability",
            "NOT_ASSESSED",
            0,
            detail="Return-base date field absent.",
        )

    # --------------------------------------------------------
    # 9. Return standardized validation records
    # --------------------------------------------------------

    return pd.DataFrame.from_records(
        validation_records,
        columns=[
            "check_id",
            "status",
            "observations",
            "max_abs_error",
            "detail",
        ],
    )
