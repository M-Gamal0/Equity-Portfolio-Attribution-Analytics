"""Data contracts: validated research calculations."""

from __future__ import annotations

import json
import numpy as np
import pandas as pd

PORTFOLIO_HOLDINGS_OPTIONAL = ('ticker', 'sector', 'weight_eop_pre_liquidation', 'contribution_total', 'contribution_price')
PORTFOLIO_HOLDINGS_REQUIRED = ('portfolio_id', 'date', 'security_id', 'weight_bop', 'weight_eop', 'return_price', 'return_total')
PORTFOLIO_METADATA_OPTIONAL = ('display_name', 'portfolio_role', 'benchmark_portfolio_id', 'reporting_cutoff', 'base_currency', 'methodology_version', 'methodology_json', 'source_versions_json', 'run_status', 'source_hashes_verified', 'publication_approved', 'code_version', 'calculation_fingerprint', 'source_system', 'source_table', 'source_run_id')
PORTFOLIO_METADATA_REQUIRED = ('portfolio_id', 'return_frequency', 'return_units')
PORTFOLIO_RETURNS_OPTIONAL = ('return_price',)
PORTFOLIO_RETURNS_REQUIRED = ('portfolio_id', 'date', 'return_total')
REBALANCE_TARGETS_OPTIONAL = ('effective_rebalance_date', 'return_base_date', 'ticker', 'sector')
REBALANCE_TARGETS_REQUIRED = ('portfolio_id', 'selection_date', 'security_id', 'weight_target')
SECURITY_RETURNS_OPTIONAL = ('ticker',)
SECURITY_RETURNS_REQUIRED = ('security_id', 'date', 'return_price', 'return_total')

# Research reference: 06_standardized_analytics_data_layer.ipynb, code cell index 3.
def validate_portfolio_returns(
    returns_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate and standardize daily equity portfolio returns.

    Required columns:
        portfolio_id : unique return-series identifier
        date         : daily return interval end date
        return_total : daily simple total return, decimal

    Optional:
        return_price : daily simple price return, decimal

    Returns:
        A validated, chronologically sorted DataFrame.

    This function is independent of the data source.
    """

    required = set(PORTFOLIO_RETURNS_REQUIRED)

    missing = required - set(returns_df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if returns_df.empty:
        raise ValueError(
            "Portfolio returns DataFrame is empty."
        )

    # Preserve only fields belonging to this contract.
    columns = list(PORTFOLIO_RETURNS_REQUIRED)

    columns.extend(
        column
        for column in PORTFOLIO_RETURNS_OPTIONAL
        if column in returns_df.columns
    )

    validated = returns_df[columns].copy()


    # --------------------------------------------------------
    # Validate portfolio identifiers
    # --------------------------------------------------------

    identifiers = validated["portfolio_id"]

    if identifiers.isna().any():
        raise ValueError("Null portfolio identifiers detected.")

    if not identifiers.map(
        lambda value: isinstance(value, str)
        and bool(value.strip())
    ).all():
        raise ValueError(
            "Portfolio identifiers must be nonempty strings."
        )


    # --------------------------------------------------------
    # Normalize and validate observation dates
    # --------------------------------------------------------

    validated["date"] = pd.to_datetime(
        validated["date"],
        errors="raise",
    )

    if validated["date"].isna().any():
        raise ValueError("Null dates detected.")

    if validated["date"].dt.tz is not None:
        raise ValueError(
            "Return dates must be timezone-naive calendar dates."
        )

    if not validated["date"].eq(
        validated["date"].dt.normalize()
    ).all():
        raise ValueError(
            "Return dates must not contain intraday timestamps."
        )


    # --------------------------------------------------------
    # Validate daily return units and numerical values
    # --------------------------------------------------------

    numeric_columns = [
        column
        for column in ("return_total", "return_price")
        if column in validated.columns
    ]

    for column in numeric_columns:

        validated[column] = pd.to_numeric(
            validated[column],
            errors="raise",
        ).astype("float64")

        values = validated[column].to_numpy()

        if not np.isfinite(values).all():
            raise ValueError(
                f"Non-finite values detected in {column}."
            )

        if (values < -1.0).any():
            raise ValueError(
                f"Invalid simple returns below -100% "
                f"detected in {column}."
            )


    # --------------------------------------------------------
    # Enforce standardized primary key
    # --------------------------------------------------------

    primary_key = ["portfolio_id", "date"]

    if validated.duplicated(primary_key).any():
        raise ValueError(
            "Duplicate (portfolio_id, date) records detected."
        )


    # --------------------------------------------------------
    # Return standardized dataset without modifying input
    # --------------------------------------------------------

    return (
        validated
        .sort_values(primary_key)
        .reset_index(drop=True)
    )


# Research reference: 06_standardized_analytics_data_layer.ipynb, code cell index 5.
def validate_portfolio_holdings(
    holdings_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate standardized daily long-only equity holdings.

    Grain:
        One row per portfolio_id, date, security_id.

    Required:
        Stable identifiers, return-interval end date,
        beginning/ending weights, and daily total return.

    Units:
        Weights and simple returns are decimal fractions.

    Notes:
        Cash exposure is permitted through weights summing
        to less than one. Portfolio-level reconciliation
        is performed separately.

    Returns:
        Validated, sorted DataFrame.

    No database or source-specific dependencies.
    """

    # --------------------------------------------------------
    # Validate required fields
    # --------------------------------------------------------

    missing = (
        set(PORTFOLIO_HOLDINGS_REQUIRED)
        - set(holdings_df.columns)
    )

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if holdings_df.empty:
        raise ValueError(
            "Portfolio holdings DataFrame is empty."
        )

    columns = list(PORTFOLIO_HOLDINGS_REQUIRED)

    columns.extend(
        column
        for column in PORTFOLIO_HOLDINGS_OPTIONAL
        if column in holdings_df.columns
    )

    validated = holdings_df[columns].copy()


    # --------------------------------------------------------
    # Validate portfolio and security identifiers
    # --------------------------------------------------------

    for column in ("portfolio_id", "security_id"):

        valid_identifiers = validated[column].map(
            lambda value: (
                isinstance(value, str)
                and bool(value.strip())
                and value == value.strip()
            )
        )

        if not valid_identifiers.all():
            raise ValueError(
                f"Invalid or missing identifiers: {column}"
            )


    # --------------------------------------------------------
    # Normalize and validate dates
    # --------------------------------------------------------

    validated["date"] = pd.to_datetime(
        validated["date"],
        errors="raise",
    )

    if validated["date"].isna().any():
        raise ValueError("Null holding dates detected.")

    if validated["date"].dt.tz is not None:
        raise ValueError(
            "Holding dates must be timezone-naive."
        )

    if not validated["date"].eq(
        validated["date"].dt.normalize()
    ).all():
        raise ValueError(
            "Holding dates must not contain intraday times."
        )


    # --------------------------------------------------------
    # Validate numerical columns
    # --------------------------------------------------------

    weight_columns = [
        column
        for column in (
            "weight_bop",
            "weight_eop",
            "weight_eop_pre_liquidation",
        )
        if column in validated.columns
    ]

    return_columns = [
        column
        for column in (
            "return_total",
            "return_price",
        )
        if column in validated.columns
    ]

    contribution_columns = [
        column
        for column in (
            "contribution_total",
            "contribution_price",
        )
        if column in validated.columns
    ]

    numeric_columns = (
        weight_columns
        + return_columns
        + contribution_columns
    )

    for column in numeric_columns:

        validated[column] = pd.to_numeric(
            validated[column],
            errors="raise",
        ).astype("float64")

        values = validated[column].to_numpy()

        if not np.isfinite(values).all():
            raise ValueError(
                f"Non-finite values detected: {column}"
            )


    # --------------------------------------------------------
    # Enforce long-only position weights
    # --------------------------------------------------------

    for column in weight_columns:

        if not validated[column].between(0.0, 1.0).all():
            raise ValueError(
                f"Invalid long-only weights: {column}"
            )


    # --------------------------------------------------------
    # Validate simple-return boundaries
    # --------------------------------------------------------

    for column in return_columns:

        if (validated[column] < -1.0).any():
            raise ValueError(
                f"Security returns below -100%: {column}"
            )


    # --------------------------------------------------------
    # Validate optional contribution identities
    # --------------------------------------------------------

    contribution_pairs = (
        ("contribution_total", "return_total"),
        ("contribution_price", "return_price"),
    )

    for contribution_column, return_column in contribution_pairs:

        if contribution_column not in validated.columns:
            continue

        if return_column not in validated.columns:
            raise ValueError(
                f"{contribution_column} requires {return_column}."
            )

        expected = (
            validated["weight_bop"].to_numpy()
            * validated[return_column].to_numpy()
        )

        actual = validated[
            contribution_column
        ].to_numpy()

        if not np.isclose(
            actual,
            expected,
            rtol=1e-10,
            atol=1e-12,
        ).all():
            raise ValueError(
                f"Security contribution identity failed: "
                f"{contribution_column}"
            )


    # --------------------------------------------------------
    # Enforce unique security-day observations
    # --------------------------------------------------------

    primary_key = [
        "portfolio_id",
        "date",
        "security_id",
    ]

    if validated.duplicated(primary_key).any():
        raise ValueError(
            "Duplicate portfolio/date/security records."
        )


    # --------------------------------------------------------
    # Return standardized holdings
    # --------------------------------------------------------

    return (
        validated
        .sort_values(primary_key)
        .reset_index(drop=True)
    )


# Research reference: 06_standardized_analytics_data_layer.ipynb, code cell index 7.
def validate_factor_returns(
    factors_df: pd.DataFrame,
    factor_columns: tuple[str, ...],
    require_rf: bool = True,
) -> pd.DataFrame:
    """
    Validate a standardized daily equity factor return matrix.

    Contract:
        One observation per date.

    Required:
        date
        At least one explicitly specified factor column
        rf, if require_rf=True

    Units:
        Daily decimal returns or factor spreads.

    Notes:
        Factor column names are configurable.
        Source metadata and factor version identifiers
        are maintained separately.

    No database or source-specific dependencies.
    """

    # --------------------------------------------------------
    # Validate factor configuration
    # --------------------------------------------------------

    if not isinstance(factors_df, pd.DataFrame):
        raise TypeError(
            "Factor input must be a pandas DataFrame."
        )

    if factors_df.empty:
        raise ValueError(
            "Factor returns DataFrame is empty."
        )

    if factors_df.columns.has_duplicates:
        raise ValueError(
            "Duplicate DataFrame column names detected."
        )

    if not factor_columns:
        raise ValueError(
            "At least one factor column is required."
        )

    if (
        len(set(factor_columns)) != len(factor_columns)
        or any(
            not isinstance(column, str)
            or not column.strip()
            or column in ("date", "rf")
            for column in factor_columns
        )
    ):
        raise ValueError(
            "Factor column configuration is invalid."
        )

    required_columns = [
        "date",
        *factor_columns,
    ]

    if require_rf:
        required_columns.append("rf")

    missing = (
        set(required_columns)
        - set(factors_df.columns)
    )

    if missing:
        raise ValueError(
            f"Missing factor columns: {sorted(missing)}"
        )

    selected_columns = required_columns.copy()

    if not require_rf and "rf" in factors_df.columns:
        selected_columns.append("rf")

    validated = factors_df[
        selected_columns
    ].copy()


    # --------------------------------------------------------
    # Validate and normalize factor observation dates
    # --------------------------------------------------------

    validated["date"] = pd.to_datetime(
        validated["date"],
        errors="raise",
    )

    if validated["date"].isna().any():
        raise ValueError(
            "Missing factor observation dates."
        )

    if validated["date"].dt.tz is not None:
        raise ValueError(
            "Factor dates must be timezone-naive."
        )

    if not validated["date"].eq(
        validated["date"].dt.normalize()
    ).all():
        raise ValueError(
            "Factor dates contain intraday timestamps."
        )

    if validated["date"].duplicated().any():
        raise ValueError(
            "Duplicate factor observation dates detected."
        )


    # --------------------------------------------------------
    # Validate factor returns and risk-free observations
    # --------------------------------------------------------

    numeric_columns = [
        column
        for column in selected_columns
        if column != "date"
    ]

    for column in numeric_columns:

        if pd.api.types.is_bool_dtype(
            validated[column]
        ):
            raise ValueError(
                f"Boolean values are invalid: {column}"
            )

        validated[column] = pd.to_numeric(
            validated[column],
            errors="raise",
        ).astype("float64")

        values = validated[
            column
        ].to_numpy(dtype="float64")

        if not np.isfinite(values).all():
            raise ValueError(
                f"Non-finite factor observations: {column}"
            )


    # --------------------------------------------------------
    # Return standardized factor observation matrix
    # --------------------------------------------------------

    return (
        validated
        .sort_values("date")
        .reset_index(drop=True)
    )


# Research reference: 06_standardized_analytics_data_layer.ipynb, code cell index 11.
def validate_security_returns(
    returns_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate daily long-only equity security-return data.

    Grain:
        One observation per security_id and date.

    Required:
        security_id
        date
        return_price
        return_total

    Optional:
        ticker

    Financial conventions:
        Daily simple returns.
        Decimal fractions.
        Price return excludes dividend adjustments.
        Total return includes applicable adjustments.

    This function does not:
        Connect to a database.
        Calculate returns from prices.
        Infer missing return observations.
        Assess rolling factor-regression eligibility.

    Returns:
        A validated and sorted pandas DataFrame.
    """

    # --------------------------------------------------------
    # Validate DataFrame structure
    # --------------------------------------------------------

    if not isinstance(returns_df, pd.DataFrame):
        raise TypeError(
            "Security returns input must be a pandas DataFrame."
        )

    if returns_df.empty:
        raise ValueError(
            "Security returns DataFrame is empty."
        )

    if returns_df.columns.has_duplicates:
        raise ValueError(
            "Duplicate DataFrame column names detected."
        )

    missing = (
        set(SECURITY_RETURNS_REQUIRED)
        - set(returns_df.columns)
    )

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    selected_columns = list(
        SECURITY_RETURNS_REQUIRED
    )

    selected_columns.extend(
        column
        for column in SECURITY_RETURNS_OPTIONAL
        if column in returns_df.columns
    )

    validated = returns_df[
        selected_columns
    ].copy()


    # --------------------------------------------------------
    # Validate permanent security identifiers
    # --------------------------------------------------------

    valid_identifiers = validated["security_id"].map(
        lambda value: (
            isinstance(value, str)
            and bool(value.strip())
            and value == value.strip()
        )
    )

    if not valid_identifiers.all():
        raise ValueError(
            "Security identifiers must be nonempty strings "
            "without surrounding whitespace."
        )


    # --------------------------------------------------------
    # Validate and normalize observation dates
    # --------------------------------------------------------

    validated["date"] = pd.to_datetime(
        validated["date"],
        errors="raise",
    )

    if validated["date"].isna().any():
        raise ValueError(
            "Missing security-return observation dates."
        )

    if validated["date"].dt.tz is not None:
        raise ValueError(
            "Security-return dates must be timezone-naive."
        )

    if not validated["date"].eq(
        validated["date"].dt.normalize()
    ).all():
        raise ValueError(
            "Security-return dates must be calendar dates."
        )


    # --------------------------------------------------------
    # Validate price and total returns
    # --------------------------------------------------------

    for column in ("return_price", "return_total"):

        if (
            pd.api.types.is_bool_dtype(validated[column])
            or validated[column].map(
                lambda value: isinstance(
                    value, (bool, np.bool_)
                )
            ).any()
        ):
            raise ValueError(
                f"Boolean return values detected: {column}"
            )

        validated[column] = pd.to_numeric(
            validated[column],
            errors="raise",
        ).astype("float64")

        values = validated[
            column
        ].to_numpy(dtype="float64")

        if not np.isfinite(values).all():
            raise ValueError(
                f"Non-finite returns detected: {column}"
            )

        if (values < -1.0).any():
            raise ValueError(
                f"Simple returns below -100%: {column}"
            )


    # --------------------------------------------------------
    # Validate optional display ticker
    # --------------------------------------------------------

    if "ticker" in validated.columns:

        valid_tickers = validated["ticker"].map(
            lambda value: (
                pd.isna(value)
                or (
                    isinstance(value, str)
                    and bool(value.strip())
                )
            )
        )

        if not valid_tickers.all():
            raise ValueError(
                "Invalid optional ticker values detected."
            )


    # --------------------------------------------------------
    # Enforce universal security/date primary key
    # --------------------------------------------------------

    primary_key = [
        "security_id",
        "date",
    ]

    if validated.duplicated(primary_key).any():
        raise ValueError(
            "Duplicate security_id/date observations detected."
        )


    # --------------------------------------------------------
    # Return standardized observations
    # --------------------------------------------------------

    return (
        validated
        .sort_values(primary_key)
        .reset_index(drop=True)
    )


# Research reference: 06_standardized_analytics_data_layer.ipynb, code cell index 12.
def validate_rebalance_targets(
    targets_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate standardized long-only equity rebalance targets.

    Grain:
        One row per portfolio_id, selection_date, security_id.

    Required:
        portfolio_id
        selection_date
        security_id
        weight_target

    Optional:
        effective_rebalance_date
        return_base_date
        ticker
        sector

    Financial conventions:
        Target weights are decimal fractions in [0, 1].
        Zero target weights may represent explicit exits.
        Cash allocations are permitted.

    Missing effective dates are preserved. A missing date
    does not universally imply a pending rebalance.

    Complete target-weight sums and implementation status
    are validated separately using full source datasets.

    No database or project-specific dependencies.
    """

    # --------------------------------------------------------
    # Validate DataFrame structure
    # --------------------------------------------------------

    if not isinstance(targets_df, pd.DataFrame):
        raise TypeError(
            "Rebalance targets must be a pandas DataFrame."
        )

    if targets_df.empty:
        raise ValueError(
            "Rebalance targets DataFrame is empty."
        )

    if targets_df.columns.has_duplicates:
        raise ValueError(
            "Duplicate DataFrame column names detected."
        )

    missing = (
        set(REBALANCE_TARGETS_REQUIRED)
        - set(targets_df.columns)
    )

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    selected_columns = list(
        REBALANCE_TARGETS_REQUIRED
    )

    selected_columns.extend(
        column
        for column in REBALANCE_TARGETS_OPTIONAL
        if column in targets_df.columns
    )

    validated = targets_df[
        selected_columns
    ].copy()


    # --------------------------------------------------------
    # Validate portfolio and security identifiers
    # --------------------------------------------------------

    for column in ("portfolio_id", "security_id"):

        valid_identifiers = validated[column].map(
            lambda value: (
                isinstance(value, str)
                and bool(value.strip())
                and value == value.strip()
            )
        )

        if not valid_identifiers.all():
            raise ValueError(
                f"Invalid identifiers detected: {column}"
            )


    # --------------------------------------------------------
    # Validate selection and effective dates
    # --------------------------------------------------------

    date_columns = [
        "selection_date",
    ]

    date_columns.extend(
        column
        for column in (
            "effective_rebalance_date",
            "return_base_date",
        )
        if column in validated.columns
    )

    for column in date_columns:

        validated[column] = pd.to_datetime(
            validated[column],
            errors="raise",
        )

        if (
            column == "selection_date"
            and validated[column].isna().any()
        ):
            raise ValueError(
                "Selection dates cannot be missing."
            )

        if validated[column].dt.tz is not None:
            raise ValueError(
                f"Timezone-aware dates are invalid: {column}"
            )

        nonnull_dates = validated[column].dropna()

        if not nonnull_dates.eq(
            nonnull_dates.dt.normalize()
        ).all():
            raise ValueError(
                f"Intraday timestamps are invalid: {column}"
            )


    # --------------------------------------------------------
    # Validate effective-date chronology
    # --------------------------------------------------------

    if "effective_rebalance_date" in validated.columns:

        effective_dates = validated[
            "effective_rebalance_date"
        ]

        invalid_chronology = (
            effective_dates.notna()
            & (
                effective_dates
                < validated["selection_date"]
            )
        )

        if invalid_chronology.any():
            raise ValueError(
                "Effective rebalance date precedes "
                "the selection date."
            )


    # --------------------------------------------------------
    # Validate optional return-base chronology
    # --------------------------------------------------------

    if all(
        column in validated.columns
        for column in (
            "return_base_date",
            "effective_rebalance_date",
        )
    ):

        base_dates = validated["return_base_date"]
        effective_dates = validated[
            "effective_rebalance_date"
        ]

        both_present = (
            base_dates.notna()
            & effective_dates.notna()
        )

        invalid_base_dates = (
            both_present
            & (base_dates >= effective_dates)
        )

        if invalid_base_dates.any():
            raise ValueError(
                "Return-base date must precede "
                "the effective rebalance date."
            )


    # --------------------------------------------------------
    # Validate long-only target weights
    # --------------------------------------------------------

    if (
        pd.api.types.is_bool_dtype(
            validated["weight_target"]
        )
        or validated["weight_target"].map(
            lambda value: isinstance(
                value, (bool, np.bool_)
            )
        ).any()
    ):
        raise ValueError(
            "Boolean target weights are invalid."
        )

    validated["weight_target"] = pd.to_numeric(
        validated["weight_target"],
        errors="raise",
    ).astype("float64")

    weights = validated[
        "weight_target"
    ].to_numpy(dtype="float64")

    if not np.isfinite(weights).all():
        raise ValueError(
            "Non-finite target weights detected."
        )

    if ((weights < 0.0) | (weights > 1.0)).any():
        raise ValueError(
            "Long-only target weights must lie in [0, 1]."
        )


    # --------------------------------------------------------
    # Validate optional descriptive fields
    # --------------------------------------------------------

    for column in ("ticker", "sector"):

        if column not in validated.columns:
            continue

        valid_values = validated[column].map(
            lambda value: (
                pd.isna(value)
                or (
                    isinstance(value, str)
                    and bool(value.strip())
                )
            )
        )

        if not valid_values.all():
            raise ValueError(
                f"Invalid descriptive values: {column}"
            )


    # --------------------------------------------------------
    # Enforce unique target security/date observations
    # --------------------------------------------------------

    primary_key = [
        "portfolio_id",
        "selection_date",
        "security_id",
    ]

    if validated.duplicated(primary_key).any():
        raise ValueError(
            "Duplicate portfolio/selection/security targets."
        )


    # --------------------------------------------------------
    # Return standardized targets
    # --------------------------------------------------------

    return (
        validated
        .sort_values(primary_key)
        .reset_index(drop=True)
    )


# Research reference: 06_standardized_analytics_data_layer.ipynb, code cell index 14.
def validate_portfolio_metadata(
    metadata_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate standardized daily equity portfolio metadata.

    Grain:
        One record per unique portfolio_id.

    Required:
        portfolio_id
        return_frequency
        return_units

    Optional:
        Portfolio descriptions, benchmark relationships,
        methodology, source provenance and validation status.

    Conventions:
        return_frequency = 'daily'
        return_units = 'daily_simple_decimal'

    The validator does not:
        Connect to a database.
        Require a benchmark.
        Require a particular research methodology.
        Require publication approval.
        Assume an unverified portfolio currency.

    Returns:
        A validated and sorted pandas DataFrame.
    """

    # --------------------------------------------------------
    # Validate structure
    # --------------------------------------------------------

    if not isinstance(metadata_df, pd.DataFrame):
        raise TypeError(
            "Portfolio metadata must be a pandas DataFrame."
        )

    if metadata_df.empty:
        raise ValueError(
            "Portfolio metadata DataFrame is empty."
        )

    if metadata_df.columns.has_duplicates:
        raise ValueError(
            "Duplicate metadata column names detected."
        )

    missing = (
        set(PORTFOLIO_METADATA_REQUIRED)
        - set(metadata_df.columns)
    )

    if missing:
        raise ValueError(
            f"Missing required metadata columns: {sorted(missing)}"
        )

    columns = list(PORTFOLIO_METADATA_REQUIRED)

    columns.extend(
        column
        for column in PORTFOLIO_METADATA_OPTIONAL
        if column in metadata_df.columns
    )

    validated = metadata_df[columns].copy()


    # --------------------------------------------------------
    # Validate required identifiers and conventions
    # --------------------------------------------------------

    for column in PORTFOLIO_METADATA_REQUIRED:

        valid_values = validated[column].map(
            lambda value: (
                isinstance(value, str)
                and bool(value.strip())
                and value == value.strip()
            )
        )

        if not valid_values.all():
            raise ValueError(
                f"Invalid required metadata values: {column}"
            )

    if not validated["return_frequency"].eq("daily").all():
        raise ValueError(
            "Only daily portfolio returns are supported."
        )

    if not validated["return_units"].eq(
        "daily_simple_decimal"
    ).all():
        raise ValueError(
            "Portfolio returns must use daily simple decimals."
        )


    # --------------------------------------------------------
    # Validate unique portfolio identities
    # --------------------------------------------------------

    if validated["portfolio_id"].duplicated().any():
        raise ValueError(
            "Duplicate portfolio metadata identities."
        )


    # --------------------------------------------------------
    # Validate optional string attributes
    # --------------------------------------------------------

    optional_string_columns = (
        "display_name",
        "portfolio_role",
        "benchmark_portfolio_id",
        "base_currency",
        "methodology_version",
        "run_status",
        "code_version",
        "calculation_fingerprint",
        "source_system",
        "source_table",
        "source_run_id",
    )

    for column in optional_string_columns:

        if column not in validated.columns:
            continue

        valid_values = validated[column].map(
            lambda value: (
                pd.isna(value)
                or (
                    isinstance(value, str)
                    and bool(value.strip())
                    and value == value.strip()
                )
            )
        )

        if not valid_values.all():
            raise ValueError(
                f"Invalid optional metadata values: {column}"
            )


    # --------------------------------------------------------
    # Validate optional reporting cutoff
    # --------------------------------------------------------

    if "reporting_cutoff" in validated.columns:

        validated["reporting_cutoff"] = pd.to_datetime(
            validated["reporting_cutoff"],
            errors="raise",
        )

        if validated["reporting_cutoff"].dt.tz is not None:
            raise ValueError(
                "Reporting cutoffs must be timezone-naive."
            )

        cutoff_values = validated[
            "reporting_cutoff"
        ].dropna()

        if not cutoff_values.eq(
            cutoff_values.dt.normalize()
        ).all():
            raise ValueError(
                "Reporting cutoffs must be calendar dates."
            )


    # --------------------------------------------------------
    # Validate optional Boolean research controls
    # --------------------------------------------------------

    boolean_columns = (
        "source_hashes_verified",
        "publication_approved",
    )

    for column in boolean_columns:

        if column not in validated.columns:
            continue

        valid_values = validated[column].map(
            lambda value: (
                pd.isna(value)
                or isinstance(value, (bool, np.bool_))
            )
        )

        if not valid_values.all():
            raise ValueError(
                f"Invalid Boolean metadata: {column}"
            )


    # --------------------------------------------------------
    # Validate JSON methodology and provenance fields
    # --------------------------------------------------------

    for column in (
        "methodology_json",
        "source_versions_json",
    ):

        if column not in validated.columns:
            continue

        for value in validated[column]:

            if pd.isna(value):
                continue

            if not isinstance(value, str):
                raise ValueError(
                    f"{column} must contain JSON text."
                )

            try:
                parsed = json.loads(value)

            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid JSON in {column}."
                ) from error

            if not isinstance(parsed, dict):
                raise ValueError(
                    f"{column} must contain a JSON object."
                )


    # --------------------------------------------------------
    # Validate optional benchmark relationships
    # --------------------------------------------------------

    if "benchmark_portfolio_id" in validated.columns:

        self_reference = (
            validated["benchmark_portfolio_id"].notna()
            & (
                validated["benchmark_portfolio_id"]
                == validated["portfolio_id"]
            )
        )

        if self_reference.any():
            raise ValueError(
                "A portfolio cannot reference itself "
                "as its benchmark."
            )

    # Benchmark existence is validated separately when
    # multiple portfolio datasets are supplied together.
    # This permits valid standalone external metadata.


    # --------------------------------------------------------
    # Return standardized metadata
    # --------------------------------------------------------

    return (
        validated
        .sort_values("portfolio_id")
        .reset_index(drop=True)
    )
