"""Performance contributions: validated research calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd



# Research reference: 07_performance_analytics.ipynb, code cell index 14.
def compute_security_contributions(
    holdings_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate daily absolute security return contributions.

    Required fields:
        portfolio_id, date, security_id,
        weight_bop, return_total

    Optional fields:
        ticker, sector, return_price

    Each portfolio/date/security observation contributes:

        weight_bop * security_daily_return

    The function does not require portfolio-level returns
    or previously calculated analytical outputs.
    """

    if not isinstance(holdings_df, pd.DataFrame):
        raise TypeError("holdings_df must be a DataFrame.")

    required = [
        "portfolio_id",
        "date",
        "security_id",
        "weight_bop",
        "return_total",
    ]

    missing = set(required).difference(holdings_df.columns)

    if missing or holdings_df.empty:
        raise ValueError(
            f"Empty holdings or missing columns: {sorted(missing)}"
        )

    if not holdings_df.columns.is_unique:
        raise ValueError("Duplicate holdings column names.")

    optional = [
        column
        for column in ("ticker", "sector", "return_price")
        if column in holdings_df.columns
    ]

    data = holdings_df[required + optional].copy()

    if data[required].isna().any().any():
        raise ValueError("Null required holdings fields.")

    data["date"] = pd.to_datetime(
        data["date"],
        errors="raise",
    )

    if (
        data["date"].dt.tz is not None
        or not data["date"].eq(
            data["date"].dt.normalize()
        ).all()
    ):
        raise ValueError(
            "Holdings dates must be timezone-naive dates."
        )

    keys = ["portfolio_id", "date", "security_id"]

    if data.duplicated(keys).any():
        raise ValueError("Duplicate security-day holdings keys.")

    numeric_columns = ["weight_bop", "return_total"]

    if "return_price" in data.columns:
        numeric_columns.append("return_price")

    for column in numeric_columns:

        if not pd.api.types.is_numeric_dtype(data[column]):
            raise TypeError(f"{column} must be numeric.")

        values = data[column].to_numpy(dtype=np.float64)

        if not np.isfinite(values).all():
            raise ValueError(
                f"{column} contains non-finite values."
            )

    weights = data["weight_bop"].to_numpy(
        dtype=np.float64
    )

    if (weights < 0.0).any():
        raise ValueError(
            "Negative beginning-of-period weights detected."
        )

    for return_column in (
        "return_total",
        "return_price",
    ):

        if return_column not in data.columns:
            continue

        returns = data[return_column].to_numpy(
            dtype=np.float64
        )

        if (returns < -1.0).any():
            raise ValueError(
                f"{return_column} contains a return below -100%."
            )

        contribution_column = (
            "contribution_total"
            if return_column == "return_total"
            else "contribution_price"
        )

        data[contribution_column] = (
            weights * returns
        )

    return (
        data
        .sort_values(keys)
        .reset_index(drop=True)
    )


# Research reference: 07_performance_analytics.ipynb, code cell index 15.
def link_security_contributions(
    daily_contributions_df: pd.DataFrame,
    portfolio_returns_df: pd.DataFrame,
    window_definitions_df: pd.DataFrame,
    *,
    tolerance: float = 1e-12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Link absolute security contributions over reporting windows.

    Required daily contribution columns:
        portfolio_id, date, security_id, contribution_total

    Required portfolio return columns:
        portfolio_id, date, return_total

    Optional paired price columns:
        contribution_price, return_price

    The function uses only its supplied DataFrames.

    Returns
    -------
    security_period_df:
        Linked security-level return contributions.

    reconciliation_df:
        One record per reporting window, including
        compounded returns and linking residuals.

    Notes
    -----
    Security contributions are not Brinson attribution.

    Summed unlinked daily contributions are retained
    for diagnostics, not used as multi-period attribution.
    """

    contribution_required = {
        "portfolio_id",
        "date",
        "security_id",
        "contribution_total",
    }

    returns_required = {
        "portfolio_id",
        "date",
        "return_total",
    }

    window_required = {
        "portfolio_id",
        "window_id",
        "as_of_date",
        "resolved_start_date",
        "first_return_date",
        "last_return_date",
        "observation_count",
        "start_anchor_source",
        "window_status",
    }

    inputs = [
        (
            daily_contributions_df,
            contribution_required,
            "daily_contributions_df",
        ),
        (
            portfolio_returns_df,
            returns_required,
            "portfolio_returns_df",
        ),
        (
            window_definitions_df,
            window_required,
            "window_definitions_df",
        ),
    ]

    for dataframe, required, name in inputs:

        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(f"{name} must be a DataFrame.")

        missing = required.difference(dataframe.columns)

        if missing:
            raise ValueError(
                f"{name}: missing {sorted(missing)}"
            )

        if dataframe.empty:
            raise ValueError(f"{name} is empty.")

        if not dataframe.columns.is_unique:
            raise ValueError(
                f"{name} contains duplicate column names."
            )

    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be positive and finite.")

    has_price_contributions = (
        "contribution_price" in daily_contributions_df.columns
    )

    has_price_returns = (
        "return_price" in portfolio_returns_df.columns
    )

    if has_price_contributions != has_price_returns:
        raise ValueError(
            "Price-return linking requires both "
            "contribution_price and return_price."
        )

    price_enabled = (
        has_price_contributions and has_price_returns
    )

    descriptor_columns = [
        column
        for column in ("ticker", "sector")
        if column in daily_contributions_df.columns
    ]

    contribution_columns = ["contribution_total"]
    return_columns = ["return_total"]

    if price_enabled:
        contribution_columns.append("contribution_price")
        return_columns.append("return_price")

    contributions = daily_contributions_df[
        [
            "portfolio_id",
            "date",
            "security_id",
            *contribution_columns,
            *descriptor_columns,
        ]
    ].copy()

    returns = portfolio_returns_df[
        [
            "portfolio_id",
            "date",
            *return_columns,
        ]
    ].copy()

    windows = window_definitions_df[
        list(window_required)
    ].copy()

    for dataframe in (contributions, returns):
        dataframe["date"] = pd.to_datetime(
            dataframe["date"],
            errors="raise",
        )

    for column in (
        "as_of_date",
        "resolved_start_date",
        "first_return_date",
        "last_return_date",
    ):
        windows[column] = pd.to_datetime(
            windows[column],
            errors="raise",
        )

    security_keys = [
        "portfolio_id",
        "date",
        "security_id",
    ]

    daily_keys = [
        "portfolio_id",
        "date",
    ]

    window_keys = [
        "portfolio_id",
        "as_of_date",
        "window_id",
    ]

    if contributions.duplicated(security_keys).any():
        raise ValueError("Duplicate security-day contributions.")

    if returns.duplicated(daily_keys).any():
        raise ValueError("Duplicate portfolio-day returns.")

    if windows.duplicated(window_keys).any():
        raise ValueError("Duplicate reporting-window definitions.")

    for dataframe, fields in (
        (contributions, contribution_columns),
        (returns, return_columns),
    ):

        if dataframe[
            ["portfolio_id", "date", *fields]
        ].isna().any().any():
            raise ValueError("Null required return fields.")

        values = dataframe[fields].to_numpy(
            dtype=np.float64
        )

        if not np.isfinite(values).all():
            raise ValueError(
                "Non-finite return or contribution values."
            )

    if (
        returns[return_columns].to_numpy(
            dtype=np.float64
        ) < -1.0
    ).any():
        raise ValueError(
            "Simple portfolio returns cannot be below -100%."
        )

    contributions = contributions.sort_values(
        security_keys
    )

    returns = returns.sort_values(daily_keys)

    # --------------------------------------------------------
    # Confirm that supplied daily contributions are
    # consistent with source portfolio returns.
    # --------------------------------------------------------

    daily_sums = (
        contributions
        .groupby(daily_keys, as_index=False)[
            contribution_columns
        ]
        .sum()
    )

    daily_comparison = returns.merge(
        daily_sums,
        on=daily_keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    if not daily_comparison["_merge"].eq("both").all():
        raise ValueError(
            "Contribution dates do not match return dates."
        )

    for basis in (
        ("total", "price")
        if price_enabled
        else ("total",)
    ):

        discrepancy = (
            daily_comparison[f"contribution_{basis}"]
            - daily_comparison[f"return_{basis}"]
        )

        if discrepancy.abs().max() > tolerance:
            raise ValueError(
                f"Daily {basis} contributions do not reconcile."
            )

    # --------------------------------------------------------
    # Independently link each portfolio and window.
    # --------------------------------------------------------

    security_results = []
    reconciliation_results = []

    for portfolio_id, portfolio_data in returns.groupby(
        "portfolio_id",
        sort=True,
    ):

        portfolio_data = (
            portfolio_data
            .sort_values("date")
            .reset_index(drop=True)
        )

        portfolio_contributions = contributions.loc[
            contributions["portfolio_id"].eq(portfolio_id)
        ]

        portfolio_windows = windows.loc[
            windows["portfolio_id"].eq(portfolio_id)
        ]

        if portfolio_windows.empty:
            continue

        dates = pd.DatetimeIndex(
            portfolio_data["date"]
        )

        # Wealth paths are calculated once per portfolio.

        wealth_paths = {}

        for basis in (
            ("total", "price")
            if price_enabled
            else ("total",)
        ):

            daily_returns = portfolio_data[
                f"return_{basis}"
            ].to_numpy(dtype=np.float64)

            wealth_eop = np.cumprod(
                1.0 + daily_returns,
                dtype=np.float64,
            )

            wealth_bop = np.concatenate(
                ([1.0], wealth_eop[:-1])
            )

            if not np.isfinite(wealth_eop).all():
                raise ValueError(
                    f"Non-finite {basis} wealth path."
                )

            wealth_paths[basis] = (
                wealth_bop,
                wealth_eop,
            )

        for _, window in portfolio_windows.iterrows():

            as_of = window["as_of_date"]
            start = window["resolved_start_date"]
            window_id = window["window_id"]

            summary = {
                "portfolio_id": portfolio_id,
                "as_of_date": as_of,
                "window_id": window_id,
                "resolved_start_date": start,
                "observation_count": (
                    window["observation_count"]
                ),
                "security_count": 0,
                "calculation_status": "WINDOW_UNAVAILABLE",
            }

            for basis in wealth_paths:
                summary[
                    f"compounded_return_{basis}"
                ] = np.nan

                summary[
                    f"linked_contribution_sum_{basis}"
                ] = np.nan

                summary[
                    f"linking_residual_{basis}"
                ] = np.nan

            if window["window_status"] != "AVAILABLE":
                reconciliation_results.append(summary)
                continue

            end = window["last_return_date"]

            if (
                pd.isna(start)
                or pd.isna(end)
                or start >= end
                or end != as_of
            ):
                raise ValueError(
                    f"Invalid {portfolio_id}/{window_id} dates."
                )

            first_position = int(
                dates.searchsorted(start, side="right")
            )

            end_position = int(
                dates.searchsorted(end, side="right")
            )

            if (
                end_position == 0
                or dates[end_position - 1] != end
                or end_position <= first_position
            ):
                raise ValueError(
                    f"Invalid {window_id} return interval."
                )

            anchor_source = window["start_anchor_source"]

            if anchor_source == "OBSERVED_RETURN_DATE":

                if (
                    first_position == 0
                    or dates[first_position - 1] != start
                ):
                    raise ValueError(
                        f"Invalid {window_id} observed anchor."
                    )

            elif anchor_source == "EXPLICIT_INCEPTION_BASELINE":

                if (
                    first_position != 0
                    or start >= dates[0]
                ):
                    raise ValueError(
                        f"Invalid {window_id} inception baseline."
                    )

            else:
                raise ValueError(
                    f"Unsupported {window_id} anchor source."
                )

            expected_dates = dates[
                first_position:end_position
            ]

            if (
                len(expected_dates)
                != int(window["observation_count"])
                or expected_dates[0]
                != window["first_return_date"]
            ):
                raise ValueError(
                    f"Incorrect {window_id} observation coverage."
                )

            window_contributions = (
                portfolio_contributions.loc[
                    portfolio_contributions["date"].gt(start)
                    & portfolio_contributions["date"].le(end)
                ].copy()
            )

            observed_dates = pd.DatetimeIndex(
                window_contributions["date"].unique()
            ).sort_values()

            if not observed_dates.equals(expected_dates):
                raise ValueError(
                    f"Missing security-day coverage in {window_id}."
                )

            # --------------------------------------------
            # Normalize each wealth path at window start.
            # --------------------------------------------

            multiplier_data = {
                "date": expected_dates,
            }

            valid_start_wealth = True

            for basis, (
                wealth_bop,
                wealth_eop,
            ) in wealth_paths.items():

                start_wealth = (
                    wealth_eop[first_position - 1]
                    if first_position > 0
                    else 1.0
                )

                if start_wealth <= 0.0:
                    valid_start_wealth = False
                    break

                multiplier_data[
                    f"wealth_multiplier_{basis}"
                ] = (
                    wealth_bop[
                        first_position:end_position
                    ]
                    / start_wealth
                )

                summary[
                    f"compounded_return_{basis}"
                ] = (
                    wealth_eop[end_position - 1]
                    / start_wealth
                    - 1.0
                )

            if not valid_start_wealth:
                summary["calculation_status"] = (
                    "UNAVAILABLE_ZERO_START_WEALTH"
                )
                reconciliation_results.append(summary)
                continue

            multipliers = pd.DataFrame(
                multiplier_data
            )

            linked_daily = window_contributions.merge(
                multipliers,
                on="date",
                how="left",
                validate="many_to_one",
            )

            aggregation = {
                "security_day_observations": (
                    "date", "size"
                ),
                "first_security_date": ("date", "min"),
                "last_security_date": ("date", "max"),
            }

            for basis in wealth_paths:

                linked_column = (
                    f"linked_contribution_{basis}"
                )

                linked_daily[linked_column] = (
                    linked_daily[
                        f"contribution_{basis}"
                    ]
                    * linked_daily[
                        f"wealth_multiplier_{basis}"
                    ]
                )

                aggregation[linked_column] = (
                    linked_column, "sum"
                )

                aggregation[
                    f"unlinked_contribution_{basis}"
                ] = (
                    f"contribution_{basis}", "sum"
                )

            security_summary = (
                linked_daily
                .groupby("security_id", as_index=False)
                .agg(**aggregation)
            )

            # Use the security's latest observed descriptor
            # within this window; never look beyond as_of.

            if descriptor_columns:

                descriptors = (
                    window_contributions
                    .sort_values("date")
                    .drop_duplicates(
                        "security_id",
                        keep="last",
                    )[
                        ["security_id", *descriptor_columns]
                    ]
                )

                security_summary = security_summary.merge(
                    descriptors,
                    on="security_id",
                    how="left",
                    validate="one_to_one",
                )

            security_summary.insert(
                0, "portfolio_id", portfolio_id
            )

            security_summary.insert(
                1, "as_of_date", as_of
            )

            security_summary.insert(
                2, "window_id", window_id
            )

            security_summary[
                "resolved_start_date"
            ] = start

            security_summary[
                "contribution_units"
            ] = "simple_decimal_of_window_start_wealth"

            # --------------------------------------------
            # Verify the defining linking identity.
            # --------------------------------------------

            for basis in wealth_paths:

                observed_sum = float(
                    security_summary[
                        f"linked_contribution_{basis}"
                    ].sum()
                )

                expected_return = summary[
                    f"compounded_return_{basis}"
                ]

                residual = (
                    observed_sum - expected_return
                )

                if (
                    not np.isfinite(residual)
                    or abs(residual) > tolerance
                ):
                    raise AssertionError(
                        f"{portfolio_id}/{window_id}: "
                        f"{basis} linking residual "
                        f"{residual:.3e}"
                    )

                summary[
                    f"linked_contribution_sum_{basis}"
                ] = observed_sum

                summary[
                    f"linking_residual_{basis}"
                ] = residual

            summary["security_count"] = (
                len(security_summary)
            )

            summary["calculation_status"] = "VALIDATED"

            security_results.append(security_summary)
            reconciliation_results.append(summary)

    if not reconciliation_results:
        raise ValueError(
            "No matching portfolio windows found."
        )

    security_period_df = (
        pd.concat(
            security_results,
            ignore_index=True,
        )
        if security_results
        else pd.DataFrame()
    )

    reconciliation_df = (
        pd.DataFrame(reconciliation_results)
        .sort_values(
            ["portfolio_id", "as_of_date", "window_id"]
        )
        .reset_index(drop=True)
    )

    if not security_period_df.empty:

        keys = [
            "portfolio_id",
            "as_of_date",
            "window_id",
            "security_id",
        ]

        if security_period_df.duplicated(keys).any():
            raise AssertionError(
                "Duplicate linked security contribution keys."
            )

        security_period_df = (
            security_period_df
            .sort_values(keys)
            .reset_index(drop=True)
        )

    return security_period_df, reconciliation_df


# Research reference: 07_performance_analytics.ipynb, code cell index 16.
def aggregate_sector_contributions(
    linked_security_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate linked security contributions by sector."""

    required = {
        "portfolio_id",
        "as_of_date",
        "window_id",
        "security_id",
        "linked_contribution_total",
    }

    missing = required.difference(linked_security_df.columns)

    if missing or linked_security_df.empty:
        raise ValueError(
            f"Missing linked contribution fields: {sorted(missing)}"
        )

    data = linked_security_df.copy()

    keys = [
        "portfolio_id",
        "as_of_date",
        "window_id",
    ]

    if data.duplicated(keys + ["security_id"]).any():
        raise ValueError(
            "Duplicate security contribution window keys."
        )

    has_price = "linked_contribution_price" in data.columns

    numeric_columns = ["linked_contribution_total"]

    if has_price:
        numeric_columns.append("linked_contribution_price")

    if not np.isfinite(
        data[numeric_columns].to_numpy(dtype=float)
    ).all():
        raise ValueError("Non-finite linked contributions.")

    # Preserve the source's existing classification.
    # Missing/blank labels receive a reporting-only
    # category; source observations are not modified.

    if "sector" not in data.columns:
        data["sector"] = pd.NA

    missing_sector = (
        data["sector"].isna()
        | data["sector"].map(
            lambda value: (
                isinstance(value, str)
                and not value.strip()
            )
        )
    )

    data["sector_reporting_label"] = (
        data["sector"]
        .astype(object)
        .where(
            ~missing_sector,
            "UNCLASSIFIED_SOURCE_MISSING",
        )
    )

    data["sector_label_status"] = np.where(
        missing_sector,
        "SOURCE_LABEL_MISSING",
        "SOURCE_LABEL_AVAILABLE",
    )

    data["positive_linked_contribution"] = (
        data["linked_contribution_total"].clip(lower=0)
    )

    data["negative_linked_contribution"] = (
        data["linked_contribution_total"].clip(upper=0)
    )

    aggregation = {
        "security_count": ("security_id", "nunique"),
        "linked_contribution_total": (
            "linked_contribution_total", "sum"
        ),
        "positive_security_contributions": (
            "positive_linked_contribution", "sum"
        ),
        "negative_security_contributions": (
            "negative_linked_contribution", "sum"
        ),
    }

    if has_price:
        aggregation["linked_contribution_price"] = (
            "linked_contribution_price", "sum"
        )

    group_keys = keys + [
        "sector_reporting_label",
        "sector_label_status",
    ]

    sectors = (
        data
        .groupby(
            group_keys,
            as_index=False,
            dropna=False,
            sort=True,
        )
        .agg(**aggregation)
    )

    # Deterministic largest-contributor-first sector ranking.

    sectors = sectors.sort_values(
        keys + [
            "linked_contribution_total",
            "sector_reporting_label",
        ],
        ascending=[True, True, True, False, True],
        kind="mergesort",
    )

    sectors["sector_rank_descending"] = (
        sectors.groupby(keys, sort=False).cumcount() + 1
    )

    sectors["contribution_units"] = (
        "simple_decimal_of_window_start_wealth"
    )

    return sectors.reset_index(drop=True)


# Research reference: 07_performance_analytics.ipynb, code cell index 16.
def rank_linked_security_contributions(
    linked_security_df: pd.DataFrame,
) -> pd.DataFrame:
    """Rank security contributions within each portfolio window."""

    keys = [
        "portfolio_id",
        "as_of_date",
        "window_id",
    ]

    required = set(
        keys + [
            "security_id",
            "linked_contribution_total",
        ]
    )

    if not required.issubset(linked_security_df.columns):
        raise ValueError("Missing security ranking fields.")

    ranked = linked_security_df.copy()

    if ranked.duplicated(keys + ["security_id"]).any():
        raise ValueError("Duplicate security ranking keys.")

    if not np.isfinite(
        ranked["linked_contribution_total"].to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError("Non-finite security contributions.")

    # Sort security IDs before ranking so ties remain
    # deterministic, irrespective of original row order.

    ranked = ranked.sort_values(
        keys + ["security_id"],
        kind="mergesort",
    )

    grouped = ranked.groupby(
        keys,
        sort=False,
    )["linked_contribution_total"]

    ranked["rank_largest_contributor"] = (
        grouped.rank(
            method="first",
            ascending=False,
        ).astype("int64")
    )

    ranked["rank_smallest_contributor"] = (
        grouped.rank(
            method="first",
            ascending=True,
        ).astype("int64")
    )

    ranked["contribution_direction"] = np.select(
        [
            ranked["linked_contribution_total"].gt(0),
            ranked["linked_contribution_total"].lt(0),
        ],
        [
            "POSITIVE",
            "NEGATIVE",
        ],
        default="ZERO",
    )

    return (
        ranked
        .sort_values(
            keys + ["rank_largest_contributor"]
        )
        .reset_index(drop=True)
    )
