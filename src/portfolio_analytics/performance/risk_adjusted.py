"""Performance risk adjusted: validated research calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd



# Research reference: 07_performance_analytics.ipynb, code cell index 12.
def align_performance_risk_free(
    aligned_returns_df: pd.DataFrame,
    risk_free_df: pd.DataFrame,
    *,
    rf_column: str = "rf",
) -> pd.DataFrame:
    """Left-align daily risk-free returns without dropping dates."""

    keys = [
        "portfolio_id",
        "benchmark_portfolio_id",
        "date",
    ]

    return_columns = [
        "portfolio_return_total",
        "benchmark_return_total",
    ]

    required_daily = set(keys + return_columns)
    required_rf = {"date", rf_column}

    if not required_daily.issubset(aligned_returns_df.columns):
        raise ValueError("Missing aligned-return fields.")

    if not required_rf.issubset(risk_free_df.columns):
        raise ValueError("Missing risk-free fields.")

    daily = aligned_returns_df[
        keys + return_columns
    ].copy()

    rf = risk_free_df[
        ["date", rf_column]
    ].copy()

    daily["date"] = pd.to_datetime(daily["date"])
    rf["date"] = pd.to_datetime(rf["date"])

    if daily[keys].isna().any().any():
        raise ValueError("Null aligned-return keys.")

    if rf["date"].isna().any():
        raise ValueError("Null risk-free dates.")

    if daily.duplicated(keys).any():
        raise ValueError("Duplicate aligned-return keys.")

    if rf["date"].duplicated().any():
        raise ValueError("Duplicate risk-free dates.")

    if any(
        not pd.api.types.is_numeric_dtype(frame[column])
        for frame, column in (
            (daily, return_columns[0]),
            (daily, return_columns[1]),
            (rf, rf_column),
        )
    ):
        raise TypeError("Returns and RF must be numeric.")

    for column in return_columns:
        values = daily[column].to_numpy(dtype=float)

        if not np.isfinite(values).all() or (values < -1).any():
            raise ValueError(f"Invalid {column} values.")

    observed_rf = rf[rf_column].dropna().to_numpy(
        dtype=float
    )

    if (
        not np.isfinite(observed_rf).all()
        or (observed_rf < -1).any()
    ):
        raise ValueError("Invalid observed risk-free returns.")

    rf = rf.rename(
        columns={rf_column: "risk_free_return"}
    )

    result = daily.merge(
        rf,
        on="date",
        how="left",
        validate="many_to_one",
        sort=False,
    )

    if len(result) != len(daily):
        raise AssertionError(
            "Risk-free alignment changed the portfolio row count."
        )

    result = (
        result
        .sort_values(keys)
        .reset_index(drop=True)
    )

    result["risk_free_available"] = (
        result["risk_free_return"].notna()
    )

    result["portfolio_excess_return"] = (
        result["portfolio_return_total"]
        - result["risk_free_return"]
    )

    result["benchmark_excess_return"] = (
        result["benchmark_return_total"]
        - result["risk_free_return"]
    )

    result["arithmetic_active_return"] = (
        result["portfolio_return_total"]
        - result["benchmark_return_total"]
    )

    return result


# Research reference: 07_performance_analytics.ipynb, code cell index 12.
def assess_performance_rf_coverage(
    rf_aligned_df: pd.DataFrame,
    windows_df: pd.DataFrame,
) -> pd.DataFrame:
    """Assess RF coverage over matched reporting windows."""

    required = {
        "portfolio_id",
        "window_id",
        "as_of_date",
        "resolved_start_date",
        "last_return_date",
        "observation_count",
        "window_status",
    }

    if not required.issubset(windows_df.columns):
        raise ValueError("Missing reporting-window fields.")

    windows = windows_df.copy()

    for column in (
        "as_of_date",
        "resolved_start_date",
        "last_return_date",
    ):
        windows[column] = pd.to_datetime(windows[column])

    window_keys = [
        "portfolio_id",
        "as_of_date",
        "window_id",
    ]

    if windows.duplicated(window_keys).any():
        raise ValueError("Duplicate window definitions.")

    indexed_windows = windows.set_index(window_keys)

    records = []

    pair_keys = [
        "portfolio_id",
        "benchmark_portfolio_id",
    ]

    for (portfolio_id, benchmark_id), group in (
        rf_aligned_df.groupby(pair_keys, sort=True)
    ):

        portfolio_windows = windows.loc[
            windows["portfolio_id"].eq(portfolio_id)
        ]

        benchmark_windows = windows.loc[
            windows["portfolio_id"].eq(benchmark_id)
        ]

        p_keys = set(
            zip(
                portfolio_windows["as_of_date"],
                portfolio_windows["window_id"],
            )
        )

        b_keys = set(
            zip(
                benchmark_windows["as_of_date"],
                benchmark_windows["window_id"],
            )
        )

        if not p_keys or p_keys != b_keys:
            raise ValueError(
                "Portfolio/benchmark windows do not match."
            )

        for as_of, window_id in sorted(p_keys):

            p_window = indexed_windows.loc[
                (portfolio_id, as_of, window_id)
            ]

            b_window = indexed_windows.loc[
                (benchmark_id, as_of, window_id)
            ]

            # Risk-free coverage is meaningful only
            # over identical portfolio/benchmark dates.

            for field in (
                "resolved_start_date",
                "last_return_date",
                "observation_count",
                "window_status",
            ):
                if p_window[field] != b_window[field]:
                    raise ValueError(
                        f"{window_id}: mismatched window {field}."
                    )

            record = {
                "portfolio_id": portfolio_id,
                "benchmark_portfolio_id": benchmark_id,
                "as_of_date": as_of,
                "window_id": window_id,
                "resolved_start_date": (
                    p_window["resolved_start_date"]
                ),
                "requested_observations": (
                    p_window["observation_count"]
                ),
                "rf_available_observations": 0,
                "rf_missing_observations": 0,
                "missing_rf_leading": 0,
                "missing_rf_internal": 0,
                "missing_rf_trailing": 0,
                "first_available_rf_date": pd.NaT,
                "last_available_rf_date": pd.NaT,
                "coverage_status": "WINDOW_UNAVAILABLE",
                "ratio_eligibility": "NOT_ELIGIBLE",
            }

            if p_window["window_status"] != "AVAILABLE":
                records.append(record)
                continue

            start = p_window["resolved_start_date"]
            end = p_window["last_return_date"]

            sample = group.loc[
                group["date"].gt(start)
                & group["date"].le(end)
            ]

            n = len(sample)

            if (
                n == 0
                or n != int(p_window["observation_count"])
                or sample["date"].iloc[-1] != end
            ):
                raise ValueError(
                    f"{window_id}: invalid RF sample coverage."
                )

            available = sample[
                "risk_free_available"
            ].to_numpy(dtype=bool)

            matched = np.flatnonzero(available)
            matched_count = len(matched)
            missing_count = n - matched_count

            record["rf_available_observations"] = matched_count
            record["rf_missing_observations"] = missing_count

            if matched_count == 0:

                record["coverage_status"] = "NO_COVERAGE"

            else:

                leading = int(matched[0])
                trailing = int(n - 1 - matched[-1])
                internal = int(
                    missing_count - leading - trailing
                )

                record.update(
                    {
                        "missing_rf_leading": leading,
                        "missing_rf_internal": internal,
                        "missing_rf_trailing": trailing,
                        "first_available_rf_date": (
                            sample["date"].iloc[matched[0]]
                        ),
                        "last_available_rf_date": (
                            sample["date"].iloc[matched[-1]]
                        ),
                    }
                )

                if missing_count == 0:

                    record["coverage_status"] = (
                        "FULL_COVERAGE"
                    )

                    record["ratio_eligibility"] = (
                        "FULL_WINDOW_ELIGIBLE"
                    )

                elif leading == 0 and internal == 0:

                    record["coverage_status"] = (
                        "TRAILING_GAP_ONLY"
                    )

                    record["ratio_eligibility"] = (
                        "EARLIER_CUTOFF_REQUIRED"
                    )

                else:

                    record["coverage_status"] = (
                        "GAPPED_COVERAGE"
                    )

                    record["ratio_eligibility"] = (
                        "NOT_ELIGIBLE"
                    )

            records.append(record)

    return (
        pd.DataFrame.from_records(records)
        .sort_values(
            [
                "portfolio_id",
                "benchmark_portfolio_id",
                "as_of_date",
                "window_id",
            ]
        )
        .reset_index(drop=True)
    )


# Research reference: 07_performance_analytics.ipynb, code cell index 13.
def compute_risk_adjusted_performance(
    aligned_rf_df: pd.DataFrame,
    windows_df: pd.DataFrame,
    *,
    sessions_per_year: int = 252,
    minimum_observations: int = 2,
) -> pd.DataFrame:
    """Calculate risk-adjusted metrics on explicit windows."""

    required_daily = {
        "portfolio_id",
        "benchmark_portfolio_id",
        "date",
        "portfolio_return_total",
        "benchmark_return_total",
        "risk_free_return",
    }

    required_window = {
        "portfolio_id",
        "as_of_date",
        "window_id",
        "resolved_start_date",
        "observation_count",
        "window_status",
    }

    if not required_daily.issubset(aligned_rf_df.columns):
        raise ValueError("Missing risk-aligned input fields.")

    if not required_window.issubset(windows_df.columns):
        raise ValueError("Missing reporting-window fields.")

    if (
        isinstance(sessions_per_year, bool)
        or not isinstance(sessions_per_year, int)
        or sessions_per_year <= 0
        or isinstance(minimum_observations, bool)
        or not isinstance(minimum_observations, int)
        or minimum_observations < 2
    ):
        raise ValueError("Invalid annualization configuration.")

    daily = aligned_rf_df[
        [
            "portfolio_id",
            "benchmark_portfolio_id",
            "date",
            "portfolio_return_total",
            "benchmark_return_total",
            "risk_free_return",
        ]
    ].copy()

    windows = windows_df[
        list(required_window)
    ].copy()

    daily["date"] = pd.to_datetime(daily["date"])

    for column in ("as_of_date", "resolved_start_date"):
        windows[column] = pd.to_datetime(windows[column])

    pair_keys = [
        "portfolio_id",
        "benchmark_portfolio_id",
    ]

    if daily.duplicated(pair_keys + ["date"]).any():
        raise ValueError("Duplicate aligned daily observations.")

    window_keys = [
        "portfolio_id",
        "as_of_date",
        "window_id",
    ]

    if windows.duplicated(window_keys).any():
        raise ValueError("Duplicate reporting-window keys.")

    for column in (
        "portfolio_return_total",
        "benchmark_return_total",
    ):
        values = daily[column].to_numpy(dtype=float)

        if not np.isfinite(values).all() or (values < -1).any():
            raise ValueError(f"Invalid {column} values.")

    rf_values = daily["risk_free_return"].to_numpy(
        dtype=float
    )

    if np.isinf(rf_values).any():
        raise ValueError("Infinite risk-free returns detected.")

    metric_names = [
        "portfolio_sharpe_ratio",
        "benchmark_sharpe_ratio",
        "portfolio_sortino_ratio_zero_target",
        "benchmark_sortino_ratio_zero_target",
        "information_ratio",
        "portfolio_annualized_arithmetic_excess_return",
        "benchmark_annualized_arithmetic_excess_return",
    ]

    window_index = windows.set_index(window_keys)

    records = []
    annualization_factor = np.sqrt(sessions_per_year)

    def standard_ratio(
        numerator: float,
        denominator: float,
    ) -> tuple[float, str]:
        """Return an annualized ratio and its calculation status."""

        if not np.isfinite(denominator) or denominator <= 1e-14:
            return np.nan, "UNDEFINED_ZERO_DENOMINATOR"

        return (
            float(annualization_factor * numerator / denominator),
            "CALCULATED",
        )

    for (portfolio_id, benchmark_id), group in (
        daily.groupby(pair_keys, sort=True)
    ):

        group = group.sort_values("date")

        p_windows = windows.loc[
            windows["portfolio_id"].eq(portfolio_id)
        ]

        b_windows = windows.loc[
            windows["portfolio_id"].eq(benchmark_id)
        ]

        p_keys = set(
            zip(p_windows["as_of_date"], p_windows["window_id"])
        )

        b_keys = set(
            zip(b_windows["as_of_date"], b_windows["window_id"])
        )

        if not p_keys or p_keys != b_keys:
            raise ValueError(
                "Missing or unmatched portfolio/benchmark windows."
            )

        for as_of, window_id in sorted(p_keys):

            p_window = window_index.loc[
                (portfolio_id, as_of, window_id)
            ]

            b_window = window_index.loc[
                (benchmark_id, as_of, window_id)
            ]

            metric_results = {
                name: (np.nan, "WINDOW_UNAVAILABLE")
                for name in metric_names
            }

            sample_count = 0
            rf_count = 0

            p_status = p_window["window_status"]
            b_status = b_window["window_status"]

            if p_status == "AVAILABLE" and b_status == "AVAILABLE":

                start = p_window["resolved_start_date"]
                benchmark_start = b_window["resolved_start_date"]

                if (
                    start != benchmark_start
                    or p_window["observation_count"]
                    != b_window["observation_count"]
                ):
                    raise ValueError(
                        f"Mismatched {window_id} window boundaries."
                    )

                sample = group.loc[
                    group["date"].gt(start)
                    & group["date"].le(as_of)
                ]

                sample_count = len(sample)

                if (
                    sample_count == 0
                    or sample_count
                    != int(p_window["observation_count"])
                    or sample["date"].iloc[-1] != as_of
                ):
                    raise ValueError(
                        f"Inconsistent {window_id} return coverage."
                    )

                rp = sample[
                    "portfolio_return_total"
                ].to_numpy(dtype=float)

                rb = sample[
                    "benchmark_return_total"
                ].to_numpy(dtype=float)

                rf = sample[
                    "risk_free_return"
                ].to_numpy(dtype=float)

                rf_available = np.isfinite(rf)
                rf_count = int(rf_available.sum())

                if sample_count < minimum_observations:

                    metric_results = {
                        name: (
                            np.nan,
                            "INSUFFICIENT_OBSERVATIONS",
                        )
                        for name in metric_names
                    }

                else:

                    # ----------------------------------------
                    # RF-independent: Sortino, zero target.
                    # ----------------------------------------

                    for label, returns in (
                        ("portfolio", rp),
                        ("benchmark", rb),
                    ):

                        downside_rms = float(
                            np.sqrt(
                                np.mean(
                                    np.minimum(returns, 0.0) ** 2
                                )
                            )
                        )

                        metric_results[
                            f"{label}_sortino_ratio_zero_target"
                        ] = standard_ratio(
                            float(np.mean(returns)),
                            downside_rms,
                        )

                    # ----------------------------------------
                    # RF-independent: Information Ratio.
                    # ----------------------------------------

                    active_returns = rp - rb

                    metric_results["information_ratio"] = (
                        standard_ratio(
                            float(np.mean(active_returns)),
                            float(
                                np.std(
                                    active_returns,
                                    ddof=1,
                                )
                            ),
                        )
                    )

                    # ----------------------------------------
                    # RF-dependent: Sharpe and excess return.
                    # All observations must have RF.
                    # ----------------------------------------

                    if rf_count != sample_count:

                        for label in ("portfolio", "benchmark"):

                            metric_results[
                                f"{label}_sharpe_ratio"
                            ] = (
                                np.nan,
                                "UNAVAILABLE_RF_COVERAGE",
                            )

                            metric_results[
                                f"{label}_annualized_arithmetic_excess_return"
                            ] = (
                                np.nan,
                                "UNAVAILABLE_RF_COVERAGE",
                            )

                    else:

                        for label, returns in (
                            ("portfolio", rp),
                            ("benchmark", rb),
                        ):

                            excess_returns = returns - rf

                            mean_excess = float(
                                np.mean(excess_returns)
                            )

                            excess_std = float(
                                np.std(
                                    excess_returns,
                                    ddof=1,
                                )
                            )

                            metric_results[
                                f"{label}_sharpe_ratio"
                            ] = standard_ratio(
                                mean_excess,
                                excess_std,
                            )

                            metric_results[
                                f"{label}_annualized_arithmetic_excess_return"
                            ] = (
                                float(
                                    mean_excess
                                    * sessions_per_year
                                ),
                                "CALCULATED",
                            )

            for metric_name in metric_names:

                value, status = metric_results[metric_name]

                records.append(
                    {
                        "portfolio_id": portfolio_id,
                        "benchmark_portfolio_id": benchmark_id,
                        "as_of_date": as_of,
                        "window_id": window_id,
                        "metric_name": metric_name,
                        "metric_value": value,
                        "metric_units": (
                            "annualized_simple_decimal"
                            if "excess_return" in metric_name
                            else "dimensionless_ratio"
                        ),
                        "calculation_status": status,
                        "return_observations": sample_count,
                        "rf_available_observations": rf_count,
                        "sessions_per_year": sessions_per_year,
                        "minimum_observations": minimum_observations,
                        "sortino_target": "zero_daily_return",
                        "standard_deviation_ddof": 1,
                    }
                )

    return (
        pd.DataFrame.from_records(records)
        .sort_values(
            [
                "as_of_date",
                "window_id",
                "metric_name",
                "portfolio_id",
            ]
        )
        .reset_index(drop=True)
    )
