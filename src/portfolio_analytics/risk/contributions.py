"""Risk contributions: validated research calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd

RISK_REALIZED_RECONCILIATION_ATOL = 1e-12

# Research reference: 09_risk_analytics.ipynb, code cell index 11.
def calculate_realized_risk_contributions(
    holdings_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    portfolio_id: str,
    as_of_date: pd.Timestamp,
    lookback_sessions: int | None = None,
    annualization_sessions: int = 252,
    variance_ddof: int = 1,
) -> tuple[pd.DataFrame, dict]:
    """
    Attribute historical portfolio volatility to securities.

    Parameters
    ----------
    holdings_df:
        Daily actual portfolio holdings, requiring:
            portfolio_id, date, security_id,
            weight_bop, return_total.

    returns_df:
        Actual daily portfolio total returns, requiring:
            portfolio_id, date, return_total.

    portfolio_id:
        Exact portfolio run identifier.

    as_of_date:
        Observed estimation cutoff.

    lookback_sessions:
        None for all available history through as-of.
        Otherwise the number of trailing return sessions.

    annualization_sessions:
        Explicit daily-to-annual variance multiplier.

    variance_ddof:
        Common variance/covariance degrees of freedom.

    Returns
    -------
    detail_df:
        Security-level realized risk contributions.

    diagnostics:
        Portfolio-level variance, volatility, contribution
        reconciliation and coverage diagnostics.

    Notes
    -----
    The security universe contains every security actually
    held during the selected estimation window.

    Absent holdings mean zero exposure, not missing returns.

    Observed holdings with invalid returns are rejected.

    Component annualized volatility:
        252 * Cov(c_i, r_p) / sigma_p_ann.

    The component values sum to portfolio volatility when
    daily security contributions reconcile to portfolio
    total returns.

    Zero portfolio variance makes component volatility
    contributions and shares mathematically unavailable.

    No other notebook state or external system is accessed.
    """

    # --------------------------------------------------------
    # Validate parameters and required columns
    # --------------------------------------------------------

    required_holdings = {
        "portfolio_id",
        "date",
        "security_id",
        "weight_bop",
        "return_total",
    }

    required_returns = {
        "portfolio_id",
        "date",
        "return_total",
    }

    if not required_holdings.issubset(holdings_df.columns):
        raise ValueError(
            "Missing required holdings contribution fields."
        )

    if not required_returns.issubset(returns_df.columns):
        raise ValueError(
            "Missing required portfolio return fields."
        )

    for name, value in (
        ("annualization_sessions", annualization_sessions),
        ("variance_ddof", variance_ddof),
    ):
        if (
            not isinstance(value, (int, np.integer))
            or isinstance(value, (bool, np.bool_))
            or value < 1
        ):
            raise ValueError(
                f"{name} must be a positive integer."
            )

    if lookback_sessions is not None:
        if (
            not isinstance(
                lookback_sessions, (int, np.integer)
            )
            or isinstance(
                lookback_sessions, (bool, np.bool_)
            )
            or lookback_sessions <= variance_ddof
        ):
            raise ValueError(
                "Lookback must exceed variance_ddof."
            )

    cutoff = pd.Timestamp(as_of_date)

    if pd.isna(cutoff):
        raise ValueError("As-of date cannot be missing.")

    # --------------------------------------------------------
    # Select exact portfolio return observations
    # --------------------------------------------------------

    portfolio_returns = returns_df.loc[
        returns_df["portfolio_id"].eq(portfolio_id),
        ["date", "return_total"],
    ].copy()

    portfolio_returns["date"] = pd.to_datetime(
        portfolio_returns["date"],
        errors="raise",
    )

    if (
        portfolio_returns.empty
        or portfolio_returns["date"].isna().any()
        or portfolio_returns["date"].duplicated().any()
    ):
        raise ValueError(
            "Portfolio return dates are missing or duplicated."
        )

    portfolio_returns = (
        portfolio_returns
        .sort_values("date")
        .reset_index(drop=True)
    )

    if cutoff not in pd.DatetimeIndex(
        portfolio_returns["date"]
    ):
        raise ValueError(
            "As-of date must exist in the portfolio calendar."
        )

    eligible_returns = portfolio_returns.loc[
        portfolio_returns["date"].le(cutoff)
    ]

    window_returns = (
        eligible_returns
        if lookback_sessions is None
        else eligible_returns.tail(lookback_sessions)
    ).copy()

    observations = len(window_returns)

    if observations <= variance_ddof:
        raise ValueError(
            "Insufficient return observations for covariance."
        )

    if (
        lookback_sessions is not None
        and observations != lookback_sessions
    ):
        raise ValueError(
            "Insufficient history for requested lookback."
        )

    if (
        not pd.api.types.is_numeric_dtype(
            window_returns["return_total"]
        )
        or pd.api.types.is_bool_dtype(
            window_returns["return_total"]
        )
        or pd.api.types.is_complex_dtype(
            window_returns["return_total"]
        )
    ):
        raise TypeError(
            "Portfolio returns must be real numeric values."
        )

    portfolio_values = window_returns[
        "return_total"
    ].to_numpy(dtype=np.float64)

    if (
        not np.isfinite(portfolio_values).all()
        or np.any(portfolio_values < -1.0)
    ):
        raise ValueError(
            "Selected portfolio returns are invalid."
        )

    window_dates = pd.DatetimeIndex(
        window_returns["date"]
    )

    # --------------------------------------------------------
    # Select actual holdings in the estimation window
    # --------------------------------------------------------

    holding_columns = [
        "date",
        "security_id",
        "weight_bop",
        "return_total",
    ]

    has_saved_contributions = (
        "contribution_total" in holdings_df.columns
    )

    if has_saved_contributions:
        holding_columns.append("contribution_total")

    selected_holdings = holdings_df.loc[
        holdings_df["portfolio_id"].eq(portfolio_id),
        holding_columns,
    ].copy()

    selected_holdings["date"] = pd.to_datetime(
        selected_holdings["date"],
        errors="raise",
    )

    selected_holdings = selected_holdings.loc[
        selected_holdings["date"].isin(window_dates)
    ].copy()

    if selected_holdings.empty:
        raise ValueError(
            "No holdings found within the estimation window."
        )

    if (
        selected_holdings["date"].isna().any()
        or selected_holdings["security_id"].isna().any()
    ):
        raise ValueError(
            "Holdings contain missing date or security IDs."
        )

    if selected_holdings.duplicated(
        ["date", "security_id"]
    ).any():
        raise ValueError(
            "Duplicate holdings security/date keys."
        )

    if not window_dates.isin(
        selected_holdings["date"]
    ).all():
        raise ValueError(
            "One or more portfolio return dates have "
            "no corresponding holdings."
        )

    # --------------------------------------------------------
    # Validate held-security returns and beginning weights
    # --------------------------------------------------------

    for column in ("weight_bop", "return_total"):

        values = selected_holdings[column]

        if (
            not pd.api.types.is_numeric_dtype(values)
            or pd.api.types.is_bool_dtype(values)
            or pd.api.types.is_complex_dtype(values)
        ):
            raise TypeError(
                f"Holdings {column} must be real numeric."
            )

        if not np.isfinite(
            values.to_numpy(dtype=np.float64)
        ).all():
            raise ValueError(
                f"Holdings {column} contains non-finite values."
            )

    if (
        selected_holdings["return_total"] < -1.0
    ).any():
        raise ValueError(
            "Held-security simple returns below -100%."
        )

    # --------------------------------------------------------
    # Reconstruct daily realized security contributions
    # --------------------------------------------------------

    calculated_contributions = (
        selected_holdings["weight_bop"].to_numpy(
            dtype=np.float64
        )
        * selected_holdings["return_total"].to_numpy(
            dtype=np.float64
        )
    )

    if not np.isfinite(
        calculated_contributions
    ).all():
        raise ArithmeticError(
            "Calculated security contributions are non-finite."
        )

    selected_holdings[
        "calculated_contribution"
    ] = calculated_contributions

    saved_contribution_error = np.nan

    if has_saved_contributions:

        saved_values = selected_holdings[
            "contribution_total"
        ]

        if (
            not pd.api.types.is_numeric_dtype(saved_values)
            or not np.isfinite(
                saved_values.to_numpy(dtype=np.float64)
            ).all()
        ):
            raise ValueError(
                "Saved contributions contain invalid values."
            )

        saved_contribution_error = float(
            np.max(
                np.abs(
                    saved_values.to_numpy(dtype=np.float64)
                    - calculated_contributions
                )
            )
        )

        np.testing.assert_allclose(
            calculated_contributions,
            saved_values.to_numpy(dtype=np.float64),
            rtol=1e-10,
            atol=RISK_REALIZED_RECONCILIATION_ATOL,
            err_msg=(
                "Calculated contributions differ from "
                "saved holdings contributions."
            ),
        )

    # --------------------------------------------------------
    # Build complete contribution matrix
    # --------------------------------------------------------

    securities = sorted(
        selected_holdings["security_id"].unique(),
        key=str,
    )

    contribution_panel = (
        selected_holdings
        .pivot(
            index="date",
            columns="security_id",
            values="calculated_contribution",
        )
        .reindex(
            index=window_dates,
            columns=securities,
        )
        .fillna(0.0)
    )

    # Zero is assigned only to absent holdings, never to
    # a missing return for a recorded holding.

    if contribution_panel.isna().any().any():
        raise ValueError(
            "Missing contribution for an observed holding."
        )

    contribution_matrix = (
        contribution_panel.to_numpy(dtype=np.float64)
    )

    # --------------------------------------------------------
    # Reconcile daily contributions to portfolio returns
    # --------------------------------------------------------

    reconstructed_returns = contribution_matrix.sum(
        axis=1
    )

    daily_reconciliation_error = float(
        np.max(
            np.abs(
                reconstructed_returns - portfolio_values
            )
        )
    )

    np.testing.assert_allclose(
        reconstructed_returns,
        portfolio_values,
        rtol=1e-10,
        atol=RISK_REALIZED_RECONCILIATION_ATOL,
        err_msg=(
            "Daily security contributions do not "
            "reconcile to portfolio total returns."
        ),
    )

    # --------------------------------------------------------
    # Calculate sample covariance attribution
    # --------------------------------------------------------

    contribution_differences = (
        contribution_matrix - contribution_matrix[:1, :]
    )

    contribution_centered = (
        contribution_differences
        - contribution_differences.mean(
            axis=0,
            keepdims=True,
        )
    )

    portfolio_differences = (
        portfolio_values - portfolio_values[0]
    )

    portfolio_centered = (
        portfolio_differences
        - portfolio_differences.mean()
    )

    denominator = observations - variance_ddof

    portfolio_variance_daily = float(
        np.dot(
            portfolio_centered,
            portfolio_centered,
        ) / denominator
    )

    covariance_contribution_daily = (
        contribution_centered.T @ portfolio_centered
    ) / denominator

    component_variance_annualized = (
        covariance_contribution_daily
        * annualization_sessions
    )

    portfolio_variance_annualized = (
        portfolio_variance_daily
        * annualization_sessions
    )

    portfolio_volatility_annualized = float(
        np.sqrt(portfolio_variance_annualized)
    )

    # --------------------------------------------------------
    # Calculate component volatility contributions
    # --------------------------------------------------------

    if portfolio_volatility_annualized > 0.0:

        component_volatility = (
            component_variance_annualized
            / portfolio_volatility_annualized
        )

        component_risk_share = (
            component_volatility
            / portfolio_volatility_annualized
        )

        attribution_status = "PASS"

    else:

        component_volatility = np.full(
            len(securities),
            np.nan,
        )

        component_risk_share = np.full(
            len(securities),
            np.nan,
        )

        attribution_status = (
            "UNAVAILABLE_ZERO_PORTFOLIO_VOLATILITY"
        )

    # --------------------------------------------------------
    # Reconcile variance, volatility and mean contributions
    # --------------------------------------------------------

    variance_contribution_sum = float(
        np.sum(
            component_variance_annualized
        )
    )

    np.testing.assert_allclose(
        variance_contribution_sum,
        portfolio_variance_annualized,
        rtol=1e-10,
        atol=1e-12,
    )

    mean_contribution_daily = (
        contribution_matrix.mean(axis=0)
    )

    np.testing.assert_allclose(
        mean_contribution_daily.sum(),
        portfolio_values.mean(),
        rtol=1e-10,
        atol=1e-12,
    )

    if portfolio_volatility_annualized > 0.0:

        np.testing.assert_allclose(
            component_volatility.sum(),
            portfolio_volatility_annualized,
            rtol=1e-10,
            atol=1e-12,
        )

        np.testing.assert_allclose(
            component_risk_share.sum(),
            1.0,
            rtol=1e-10,
            atol=1e-12,
        )

    # --------------------------------------------------------
    # Count actual holding observations per security
    # --------------------------------------------------------

    holding_session_counts = (
        selected_holdings
        .groupby("security_id")["date"]
        .nunique()
        .reindex(securities)
        .to_numpy(dtype=np.int64)
    )

    # --------------------------------------------------------
    # Return standardized detail and diagnostics
    # --------------------------------------------------------

    detail = pd.DataFrame(
        {
            "security_id": securities,
            "observations": observations,
            "held_sessions": holding_session_counts,
            "mean_contribution_daily": (
                mean_contribution_daily
            ),
            "covariance_with_portfolio_daily": (
                covariance_contribution_daily
            ),
            "contribution_to_annualized_variance": (
                component_variance_annualized
            ),
            "contribution_to_annualized_volatility": (
                component_volatility
            ),
            "risk_share_of_volatility": (
                component_risk_share
            ),
            "status": attribution_status,
        }
    )

    diagnostics = {
        "as_of_date": cutoff,
        "window_start": window_dates.min(),
        "window_end": window_dates.max(),
        "observations": observations,
        "historically_held_securities": len(securities),
        "annualization_sessions": annualization_sessions,
        "variance_ddof": variance_ddof,
        "portfolio_variance_daily": (
            portfolio_variance_daily
        ),
        "portfolio_variance_annualized": (
            portfolio_variance_annualized
        ),
        "portfolio_volatility_annualized": (
            portfolio_volatility_annualized
        ),
        "sum_component_variance_annualized": (
            variance_contribution_sum
        ),
        "sum_component_volatility": (
            float(np.nansum(component_volatility))
            if attribution_status == "PASS"
            else np.nan
        ),
        "sum_risk_shares": (
            float(np.nansum(component_risk_share))
            if attribution_status == "PASS"
            else np.nan
        ),
        "maximum_daily_return_reconciliation_error": (
            daily_reconciliation_error
        ),
        "maximum_saved_contribution_error": (
            saved_contribution_error
        ),
        "negative_risk_contributors": int(
            np.count_nonzero(
                component_variance_annualized < 0.0
            )
        ),
        "status": attribution_status,
    }

    return detail, diagnostics


# Research reference: 09_risk_analytics.ipynb, code cell index 12.
def calculate_active_risk_contributions(
    holdings_df: pd.DataFrame,
    aligned_returns_df: pd.DataFrame,
    portfolio_id: str,
    benchmark_portfolio_id: str,
    as_of_date: pd.Timestamp,
    lookback_sessions: int | None = None,
    annualization_sessions: int = 252,
    variance_ddof: int = 1,
    reconciliation_atol: float = 1e-12,
) -> tuple[pd.DataFrame, dict]:
    """
    Attribute realized arithmetic active-return volatility.

    Returns
    -------
    detail_df:
        One row per security held in either portfolio
        during the estimation window.

    diagnostics:
        Tracking error, component sums, holding-universe
        statistics, and return reconciliation errors.

    Definitions
    -----------
    c_active(i,t) = c_portfolio(i,t) - c_benchmark(i,t)

    active_return(t) = sum_i c_active(i,t)

    annualized_TE = sqrt(
        annualization_sessions * Var(active_return)
    )

    component_TE(i) = (
        annualization_sessions
        * Cov(c_active(i), active_return)
        / annualized_TE
    )

    Risk shares sum to one when tracking error is positive.

    If tracking error is zero, volatility contributions
    and risk shares are mathematically unavailable.

    All covariance calculations use a common sample
    and the same degrees-of-freedom convention.

    No source DataFrames are modified.
    """

    # --------------------------------------------------------
    # Validate parameters and required fields
    # --------------------------------------------------------

    if portfolio_id == benchmark_portfolio_id:
        raise ValueError(
            "Portfolio and benchmark IDs must differ."
        )

    required_holdings = {
        "portfolio_id",
        "date",
        "security_id",
        "weight_bop",
        "return_total",
    }

    required_returns = {
        "portfolio_id",
        "benchmark_portfolio_id",
        "date",
        "portfolio_return_total",
        "benchmark_return_total",
        "active_return_arithmetic",
    }

    if not required_holdings.issubset(holdings_df.columns):
        raise ValueError(
            "Missing required holdings fields."
        )

    if not required_returns.issubset(
        aligned_returns_df.columns
    ):
        raise ValueError(
            "Missing required aligned-return fields."
        )

    for name, value in (
        ("annualization_sessions", annualization_sessions),
        ("variance_ddof", variance_ddof),
    ):
        if (
            not isinstance(value, (int, np.integer))
            or isinstance(value, (bool, np.bool_))
            or value < 1
        ):
            raise ValueError(
                f"{name} must be a positive integer."
            )

    if lookback_sessions is not None:
        if (
            not isinstance(
                lookback_sessions, (int, np.integer)
            )
            or isinstance(
                lookback_sessions, (bool, np.bool_)
            )
            or lookback_sessions <= variance_ddof
        ):
            raise ValueError(
                "Lookback must exceed variance_ddof."
            )

    if (
        not np.isscalar(reconciliation_atol)
        or not np.isfinite(reconciliation_atol)
        or reconciliation_atol < 0.0
    ):
        raise ValueError(
            "Reconciliation tolerance must be "
            "a nonnegative finite number."
        )

    cutoff = pd.Timestamp(as_of_date)

    if pd.isna(cutoff):
        raise ValueError(
            "As-of date cannot be missing."
        )

    # --------------------------------------------------------
    # Select and validate aligned return history
    # --------------------------------------------------------

    aligned = aligned_returns_df[
        list(required_returns)
    ].copy()

    aligned["date"] = pd.to_datetime(
        aligned["date"],
        errors="raise",
    )

    aligned = aligned.loc[
        aligned["date"].le(cutoff)
    ].copy()

    if aligned.empty:
        raise ValueError(
            "No eligible aligned return observations."
        )

    if (
        aligned["date"].isna().any()
        or aligned["date"].duplicated().any()
    ):
        raise ValueError(
            "Aligned return dates must be non-null "
            "and unique."
        )

    if not aligned["portfolio_id"].eq(
        portfolio_id
    ).all():
        raise ValueError(
            "Aligned portfolio identity mismatch."
        )

    if not aligned["benchmark_portfolio_id"].eq(
        benchmark_portfolio_id
    ).all():
        raise ValueError(
            "Aligned benchmark identity mismatch."
        )

    aligned = (
        aligned
        .sort_values("date")
        .reset_index(drop=True)
    )

    if cutoff not in pd.DatetimeIndex(
        aligned["date"]
    ):
        raise ValueError(
            "As-of date must exist in the return calendar."
        )

    window = (
        aligned
        if lookback_sessions is None
        else aligned.tail(lookback_sessions)
    ).copy()

    observations = len(window)

    if observations <= variance_ddof:
        raise ValueError(
            "Insufficient observations for active covariance."
        )

    if (
        lookback_sessions is not None
        and observations != lookback_sessions
    ):
        raise ValueError(
            "Insufficient history for requested lookback."
        )

    return_columns = [
        "portfolio_return_total",
        "benchmark_return_total",
        "active_return_arithmetic",
    ]

    for column in return_columns:

        series = window[column]

        if (
            not pd.api.types.is_numeric_dtype(series)
            or pd.api.types.is_bool_dtype(series)
            or pd.api.types.is_complex_dtype(series)
        ):
            raise TypeError(
                f"{column} must contain real numeric returns."
            )

    return_matrix = window[
        return_columns
    ].to_numpy(dtype=np.float64)

    if not np.isfinite(return_matrix).all():
        raise ValueError(
            "Aligned returns contain non-finite values."
        )

    portfolio_returns = return_matrix[:, 0]
    benchmark_returns = return_matrix[:, 1]
    recorded_active_returns = return_matrix[:, 2]

    if (
        np.any(portfolio_returns < -1.0)
        or np.any(benchmark_returns < -1.0)
    ):
        raise ValueError(
            "Portfolio or benchmark simple returns "
            "fall below -100%."
        )

    active_returns = (
        portfolio_returns - benchmark_returns
    )

    np.testing.assert_allclose(
        active_returns,
        recorded_active_returns,
        rtol=1e-10,
        atol=reconciliation_atol,
        err_msg=(
            "Arithmetic active returns do not reconcile."
        ),
    )

    window_dates = pd.DatetimeIndex(
        window["date"]
    )

    # --------------------------------------------------------
    # Select holdings from both portfolios
    # --------------------------------------------------------

    has_saved_contributions = (
        "contribution_total" in holdings_df.columns
    )

    holding_columns = [
        "portfolio_id",
        "date",
        "security_id",
        "weight_bop",
        "return_total",
    ]

    if has_saved_contributions:
        holding_columns.append("contribution_total")

    selected_holdings = holdings_df.loc[
        holdings_df["portfolio_id"].isin(
            [portfolio_id, benchmark_portfolio_id]
        ),
        holding_columns,
    ].copy()

    selected_holdings["date"] = pd.to_datetime(
        selected_holdings["date"],
        errors="raise",
    )

    selected_holdings = selected_holdings.loc[
        selected_holdings["date"].isin(window_dates)
    ].copy()

    if selected_holdings.empty:
        raise ValueError(
            "No holdings found in the estimation window."
        )

    if (
        selected_holdings["date"].isna().any()
        or selected_holdings["security_id"].isna().any()
    ):
        raise ValueError(
            "Holdings contain missing date or security IDs."
        )

    holding_keys = [
        "portfolio_id",
        "date",
        "security_id",
    ]

    if selected_holdings.duplicated(
        subset=holding_keys
    ).any():
        raise ValueError(
            "Duplicate portfolio/security/date holdings."
        )

    # --------------------------------------------------------
    # Validate recorded holding values before zero-filling
    # --------------------------------------------------------

    for column in ("weight_bop", "return_total"):

        series = selected_holdings[column]

        if (
            not pd.api.types.is_numeric_dtype(series)
            or pd.api.types.is_bool_dtype(series)
            or pd.api.types.is_complex_dtype(series)
        ):
            raise TypeError(
                f"Holdings {column} must be real numeric."
            )

        if not np.isfinite(
            series.to_numpy(dtype=np.float64)
        ).all():
            raise ValueError(
                f"Holdings {column} contains invalid values."
            )

    if (
        selected_holdings["return_total"] < -1.0
    ).any():
        raise ValueError(
            "Held-security return below -100%."
        )

    selected_holdings["calculated_contribution"] = (
        selected_holdings["weight_bop"].to_numpy(
            dtype=np.float64
        )
        * selected_holdings["return_total"].to_numpy(
            dtype=np.float64
        )
    )

    if not np.isfinite(
        selected_holdings[
            "calculated_contribution"
        ].to_numpy()
    ).all():
        raise ArithmeticError(
            "Calculated holding contributions are invalid."
        )

    # --------------------------------------------------------
    # Reconcile against saved security contributions
    # --------------------------------------------------------

    maximum_saved_contribution_error = np.nan

    if has_saved_contributions:

        saved_series = selected_holdings[
            "contribution_total"
        ]

        if (
            not pd.api.types.is_numeric_dtype(saved_series)
            or not np.isfinite(
                saved_series.to_numpy(dtype=np.float64)
            ).all()
        ):
            raise ValueError(
                "Saved security contributions are invalid."
            )

        calculated = selected_holdings[
            "calculated_contribution"
        ].to_numpy(dtype=np.float64)

        saved = saved_series.to_numpy(
            dtype=np.float64
        )

        maximum_saved_contribution_error = float(
            np.max(np.abs(calculated - saved))
        )

        np.testing.assert_allclose(
            calculated,
            saved,
            rtol=1e-10,
            atol=reconciliation_atol,
            err_msg=(
                "Calculated and saved holding "
                "contributions do not reconcile."
            ),
        )

    # --------------------------------------------------------
    # Construct separate daily contribution panels
    # --------------------------------------------------------

    portfolio_panels = {}
    held_session_counts = {}
    universe_by_portfolio = {}

    for current_id, expected_returns in (
        (portfolio_id, portfolio_returns),
        (benchmark_portfolio_id, benchmark_returns),
    ):

        leg = selected_holdings.loc[
            selected_holdings["portfolio_id"].eq(
                current_id
            )
        ].copy()

        if leg.empty:
            raise ValueError(
                f"No holdings for portfolio: {current_id}"
            )

        if not window_dates.isin(
            leg["date"]
        ).all():
            raise ValueError(
                "Portfolio return date has no holdings: "
                f"{current_id}"
            )

        securities = sorted(
            leg["security_id"].unique(),
            key=str,
        )

        universe_by_portfolio[current_id] = set(
            securities
        )

        counts = (
            leg.groupby("security_id")["date"]
            .nunique()
        )

        held_session_counts[current_id] = counts

        panel = (
            leg
            .pivot(
                index="date",
                columns="security_id",
                values="calculated_contribution",
            )
            .reindex(
                index=window_dates,
                columns=securities,
            )
            .fillna(0.0)
        )

        # Every observed holding contribution was
        # validated before this operation. The zeros
        # therefore represent absent exposure only.

        if not np.isfinite(
            panel.to_numpy(dtype=np.float64)
        ).all():
            raise ValueError(
                "Non-finite contribution panel."
            )

        reconstructed = panel.to_numpy(
            dtype=np.float64
        ).sum(axis=1)

        np.testing.assert_allclose(
            reconstructed,
            expected_returns,
            rtol=1e-10,
            atol=reconciliation_atol,
            err_msg=(
                f"Daily return reconciliation failed: "
                f"{current_id}"
            ),
        )

        portfolio_panels[current_id] = panel

    # --------------------------------------------------------
    # Net both portfolios over their union security universe
    # --------------------------------------------------------

    portfolio_universe = universe_by_portfolio[
        portfolio_id
    ]

    benchmark_universe = universe_by_portfolio[
        benchmark_portfolio_id
    ]

    union_securities = sorted(
        portfolio_universe | benchmark_universe,
        key=str,
    )

    portfolio_panel = portfolio_panels[
        portfolio_id
    ].reindex(
        columns=union_securities,
        fill_value=0.0,
    )

    benchmark_panel = portfolio_panels[
        benchmark_portfolio_id
    ].reindex(
        columns=union_securities,
        fill_value=0.0,
    )

    active_contribution_matrix = (
        portfolio_panel.to_numpy(dtype=np.float64)
        - benchmark_panel.to_numpy(dtype=np.float64)
    )

    reconstructed_active_returns = (
        active_contribution_matrix.sum(axis=1)
    )

    maximum_active_return_error = float(
        np.max(
            np.abs(
                reconstructed_active_returns
                - active_returns
            )
        )
    )

    np.testing.assert_allclose(
        reconstructed_active_returns,
        active_returns,
        rtol=1e-10,
        atol=reconciliation_atol,
        err_msg=(
            "Daily security active contributions do not "
            "reconcile to arithmetic active returns."
        ),
    )

    # --------------------------------------------------------
    # Calculate sample active-return variance and covariance
    # --------------------------------------------------------

    contribution_differences = (
        active_contribution_matrix
        - active_contribution_matrix[:1, :]
    )

    contribution_centered = (
        contribution_differences
        - contribution_differences.mean(
            axis=0,
            keepdims=True,
        )
    )

    active_differences = (
        active_returns - active_returns[0]
    )

    active_centered = (
        active_differences
        - active_differences.mean()
    )

    denominator = observations - variance_ddof

    active_variance_daily = float(
        np.dot(
            active_centered,
            active_centered,
        ) / denominator
    )

    covariance_with_active_daily = (
        contribution_centered.T @ active_centered
    ) / denominator

    component_active_variance_annualized = (
        covariance_with_active_daily
        * annualization_sessions
    )

    active_variance_annualized = (
        active_variance_daily
        * annualization_sessions
    )

    tracking_error_annualized = float(
        np.sqrt(active_variance_annualized)
    )

    # --------------------------------------------------------
    # Calculate tracking-error component contributions
    # --------------------------------------------------------

    if tracking_error_annualized > 0.0:

        component_tracking_error = (
            component_active_variance_annualized
            / tracking_error_annualized
        )

        tracking_error_risk_share = (
            component_tracking_error
            / tracking_error_annualized
        )

        attribution_status = "PASS"

    else:

        component_tracking_error = np.full(
            len(union_securities),
            np.nan,
        )

        tracking_error_risk_share = np.full(
            len(union_securities),
            np.nan,
        )

        attribution_status = (
            "UNAVAILABLE_ZERO_TRACKING_ERROR"
        )

    # --------------------------------------------------------
    # Reconcile component variance and tracking error
    # --------------------------------------------------------

    np.testing.assert_allclose(
        component_active_variance_annualized.sum(),
        active_variance_annualized,
        rtol=1e-10,
        atol=1e-12,
        err_msg=(
            "Active variance contributions do not reconcile."
        ),
    )

    mean_active_contribution_daily = (
        active_contribution_matrix.mean(axis=0)
    )

    np.testing.assert_allclose(
        mean_active_contribution_daily.sum(),
        active_returns.mean(),
        rtol=1e-10,
        atol=1e-12,
    )

    if attribution_status == "PASS":

        np.testing.assert_allclose(
            component_tracking_error.sum(),
            tracking_error_annualized,
            rtol=1e-10,
            atol=1e-12,
        )

        np.testing.assert_allclose(
            tracking_error_risk_share.sum(),
            1.0,
            rtol=1e-10,
            atol=1e-12,
        )

    # --------------------------------------------------------
    # Identify historical exposure categories
    # --------------------------------------------------------

    portfolio_counts = (
        held_session_counts[portfolio_id]
        .reindex(
            union_securities,
            fill_value=0,
        )
        .to_numpy(dtype=np.int64)
    )

    benchmark_counts = (
        held_session_counts[benchmark_portfolio_id]
        .reindex(
            union_securities,
            fill_value=0,
        )
        .to_numpy(dtype=np.int64)
    )

    exposure_category = np.select(
        [
            (portfolio_counts > 0)
            & (benchmark_counts > 0),
            (portfolio_counts > 0)
            & (benchmark_counts == 0),
            (portfolio_counts == 0)
            & (benchmark_counts > 0),
        ],
        [
            "BOTH_PORTFOLIOS",
            "ACTIVE_ONLY",
            "BENCHMARK_ONLY",
        ],
        default="UNCLASSIFIED",
    )

    if np.any(exposure_category == "UNCLASSIFIED"):
        raise ArithmeticError(
            "Unclassified active-risk security exposure."
        )

    # --------------------------------------------------------
    # Construct reusable detail and diagnostics
    # --------------------------------------------------------

    detail = pd.DataFrame(
        {
            "security_id": union_securities,
            "observations": observations,
            "portfolio_held_sessions": portfolio_counts,
            "benchmark_held_sessions": benchmark_counts,
            "exposure_category": exposure_category,
            "mean_active_contribution_daily": (
                mean_active_contribution_daily
            ),
            "covariance_with_active_return_daily": (
                covariance_with_active_daily
            ),
            "contribution_to_annualized_active_variance": (
                component_active_variance_annualized
            ),
            "contribution_to_annualized_tracking_error": (
                component_tracking_error
            ),
            "tracking_error_risk_share": (
                tracking_error_risk_share
            ),
            "status": attribution_status,
        }
    )

    diagnostics = {
        "as_of_date": cutoff,
        "window_start": window_dates.min(),
        "window_end": window_dates.max(),
        "observations": observations,
        "union_security_count": len(union_securities),
        "active_historical_securities": len(
            portfolio_universe
        ),
        "benchmark_historical_securities": len(
            benchmark_universe
        ),
        "overlapping_securities": len(
            portfolio_universe & benchmark_universe
        ),
        "annualization_sessions": annualization_sessions,
        "variance_ddof": variance_ddof,
        "mean_active_return_daily": float(
            active_returns.mean()
        ),
        "active_variance_daily": active_variance_daily,
        "active_variance_annualized": (
            active_variance_annualized
        ),
        "tracking_error_annualized": (
            tracking_error_annualized
        ),
        "sum_component_active_variance": float(
            component_active_variance_annualized.sum()
        ),
        "sum_component_tracking_error": (
            float(component_tracking_error.sum())
            if attribution_status == "PASS"
            else np.nan
        ),
        "sum_tracking_error_risk_shares": (
            float(tracking_error_risk_share.sum())
            if attribution_status == "PASS"
            else np.nan
        ),
        "negative_tracking_error_contributors": int(
            np.count_nonzero(
                component_active_variance_annualized < 0.0
            )
        ),
        "maximum_active_return_reconciliation_error": (
            maximum_active_return_error
        ),
        "maximum_saved_contribution_error": (
            maximum_saved_contribution_error
        ),
        "saved_contribution_status": (
            "VERIFIED"
            if has_saved_contributions
            else "NOT_PROVIDED"
        ),
        "status": attribution_status,
    }

    return detail, diagnostics


# Research reference: 09_risk_analytics.ipynb, code cell index 13.
def calculate_sector_risk_contributions(
    holdings_df: pd.DataFrame,
    aligned_returns_df: pd.DataFrame,
    portfolio_id: str,
    benchmark_portfolio_id: str,
    as_of_date: pd.Timestamp,
    lookback_sessions: int | None = None,
    annualization_sessions: int = 252,
    variance_ddof: int = 1,
    reconciliation_atol: float = 1e-12,
    missing_sector_label: str = (
        "UNCLASSIFIED_MISSING_SOURCE_LABEL"
    ),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Attribute realized volatility and tracking error by sector.

    Returns
    -------
    detail_df:
        One row per sector and risk series.

    diagnostics_df:
        One row for each of:
            Active absolute volatility.
            Benchmark absolute volatility.
            Arithmetic active-return tracking error.

    Methodology
    -----------
    The sector contribution series is constructed from
    dated holdings before covariance estimation.

    This means a security changing its recorded sector
    during the window contributes to the sector recorded
    on each respective holdings date.

    Sector component risk is:
        annualization_sessions
        * Cov(sector_contribution, reference_return)
        / annualized_reference_volatility.

    Unclassified source labels remain visible and contribute
    normally to portfolio or active-risk reconciliation.

    Zero reference volatility produces unavailable component
    volatility contributions and risk shares.

    No source inputs are modified.
    """

    # --------------------------------------------------------
    # Validate required columns and parameters
    # --------------------------------------------------------

    required_holdings = {
        "portfolio_id",
        "date",
        "security_id",
        "weight_bop",
        "return_total",
        "sector",
    }

    required_returns = {
        "portfolio_id",
        "benchmark_portfolio_id",
        "date",
        "portfolio_return_total",
        "benchmark_return_total",
        "active_return_arithmetic",
    }

    if not required_holdings.issubset(holdings_df.columns):
        raise ValueError(
            "Missing required sector holdings fields."
        )

    if not required_returns.issubset(
        aligned_returns_df.columns
    ):
        raise ValueError(
            "Missing required aligned-return fields."
        )

    if portfolio_id == benchmark_portfolio_id:
        raise ValueError(
            "Portfolio and benchmark must be distinct."
        )

    for name, value in (
        ("annualization_sessions", annualization_sessions),
        ("variance_ddof", variance_ddof),
    ):
        if (
            not isinstance(value, (int, np.integer))
            or isinstance(value, (bool, np.bool_))
            or value < 1
        ):
            raise ValueError(
                f"{name} must be a positive integer."
            )

    if lookback_sessions is not None:
        if (
            not isinstance(lookback_sessions, (int, np.integer))
            or isinstance(lookback_sessions, (bool, np.bool_))
            or lookback_sessions <= variance_ddof
        ):
            raise ValueError(
                "Lookback must exceed variance_ddof."
            )

    if (
        not isinstance(
            reconciliation_atol,
            (int, float, np.integer, np.floating),
        )
        or isinstance(reconciliation_atol, (bool, np.bool_))
        or not np.isfinite(reconciliation_atol)
        or reconciliation_atol < 0.0
    ):
        raise ValueError(
            "Reconciliation tolerance must be finite "
            "and nonnegative."
        )

    if (
        not isinstance(missing_sector_label, str)
        or not missing_sector_label
    ):
        raise ValueError(
            "Missing-label classification must be a string."
        )

    cutoff = pd.Timestamp(as_of_date)

    if pd.isna(cutoff):
        raise ValueError(
            "As-of date cannot be missing."
        )

    # --------------------------------------------------------
    # Select and validate aligned return observations
    # --------------------------------------------------------

    aligned = aligned_returns_df[
        [
            "portfolio_id",
            "benchmark_portfolio_id",
            "date",
            "portfolio_return_total",
            "benchmark_return_total",
            "active_return_arithmetic",
        ]
    ].copy()

    aligned["date"] = pd.to_datetime(
        aligned["date"],
        errors="raise",
    )

    if not aligned["portfolio_id"].eq(portfolio_id).all():
        raise ValueError(
            "Aligned portfolio identifier mismatch."
        )

    if not aligned["benchmark_portfolio_id"].eq(
        benchmark_portfolio_id
    ).all():
        raise ValueError(
            "Aligned benchmark identifier mismatch."
        )

    aligned = aligned.loc[
        aligned["date"].le(cutoff)
    ].copy()

    if (
        aligned.empty
        or aligned["date"].isna().any()
        or aligned["date"].duplicated().any()
    ):
        raise ValueError(
            "Aligned return dates are missing or duplicated."
        )

    aligned = (
        aligned
        .sort_values("date")
        .reset_index(drop=True)
    )

    if cutoff not in pd.DatetimeIndex(aligned["date"]):
        raise ValueError(
            "As-of date must exist in the return calendar."
        )

    window = (
        aligned
        if lookback_sessions is None
        else aligned.tail(lookback_sessions)
    ).copy()

    observations = len(window)

    if observations <= variance_ddof:
        raise ValueError(
            "Insufficient observations for covariance."
        )

    if (
        lookback_sessions is not None
        and observations != lookback_sessions
    ):
        raise ValueError(
            "Insufficient history for requested lookback."
        )

    return_columns = [
        "portfolio_return_total",
        "benchmark_return_total",
        "active_return_arithmetic",
    ]

    for column in return_columns:
        series = window[column]

        if (
            not pd.api.types.is_numeric_dtype(series)
            or pd.api.types.is_bool_dtype(series)
            or pd.api.types.is_complex_dtype(series)
        ):
            raise TypeError(
                f"{column} must contain real numeric returns."
            )

    return_values = window[
        return_columns
    ].to_numpy(dtype=np.float64)

    if not np.isfinite(return_values).all():
        raise ValueError(
            "Aligned returns contain non-finite values."
        )

    portfolio_returns = return_values[:, 0]
    benchmark_returns = return_values[:, 1]
    active_returns = portfolio_returns - benchmark_returns

    if (
        np.any(portfolio_returns < -1.0)
        or np.any(benchmark_returns < -1.0)
    ):
        raise ValueError(
            "Portfolio simple returns cannot be below -100%."
        )

    np.testing.assert_allclose(
        active_returns,
        return_values[:, 2],
        rtol=1e-10,
        atol=reconciliation_atol,
        err_msg="Arithmetic active-return identity failed.",
    )

    window_dates = pd.DatetimeIndex(
        window["date"]
    )

    # --------------------------------------------------------
    # Select dated holdings for both portfolios
    # --------------------------------------------------------

    holding_columns = [
        "portfolio_id",
        "date",
        "security_id",
        "weight_bop",
        "return_total",
        "sector",
    ]

    has_saved_contributions = (
        "contribution_total" in holdings_df.columns
    )

    if has_saved_contributions:
        holding_columns.append("contribution_total")

    selected_holdings = holdings_df.loc[
        holdings_df["portfolio_id"].isin(
            [portfolio_id, benchmark_portfolio_id]
        ),
        holding_columns,
    ].copy()

    selected_holdings["date"] = pd.to_datetime(
        selected_holdings["date"],
        errors="raise",
    )

    selected_holdings = selected_holdings.loc[
        selected_holdings["date"].isin(window_dates)
    ].copy()

    if (
        selected_holdings.empty
        or selected_holdings["date"].isna().any()
        or selected_holdings["security_id"].isna().any()
    ):
        raise ValueError(
            "Selected holdings have missing identifiers."
        )

    if selected_holdings.duplicated(
        ["portfolio_id", "date", "security_id"]
    ).any():
        raise ValueError(
            "Duplicate portfolio/date/security holdings."
        )

    for column in ("weight_bop", "return_total"):
        series = selected_holdings[column]

        if (
            not pd.api.types.is_numeric_dtype(series)
            or pd.api.types.is_bool_dtype(series)
            or pd.api.types.is_complex_dtype(series)
        ):
            raise TypeError(
                f"Holdings {column} must be real numeric."
            )

        if not np.isfinite(
            series.to_numpy(dtype=np.float64)
        ).all():
            raise ValueError(
                f"Holdings {column} contains invalid values."
            )

    if selected_holdings["return_total"].lt(-1.0).any():
        raise ValueError(
            "Held-security return below -100%."
        )

    # --------------------------------------------------------
    # Preserve original sector labels except genuine missing
    # --------------------------------------------------------

    if any(
        not (pd.isna(value) or isinstance(value, str))
        for value in selected_holdings["sector"]
    ):
        raise TypeError(
            "Recorded sector labels must be strings or missing."
        )

    missing_sector = (
        selected_holdings["sector"].isna()
        | selected_holdings["sector"].map(
            lambda value: (
                isinstance(value, str)
                and not value.strip()
            )
        )
    )

    if selected_holdings["sector"].eq(
        missing_sector_label
    ).fillna(False).any():
        raise ValueError(
            "Missing-sector classification conflicts "
            "with an existing recorded sector label."
        )

    selected_holdings["risk_sector"] = (
        selected_holdings["sector"].astype(object)
    )

    selected_holdings.loc[
        missing_sector,
        "risk_sector",
    ] = missing_sector_label

    selected_holdings["missing_sector_label"] = (
        missing_sector.to_numpy(dtype=bool)
    )

    # --------------------------------------------------------
    # Reconstruct security contributions
    # --------------------------------------------------------

    selected_holdings["calculated_contribution"] = (
        selected_holdings["weight_bop"].to_numpy(
            dtype=np.float64
        )
        * selected_holdings["return_total"].to_numpy(
            dtype=np.float64
        )
    )

    calculated = selected_holdings[
        "calculated_contribution"
    ].to_numpy(dtype=np.float64)

    if not np.isfinite(calculated).all():
        raise ArithmeticError(
            "Non-finite calculated security contributions."
        )

    maximum_saved_error = np.nan

    if has_saved_contributions:
        saved_series = selected_holdings[
            "contribution_total"
        ]

        if (
            not pd.api.types.is_numeric_dtype(saved_series)
            or not np.isfinite(
                saved_series.to_numpy(dtype=np.float64)
            ).all()
        ):
            raise ValueError(
                "Invalid saved holding contributions."
            )

        saved = saved_series.to_numpy(dtype=np.float64)

        maximum_saved_error = float(
            np.max(np.abs(calculated - saved))
        )

        np.testing.assert_allclose(
            calculated,
            saved,
            rtol=1e-10,
            atol=reconciliation_atol,
            err_msg="Saved contribution reconciliation failed.",
        )

    # --------------------------------------------------------
    # Assess recorded sector-label consistency
    # --------------------------------------------------------

    security_sector_counts = (
        selected_holdings
        .groupby(
            ["portfolio_id", "security_id"]
        )["risk_sector"]
        .nunique()
    )

    changed_security_sector_assignments = int(
        security_sector_counts.gt(1).sum()
    )

    conflicting_cross_portfolio_dates = int(
        selected_holdings
        .groupby(["date", "security_id"])["risk_sector"]
        .nunique()
        .gt(1)
        .sum()
    )

    # Changes and cross-portfolio disagreements are
    # diagnostic. Dated recorded labels are retained.

    # --------------------------------------------------------
    # Construct and reconcile daily sector contribution panels
    # --------------------------------------------------------

    sector_panels = {}
    sector_holding_counts = {}
    leg_reconciliation_errors = {}

    for leg_id, leg_reference_returns in (
        (portfolio_id, portfolio_returns),
        (benchmark_portfolio_id, benchmark_returns),
    ):

        leg = selected_holdings.loc[
            selected_holdings["portfolio_id"].eq(leg_id)
        ].copy()

        if (
            leg.empty
            or not window_dates.isin(leg["date"]).all()
        ):
            raise ValueError(
                f"Incomplete holdings calendar: {leg_id}"
            )

        sector_daily = (
            leg.groupby(
                ["date", "risk_sector"],
                as_index=False,
                sort=False,
            )["calculated_contribution"]
            .sum()
        )

        sector_panel = (
            sector_daily
            .pivot(
                index="date",
                columns="risk_sector",
                values="calculated_contribution",
            )
            .reindex(index=window_dates)
            .fillna(0.0)
        )

        if not np.isfinite(
            sector_panel.to_numpy(dtype=np.float64)
        ).all():
            raise ValueError(
                "Non-finite sector contribution panel."
            )

        reconstructed_returns = (
            sector_panel.to_numpy(dtype=np.float64)
            .sum(axis=1)
        )

        maximum_error = float(
            np.max(
                np.abs(
                    reconstructed_returns
                    - leg_reference_returns
                )
            )
        )

        np.testing.assert_allclose(
            reconstructed_returns,
            leg_reference_returns,
            rtol=1e-10,
            atol=reconciliation_atol,
            err_msg=(
                f"Sector/portfolio return mismatch: {leg_id}"
            ),
        )

        sector_panels[leg_id] = sector_panel

        sector_holding_counts[leg_id] = (
            leg.groupby("risk_sector")
            .size()
        )

        leg_reconciliation_errors[leg_id] = maximum_error

    # --------------------------------------------------------
    # Align sector universes and derive active contributions
    # --------------------------------------------------------

    active_sector_panel = sector_panels[portfolio_id]

    benchmark_sector_panel = sector_panels[
        benchmark_portfolio_id
    ]

    all_sectors = sorted(
        set(active_sector_panel.columns)
        | set(benchmark_sector_panel.columns)
    )

    active_sector_union = active_sector_panel.reindex(
        columns=all_sectors,
        fill_value=0.0,
    )

    benchmark_sector_union = (
        benchmark_sector_panel.reindex(
            columns=all_sectors,
            fill_value=0.0,
        )
    )

    active_contribution_panel = (
        active_sector_union - benchmark_sector_union
    )

    reconstructed_active_returns = (
        active_contribution_panel.to_numpy(
            dtype=np.float64
        ).sum(axis=1)
    )

    maximum_active_error = float(
        np.max(
            np.abs(
                reconstructed_active_returns
                - active_returns
            )
        )
    )

    np.testing.assert_allclose(
        reconstructed_active_returns,
        active_returns,
        rtol=1e-10,
        atol=reconciliation_atol,
        err_msg=(
            "Active sector contributions do not "
            "reconcile to arithmetic active returns."
        ),
    )

    # --------------------------------------------------------
    # Define the three historical risk series
    # --------------------------------------------------------

    risk_specs = [
        (
            "ABSOLUTE",
            portfolio_id,
            None,
            active_sector_panel,
            portfolio_returns,
            "PORTFOLIO_VOLATILITY",
        ),
        (
            "ABSOLUTE",
            benchmark_portfolio_id,
            None,
            benchmark_sector_panel,
            benchmark_returns,
            "PORTFOLIO_VOLATILITY",
        ),
        (
            "ACTIVE_ARITHMETIC",
            portfolio_id,
            benchmark_portfolio_id,
            active_contribution_panel,
            active_returns,
            "TRACKING_ERROR",
        ),
    ]

    detail_frames = []
    diagnostic_records = []

    # --------------------------------------------------------
    # Attribute variance and volatility for each risk series
    # --------------------------------------------------------

    for (
        risk_type,
        result_portfolio_id,
        result_benchmark_id,
        sector_panel,
        reference_returns,
        risk_measure,
    ) in risk_specs:

        sectors = sector_panel.columns.tolist()

        contribution_matrix = sector_panel.to_numpy(
            dtype=np.float64
        )

        reference_differences = (
            reference_returns - reference_returns[0]
        )

        reference_centered = (
            reference_differences
            - reference_differences.mean()
        )

        contribution_differences = (
            contribution_matrix
            - contribution_matrix[:1, :]
        )

        contribution_centered = (
            contribution_differences
            - contribution_differences.mean(
                axis=0,
                keepdims=True,
            )
        )

        denominator = observations - variance_ddof

        variance_daily = float(
            np.dot(
                reference_centered,
                reference_centered,
            ) / denominator
        )

        covariance_daily = (
            contribution_centered.T @ reference_centered
        ) / denominator

        component_variance = (
            covariance_daily * annualization_sessions
        )

        variance_annualized = (
            variance_daily * annualization_sessions
        )

        risk_annualized = float(
            np.sqrt(variance_annualized)
        )

        if risk_annualized > 0.0:
            component_risk = (
                component_variance / risk_annualized
            )

            risk_share = (
                component_risk / risk_annualized
            )

            status = "PASS"

        else:
            component_risk = np.full(
                len(sectors),
                np.nan,
            )

            risk_share = np.full(
                len(sectors),
                np.nan,
            )

            status = "UNAVAILABLE_ZERO_REFERENCE_VOLATILITY"

        # ----------------------------------------------------
        # Reconcile component sums
        # ----------------------------------------------------

        np.testing.assert_allclose(
            component_variance.sum(),
            variance_annualized,
            rtol=1e-10,
            atol=1e-12,
        )

        mean_contributions = (
            contribution_matrix.mean(axis=0)
        )

        np.testing.assert_allclose(
            mean_contributions.sum(),
            reference_returns.mean(),
            rtol=1e-10,
            atol=1e-12,
        )

        if status == "PASS":
            np.testing.assert_allclose(
                component_risk.sum(),
                risk_annualized,
                rtol=1e-10,
                atol=1e-12,
            )

            np.testing.assert_allclose(
                risk_share.sum(),
                1.0,
                rtol=1e-10,
                atol=1e-12,
            )

        # ----------------------------------------------------
        # Preserve attribution provenance and sector coverage
        # ----------------------------------------------------

        portfolio_counts = (
            sector_holding_counts[portfolio_id]
            .reindex(sectors, fill_value=0)
            .to_numpy(dtype=np.int64)
        )

        benchmark_counts = (
            sector_holding_counts[benchmark_portfolio_id]
            .reindex(sectors, fill_value=0)
            .to_numpy(dtype=np.int64)
        )

        detail_frames.append(
            pd.DataFrame(
                {
                    "risk_type": risk_type,
                    "portfolio_id": result_portfolio_id,
                    "benchmark_portfolio_id": (
                        result_benchmark_id
                    ),
                    "sector": sectors,
                    "observations": observations,
                    "risk_measure": risk_measure,
                    "portfolio_holding_rows": (
                        portfolio_counts
                    ),
                    "benchmark_holding_rows": (
                        benchmark_counts
                    ),
                    "mean_sector_contribution_daily": (
                        mean_contributions
                    ),
                    "covariance_with_reference_daily": (
                        covariance_daily
                    ),
                    "contribution_to_annualized_variance": (
                        component_variance
                    ),
                    "contribution_to_annualized_risk": (
                        component_risk
                    ),
                    "risk_share": risk_share,
                    "sector_label_status": [
                        (
                            "MISSING_SOURCE_LABEL"
                            if sector == missing_sector_label
                            else "RECORDED_LABEL_NOT_PIT_CERTIFIED"
                        )
                        for sector in sectors
                    ],
                    "status": status,
                }
            )
        )

        if risk_type == "ACTIVE_ARITHMETIC":
            return_error = maximum_active_error

        else:
            return_error = leg_reconciliation_errors[
                result_portfolio_id
            ]

        diagnostic_records.append(
            {
                "risk_type": risk_type,
                "portfolio_id": result_portfolio_id,
                "benchmark_portfolio_id": (
                    result_benchmark_id
                ),
                "risk_measure": risk_measure,
                "observations": observations,
                "sector_count": len(sectors),
                "variance_daily": variance_daily,
                "variance_annualized": variance_annualized,
                "risk_annualized": risk_annualized,
                "sum_component_variance": float(
                    component_variance.sum()
                ),
                "sum_component_risk": (
                    float(component_risk.sum())
                    if status == "PASS"
                    else np.nan
                ),
                "sum_risk_shares": (
                    float(risk_share.sum())
                    if status == "PASS"
                    else np.nan
                ),
                "negative_risk_contributors": int(
                    np.count_nonzero(
                        component_variance < 0.0
                    )
                ),
                "maximum_daily_return_error": return_error,
                "maximum_saved_contribution_error": (
                    maximum_saved_error
                ),
                "missing_sector_holding_rows": int(
                    selected_holdings[
                        "missing_sector_label"
                    ].sum()
                ),
                "changed_security_sector_assignments": (
                    changed_security_sector_assignments
                ),
                "conflicting_cross_portfolio_sector_dates": (
                    conflicting_cross_portfolio_dates
                ),
                "sector_label_coverage_status": (
                    "PARTIAL_MISSING_SOURCE_LABELS"
                    if selected_holdings[
                        "missing_sector_label"
                    ].any()
                    else "ALL_SOURCE_LABELS_PRESENT"
                ),
                "status": status,
            }
        )

    return (
        pd.concat(detail_frames, ignore_index=True),
        pd.DataFrame(diagnostic_records),
    )
