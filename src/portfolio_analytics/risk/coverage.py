"""Risk coverage: validated research calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd



# Research reference: 09_risk_analytics.ipynb, code cell index 1.
def assess_covariance_history_coverage(
    security_returns: pd.DataFrame,
    holdings: pd.DataFrame,
    reference_dates: pd.DatetimeIndex,
    portfolio_ids: tuple[str, ...],
    lookbacks: tuple[int, ...],
) -> pd.DataFrame:
    """
    Assess historical return coverage for current holdings.

    Parameters
    ----------
    security_returns:
        Standardized daily security returns. Requires
        security_id, date, and return_total.

    holdings:
        Current positive-EOP-weight holdings. Requires
        portfolio_id and security_id.

    reference_dates:
        Ordered, eligible portfolio trading dates ending
        at the risk estimation as-of date.

    portfolio_ids:
        Portfolio identifiers to assess independently.

    lookbacks:
        Required trading-observation counts.

    Returns
    -------
    pd.DataFrame
        One row per portfolio and lookback containing
        security counts, complete-panel observations,
        and coverage status.

    Notes
    -----
    Coverage is measured using exact reference dates.
    Missing returns remain missing.
    No covariance estimation is performed.
    """

    if not lookbacks or any(
        not isinstance(n, (int, np.integer)) or n < 2
        for n in lookbacks
    ):
        raise ValueError(
            "Lookbacks must contain integers >= 2."
        )

    if reference_dates.empty:
        raise ValueError("Reference calendar is empty.")

    if not reference_dates.is_unique:
        raise ValueError(
            "Reference calendar contains duplicate dates."
        )

    if not reference_dates.is_monotonic_increasing:
        raise ValueError(
            "Reference calendar must be sorted ascending."
        )

    max_lookback = max(lookbacks)
    history_dates = reference_dates[-max_lookback:]

    security_universe = holdings["security_id"].unique()

    relevant_returns = security_returns.loc[
        security_returns["security_id"].isin(
            security_universe
        )
        & security_returns["date"].isin(history_dates),
        ["date", "security_id", "return_total"],
    ].copy()

    relevant_returns["date"] = pd.to_datetime(
        relevant_returns["date"],
        errors="raise",
    )

    return_panel = relevant_returns.pivot(
        index="date",
        columns="security_id",
        values="return_total",
    )

    coverage_records = []

    for portfolio_id in portfolio_ids:

        portfolio_securities = (
            holdings.loc[
                holdings["portfolio_id"].eq(portfolio_id),
                "security_id",
            ]
            .drop_duplicates()
            .tolist()
        )

        if not portfolio_securities:
            raise ValueError(
                f"{portfolio_id}: no current securities."
            )

        for lookback in lookbacks:

            window_dates = reference_dates[-lookback:]

            panel = return_panel.reindex(
                index=window_dates,
                columns=portfolio_securities,
            )

            observed_by_security = (
                panel.notna().sum(axis=0)
            )

            complete_security_count = int(
                observed_by_security.eq(lookback).sum()
            )

            complete_panel_days = int(
                panel.notna().all(axis=1).sum()
            )

            if len(window_dates) < lookback:
                coverage_status = (
                    "INSUFFICIENT_PORTFOLIO_HISTORY"
                )

            elif complete_panel_days == lookback:
                coverage_status = "FULL_PANEL_COVERAGE"

            else:
                coverage_status = "PARTIAL_PANEL_COVERAGE"

            coverage_records.append(
                {
                    "portfolio_id": portfolio_id,
                    "as_of_date": window_dates.max(),
                    "lookback_sessions": lookback,
                    "reference_sessions": len(window_dates),
                    "security_count": len(
                        portfolio_securities
                    ),
                    "fully_observed_securities": (
                        complete_security_count
                    ),
                    "minimum_security_observations": (
                        int(observed_by_security.min())
                    ),
                    "complete_panel_sessions": (
                        complete_panel_days
                    ),
                    "coverage_status": coverage_status,
                }
            )

    return pd.DataFrame(coverage_records)
