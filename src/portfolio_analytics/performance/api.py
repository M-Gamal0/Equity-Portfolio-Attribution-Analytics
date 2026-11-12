"""Independent performance application entry point."""

import pandas as pd

from portfolio_analytics.common.selection import bind, require_inception, select_pair
from portfolio_analytics.config import AnalysisSettings
from . import (inputs, wealth, periods, relative, windows, rolling, capture,
               risk_adjusted, contributions, return_basis, distributions)


def run(*, portfolio_returns: pd.DataFrame, portfolio_holdings: pd.DataFrame,
        factor_returns: pd.DataFrame, settings: AnalysisSettings) -> dict[str, pd.DataFrame]:
    """Return performance tables using BOP contributions and calendar windows.

    No I/O is performed. RF availability retains its own explicit cutoff.
    """
    returns = inputs.validate_performance_returns(select_pair(portfolio_returns, settings))
    holdings = select_pair(portfolio_holdings, settings)
    baseline = require_inception(settings)
    baselines = dict.fromkeys(settings.portfolio_ids, baseline)
    pair = inputs.align_performance_pair(returns, settings.portfolio_id, settings.benchmark_id)
    definitions = windows.resolve_performance_windows(
        returns, as_of_date=settings.as_of_date, inception_baseline_dates=baselines,
        window_ids=settings.performance_windows)
    out = {"performance_window_definitions": definitions,
           "performance_daily": wealth.compute_wealth_index(returns),
           "performance_period": periods.aggregate_period_returns(returns),
           "performance_relative_daily": relative.compute_relative_performance(pair),
           "performance_relative_period": relative.compute_relative_period_performance(pair),
           "performance_window": windows.compute_absolute_performance(
               returns, definitions, arithmetic_trading_days=settings.annualization_sessions),
           "performance_relative_window": windows.compute_relative_window_performance(
               pair, definitions, arithmetic_trading_days=settings.annualization_sessions),
           "performance_market_capture": capture.compute_market_capture(
               pair, definitions, portfolio_id=settings.portfolio_id,
               benchmark_portfolio_id=settings.benchmark_id)}
    rolled = rolling.compute_rolling_performance(returns, inception_baselines=baselines)
    rel_rolled = rolling.compute_rolling_relative_performance(
        rolled, portfolio_id=settings.portfolio_id, benchmark_portfolio_id=settings.benchmark_id)
    out.update(performance_rolling=rolled, performance_rolling_relative=rel_rolled,
               performance_rolling_consistency=rolling.summarize_rolling_outperformance(rel_rolled))
    rf_aligned = risk_adjusted.align_performance_risk_free(pair, factor_returns)
    out["performance_rf_coverage"] = risk_adjusted.assess_performance_rf_coverage(rf_aligned, definitions)
    rf_end = pd.to_datetime(factor_returns["date"]).max()
    cutoff = min(rf_end, pd.Timestamp(settings.as_of_date))
    rf_windows = windows.resolve_performance_windows(
        returns, as_of_date=cutoff, inception_baseline_dates=baselines,
        window_ids=settings.performance_windows)
    # Keep both requested-cutoff and RF-supported-cutoff results, as in Notebook 07.
    ratio_windows = pd.concat([definitions, rf_windows], ignore_index=True).drop_duplicates()
    out["performance_risk_adjusted"] = risk_adjusted.compute_risk_adjusted_performance(
        rf_aligned, ratio_windows, sessions_per_year=settings.annualization_sessions)
    daily = contributions.compute_security_contributions(holdings)
    out["performance_security_contributions_daily"] = daily
    bind(out, ("performance_security_contributions_period", "performance_security_contribution_reconciliation"),
         contributions.link_security_contributions(daily, returns, definitions))
    linked = out["performance_security_contributions_period"]
    out["performance_sector_contributions_period"] = contributions.aggregate_sector_contributions(linked)
    out["performance_security_contribution_rankings"] = contributions.rank_linked_security_contributions(linked)
    if "return_price" in returns:
        out["performance_return_basis_window"] = return_basis.compute_return_basis_windows(returns, definitions)
    bind(out, ("performance_rolling_relative_chart", "performance_rolling_distribution_summary"),
         distributions.prepare_rolling_relative_visuals(
             rel_rolled, portfolio_id=settings.portfolio_id, benchmark_portfolio_id=settings.benchmark_id))
    return out
