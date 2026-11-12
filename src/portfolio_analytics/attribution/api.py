"""Independent Brinson and security-attribution application."""

import pandas as pd

from portfolio_analytics.common.selection import bind, require_inception, select_pair
from portfolio_analytics.config import AnalysisSettings
from . import inputs, security, sectors, brinson, carino, menchero, windows, periods, linking, summaries, cumulative


def run(*, portfolio_returns: pd.DataFrame, portfolio_holdings: pd.DataFrame,
        settings: AnalysisSettings) -> dict[str, pd.DataFrame]:
    """Compute daily and linked attribution, preserving both validated linkers."""
    returns = select_pair(portfolio_returns, settings)
    holdings = select_pair(portfolio_holdings, settings)
    pair_args = dict(portfolio_id=settings.portfolio_id, benchmark_id=settings.benchmark_id)
    out = {}
    pair, prepared, coverage = inputs.prepare_attribution_inputs(
        returns, holdings, **pair_args, require_fully_invested=settings.fully_invested,
        require_static_sectors=settings.fixed_sectors)
    out["attribution_sector_coverage"] = coverage
    bind(out, ("security_contributions_daily", "security_active_contributions_daily", "security_active_reconciliation"),
         security.calculate_security_active_contributions(prepared, pair, **pair_args))
    sector, audit = sectors.aggregate_sector_returns(
        prepared, returns, portfolio_ids=settings.portfolio_ids,
        require_fully_invested=settings.fully_invested)
    out.update(sector_returns_daily=sector, sector_return_reconciliation=audit)
    comparison = sectors.build_brinson_sector_comparison(sector, pair, **pair_args)
    bind(out, ("brinson_sector_daily", "brinson_total_daily"), brinson.calculate_daily_brinson_fachler(comparison))
    definitions = windows.build_attribution_reporting_windows(
        pair, **pair_args, inception_valuation_date=require_inception(settings),
        trailing_years=settings.reporting_years)
    out["attribution_reporting_windows"] = definitions
    bind(out, ("brinson_sector_period", "brinson_total_period"), periods.build_period_attribution(
        out["brinson_sector_daily"], definitions,
        linkers={"carino": carino.link_attribution_carino, "menchero": menchero.link_attribution_menchero}))
    bind(out, ("brinson_method_comparison_period", "brinson_method_comparison_sector"),
         periods.compare_attribution_linking_methods(out["brinson_total_period"], out["brinson_sector_period"]))
    bind(out, ("security_contributions_period", "security_contributions_period_reconciliation"),
         linking.link_security_contributions_by_period(
             out["security_active_contributions_daily"], pair, out["brinson_total_period"]))
    bind(out, ("attribution_period_report", "attribution_sector_report", "attribution_security_report"),
         summaries.build_attribution_reporting_tables(out["brinson_total_period"],
             out["brinson_sector_period"], out["security_contributions_period"]))
    out["brinson_cumulative_daily"] = cumulative.calculate_cumulative_brinson_attribution(out["brinson_total_daily"])
    return out
