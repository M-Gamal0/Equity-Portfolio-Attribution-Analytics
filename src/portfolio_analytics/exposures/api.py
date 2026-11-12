"""Independent holdings and exposure application."""

import pandas as pd

from portfolio_analytics.common.selection import bind, select_pair
from portfolio_analytics.config import AnalysisSettings
from . import inputs, security, sectors, active_share, concentration, overlap, changes, rankings, windows


def run(*, portfolio_holdings: pd.DataFrame, settings: AnalysisSettings) -> dict[str, pd.DataFrame]:
    """Compute both BOP/EOP exposures and strict calendar holdings windows."""
    holdings = select_pair(portfolio_holdings, settings)
    policy = dict(require_fully_invested=settings.fully_invested)
    pair = dict(portfolio_id=settings.portfolio_id, benchmark_portfolio_id=settings.benchmark_id)
    out = {"exposure_input_validation": inputs.validate_exposure_inputs(
        holdings, portfolio_ids=settings.portfolio_ids, require_aligned_calendars=True,
        require_fixed_sector_history=settings.fixed_sectors, **policy)}
    sec = security.prepare_security_exposures(holdings)
    sector = sectors.calculate_sector_exposures(sec)
    act_sec = security.calculate_active_security_exposures(sec, **pair, **policy)
    act_sector = sectors.calculate_active_sector_exposures(sector, **pair, **policy)
    share = active_share.calculate_active_share(act_sec, **policy)
    ov = overlap.calculate_holdings_overlap(act_sec, **policy)
    out.update(security_exposures=sec, sector_exposures=sector,
               active_security_exposures=act_sec, active_sector_exposures=act_sector,
               active_share=share, holdings_overlap=ov)
    bind(out, ("security_concentration", "top_k_concentration"), concentration.calculate_security_concentration(sec, **policy))
    bind(out, ("sector_concentration", "top_k_sector_concentration"), concentration.calculate_sector_concentration(sector, **policy))
    out["exposure_changes"] = changes.calculate_exposure_changes(sec, require_fixed_sector_history=settings.fixed_sectors, **policy)
    out["top_holdings"] = rankings.rank_portfolio_holdings(sec)
    out["active_security_rankings"] = rankings.rank_active_security_positions(act_sec)
    definitions = windows.resolve_exposure_reporting_windows(
        pd.DatetimeIndex(holdings["date"].unique()).sort_values(),
        as_of_date=pd.Timestamp(settings.as_of_date), trailing_years=settings.reporting_years)
    out["exposure_reporting_windows"] = definitions
    bind(out, ("portfolio_window_summary", "active_window_summary", "sector_tilt_window_summary"),
         windows.build_exposure_window_summaries(out["security_concentration"],
             out["sector_concentration"], out["top_k_concentration"], share, ov, act_sector, definitions))
    return out
