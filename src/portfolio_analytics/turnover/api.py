"""Independent scheduled-turnover and price-drift application."""

import pandas as pd

from portfolio_analytics.common.selection import bind, select_pair
from portfolio_analytics.config import AnalysisSettings
from . import inputs, schedule, implementation, trades, sectors, drift, transitions, periods, concentration, costs


def run(*, portfolio_holdings: pd.DataFrame, rebalance_targets: pd.DataFrame,
        settings: AnalysisSettings) -> dict[str, pd.DataFrame]:
    """Return scheduled turnover separately from passive drift and hypothetical costs."""
    holdings = select_pair(portfolio_holdings, settings)
    targets = select_pair(rebalance_targets, settings, date_column="selection_date")
    # A target beyond the selected return cutoff has not become effective in this run.
    if "effective_rebalance_date" in targets:
        future = pd.to_datetime(targets["effective_rebalance_date"]).gt(pd.Timestamp(settings.as_of_date))
        targets.loc[future, "effective_rebalance_date"] = pd.NaT
    out = {"turnover_input_validation": inputs.validate_turnover_inputs(
        holdings, targets, require_sector=True, require_full_investment=settings.fully_invested)}
    events = schedule.build_rebalance_event_schedule(holdings, targets)
    out["rebalance_events"] = events
    bind(out, ("rebalance_target_security", "rebalance_target_reconciliation"),
         implementation.reconcile_rebalance_target_implementation(holdings, targets, events, require_sector_match=True))
    bind(out, ("rebalance_security_changes", "rebalance_turnover_events"),
         trades.calculate_scheduled_rebalance_turnover_portable(holdings, targets, events))
    event_turnover = out["rebalance_turnover_events"]
    sec = out["rebalance_security_changes"]
    bind(out, ("rebalance_sector_turnover", "rebalance_sector_reconciliation"),
         sectors.calculate_sector_rebalance_turnover(sec, holdings, targets, event_turnover))
    bind(out, ("daily_weight_drift", "daily_weight_drift_summary"), drift.calculate_daily_price_weight_drift(holdings))
    bind(out, ("daily_weight_transition_security", "daily_rebalancing_diagnostics"),
         transitions.reconcile_daily_weight_transitions(holdings, event_turnover, sec, out["daily_weight_drift_summary"]))
    bind(out, ("quarterly_turnover", "annual_turnover", "portfolio_turnover_summary"),
         periods.summarize_rebalance_turnover_periods_portable(event_turnover, out["daily_rebalancing_diagnostics"]))
    bind(out, ("rebalance_security_concentration", "rebalance_sector_concentration", "rebalance_event_concentration",
               "cumulative_security_turnover_concentration", "cumulative_sector_turnover_concentration", "portfolio_turnover_concentration"),
         concentration.analyze_scheduled_turnover_concentration(sec, out["rebalance_sector_turnover"], event_turnover))
    bind(out, ("rebalance_event_cost_sensitivity", "quarterly_cost_sensitivity", "annual_cost_sensitivity", "portfolio_cost_sensitivity"),
         costs.calculate_turnover_unit_cost_sensitivity(event_turnover, out["quarterly_turnover"],
             out["annual_turnover"], out["portfolio_turnover_summary"]))
    return out
