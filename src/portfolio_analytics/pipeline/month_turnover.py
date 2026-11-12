"""Calendar-inclusive pending decisions, distinct from observed implementation."""
import numpy as np
import pandas as pd

PENDING_MEASUREMENTS = [
    "total_purchases", "total_sales", "gross_two_way_turnover", "one_way_turnover",
    "net_weight_change", "new_positions", "exited_positions", "increased_positions", "decreased_positions",
    "securities_compared", "implemented_weight_sum", "max_abs_implementation_weight_error",
    "buy_sell_balance_error", "turnover_identity_error", "gross_flow_identity_error",
]


def select_turnover_events(events, window):
    start, end = pd.Timestamp(window["first_return_date"]), pd.Timestamp(window["last_return_date"])
    selection_start, selection_end = pd.Timestamp(window["calendar_start"]), pd.Timestamp(window["calendar_end"])
    effective = pd.to_datetime(events.effective_rebalance_date)
    decisions = pd.to_datetime(events.selection_date)
    pending = decisions.between(selection_start, selection_end) & (effective.gt(end) | effective.isna())
    selected = events.loc[effective.between(start, end) | pending].copy()
    future = pd.to_datetime(selected.effective_rebalance_date).gt(end)
    selected.loc[future, "effective_rebalance_date"] = pd.NaT
    selected.loc[future, "event_status"] = "PENDING_AFTER_CUTOFF"
    selected.loc[future, "turnover_status"] = "NOT_ASSESSED_PENDING_SELECTION"
    selected.loc[future, [c for c in PENDING_MEASUREMENTS if c in selected]] = np.nan
    return selected


def validate_pending_events(source, selected, window):
    expected = select_turnover_events(source, window)
    expected = expected.loc[expected.event_status.eq("PENDING_AFTER_CUTOFF")]
    actual = selected.loc[selected.event_status.eq("PENDING_AFTER_CUTOFF")]
    keys = ["portfolio_id", "selection_date"]
    if expected.duplicated(keys).any() or actual.duplicated(keys).any():
        raise ValueError("Pending decisions must have unique portfolio/selection keys.")
    expected_keys = set(map(tuple, expected[keys].itertuples(index=False, name=None)))
    actual_keys = set(map(tuple, actual[keys].itertuples(index=False, name=None)))
    if expected_keys != actual_keys:
        raise ValueError("Pending calendar-month selection completeness failed.")
    columns = [c for c in PENDING_MEASUREMENTS if c in actual]
    if actual[columns].notna().any().any() or actual.effective_rebalance_date.notna().any():
        raise ValueError("Pending selections contain future implementation values.")
    if not actual.turnover_status.eq("NOT_ASSESSED_PENDING_SELECTION").all():
        raise ValueError("Pending selections lack the saved unavailable status.")
    return {"domain": "turnover_pending_calendar_boundary", "status": "PASSED",
            "expected_pending_records": len(expected), "saved_pending_records": len(actual)}
