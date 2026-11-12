"""Turnover event gates, calendar labels, complete ranking and gross-cost units."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("streamlit")


@pytest.fixture
def ui(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "app"))
    import turnover_details
    return turnover_details


def test_only_passed_effective_finite_events_are_measured(ui):
    source = pd.DataFrame({"portfolio_id": ["Active portfolio"] * 4,
                           "event_status": ["EFFECTIVE_REBALANCE", "PENDING_AFTER_CUTOFF", "INITIAL_ESTABLISHMENT", "EFFECTIVE_REBALANCE"],
                           "turnover_status": ["PASSED", "NOT_ASSESSED_PENDING_SELECTION", "NOT_ASSESSED_INITIAL_ESTABLISHMENT", "FAILED"],
                           "effective_rebalance_date": pd.to_datetime(["2020-01-02", None, "2019-01-02", "2020-04-01"]),
                           **{c: [.2, np.nan, np.nan, .3] for c in ["one_way_turnover", "total_purchases", "total_sales", "gross_two_way_turnover"]}})
    assert len(ui.measured_events(source)) == 1
    scope = ui.event_scope(source).iloc[0]
    assert (scope.measured_rebalances, scope.initial_establishment_events, scope.pending_events, scope.other_or_ineligible_events) == (1, 1, 1, 1)
    assert pd.isna(source.one_way_turnover.iloc[1])


def test_cross_year_event_labels_put_years_beneath_months(ui):
    axis = ui.event_axis(pd.Series(pd.to_datetime(["2017-01-03", "2017-04-03", "2018-01-02"]))).to_dict()
    assert '["Jan", "2017"]' in axis["labelExpr"]
    assert '["Jan", "2018"]' in axis["labelExpr"]
    assert axis["labelLineHeight"] == 14


def test_turnover_rank_balances_reconcile_and_duplicate_tickers_stay_separate(ui, monkeypatch):
    source = pd.DataFrame({"security_id": [str(i) for i in range(12)], "security_label": ["SAME"] * 12,
                           "cumulative_one_way": [.1] * 12, "portfolio_turnover_share": [1 / 12] * 12, "cumulative_rank": range(1, 13)})
    ranked, balance = ui.turnover_ranking(source, 1.2)
    assert balance.security_count.tolist() == [10, 2, 12]
    assert balance.turnover_share.iloc[:2].sum() == pytest.approx(1.)
    assert balance.cumulative_one_way_turnover.iloc[:2].sum() == pytest.approx(1.2)
    specs = []
    monkeypatch.setattr(ui, "render_chart", lambda chart, **kwargs: specs.append(chart.to_dict()))
    ui.concentration_chart(ranked, "Active portfolio")
    assert specs[0]["encoding"]["y"]["field"] == "security_key"
    with pytest.raises(ValueError, match="reconcile"):
        ui.turnover_ranking(source, 2.)


def test_cost_scenarios_scale_saved_two_sided_coefficient_in_rate_order(ui):
    source = pd.DataFrame({"portfolio_id": ["Active portfolio", "Benchmark"], "nav_bp_per_1bp_traded": [.6, .2],
                           "cost_sensitivity_status": ["UNIT_RATE_SENSITIVITY_ONLY"] * 2})
    scenarios = ui.cost_scenarios(source)
    active = scenarios.loc[scenarios.portfolio_id.eq("Active portfolio")]
    assert active.assumed_traded_weight_rate_bp.tolist() == [1, 5, 10, 25]
    assert active.arithmetic_nav_cost_bp.tolist() == [.6, 3., 6., 15.]
    assert source.nav_bp_per_1bp_traded.tolist() == [.6, .2]


def test_unavailable_costs_are_not_converted_to_zero(ui):
    source = pd.DataFrame({"portfolio_id": ["Active portfolio"], "nav_bp_per_1bp_traded": [0.], "cost_sensitivity_status": ["NOT_ASSESSED"]})
    assert ui.cost_scenarios(source).arithmetic_nav_cost_bp.isna().all()
    zero = source.assign(cost_sensitivity_status="UNIT_RATE_SENSITIVITY_ONLY")
    assert ui.cost_scenarios(zero).arithmetic_nav_cost_bp.eq(0.).all()
