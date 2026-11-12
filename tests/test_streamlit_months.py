"""Selection and chart-boundary behavior without private data or an analysis run."""
import json
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app/streamlit_app.py"


@pytest.fixture
def ui(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "app"))
    import research_ui
    return research_ui


def test_risk_tails_select_exact_basis_and_keep_sample_gate(ui):
    from risk_details import tail_selection, eligible_tail
    rows = pd.DataFrame({"portfolio_id": ["Active portfolio"] * 3,
                         "confidence_level": [.95, .95, .99],
                         "return_kind": ["ARITHMETIC_ACTIVE", "SIMPLE_TOTAL", "SIMPLE_TOTAL"],
                         "coverage_status": ["PASS", "PASS", "LOW_EFFECTIVE_TAIL_SAMPLE"],
                         "var_loss_daily": [.01, .03, .05], "expected_shortfall_loss_daily": [.02, .04, .06]})
    total = tail_selection(rows, "Active portfolio", .95, "SIMPLE_TOTAL")
    assert total.var_loss_daily == .03 and eligible_tail(total)
    assert not eligible_tail(tail_selection(rows, "Active portfolio", .99, "SIMPLE_TOTAL"))
    assert tail_selection(pd.concat([rows, rows]), "Active portfolio", .95, "SIMPLE_TOTAL") is None
    assert not eligible_tail(None)


def test_risk_rolling_unavailable_numeric_estimates_are_not_plotted(ui):
    from risk_details import rolling_selection
    source = pd.DataFrame({"lookback_sessions": [252, 252, 63], "annualized_volatility": [.2, .3, .1],
                           "status": ["UNAVAILABLE_INSUFFICIENT_HISTORY", "PASS", "PASS"]})
    plotted, eligible = rolling_selection(source, 252, "annualized_volatility", "status")
    assert len(eligible) == 1 and pd.isna(plotted.annualized_volatility.iloc[0])
    assert source.annualized_volatility.iloc[0] == .2


def test_risk_ranking_keeps_sign_identity_and_remainder(ui):
    from risk_details import risk_evidence, risk_ranking
    source = pd.DataFrame({"security_id": ["a", "b", "c"], "security_label": ["SAME", "SAME", "OTHER"],
                           "contribution": [.02, -.03, .01], "share": [2., -3., 1.], "status": ["PASS"] * 3})
    evidence = risk_evidence(source, "contribution", "share")
    ranked, balance = risk_ranking(evidence, 2)
    assert ranked.security_id.tolist() == ["b", "a"]
    assert ranked.risk_contribution_pp.tolist() == [-3., 2.]
    assert balance.security_count.tolist() == [2, 1, 3]
    assert balance.risk_contribution_pp.tolist() == [-1., 1., 0.]
    assert source.contribution.tolist() == [.02, -.03, .01]
    with pytest.raises(ValueError, match="ineligible"):
        risk_ranking(evidence.assign(status="UNAVAILABLE"))


def test_risk_episode_uses_opening_anchor_and_does_not_invent_recovery(ui, monkeypatch):
    import risk_details
    charts = []
    monkeypatch.setattr(risk_details, "render_chart", lambda chart, **kwargs: charts.append(chart.to_dict()))
    source = pd.DataFrame({"portfolio_id": ["Active portfolio"], "episode_number": [1], "maximum_drawdown": [-.1],
                           "peak_date": [pd.NaT], "peak_reference": ["INITIAL_BASELINE"],
                           "trough_date": pd.to_datetime(["2017-02-01"]), "recovery_date": [pd.NaT],
                           "censor_date": pd.to_datetime(["2017-09-29"]), "episode_status": ["OPEN_AT_WINDOW_END"], "underwater_sessions": [251]})
    risk_details.episode_chart(source, {"opening_valuation_date": "2016-09-30"})
    row = next(iter(charts[0]["datasets"].values()))[0]
    assert row["peak_display_date"].startswith("2016-09-30")
    assert row["end_display_date"].startswith("2017-09-29") and row["recovery_date"] in (None, "")
    assert row["episode_end"] == "Open at cutoff"
    for layer in charts[0]["layer"]:
        assert "'%Y'" in layer["encoding"]["x"]["axis"]["labelExpr"]
    assert pd.isna(source.peak_date.iloc[0]) and pd.isna(source.recovery_date.iloc[0])


def test_risk_rolling_single_series_has_no_benchmark_legend_or_empty_chart(ui, monkeypatch):
    import risk_details
    charts = []
    monkeypatch.setattr(risk_details, "render_chart", lambda chart, **kwargs: charts.append(chart.to_dict()))
    source = pd.DataFrame({"date": pd.to_datetime(["2017-09-28", "2017-09-29"]), "portfolio_id": ["Active portfolio"] * 2,
                           "lookback_sessions": [252, 252], "tracking_error_annualized": [.1, .2], "status": ["UNAVAILABLE_INSUFFICIENT_HISTORY", "PASS"]})
    risk_details.rolling_chart(source, 252, "tracking_error_annualized", "status", "Tracking error")
    assert charts[0]["layer"][0]["encoding"]["color"]["legend"] is None
    charts.clear()
    risk_details.rolling_chart(source.assign(status="UNAVAILABLE_INSUFFICIENT_HISTORY"), 252, "tracking_error_annualized", "status", "Tracking error")
    assert not charts


def test_empty_month_workspace(monkeypatch, tmp_path):
    path = tmp_path / "missing.duckdb"
    monkeypatch.setenv("PORTFOLIO_MONTH_DATABASE", str(path))
    app = AppTest.from_file(str(APP)).run(timeout=20)
    assert not app.exception
    assert "ready" in app.info[0].value
    assert not path.exists()


def test_growth_keeps_opening_and_first_session(ui):
    paths = pd.DataFrame({"date": pd.to_datetime(["2020-01-02", "2020-12-31"]),
                          "portfolio_id": ["P", "P"], "wealth_index": [0.9, 1.2], "drawdown": [-0.1, 0.0]})
    plotted = ui.growth_frame(paths, {"opening_valuation_date": "2019-12-31"})
    assert plotted.date.iloc[0] == pd.Timestamp("2019-12-31")
    assert plotted.growth.tolist() == [100, 90, 120]
    assert plotted.drawdown.tolist() == [0, -0.1, 0]


def test_apply_months_updates_metrics_and_rejects_short_period(ui, monkeypatch, tmp_path):
    monkeypatch.setenv("PORTFOLIO_ANALYTICS_DATABASE", str(tmp_path / "missing_original.duckdb"))
    settings = {"portfolio_id": "P", "benchmark_id": "B"}
    monkeypatch.setattr(ui, "builds", lambda path: pd.DataFrame([{
        "build_id": "test", "config_json": json.dumps(settings), "snapshot_id": "fixture"}]))
    names = ["performance_window", "performance_relative_window", "performance_risk_adjusted", "performance_period"]
    catalog = pd.DataFrame({"domain": "performance", "scope": "PERIOD", "output_name": names})
    definitions = []
    for iid, count in [("2016-10_2026-09", 120), ("2020-01_2020-12", 12)]:
        definitions.append({"interval_id": iid, "definition_json": json.dumps({
            "opening_valuation_date": "2019-12-31", "first_return_date": "2020-01-02", "last_return_date": "2020-12-31",
            "inclusive_months": count, "observation_count": 252}), "validation_json": "[]"})
    monkeypatch.setattr(ui, "metadata", lambda *args: (catalog, pd.DataFrame(definitions)))

    def saved(path, build, iid, domain, name):
        ret = 0.2 if iid.startswith("2020") else 0.5
        if name == "risk_absolute_summary":
            return pd.DataFrame({"portfolio_id": ["P", "B"], "annualized_volatility": [0.25, 0.2], "maximum_drawdown": [-0.2, -0.1]})
        if name == "risk_active_summary":
            return pd.DataFrame({"tracking_error_annualized": [0.1], "tracking_error_status": ["PASS"]})
        if name == "attribution_sector_report":
            return pd.DataFrame({"linking_method": ["carino"], "sector": ["Sector"], "linked_total_effect": [ret - 0.1]})
        if name == "attribution_period_report":
            return pd.DataFrame({"linking_method": ["carino"], "linked_total_effect": [ret - 0.1],
                                 "compounded_active_return": [ret - 0.1], "reconciliation_error": [0.0]})
        if name == "performance_window":
            return pd.DataFrame({"portfolio_id": ["P", "B"], "period_return": [ret, 0.1],
                                 "annualized_return_cagr": [ret, 0.1], "cagr_status": ["CALCULATED"] * 2})
        if name == "performance_relative_window":
            return pd.DataFrame({"arithmetic_active_period_return": [ret - 0.1]})
        if name == "performance_risk_adjusted":
            return pd.DataFrame({"metric_name": ["portfolio_sharpe_ratio"], "metric_value": [1.2], "calculation_status": ["CALCULATED"]})
        if name == "risk_drawdown_daily":
            return pd.DataFrame({"date": pd.to_datetime(["2020-12-31"] * 2), "portfolio_id": ["P", "B"],
                                 "wealth_index": [1 + ret, 1.1], "drawdown": [0.0, 0.0]})
        if name == "performance_period":
            return pd.DataFrame({"period_type": ["monthly", "annual"] * 2, "portfolio_id": ["P", "P", "B", "B"],
                                 "period_start": pd.to_datetime(["2020-01-01"] * 4), "period_return": [0.02, ret, 0.01, 0.1],
                                 "observed_calendar_months": [1, 12, 1, 12]})
        raise AssertionError(name)

    monkeypatch.setattr(ui, "result", saved)
    app = AppTest.from_file(str(APP)).run(timeout=20)
    assert not app.exception
    assert app.metric[0].value == "50.00%"
    assert len(app.get("bidi_component")) == 4
    app.selectbox(key="start_month").set_value("January")
    app.selectbox(key="start_year").set_value(2020)
    app.selectbox(key="end_month").set_value("December")
    app.selectbox(key="end_year").set_value(2020)
    app.button[0].click().run(timeout=20)
    assert not app.exception
    assert app.metric[0].value == "20.00%"
    app.selectbox(key="end_month").set_value("November")
    app.button[0].click().run(timeout=20)
    assert "12 inclusive months" in app.error[0].value
    assert app.metric[0].value == "20.00%"
    app.switch_page("app_pages/overview.py").run(timeout=20)
    assert not app.exception
    assert app.session_state["month_window"] == "2020-01_2020-12"
    assert app.metric[0].value == "20.00%"
    # Entry-point widgets retain the staged controls as well as the applied interval.
    assert app.selectbox(key="start_year").value == 2020
    app.switch_page("app_pages/performance.py").run(timeout=20)
    assert not app.exception
    assert app.metric[0].value == "20.00%"
    app.selectbox(key="end_month").set_value("December")
    app.button[0].click().run(timeout=20)
    app.switch_page("app_pages/overview.py").run(timeout=20)
    assert not app.exception
    assert app.selectbox(key="start_month").value == "January"
    assert app.selectbox(key="start_year").value == 2020
    assert app.selectbox(key="end_month").value == "December"
    assert app.selectbox(key="end_year").value == 2020
    assert app.session_state["month_window"] == "2020-01_2020-12"
    assert app.metric[0].value == "20.00%"
    import streamlit as st
    assert st.get_option("client.disableDataExport") is True

    # Overview exposes the same selected-period chart data without losing the anchor.
    app.selectbox(key="overview_supporting_data").set_value("Growth and drawdown observations").run(timeout=20)
    assert not app.exception
    evidence = app.dataframe[-2].value  # Last dataframe belongs to methodology validations.
    assert len(evidence) == 4
    assert evidence.date.min() == pd.Timestamp("2019-12-31")
    assert set(evidence.loc[evidence.date.eq("2019-12-31"), "growth"]) == {100}


def test_table_mapping_keeps_estimators_and_rejects_old_summaries(ui):
    catalog = pd.DataFrame([
        {"domain": "risk", "output_name": "selected", "scope": "PERIOD", "schema_json": '{}'},
        {"domain": "risk", "output_name": "covariance", "scope": "CLOSING_MONTH", "schema_json": '{}'},
        {"domain": "risk", "output_name": "rolling", "scope": "SOURCE_REFERENCE", "schema_json": '{"date":"date"}'},
        {"domain": "risk", "output_name": "old_summary", "scope": "SOURCE_REFERENCE", "schema_json": '{"date":"date","window_id":"str"}'},
        {"domain": "risk", "output_name": "rolling_fit", "scope": "SOURCE_REFERENCE", "schema_json": '{"estimation_date":"date"}'},
    ])
    options = ui.detail_options(catalog, "risk")
    assert ("selected", "PERIOD", None) in options
    assert ("covariance", "CLOSING_MONTH", None) in options
    assert ("rolling", "SOURCE_REFERENCE", "date") in options
    assert ("rolling_fit", "SOURCE_REFERENCE", "estimation_date") in options
    assert all(name != "old_summary" for name, _, _ in options)


def test_date_axes_separate_months_from_years(ui):
    long = pd.DataFrame({"date": pd.to_datetime(["2016-09-30", "2026-09-30"])})
    axis = ui.date_axis(long).to_dict()
    assert axis["format"] == "%Y"
    assert all(value[5:10] == "01-01" for value in axis["values"])
    short = pd.DataFrame({"date": pd.to_datetime(["2020-01-02", "2020-12-31"])})
    axis = ui.date_axis(short).to_dict()
    assert axis["format"] == "%b"
    assert axis["title"] == "2020"
    cross_year = pd.DataFrame({"date": pd.to_datetime(["2019-09-30", "2021-03-31"])})
    axis = ui.date_axis(cross_year).to_dict()
    assert "'%b'" in axis["labelExpr"] and "'%Y'" in axis["labelExpr"]
    assert "2020-01-01T00:00:00" in axis["values"]
    assert "2021-01-01T00:00:00" in axis["values"]
    assert axis["title"] is None  # Individual year markers occupy the second label row.


def test_chart_precision_legends_and_event_grouping(ui, monkeypatch):
    charts = []
    monkeypatch.setattr(ui, "render_chart", lambda chart, **kwargs: charts.append(chart.to_dict()))
    frame = pd.DataFrame({"date": pd.to_datetime(["2020-01-02", "2020-12-31"]),
                          "portfolio_id": ["Active portfolio", "Benchmark"], "beta": [0.91, 1.12]})
    ui.line_chart(frame, "beta", "Exposure")
    spec = charts[-1]
    assert spec["encoding"]["y"]["axis"]["format"] == ".2f"
    assert spec["encoding"]["color"]["legend"]["orient"] == "top"
    assert spec["config"]["axis"]["labelFont"] == "Roboto"
    assert spec["config"]["view"]["stroke"] is None
    ui.grouped_date_bars(frame.rename(columns={"beta": "turnover"}), "date", "turnover", "portfolio_id", "Turnover")
    spec = charts[-1]
    assert spec["encoding"]["x"]["type"] == "ordinal"
    assert spec["encoding"]["xOffset"]["field"] == "portfolio_id"
    assert spec["encoding"]["y"]["stack"] is None
    ui.bars(pd.DataFrame({"security": ["P"], "contribution": [0.0012]}), "security", "contribution", "Contribution")
    assert charts[-1]["encoding"]["x"]["axis"]["format"] == ".1%"


def test_comparison_aligns_portfolios_and_preserves_eligibility(ui):
    from comparison import comparison_frames
    performance = pd.DataFrame({"portfolio_id": ["Active portfolio", "Benchmark"],
        "period_return": [1.0, 0.6], "annualized_return_cagr": [0.2, 0.1],
        "cagr_status": ["INSUFFICIENT_CALENDAR_DAYS", "CALCULATED"], "wealth_end_normalized": [2.0, 1.6]})
    risk = pd.DataFrame({"portfolio_id": ["Benchmark", "Active portfolio"],
                         "annualized_volatility": [0.15, 0.25], "maximum_drawdown": [-0.3, -0.2]})
    ratios = pd.DataFrame({"metric_name": ["portfolio_sharpe_ratio", "benchmark_sharpe_ratio"],
                           "metric_value": [0.75, 0.5], "calculation_status": ["CALCULATED", "CALCULATED"]})
    original = performance.copy(deep=True)
    frames = comparison_frames(performance, risk, ratios)
    returns = frames["Return and growth"].set_index("Metric")
    assert returns.loc["Total return", "Difference"] == "+40.00 pp"
    assert returns.loc["Ending value of 100", "Active portfolio"] == "200.00"
    assert returns.loc["CAGR", "Active portfolio"] == "Unavailable"
    assert returns.loc["CAGR", "Benchmark"] == "10.00%"
    assert returns.loc["CAGR", "Difference"] == "—"
    risks = frames["Risk and drawdowns"].set_index("Metric")
    assert risks.loc["Annualized volatility", "Active portfolio"] == "25.00%"
    assert risks.loc["Annualized volatility", "Benchmark"] == "15.00%"
    assert risks.loc["Maximum drawdown", "Difference"] == "+10.00 pp"
    assert frames["Risk-adjusted performance"].iloc[0]["Difference"] == "+0.250"
    assert all("CAGR status" not in f.columns for f in frames.values())
    pd.testing.assert_frame_equal(original, performance)


def test_overview_pp_evidence_preserves_saved_units_and_gates_active_risk(ui):
    from overview_details import attribution_evidence, active_risk_values
    source = pd.DataFrame({"sector": ["Communication Services", "Technology"],
                           "linked_total_effect": [0.125, -0.025], "brinson_total_pp": [12.5, -2.5],
                           "linked_allocation_effect": [0.1, -0.01], "linking_method": "carino",
                           "coverage_status": ["COMPLETE", "PARTIAL"]})
    original = source.copy(deep=True)
    result = attribution_evidence(source)
    assert result.brinson_total_pp.tolist() == [12.5, -2.5]  # Saved pp must never be multiplied again.
    assert result.allocation_pp.tolist() == [10, -1]
    assert result.coverage_status.tolist() == ["COMPLETE", "PARTIAL"]
    assert result.sector.tolist() == source.sector.tolist()
    pd.testing.assert_frame_equal(source, original)
    risk = pd.DataFrame({"tracking_error_annualized": [0.2], "tracking_error_status": ["INSUFFICIENT_HISTORY"]})
    ratios = pd.DataFrame({"metric_name": ["information_ratio"], "metric_value": [9], "calculation_status": ["UNDEFINED_RATIO"]})
    te, ir, te_status, ir_status = active_risk_values(risk, ratios)
    assert te is None and ir is None
    assert te_status == "INSUFFICIENT_HISTORY" and ir_status == "UNDEFINED_RATIO"


def test_performance_shared_scale_partial_years_and_capture_units(ui, monkeypatch):
    import performance_details as view
    monthly = pd.DataFrame({"portfolio_id": [owner for owner in ui.SERIES for _ in range(3)],
                            "period_type": "monthly", "period_start": pd.to_datetime(["2016-10-01", "2016-11-01", "2016-12-01"] * 2),
                            "period_return": [0.2, -0.05, 0.01, 0.1, -0.1, 0.02]})
    annual = pd.DataFrame({"portfolio_id": ui.SERIES, "period_type": "annual", "period_start": pd.to_datetime(["2016-01-01"] * 2),
                           "period_return": [0.1514, 0.0098], "observed_calendar_months": 3})
    frame = pd.concat([monthly, annual], ignore_index=True)
    original = frame.copy(deep=True)
    charts = []
    monkeypatch.setattr(view, "render_chart", lambda chart, **kwargs: charts.append(chart.to_dict()))
    for owner in ui.SERIES:
        view.monthly_chart(frame, owner)
    assert [c['layer'][0]['encoding']['color']['scale']['domain'] for c in charts] == [[-0.2, 0, 0.2]] * 2
    result = view.annual_frame(frame)
    assert result.calendar_label.tolist() == ["2016|Oct–Dec (3m)"] * 2
    assert sorted(result.period_return) == sorted(annual.period_return)  # No partial-year annualization.
    capture = view.capture_frame(pd.Series({"up_geometric_capture_pct": 120, "up_capture_status": "CALCULATED",
                                           "down_geometric_capture_pct": 999, "down_capture_status": "INSUFFICIENT_SAMPLE"}))
    assert capture.Regime.tolist() == ["Benchmark up days", "Benchmark down days"]
    assert capture['Capture (%)'].iloc[0] == 120  # Already in percent units.
    assert pd.isna(capture['Capture (%)'].iloc[1])
    pd.testing.assert_frame_equal(frame, original)


def test_performance_rolling_eligibility_empty_and_single_point(ui, monkeypatch):
    import performance_details as view
    frame = pd.DataFrame({"date": pd.to_datetime(["2020-01-02", "2020-01-03"]), "horizon_id": "12M",
                          "calculation_status": ["UNAVAILABLE_INPUT_WINDOW", "AVAILABLE"],
                          "geometric_relative_return": [999, 0.05]})
    original = frame.copy(deep=True)
    charts, messages = [], []
    monkeypatch.setattr(view, "render_chart", lambda chart, **kwargs: charts.append(chart.to_dict()))
    monkeypatch.setattr(view.st, "info", messages.append)
    monkeypatch.setattr(view.st, "caption", lambda *args: None)
    view.rolling_chart(frame.iloc[:1], "12M")
    assert not charts and "No eligible 12M" in messages[0]
    view.rolling_chart(frame, "12M")
    assert charts[0]["layer"][0]["mark"]["point"] is True
    assert "color" not in charts[0]["layer"][0]["encoding"]  # No phantom two-series legend.
    assert charts[0]["layer"][0]["mark"]["invalid"] == "break-paths-show-domains"
    pd.testing.assert_frame_equal(frame, original)


def test_attribution_pp_ranking_balance_and_benchmark_only_context(ui):
    from attribution_details import security_evidence, security_ranking, bridge_evidence
    source = pd.DataFrame({"security_id": range(13), "security_label": ["SAME"] * 13,
                           "linked_active_contribution": [-0.06, -0.05, -0.04, -0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
                           "linked_portfolio_leg": 0.1, "linked_benchmark_leg": 0.08,
                           "portfolio_held_days": [0] + [20] * 12, "benchmark_held_days": 20})
    original = source.copy(deep=True)
    evidence = security_evidence(source)
    assert evidence.active_contribution_pp.iloc[0] == -6
    assert evidence.holding_context.iloc[0] == "Benchmark only during selection"
    ranked, balance = security_ranking(evidence)
    assert len(ranked) == 10 and ranked.security_id.nunique() == 10  # Duplicate ticker labels never merge identities.
    assert balance.security_count.tolist() == [10, 3, 13]
    assert abs(balance.active_contribution_pp.iloc[:2].sum() - balance.active_contribution_pp.iloc[2]) < 1e-12
    pd.testing.assert_frame_equal(source, original)
    sectors = pd.DataFrame({"sector": ["A"], "linked_total_effect": [0.1], "brinson_total_pp": [10],
                            "linked_security_active_contribution": [0.12], "linked_benchmark_reference_bridge": [0.02]})
    bridge = bridge_evidence(sectors).iloc[0]
    assert bridge.brinson_total_pp == 10  # Existing pp values are not multiplied again.
    assert bridge.security_active_contribution_pp == bridge.brinson_total_pp + bridge.benchmark_reference_bridge_pp


def test_attribution_ranking_rejects_unavailable_values_and_duplicate_ids(ui):
    from attribution_details import security_ranking
    with pytest.raises(ValueError, match="incomplete"):
        security_ranking(pd.DataFrame({"security_id": [1], "active_contribution_pp": [float("nan")]}))
    with pytest.raises(ValueError, match="incomplete"):
        security_ranking(pd.DataFrame({"security_id": [1, 1], "active_contribution_pp": [1, 2]}))


def test_attribution_method_comparison_aligns_names_and_preserves_target(ui):
    from attribution_details import method_comparison
    report = pd.DataFrame({"linking_method": ["menchero", "carino"], "linked_allocation_effect": [0.2, 0.3],
                           "linked_selection_effect": [-0.15, -0.2], "linked_interaction_effect": [0.05, 0],
                           "linked_total_effect": [0.1, 0.1]})
    original = report.copy(deep=True)
    result = method_comparison(report).set_index("Component")
    assert result.loc["Allocation", "Carino (pp)"] == 30
    assert result.loc["Allocation", "Menchero (pp)"] == 20
    assert result.loc["Allocation", "Difference (pp)"] == -10
    assert result.loc["Linked total", "Difference (pp)"] == 0
    pd.testing.assert_frame_equal(report, original)
