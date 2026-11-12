"""Full-month boundaries, first-session returns and atomic interval storage."""

import json
from types import SimpleNamespace

import duckdb
import numpy as np
import pandas as pd
import pytest

from portfolio_analytics.common.month_windows import MonthWindow, enumerate_month_windows
from portfolio_analytics.database.month_results import MonthResultWriter, read_month_result, list_month_builds
from portfolio_analytics import performance
from portfolio_analytics.performance import compute_absolute_performance
from portfolio_analytics.pipeline.month_calculations import (
    performance_definitions, complete_month_cagr_policy, compute_month_performance,
)
from portfolio_analytics.pipeline.month_runner import validate_month_outputs, validate_month_cagr
from portfolio_analytics.risk import calculate_drawdown_path
from portfolio_analytics.factors.stable import fit_period_ols


def calendar():
    return pd.bdate_range("2016-10-03", "2026-09-30")


def cagr_sample(start="2022-01", end="2022-12"):
    # Synthetic sessions; the removed Good Friday reproduces March 2018's close.
    dates = calendar().difference(pd.DatetimeIndex(["2018-03-30"]))
    w = MonthWindow(pd.Period(start, "M"), pd.Period(end, "M")).resolve(dates, "2016-09-30")
    returns = pd.DataFrame({"portfolio_id": ["P"] * len(dates) + ["B"] * len(dates),
                           "date": list(dates) * 2,
                           "return_total": [.0002] * len(dates) + [.0001] * len(dates)})
    return w, returns, performance.align_performance_pair(returns, "P", "B")


@pytest.mark.parametrize("start,end,days", [
    ("2022-01", "2022-12", 364), ("2016-10", "2017-09", 364), ("2017-04", "2018-03", 363),
])
def test_full_twelve_month_cagr_with_short_trading_endpoint_duration(start, end, days):
    w, returns, pair = cagr_sample(start, end)
    definitions = performance_definitions(w, ("P", "B"))
    original = {
        "performance_window": performance.compute_absolute_performance(returns, definitions),
        "performance_relative_window": performance.compute_relative_window_performance(pair, definitions),
    }
    assert definitions.calendar_days.eq(days).all()
    assert all(frame.cagr_status.eq("NOT_ANNUALIZED_SHORT_WINDOW").all() for frame in original.values())
    corrected = compute_month_performance(returns, pair, w, ("P", "B"))
    for name, frame in corrected.items():
        assert frame.cagr_status.eq("CALCULATED").all()
        assert frame.inclusive_months.eq(12).all()
        assert frame.minimum_cagr_calendar_days.eq(days).all()
        unchanged = [c for c in original[name] if "cagr" not in c
                     and c != "annualized_geometric_relative_return"]
        pd.testing.assert_frame_equal(frame[unchanged], original[name][unchanged])
    absolute = corrected["performance_window"]
    expected = np.expm1(np.log1p(absolute.period_return) * 365.25 / days)
    np.testing.assert_allclose(absolute.annualized_return_cagr, expected, atol=1e-12, rtol=1e-10)
    validate_month_cagr(w, corrected, ("P", "B"))


@pytest.mark.parametrize("start,end", [("2017-01", "2018-01"), ("2021-10", "2026-09")])
def test_previously_eligible_and_public_five_year_cagr_are_numerically_unchanged(start, end):
    w, returns, pair = cagr_sample(start, end)
    definitions = performance_definitions(w, ("P", "B"))
    original = {"performance_window": performance.compute_absolute_performance(returns, definitions),
                "performance_relative_window": performance.compute_relative_window_performance(pair, definitions)}
    corrected = compute_month_performance(returns, pair, w, ("P", "B"))
    for name, frame in original.items():
        pd.testing.assert_frame_equal(corrected[name][frame.columns], frame)
    validate_month_cagr(w, corrected, ("P", "B"))


@pytest.mark.parametrize("change", ["short", "count", "boolean", "partial", "opening"])
def test_cagr_month_policy_rejects_short_or_malformed_intervals(change):
    w, _, _ = cagr_sample()
    if change == "short":
        w.update(calendar_end=pd.Timestamp("2022-11-30"), inclusive_months=11)
    elif change == "count":
        w["inclusive_months"] = 13
    elif change == "boolean":
        w["inclusive_months"] = True
    elif change == "partial":
        w["calendar_end"] = pd.Timestamp("2022-12-29")
    else:
        w["opening_valuation_date"] = pd.Timestamp("2022-01-01")
    with pytest.raises(ValueError):
        complete_month_cagr_policy(w)


@pytest.mark.parametrize("corruption", ["status", "absolute", "relative", "policy", "nan"])
def test_cagr_validation_rejects_missing_metrics_and_wrong_formula(corruption):
    w, returns, pair = cagr_sample()
    outputs = compute_month_performance(returns, pair, w, ("P", "B"))
    absolute, relative = outputs["performance_window"], outputs["performance_relative_window"]
    if corruption == "status":
        absolute.loc[0, "cagr_status"] = "NOT_ANNUALIZED_SHORT_WINDOW"
    elif corruption == "absolute":
        absolute.loc[0, "annualized_return_cagr"] += .01
    elif corruption == "relative":
        relative.loc[0, "annualized_cagr_spread"] += .01
    elif corruption == "policy":
        relative.loc[0, "inclusive_months"] = 11
    else:
        absolute.loc[0, "annualized_return_cagr"] = np.nan
    with pytest.raises((ValueError, AssertionError), match="CAGR"):
        validate_month_cagr(w, outputs, ("P", "B"))


def test_cagr_month_policy_preserves_genuine_unavailable_status():
    w, returns, pair = cagr_sample()
    outputs = compute_month_performance(returns, pair, w, ("P", "B"))
    row = outputs["performance_window"].index[0]
    outputs["performance_window"].loc[row, ["calculation_status", "cagr_status"]] = "UNAVAILABLE_ZERO_START_WEALTH"
    outputs["performance_window"].loc[row, ["period_return", "annualized_return_cagr"]] = np.nan
    validate_month_cagr(w, outputs, ("P", "B"))


def test_interval_validation_rejects_the_previously_accepted_short_cagr_status():
    w, returns, pair = cagr_sample()
    result = outputs()
    result["performance"] = compute_month_performance(returns, pair, w, ("P", "B"))
    result["performance"]["performance_relative_window"].loc[0, "cagr_status"] = "NOT_ANNUALIZED_SHORT_WINDOW"
    context = SimpleNamespace(settings=SimpleNamespace(portfolio_ids=("P", "B")), returns=returns)
    # Other domain tables are irrelevant: the monthly CAGR audit must reject
    # this soft status before continuing to those domain-specific checks.
    with pytest.raises(ValueError, match="Eligible complete-month CAGR"):
        validate_month_outputs(context, w, result)


def test_complete_interval_records_the_passed_cagr_check_with_other_validations():
    w, returns, pair = cagr_sample()
    result = outputs()
    result["performance"] = compute_month_performance(returns, pair, w, ("P", "B"))
    result["attribution"]["brinson_total_period"] = pd.DataFrame({"linking_method": ["carino", "menchero"]})
    for name in ("portfolio_window_summary", "active_window_summary"):
        result["exposures"][name] = pd.DataFrame({"observations": [w["observation_count"]]})
    fields = ("one_way_turnover", "total_purchases", "total_sales", "gross_two_way_turnover",
              "implemented_weight_sum", "max_abs_implementation_weight_error")
    pending = pd.DataFrame({"portfolio_id": ["P"], "selection_date": pd.to_datetime(["2022-12-31"]),
                           "effective_rebalance_date": [pd.NaT],
                           "event_status": ["PENDING_AFTER_CUTOFF"],
                           "turnover_status": ["NOT_ASSESSED_PENDING_SELECTION"],
                           **{field: [np.nan] for field in fields}})
    result["turnover"]["rebalance_turnover_events"] = pending
    source_events = pending.copy()
    source_events["event_status"] = "EFFECTIVE_REBALANCE"
    source_events["effective_rebalance_date"] = pd.Timestamp("2023-01-03")
    context = SimpleNamespace(settings=SimpleNamespace(portfolio_ids=("P", "B")), returns=returns,
                              shared={("turnover", "rebalance_turnover_events"): source_events})
    report = validate_month_outputs(context, w, result)
    assert all(row["status"] == "PASSED" for row in report)
    assert next(row for row in report if row["domain"] == "performance")["complete_month_cagr_check"] == "PASSED"


def test_inclusive_month_lengths_and_entire_universe():
    window = MonthWindow.from_names("January", 2020, "December", 2020)
    assert window.months == 12
    assert MonthWindow.from_names(" OCT ", 2016, "Sep", 2026).months == 120
    intervals = enumerate_month_windows(calendar(), "2016-09-30")
    assert len(intervals) == 5995
    assert not intervals.interval_id.duplicated().any()
    assert intervals.inclusive_months.between(12, 120).all()
    assert sum(intervals.inclusive_months.eq(12)) == 109


@pytest.mark.parametrize("arguments", [
    ("January", 2020, "November", 2020),
    ("December", 2020, "January", 2020),
    ("September", 2016, "September", 2017),
    ("January", 2026, "December", 2026),
    ("Jannuary", 2020, "December", 2020),
    (1, 2020, "December", 2020),
    ("January", True, "December", 2020),
    ("January", "2020", "December", 2020),
])
def test_invalid_selection_is_rejected(arguments):
    with pytest.raises(ValueError):
        MonthWindow.from_names(*arguments)


def test_prior_month_close_and_weekend_end_are_preserved():
    window = MonthWindow.from_names("January", 2022, "December", 2022)
    resolved = window.resolve(calendar(), "2016-09-30")
    assert resolved["opening_valuation_date"] == pd.Timestamp("2021-12-31")
    assert resolved["first_return_date"] == pd.Timestamp("2022-01-03")
    assert resolved["last_return_date"] == pd.Timestamp("2022-12-30")
    assert resolved["calendar_end"] == pd.Timestamp("2022-12-31")
    inception = MonthWindow.from_names("October", 2016, "September", 2017).resolve(calendar(), "2016-09-30")
    assert inception["opening_valuation_date"] == pd.Timestamp("2016-09-30")


def test_first_trading_day_return_is_included_in_the_full_month():
    dates = calendar()
    w = MonthWindow.from_names("January", 2022, "December", 2022).resolve(dates, "2016-09-30")
    values = np.zeros(len(dates))
    values[w["start_index"]] = 0.1  # Prior December close 100 -> first January close 110.
    values[w["stop_index"] - 1] = 1.0  # Final December close 220.
    source = pd.DataFrame(dict(portfolio_id="P", date=dates, return_total=values))
    result = compute_absolute_performance(source, performance_definitions(w, ("P",)))
    assert result.iloc[0].period_return == pytest.approx(1.2)
    assert result.iloc[0].observation_count == w["observation_count"]


def test_missing_month_and_missing_opening_close_are_rejected():
    dates = calendar()
    selected = MonthWindow.from_names("January", 2022, "December", 2022)
    with pytest.raises(ValueError, match="missing"):
        selected.resolve(dates[dates.to_period("M") != pd.Period("2022-06")], "2016-09-30")
    with pytest.raises(ValueError, match="preceding month"):
        selected.resolve(dates[dates >= "2022-01-01"], "2016-09-30")


def test_drawdown_includes_first_day_loss_from_the_opening_close():
    path = calculate_drawdown_path(np.array([-0.1, 1 / 9]), pd.to_datetime(["2022-01-03", "2022-01-04"]))
    assert path.drawdown.iloc[0] == pytest.approx(-0.1)
    assert path.drawdown.iloc[-1] == pytest.approx(0, abs=1e-12)


def test_validation_rejects_future_implementation_values_on_pending_events():
    context = SimpleNamespace(settings=SimpleNamespace(portfolio_ids=("P", "B")),
        returns=pd.DataFrame(dict(portfolio_id=["P", "B"],
            date=pd.to_datetime(["2020-12-31"] * 2), return_total=[0.1, 0.1])))
    w = dict(first_return_date=pd.Timestamp("2020-12-31"), last_return_date=pd.Timestamp("2020-12-31"),
        opening_valuation_date=pd.Timestamp("2020-12-30"), observation_count=1)
    w.update(calendar_start=pd.Timestamp("2020-12-01"), calendar_end=pd.Timestamp("2020-12-31"))
    context.shared = {("turnover", "rebalance_turnover_events"): pd.DataFrame({
        "portfolio_id": ["P"], "selection_date": pd.to_datetime(["2020-12-31"]),
        "effective_rebalance_date": pd.to_datetime(["2021-01-04"]),
        "event_status": ["EFFECTIVE_REBALANCE"], "turnover_status": ["PASSED"]})}
    result = outputs()
    result["performance"]["performance_window"] = pd.DataFrame(dict(portfolio_id=["P", "B"], period_return=[0.1, 0.1]))
    result["attribution"]["brinson_total_period"] = pd.DataFrame(dict(linking_method=["carino", "menchero"]))
    for name in ("portfolio_window_summary", "active_window_summary"):
        result["exposures"][name] = pd.DataFrame({"observations": [1]})
    fields = ("one_way_turnover", "total_purchases", "total_sales", "gross_two_way_turnover",
              "implemented_weight_sum", "max_abs_implementation_weight_error")
    pending = pd.DataFrame({"event_status": ["PENDING_AFTER_CUTOFF"], **{field: [np.nan] for field in fields}})
    pending["portfolio_id"] = "P"
    pending["selection_date"] = pd.to_datetime(["2020-12-31"])
    pending["effective_rebalance_date"] = pd.NaT
    pending["turnover_status"] = "NOT_ASSESSED_PENDING_SELECTION"
    pending["implemented_weight_sum"] = 1.0
    result["turnover"]["rebalance_turnover_events"] = pending
    with pytest.raises(ValueError, match="future implementation"):
        validate_month_outputs(context, w, result)
    pending["implemented_weight_sum"] = np.nan
    assert all(row["status"] == "PASSED" for row in validate_month_outputs(context, w, result))


def metadata():
    return dict(snapshot_id="test", config_json="{}", source_database="unused",
        source_run_id="source", source_fingerprint="source-code", manifest_sha256="manifest",
        code_fingerprint="current-code", input_validation={"status": "PASSED"})


def outputs(value=0.25):
    return {domain: {"summary": pd.DataFrame({"value": [value]})} for domain in (
        "performance", "attribution", "risk", "exposures", "factors", "turnover")}


def test_checkpoint_resume_completed_reader_and_immutability(tmp_path):
    path = tmp_path / "months.duckdb"
    a = MonthWindow.from_names("January", 2020, "December", 2020)
    b = MonthWindow.from_names("February", 2020, "January", 2021)
    windows = [w.resolve(calendar(), "2016-09-30") for w in (a, b)]
    writer = MonthResultWriter(path)
    try:
        writer.begin_build("test", metadata(), windows)
        writer.save_batch("test", [(windows[0], outputs(), [{"status": "PASSED"}], [])])
        assert writer.completed("test") == {a.interval_id}
        with pytest.raises(ValueError, match="Incomplete"):
            writer.finish("test")
        writer.fail("test", RuntimeError("interrupted"))
    finally:
        writer.close()
    assert list_month_builds(path).empty
    with pytest.raises(ValueError, match="completed"):
        read_month_result(path, "test", a, "performance", "summary")
    writer = MonthResultWriter(path)
    try:
        with pytest.raises(ValueError, match="changed code_fingerprint"):
            writer.begin_build("test", {**metadata(), "code_fingerprint": "different"}, windows, resume=True)
        writer.begin_build("test", metadata(), windows, resume=True)
        writer.save_batch("test", [(windows[1], outputs(0.5), [{"status": "PASSED"}], [])])
        with pytest.raises(ValueError, match="already completed"):
            writer.save_batch("test", [(windows[0], outputs(), [], [])])
        writer.finish("test")
        writer.fail("test", RuntimeError("must not invalidate a completed build"))
    finally:
        writer.close()
    assert len(list_month_builds(path)) == 1
    assert read_month_result(path, "test", a, "performance", "summary").iloc[0].value == 0.25
    assert read_month_result(path, "test", b, "risk", "summary").iloc[0].value == 0.5


def test_invalid_result_rolls_back_the_whole_checkpoint(tmp_path):
    path = tmp_path / "months.duckdb"
    window = MonthWindow.from_names("January", 2020, "December", 2020).resolve(calendar(), "2016-09-30")
    writer = MonthResultWriter(path)
    try:
        writer.begin_build("test", metadata(), [window])
        with pytest.raises(ValueError, match="Infinite"):
            writer.save_batch("test", [(window, outputs(float("inf")), [], [])])
        assert writer.completed("test") == set()
    finally:
        writer.close()
    with duckdb.connect(str(path), read_only=True) as db:
        assert db.execute("SELECT count(*) FROM control.month_outputs").fetchone()[0] == 0
        assert json.loads(db.execute("SELECT definition_json FROM control.month_intervals").fetchone()[0])["interval_id"] == window["interval_id"]


def test_scaled_ols_preserves_original_units_predictions_and_strict_validation():
    rng = np.random.default_rng(42)
    f = rng.normal(size=(400, 3)) * [0.02, 0.001, 0.004] + [0.001, -0.0001, 0.0002]
    y = 0.0003 + f @ [1.2, -0.4, 0.7] + rng.normal(scale=0.001, size=len(f))
    source = pd.DataFrame(f, columns=["a", "b", "c"])
    source["date"] = pd.bdate_range("2020-01-01", periods=len(f))
    source["excess_return"] = y
    fit, coeff, residuals = fit_period_ols(source, factor_columns=("a", "b", "c"), model_id="test")
    x = np.column_stack([np.ones(len(f)), f])
    np.testing.assert_allclose(coeff.coefficient, np.linalg.lstsq(x, y, rcond=None)[0], atol=1e-10, rtol=1e-9)
    np.testing.assert_allclose(residuals.fitted_excess_return + residuals.residual_excess_return, y, atol=1e-12)
    assert fit.iloc[0].max_normal_equation_error <= 1e-12 * max(1, np.linalg.norm(y, ord=1))
    percent = source.copy()
    percent[["a", "b", "c"]] *= 100
    _, percent_coeff, percent_residuals = fit_period_ols(percent, factor_columns=("a", "b", "c"), model_id="test")
    np.testing.assert_allclose(percent_residuals.fitted_excess_return, residuals.fitted_excess_return, atol=1e-12)
    np.testing.assert_allclose(percent_coeff.coefficient.iloc[1:] * 100, coeff.coefficient.iloc[1:], atol=1e-10)


def test_scaling_does_not_hide_original_design_condition_failure():
    source = pd.DataFrame(dict(date=pd.bdate_range("2020-01-01", periods=100),
        tiny=np.linspace(-1e-12, 1e-12, 100), excess_return=np.linspace(-0.01, 0.01, 100)))
    with pytest.raises(ValueError, match="condition-number"):
        fit_period_ols(source, factor_columns=("tiny",), model_id="test")


def test_source_database_uses_recorded_path_when_present(tmp_path, monkeypatch):
    from portfolio_analytics.database.month_results import SOURCE_DATABASE_ENV, resolve_source_database
    monkeypatch.delenv(SOURCE_DATABASE_ENV, raising=False)
    source = tmp_path / "elsewhere" / "analytics.duckdb"
    source.parent.mkdir()
    source.touch()
    assert resolve_source_database(str(source), tmp_path / "month_windows.duckdb") == source


def test_source_database_falls_back_next_to_month_database(tmp_path, monkeypatch):
    from portfolio_analytics.database.month_results import SOURCE_DATABASE_ENV, resolve_source_database
    monkeypatch.delenv(SOURCE_DATABASE_ENV, raising=False)
    (tmp_path / "analytics.duckdb").touch()
    recorded = r"Z:\Missing\Machine\local\analytics_outputs\analytics.duckdb"
    resolved = resolve_source_database(recorded, tmp_path / "month_windows.duckdb")
    assert resolved == (tmp_path / "analytics.duckdb").resolve()


def test_source_database_override_and_missing_file(tmp_path, monkeypatch):
    from portfolio_analytics.database.month_results import SOURCE_DATABASE_ENV, resolve_source_database
    override = tmp_path / "shared.duckdb"
    override.touch()
    monkeypatch.setenv(SOURCE_DATABASE_ENV, str(override))
    assert resolve_source_database(r"Z:\gone\analytics.duckdb", tmp_path / "m.duckdb") == override
    monkeypatch.setenv(SOURCE_DATABASE_ENV, str(tmp_path / "absent.duckdb"))
    with pytest.raises(FileNotFoundError):
        resolve_source_database(r"Z:\gone\analytics.duckdb", tmp_path / "m.duckdb")
    monkeypatch.delenv(SOURCE_DATABASE_ENV)
    recorded = r"Z:\Private\Path\analytics.duckdb"
    with pytest.raises(FileNotFoundError) as error:
        resolve_source_database(recorded, tmp_path / "m.duckdb")
    assert "Private" not in str(error.value)
