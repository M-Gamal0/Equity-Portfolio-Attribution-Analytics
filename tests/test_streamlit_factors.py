"""Saved factor presentation units, reconciliation and model-coverage safeguards."""
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit")


@pytest.fixture
def ui(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "app"))
    import factor_details
    return factor_details


def test_daily_alpha_and_interval_convert_once_without_annualizing(ui):
    source = pd.DataFrame({"term": ["intercept", "mkt_rf"], "coefficient": [.00002, 1.],
                           "hac_ci_lower": [-.0001, .8], "hac_ci_upper": [.0002, 1.2], "hac_p_value": [.8, .01], "classical_ci_lower": [-.0002, .7]})
    converted = ui.alpha_evidence(source)
    assert len(converted) == 1
    assert converted.coefficient_daily_bp.iloc[0] == pytest.approx(.2)
    assert converted.hac_ci_lower_daily_bp.iloc[0] == -1
    assert converted.hac_ci_upper_daily_bp.iloc[0] == 2
    assert converted.hac_p_value.iloc[0] == .8
    assert "classical_ci_lower" not in converted  # Original-unit inference stays in its separate complete table.
    assert source.coefficient.iloc[0] == .00002


def linked_row():
    return pd.Series({"portfolio_id": "Active portfolio", "link_status": "LINKED_PARTIAL_MODEL_COVERAGE",
                      "realized_compounded_total_return": .2, "linked_attributed_total_return": .2, "period_reconciliation_error": 0.,
                      "linked_risk_free_contribution": .01, "linked_estimated_intercept_contribution": .03,
                      **{"linked_factor_contribution_" + f: v for f, v in zip(["mkt_rf", "smb", "hml", "rmw", "cma", "mom"], [.1, -.02, .03, .01, .02, .01])},
                      "linked_modeled_residual_contribution": -.01, "linked_unmodeled_excess_contribution": .02})


def test_linked_bridge_keeps_every_component_partial_coverage_and_target(ui):
    row = linked_row()
    evidence = ui.linked_evidence(row, ["mkt_rf", "smb", "hml", "rmw", "cma", "mom"])
    bridge = ui.bridge_rows(evidence)
    assert len(evidence) == 10 and len(bridge) == 11
    assert evidence.linked_contribution_pp.sum() == pytest.approx(20)
    assert bridge.end_pp.iloc[-1] == pytest.approx(20)
    assert bridge.contribution_pp.min() == -2
    assert "Unmodeled excess" in evidence.component.tolist()
    assert row.linked_estimated_intercept_contribution == .03
    with pytest.raises(ValueError, match="reconcile"):
        ui.bridge_rows(evidence.assign(realized_total_return_pp=21.))


def test_linked_unavailable_or_missing_components_cannot_appear_complete(ui):
    evidence = ui.linked_evidence(linked_row(), ["mkt_rf", "smb", "hml", "rmw", "cma", "mom"])
    with pytest.raises(ValueError, match="saved status"):
        ui.bridge_rows(evidence.assign(link_status="UNAVAILABLE_MISSING_FACTOR_DATES"))
    with pytest.raises(ValueError, match="complete"):
        ui.bridge_rows(evidence.assign(linked_contribution_pp=float("nan")))


def test_direct_plot_keeps_zero_reference_and_alpha_separate(ui, monkeypatch):
    specs = []
    monkeypatch.setattr(ui, "render_chart", lambda chart, **kwargs: specs.append(chart.to_dict()))
    coefficients = pd.DataFrame({"term": ["intercept", "mkt_rf", "smb"], "coefficient": [.00002, 1., -.2]})
    inference = pd.DataFrame({"term": ["intercept", "mkt_rf", "smb"], "hac_ci_lower": [-.1, .9, -.3], "hac_ci_upper": [.1, 1.1, -.1], "hac_p_value": [.8, .01, .02]})
    ui.coefficient_chart(coefficients, inference, ["mkt_rf", "smb"], True)
    datasets = [row for rows in specs[0]["datasets"].values() for row in rows]
    assert any(row.get("zero") == 0 for row in datasets)
    assert {row["term"] for row in datasets if "term" in row} == {"mkt_rf", "smb"}
    specs.clear()
    ui.coefficient_chart(coefficients, inference, ["mkt_rf", "smb"], False)
    assert not specs


def test_partial_exposures_are_not_renormalized_and_missing_models_keep_gaps(ui, monkeypatch):
    specs = []
    monkeypatch.setattr(ui, "render_chart", lambda chart, **kwargs: specs.append(chart.to_dict()))
    source = pd.DataFrame({"portfolio_id": ["Active portfolio"] * 3, "date": pd.date_range("2020-01-02", periods=3),
                           "exposure_mkt_rf": [.8, .6, 9.], "modeled_gross_weight_coverage": [1., .75, 0.],
                           "factor_available_on_return_date": [True, True, False],
                           "exposure_coverage_status": ["FULLY_MODELED", "PARTIALLY_MODELED", "NO_MODELS"]})
    coverage = ui.coverage_evidence(source).iloc[0]
    assert (coverage.fully_modeled_days, coverage.partially_modeled_days, coverage.other_coverage_status_days) == (1, 1, 1)
    assert coverage.factor_available_days == 2
    ui.exposure_chart(source, "exposure_mkt_rf", "Market exposure")
    plotted = next(iter(specs[0]["datasets"].values()))
    assert plotted[1]["exposure_mkt_rf"] == .6
    assert plotted[2]["exposure_mkt_rf"] is None
    assert source.exposure_mkt_rf.iloc[2] == 9.
