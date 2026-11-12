"""A fixed public UI must never select private data or compute on rerun."""
import json
import os
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from portfolio_analytics.config import AnalysisSettings
from portfolio_analytics.database.month_results import MonthResultWriter
from portfolio_analytics.demo import view
from portfolio_analytics.demo.build import ACTIVE_ID, BENCHMARK_ID

APP = Path(__file__).resolve().parents[1] / "app/public_demo_app.py"


def test_missing_public_view_shows_setup_without_private_fallback_or_calculations(monkeypatch, tmp_path):
    monkeypatch.setenv("PORTFOLIO_DEMO_WORKDIR", str(tmp_path))
    def forbidden(*a, **kw):
        raise AssertionError("UI attempted preparation or a financial run")
    monkeypatch.setattr(view, "prepare_demo_view", forbidden)
    monkeypatch.setattr(view, "run_analysis", forbidden)
    app = AppTest.from_file(str(APP)).run(timeout=30)
    assert not app.exception and app.info
    assert "prepare-app" in app.code[0].value
    assert not app.date_input and not app.selectbox and not app.dataframe
    assert not (tmp_path / "streamlit").exists()


def test_public_view_rejects_wrong_profile_before_opening_any_database():
    with pytest.raises(ValueError, match="fixed public"):
        view.verify_demo_view({"profile": "research", "interval_id": "2016-10_2026-09"})


@pytest.mark.parametrize("status,expected", [("RUNNING", 1), ("COMPLETED", 2)])
def test_public_view_requires_one_completed_interval_without_mutating_database(tmp_path, status, expected):
    database = tmp_path / "view.duckdb"
    source = str(tmp_path / "source.duckdb")
    settings = AnalysisSettings(ACTIVE_ID, BENCHMARK_ID, "2026-09-30", "2021-09-30")
    metadata = {"snapshot_id": "0" * 16, "config_json": settings.to_json(), "source_database": source,
                "source_run_id": "fixture", "source_fingerprint": "fixture", "manifest_sha256": "fixture",
                "code_fingerprint": "fixture", "input_validation": {}}
    writer = MonthResultWriter(database)
    writer.begin_build("fixture", metadata, [{"interval_id": view.INTERVAL_ID, "inclusive_months": 60}])
    writer.db.execute("UPDATE control.month_builds SET status=?,expected_intervals=?", [status, expected])
    writer.close()
    before = (database.stat().st_size, database.stat().st_mtime_ns)
    with pytest.raises(ValueError, match="incomplete"):
        view.verify_demo_view({"profile": "public_demo_fixed_five_years_v1", "interval_id": view.INTERVAL_ID,
                              "database": str(database), "build_id": "fixture", "snapshot_id": "0" * 16,
                              "source_database": source})
    assert (database.stat().st_size, database.stat().st_mtime_ns) == before


@pytest.mark.public_demo
def test_prepared_public_app_all_sections_are_read_only_and_fixed(monkeypatch):
    workspace = os.environ.get("PORTFOLIO_DEMO_TEST_WORKDIR")
    if not workspace:
        pytest.skip("Set PORTFOLIO_DEMO_TEST_WORKDIR to a prepared public workspace")
    monkeypatch.setenv("PORTFOLIO_DEMO_WORKDIR", workspace)
    def forbidden(*a, **kw):
        raise AssertionError("Rerun attempted preparation or analytics")
    monkeypatch.setattr(view, "prepare_demo_view", forbidden)
    monkeypatch.setattr(view, "run_analysis", forbidden)
    pointer = json.loads((Path(workspace) / "streamlit/latest.json").read_text(encoding="utf-8"))
    files = [Path(pointer[k]) for k in ("database", "source_database")]
    before = [(p.stat().st_size, p.stat().st_mtime_ns) for p in files]
    app = AppTest.from_file(str(APP)).run(timeout=90)
    expected = {"Overview": 3, "Performance": 3, "Attribution": 3, "Risk": 2,
                "Holdings & exposures": 3, "Factors": 3, "Turnover & rebalancing": 2, "About": 0}
    details = 0
    for section, charts in expected.items():
        app.radio(key="public_demo_section").set_value(section).run(timeout=90)
        assert not app.exception and not app.error, section
        assert len(app.get("bidi_component")) == charts
        assert not app.date_input
        assert not any(s.label in ("Beginning month", "Beginning year", "Ending month", "Ending year") for s in app.selectbox)
        assert app.session_state["research_context"].iid == view.INTERVAL_ID
        assert app.session_state["research_context"].settings["portfolio_id"] == "demo_active20"
        for table in app.dataframe:
            assert not {"security_id", "other_security_id"}.intersection(table.value.columns)
        for selector in app.selectbox:
            if selector.label == "Result table":
                details += len(selector.options)
    assert details == 133
    assert [(p.stat().st_size, p.stat().st_mtime_ns) for p in files] == before
