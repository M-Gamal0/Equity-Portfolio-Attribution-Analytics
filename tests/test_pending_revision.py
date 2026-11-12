"""Calendar-end membership and atomic, inherited month-result revisions."""
import json

import duckdb
import numpy as np
import pandas as pd
import pytest

from portfolio_analytics.database.month_results import MonthResultWriter
from portfolio_analytics.database.pending_revision import revise_pending_build, TABLES
from portfolio_analytics.pipeline.month_turnover import select_turnover_events, validate_pending_events, PENDING_MEASUREMENTS


def example():
    w = dict(interval_id="2016-10_2017-09", calendar_start="2016-10-01", calendar_end="2017-09-30",
             first_return_date="2016-10-03", last_return_date="2017-09-29")
    frame = pd.DataFrame({"portfolio_id": ["P", "P"], "selection_date": pd.to_datetime(["2017-06-30", "2017-09-30"]),
                          "effective_rebalance_date": pd.to_datetime(["2017-07-03", "2017-10-02"]),
                          "event_status": ["EFFECTIVE_REBALANCE"] * 2, "turnover_status": ["PASSED"] * 2,
                          **{field: [.1, .2] for field in PENDING_MEASUREMENTS}})
    return w, frame


def test_weekend_selection_is_pending_with_no_future_measurements():
    w, source = example()
    selected = select_turnover_events(source, w)
    assert len(selected) == 2
    pending = selected.iloc[1]
    assert pending.event_status == "PENDING_AFTER_CUTOFF" and pd.isna(pending.effective_rebalance_date)
    assert pending[PENDING_MEASUREMENTS].isna().all()
    assert source.effective_rebalance_date.iloc[1] == pd.Timestamp("2017-10-02")
    assert selected.one_way_turnover.iloc[0] == .1
    assert validate_pending_events(source, selected, w)["saved_pending_records"] == 1


def test_calendar_pending_audit_rejects_missing_or_leaked_future_events():
    w, source = example()
    selected = select_turnover_events(source, w)
    with pytest.raises(ValueError, match="completeness"):
        validate_pending_events(source, selected.iloc[:1], w)
    selected.loc[selected.event_status.eq("PENDING_AFTER_CUTOFF"), "one_way_turnover"] = 0.
    with pytest.raises(ValueError, match="future implementation"):
        validate_pending_events(source, selected, w)


def parent_database(tmp_path):
    w, source = example()
    source_path = tmp_path / "source.duckdb"
    with duckdb.connect(str(source_path)) as db:
        db.execute("CREATE SCHEMA control")
        db.execute("CREATE SCHEMA analytics")
        db.execute("CREATE TABLE control.analytics_runs AS SELECT 'source' run_id,'COMPLETED' status,'snapshot' snapshot_id,'sourcefp' source_fingerprint,'manifest' snapshot_manifest_sha256")
        db.register("source_frame", source)
        db.execute("CREATE TABLE analytics.pinned_events AS SELECT * FROM source_frame")
    measured = source.iloc[:1].copy()
    cost = measured[["portfolio_id", "selection_date", "effective_rebalance_date", "turnover_status", "one_way_turnover"]].copy()
    cost["cost_sensitivity_status"] = "UNIT_RATE_SENSITIVITY_ONLY"
    cost["nav_bp_per_1bp_traded"] = .2
    concentration = measured[["portfolio_id", "selection_date", "effective_rebalance_date", "turnover_status", "one_way_turnover"]].copy()
    concentration["concentration_status"] = "PASSED"
    concentration["security_hhi"] = .4
    outputs = {"turnover": dict(zip(TABLES, [measured, cost, concentration])), "performance": {"unchanged": pd.DataFrame({"return_value": [.25]})}}
    path = tmp_path / "months.duckdb"
    writer = MonthResultWriter(path)
    try:
        writer.begin_build("parent", dict(snapshot_id="snapshot", config_json="{}", source_database=str(source_path), source_run_id="source",
                           source_fingerprint="sourcefp", manifest_sha256="manifest", code_fingerprint="original", input_validation={}), [w])
        writer.register_shared("parent", pd.DataFrame({"domain": ["turnover"], "output_name": [TABLES[0]], "table_name": ["pinned_events"], "schema_json": ["{}"]}))
        writer.save_batch("parent", [(w, outputs, [{"domain": "turnover", "status": "PASSED", "rows": 3, "statuses": {}}], {})])
        writer.finish("parent")
    finally:
        writer.close()
    return path


def test_revision_preserves_parent_and_all_inherited_values_and_is_idempotent(tmp_path):
    path = parent_database(tmp_path)
    report = revise_pending_build(path, "parent")
    assert report["rows_added"] == 3 and report["parent_preserved"]
    assert report["pending_completeness"] == "PASSED"
    assert revise_pending_build(path, "parent") == report
    with duckdb.connect(str(path), read_only=True) as db:
        assert db.execute("SELECT count(*) FROM control.month_builds").fetchone()[0] == 2
        assert db.execute("SELECT rows_saved FROM control.month_intervals WHERE build_id='parent'").fetchone()[0] == 4
        assert db.execute("SELECT rows_saved FROM control.month_intervals WHERE build_id=?", [report["build_id"]]).fetchone()[0] == 7
        for name in TABLES:
            rows = db.execute("SELECT table_name FROM control.month_outputs WHERE build_id=? AND output_name=? AND scope='PERIOD'", [report["build_id"], name]).fetchone()
            data = db.execute('SELECT * FROM month_results."' + rows[0] + '" ORDER BY result_row').df()
            assert len(data) == 2 and np.isnan(data.one_way_turnover.iloc[1])
        tables = db.execute("SELECT table_name FROM control.month_outputs WHERE output_name='unchanged' ORDER BY build_id").fetchall()
        assert tables[0] == tables[1]
        validations = json.loads(db.execute("SELECT validation_json FROM control.month_intervals WHERE build_id=?", [report["build_id"]]).fetchone()[0])
        assert validations[-1]["domain"] == "turnover_pending_calendar_boundary"
        assert validations[0]["inherited_from_build_id"] == "parent"


def test_failed_revision_rolls_back_every_new_table_and_manifest(tmp_path, monkeypatch):
    path = parent_database(tmp_path)
    import portfolio_analytics.database.pending_revision as module
    original = module._unavailable_rows

    def duplicate(*args):
        rows = original(*args)
        return pd.concat([rows, rows], ignore_index=True)

    monkeypatch.setattr(module, "_unavailable_rows", duplicate)
    with pytest.raises(ValueError, match="row delta"):
        revise_pending_build(path, "parent")
    with duckdb.connect(str(path), read_only=True) as db:
        assert db.execute("SELECT count(*) FROM control.month_builds").fetchone()[0] == 1
        assert db.execute("SELECT count(*) FROM information_schema.tables WHERE table_schema='month_results'").fetchone()[0] == 4
