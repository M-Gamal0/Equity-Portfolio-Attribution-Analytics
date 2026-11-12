"""Atomic derived month build correcting pending metadata, never parent results."""
import hashlib
import json
from pathlib import Path

import duckdb
import pandas as pd

from .month_results import result_table, utc_now
from .results import identifier, qualified_result
from portfolio_analytics.pipeline.month_turnover import select_turnover_events, PENDING_MEASUREMENTS
from portfolio_analytics.pipeline.runner import code_provenance

TABLES = ("rebalance_turnover_events", "rebalance_event_cost_sensitivity", "rebalance_event_concentration")
KEYS = ["interval_id", "portfolio_id", "selection_date"]


def _source(parent, catalog):
    record = catalog.loc[(catalog.output_name == TABLES[0]) & catalog.scope.eq("SOURCE_REFERENCE")]
    if len(record) != 1:
        raise ValueError("A unique pinned source-event reference is required.")
    with duckdb.connect(parent.source_database, read_only=True) as db:
        run = db.execute("SELECT status,snapshot_id,source_fingerprint,snapshot_manifest_sha256 FROM control.analytics_runs WHERE run_id=?",
                         [parent.source_run_id]).fetchone()
        if run != ("COMPLETED", parent.snapshot_id, parent.source_fingerprint, parent.manifest_sha256):
            raise ValueError("The source run no longer matches the parent snapshot/provenance.")
        return db.execute("SELECT * FROM " + qualified_result(db, record.iloc[0].table_name)).df()


def pending_plan(intervals, source, existing):
    """All-window expected pending keys; no regression or financial calculation."""
    meta = intervals[["interval_id", "definition_json"]].copy()
    definitions = meta.definition_json.map(json.loads)
    for column in ["calendar_start", "calendar_end", "first_return_date", "last_return_date"]:
        meta[column] = pd.to_datetime(definitions.map(lambda row: row[column]))
    candidates = meta.drop(columns="definition_json").merge(source[["portfolio_id", "selection_date", "effective_rebalance_date", "event_status"]], how="cross")
    future = candidates.effective_rebalance_date.gt(candidates.last_return_date)
    already_pending = candidates.effective_rebalance_date.isna() & candidates.event_status.eq("PENDING_AFTER_CUTOFF")
    expected = candidates.loc[candidates.selection_date.ge(candidates.calendar_start) & candidates.selection_date.le(candidates.calendar_end) & (future | already_pending), KEYS].copy()
    actual = existing.loc[existing.event_status.eq("PENDING_AFTER_CUTOFF"), KEYS]
    if expected.duplicated(KEYS).any() or actual.duplicated(KEYS).any():
        raise ValueError("Pending keys are not unique.")
    comparison = expected.merge(actual, how="outer", on=KEYS, indicator=True)
    if comparison._merge.eq("right_only").any():
        raise ValueError("Unexpected saved pending records; automatic append is unsafe.")
    missing = comparison.loc[comparison._merge.eq("left_only"), KEYS]
    additions = []
    next_rows = existing.groupby("interval_id").result_row.max().add(1).to_dict()
    source_index = source.set_index(["portfolio_id", "selection_date"])
    if not source_index.index.is_unique:
        raise ValueError("Pinned source event keys are not unique.")
    window_index = dict(zip(intervals.interval_id, definitions))
    for key in missing.itertuples(index=False):
        row = source_index.loc[[(key.portfolio_id, key.selection_date)]].reset_index()
        row = select_turnover_events(row, window_index[key.interval_id])
        if len(row) != 1 or row.iloc[0].event_status != "PENDING_AFTER_CUTOFF":
            raise ValueError("A missing decision cannot be safely converted to pending.")
        row.insert(0, "interval_id", key.interval_id)
        row.insert(1, "result_row", next_rows[key.interval_id])
        next_rows[key.interval_id] += 1
        additions.append(row)
    added = pd.concat(additions, ignore_index=True) if additions else existing.iloc[:0].copy()
    return expected, missing, added


def _unavailable_rows(pending, columns, name, maxima):
    rows = pd.DataFrame(index=pending.index, columns=columns)
    for column in ["interval_id", "portfolio_id", "selection_date", "turnover_status"]:
        rows[column] = pending[column]
    rows["effective_rebalance_date"] = pd.NaT
    rows["result_row"] = [maxima[iid] + ordinal for iid, ordinal in zip(pending.interval_id, pending.groupby("interval_id").cumcount() + 1)]
    if name == TABLES[1]:
        rows["cost_sensitivity_status"] = "NOT_ASSESSED"
    else:
        rows["concentration_status"] = "NOT_ASSESSED"
    return rows[columns]


def revise_pending_build(database, parent_build_id, *, correction_id="calendar_pending_v1", dry_run=False):
    """Share inherited tables; new event tables and provenance commit atomically."""
    path = Path(database).resolve()
    _, fingerprint = code_provenance()
    with duckdb.connect(str(path), read_only=True) as db:
        parents = db.execute("SELECT * FROM control.month_builds WHERE build_id=? AND status='COMPLETED'", [parent_build_id]).df()
        if len(parents) != 1:
            raise ValueError("An explicit completed parent build is required.")
        parent = parents.iloc[0]
        catalog = db.execute("SELECT * FROM control.month_outputs WHERE build_id=?", [parent_build_id]).df()
        intervals = db.execute("SELECT * FROM control.month_intervals WHERE build_id=? ORDER BY interval_id", [parent_build_id]).df()
        if len(intervals) != parent.expected_intervals or not intervals.status.eq("COMPLETED").all():
            raise ValueError("The parent interval universe is incomplete.")
        source = _source(parent, catalog)
        old_tables, frames = {}, {}
        for name in TABLES:
            records = catalog.loc[catalog.output_name.eq(name) & catalog.scope.eq("PERIOD")]
            if len(records) != 1:
                raise ValueError("Missing unique event output: " + name)
            old_tables[name] = records.iloc[0].table_name
            frames[name] = db.execute("SELECT * FROM month_results." + identifier(old_tables[name])).df()
        expected, missing, pending = pending_plan(intervals, source, frames[TABLES[0]])
    source_hash = hashlib.sha256(pd.util.hash_pandas_object(source, index=False).values.tobytes()).hexdigest()
    identity = dict(parent_build_id=parent_build_id, correction_id=correction_id, correction_code_fingerprint=fingerprint, source_event_hash=source_hash)
    child = "months_pending_" + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    report = dict(**identity, build_id=child, intervals=len(intervals), expected_pending_records=len(expected),
                  missing_parent_pending_records=len(missing), affected_intervals=int(missing.interval_id.nunique()),
                  event_tables_revised=list(TABLES), rows_added=len(pending) * len(TABLES),
                  inherited_analytics_code_fingerprint=parent.code_fingerprint, numerical_recomputation=False)
    if dry_run:
        return report
    if missing.empty:
        raise ValueError("No missing pending decisions in this parent; no revision is necessary.")
    additions = {TABLES[0]: pending[frames[TABLES[0]].columns]}
    for name in TABLES[1:]:
        frame = frames[name]
        additions[name] = _unavailable_rows(pending, frame.columns.tolist(), name, frame.groupby("interval_id").result_row.max().to_dict())
        if frame.merge(missing, on=KEYS).shape[0]:
            raise ValueError("Dependent event keys already exist; append would duplicate them.")
    deltas = pending.groupby("interval_id").size().mul(len(TABLES)).to_dict()
    pending_counts = expected.groupby("interval_id").size().to_dict()
    status_deltas = {}
    for name, frame in additions.items():
        for iid, rows in frame.groupby("interval_id"):
            updates = status_deltas.setdefault(iid, {})
            for column in rows:
                if column.endswith("status"):
                    updates[name + "." + column] = rows[column].value_counts().to_dict()
    revised_intervals = intervals.copy()
    revised_intervals["build_id"] = child
    revised_intervals["rows_saved"] += revised_intervals.interval_id.map(deltas).fillna(0).astype(int)
    validations = []
    for row in intervals.itertuples(index=False):
        checks = json.loads(row.validation_json)
        for check in checks:
            check["inherited_from_build_id"] = parent_build_id
            if check.get("domain") == "turnover" and row.interval_id in deltas:
                check["rows"] += deltas[row.interval_id]
                for key, counts in status_deltas[row.interval_id].items():
                    current = check.setdefault("statuses", {}).setdefault(key, {})
                    for status, count in counts.items():
                        current[status] = current.get(status, 0) + int(count)
        checks.append(dict(domain="turnover_pending_calendar_boundary", status="PASSED",
                           expected_pending_records=int(pending_counts.get(row.interval_id, 0)),
                           saved_pending_records=int(pending_counts.get(row.interval_id, 0))))
        validations.append(json.dumps(checks, sort_keys=True))
    revised_intervals["validation_json"] = validations
    now = utc_now()
    report.update(created_at=now, validation_basis="Inherited numerical/boundary validations plus fresh pending completeness per interval")
    with duckdb.connect(str(path)) as db:
        db.execute("BEGIN TRANSACTION")
        try:
            db.execute("CREATE TABLE IF NOT EXISTS control.month_build_derivations (build_id VARCHAR PRIMARY KEY,parent_build_id VARCHAR,correction_id VARCHAR,manifest_json VARCHAR, UNIQUE(parent_build_id,correction_id))")
            prior = db.execute("SELECT build_id,manifest_json FROM control.month_build_derivations WHERE parent_build_id=? AND correction_id=?", [parent_build_id, correction_id]).fetchone()
            if prior:
                if prior[0] != child:
                    raise ValueError("This correction identity already exists with different provenance.")
                db.execute("ROLLBACK")
                return json.loads(prior[1])
            revised_catalog = catalog.copy()
            revised_catalog["build_id"] = child
            table_audits = {}
            for name in TABLES:
                old = "month_results." + identifier(old_tables[name])
                new_name = result_table(child, "turnover", name)
                new = "month_results." + identifier(new_name)
                db.execute(f"CREATE TABLE {new} AS SELECT * FROM {old}")
                db.register("pending_stage", additions[name])
                db.execute(f"INSERT INTO {new} BY NAME SELECT * FROM pending_stage")
                # Exact multiset proof: inherited rows are unchanged, the only extra rows are the staged pending records.
                columns = ",".join(identifier(c) for c in frames[name].columns)
                staged = f"SELECT {columns} FROM {new} EXCEPT ALL SELECT {columns} FROM {old}"
                mismatch = db.execute(f"SELECT count(*) FROM (({staged} EXCEPT ALL SELECT {columns} FROM pending_stage) UNION ALL (SELECT {columns} FROM pending_stage EXCEPT ALL ({staged})))").fetchone()[0]
                removed = db.execute(f"SELECT count(*) FROM (SELECT * FROM {old} EXCEPT ALL SELECT * FROM {new})").fetchone()[0]
                if mismatch or removed:
                    raise ValueError("Exact inherited/pending readback failed: " + name)
                old_count = len(frames[name])
                new_count = db.execute(f"SELECT count(*) FROM {new}").fetchone()[0]
                if new_count != old_count + len(pending):
                    raise ValueError("Unexpected event-table row delta.")
                table_audits[name] = dict(parent_rows=old_count, revised_rows=new_count, added_pending_rows=len(pending), exact_inheritance="PASSED")
                revised_catalog.loc[revised_catalog.output_name.eq(name) & revised_catalog.scope.eq("PERIOD"), "table_name"] = new_name
                db.unregister("pending_stage")
            events_table = "month_results." + identifier(result_table(child, "turnover", TABLES[0]))
            db.register("expected_pending_keys", expected)
            actual = f"SELECT interval_id,portfolio_id,selection_date FROM {events_table} WHERE event_status='PENDING_AFTER_CUTOFF'"
            if db.execute(f"SELECT count(*) FROM (({actual} EXCEPT ALL SELECT * FROM expected_pending_keys) UNION ALL (SELECT * FROM expected_pending_keys EXCEPT ALL {actual}))").fetchone()[0]:
                raise ValueError("All-window pending completeness failed.")
            forbidden = " OR ".join(identifier(c) + " IS NOT NULL" for c in PENDING_MEASUREMENTS)
            if db.execute(f"SELECT count(*) FROM {events_table} WHERE event_status='PENDING_AFTER_CUTOFF' AND (effective_rebalance_date IS NOT NULL OR {forbidden})").fetchone()[0]:
                raise ValueError("Pending measurements leaked into revised results.")
            report["table_audits"] = table_audits
            total_rows = sum(db.execute("SELECT count(*) FROM month_results." + identifier(row.table_name)).fetchone()[0] for row in revised_catalog.loc[revised_catalog.scope.eq("PERIOD")].itertuples())
            if total_rows != int(revised_intervals.rows_saved.sum()):
                raise ValueError("Revised interval row accounting does not match the output catalog.")
            report.update(period_rows=total_rows, pending_completeness="PASSED", unchanged_catalog_outputs=len(catalog) - len(TABLES), parent_preserved=True)
            new_parent = parents.copy()
            new_parent.loc[0, ["build_id", "created_at", "completed_at", "code_fingerprint", "message"]] = [child, now, now, fingerprint, "Derived pending calendar-boundary repair; financial results inherited from " + parent_build_id]
            for temporary, frame, table in [("revision_build", new_parent, "control.month_builds"), ("revision_catalog", revised_catalog, "control.month_outputs"), ("revision_intervals", revised_intervals, "control.month_intervals")]:
                db.register(temporary, frame)
                db.execute(f"INSERT INTO {table} BY NAME SELECT * FROM {temporary}")
            db.execute("INSERT INTO control.month_build_derivations VALUES (?,?,?,?)", [child, parent_build_id, correction_id, json.dumps(report, sort_keys=True)])
            db.execute("COMMIT")
        except Exception:
            db.execute("ROLLBACK")
            raise
    with duckdb.connect(str(path), read_only=True) as db:
        if db.execute("SELECT status FROM control.month_builds WHERE build_id=?", [child]).fetchone() != ("COMPLETED",):
            raise ValueError("Independent completed revision readback failed.")
    return report
