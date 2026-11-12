"""Versioned month-window tables, checkpoints and read-only interval selection."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import duckdb
import pandas as pd

from .results import identifier, storage_frame


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def result_table(build_id, domain, name, scope="PERIOD"):
    token = hashlib.sha256(build_id.encode()).hexdigest()[:16]
    return f"{scope.lower()}_{domain}_{name}_{token}"


class MonthResultWriter:
    """One writer with atomic multi-interval checkpoints and exact readbacks."""

    def __init__(self, database):
        path = Path(database).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.db = duckdb.connect(str(path))
        self.db.execute("CREATE SCHEMA IF NOT EXISTS control")
        self.db.execute("CREATE SCHEMA IF NOT EXISTS month_results")
        self.db.execute("""CREATE TABLE IF NOT EXISTS control.month_builds (
            build_id VARCHAR PRIMARY KEY, created_at VARCHAR, completed_at VARCHAR,
            status VARCHAR, snapshot_id VARCHAR, config_json VARCHAR, source_database VARCHAR,
            source_run_id VARCHAR, source_fingerprint VARCHAR, manifest_sha256 VARCHAR,
            code_fingerprint VARCHAR, expected_intervals BIGINT, input_validation_json VARCHAR,
            message VARCHAR, publication_approved BOOLEAN DEFAULT FALSE)""")
        self.db.execute("""CREATE TABLE IF NOT EXISTS control.month_outputs (
            build_id VARCHAR, domain VARCHAR, output_name VARCHAR, scope VARCHAR,
            table_name VARCHAR, schema_json VARCHAR,
            PRIMARY KEY (build_id,domain,output_name,scope))""")
        self.db.execute("""CREATE TABLE IF NOT EXISTS control.month_intervals (
            build_id VARCHAR, interval_id VARCHAR, definition_json VARCHAR, status VARCHAR,
            rows_saved BIGINT, completed_at VARCHAR, validation_json VARCHAR, timings_json VARCHAR,
            PRIMARY KEY (build_id,interval_id))""")

    def close(self):
        self.db.close()

    def begin_build(self, build_id, metadata, windows, *, resume=False):
        existing = self.db.execute("SELECT * FROM control.month_builds WHERE build_id=?", [build_id]).df()
        if len(existing):
            if not resume:
                raise ValueError("Build already exists; use explicit resume to retain its checkpoints.")
            row = existing.iloc[0]
            for key in ("snapshot_id", "config_json", "source_database", "source_run_id",
                        "source_fingerprint", "manifest_sha256", "code_fingerprint"):
                if row[key] != metadata[key]:
                    raise ValueError(f"Cannot resume with changed {key}.")
            actual = self.db.execute("SELECT interval_id,definition_json FROM control.month_intervals WHERE build_id=?",
                                     [build_id]).fetchall()
            expected = {w["interval_id"]: json.dumps(w, sort_keys=True, default=str) for w in windows}
            if dict(actual) != expected:
                raise ValueError("Cannot resume a different interval universe.")
            if row.status != "COMPLETED":
                self.db.execute("UPDATE control.month_builds SET status='RUNNING',message='' WHERE build_id=?", [build_id])
            return
        self.db.execute("BEGIN TRANSACTION")
        try:
            self.db.execute("""INSERT INTO control.month_builds VALUES
                (?,?,NULL,'RUNNING',?,?,?,?,?,?,?,?,?,'',FALSE)""",
                [build_id, utc_now(), metadata["snapshot_id"], metadata["config_json"],
                 metadata["source_database"], metadata["source_run_id"], metadata["source_fingerprint"],
                 metadata["manifest_sha256"], metadata["code_fingerprint"], len(windows),
                 json.dumps(metadata["input_validation"], default=str, sort_keys=True)])
            self.db.executemany("INSERT INTO control.month_intervals VALUES (?,?,?,'PENDING',0,NULL,NULL,NULL)",
                [(build_id, w["interval_id"], json.dumps(w, sort_keys=True, default=str)) for w in windows])
            self.db.execute("COMMIT")
        except Exception:
            self.db.execute("ROLLBACK")
            raise

    def completed(self, build_id):
        return {r[0] for r in self.db.execute(
            "SELECT interval_id FROM control.month_intervals WHERE build_id=? AND status='COMPLETED'", [build_id]).fetchall()}

    def register_shared(self, build_id, catalog):
        """References preserve full original output inventory without copying histories."""
        for row in catalog.itertuples(index=False):
            self.db.execute("INSERT INTO control.month_outputs VALUES (?,?,?,?,?,?) ON CONFLICT DO NOTHING",
                [build_id, row.domain, row.output_name, "SOURCE_REFERENCE", row.table_name, row.schema_json])

    def _append(self, build_id, domain, name, scope, frame):
        identifier(domain)
        identifier(name)
        table_name = result_table(build_id, domain, name, scope)
        table = "month_results." + identifier(table_name)
        prepared = storage_frame(frame)
        if not prepared.columns.is_unique or not len(prepared.columns):
            raise ValueError("Result tables must have explicit unique column names.")
        self.db.register("month_stage", prepared)
        try:
            record = self.db.execute("SELECT table_name FROM control.month_outputs WHERE build_id=? AND domain=? AND output_name=? AND scope=?",
                [build_id, domain, name, scope]).fetchone()
            if record is None:
                self.db.execute(f"CREATE TABLE {table} AS SELECT * FROM month_stage WHERE FALSE")
                self.db.execute("INSERT INTO control.month_outputs VALUES (?,?,?,?,?,?)",
                    [build_id, domain, name, scope, table_name,
                     json.dumps(dict(zip(prepared.columns, map(str, prepared.dtypes))), sort_keys=True)])
            else:
                if record[0] != table_name:
                    raise ValueError("Conflicting output identity.")
                # Availability diagnostics legitimately have different fields
                # when the first endpoint lacks a complete estimator history.
                present = {r[0]: r[1] for r in self.db.execute(f"DESCRIBE {table}").fetchall()}
                observed = self.db.execute("DESCRIBE month_stage").fetchall()
                for column, sql_type, *_ in observed:
                    quoted = '"' + column.replace('"', '""') + '"'
                    if column not in present:
                        self.db.execute(f"ALTER TABLE {table} ADD COLUMN {quoted} {sql_type}")
                # Check readback in the exact source frame's column order;
                # older availability-only rows retain NULL for added fields.
            # Nullable metrics may be all NULL in the first interval. Later frames
            # must retain the logical schema rather than inherit accidental types.
            # These private tables are append-only. Verify the newly appended
            # row-id range, rather than repeatedly scanning every older period.
            previous_row = self.db.execute(f"SELECT coalesce(max(rowid),-1) FROM {table}").fetchone()[0]
            self.db.execute(f"INSERT INTO {table} BY NAME SELECT * FROM month_stage")
            columns = ','.join('t."' + c.replace('"', '""') + '"' for c in prepared.columns)
            selected = f"SELECT {columns} FROM {table} t WHERE rowid>{int(previous_row)}"
            mismatch = self.db.execute(f"""SELECT count(*) FROM (
                ({selected} EXCEPT ALL SELECT * FROM month_stage)
                UNION ALL (SELECT * FROM month_stage EXCEPT ALL {selected}))""").fetchone()[0]
            if mismatch:
                raise ValueError(f"Exact result readback mismatch: {domain}/{name}")
        finally:
            self.db.unregister("month_stage")

    def save_batch(self, build_id, results):
        """An interval becomes visible as completed only with all six validated domains."""
        grouped = {}
        self.db.execute("BEGIN TRANSACTION")
        try:
            for w, outputs, validations, timings in results:
                iid = w["interval_id"]
                state = self.db.execute("SELECT status FROM control.month_intervals WHERE build_id=? AND interval_id=?",
                                        [build_id, iid]).fetchone()
                if state is None or state[0] != "PENDING":
                    raise ValueError("Unknown or already completed interval.")
                rows = 0
                for domain, tables in outputs.items():
                    for name, frame in tables.items():
                        if "interval_id" in frame or "result_row" in frame:
                            raise ValueError("Reserved output key collision.")
                        frame = frame.copy()
                        frame.insert(0, "interval_id", iid)
                        frame.insert(1, "result_row", range(len(frame)))
                        grouped.setdefault((domain, name), []).append(frame)
                        rows += len(frame)
                self.db.execute("""UPDATE control.month_intervals SET status='COMPLETED',rows_saved=?,
                    completed_at=?,validation_json=?,timings_json=? WHERE build_id=? AND interval_id=?""",
                    [rows, utc_now(), json.dumps(validations, default=str), json.dumps(timings), build_id, iid])
            for (domain, name), frames in grouped.items():
                self._append(build_id, domain, name, "PERIOD", pd.concat(frames, ignore_index=True))
            self.db.execute("COMMIT")
        except Exception:
            self.db.execute("ROLLBACK")
            raise

    def save_closing(self, build_id, end_month, outputs):
        """Store end-date estimator tables once, independently of reporting starts."""
        name = next(iter(outputs))
        match = self.db.execute("SELECT table_name FROM control.month_outputs WHERE build_id=? AND output_name=? AND scope='CLOSING_MONTH'",
                                [build_id, name]).fetchone()
        if match and self.db.execute("SELECT count(*) FROM month_results." + identifier(match[0]) + " WHERE end_month=?",
                                     [end_month]).fetchone()[0]:
            return
        self.db.execute("BEGIN TRANSACTION")
        try:
            for name, frame in outputs.items():
                frame = frame.copy()
                frame.insert(0, "end_month", end_month)
                frame.insert(1, "result_row", range(len(frame)))
                self._append(build_id, "risk", name, "CLOSING_MONTH", frame)
            self.db.execute("COMMIT")
        except Exception:
            self.db.execute("ROLLBACK")
            raise

    def finish(self, build_id):
        expected = self.db.execute("SELECT expected_intervals FROM control.month_builds WHERE build_id=?", [build_id]).fetchone()[0]
        count = self.db.execute("SELECT count(*) FROM control.month_intervals WHERE build_id=? AND status='COMPLETED'", [build_id]).fetchone()[0]
        if count != expected:
            raise ValueError(f"Incomplete build: {count}/{expected} intervals.")
        self.db.execute("UPDATE control.month_builds SET status='COMPLETED',completed_at=?,message='' WHERE build_id=?",
                        [utc_now(), build_id])

    def fail(self, build_id, exc):
        self.db.execute("UPDATE control.month_builds SET status='FAILED',message=? WHERE build_id=? AND status<>'COMPLETED'",
                        [f"{type(exc).__name__}: {exc}", build_id])


def list_month_builds(database, *, include_incomplete=False):
    if not Path(database).is_file():
        return pd.DataFrame()
    with duckdb.connect(str(database), read_only=True) as db:
        where = "" if include_incomplete else " WHERE status='COMPLETED'"
        return db.execute("SELECT * FROM control.month_builds" + where + " ORDER BY created_at DESC").df()


def read_month_result(database, build_id, window, domain, output_name):
    """Read a completed selected interval. No analytics functions are called."""
    from portfolio_analytics.common.month_windows import MonthWindow
    if not isinstance(window, MonthWindow):
        raise TypeError("Supply a validated MonthWindow selection.")
    with duckdb.connect(str(database), read_only=True) as db:
        build = db.execute("SELECT status,source_database FROM control.month_builds WHERE build_id=?", [build_id]).fetchone()
        if not build or build[0] != "COMPLETED":
            raise ValueError("Only completed validated builds are available for selection.")
        interval = db.execute("SELECT status,definition_json FROM control.month_intervals WHERE build_id=? AND interval_id=?",
                              [build_id, window.interval_id]).fetchone()
        if not interval or interval[0] != "COMPLETED":
            raise KeyError("The selected interval is unavailable.")
        record = db.execute("""SELECT scope,table_name FROM control.month_outputs
            WHERE build_id=? AND domain=? AND output_name=?
            ORDER BY CASE scope WHEN 'PERIOD' THEN 0 WHEN 'CLOSING_MONTH' THEN 1 ELSE 2 END LIMIT 1""",
            [build_id, domain, output_name]).fetchone()
        if not record:
            raise KeyError(f"Output unavailable: {domain}/{output_name}")
        scope, table = record
        if scope == "SOURCE_REFERENCE":
            raise ValueError("This is a shared history/reference, not an interval summary; use read_month_history.")
        key, value = ("interval_id", window.interval_id) if scope == "PERIOD" else ("end_month", str(window.end))
        return db.execute("SELECT * FROM month_results." + identifier(table) + f" WHERE {key}=? ORDER BY result_row", [value]).df()


def read_month_history(database, build_id, window, domain, output_name, *, date_column="date"):
    """Filter a shared dated history; never mislabel an old summary as a new interval."""
    from .results import qualified_result
    identifier(date_column)
    with duckdb.connect(str(database), read_only=True) as db:
        build = db.execute("SELECT status,source_database FROM control.month_builds WHERE build_id=?", [build_id]).fetchone()
        if not build or build[0] != "COMPLETED":
            raise ValueError("Only completed validated builds are available.")
        row = db.execute("SELECT definition_json FROM control.month_intervals WHERE build_id=? AND interval_id=? AND status='COMPLETED'",
                         [build_id, window.interval_id]).fetchone()
        record = db.execute("SELECT table_name,schema_json FROM control.month_outputs WHERE build_id=? AND domain=? AND output_name=? AND scope='SOURCE_REFERENCE'",
                            [build_id, domain, output_name]).fetchone()
        if not row or not record:
            raise KeyError("Interval or shared history unavailable.")
        if date_column not in json.loads(record[1]):
            raise ValueError("This source table has no requested date column; it is not a selectable daily history.")
        if any(key in json.loads(record[1]) for key in ("period_id", "window_id", "report_window")):
            raise ValueError("A historical period-specific table cannot be reused as a new interval history.")
        definition = json.loads(row[0])
        source = build[1]
    with duckdb.connect(source, read_only=True) as db:
        return db.execute(f"SELECT * FROM {qualified_result(db, record[0])} WHERE {identifier(date_column)} BETWEEN ? AND ?",
            [definition["first_return_date"], definition["last_return_date"]]).df()
