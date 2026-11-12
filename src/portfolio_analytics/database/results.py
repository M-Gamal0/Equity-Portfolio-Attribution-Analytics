"""Single-writer transactional results; existing source databases are untouched.

Each result is a typed immutable table in analytics, indexed by a run manifest.
Run-specific tables allow explicit schema evolution without rewriting prior runs.
"""

import hashlib
import json
from pathlib import Path
import re

import duckdb
import numpy as np
import pandas as pd

from portfolio_analytics.pipeline.runner import AnalysisRun


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", value):
        raise ValueError("Invalid result identifier.")
    return '"' + value + '"'


def qualified_result(db, table):
    # DuckDB requires three parts when a database basename equals its schema.
    catalog = db.execute("SELECT current_database()").fetchone()[0]
    return '"' + catalog.replace('"', '""') + '".analytics.' + identifier(table)


def storage_frame(frame):
    result = frame.copy()
    if result.columns.duplicated().any() or not all(
        isinstance(c, str) for c in result.columns
    ):
        raise ValueError("Output column names must be unique strings.")
    for col in result:
        if isinstance(result[col].dtype, pd.PeriodDtype):
            # Preserve calendar labels (including fiscal quarters), not an
            # arbitrary timestamp at the start/end of the reporting period.
            # Nullable strings retain NaT as SQL NULL rather than the text 'NaT'.
            result[col] = result[col].astype("string")
        if (
            pd.api.types.is_numeric_dtype(result[col])
            and np.isinf(result[col].to_numpy(dtype=float, na_value=np.nan)).any()
        ):
            raise ValueError(f"Infinite result values in {col}.")
        if result[col].dtype == object:
            result[col] = result[col].map(
                lambda x: (
                    json.dumps(x, sort_keys=True, default=str)
                    if isinstance(x, (dict, list, tuple, np.ndarray))
                    else x
                )
            )
    return result


def save_run(run: AnalysisRun, database: str | Path):
    """Persist one complete manifest atomically; reject duplicate run identities."""
    path = Path(database).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    prepared = []
    for domain, outputs in run.outputs.items():
        for name, frame in outputs.items():
            identifier(domain)
            identifier(name)
            table = f"{domain}_{name}_{hashlib.sha256(run.run_id.encode()).hexdigest()[:16]}"
            prepared.append((domain, name, table, storage_frame(frame)))
    with duckdb.connect(str(path)) as db:
        db.execute("BEGIN TRANSACTION")
        try:
            db.execute("CREATE SCHEMA IF NOT EXISTS control")
            db.execute("CREATE SCHEMA IF NOT EXISTS analytics")
            db.execute("""CREATE TABLE IF NOT EXISTS control.analytics_runs (
                run_id VARCHAR PRIMARY KEY, created_at VARCHAR, status VARCHAR,
                snapshot_id VARCHAR, config_json VARCHAR, code_version VARCHAR,
                source_fingerprint VARCHAR, snapshot_manifest_sha256 VARCHAR,
                statuses_json VARCHAR, publication_approved BOOLEAN DEFAULT FALSE)""")
            db.execute("""CREATE TABLE IF NOT EXISTS control.analytics_outputs (
                run_id VARCHAR, domain VARCHAR, output_name VARCHAR, table_name VARCHAR,
                row_count BIGINT, schema_json VARCHAR,
                PRIMARY KEY (run_id, domain, output_name))""")
            db.execute(
                "INSERT INTO control.analytics_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, FALSE)",
                [
                    run.run_id,
                    run.created_at,
                    run.status,
                    run.snapshot_id,
                    run.config_json,
                    run.code_version,
                    run.source_fingerprint,
                    run.snapshot_manifest_sha256,
                    json.dumps(run.statuses),
                ],
            )
            for domain, name, table, frame in prepared:
                target = qualified_result(db, table)
                db.register("result_stage", frame)
                db.execute(f"CREATE TABLE {target} AS SELECT * FROM result_stage")
                count = db.execute(f"SELECT count(*) FROM {target}").fetchone()[0]
                if count != len(frame):
                    raise RuntimeError(f"Readback count mismatch: {domain}/{name}")
                # Compare all values through SQL multiset equality, including duplicates/nulls.
                mismatch = db.execute(f"""SELECT count(*) FROM (
                    (SELECT * FROM {target} EXCEPT ALL SELECT * FROM result_stage)
                    UNION ALL
                    (SELECT * FROM result_stage EXCEPT ALL SELECT * FROM {target})
                )""").fetchone()[0]
                if mismatch:
                    raise RuntimeError(f"Readback value mismatch: {domain}/{name}")
                db.execute(
                    "INSERT INTO control.analytics_outputs VALUES (?, ?, ?, ?, ?, ?)",
                    [
                        run.run_id,
                        domain,
                        name,
                        table,
                        len(frame),
                        json.dumps(dict(zip(frame.columns, map(str, frame.dtypes)))),
                    ],
                )
                db.unregister("result_stage")
            db.execute("COMMIT")
        except Exception:
            db.execute("ROLLBACK")
            raise
    # Independent post-commit connection checks the saved manifest and table counts.
    with duckdb.connect(str(path), read_only=True) as db:
        for _, _, table, frame in prepared:
            if db.execute(
                f"SELECT count(*) FROM {qualified_result(db, table)}"
            ).fetchone()[0] != len(frame):
                raise RuntimeError("Post-commit output verification failed.")
    return path


def list_runs(database):
    path = Path(database)
    if not path.is_file():
        return pd.DataFrame()
    with duckdb.connect(str(path), read_only=True) as db:
        exists = db.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_schema='control' AND table_name='analytics_runs'"
        ).fetchone()[0]
        if not exists:
            return pd.DataFrame()
        return db.execute(
            "SELECT * FROM control.analytics_runs ORDER BY created_at DESC"
        ).df()


def list_outputs(database, run_id):
    with duckdb.connect(str(database), read_only=True) as db:
        return db.execute(
            "SELECT * FROM control.analytics_outputs WHERE run_id=? ORDER BY domain, output_name",
            [run_id],
        ).df()


def read_output(database, run_id, domain, output_name, *, limit=None):
    if limit is not None and (type(limit) is not int or not 1 <= limit <= 1_000_000):
        raise ValueError("limit must be a positive integer up to 1,000,000.")
    with duckdb.connect(str(database), read_only=True) as db:
        row = db.execute(
            "SELECT table_name FROM control.analytics_outputs WHERE run_id=? AND domain=? AND output_name=?",
            [run_id, domain, output_name],
        ).fetchone()
        if row is None:
            raise KeyError(f"Output unavailable: {domain}/{output_name}")
        suffix = " LIMIT ?" if limit else ""
        return db.execute(
            f"SELECT * FROM {qualified_result(db, row[0])}{suffix}",
            [limit] if limit else [],
        ).df()
