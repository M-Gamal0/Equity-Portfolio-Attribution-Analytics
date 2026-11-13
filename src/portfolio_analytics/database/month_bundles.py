"""Remote-friendly bundles of a completed month-window build.

A completed build stays authoritative in DuckDB. This module copies it into files that a
hosted dashboard can fetch with very few HTTP requests:

* ``core.zip``: the output catalog, interval definitions and the small dated histories.
* ``periods/<interval_id>.zip``: every period result table for one complete-month window.
* ``closing/<end_month>.zip``: every closing-month estimator table for one ending month.
* ``history/<domain>__<name>/<year>.parquet``: large dated histories, split by year.
* ``manifest.json``: build identity, file checksums and the history layout.

The reader returns exactly what ``read_month_result`` and ``read_month_history`` return for
the same build, window and table. No analytics are recalculated, and no local file paths
are copied into the bundles.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import threading
import zipfile
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .month_results import identifier, resolve_source_database

FORMAT_VERSION = 1
MANIFEST = "manifest.json"
CORE = "core.zip"
ROW_ORDER = "__row_order"
PERIOD_COLUMNS = {"period_id", "window_id", "report_window"}
DATE_COLUMNS = ("date", "effective_rebalance_date", "estimation_date")
# Histories the dashboard opens on every page are kept in core.zip regardless of size.
ALWAYS_CORE = {("exposures", "security_exposures"), ("exposures", "sector_exposures"),
               ("exposures", "active_share"), ("factors", "portfolio_factor_exposures"),
               ("performance", "performance_rolling_relative"), ("risk", "active_risk_rolling"),
               ("risk", "risk_rolling")}


# --------------------------------------------------------------------------- helpers

def _parquet_bytes(table: pa.Table) -> bytes:
    buffer = io.BytesIO()
    pq.write_table(table, buffer, compression="zstd")
    return buffer.getvalue()


def _frame_from_parquet(data: bytes, sql: str = "SELECT * FROM t", params=None) -> pd.DataFrame:
    """Convert through DuckDB so pandas dtypes match the original DuckDB readers."""
    table = pq.read_table(io.BytesIO(data))
    with duckdb.connect() as con:
        con.register("t", table)
        return con.execute(sql, params or []).df()


def _enum_categories(table: pa.Table) -> dict[str, list[str]]:
    """DuckDB ENUM columns arrive as dictionary arrays; keep their full category order."""
    enums = {}
    for field in table.schema:
        if pa.types.is_dictionary(field.type):
            column = table.column(field.name)
            if column.num_chunks == 0:
                raise ValueError(f"Cannot record categories for empty ENUM column {field.name}.")
            enums[field.name] = column.chunk(0).dictionary.to_pylist()
    return enums


def _restore_enums(frame: pd.DataFrame, enums: dict[str, list[str]]) -> pd.DataFrame:
    for column, categories in (enums or {}).items():
        # DuckDB ENUMs convert to ordered pandas categoricals.
        frame[column] = pd.Categorical(frame[column], categories=categories, ordered=True)
    return frame


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def history_key(domain: str, name: str) -> str:
    return f"{domain}__{name}"


def selectable_history(schema: dict) -> str | None:
    """Return the dashboard's date column for a dated shared history, else None."""
    if PERIOD_COLUMNS.intersection(schema):
        return None
    return next((c for c in DATE_COLUMNS if c in schema), None)


def _zip(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(members):
            info = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
            archive.writestr(info, members[name])
    return buffer.getvalue()


def _unzip(data: bytes) -> dict[str, bytes]:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


# --------------------------------------------------------------------------- export

class _Writer:
    def __init__(self, root: Path):
        self.root = root
        self.files: dict[str, str] = {}

    def write(self, relative: str, data: bytes):
        target = self.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        self.files[relative] = _sha256(data)


def export_month_bundles(database, output, *, build_id=None, core_limit_bytes=3_000_000,
                         chunk_size=100, interval_limit=None, progress=print):
    """Write bundles for a completed build. Returns the manifest dictionary.

    ``interval_limit`` exports only the first N intervals and is intended for size and
    timing trials; such a manifest is marked incomplete.
    """
    database = Path(database)
    root = Path(output)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("The output folder must be empty; bundles are never overwritten.")
    root.mkdir(parents=True, exist_ok=True)
    writer = _Writer(root)
    with duckdb.connect(str(database), read_only=True) as db:
        builds = db.execute("""SELECT build_id, status, snapshot_id, config_json, source_database
                               FROM control.month_builds WHERE status='COMPLETED'
                               ORDER BY completed_at DESC""").fetchall()
        if not builds:
            raise ValueError("No completed month-window build is available.")
        build = next((b for b in builds if build_id in (None, b[0])), None)
        if build is None:
            raise ValueError("The requested build is not a completed build.")
        build_id, _, snapshot_id, config_json, stored_source = build
        catalog = db.execute("SELECT * FROM control.month_outputs WHERE build_id=?", [build_id]).to_arrow_table()
        intervals = db.execute("""SELECT interval_id, definition_json, validation_json
                                  FROM control.month_intervals
                                  WHERE build_id=? AND status='COMPLETED' ORDER BY interval_id""",
                               [build_id]).to_arrow_table()
        outputs = db.execute("""SELECT domain, output_name, scope, table_name, schema_json
                                FROM control.month_outputs WHERE build_id=?""", [build_id]).fetchall()
        interval_ids = intervals.column("interval_id").to_pylist()
        if interval_limit:
            interval_ids = interval_ids[:interval_limit]
        period = [(d, n, t) for d, n, s, t, _ in outputs if s == "PERIOD"]
        closing = [(d, n, t) for d, n, s, t, _ in outputs if s == "CLOSING_MONTH"]

        # Period tables: read chunks of intervals once per table, then split per interval.
        for start in range(0, len(interval_ids), chunk_size):
            chunk = interval_ids[start:start + chunk_size]
            members: dict[str, dict[str, bytes]] = {iid: {} for iid in chunk}
            for domain, name, table in period:
                arrow = db.execute(
                    f"SELECT * FROM month_results.{identifier(table)} WHERE interval_id IN "
                    f"(SELECT unnest(?)) ORDER BY interval_id, result_row", [chunk]).to_arrow_table()
                if any(pa.types.is_dictionary(f.type) for f in arrow.schema):
                    raise ValueError(f"ENUM columns are not supported in period table {domain}/{name}.")
                ids = arrow.column("interval_id").to_pylist()
                bounds = {}
                for position, value in enumerate(ids):
                    first, _ = bounds.get(value, (position, position))
                    bounds[value] = (first, position + 1)
                for iid in chunk:
                    first, end = bounds.get(iid, (0, 0))
                    members[iid][history_key(domain, name) + ".parquet"] = _parquet_bytes(arrow.slice(first, end - first))
            for iid in chunk:
                writer.write(f"periods/{iid}.zip", _zip(members[iid]))
            progress(f"periods: {min(start + chunk_size, len(interval_ids))}/{len(interval_ids)}")

        # Closing-month estimators: one file per ending month.
        end_months = sorted({iid.split("_")[1] for iid in interval_ids})
        for end_month in end_months:
            files = {}
            for domain, name, table in closing:
                arrow = db.execute(f"SELECT * FROM month_results.{identifier(table)} WHERE end_month=? "
                                   "ORDER BY result_row", [end_month]).to_arrow_table()
                if any(pa.types.is_dictionary(f.type) for f in arrow.schema):
                    raise ValueError(f"ENUM columns are not supported in closing table {domain}/{name}.")
                files[history_key(domain, name) + ".parquet"] = _parquet_bytes(arrow)
            writer.write(f"closing/{end_month}.zip", _zip(files))
        progress(f"closing months: {len(end_months)}")

    # Shared dated histories from the source database.
    source = resolve_source_database(stored_source, database)
    histories = {}
    core_files = {"catalog.parquet": _parquet_bytes(catalog),
                  "intervals.parquet": _parquet_bytes(intervals)}
    with duckdb.connect(str(source), read_only=True) as src:
        schema = src.execute("SELECT current_database()").fetchone()[0].replace('"', '""')
        for domain, name, scope, table, schema_json in outputs:
            if scope != "SOURCE_REFERENCE":
                continue
            date_column = selectable_history(json.loads(schema_json))
            if date_column is None:
                continue
            qualified = f'"{schema}".analytics.{identifier(table)}'
            # Same scan as read_month_history; number rows in the order they are returned.
            arrow = src.execute(f"SELECT * FROM {qualified}").to_arrow_table()
            arrow = arrow.append_column(ROW_ORDER, pa.array(range(arrow.num_rows), pa.int64()))
            enums = _enum_categories(arrow)
            data = _parquet_bytes(arrow)
            key = history_key(domain, name)
            if (domain, name) in ALWAYS_CORE or len(data) <= core_limit_bytes:
                core_files[f"histories/{key}.parquet"] = data
                histories[key] = {"layout": "core", "split_column": None, "years": [], "enums": enums}
                continue
            years = sorted({d.year for d in arrow.column(date_column).to_pylist() if d is not None})
            with duckdb.connect() as con:
                con.register("t", arrow)
                for year in years:
                    part = con.execute(f"SELECT * FROM t WHERE year({identifier(date_column)})=? "
                                       f"ORDER BY {ROW_ORDER}", [year]).to_arrow_table()
                    writer.write(f"history/{key}/{year}.parquet", _parquet_bytes(part))
                missing = con.execute(f"SELECT * FROM t WHERE {identifier(date_column)} IS NULL "
                                      f"ORDER BY {ROW_ORDER}").to_arrow_table()
            if missing.num_rows:
                writer.write(f"history/{key}/none.parquet", _parquet_bytes(missing))
            histories[key] = {"layout": "yearly", "split_column": date_column, "years": years,
                              "has_missing_dates": bool(missing.num_rows), "enums": enums}
    writer.write(CORE, _zip(core_files))
    progress(f"histories: {len(histories)} ({sum(h['layout'] == 'yearly' for h in histories.values())} split by year)")

    manifest = {"format_version": FORMAT_VERSION, "build_id": build_id, "snapshot_id": snapshot_id,
                "config_json": config_json, "status": "COMPLETED",
                "complete": interval_limit is None,
                "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "interval_count": len(interval_ids), "histories": histories, "files": writer.files}
    (root / MANIFEST).write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    return manifest


def pack_month_bundles(source, output, *, pack_bytes=500_000_000, progress=print):
    """Combine exported bundle files into a few large packs, byte for byte.

    Hosting services throttle uploads of thousands of small files. Packing keeps each
    original file intact at a recorded offset, so a reader still fetches exactly one file
    per request, using an HTTP range request. Checksums of the original files are kept.
    """
    source, root = Path(source), Path(output)
    manifest = json.loads((source / MANIFEST).read_text(encoding="utf-8"))
    if not manifest.get("complete"):
        raise ValueError("Only complete exports can be packed.")
    if manifest.get("packed"):
        raise ValueError("These bundles are already packed.")
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("The output folder must be empty; bundles are never overwritten.")
    (root / "packs").mkdir(parents=True, exist_ok=True)
    (root / CORE).write_bytes((source / CORE).read_bytes())
    packed, pack_files = {}, {}
    pack_number, handle, digest, offset = -1, None, None, 0

    def close_pack():
        if handle is not None:
            handle.close()
            pack_files[name] = digest.hexdigest()

    members = sorted(r for r in manifest["files"] if r != CORE)
    for relative in members:
        data = (source / relative).read_bytes()
        if _sha256(data) != manifest["files"][relative]:
            raise ValueError(f"Source bundle file failed its checksum: {relative}")
        if handle is None or offset + len(data) > pack_bytes and offset > 0:
            close_pack()
            pack_number += 1
            name = f"packs/pack-{pack_number:03d}.bin"
            handle, digest, offset = open(root / name, "wb"), hashlib.sha256(), 0
        handle.write(data)
        digest.update(data)
        packed[relative] = [name, offset, len(data)]
        offset += len(data)
    close_pack()
    manifest["packed"] = packed
    manifest["pack_files"] = pack_files
    (root / MANIFEST).write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    progress(f"packed {len(packed)} files into {len(pack_files)} packs")
    return manifest


# --------------------------------------------------------------------------- reader

class BundleReader:
    """Read bundles from a local folder or a Hugging Face dataset (``hf://datasets/owner/name``)."""

    def __init__(self, source, *, revision=None, cache_entries=24, allow_incomplete=False):
        self.source = str(source)
        self.revision = revision
        self.allow_incomplete = allow_incomplete
        self.cache_entries = cache_entries
        self._manifest = None
        self._core = None
        self._metadata = None
        self._files = OrderedDict()
        self._lock = threading.RLock()

    def _cached(self, relative: str) -> bytes:
        """Keep recently used bundle files in memory so one Apply fetches each file once."""
        with self._lock:
            if relative in self._files:
                self._files.move_to_end(relative)
                return self._files[relative]
        data = self.fetch(relative)
        with self._lock:
            self._files[relative] = data
            while len(self._files) > self.cache_entries:
                self._files.popitem(last=False)
        return data

    @property
    def _remote(self) -> bool:
        return self.source.startswith("hf://datasets/")

    def _repo(self) -> str:
        return self.source[len("hf://datasets/"):].strip("/")

    def _read_whole(self, relative: str) -> bytes:
        if self._remote:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(repo_id=self._repo(), filename=relative, repo_type="dataset",
                                   revision=self.revision)
            return Path(path).read_bytes()
        return (Path(self.source) / relative).read_bytes()

    def _read_range(self, pack: str, offset: int, length: int) -> bytes:
        """Read one packed file with a single HTTP range request (or a local seek)."""
        if self._remote:
            from huggingface_hub import HfFileSystem
            with self._lock:
                if getattr(self, "_fs", None) is None:
                    self._fs = HfFileSystem()
            revision = f"@{self.revision}" if self.revision else ""
            data = self._fs.cat_file(f"datasets/{self._repo()}{revision}/{pack}",
                                     start=offset, end=offset + length)
        else:
            with open(Path(self.source) / pack, "rb") as handle:
                handle.seek(offset)
                data = handle.read(length)
        if len(data) != length:
            raise ValueError(f"Short read from bundle pack: {pack}")
        return data

    def fetch(self, relative: str) -> bytes:
        if relative == MANIFEST:
            return self._read_whole(relative)
        location = self.manifest().get("packed", {}).get(relative)
        data = self._read_range(*location) if location else self._read_whole(relative)
        expected = self.manifest()["files"].get(relative)
        if expected is not None and _sha256(data) != expected:
            raise ValueError(f"Bundle file failed its checksum: {relative}")
        return data

    def manifest(self) -> dict:
        if self._manifest is None:
            raw = self.fetch(MANIFEST)
            manifest = json.loads(raw.decode("utf-8"))
            if manifest.get("format_version") != FORMAT_VERSION:
                raise ValueError("Unsupported bundle format.")
            if not manifest.get("complete") and not self.allow_incomplete:
                raise ValueError("These bundles are an incomplete trial export.")
            self._manifest = manifest
        return self._manifest

    def _core_members(self) -> dict[str, bytes]:
        with self._lock:
            if self._core is None:
                self._core = _unzip(self.fetch(CORE))
            return self._core

    def builds(self) -> pd.DataFrame:
        m = self.manifest()
        return pd.DataFrame([{"build_id": m["build_id"], "status": m["status"],
                              "snapshot_id": m["snapshot_id"], "config_json": m["config_json"]}])

    def _check_build(self, build_id):
        if build_id != self.manifest()["build_id"]:
            raise ValueError("Only completed validated builds are available for selection.")

    def metadata(self, build_id):
        self._check_build(build_id)
        if self._metadata is None:
            core = self._core_members()
            self._metadata = (_frame_from_parquet(core["catalog.parquet"]),
                              _frame_from_parquet(core["intervals.parquet"]))
        catalog, intervals = self._metadata
        return catalog.copy(), intervals.copy()

    def _catalog_entry(self, domain, name, scopes):
        catalog, _ = self.metadata(self.manifest()["build_id"])
        rows = catalog.loc[catalog.domain.eq(domain) & catalog.output_name.eq(name) & catalog.scope.isin(scopes)]
        order = {"PERIOD": 0, "CLOSING_MONTH": 1, "SOURCE_REFERENCE": 2}
        return rows.assign(rank=rows.scope.map(order)).sort_values("rank").head(1)

    def _interval(self, window):
        _, intervals = self.metadata(self.manifest()["build_id"])
        row = intervals.loc[intervals.interval_id.eq(window.interval_id)]
        if row.empty:
            raise KeyError("The selected interval is unavailable.")
        return json.loads(row.iloc[0].definition_json)

    def result(self, build_id, window, domain, output_name) -> pd.DataFrame:
        """Equivalent to ``read_month_result`` for the same build."""
        self._check_build(build_id)
        self._interval(window)
        entry = self._catalog_entry(domain, output_name, ["PERIOD", "CLOSING_MONTH", "SOURCE_REFERENCE"])
        if entry.empty:
            raise KeyError(f"Output unavailable: {domain}/{output_name}")
        scope = entry.iloc[0].scope
        if scope == "SOURCE_REFERENCE":
            raise ValueError("This is a shared history/reference, not an interval summary; use read_month_history.")
        archive = f"periods/{window.interval_id}.zip" if scope == "PERIOD" else f"closing/{window.end}.zip"
        members = _unzip(self._cached(archive))
        return _frame_from_parquet(members[history_key(domain, output_name) + ".parquet"],
                                   "SELECT * FROM t ORDER BY result_row")

    def history(self, build_id, window, domain, output_name, *, date_column="date") -> pd.DataFrame:
        """Equivalent to ``read_month_history`` for the same build."""
        identifier(date_column)
        self._check_build(build_id)
        definition = self._interval(window)
        entry = self._catalog_entry(domain, output_name, ["SOURCE_REFERENCE"])
        if entry.empty:
            raise KeyError("Interval or shared history unavailable.")
        schema = json.loads(entry.iloc[0].schema_json)
        if date_column not in schema:
            raise ValueError("This source table has no requested date column; it is not a selectable daily history.")
        if PERIOD_COLUMNS.intersection(schema):
            raise ValueError("A historical period-specific table cannot be reused as a new interval history.")
        key = history_key(domain, output_name)
        layout = self.manifest()["histories"].get(key)
        if layout is None:
            raise KeyError("Interval or shared history unavailable.")
        if layout["layout"] == "core":
            parts = [self._core_members()[f"histories/{key}.parquet"]]
        else:
            first = pd.Timestamp(definition["first_return_date"]).year
            last = pd.Timestamp(definition["last_return_date"]).year
            years = [y for y in layout["years"]
                     if layout["split_column"] != date_column or first <= y <= last]
            names = [f"history/{key}/{y}.parquet" for y in years]
            if layout["split_column"] != date_column and layout.get("has_missing_dates"):
                names.append(f"history/{key}/none.parquet")
            parts = [self._cached(n) for n in names]
        tables = [pq.read_table(io.BytesIO(p)) for p in parts]
        table = pa.concat_tables(tables) if tables else None
        with duckdb.connect() as con:
            con.register("t", table)
            frame = con.execute(f"SELECT * FROM t WHERE {identifier(date_column)} BETWEEN ? AND ? "
                                f"ORDER BY {ROW_ORDER}",
                                [definition["first_return_date"], definition["last_return_date"]]).df()
        frame = frame.drop(columns=[ROW_ORDER]).reset_index(drop=True)
        return _restore_enums(frame, layout.get("enums"))


def verify_packed_files(source, *, progress=print):
    """Read every packed file through the reader; each must match its original checksum."""
    reader = BundleReader(source)
    manifest = reader.manifest()
    for name, expected in manifest.get("pack_files", {}).items():
        digest = hashlib.sha256()
        with open(Path(source) / name, "rb") as handle:
            for block in iter(lambda: handle.read(16 << 20), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError(f"Pack failed its checksum: {name}")
    for relative in manifest.get("packed", {}):
        reader.fetch(relative)  # raises on any checksum difference
    progress(f"all {len(manifest.get('packed', {}))} packed files match their original checksums")


def verify_bundles(database, source, interval_ids, *, progress=print):
    """Compare every table for the given intervals with the DuckDB readers. Returns mismatches."""
    from portfolio_analytics.common.month_windows import MonthWindow
    from .month_results import read_month_history, read_month_result

    reader = BundleReader(source)
    build_id = reader.manifest()["build_id"]
    catalog, _ = reader.metadata(build_id)
    mismatches, checked = [], 0
    for number, iid in enumerate(interval_ids, 1):
        start, end = iid.split("_")
        window = MonthWindow(pd.Period(start, "M"), pd.Period(end, "M"))
        for row in catalog.itertuples():
            if row.scope in ("PERIOD", "CLOSING_MONTH"):
                expected = read_month_result(database, build_id, window, row.domain, row.output_name)
                actual = reader.result(build_id, window, row.domain, row.output_name)
            else:
                column = selectable_history(json.loads(row.schema_json))
                if column is None:
                    continue
                expected = read_month_history(database, build_id, window, row.domain, row.output_name,
                                              date_column=column)
                actual = reader.history(build_id, window, row.domain, row.output_name, date_column=column)
            try:
                pd.testing.assert_frame_equal(expected, actual, check_exact=True)
            except AssertionError as error:
                mismatches.append((iid, row.scope, row.domain, row.output_name, str(error).splitlines()[0]))
            checked += 1
        progress(f"verified {number}/{len(interval_ids)} periods ({checked} tables, {len(mismatches)} mismatches)")
    return mismatches


def bundle_source_from_environment():
    """Return the configured bundle source, or None to use the local DuckDB files."""
    return os.environ.get("PORTFOLIO_BUNDLE_SOURCE") or None
