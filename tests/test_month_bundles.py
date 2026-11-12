"""Remote-friendly month bundles must return exactly what the DuckDB readers return."""
import duckdb
import pandas as pd
import pytest

from portfolio_analytics.database.month_bundles import BundleReader, export_month_bundles, verify_bundles
from portfolio_analytics.database.month_results import MonthResultWriter

WINDOWS = [
    dict(interval_id="2016-10_2017-09", calendar_start="2016-10-01", calendar_end="2017-09-30",
         first_return_date="2016-10-03", last_return_date="2017-09-29"),
    dict(interval_id="2016-11_2017-12", calendar_start="2016-11-01", calendar_end="2017-12-31",
         first_return_date="2016-11-01", last_return_date="2017-12-29"),
]


def build_databases(tmp_path):
    source = tmp_path / "private" / "source.duckdb"
    source.parent.mkdir()
    dates = pd.bdate_range("2016-09-30", "2018-01-31")
    history = pd.DataFrame({"portfolio_id": "P", "date": dates, "value": range(len(dates)),
                            "presence": ["both", "left_only"] * (len(dates) // 2) + ["both"] * (len(dates) % 2)})
    with duckdb.connect(str(source)) as db:
        db.execute("CREATE SCHEMA analytics")
        db.execute("CREATE TYPE presence_type AS ENUM ('left_only', 'right_only', 'both')")
        db.register("frame", history.sample(frac=1, random_state=3))  # deliberately unsorted
        db.execute("CREATE TABLE analytics.daily_history AS SELECT portfolio_id, date, value, "
                   "presence::presence_type AS presence FROM frame")
    months = tmp_path / "months.duckdb"
    writer = MonthResultWriter(months)
    try:
        writer.begin_build("build", dict(snapshot_id="snapshot", config_json='{"portfolio_id": "P", "benchmark_id": "B"}',
                                         source_database=str(source), source_run_id="run", source_fingerprint="fp",
                                         manifest_sha256="sha", code_fingerprint="code", input_validation={}), WINDOWS)
        writer.register_shared("build", pd.DataFrame({
            "domain": ["performance"], "output_name": ["daily_history"], "table_name": ["daily_history"],
            "schema_json": ['{"portfolio_id": "str", "date": "datetime64[ns]", "value": "int64", "presence": "category"}']}))
        batch = []
        for number, w in enumerate(WINDOWS):
            outputs = {"performance": {"summary": pd.DataFrame({"portfolio_id": ["P", "B"], "period_return": [0.1 + number, 0.05]}),
                                       "empty_table": pd.DataFrame({"portfolio_id": pd.Series([], dtype=str)})}}
            batch.append((w, outputs, [{"domain": "performance", "status": "PASSED", "rows": 2, "statuses": {}}], {}))
        writer.save_batch("build", batch)
        writer.finish("build")
    finally:
        writer.close()
    return months, source


@pytest.mark.parametrize("core_limit", [10_000_000, 0], ids=["history-in-core", "history-split-by-year"])
def test_bundles_match_duckdb_readers_exactly(tmp_path, core_limit):
    months, source = build_databases(tmp_path)
    output = tmp_path / "bundles"
    manifest = export_month_bundles(months, output, core_limit_bytes=core_limit, progress=lambda message: None)
    assert manifest["complete"] and manifest["interval_count"] == 2
    layout = manifest["histories"]["performance__daily_history"]
    assert layout["layout"] == ("core" if core_limit else "yearly")
    assert verify_bundles(months, output, [w["interval_id"] for w in WINDOWS], progress=lambda message: None) == []
    # No local file paths are copied into the published bundle files.
    for path in output.rglob("*"):
        if path.is_file():
            assert str(source).encode() not in path.read_bytes()
            assert str(source.parent).encode() not in path.read_bytes()


def test_tampered_or_incomplete_bundles_are_refused(tmp_path):
    months, _ = build_databases(tmp_path)
    output = tmp_path / "bundles"
    export_month_bundles(months, output, progress=lambda message: None)
    archive = output / "periods" / "2016-10_2017-09.zip"
    archive.write_bytes(archive.read_bytes() + b"x")
    reader = BundleReader(output)
    from portfolio_analytics.common.month_windows import MonthWindow
    window = MonthWindow(pd.Period("2016-10", "M"), pd.Period("2017-09", "M"))
    with pytest.raises(ValueError, match="checksum"):
        reader.result("build", window, "performance", "summary")

    trial = tmp_path / "trial"
    export_month_bundles(months, trial, interval_limit=1, progress=lambda message: None)
    with pytest.raises(ValueError, match="incomplete"):
        BundleReader(trial).manifest()
    assert BundleReader(trial, allow_incomplete=True).manifest()["interval_count"] == 1
    with pytest.raises(FileExistsError):
        export_month_bundles(months, trial, progress=lambda message: None)
