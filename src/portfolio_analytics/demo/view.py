"""Prepare exactly one public five-year view using the existing month adapters."""
from dataclasses import asdict
from hashlib import sha256
import json

import duckdb
import pandas as pd

from portfolio_analytics.common.month_windows import MonthWindow
from portfolio_analytics.config import AnalysisSettings
from portfolio_analytics.database.month_results import MonthResultWriter, read_month_result
from portfolio_analytics.database.results import save_run
from portfolio_analytics.pipeline.month_calculations import MonthCalculationContext
from portfolio_analytics.pipeline.month_runner import validate_month_outputs
from portfolio_analytics.pipeline.runner import DOMAIN_INPUTS, code_provenance, run_analysis
from .build import ACTIVE_ID, BENCHMARK_ID, DemoConfig

INTERVAL_ID = "2021-10_2026-09"


def prepare_demo_view(workspace, *, snapshot_id=None, progress=print):
    """Offline-only preparation; the Streamlit entry point never calls this."""
    from .cli import _workspace
    workspace = _workspace(workspace)
    snapshots = workspace / "snapshots"
    if snapshot_id is None:
        snapshot_id = json.loads((snapshots / "latest.json").read_text(encoding="utf-8"))["snapshot_id"]
    if len(snapshot_id) != 16 or any(c not in "0123456789abcdef" for c in snapshot_id):
        raise ValueError("Expected a validated public snapshot ID")
    manifest = json.loads((snapshots / snapshot_id / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("profile") != "public_yahoo_quarterly_equal_weight_v1" or manifest["construction"] != asdict(DemoConfig()):
        raise ValueError("This viewer requires the fixed 50/20 public demo for September 2021–September 2026")
    _, fingerprint = code_provenance()
    folder = workspace / "streamlit" / f"{snapshot_id}_{fingerprint[:12]}"
    folder.mkdir(parents=True, exist_ok=True)
    view_file = folder / "view.json"
    pointer = workspace / "streamlit" / "latest.json"
    if view_file.exists():
        view = json.loads(view_file.read_text(encoding="utf-8"))
        verify_demo_view(view)
        pointer.write_text(json.dumps(view, indent=2), encoding="utf-8")
        progress("Reusing the completed fixed five-year demo view.", flush=True)
        return view
    settings = AnalysisSettings(ACTIVE_ID, BENCHMARK_ID, "2026-09-30", "2021-09-30",
        reporting_years=(1, 3, 5), performance_windows=("SI", "YTD", "1Y", "3Y", "5Y"),
        include_security_factor_history=True)
    source = folder / "analytics.duckdb"
    run = run_analysis(snapshot_root=snapshots, snapshot_id=snapshot_id, settings=settings,
        domains=tuple(DOMAIN_INPUTS), on_progress=lambda d, s: progress(f"{d}: {s}", flush=True))
    if run.status != "COMPLETED":
        raise ValueError(f"Public full-history preparation failed: {run.statuses}")
    save_run(run, source)
    context = MonthCalculationContext(snapshot_root=snapshots, snapshot_id=snapshot_id,
        settings=settings, source_database=source, source_run_id=run.run_id)
    window = MonthWindow(pd.Period("2021-10", "M"), pd.Period("2026-09", "M"))
    definition = window.resolve(context.calendar, settings.inception_date)
    closing = context.closing_risk(definition)
    outputs, timings = context.calculate(definition)
    validations = validate_month_outputs(context, definition, outputs)
    database = folder / "fixed_view.duckdb"
    metadata = dict(snapshot_id=snapshot_id, config_json=settings.to_json(), source_database=str(source),
        source_run_id=run.run_id, source_fingerprint=context.source_fingerprint,
        manifest_sha256=context.manifest_sha256, code_fingerprint=fingerprint,
        input_validation=context.input_validation)
    build_id = "demo_fixed_" + sha256((snapshot_id + fingerprint).encode()).hexdigest()[:20]
    writer = MonthResultWriter(database)
    try:
        writer.begin_build(build_id, metadata, [definition])
        writer.register_shared(build_id, context.catalog)
        writer.save_closing(build_id, definition["end_month"], closing)
        writer.save_batch(build_id, [(definition, outputs, validations, timings)])
        writer.finish(build_id)
    except Exception as exc:
        writer.fail(build_id, exc)
        raise
    finally:
        writer.close()
    view = {"profile": "public_demo_fixed_five_years_v1", "snapshot_id": snapshot_id,
            "build_id": build_id, "interval_id": INTERVAL_ID, "database": str(database),
            "source_database": str(source), "source_run_id": run.run_id,
            "validation_records": len(validations), "inclusive_months": 60}
    verify_demo_view(view)
    view_file.write_text(json.dumps(view, indent=2), encoding="utf-8")
    pointer.write_text(json.dumps(view, indent=2), encoding="utf-8")
    progress(f"Prepared one validated 60-month view: {view_file}", flush=True)
    return view


def verify_demo_view(view):
    """Read-only completion/identity check, shared by preparation and the UI."""
    if view.get("profile") != "public_demo_fixed_five_years_v1" or view.get("interval_id") != INTERVAL_ID:
        raise ValueError("Not the fixed public five-year profile")
    with duckdb.connect(str(view["database"]), read_only=True) as db:
        build = db.execute("SELECT status,config_json,expected_intervals,snapshot_id,source_database FROM control.month_builds WHERE build_id=?", [view["build_id"]]).fetchone()
        rows = db.execute("SELECT interval_id,status,validation_json,definition_json FROM control.month_intervals WHERE build_id=?", [view["build_id"]]).fetchall()
    if not build or build[0] != "COMPLETED" or build[2] != 1 or build[3] != view["snapshot_id"] or build[4] != view["source_database"]:
        raise ValueError("The fixed public view is incomplete or has conflicting provenance")
    settings = json.loads(build[1])
    if (settings["portfolio_id"], settings["benchmark_id"], settings["inception_date"], settings["as_of_date"]) != (ACTIVE_ID, BENCHMARK_ID, "2021-09-30", "2026-09-30"):
        raise ValueError("The demo viewer cannot open the private research profile")
    if len(rows) != 1 or rows[0][0] != INTERVAL_ID or rows[0][1] != "COMPLETED":
        raise ValueError("Exactly one completed 60-month interval is required")
    checks, definition = json.loads(rows[0][2]), json.loads(rows[0][3])
    if definition["inclusive_months"] != 60 or not checks or any(r["status"] != "PASSED" for r in checks):
        raise ValueError("Saved five-year validations did not pass")
    window = MonthWindow(pd.Period("2021-10", "M"), pd.Period("2026-09", "M"))
    returns = read_month_result(view["database"], view["build_id"], window, "performance", "performance_window")
    if len(returns) != 2:
        raise ValueError("The demo must contain both portfolio and benchmark results")
    return settings, definition, checks
