"""Offline, resumable precomputation of every approved complete-month interval."""

from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import os
from pathlib import Path
import time

import duckdb
import numpy as np
import pandas as pd

from portfolio_analytics.common.month_windows import enumerate_month_windows
from portfolio_analytics.database.month_results import MonthResultWriter
from portfolio_analytics.pipeline.month_calculations import (
    MonthCalculationContext, SOURCE_RUN_ID, complete_month_cagr_policy,
)
from portfolio_analytics.pipeline.runner import code_provenance, DOMAIN_INPUTS

_CONTEXT = None


def initialize_worker(arguments):
    global _CONTEXT
    _CONTEXT = MonthCalculationContext(**arguments)


def validate_month_cagr(w, outputs, portfolio_ids):
    """Require eligible CAGR values, their exact formula and pair reconciliation."""
    policy = complete_month_cagr_policy(w)
    absolute = outputs["performance_window"]
    relative = outputs["performance_relative_window"]
    ids = tuple(portfolio_ids)
    if len(ids) != 2 or len(set(ids)) != 2 or len(absolute) != 2 or set(absolute.portfolio_id) != set(ids):
        raise ValueError("CAGR requires exactly one result for each portfolio owner.")
    if (len(relative) != 1 or relative.iloc[0].portfolio_id != ids[0]
            or relative.iloc[0].benchmark_portfolio_id != ids[1]):
        raise ValueError("CAGR requires exactly one correctly oriented portfolio pair.")
    days = (pd.Timestamp(w["last_return_date"]) - pd.Timestamp(w["opening_valuation_date"])).days
    for frame in (absolute, relative):
        for column, value in policy.items():
            if column not in frame or not frame[column].eq(value).fillna(False).all():
                raise ValueError(f"CAGR eligibility provenance mismatch: {column}")
        if (not frame.calendar_days.eq(days).fillna(False).all()
                or not frame.window_id.eq(w["interval_id"]).fillna(False).all()):
            raise ValueError("CAGR interval or actual-day duration mismatch.")
        if not frame.annualization_day_basis.eq(365.25).fillna(False).all():
            raise ValueError("CAGR must retain the existing 365.25-day annualization basis.")
        for row in frame.itertuples(index=False):
            status = str(row.calculation_status)
            if status == "CALCULATED":
                if row.cagr_status != "CALCULATED":
                    raise ValueError("Eligible complete-month CAGR was not calculated.")
            elif not ((status.startswith("UNAVAILABLE_") or status == "WINDOW_UNAVAILABLE")
                      and row.cagr_status == status):
                raise ValueError("CAGR has no calculated result or genuine unavailability reason.")
    exponent = 365.25 / days
    for row in absolute.itertuples(index=False):
        if row.calculation_status == "CALCULATED":
            if not np.isfinite([row.period_return, row.annualized_return_cagr]).all():
                raise ValueError("Calculated CAGR and period return must be finite.")
            np.testing.assert_allclose(row.annualized_return_cagr,
                (1 + row.period_return) ** exponent - 1, atol=1e-12, rtol=1e-10,
                err_msg="CAGR formula mismatch")
    row = relative.iloc[0]
    if row.calculation_status == "CALCULATED":
        p, b = 1 + row.portfolio_period_return, 1 + row.benchmark_period_return
        pc, bc = p ** exponent - 1, b ** exponent - 1
        observed = [row.portfolio_annualized_return_cagr, row.benchmark_annualized_return_cagr,
                    row.annualized_cagr_spread, row.annualized_geometric_relative_return]
        if not np.isfinite(observed).all():
            raise ValueError("Calculated relative CAGR metrics must be finite.")
        np.testing.assert_allclose(observed, [pc, bc, pc - bc, (p / b) ** exponent - 1],
            atol=1e-12, rtol=1e-10, err_msg="Relative CAGR formula mismatch")
        for owner, value, period in ((ids[0], pc, row.portfolio_period_return),
                                     (ids[1], bc, row.benchmark_period_return)):
            match = absolute.loc[absolute.portfolio_id.eq(owner)].iloc[0]
            if match.calculation_status == "CALCULATED":
                np.testing.assert_allclose([match.annualized_return_cagr, match.period_return],
                    [value, period], atol=1e-12, rtol=1e-10, err_msg="Absolute/relative CAGR mismatch")


def validate_month_outputs(context, w, outputs):
    """Retain kernel checks and add interval, compounding and output audits."""
    if set(outputs) != set(DOMAIN_INPUTS):
        raise ValueError("Every interval requires all six domains.")
    report = []
    for domain, tables in outputs.items():
        if not tables:
            raise ValueError(f"Empty domain: {domain}")
        errors, statuses = {}, {}
        for name, frame in tables.items():
            if not isinstance(frame, pd.DataFrame) or not len(frame.columns) or not frame.columns.is_unique:
                raise ValueError(f"Invalid output structure: {domain}/{name}")
            for column in frame:
                series = frame[column]
                if pd.api.types.is_numeric_dtype(series):
                    values = series.to_numpy(dtype=float, na_value=np.nan)
                    if np.isinf(values).any():
                        raise ValueError(f"Infinite metric: {domain}/{name}/{column}")
                    if (column.endswith("_error") or column.startswith("linking_residual_")) and (
                        not column.endswith("standard_error") and not column.endswith("tracking_error")):
                        finite = values[np.isfinite(values)]
                        error = float(np.max(np.abs(finite))) if len(finite) else None
                        errors[f"{name}.{column}"] = error
                        if error is not None and error > 1e-8:
                            raise ValueError(f"Reconciliation failed: {domain}/{name}/{column}: {error}")
                if column.endswith("status"):
                    counts = series.fillna("NULL").astype(str).value_counts().to_dict()
                    if any(str(value).startswith(("FAIL", "INVALID")) for value in counts):
                        raise ValueError(f"Failed status: {domain}/{name}/{column}: {counts}")
                    statuses[f"{name}.{column}"] = counts
        report.append(dict(domain=domain, status="PASSED", tables=len(tables),
            rows=sum(len(frame) for frame in tables.values()), reconciliation_errors=errors, statuses=statuses))
    performance = outputs["performance"]["performance_window"]
    source = context.returns.loc[context.returns.date.between(w["first_return_date"], w["last_return_date"])]
    for pid in context.settings.portfolio_ids:
        sample = source.loc[source.portfolio_id.eq(pid), "return_total"].to_numpy()
        if len(sample) != w["observation_count"]:
            raise ValueError("The selected period dropped or added a return session.")
        actual = performance.loc[performance.portfolio_id.eq(pid)].iloc[0]
        # Kernel output field is intentionally explicit: catch incompatible
        # contract changes rather than guessing a similarly named metric.
        expected = np.prod(1 + sample) - 1
        np.testing.assert_allclose(actual["period_return"], expected, atol=1e-10, rtol=1e-10)
    # Full MonthWindow definitions carry this provenance; standalone diagnostic
    # samples without a monthly definition retain the existing validation path.
    if "inclusive_months" in w:
        validate_month_cagr(w, outputs["performance"], context.settings.portfolio_ids)
        next(row for row in report if row["domain"] == "performance").update(
            complete_month_cagr_check="PASSED", cagr_eligibility_basis="AT_LEAST_12_COMPLETE_MONTHS")
    totals = outputs["attribution"]["brinson_total_period"]
    if len(totals) != 2 or set(totals.linking_method) != {"carino", "menchero"}:
        raise ValueError("Both linked attribution methods must be present.")
    for frame in (outputs["exposures"]["portfolio_window_summary"], outputs["exposures"]["active_window_summary"]):
        if not frame.observations.eq(w["observation_count"]).all():
            raise ValueError("Exposure summaries do not cover every in-month session.")
    events = outputs["turnover"]["rebalance_turnover_events"]
    pending = events.event_status.eq("PENDING_AFTER_CUTOFF")
    forbidden = ["one_way_turnover", "total_purchases", "total_sales", "gross_two_way_turnover",
                 "implemented_weight_sum", "max_abs_implementation_weight_error"]
    if events.loc[pending, forbidden].notna().any().any():
        raise ValueError("Pending rebalances contain future implementation values.")
    report.append(dict(domain="boundaries", status="PASSED", opening_valuation_date=str(w["opening_valuation_date"]),
        first_return_date=str(w["first_return_date"]), last_return_date=str(w["last_return_date"]),
        observations=w["observation_count"]))
    from .month_turnover import validate_pending_events
    report.append(validate_pending_events(context.shared["turnover", "rebalance_turnover_events"], events, w))
    return report


def calculate_chunk(windows):
    results = []
    for w in windows:
        outputs, timings = _CONTEXT.calculate(w)
        validation = validate_month_outputs(_CONTEXT, w, outputs)
        results.append((w, outputs, validation, timings))
    return results


def precompute_month_windows(*, snapshot_root, snapshot_id, settings, source_database,
                            database, source_run_id=SOURCE_RUN_ID, workers=1,
                            resume=False, limit=None, progress=print):
    """Build compact period summaries and genuinely interval-dependent paths.

    A limited build is a separate benchmark universe. It is never confused with
    the complete 5,995-period build, which has its own deterministic identity.
    """
    if type(workers) is not int or not 1 <= workers <= 12:
        raise ValueError("workers must be an integer between 1 and 12.")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("limit must be a positive integer.")
    if Path(database).resolve() == Path(source_database).resolve():
        raise ValueError("Month-window storage must be separate from the immutable source-results database.")
    arguments = dict(snapshot_root=Path(snapshot_root).resolve(), snapshot_id=snapshot_id,
        settings=settings, source_database=Path(source_database).resolve(), source_run_id=source_run_id)
    context = MonthCalculationContext(**arguments)
    universe = enumerate_month_windows(context.calendar, settings.inception_date)
    if len(universe) != 5995 or universe.interval_id.duplicated().any():
        raise ValueError("The approved research universe must contain exactly 5,995 unique intervals.")
    if limit:
        # Include the shortest, a middle interval and full history in benchmarks.
        picks = np.linspace(0, len(universe) - 1, min(limit, len(universe)), dtype=int)
        picks[-1] = int(universe.inclusive_months.to_numpy().argmax())
        picks = list(dict.fromkeys(picks.tolist()))
        for index in range(len(universe)):
            if len(picks) == min(limit, len(universe)):
                break
            if index not in picks:
                picks.append(index)
        universe = universe.iloc[picks].reset_index(drop=True)
    windows = universe.to_dict("records")
    _, code = code_provenance()
    metadata = dict(snapshot_id=snapshot_id, config_json=context.settings.to_json(),
        source_database=str(Path(source_database).resolve()), source_run_id=source_run_id,
        source_fingerprint=context.source_fingerprint, manifest_sha256=context.manifest_sha256,
        code_fingerprint=code, input_validation=context.input_validation)
    identity = {k: v for k, v in metadata.items() if k != "input_validation"}
    identity["intervals"] = [w["interval_id"] for w in windows]
    build_id = "months_" + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    writer = MonthResultWriter(database)
    started = time.perf_counter()
    try:
        writer.begin_build(build_id, metadata, windows, resume=resume)
        writer.register_shared(build_id, context.catalog)
        completed = writer.completed(build_id)
        pending = [w for w in windows if w["interval_id"] not in completed]
        progress(f"Build {build_id}: {len(completed)}/{len(windows)} intervals already saved.")
        for _, group in universe.groupby("end_month", sort=True):
            w = group.iloc[0].to_dict()
            writer.save_closing(build_id, w["end_month"], context.closing_risk(w))
        # One native math thread per worker avoids oversubscribing BLAS libraries.
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[name] = "1"
        chunks = [pending[i:i + 25] for i in range(0, len(pending), 25)]
        if workers == 1:
            global _CONTEXT
            _CONTEXT = context
            for chunk in chunks:
                writer.save_batch(build_id, calculate_chunk(chunk))
                completed.update(w["interval_id"] for w in chunk)
                progress(f"Saved {len(completed)}/{len(windows)} intervals; {time.perf_counter()-started:.1f}s elapsed.")
        elif chunks:
            with ProcessPoolExecutor(max_workers=workers, initializer=initialize_worker, initargs=(arguments,)) as pool:
                queue = iter(chunks)
                # Bound queued results: completed futures otherwise retain every
                # DataFrame and can exhaust memory during a long population.
                futures = {pool.submit(calculate_chunk, chunk) for chunk in [next(queue, None) for _ in range(workers)] if chunk}
                buffered = []
                while futures:
                    done, futures = wait(futures, return_when=FIRST_COMPLETED)
                    for future in done:
                        results = future.result()
                        buffered.extend(results)
                        if len(buffered) >= 100:
                            writer.save_batch(build_id, buffered)
                            completed.update(w["interval_id"] for w, *_ in buffered)
                            buffered.clear()
                            progress(f"Saved {len(completed)}/{len(windows)} intervals; {time.perf_counter()-started:.1f}s elapsed.")
                        chunk = next(queue, None)
                        if chunk:
                            futures.add(pool.submit(calculate_chunk, chunk))
                if buffered:
                    writer.save_batch(build_id, buffered)
                    completed.update(w["interval_id"] for w, *_ in buffered)
                    progress(f"Saved {len(completed)}/{len(windows)} intervals; {time.perf_counter()-started:.1f}s elapsed.")
        # Verify every stored period table's keys and total population, not just
        # the last checkpoint, before making this build reader-visible.
        catalog = writer.db.execute("SELECT * FROM control.month_outputs WHERE build_id=? AND scope='PERIOD'", [build_id]).df()
        for row in catalog.itertuples():
            table = 'month_results."' + row.table_name + '"'
            duplicates = writer.db.execute(f"SELECT count(*) FROM (SELECT interval_id,result_row FROM {table} GROUP BY ALL HAVING count(*)>1)").fetchone()[0]
            if duplicates:
                raise ValueError(f"Duplicate persisted output keys: {row.output_name}")
        actual_rows = sum(writer.db.execute('SELECT count(*) FROM month_results."' + r.table_name + '"').fetchone()[0]
                          for r in catalog.itertuples())
        expected_rows = writer.db.execute("SELECT sum(rows_saved) FROM control.month_intervals WHERE build_id=?", [build_id]).fetchone()[0]
        if actual_rows != expected_rows:
            raise ValueError("Persisted result totals differ from checkpoint totals.")
        writer.finish(build_id)
        summary = dict(build_id=build_id, snapshot_id=snapshot_id, intervals=len(windows),
            period_tables=len(catalog), period_rows=int(actual_rows), elapsed_seconds=time.perf_counter()-started,
            database=str(Path(database).resolve()), universe="FULL" if len(windows) == 5995 else "BENCHMARK")
        progress(json.dumps(summary, indent=2))
    except Exception as exc:
        writer.fail(build_id, exc)
        raise
    finally:
        writer.close()
    # Independent connection verifies the committed completion state.
    with duckdb.connect(str(database), read_only=True) as db:
        if db.execute("SELECT status FROM control.month_builds WHERE build_id=?", [build_id]).fetchone()[0] != "COMPLETED":
            raise ValueError("Independent completion readback failed.")
    summary["database_bytes"] = Path(database).stat().st_size
    return summary
