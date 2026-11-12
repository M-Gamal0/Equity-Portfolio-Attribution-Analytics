"""Standalone public-demo acquisition/execution; never select the research profile."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pandas as pd

from portfolio_analytics.config import AnalysisSettings
from portfolio_analytics.pipeline.runner import DOMAIN_INPUTS, run_analysis
from .build import BENCHMARK_ID, DemoConfig, construct_inputs
from .snapshot import save_demo_snapshot
from .sources import SourceCache, fetch_factors, fetch_prices, fetch_quote, fetch_universe


def _workspace(value):
    path = Path(value).resolve()
    project = Path(__file__).resolve().parents[3]
    for protected in (project / "app", project / "src", project / "source",
                      project / "local/analytics_outputs", project / "local/standardized_inputs"):
        if path == protected or path.is_relative_to(protected):
            raise ValueError("Use a separate demo workspace, not a research/source/app directory")
    path.mkdir(parents=True, exist_ok=True)
    return path


def cached_history_index(cache, config, fetch):
    """Acquisition depends on dates, not how many of the cached names we hold.

    Preserve and validate the original full-config descriptor in older caches;
    changing 10 to 20 holdings must not require another Yahoo download.
    """
    prefix = "derived:yfinance-price-acquisition:"
    dates = {key: getattr(config, key) for key in ("history_start", "cutoff")}
    source = prefix + json.dumps(dates, sort_keys=True)
    metadata = cache.folder / "history_index.json.source.json"
    if metadata.exists():
        recorded = json.loads(metadata.read_text(encoding="utf-8"))["source"]
        if not recorded.startswith(prefix):
            raise ValueError("Unexpected cached history acquisition source")
        request = json.loads(recorded[len(prefix):])
        if any(request.get(key) != value for key, value in dates.items()):
            raise ValueError("Cached price dates differ; use a new acquisition cache")
        source = recorded
    return json.loads(cache.get("history_index.json", source, fetch))


def acquire_demo(workspace, *, offline=False, refresh=False, progress=print):
    workspace = _workspace(workspace)
    config = DemoConfig()
    pointer = workspace / "cache_latest.json"
    if offline and refresh:
        raise ValueError("--offline and --refresh cannot be combined")
    if pointer.exists() and not refresh:
        name = json.loads(pointer.read_text(encoding="utf-8"))["cache"]
        if Path(name).name != name:
            raise ValueError("Invalid cache pointer")
    elif offline:
        raise FileNotFoundError("No frozen public cache; build online before --offline")
    else:
        name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        pointer.write_text(json.dumps({"cache": name}), encoding="utf-8")
    cache = SourceCache(workspace / "cache" / name, offline=offline)
    universe = fetch_universe(cache)
    progress(f"Current Nasdaq-100 security universe: {len(universe)} separate tickers", flush=True)

    def collect_quotes():
        def one(ticker):
            try:
                return fetch_quote(cache, ticker)
            except Exception as exc:
                return {"ticker": ticker, "marketCap": None, "source_error": f"QUOTE_REQUEST_FAILED:{type(exc).__name__}"}
        records = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            for record in pool.map(one, universe.ticker):
                records.append(record)
                if len(records) % 10 == 0:
                    progress(f"Market-cap/sector responses: {len(records)}/{len(universe)}", flush=True)
        return json.dumps(records, sort_keys=True, allow_nan=False).encode()

    quotes = pd.DataFrame(json.loads(cache.get("quotes_index.json", "derived:yfinance-current-quotes:v1", collect_quotes)))
    quotes = universe.merge(quotes, on="ticker", validate="one_to_one")
    if "source_error" in quotes:
        quotes["source_error"] = quotes.source_error.fillna("")
    calendar = fetch_prices(cache, "SPY", config.history_start, config.cutoff)
    if calendar.empty or calendar.index.max() != pd.Timestamp(config.cutoff):
        raise ValueError("Yahoo reference calendar does not cover September 30, 2026")

    def collect_histories():
        results = []
        for row in quotes.to_dict("records"):
            if row.get("source_error") or not row.get("marketCap") or row.get("currency") != "USD" or not row.get("sector"):
                results.append({"ticker": row["ticker"], "status": "NOT_REQUESTED_INVALID_QUOTE"})
                continue
            try:
                fetch_prices(cache, row["ticker"], config.history_start, config.cutoff)
                results.append({"ticker": row["ticker"], "status": "DOWNLOADED"})
            except Exception as exc:
                results.append({"ticker": row["ticker"], "status": f"PRICE_REQUEST_FAILED:{type(exc).__name__}"})
            if len(results) % 10 == 0:
                progress(f"Price histories assessed: {len(results)}/{len(quotes)}", flush=True)
        return json.dumps(results, sort_keys=True).encode()

    assessed = cached_history_index(cache, config, collect_histories)
    histories = {row["ticker"]: fetch_prices(cache, row["ticker"], config.history_start, config.cutoff)
                 for row in assessed if row["status"] == "DOWNLOADED"}
    factors = fetch_factors(cache, config.history_start, config.cutoff)
    frames, audit = construct_inputs(quotes, histories, calendar, factors, config)
    audit.to_json(workspace / "selection_audit.json", orient="records", indent=2)
    snapshot_id = save_demo_snapshot(workspace / "snapshots", frames, config,
                                     sources=cache.provenance(), selection_audit=audit)
    progress(f"Validated snapshot: {snapshot_id}; selected {config.benchmark_size}/{config.active_size}; exclusions {int((~audit.eligible).sum())}", flush=True)
    return snapshot_id


def run_demo(workspace, *, snapshot_id=None, domains=None, security_factor_history=False):
    workspace = _workspace(workspace)
    root = workspace / "snapshots"
    if snapshot_id is None:
        snapshot_id = json.loads((root / "latest.json").read_text(encoding="utf-8"))["snapshot_id"]
    if len(snapshot_id) != 16 or any(c not in "0123456789abcdef" for c in snapshot_id):
        raise ValueError("Expected a 16-character snapshot ID")
    manifest = json.loads((root / snapshot_id / "manifest.json").read_text(encoding="utf-8"))
    config = DemoConfig(**manifest["construction"])
    # Old immutable ten-stock snapshots remain runnable under their saved identity.
    active_id = f"demo_active{config.active_size}"
    settings = AnalysisSettings(portfolio_id=active_id, benchmark_id=BENCHMARK_ID,
        inception_date=config.inception, as_of_date=config.cutoff, reporting_years=(1, 3, 5),
        performance_windows=("SI", "YTD", "1Y", "3Y", "5Y"),
        include_security_factor_history=security_factor_history)
    run = run_analysis(snapshot_root=root, snapshot_id=snapshot_id, settings=settings,
                       domains=tuple(domains or DOMAIN_INPUTS),
                       on_progress=lambda domain, status: print(f"{domain}: {status}", flush=True))
    output = workspace / "results" / run.run_id
    output.mkdir(parents=True, exist_ok=False)
    for domain, tables in run.outputs.items():
        folder = output / domain
        folder.mkdir()
        for name, frame in tables.items():
            frame.to_parquet(folder / (name + ".parquet"), index=False)
    summary = {"run_id": run.run_id, "snapshot_id": snapshot_id, "status": run.status,
               "settings": json.loads(run.config_json), "domains": run.statuses,
               "outputs": {domain: {name: len(frame) for name, frame in tables.items()}
                           for domain, tables in run.outputs.items()},
               "snapshot_manifest_sha256": run.snapshot_manifest_sha256,
               "source_fingerprint": run.source_fingerprint}
    (output / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"{run.status}: {output / 'run_summary.json'}", flush=True)
    return run, output


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build and run the independent public-data portfolio demo.")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="Fetch/cache public sources and validate six standardized Parquet datasets.")
    build.add_argument("--workdir", default="demo_workspace", type=Path)
    build.add_argument("--offline", action="store_true", help="Replay the frozen source cache without network requests.")
    build.add_argument("--refresh", action="store_true", help="Acquire a new source cache; preserve old caches and snapshots.")
    run = sub.add_parser("run", help="Run existing analytics against the public snapshot, without private files.")
    run.add_argument("--workdir", default="demo_workspace", type=Path)
    run.add_argument("--snapshot-id")
    run.add_argument("--domains", nargs="+", choices=list(DOMAIN_INPUTS))
    run.add_argument("--security-factor-history", action="store_true", help="Also run expensive rolling security models/lagged holdings attribution.")
    view = sub.add_parser("prepare-app", help="Prepare exactly one validated five-year view for the separate Streamlit demo.")
    view.add_argument("--workdir", default="demo_workspace", type=Path)
    view.add_argument("--snapshot-id")
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            acquire_demo(args.workdir, offline=args.offline, refresh=args.refresh)
            return 0
        if args.command == "prepare-app":
            from .view import prepare_demo_view
            prepare_demo_view(args.workdir, snapshot_id=args.snapshot_id)
            return 0
        result, _ = run_demo(args.workdir, snapshot_id=args.snapshot_id,
                             domains=args.domains, security_factor_history=args.security_factor_history)
        return 0 if result.status == "COMPLETED" else 1
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"Demo failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
