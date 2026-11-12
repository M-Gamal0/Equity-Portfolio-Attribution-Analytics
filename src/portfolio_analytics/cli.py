"""Command-line applications for individual or selected analytical domains."""

import argparse
from dataclasses import replace
from pathlib import Path
import sys

from .config import AnalysisSettings, RESEARCH_SNAPSHOT_ID, research_settings
from .pipeline.runner import DOMAIN_INPUTS, run_analysis, run_summary
from .database.results import list_runs, save_run


def project_root():
    for candidate in (Path.cwd(), *Path.cwd().parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return Path.cwd()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run independent equity analytics offline.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="List analytical domains and their required inputs.")
    run = sub.add_parser("run", help="Run one domain, a subset, or all.")
    run.add_argument("domains", nargs="+", choices=[*DOMAIN_INPUTS, "all"])
    run.add_argument("--research", action="store_true", help="Use the explicit approved research profile.")
    run.add_argument("--snapshot-root", type=Path, default=project_root()/"local/standardized_inputs")
    run.add_argument("--snapshot-id")
    run.add_argument("--portfolio-id")
    run.add_argument("--benchmark-id")
    run.add_argument("--as-of")
    run.add_argument("--inception-date")
    run.add_argument("--security-factor-history", action="store_true", help="Include expensive offline rolling security regressions.")
    run.add_argument("--database", type=Path, default=project_root()/"local/analytics_outputs/analytics.duckdb")
    run.add_argument("--no-save", action="store_true", help="Calculate only; do not persist outputs.")
    runs = sub.add_parser("runs", help="List saved private analysis runs.")
    runs.add_argument("--database", type=Path, default=project_root()/"local/analytics_outputs/analytics.duckdb")
    months = sub.add_parser("precompute-months", help="Precompute every >=12-complete-month interval offline.")
    months.add_argument("--research", action="store_true", required=True)
    months.add_argument("--snapshot-root", type=Path, default=project_root()/"local/standardized_inputs")
    months.add_argument("--snapshot-id", default=RESEARCH_SNAPSHOT_ID)
    months.add_argument("--source-database", type=Path, default=project_root()/"local/analytics_outputs/analytics.duckdb")
    months.add_argument("--source-run-id", default="0ddde9043b9b4189b235102471de3d8f")
    months.add_argument("--database", type=Path, default=project_root()/"local/analytics_outputs/month_windows.duckdb")
    months.add_argument("--workers", type=int, default=1)
    months.add_argument("--resume", action="store_true")
    months.add_argument("--limit", type=int, help="Build a separate benchmark subset, not the complete universe.")
    selected = sub.add_parser("month-result", help="Read one precomputed interval without recalculating analytics.")
    selected.add_argument("domain", choices=list(DOMAIN_INPUTS))
    selected.add_argument("output_name")
    selected.add_argument("--build-id", required=True)
    selected.add_argument("--start-month", required=True)
    selected.add_argument("--start-year", type=int, required=True)
    selected.add_argument("--end-month", required=True)
    selected.add_argument("--end-year", type=int, required=True)
    selected.add_argument("--database", type=Path, default=project_root()/"local/analytics_outputs/month_windows.duckdb")
    args = parser.parse_args(argv)
    if args.command == "list":
        for domain, datasets in DOMAIN_INPUTS.items():
            print(f"{domain:14s} {', '.join(datasets)}")
        return 0
    if args.command == "runs":
        frame = list_runs(args.database)
        print("No saved runs." if frame.empty else frame[["run_id", "created_at", "status"]].to_string(index=False))
        return 0
    try:
        if args.command == "precompute-months":
            from .pipeline.month_runner import precompute_month_windows
            precompute_month_windows(snapshot_root=args.snapshot_root, snapshot_id=args.snapshot_id,
                settings=research_settings(), source_database=args.source_database, source_run_id=args.source_run_id,
                database=args.database, workers=args.workers, resume=args.resume, limit=args.limit,
                progress=lambda message: print(message, flush=True))
            return 0
        if args.command == "month-result":
            from .common.month_windows import MonthWindow
            from .database.month_results import read_month_result
            window = MonthWindow.from_names(args.start_month, args.start_year, args.end_month, args.end_year)
            frame = read_month_result(args.database, args.build_id, window, args.domain, args.output_name)
            print(frame.to_string(index=False))
            return 0
        if args.research:
            settings = research_settings()
            overrides = {k: v for k, v in dict(portfolio_id=args.portfolio_id, benchmark_id=args.benchmark_id,
                as_of_date=args.as_of, inception_date=args.inception_date).items() if v is not None}
            settings = replace(settings, **overrides, include_security_factor_history=args.security_factor_history)
            snapshot_id = args.snapshot_id or RESEARCH_SNAPSHOT_ID
        else:
            if not all((args.portfolio_id, args.benchmark_id, args.as_of, args.snapshot_id)):
                parser.error("Supply --research or explicit --portfolio-id, --benchmark-id, --as-of and --snapshot-id.")
            snapshot_id = args.snapshot_id
            settings = AnalysisSettings(args.portfolio_id, args.benchmark_id, args.as_of, args.inception_date,
                                        include_security_factor_history=args.security_factor_history)
        domains = tuple(DOMAIN_INPUTS) if args.domains == ["all"] else tuple(args.domains)
        result = run_analysis(snapshot_root=args.snapshot_root, snapshot_id=snapshot_id,
            settings=settings, domains=domains, on_progress=lambda d, s: print(f"{d}: {s}", flush=True))
        if not args.no_save:
            print(f"Saved: {save_run(result, args.database)}")
        print(run_summary(result))
        return 0 if result.status == "COMPLETED" else 1
    except (OSError, ValueError, KeyError) as exc:
        print(f"Unable to run analysis: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
