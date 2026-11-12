"""Run installed analytics on the public snapshot; no private defaults or notebooks."""
import argparse
import json
from pathlib import Path

from portfolio_analytics.config import AnalysisSettings
from portfolio_analytics.data.snapshots import load_standardized_inputs
from portfolio_analytics.performance.api import run as performance
from portfolio_analytics.demo.cli import run_demo
from portfolio_analytics.demo.build import ACTIVE_ID, BENCHMARK_ID


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, default=Path("demo_workspace"))
    args = parser.parse_args()
    root = args.workdir / "snapshots"
    snapshot = json.loads((root / "latest.json").read_text(encoding="utf-8"))["snapshot_id"]
    frames = load_standardized_inputs(root, snapshot_id=snapshot, verify=True)
    settings = AnalysisSettings(ACTIVE_ID, BENCHMARK_ID, "2026-09-30", "2021-09-30",
                                 reporting_years=(1, 3, 5), performance_windows=("SI", "1Y", "3Y", "5Y"))
    result = performance(portfolio_returns=frames["portfolio_returns"],
                         portfolio_holdings=frames["portfolio_holdings"],
                         factor_returns=frames["factor_returns"], settings=settings)
    print(result["performance_window"][["portfolio_id", "window_id", "period_return"]].to_string(index=False))
    # The same installed package runs all domains with explicit demo identities.
    run, _ = run_demo(args.workdir, snapshot_id=snapshot)
    return 0 if run.status == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
