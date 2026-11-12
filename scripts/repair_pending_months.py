"""Bounded, versioned repair; defaults to a read-only plan."""
import argparse
import json
from pathlib import Path

from portfolio_analytics.database.pending_revision import revise_pending_build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default="local/analytics_outputs/month_windows.duckdb")
    parser.add_argument("--parent", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--audit", default="local/month_window_pending_repair_validation.json")
    args = parser.parse_args()
    report = revise_pending_build(args.database, args.parent, dry_run=not args.apply)
    if args.apply:
        path = Path(args.audit)
        text = json.dumps(report, indent=2, sort_keys=True) + "\n"
        path.write_text(text, encoding="utf-8")
        if json.loads(path.read_text(encoding="utf-8")) != report:
            raise ValueError("Audit readback failed.")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
