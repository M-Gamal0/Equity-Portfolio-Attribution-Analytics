"""Export a completed month-window build to remote-friendly bundles, then verify them.

Reads the existing DuckDB results only; nothing is recalculated or modified. The output
folder must be new or empty. Verification compares every result table for a spread of
periods (including all 12-month periods) with the original DuckDB readers.

Hosting services throttle uploads of thousands of small files, so the verified export can
then be packed into a few large files. Packing copies every file byte for byte; readers
fetch one file per request with an HTTP range request.

Examples:
    python scripts/export_month_bundles.py --output local/month_bundles
    python scripts/export_month_bundles.py --pack-from local/month_bundles --output local/month_packs
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from portfolio_analytics.database.month_bundles import (  # noqa: E402
    BundleReader, export_month_bundles, pack_month_bundles, verify_bundles, verify_packed_files,
)


def verification_sample(interval_ids, spread, *, all_twelve_month=True):
    """Every 12-month period (optional) plus an even spread across all lengths and dates."""
    ids = sorted(interval_ids)
    twelve = [i for i in ids if _months(i) == 12] if all_twelve_month else []
    step = max(1, len(ids) // spread)
    chosen = dict.fromkeys(twelve + ids[::step] + [ids[0], ids[-1]])
    return list(chosen)


def _months(interval_id):
    start, end = interval_id.split("_")
    (sy, sm), (ey, em) = (map(int, start.split("-")), map(int, end.split("-")))
    return (ey - sy) * 12 + em - sm + 1


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--database", default=str(ROOT / "local/analytics_outputs/month_windows.duckdb"))
    parser.add_argument("--output", required=True, help="New or empty folder for the bundles.")
    parser.add_argument("--spread", type=int, default=60, help="Extra periods spread across all lengths.")
    parser.add_argument("--verify-only", action="store_true", help="Skip export; verify an existing folder.")
    parser.add_argument("--pack-from", help="Pack an existing verified export instead of exporting.")
    parser.add_argument("--pack-mb", type=int, default=500, help="Approximate size of each pack in MB.")
    args = parser.parse_args()

    packing = bool(args.pack_from)
    if packing and not args.verify_only:
        pack_month_bundles(args.pack_from, args.output, pack_bytes=args.pack_mb * 1_000_000)
    elif not args.verify_only:
        manifest = export_month_bundles(args.database, args.output)
        print(f"Exported build {manifest['build_id']}: {manifest['interval_count']} periods, "
              f"{len(manifest['files'])} files.")
    reader = BundleReader(args.output)
    if reader.manifest().get("packed"):
        verify_packed_files(args.output)
    _, intervals = reader.metadata(reader.manifest()["build_id"])
    # Packed copies are byte-identical, so a smaller table comparison suffices after packing.
    sample = verification_sample(intervals.interval_id.tolist(), args.spread, all_twelve_month=not packing)
    print(f"Verifying {len(sample)} periods against the DuckDB readers...")
    mismatches = verify_bundles(args.database, args.output, sample)
    if mismatches:
        for item in mismatches[:20]:
            print("MISMATCH", *item)
        raise SystemExit(f"Verification failed: {len(mismatches)} table mismatches.")
    print("Verification passed: every compared table matches exactly.")


if __name__ == "__main__":
    main()
