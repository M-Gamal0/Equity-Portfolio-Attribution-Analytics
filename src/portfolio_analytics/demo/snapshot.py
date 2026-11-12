"""Versioned public-demo snapshots compatible with the unchanged package loader."""
from dataclasses import asdict
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import tempfile

from portfolio_analytics.data.snapshots import load_standardized_inputs
from .build import validate_demo
from .sources import utc_now

PRIMARY_KEYS = {
    "portfolio_returns": ["portfolio_id", "date"],
    "portfolio_holdings": ["portfolio_id", "date", "security_id"],
    "security_returns": ["security_id", "date"],
    "rebalance_targets": ["portfolio_id", "selection_date", "security_id"],
    "factor_returns": ["date"],
    "portfolio_metadata": ["portfolio_id"],
}


def save_demo_snapshot(root, frames, config, *, sources, selection_audit):
    """Validate first, commit immutable files, verify readback, then move latest."""
    report = validate_demo(frames, config)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".incomplete-demo-", dir=root))
    try:
        datasets = {}
        for name, keys in PRIMARY_KEYS.items():
            frame = frames[name].sort_values(keys, kind="stable").reset_index(drop=True)
            filename = name + ".parquet"
            frame.to_parquet(temporary / filename, index=False)
            datasets[name] = {"file": filename, "sha256": sha256((temporary / filename).read_bytes()).hexdigest(),
                              "rows": len(frame), "columns": list(frame.columns),
                              "dtypes": {c: str(d) for c, d in frame.dtypes.items()}, "primary_key": keys}
        selection_audit.to_json(temporary / "selection_audit.json", orient="records", indent=2)
        audit_hash = sha256((temporary / "selection_audit.json").read_bytes()).hexdigest()
        identity = {"format_version": 1, "profile": "public_yahoo_quarterly_equal_weight_v1",
                    "construction": asdict(config), "datasets": datasets, "sources": sources,
                    "selection_audit_sha256": audit_hash}
        snapshot_id = sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
        manifest = {**identity, "snapshot_id": snapshot_id, "created_at_utc": utc_now(),
                    "validation": report, "data_kind": "real_public_source_derived_demo_not_synthetic",
                    "software_versions": {name: version(name) for name in ("portfolio-analytics", "pandas", "numpy", "pyarrow")},
                    "notes": ["Current constituents/current caps applied retrospectively; separate share classes.",
                              "Fixed baskets; quarterly equal-weight targets; price-only drift between rebalances.",
                              "Index-level dividend reinvestment; not live executed returns.",
                              "Source access is not an independent redistribution licence; fetch locally."]}
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        destination = root / snapshot_id
        if destination.exists():
            existing = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
            if any(existing.get(k) != v for k, v in identity.items()):
                raise ValueError("Existing demo snapshot identity conflicts; never overwrite it")
            if sha256((destination / "selection_audit.json").read_bytes()).hexdigest() != audit_hash:
                raise ValueError("Existing selection audit checksum mismatch")
        else:
            temporary.rename(destination)
        loaded = load_standardized_inputs(root, snapshot_id=snapshot_id, verify=True)
        validate_demo(loaded, config)
        for name in datasets:
            expected = frames[name].sort_values(PRIMARY_KEYS[name], kind="stable").reset_index(drop=True)
            if not loaded[name].equals(expected):
                raise ValueError(f"Exact Parquet readback failed: {name}")
        pointer = root / ".latest-demo.tmp"
        pointer.write_text(json.dumps({"snapshot_id": snapshot_id}, indent=2), encoding="utf-8")
        pointer.replace(root / "latest.json")
        return snapshot_id
    finally:
        # Only the exact temporary directory created above; never a caller path.
        if temporary.exists():
            if temporary.resolve().parent != root.resolve() or not temporary.name.startswith(".incomplete-demo-"):
                raise ValueError("Refusing to remove a directory outside the demo temporary scope")
            shutil.rmtree(temporary)
