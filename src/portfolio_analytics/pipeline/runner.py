"""Load once and run any selected domain without inter-domain dependencies."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
from importlib import import_module
import json
from pathlib import Path
import subprocess
import time
from uuid import uuid4

import pandas as pd

from portfolio_analytics.config import AnalysisSettings
from portfolio_analytics.data.snapshots import load_standardized_inputs

DOMAIN_INPUTS = {
    "performance": ("portfolio_returns", "portfolio_holdings", "factor_returns"),
    "attribution": ("portfolio_returns", "portfolio_holdings"),
    "risk": ("portfolio_returns", "portfolio_holdings", "security_returns"),
    "exposures": ("portfolio_holdings",),
    "factors": ("portfolio_returns", "factor_returns"),
    "turnover": ("portfolio_holdings", "rebalance_targets"),
}


@dataclass
class AnalysisRun:
    run_id: str
    snapshot_id: str
    config_json: str
    created_at: str
    code_version: str
    source_fingerprint: str
    snapshot_manifest_sha256: str
    outputs: dict[str, dict[str, pd.DataFrame]] = field(default_factory=dict)
    statuses: list[dict] = field(default_factory=list)

    @property
    def status(self):
        failed = sum(s["status"] == "FAILED" for s in self.statuses)
        return "FAILED" if failed == len(self.statuses) else "PARTIAL" if failed else "COMPLETED"


def code_provenance():
    package = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for file in sorted(package.rglob("*.py")):
        digest.update(file.relative_to(package).as_posix().encode())
        digest.update(file.read_bytes())
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=package,
            capture_output=True, text=True, check=True, timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=package,
            capture_output=True, text=True, check=True, timeout=5).stdout.strip()
        revision += "+dirty" if dirty else ""
    except (OSError, subprocess.SubprocessError):
        revision = "unavailable"
    return revision, digest.hexdigest()


def run_analysis(*, snapshot_root: str | Path, snapshot_id: str,
                 settings: AnalysisSettings, domains=("performance",), on_progress=None) -> AnalysisRun:
    """Verify the pinned snapshot and run selected domains, capturing failures.

    Per-domain failures do not erase successful results. Shared input failures
    raise before execution. Results remain private and are not auto-published.
    """
    if isinstance(domains, str) or not domains or len(set(domains)) != len(domains):
        raise ValueError("domains must be a nonempty sequence of unique domain names.")
    unknown = set(domains) - DOMAIN_INPUTS.keys()
    if unknown:
        raise ValueError(f"Unknown domains: {sorted(unknown)}")
    requirements = {d: DOMAIN_INPUTS[d] for d in domains}
    if "factors" in domains and settings.include_security_factor_history:
        requirements["factors"] += ("security_returns", "portfolio_holdings")
    required = sorted({key for names in requirements.values() for key in names} | {"portfolio_metadata"})
    loaded = load_standardized_inputs(snapshot_root, required, snapshot_id=snapshot_id, verify=True)
    metadata = loaded["portfolio_metadata"].set_index("portfolio_id")
    if not set(settings.portfolio_ids).issubset(metadata.index):
        raise ValueError("Selected portfolio identities are absent from snapshot metadata.")
    if "benchmark_portfolio_id" in metadata:
        linked = metadata.loc[settings.portfolio_id, "benchmark_portfolio_id"]
        if pd.notna(linked) and str(linked) != settings.benchmark_id:
            raise ValueError("Selected benchmark conflicts with saved portfolio lineage.")
    revision, fingerprint = code_provenance()
    manifest = Path(snapshot_root) / snapshot_id / "manifest.json"
    run = AnalysisRun(uuid4().hex, snapshot_id, settings.to_json(),
        datetime.now(timezone.utc).isoformat(), revision, fingerprint,
        hashlib.sha256(manifest.read_bytes()).hexdigest())
    for domain in domains:
        if on_progress:
            on_progress(domain, "RUNNING")
        start = time.perf_counter()
        try:
            module = import_module(f"portfolio_analytics.{domain}.api")
            result = module.run(**{name: loaded[name] for name in requirements[domain]}, settings=settings)
            if not result or any(not isinstance(frame, pd.DataFrame) or not len(frame.columns) for frame in result.values()):
                raise ValueError("A domain must return named DataFrames with explicit columns.")
            run.outputs[domain] = result
            status = dict(domain=domain, status="COMPLETED", message="", outputs=len(result))
        except Exception as exc:
            status = dict(domain=domain, status="FAILED", message=f"{type(exc).__name__}: {exc}", outputs=0)
        status["elapsed_seconds"] = time.perf_counter() - start
        run.statuses.append(status)
        if on_progress:
            on_progress(domain, status["status"])
    return run


def run_summary(run):
    return json.dumps(dict(run_id=run.run_id, status=run.status,
        snapshot_id=run.snapshot_id, domains=run.statuses), indent=2)
