"""Load versioned standardized analytics DataFrames."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd


def load_standardized_inputs(
    snapshot_root: str | Path,
    datasets: list[str] | tuple[str, ...] | None = None,
    *,
    snapshot_id: str | None = None,
    verify: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Load selected standardized DataFrames from Parquet.

    Parameters
    ----------
    snapshot_root:
        Directory containing latest.json and snapshots.
    datasets:
        Dataset names to load. None loads all datasets.
    snapshot_id:
        Specific immutable version. None selects latest.
    verify:
        Check SHA-256, row counts, schemas and primary keys.

    Returns
    -------
    dict[str, pd.DataFrame]
        DataFrames keyed by standardized dataset name.
    """

    root = Path(snapshot_root)

    if snapshot_id is None:
        pointer = json.loads(
            (root / "latest.json").read_text(
                encoding="utf-8"
            )
        )
        snapshot_id = pointer["snapshot_id"]

    if not re.fullmatch(r"[0-9a-f]{16}", snapshot_id):
        raise ValueError("Invalid snapshot identifier.")

    folder = root / snapshot_id

    manifest = json.loads(
        (folder / "manifest.json").read_text(
            encoding="utf-8"
        )
    )

    if manifest["snapshot_id"] != snapshot_id:
        raise ValueError("Snapshot manifest mismatch.")

    available = manifest["datasets"]

    if datasets is None:
        datasets = list(available)

    if isinstance(datasets, str):
        raise TypeError(
            "datasets must be a list or tuple, not a string."
        )

    if len(set(datasets)) != len(datasets):
        raise ValueError("Duplicate dataset requests.")

    unknown = set(datasets) - set(available)

    if unknown:
        raise ValueError(
            f"Unknown standardized datasets: {sorted(unknown)}"
        )

    results = {}

    for name in datasets:
        metadata = available[name]

        expected_filename = f"{name}.parquet"

        if metadata["file"] != expected_filename:
            raise ValueError(
                f"Unexpected file mapping: {name}"
            )

        file_path = folder / expected_filename

        if verify:
            digest = hashlib.sha256()

            with file_path.open("rb") as handle:
                for chunk in iter(
                    lambda: handle.read(1024 * 1024),
                    b"",
                ):
                    digest.update(chunk)

            if digest.hexdigest() != metadata["sha256"]:
                raise ValueError(
                    f"SHA-256 verification failed: {name}"
                )

        frame = pd.read_parquet(
            file_path,
            engine="pyarrow",
        )

        if verify:
            if len(frame) != metadata["rows"]:
                raise ValueError(
                    f"Row count mismatch: {name}"
                )

            if list(frame.columns) != metadata["columns"]:
                raise ValueError(
                    f"Column mismatch: {name}"
                )

            observed_dtypes = {
                column: str(dtype)
                for column, dtype in frame.dtypes.items()
            }

            if observed_dtypes != metadata["dtypes"]:
                raise ValueError(
                    f"Data type mismatch: {name}"
                )

            key = metadata["primary_key"]

            if (
                frame[key].isna().any().any()
                or frame.duplicated(key).any()
            ):
                raise ValueError(
                    f"Invalid primary keys: {name}"
                )

        results[name] = frame

    return results
