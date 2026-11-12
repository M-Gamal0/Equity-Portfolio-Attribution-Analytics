import json

import duckdb
import pandas as pd
import pytest

from portfolio_analytics.cli import main
from portfolio_analytics.config import research_settings
from portfolio_analytics.database.results import (
    save_run,
    list_runs,
    list_outputs,
    read_output,
)
from portfolio_analytics.pipeline.runner import AnalysisRun


def sample_run(run_id="test_run"):
    return AnalysisRun(
        run_id,
        "frozen_input",
        research_settings().to_json(),
        "2026-10-26T00:00:00+00:00",
        "test_revision",
        "test_fingerprint",
        "test_manifest",
        outputs={
            "performance": {
                "performance_window": pd.DataFrame(
                    dict(
                        portfolio_id=["P", "B"],
                        value=[0.12, float("nan")],
                        date=pd.to_datetime(["2020-01-01"] * 2),
                    )
                )
            }
        },
        statuses=[
            dict(domain="performance", status="COMPLETED", message="", outputs=1)
        ],
    )


def test_storage_roundtrip_and_run_immutability(tmp_path):
    path = (
        tmp_path / "analytics.duckdb"
    )  # regression: catalog and schema names may coincide
    run = sample_run()
    save_run(run, path)
    assert not list_runs(path).iloc[0]["publication_approved"]
    assert list_outputs(path, run.run_id).iloc[0]["row_count"] == 2
    pd.testing.assert_frame_equal(
        read_output(path, run.run_id, "performance", "performance_window"),
        run.outputs["performance"]["performance_window"],
        check_dtype=False,
    )
    with pytest.raises(duckdb.ConstraintException):
        save_run(run, path)
    assert len(list_runs(path)) == 1
    assert (
        json.loads(list_runs(path).iloc[0]["config_json"])["as_of_date"] == "2026-09-30"
    )


def test_read_missing_database_does_not_create_file(tmp_path):
    path = tmp_path / "absent.duckdb"
    assert list_runs(path).empty
    assert not path.exists()


def test_invalid_values_do_not_create_run(tmp_path):
    run = sample_run()
    run.outputs["performance"]["performance_window"].loc[0, "value"] = float("inf")
    with pytest.raises(ValueError, match="Infinite"):
        save_run(run, tmp_path / "analytics.duckdb")
    assert list_runs(tmp_path / "analytics.duckdb").empty


def test_cli_domain_listing(capsys):
    assert main(["list"]) == 0
    assert "turnover" in capsys.readouterr().out


def test_explicit_profile_required():
    with pytest.raises(SystemExit):
        main(["run", "performance"])


@pytest.mark.parametrize(
    "frequency,labels",
    [
        ("Q-DEC", ["2026Q1", None, "2026Q3"]),
        ("Q-MAR", ["2026Q1", None, "2026Q3"]),
        ("M", ["2026-01", None, "2026-03"]),
    ],
)
def test_period_labels_and_nulls_roundtrip(tmp_path, frequency, labels):
    run = sample_run()
    source = pd.DataFrame(
        {"period": pd.PeriodIndex(labels, freq=frequency), "value": [1.0, 2.0, 3.0]}
    )
    run.outputs["performance"]["period_summary"] = source
    before = source.copy(deep=True)
    path = tmp_path / "analytics.duckdb"
    save_run(run, path)
    restored = read_output(path, run.run_id, "performance", "period_summary")
    assert restored["period"].dropna().tolist() == [labels[0], labels[2]]
    assert restored["period"].isna().tolist() == [False, True, False]
    pd.testing.assert_frame_equal(source, before)
