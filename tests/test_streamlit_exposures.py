"""Presentation safeguards for complete snapshots and holdings identity."""
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit")


@pytest.fixture
def ui(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "app"))
    import exposure_details
    return exposure_details


def test_monthly_sector_rows_are_closing_observations_not_averages(ui):
    source = pd.DataFrame({"portfolio_id": ["Active portfolio"] * 3, "date": pd.to_datetime(["2020-01-02", "2020-01-31", "2020-02-28"]),
                           "weight_basis": ["eop"] * 3, "sector": ["Technology", "Healthcare", "Technology"], "sector_weight": [1., 1., 1.]})
    raw, plotted = ui.monthly_sectors(source)
    assert len(raw) == 2 and raw.date.min() == pd.Timestamp("2020-01-31")
    january = plotted.loc[plotted.date.eq("2020-01-31")]
    assert january.loc[january.sector.eq("Healthcare"), "sector_weight"].iloc[0] == 1.
    assert january.loc[january.sector.eq("Technology"), "sector_weight"].iloc[0] == 0.
    assert january.observation_basis.str.contains("zero weight").sum() == 10
    assert len(source) == 3 and "observation_basis" not in source


def test_incomplete_sector_snapshot_is_not_filled_as_zero(ui):
    source = pd.DataFrame({"portfolio_id": ["Active portfolio"], "date": pd.to_datetime(["2020-01-31"]),
                           "weight_basis": ["eop"], "sector": ["Technology"], "sector_weight": [.9]})
    with pytest.raises(ValueError, match="full investment"):
        ui.monthly_sectors(source)
    with pytest.raises(ValueError, match="Unique"):
        ui.monthly_sectors(pd.concat([source, source]))


def test_closing_positions_use_exact_date_and_no_numeric_ticker_fallback(ui):
    source = pd.DataFrame({"portfolio_id": ["Active portfolio", "Benchmark"], "date": pd.to_datetime(["2020-12-31"] * 2),
                           "weight_basis": ["eop"] * 2, "security_id": ["123", "456"], "ticker": [None, "B"], "security_weight": [1., 1.], "is_held": [True, True]})
    closing = ui.closing_positions(source, "2020-12-31")
    assert closing.holding_label.tolist() == ["Ticker unavailable", "B"]
    assert pd.isna(source.ticker.iloc[0])
    with pytest.raises(ValueError, match="unavailable"):
        ui.closing_positions(source, "2020-12-30")


def test_top_ten_keeps_duplicate_ticker_identities_and_complete_remainder(ui, monkeypatch):
    source = pd.DataFrame({"portfolio_id": ["Active portfolio"] * 12, "security_id": [str(i) for i in range(12)],
                           "holding_label": ["SAME"] * 12, "security_weight": [1 / 12] * 12, "is_held": [True] * 12})
    ranked, balance = ui.holdings_ranking(source, "Active portfolio")
    assert ranked.security_id.nunique() == 10
    assert balance.position_count.tolist() == [10, 2, 12]
    assert balance.position_weight.iloc[:2].sum() == pytest.approx(balance.position_weight.iloc[2])
    specs = []
    monkeypatch.setattr(ui, "render_chart", lambda chart, **kwargs: specs.append(chart.to_dict()))
    ui.holdings_chart(ranked, "Active portfolio")
    assert specs[0]["encoding"]["y"]["field"] == "security_key"
    assert '"SAME"' in specs[0]["encoding"]["y"]["axis"]["labelExpr"]


def test_active_share_chart_has_no_fictitious_benchmark_series(ui, monkeypatch):
    specs = []
    monkeypatch.setattr(ui, "render_chart", lambda chart, **kwargs: specs.append(chart.to_dict()))
    source = pd.DataFrame({"date": pd.to_datetime(["2020-01-02", "2020-12-31"]), "security_active_share": [.7, .8]})
    ui.active_share_chart(source, "security_active_share", "Security Active Share")
    assert "color" not in specs[0]["encoding"]
    assert specs[0]["encoding"]["y"]["scale"]["domain"] == [0, 1]
