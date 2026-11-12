"""Provider identities must stay server-side without merging securities."""
from pathlib import Path
from types import SimpleNamespace
import json

import pandas as pd
import pytest

pytest.importorskip("streamlit")


@pytest.fixture
def ui(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "app"))
    import research_ui
    monkeypatch.setattr(research_ui, "security_ticker_lookup", lambda *args: {
        "provider-a": "SAME", "provider-b": "SAME", "provider-c": "OTHER"})
    return research_ui


def test_table_copy_replaces_both_covariance_identifiers_and_preserves_values(ui):
    source = pd.DataFrame({"security_id": ["provider-a", "provider-b"],
                           "other_security_id": ["provider-b", "provider-a"],
                           "covariance_daily": [.0000123456789, -.0000987654321]})
    before = source.copy(deep=True)
    ctx = SimpleNamespace(path="fixture", build_id="fixture", iid="fixture")
    displayed = ui.public_security_frame(source, ctx)
    assert displayed.ticker.tolist() == ["SAME (security 1)", "SAME (security 2)"]
    assert displayed.other_ticker.tolist() == ["SAME (security 2)", "SAME (security 1)"]
    assert displayed.covariance_daily.tolist() == source.covariance_daily.tolist()
    assert "provider-" not in displayed.to_json()
    pd.testing.assert_frame_equal(source, before)


def test_saved_tickers_retained_and_missing_tickers_never_fall_back_to_ids(ui):
    source = pd.DataFrame({"security_id": ["provider-c", "unknown-a", "unknown-b"],
                           "ticker": ["HISTORICAL", None, ""], "security_weight": [.5, .3, .2]})
    ctx = SimpleNamespace(path="fixture", build_id="fixture", iid="fixture")
    displayed = ui.public_security_frame(source, ctx)
    assert displayed.ticker.tolist() == ["HISTORICAL", "Ticker unavailable (security 1)", "Ticker unavailable (security 2)"]
    assert displayed.security_weight.tolist() == [.5, .3, .2]
    assert "security_id" in source and "security_id" not in displayed


def test_chart_transport_drops_ids_and_keeps_duplicate_ticker_bars(ui, monkeypatch):
    import chart_renderer
    import exposure_details
    payloads = []
    monkeypatch.setattr(chart_renderer, "_CHART", lambda **kwargs: payloads.append(kwargs["data"]["spec"]))
    source = pd.DataFrame({"security_id": ["provider-a", "provider-b"],
                           "other_security_id": ["provider-b", "provider-a"],
                           "holding_label": ["SAME", "SAME"], "security_weight": [.6, .4],
                           "sector": ["Technology"] * 2, "date": pd.to_datetime(["2026-09-30"] * 2)})
    exposure_details.holdings_chart(source, "Active portfolio")
    spec = payloads[0]
    assert "provider-" not in json.dumps(spec)
    rows = next(iter(spec["datasets"].values()))
    assert len({r["security_key"] for r in rows}) == 2
    assert [r["security_weight"] for r in rows] == [.6, .4]
    assert all("security_id" not in r and "other_security_id" not in r for r in rows)
    assert source.security_id.tolist() == ["provider-a", "provider-b"]
