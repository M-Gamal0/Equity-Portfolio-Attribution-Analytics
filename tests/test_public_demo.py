"""Offline synthetic fixtures test public adapters without vendor/private files."""
from io import BytesIO
from dataclasses import asdict
import json
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from portfolio_analytics.config import AnalysisSettings
from portfolio_analytics.data.snapshots import load_standardized_inputs
from portfolio_analytics.demo import DemoConfig, construct_inputs, save_demo_snapshot, validate_demo
from portfolio_analytics.demo.build import ACTIVE_ID, BENCHMARK_ID, eligible_ranking
from portfolio_analytics.demo.sources import FACTOR_COLUMNS, SourceCache, parse_french_zip, parse_universe
from portfolio_analytics.demo.cli import cached_history_index
from portfolio_analytics.pipeline.runner import DOMAIN_INPUTS, run_analysis


@pytest.fixture(scope="module")
def inputs():
    config = DemoConfig(cutoff="2025-03-31", benchmark_size=3, active_size=2)
    dates = pd.bdate_range(config.history_start, config.cutoff, name="date")
    rng = np.random.default_rng(7291)
    histories = {}
    for ticker in ("GOOG", "GOOGL", "MSFT", "SHORT"):
        price = 100 * np.cumprod(1 + rng.normal(.0003, .012, len(dates)))
        histories[ticker] = pd.DataFrame({"Close": price, "Adj Close": price * np.exp(np.arange(len(dates)) * .00002)}, index=dates)
    # A high-ranked IPO after inception must not be backfilled or selected.
    histories["SHORT"] = histories["SHORT"].loc["2022-01-01":]
    quotes = pd.DataFrame({"ticker": ["GOOG", "GOOGL", "MSFT", "SHORT"],
                           "marketCap": [900., 900., 700., 1000.], "currency": ["USD"] * 4,
                           "sector": ["Communication Services", "Communication Services", "Technology", "Technology"]})
    factors = pd.DataFrame({"date": dates, "rf": .00003,
                            **{name: rng.normal(0, .005, len(dates)) for name in FACTOR_COLUMNS}})
    frames, audit = construct_inputs(quotes, histories, histories["MSFT"], factors, config)
    return config, quotes, histories, factors, frames, audit


def test_current_cap_ranking_preserves_classes_and_excludes_incomplete_history(inputs):
    config, quotes, histories, _, _, audit = inputs
    selected, _ = eligible_ranking(quotes, histories, histories["MSFT"], config)
    assert selected.ticker.tolist() == ["GOOG", "GOOGL", "MSFT"]
    assert audit.loc[audit.ticker.eq("SHORT"), "exclusion_reason"].item() == "INCOMPLETE_OR_INVALID_PORTFOLIO_PERIOD_PRICES"
    assert set(audit.loc[audit.selected_active, "ticker"]) == {"GOOG", "GOOGL"}


def test_quarterly_targets_full_first_return_and_passive_drift(inputs):
    config, _, _, _, frames, _ = inputs
    report = validate_demo(frames, config)
    assert report["status"] == "PASSED" and report["factor_coverage_ratio"] == 1
    holdings, targets = frames["portfolio_holdings"], frames["rebalance_targets"]
    assert holdings.date.min() == pd.Timestamp("2021-10-01")
    assert holdings.date.max() == pd.Timestamp(config.cutoff)
    assert targets.selection_date.nunique() == 15
    for owner, size in ((ACTIVE_ID, 2), (BENCHMARK_ID, 3)):
        own = holdings.loc[holdings.portfolio_id.eq(owner)]
        events = targets.loc[targets.portfolio_id.eq(owner)]
        reset_dates = set(events.effective_rebalance_date.dropna())
        for date, rows in own.groupby("date"):
            if date in reset_dates:
                np.testing.assert_allclose(rows.weight_bop, 1 / size, rtol=0, atol=1e-15)
        pivot = own.pivot(index="date", columns="security_id", values="weight_eop")
        opening = own.pivot(index="date", columns="security_id", values="weight_bop")
        no_trade = [d for d in pivot.index[1:] if d not in reset_dates]
        np.testing.assert_allclose(opening.loc[no_trade], pivot.shift().loc[no_trade], rtol=0, atol=0)
        pending = events.loc[events.selection_date.eq(config.cutoff)]
        assert len(pending) == size and pending.effective_rebalance_date.isna().all()


def test_bundle_detects_corrupt_returns_and_missing_factor_day(inputs):
    config, _, _, _, frames, _ = inputs
    damaged = {k: v.copy(deep=True) for k, v in frames.items()}
    damaged["portfolio_returns"].loc[0, "return_total"] += .01
    with pytest.raises(ValueError):
        validate_demo(damaged, config)
    damaged = {k: v.copy(deep=True) for k, v in frames.items()}
    date = frames["portfolio_returns"].date.min()
    damaged["factor_returns"] = damaged["factor_returns"].loc[lambda f: f.date.ne(date)]
    with pytest.raises(ValueError, match="calendar coverage"):
        validate_demo(damaged, config)


def test_immutable_snapshot_exact_readback_and_tamper_detection(inputs, tmp_path):
    config, _, _, _, frames, audit = inputs
    snapshot = save_demo_snapshot(tmp_path, frames, config, sources={"fixture": "synthetic_offline_test"}, selection_audit=audit)
    same = save_demo_snapshot(tmp_path, frames, config, sources={"fixture": "synthetic_offline_test"}, selection_audit=audit)
    assert same == snapshot
    loaded = load_standardized_inputs(tmp_path, snapshot_id=snapshot, verify=True)
    assert set(loaded) == set(frames)
    for name in loaded:
        assert len(loaded[name]) == len(frames[name])
    p = tmp_path / snapshot / "portfolio_returns.parquet"
    p.write_bytes(p.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="SHA-256"):
        load_standardized_inputs(tmp_path, snapshot_id=snapshot)


def test_all_six_existing_domains_run_without_private_files(inputs, tmp_path):
    config, _, _, _, frames, audit = inputs
    snapshot = save_demo_snapshot(tmp_path, frames, config, sources={"fixture": "synthetic_offline_test"}, selection_audit=audit)
    settings = AnalysisSettings(ACTIVE_ID, BENCHMARK_ID, config.cutoff, config.inception,
                                 reporting_years=(1,), performance_windows=("SI", "1Y"))
    result = run_analysis(snapshot_root=tmp_path, snapshot_id=snapshot, settings=settings, domains=tuple(DOMAIN_INPUTS))
    assert result.status == "COMPLETED", result.statuses
    assert set(result.outputs) == set(DOMAIN_INPUTS)
    turnover = result.outputs["turnover"]["rebalance_turnover_events"]
    assert turnover.loc[turnover.event_status.eq("EFFECTIVE_REBALANCE"), "one_way_turnover"].gt(0).any()


def test_french_parser_converts_percent_once_and_rejects_sentinels():
    def archive(text):
        out = BytesIO()
        with ZipFile(out, "w") as z:
            z.writestr("daily.csv", text)
        return out.getvalue()
    text = "Public explanatory header\n,Mkt-RF,SMB,HML,RMW,CMA,RF\n20210930,1,2,3,4,5,0.01\n\n2021,10,20,30,40,50,1\n"
    row = parse_french_zip(archive(text)).iloc[0]
    assert row.mkt_rf == .01 and row.rf == .0001
    assert len(parse_french_zip(archive(text))) == 1
    with pytest.raises(ValueError, match="sentinel"):
        parse_french_zip(archive(text.replace("20210930,1", "20210930,-99.99")))
    momentum = parse_french_zip(archive("Notes\n,Mom\n20210930,1.50\n"), momentum=True)
    assert momentum.mom.iloc[0] == .015


def test_universe_parser_keeps_two_classes_and_rejects_duplicate_tickers():
    symbols = ["GOOG", "GOOGL", *[f"T{i}" for i in range(99)]]
    data = {"data": {"data": {"rows": [{"symbol": t, "companyName": t} for t in symbols]}}}
    frame = parse_universe(json.dumps(data).encode())
    assert len(frame) == 101 and {"GOOG", "GOOGL"}.issubset(set(frame.ticker))
    data["data"]["data"]["rows"][1]["symbol"] = "GOOG"
    with pytest.raises(ValueError, match="universe"):
        parse_universe(json.dumps(data).encode())


def test_source_cache_offline_replay_verifies_hash_without_network(tmp_path):
    online = SourceCache(tmp_path)
    assert online.get("source.txt", "fixture", lambda: b"public") == b"public"
    offline = SourceCache(tmp_path, offline=True)
    def no_network():
        raise AssertionError("Offline replay called network")
    assert offline.get("source.txt", "fixture", no_network) == b"public"
    with pytest.raises(FileNotFoundError, match="Offline"):
        offline.get("missing.txt", "fixture", no_network)
    (tmp_path / "source.txt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="mismatch"):
        offline.get("source.txt", "fixture", no_network)


def test_twenty_stock_targets_are_five_percent_and_benchmark_stays_fifty():
    config = DemoConfig(cutoff="2021-12-31")
    dates = pd.bdate_range("2021-09-01", config.cutoff, name="date")
    names = [f"T{i:02d}" for i in range(50)]
    histories = {ticker: pd.DataFrame({"Close": 100 * np.exp(np.arange(len(dates)) * (.0001 + i * .00001)),
                                      "Adj Close": 100 * np.exp(np.arange(len(dates)) * (.0002 + i * .00001))}, index=dates)
                 for i, ticker in enumerate(names)}
    quotes = pd.DataFrame({"ticker": names, "marketCap": np.arange(50, 0, -1).astype(float),
                           "currency": "USD", "sector": [f"Sector {i % 5}" for i in range(50)]})
    factors = pd.DataFrame({"date": dates, "rf": .00001, **{name: .001 for name in FACTOR_COLUMNS}})
    frames, audit = construct_inputs(quotes, histories, histories[names[0]], factors, config)
    targets = frames["rebalance_targets"]
    active = targets.loc[targets.portfolio_id.eq(ACTIVE_ID)]
    benchmark = targets.loc[targets.portfolio_id.eq(BENCHMARK_ID)]
    assert ACTIVE_ID == "demo_active20" and config.active_size == 20
    assert active.groupby("selection_date").size().eq(20).all()
    assert active.weight_target.eq(.05).all()
    assert benchmark.groupby("selection_date").size().eq(50).all()
    assert benchmark.weight_target.eq(.02).all()
    assert audit.selected_active.sum() == 20 and audit.selected_benchmark.sum() == 50


def test_twenty_holdings_reuses_old_ten_holdings_acquisition_cache(tmp_path):
    cache = SourceCache(tmp_path)
    legacy = "derived:yfinance-price-acquisition:" + json.dumps(asdict(DemoConfig(active_size=10)), sort_keys=True)
    cache.get("history_index.json", legacy, lambda: b'[{"ticker":"GOOG","status":"DOWNLOADED"}]')
    before = (tmp_path / "history_index.json.source.json").read_bytes()
    def forbidden():
        raise AssertionError("Changing holdings count must not download market data")
    rows = cached_history_index(SourceCache(tmp_path, offline=True), DemoConfig(), forbidden)
    assert rows[0]["ticker"] == "GOOG"
    assert (tmp_path / "history_index.json.source.json").read_bytes() == before
    with pytest.raises(ValueError, match="price dates differ"):
        cached_history_index(cache, DemoConfig(cutoff="2025-12-31"), forbidden)
