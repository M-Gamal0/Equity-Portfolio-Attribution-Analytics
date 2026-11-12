"""Turnover sectors: validated research calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd



# Research reference: 12_turnover_rebalancing_analytics.ipynb, code cell index 7.
def calculate_sector_rebalance_turnover(
    security_changes_df: pd.DataFrame,
    holdings_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    turnover_events_df: pd.DataFrame,
    *,
    tolerance: float = 1e-10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate security-level scheduled turnover by saved sector.

    Returns
    -------
    sector_df
        One row per (portfolio_id, selection_date, sector) for
        implemented subsequent rebalances only.
    audit_df
        One row per (portfolio_id, selection_date), including initial
        and pending selections marked as not assessed.
    """
    frames = {
        "security changes": security_changes_df,
        "holdings": holdings_df,
        "targets": targets_df,
        "turnover events": turnover_events_df,
    }
    required = {
        "security changes": {
            "portfolio_id", "selection_date", "security_id",
            "preceding_holdings_date", "weight_pre_rebalance",
            "weight_target", "weight_change", "implied_buy_weight",
            "implied_sell_weight", "one_way_turnover_contribution",
        },
        "holdings": {"portfolio_id", "date", "security_id", "sector"},
        "targets": {"portfolio_id", "selection_date", "security_id", "sector"},
        "turnover events": {
            "portfolio_id", "selection_date", "turnover_status",
            "one_way_turnover", "gross_two_way_turnover",
            "total_purchases", "total_sales", "effective_rebalance_date",
        },
    }
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    for label, frame in frames.items():
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{label} must be a pandas DataFrame")
        missing = required[label] - set(frame.columns)
        if missing:
            raise ValueError(f"{label} missing fields: {sorted(missing)}")

    s = security_changes_df.copy()
    h = holdings_df[["portfolio_id", "date", "security_id", "sector"]].copy()
    t = targets_df[["portfolio_id", "selection_date", "security_id", "sector"]].copy()
    e = turnover_events_df.copy()

    s["selection_date"] = pd.to_datetime(s["selection_date"], errors="raise")
    s["preceding_holdings_date"] = pd.to_datetime(
        s["preceding_holdings_date"], errors="raise"
    )
    h["date"] = pd.to_datetime(h["date"], errors="raise")
    t["selection_date"] = pd.to_datetime(t["selection_date"], errors="raise")
    e["selection_date"] = pd.to_datetime(e["selection_date"], errors="raise")
    e["effective_rebalance_date"] = pd.to_datetime(
        e["effective_rebalance_date"], errors="raise"
    )

    security_key = ["portfolio_id", "selection_date", "security_id"]
    event_key = ["portfolio_id", "selection_date"]
    for label, frame, key in (
        ("security changes", s, security_key),
        ("holdings", h, ["portfolio_id", "date", "security_id"]),
        ("targets", t, security_key),
        ("turnover events", e, event_key),
    ):
        if frame[key].isna().any().any() or frame.duplicated(key).any():
            raise ValueError(f"{label} contains null or duplicate primary keys")

    numerical_fields = [
        "weight_pre_rebalance", "weight_target", "weight_change",
        "implied_buy_weight", "implied_sell_weight",
        "one_way_turnover_contribution",
    ]
    for column in numerical_fields:
        values = pd.to_numeric(s[column], errors="raise").to_numpy(
            dtype=float, na_value=np.nan
        )
        if not np.isfinite(values).all():
            raise ValueError(f"Invalid numerical values: {column}")
    for column in ["weight_pre_rebalance", "weight_target"]:
        if ((s[column] < -tolerance) | (s[column] > 1 + tolerance)).any():
            raise ValueError(f"Invalid long-only weights: {column}")

    delta = s["weight_target"] - s["weight_pre_rebalance"]
    expected_numeric = {
        "weight_change": delta,
        "implied_buy_weight": delta.clip(lower=0),
        "implied_sell_weight": (-delta).clip(lower=0),
        "one_way_turnover_contribution": 0.5 * delta.abs(),
    }
    for column, expected in expected_numeric.items():
        if (s[column] - expected).abs().gt(tolerance).any():
            raise ValueError(f"Security turnover identity mismatch: {column}")

    eligible = e.loc[e["turnover_status"].eq("PASSED"), event_key]
    observed = s[event_key].drop_duplicates()
    if (
        len(observed) != len(eligible)
        or not pd.MultiIndex.from_frame(observed).isin(
            pd.MultiIndex.from_frame(eligible)
        ).all()
    ):
        raise ValueError("Security change events differ from eligible turnover events")

    s["absolute_weight_change"] = delta.abs()
    s["sector_one_way_contribution"] = 0.5 * delta.abs()

    # Exact-date joins use classifications persisted with the
    # actual previous-close holdings and selected targets.
    prev_sector = h.rename(columns={
        "date": "preceding_holdings_date", "sector": "previous_sector"
    })
    target_sector = t.rename(columns={"sector": "target_sector"})
    s = s.merge(
        prev_sector,
        on=["portfolio_id", "preceding_holdings_date", "security_id"],
        how="left", validate="many_to_one",
    ).merge(
        target_sector,
        on=security_key,
        how="left", validate="one_to_one",
    )

    for label in ["previous_sector", "target_sector"]:
        s[label] = s[label].astype("string").str.strip().replace("", pd.NA)

    prev_missing = s["previous_sector"].isna() & s["weight_pre_rebalance"].gt(tolerance)
    target_missing = s["target_sector"].isna() & s["weight_target"].gt(tolerance)
    both_missing = s["previous_sector"].isna() & s["target_sector"].isna()
    if (prev_missing | target_missing | both_missing).any():
        raise ValueError("Missing saved sector for a rebalance security")

    shared = s["previous_sector"].notna() & s["target_sector"].notna()
    mismatched = shared & s["previous_sector"].ne(s["target_sector"]).fillna(False)
    if mismatched.any():
        raise ValueError("Saved sector disagrees between previous and target positions")

    s["sector"] = s["target_sector"].combine_first(s["previous_sector"])

    sector_df = (
        s.groupby(event_key + ["sector"], as_index=False)
        .agg(
            security_rows=("security_id", "size"),
            previous_sector_weight=("weight_pre_rebalance", "sum"),
            target_sector_weight=("weight_target", "sum"),
            buy_weight=("implied_buy_weight", "sum"),
            sell_weight=("implied_sell_weight", "sum"),
            gross_two_way_turnover=("absolute_weight_change", "sum"),
            one_way_turnover_contribution=("sector_one_way_contribution", "sum"),
        )
    )
    sector_df = sector_df.merge(
        e[event_key + ["effective_rebalance_date"]],
        on=event_key, how="left", validate="many_to_one",
    )
    sector_df["net_sector_weight_change"] = (
        sector_df["target_sector_weight"] - sector_df["previous_sector_weight"]
    )
    sector_df["within_sector_offsetting_trade_weight"] = (
        np.minimum(sector_df["buy_weight"], sector_df["sell_weight"])
    )

    totals = (
        sector_df.groupby(event_key, as_index=False)
        .agg(
            sector_rows=("sector", "size"),
            security_rows=("security_rows", "sum"),
            previous_weight_sum=("previous_sector_weight", "sum"),
            target_weight_sum=("target_sector_weight", "sum"),
            sector_buy_total=("buy_weight", "sum"),
            sector_sell_total=("sell_weight", "sum"),
            sector_gross_total=("gross_two_way_turnover", "sum"),
            sector_one_way_total=("one_way_turnover_contribution", "sum"),
        )
    )
    audit_df = e[event_key + [
        "turnover_status", "one_way_turnover", "gross_two_way_turnover",
        "total_purchases", "total_sales",
    ]].merge(totals, on=event_key, how="left", validate="one_to_one")

    assessed = audit_df["turnover_status"].eq("PASSED")
    numeric_diffs = {
        "one_way_error": (audit_df["sector_one_way_total"] - audit_df["one_way_turnover"]).abs(),
        "gross_error": (audit_df["sector_gross_total"] - audit_df["gross_two_way_turnover"]).abs(),
        "buy_error": (audit_df["sector_buy_total"] - audit_df["total_purchases"]).abs(),
        "sell_error": (audit_df["sector_sell_total"] - audit_df["total_sales"]).abs(),
        "previous_weight_error": (audit_df["previous_weight_sum"] - 1).abs(),
        "target_weight_error": (audit_df["target_weight_sum"] - 1).abs(),
    }
    for label, errors in numeric_diffs.items():
        audit_df[label] = errors
    audit_df["sector_reconciliation_status"] = "NOT_ASSESSED"
    is_valid = assessed.copy()
    for errors in numeric_diffs.values():
        is_valid &= errors.notna() & errors.le(tolerance)
    audit_df.loc[assessed, "sector_reconciliation_status"] = np.where(
        is_valid.loc[assessed], "PASSED", "FAILED"
    )
    if (audit_df.loc[~assessed, "sector_one_way_total"].notna().any()
            or (audit_df["sector_reconciliation_status"] == "FAILED").any()):
        raise ValueError("Sector reconciliation failed or unavailable event has sector turnover")
    if int(sector_df["security_rows"].sum()) != len(s):
        raise ValueError("Security rows not conserved in sector aggregation")

    sector_df = sector_df.sort_values(event_key + ["sector"]).reset_index(drop=True)
    audit_df = audit_df.sort_values(event_key).reset_index(drop=True)
    return sector_df, audit_df
