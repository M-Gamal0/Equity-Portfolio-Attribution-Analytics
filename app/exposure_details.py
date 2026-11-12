"""Presentation of saved EOP weights; no analytical estimation or database writes."""
import json

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from chart_renderer import render_chart
from research_ui import SERIES, date_encoding, display_table, pct, status_text, styled

SECTORS = ["Basic Materials", "Communication Services", "Consumer Cyclical", "Consumer Defensive", "Energy",
           "Financial Services", "Healthcare", "Industrials", "Real Estate", "Technology", "Utilities"]
SECTOR_COLORS = ["#0F2A47", "#0D9488", "#D97706", "#94A3B8", "#3B6F8F", "#71B8AE", "#B58B58", "#6B7280", "#BDC8D6", "#7F8D64", "#B79BA5"]


def monthly_sectors(source):
    """Complete only verified, fully invested monthly snapshots for area stacking."""
    frame = source.loc[source.weight_basis.eq("eop")].copy()
    frame["date"] = pd.to_datetime(frame.date)
    month = frame.date.dt.to_period("M")
    last = frame.groupby([frame.portfolio_id, month]).date.transform("max")
    observed = frame.loc[frame.date.eq(last)].copy()
    if observed.empty or observed.duplicated(["portfolio_id", "date", "sector"]).any():
        raise ValueError("Unique monthly sector snapshots are unavailable.")
    if not np.isfinite(observed.sector_weight).all() or observed.sector_weight.lt(0).any():
        raise ValueError("Saved sector weights are incomplete or outside the long-only weight contract.")
    sums = observed.groupby(["portfolio_id", "date"]).sector_weight.sum()
    if not np.allclose(sums, 1., atol=1e-8, rtol=0):
        raise ValueError("Monthly sector snapshots do not reconcile to full investment; absent sectors cannot be displayed as zero.")
    domain = list(dict.fromkeys(SECTORS + sorted(observed.sector.unique().tolist())))
    keys = observed[["portfolio_id", "date"]].drop_duplicates().merge(pd.DataFrame({"sector": domain}), how="cross")
    plotted = keys.merge(observed, on=["portfolio_id", "date", "sector"], how="left", validate="one_to_one", indicator=True)
    absent = plotted._merge.eq("left_only")
    plotted["observation_basis"] = np.where(absent, "Absent in complete saved snapshot · zero weight", "Saved sector observation")
    plotted.loc[absent, "sector_weight"] = 0.
    plotted.loc[absent, "weight_basis"] = "eop"
    return observed, plotted.drop(columns="_merge")


def closing_positions(source, closing_date):
    frame = source.loc[source.weight_basis.eq("eop") & pd.to_datetime(source.date).eq(pd.Timestamp(closing_date))].copy()
    if frame.empty or frame.duplicated(["portfolio_id", "security_id"]).any():
        raise ValueError("Unique closing security observations are unavailable on the saved summary date.")
    if not np.isfinite(frame.security_weight).all() or frame.security_weight.lt(0).any():
        raise ValueError("Closing security weights are incomplete or outside the long-only weight contract.")
    for owner in SERIES:
        own = frame.loc[frame.portfolio_id.eq(owner)]
        if own.empty or not np.isclose(own.security_weight.sum(), 1., atol=1e-8, rtol=0):
            raise ValueError(f"{owner} closing weights do not reconcile to full investment.")
    frame["holding_label"] = frame.ticker.fillna("Ticker unavailable").replace("", "Ticker unavailable")
    return frame


def holdings_ranking(closing, owner):
    frame = closing.loc[closing.portfolio_id.eq(owner) & closing.is_held].copy()
    if frame.empty or frame.security_id.duplicated().any() or not np.isfinite(frame.security_weight).all():
        raise ValueError("Eligible unique invested positions are unavailable.")
    ranked = frame.sort_values(["security_weight", "security_id"], ascending=[False, True], kind="stable").head(10)
    other = frame.loc[~frame.security_id.isin(ranked.security_id)]
    balance = pd.DataFrame({"group": ["Top ten positions", "Remaining positions", "All invested positions"],
                            "position_count": [len(ranked), len(other), len(frame)],
                            "position_weight": [ranked.security_weight.sum(), other.security_weight.sum(), frame.security_weight.sum()]})
    return ranked, balance


def headline_comparison(summary):
    fields = [("Invested holdings", "latest_invested_security_count", "count"),
              ("Effective holdings", "latest_effective_holdings", "ratio"), ("Top-five weight", "latest_top_five_weight", "percent")]
    with st.container(horizontal=True):
        for title, field, kind in fields:
            with st.container(border=True, width=260):
                st.markdown(f"**{title} · closing EOP**")
                with st.container(horizontal=True):
                    for owner in SERIES:
                        rows = summary.loc[summary.portfolio_id.eq(owner) & summary.weight_basis.eq("eop")]
                        row = rows.iloc[0] if len(rows) == 1 else None
                        available = row is not None and row.availability_status == "AVAILABLE" and pd.notna(row[field])
                        value = row[field] if available else float("nan")
                        shown = "Unavailable" if not available else pct(value) if kind == "percent" else f"{value:.1f}" if kind == "ratio" else f"{int(value):,}"
                        st.metric(owner, shown)
                        if not available:
                            st.caption(status_text(row.availability_status) if row is not None else "No unique saved summary")


def sector_history(frame, owner):
    plotted = frame.loc[frame.portfolio_id.eq(owner)]
    domain = list(dict.fromkeys(SECTORS + sorted(plotted.sector.unique().tolist())))
    palette = SECTOR_COLORS + ["#52657A"] * (len(domain) - len(SECTORS))
    chart = alt.Chart(plotted).mark_area(interpolate="step-after").encode(
        x=date_encoding(plotted), y=alt.Y("sector_weight:Q", title="Monthly closing EOP weight", stack="zero", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format=".0%")),
        color=alt.Color("sector:N", scale=alt.Scale(domain=domain, range=palette),
                        legend=alt.Legend(orient="bottom", columns=2, columnPadding=12, labelLimit=175)),
        order=alt.Order("sector:N", sort="ascending"),
        tooltip=[alt.Tooltip("date:T", title="Saved closing date"), "sector:N", alt.Tooltip("sector_weight:Q", title="EOP weight", format=".2%"), "observation_basis:N"],
    ).properties(height=340)
    render_chart(styled(chart), alt=f"Monthly closing sector weights for {owner}; all sector labels visible")


def tilt_chart(frame):
    chart = alt.Chart(frame).mark_bar(cornerRadiusEnd=3).encode(
        y=alt.Y("sector:N", title=None, sort=alt.SortField("active_sector_weight_pp", order="descending"), axis=alt.Axis(labelLimit=250)),
        x=alt.X("active_sector_weight_pp:Q", title="Portfolio minus benchmark sector weight (pp)"),
        color=alt.condition("datum.active_sector_weight_pp >= 0", alt.value("#0F2A47"), alt.value("#8293A8")),
        tooltip=["sector:N", alt.Tooltip("active_sector_weight_pp:Q", title="Active weight (pp)", format="+.2f"), alt.Tooltip("as_of_date:T", title="Closing date")],
    ).properties(height=max(300, len(frame) * 32))
    render_chart(styled(chart), alt="Closing sector overweights and underweights in percentage points")


def holdings_chart(frame, owner):
    plotted = frame.copy()
    plotted["security_key"] = [f"item_{i}" for i in range(len(plotted))]
    labels = dict(zip(plotted.security_key, plotted.holding_label))
    chart = alt.Chart(plotted).mark_bar(cornerRadiusEnd=3, color="#0D9488" if owner == SERIES[0] else "#D97706").encode(
        y=alt.Y("security_key:N", sort=plotted.security_key.tolist(), title=None, axis=alt.Axis(labelExpr=f"({json.dumps(labels)})[datum.label]", labelLimit=180)),
        x=alt.X("security_weight:Q", title="Closing EOP position weight", axis=alt.Axis(format=".1%", tickCount=5)),
        tooltip=[alt.Tooltip("holding_label:N", title="Ticker"), "sector:N", alt.Tooltip("security_weight:Q", title="Position weight", format=".2%"), alt.Tooltip("date:T", title="Closing date")],
    ).properties(height=max(250, len(frame) * 32))
    render_chart(styled(chart), alt=f"Top ten closing positions for {owner}; distinct security identities with ticker labels")


def active_share_chart(frame, field, title):
    if frame.empty or not np.isfinite(frame[field]).any():
        st.info("No saved Active Share observations are available for this selection.")
        return
    chart = alt.Chart(frame).mark_line(strokeWidth=2.4, color="#0D9488").encode(
        x=date_encoding(frame), y=alt.Y(f"{field}:Q", title=title, scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format=".0%")),
        tooltip=[alt.Tooltip("date:T", title="EOP date"), alt.Tooltip(f"{field}:Q", title=title, format=".2%")],
    ).properties(height=320)
    render_chart(styled(chart), alt=f"{title}: portfolio relative to benchmark, one pairwise series")


def supporting_data(choices):
    st.subheader("Supporting chart data")
    selected = st.selectbox("Holdings supporting data", list(choices))
    display_table(choices[selected], alt="Complete saved evidence for " + selected)
