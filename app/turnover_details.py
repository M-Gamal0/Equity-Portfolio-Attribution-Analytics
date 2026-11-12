"""Saved turnover event scope, complete concentration and illustrative cost views."""
import json

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from chart_renderer import render_chart
from research_ui import COLORS, SERIES, display_table, pct, status_text, styled


def measured_events(events):
    fields = ["one_way_turnover", "total_purchases", "total_sales", "gross_two_way_turnover"]
    valid = events.event_status.eq("EFFECTIVE_REBALANCE") & events.turnover_status.eq("PASSED") & events.effective_rebalance_date.notna()
    valid &= np.isfinite(events[fields]).all(axis=1)
    return events.loc[valid].copy()


def event_scope(events):
    measured = measured_events(events)
    rows = []
    for owner, frame in events.groupby("portfolio_id", sort=False):
        initial = int(frame.event_status.eq("INITIAL_ESTABLISHMENT").sum())
        pending = int(frame.event_status.eq("PENDING_AFTER_CUTOFF").sum())
        count = int(measured.portfolio_id.eq(owner).sum())
        rows.append({"portfolio_id": owner, "measured_rebalances": count, "initial_establishment_events": initial,
                     "pending_events": pending, "other_or_ineligible_events": len(frame) - initial - pending - count})
    return pd.DataFrame(rows)


def paired_headlines(summary):
    fields = [("Measured recurring rebalances", "measured_rebalances", True), ("Cumulative one-way turnover", "cumulative_one_way_turnover", False),
              ("Average per measured rebalance", "average_rebalance_turnover", False), ("Cumulative gross two-way turnover", "cumulative_gross_two_way_turnover", False)]
    with st.container(horizontal=True):
        for title, field, count in fields:
            with st.container(border=True, width=380, height="stretch"):
                st.markdown("**" + title + "**")
                with st.container(horizontal=True):
                    for owner in SERIES:
                        rows = summary.loc[summary.portfolio_id.eq(owner)]
                        row = rows.iloc[0] if len(rows) == 1 else None
                        available = row is not None and row.turnover_measurement_status == "MEASURED" and pd.notna(row[field])
                        value = (f"{int(row[field]):,}" if count else pct(row[field])) if available else "Unavailable"
                        st.metric(owner, value)
                        if not available:
                            st.caption(status_text(row.turnover_measurement_status) if row is not None else "No unique saved summary")


def event_axis(dates):
    parsed = pd.to_datetime(dates).sort_values().drop_duplicates()
    events = parsed.dt.strftime("%Y-%m-%d").tolist()
    if (parsed.max() - parsed.min()).days > 730:
        first = parsed.groupby(parsed.dt.year).first()
        return alt.Axis(values=first.dt.strftime("%Y-%m-%d").tolist(), labelExpr="slice(datum.label, 0, 4)", labelAngle=0, title="Effective rebalance date")
    labels, years = {}, set()
    for date, event in zip(parsed, events):
        labels[event] = [date.strftime("%b"), str(date.year) if date.year not in years else ""]
        years.add(date.year)
    return alt.Axis(labelExpr=f"({json.dumps(labels)})[datum.label]", labelAngle=0, labelLineHeight=14, title="Effective rebalance date", titlePadding=18)


def event_chart(frame, value, group, groups, colors, title):
    if frame.empty:
        st.info("No eligible measured rebalance observations for this view.")
        return
    plotted = frame.copy()
    plotted["event"] = pd.to_datetime(plotted.effective_rebalance_date).dt.strftime("%Y-%m-%d")
    chart = alt.Chart(plotted).mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(
        x=alt.X("event:O", sort=sorted(plotted.event.unique()), axis=event_axis(plotted.effective_rebalance_date)),
        xOffset=alt.XOffset(f"{group}:N", sort=groups), y=alt.Y(f"{value}:Q", title=title, axis=alt.Axis(format=".0%", tickCount=6)),
        color=alt.Color(f"{group}:N", scale=alt.Scale(domain=groups, range=colors), title=None),
        tooltip=[alt.Tooltip("effective_rebalance_date:T", title="Effective date", format="%d %b %Y"), alt.Tooltip(f"{group}:N", title="Series"), alt.Tooltip(f"{value}:Q", title=title, format=".2%")],
    ).properties(height=300)
    render_chart(styled(chart), alt=title + " by saved effective rebalance date")


def turnover_ranking(frame, target):
    if frame.empty or frame.security_id.duplicated().any() or not np.isfinite(frame[["cumulative_one_way", "portfolio_turnover_share"]]).all().all():
        raise ValueError("Complete unique security turnover amounts/shares are unavailable.")
    if not np.isfinite(target) or target <= 0:
        raise ValueError("Turnover shares are unavailable without a positive measured turnover total.")
    if abs(frame.cumulative_one_way.sum() - target) > 1e-8 or abs(frame.portfolio_turnover_share.sum() - 1) > 1e-8:
        raise ValueError("Security turnover does not reconcile to the saved owner total.")
    ranked = frame.sort_values(["cumulative_rank", "security_id"], kind="stable").head(10).copy()
    other = frame.loc[~frame.security_id.isin(ranked.security_id)]
    balance = pd.DataFrame({"group": ["Top ten securities", "Remaining securities", "All securities"], "security_count": [len(ranked), len(other), len(frame)],
                            "cumulative_one_way_turnover": [ranked.cumulative_one_way.sum(), other.cumulative_one_way.sum(), frame.cumulative_one_way.sum()],
                            "turnover_share": [ranked.portfolio_turnover_share.sum(), other.portfolio_turnover_share.sum(), frame.portfolio_turnover_share.sum()]})
    return ranked, balance


def concentration_chart(frame, owner):
    plotted = frame.copy()
    plotted["security_key"] = [f"item_{i}" for i in range(len(plotted))]
    labels = dict(zip(plotted.security_key, plotted.security_label))
    chart = alt.Chart(plotted).mark_bar(cornerRadiusEnd=3, color=COLORS[SERIES.index(owner)]).encode(
        y=alt.Y("security_key:N", sort=plotted.security_key.tolist(), title=None, axis=alt.Axis(labelLimit=190, labelExpr=f"({json.dumps(labels)})[datum.label]")),
        x=alt.X("portfolio_turnover_share:Q", title="Share of owner's cumulative one-way turnover", axis=alt.Axis(format=".1%", tickCount=6)),
        tooltip=[alt.Tooltip("security_label:N", title="Ticker (latest within selection)"), alt.Tooltip("portfolio_turnover_share:Q", title="Turnover share", format=".2%"), alt.Tooltip("cumulative_one_way:Q", title="Cumulative one-way NAV turnover", format=".2%")],
    ).properties(height=max(250, len(frame) * 32))
    render_chart(styled(chart), alt="Top ten security shares of cumulative one-way turnover for " + owner)


def cost_scenarios(costs, rates=(1, 5, 10, 25)):
    rows = []
    for row in costs.itertuples(index=False):
        available = row.cost_sensitivity_status == "UNIT_RATE_SENSITIVITY_ONLY" and np.isfinite(row.nav_bp_per_1bp_traded)
        for rate in rates:
            rows.append({"portfolio_id": row.portfolio_id, "assumed_traded_weight_rate_bp": rate,
                         "arithmetic_nav_cost_bp": rate * row.nav_bp_per_1bp_traded if available else float("nan"),
                         "nav_bp_per_1bp_traded": row.nav_bp_per_1bp_traded, "cost_sensitivity_status": row.cost_sensitivity_status})
    return pd.DataFrame(rows)


def cost_chart(scenarios):
    valid = scenarios.loc[np.isfinite(scenarios.arithmetic_nav_cost_bp)]
    if valid.empty:
        st.info("No eligible saved unit-cost sensitivity for these scenarios.")
        return
    chart = alt.Chart(valid).mark_bar(cornerRadiusTopLeft=3).encode(
        x=alt.X("assumed_traded_weight_rate_bp:O", sort=[1, 5, 10, 25], title="Assumed rate on purchases AND sales (bp)", axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("portfolio_id:N", sort=SERIES), y=alt.Y("arithmetic_nav_cost_bp:Q", title="Hypothetical arithmetic NAV cost (bp)", axis=alt.Axis(tickCount=6)),
        color=alt.Color("portfolio_id:N", scale=alt.Scale(domain=SERIES, range=COLORS), title=None),
        tooltip=["portfolio_id:N", alt.Tooltip("assumed_traded_weight_rate_bp:Q", title="Assumed rate (bp)"), alt.Tooltip("arithmetic_nav_cost_bp:Q", title="Arithmetic NAV cost (bp)", format=".2f")],
    ).properties(height=290)
    render_chart(styled(chart), alt="Portfolio and benchmark hypothetical arithmetic NAV cost at four ordered traded-weight rates")


def supporting_data(choices):
    st.subheader("Supporting chart data")
    selected = st.selectbox("Turnover supporting data", list(choices))
    display_table(choices[selected], alt="Complete saved evidence for " + selected)
