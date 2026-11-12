"""Read-only presentation of saved risk estimates and their eligibility."""
import json

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from chart_renderer import render_chart
from research_ui import COLORS, SERIES, date_axis, date_encoding, display_table, pct, status_text, styled


def tail_selection(frame, owner, confidence, kind):
    rows = frame.loc[frame.portfolio_id.eq(owner) & frame.confidence_level.eq(confidence) & frame.return_kind.eq(kind)]
    if len(rows) != 1:
        return None
    return rows.iloc[0]


def eligible_tail(row):
    return row is not None and row.coverage_status == "PASS" and np.isfinite([row.var_loss_daily, row.expected_shortfall_loss_daily]).all()


def rolling_selection(frame, lookback, value, status):
    selected = frame.loc[frame.lookback_sessions.eq(lookback)].copy()
    valid = selected[status].eq("PASS") & np.isfinite(selected[value])
    selected.loc[~valid, value] = float("nan")
    return selected, selected.loc[valid].copy()


def rolling_chart(frame, lookback, value, status, title, paired=False):
    selected, eligible = rolling_selection(frame, lookback, value, status)
    st.markdown(f"**{title}**")
    for owner, rows in selected.groupby("portfolio_id", sort=False):
        count = int(rows[value].notna().sum())
        st.caption(f"{owner}: {count:,} eligible endpoints · {len(rows) - count:,} unavailable · {lookback} sessions.")
    if eligible.empty:
        st.info("No eligible rolling estimates end in this selection for this lookback. Saved history requirements still apply; choose a shorter lookback or a later period.")
        return
    base = alt.Chart(selected).encode(
        x=date_encoding(selected), y=alt.Y(f"{value}:Q", title=title, axis=alt.Axis(format=".0%")),
        color=alt.Color("portfolio_id:N", scale=alt.Scale(domain=SERIES, range=COLORS), title=None, legend=alt.Legend() if paired else None),
        tooltip=[alt.Tooltip("date:T", title="Endpoint"), alt.Tooltip("portfolio_id:N", title="Portfolio"), alt.Tooltip(f"{value}:Q", title=title, format=".2%")],
    )
    chart = base.mark_line(strokeWidth=2) + base.transform_filter(f"isValid(datum.{value})").mark_point(size=12, opacity=.35)
    render_chart(styled(chart.properties(height=290)), alt=title + (" for portfolio and benchmark" if paired else " for the portfolio relative to the benchmark"))


def risk_evidence(source, value, share):
    frame = source.copy()
    frame["risk_contribution_pp"] = frame[value] * 100
    frame["risk_share"] = frame[share]
    return frame.drop(columns=[value, share] if share != "risk_share" else [value])


def risk_ranking(frame, count=12):
    if frame.security_id.duplicated().any() or not np.isfinite(frame.risk_contribution_pp).all() or not frame.status.eq("PASS").all():
        raise ValueError("Saved security contributions are incomplete or ineligible; inspect their statuses below.")
    ranked = frame.loc[frame.risk_contribution_pp.abs().sort_values(ascending=False, kind="stable").head(count).index].copy()
    other = frame.loc[~frame.security_id.isin(ranked.security_id)]
    balance = pd.DataFrame({"group": ["Ranked securities", "Other securities", "All securities"],
                            "security_count": [len(ranked), len(other), len(frame)],
                            "risk_contribution_pp": [ranked.risk_contribution_pp.sum(), other.risk_contribution_pp.sum(), frame.risk_contribution_pp.sum()]})
    return ranked, balance


def risk_bars(frame, label, measure, security=False):
    plotted = frame.copy()
    category = label
    axis = alt.Axis(labelLimit=290)
    if security:
        category = "security_key"
        plotted[category] = [f"item_{i}" for i in range(len(plotted))]
        labels = dict(zip(plotted[category], plotted.security_label))
        axis = alt.Axis(labelExpr=f"({json.dumps(labels)})[datum.label]", labelLimit=200)
    chart = alt.Chart(plotted).mark_bar(cornerRadiusEnd=3).encode(
        y=alt.Y(f"{category}:N", sort=alt.SortField("risk_contribution_pp", order="descending"), title=None, axis=axis),
        x=alt.X("risk_contribution_pp:Q", title=f"Contribution to annualized {measure} (pp)"),
        color=alt.condition("datum.risk_contribution_pp >= 0", alt.value("#0F2A47"), alt.value("#8293A8")),
        tooltip=[alt.Tooltip(f"{label}:N", title="Ticker" if security else "Sector"),
                 alt.Tooltip("risk_contribution_pp:Q", title="Contribution (pp)", format="+.3f"),
                 alt.Tooltip("risk_share:Q", title="Share of total risk", format=".2%"), "status:N"],
    ).properties(height=max(260, len(plotted) * 32))
    render_chart(styled(chart), alt=f"Signed {'security' if security else 'sector'} contributions to annualized {measure} in percentage points")


def reconciliation_caption(frame, target, measure):
    valid = frame.status.eq("PASS") & np.isfinite(frame.risk_contribution_pp)
    if not valid.all() or not np.isfinite(target):
        st.warning("The contribution set or saved risk target is unavailable. Inspect the complete status evidence below.")
        return False
    total = frame.risk_contribution_pp.sum()
    residual = total - target * 100
    st.caption(f"All contributions: {total:.6f} pp · Saved {measure} target: {target * 100:.6f}% · Display reconciliation residual: {residual:+.3e} pp.")
    if abs(residual) > 1e-6:
        st.error("Contributions do not reconcile to the saved target. Review the saved reconciliation before interpreting this chart.")
        return False
    return True


def tail_cards(tails):
    st.subheader("Historical one-day tail losses")
    kind = st.segmented_control("Tail return basis", ["Total returns", "Active returns"], default="Total returns") or "Total returns"
    confidence = st.segmented_control("Confidence level", [.95, .99], default=.95, format_func=lambda x: f"{x:.0%}") or .95
    owners = SERIES if kind == "Total returns" else SERIES[:1]
    for owner in owners:
        row = tail_selection(tails, owner, confidence, "SIMPLE_TOTAL" if kind == "Total returns" else "ARITHMETIC_ACTIVE")
        with st.container(border=True):
            st.markdown(f"**{owner}{' relative to benchmark' if kind == 'Active returns' else ''} · {confidence:.0%} confidence**")
            if row is None:
                st.warning("A unique saved estimate is unavailable for this return basis and confidence.")
                continue
            if eligible_tail(row):
                with st.container(horizontal=True):
                    st.metric("Daily VaR loss", pct(row.var_loss_daily))
                    st.metric("Daily expected shortfall", pct(row.expected_shortfall_loss_daily))
            else:
                st.warning("Tail estimate unavailable: " + status_text(row.coverage_status) + ".")
            st.caption(f"Effective tail mass {row.effective_tail_observation_mass:.2f} observations · Minimum required {row.minimum_tail_observation_mass:.0f} · {int(row.observations):,} daily observations.")
    st.caption("Historical daily loss estimates are not forecasts. Active returns are daily portfolio minus benchmark returns. Ineligible estimates remain hidden here and retain their saved statuses in supporting data.")


def episode_chart(episodes, definition):
    ranked = episodes.sort_values("maximum_drawdown", kind="stable").groupby("portfolio_id", sort=False).head(3).copy()
    if ranked.empty:
        st.info("No saved underwater episodes for this period.")
        return
    ranked["peak_display_date"] = pd.to_datetime(ranked.peak_date)
    baseline = ranked.peak_reference.eq("INITIAL_BASELINE")
    ranked.loc[baseline, "peak_display_date"] = pd.Timestamp(definition["opening_valuation_date"])
    ranked["end_display_date"] = pd.to_datetime(ranked.recovery_date).fillna(pd.to_datetime(ranked.censor_date))
    ranked["episode_label"] = ranked.portfolio_id + " · episode " + ranked.episode_number.astype(str)
    ranked["episode_end"] = ranked.episode_status.map({"RECOVERED": "Recovered", "OPEN_AT_WINDOW_END": "Open at cutoff"}).fillna(ranked.episode_status)
    axis_frame = pd.DataFrame({"peak_display_date": pd.concat([ranked.peak_display_date, ranked.end_display_date])})
    axis = date_axis(axis_frame, "peak_display_date")
    axes = dict(y=alt.Y("episode_label:N", sort=ranked.episode_label.tolist(), title=None, axis=alt.Axis(labelLimit=250)),
                color=alt.Color("portfolio_id:N", scale=alt.Scale(domain=SERIES, range=COLORS), title=None),
                tooltip=["portfolio_id:N", "peak_reference:N", "peak_display_date:T", "trough_date:T", "recovery_date:T", "censor_date:T", "episode_status:N",
                         alt.Tooltip("maximum_drawdown:Q", format=".2%"), "underwater_sessions:Q"])
    base = alt.Chart(ranked).encode(**axes)
    chart = (base.mark_rule(strokeWidth=5).encode(x=alt.X("peak_display_date:T", axis=axis), x2="end_display_date:T")
             + base.mark_point(filled=True, size=85).encode(x=alt.X("trough_date:T", axis=axis))
             + base.mark_point(filled=True, size=80).encode(x=alt.X("end_display_date:T", axis=axis),
                 shape=alt.Shape("episode_end:N", title="Episode end", scale=alt.Scale(domain=["Recovered", "Open at cutoff"], range=["diamond", "triangle-right"]))))
    render_chart(styled(chart.properties(height=max(160, len(ranked) * 42))), alt="Three deepest saved drawdown episodes per portfolio, from selected-period peak to recovery or censor date, with trough markers")
    st.caption("Three deepest episodes per portfolio. Dots mark troughs; diamonds mark recovery and right-pointing triangles mark an open episode at the observation cutoff. Initial-baseline peaks use the preceding-month opening close. Open episodes have no assumed recovery. All episodes and exact dates remain below.")


def supporting_data(choices):
    st.subheader("Supporting chart data")
    selected = st.selectbox("Risk supporting data", list(choices))
    display_table(choices[selected], alt="Complete saved evidence for " + selected)
