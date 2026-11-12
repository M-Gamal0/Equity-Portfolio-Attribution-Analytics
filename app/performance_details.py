"""Read-only Performance presentation; no financial estimators or writes."""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from chart_renderer import render_chart
from research_ui import SERIES, COLORS, date_encoding, display_table, pct, status_text, styled


def paired_card(title, values, kind="percent", *, width="stretch"):
    with st.container(border=True, width=width):
        st.markdown(f"**{title} · selected period**")
        with st.container(horizontal=True):
            for owner, value in zip(SERIES, values):
                shown = "Unavailable" if pd.isna(value) else pct(value) if kind == "percent" else f"{int(value):,}" if kind == "sessions" else f"{value:.3f}"
                st.metric(owner, shown)


def saved_ratio(ratios, name):
    rows = ratios.loc[ratios.metric_name.eq(name)]
    if rows.empty:
        return float("nan"), "No saved estimate"
    row = rows.iloc[0]
    return (row.metric_value if row.calculation_status == "CALCULATED" else float("nan")), row.calculation_status


def return_comparisons(summary):
    values = [summary.loc[o].get("annualized_return_cagr") if summary.loc[o].get("cagr_status") == "CALCULATED" else float("nan") for o in SERIES]
    paired_card("CAGR", values)
    for owner, value in zip(SERIES, values):
        if pd.isna(value):
            st.caption(f"{owner} CAGR unavailable: {status_text(summary.loc[owner].get('cagr_status', 'No saved estimate'))}.")


def ratio_comparisons(ratios):
    names = [("sharpe_ratio", "Sharpe ratio"), ("sortino_ratio_zero_target", "Sortino ratio")]
    with st.container(horizontal=True, vertical_alignment="distribute"):
        for suffix, title in names:
            paired_card(title, [saved_ratio(ratios, p + "_" + suffix)[0] for p in ("portfolio", "benchmark")], "ratio")
    for suffix, title in names:
        for owner, prefix in zip(SERIES, ("portfolio", "benchmark")):
            value, status = saved_ratio(ratios, prefix + "_" + suffix)
            if pd.isna(value):
                st.caption(f"{owner} {title.lower()} unavailable: {status_text(status)}.")
    value, status = saved_ratio(ratios, "information_ratio")
    st.metric("Information ratio · portfolio relative to benchmark", f"{value:.3f}" if pd.notna(value) else "Unavailable", border=True)
    if pd.isna(value):
        st.caption("Information ratio unavailable: " + status_text(status))
    st.caption("Sharpe uses the saved risk-free series; Sortino uses a zero daily-return target. Information ratio describes active return relative to tracking error and is shown once. Differences are not automatically labeled favorable or unfavorable.")


def monthly_scale(periods):
    monthly = periods.loc[periods.period_type.eq("monthly") & periods.portfolio_id.isin(SERIES), "period_return"]
    finite = monthly[np.isfinite(monthly)]
    return max(float(finite.abs().max()), 0.01) if not finite.empty else 0.01


def monthly_chart(periods, series):
    frame = periods.loc[periods.period_type.eq("monthly") & periods.portfolio_id.eq(series)].copy()
    if frame.empty:
        st.info("No saved monthly returns are available for this selection.")
        return
    frame["year"] = pd.to_datetime(frame.period_start).dt.year.astype(str)
    frame["month"] = pd.to_datetime(frame.period_start).dt.strftime("%b")
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    limit = monthly_scale(periods)
    axes = dict(x=alt.X("month:O", sort=months, title=None, axis=alt.Axis(labelAngle=0)), y=alt.Y("year:O", title=None))
    cells = alt.Chart(frame).mark_rect(cornerRadius=3, stroke="white", strokeWidth=3).encode(
        **axes, color=alt.Color("period_return:Q", title="Monthly return", scale=alt.Scale(domain=[-limit, 0, limit],
        range=["#D97706", "#F7F9FC", "#0D9488"], interpolate="rgb", clamp=True), legend=alt.Legend(format=".0%")),
        tooltip=["year", "month", alt.Tooltip("period_return:Q", title="Return", format=".2%")],
    ).properties(height=max(120, frame.year.nunique() * 32))
    labels = alt.Chart(frame).mark_text(fontSize=11, fontWeight=500, color="black").encode(**axes, text=alt.Text("period_return:Q", format=".1%"))
    render_chart(styled(cells + labels), alt=f"Monthly total returns for {series}; shared portfolio and benchmark color scale")


def annual_frame(periods):
    """Label partial years using saved monthly rows; never annualize returns."""
    annual = periods.loc[periods.period_type.eq("annual")].copy()
    monthly = periods.loc[periods.period_type.eq("monthly")].copy()
    monthly["year"] = pd.to_datetime(monthly.period_start).dt.year
    annual["year"] = pd.to_datetime(annual.period_start).dt.year
    labels = {}
    for year, rows in monthly.groupby("year"):
        months = pd.to_datetime(rows.period_start).sort_values().drop_duplicates()
        count = len(months)
        span = months.iloc[0].strftime("%b") if count == 1 else f"{months.iloc[0]:%b}–{months.iloc[-1]:%b}"
        labels[year] = str(year) if count == 12 else f"{year}|{span} ({count}m)"
    annual["calendar_label"] = annual.year.map(labels).fillna(annual.year.astype(str))
    return annual.sort_values(["year", "portfolio_id"])


def annual_chart(periods):
    frame = annual_frame(periods)
    chart = alt.Chart(frame).mark_bar(cornerRadiusTopLeft=3).encode(
        x=alt.X("calendar_label:O", title=None, sort=frame.calendar_label.drop_duplicates().tolist(),
                axis=alt.Axis(labelExpr="split(datum.label, '|')", labelAngle=0, labelLineHeight=14, labelOverlap=False)),
        xOffset="portfolio_id:N", y=alt.Y("period_return:Q", title="Selected-month total return", axis=alt.Axis(format=".0%")),
        color=alt.Color("portfolio_id:N", scale=alt.Scale(domain=SERIES, range=COLORS), title=None),
        tooltip=[alt.Tooltip("portfolio_id:N", title="Series"), alt.Tooltip("year:O", title="Calendar year"),
                 alt.Tooltip("period_return:Q", title="Selected-month return", format=".2%"),
                 alt.Tooltip("observed_calendar_months:Q", title="Months included")],
    ).properties(height=300)
    render_chart(styled(chart), alt="Portfolio and benchmark returns by calendar year; partial years explicitly labeled")


def drawdown_comparisons(risk, *, card_width="stretch"):
    fields = [("Maximum drawdown", "maximum_drawdown", "percent"), ("Drawdown at period end", "ending_drawdown", "percent"),
              ("Longest observed underwater spell (sessions)", "longest_observed_underwater_duration_sessions", "sessions"),
              ("Underwater spell at period end (sessions)", "ending_underwater_duration_sessions", "sessions")]
    with st.container(horizontal=True, vertical_alignment="distribute"):
        for title, field, kind in fields:
            paired_card(title, [risk.loc[o].get(field, float("nan")) for o in SERIES], kind, width=card_width)
    for owner in SERIES:
        ending = risk.loc[owner].get("ending_drawdown", float("nan"))
        if pd.notna(ending):
            st.caption(f"{owner}: {'below its selected-period peak at the end; the ending underwater spell is still open' if ending < -1e-12 else 'at its selected-period peak at the end'}. No future recovery date is assumed.")
    st.caption("Durations count observed trading sessions inside the selection. An open ending spell is incomplete; the longest observed spell need not be a completed recovery. These figures do not describe a peak before the selected opening valuation.")


def rolling_selection(frame, horizon):
    selected = frame.loc[frame.horizon_id.eq(horizon)].copy()
    valid = selected.calculation_status.eq("AVAILABLE") & np.isfinite(selected.geometric_relative_return)
    return selected, selected.loc[valid].copy()


def rolling_chart(frame, horizon):
    selected, eligible = rolling_selection(frame, horizon)
    st.caption(f"{len(eligible):,} eligible endpoints · {len(selected) - len(eligible):,} unavailable endpoints · {horizon} saved lookback.")
    if eligible.empty:
        st.info(f"No eligible {horizon} rolling windows end in this selection. Saved lookback requirements are not met. Choose another horizon or a later period; no estimates have been filled or recalculated.")
        return
    plotted = selected.copy()
    plotted.loc[~plotted.index.isin(eligible.index), "geometric_relative_return"] = float("nan")
    line = alt.Chart(plotted).mark_line(color=COLORS[0], strokeWidth=2.4, point=len(eligible) == 1, invalid="break-paths-show-domains").encode(
        x=date_encoding(plotted), y=alt.Y("geometric_relative_return:Q", title="Relative wealth return", axis=alt.Axis(format=".1%")),
        tooltip=[alt.Tooltip("date:T", title="Window end", format="%d %b %Y"),
                 alt.Tooltip("resolved_start_date:T", title="Window opening close", format="%d %b %Y"),
                 alt.Tooltip("geometric_relative_return:Q", title="Relative wealth return", format=".2%"),
                 alt.Tooltip("observation_count:Q", title="Return sessions")],
    )
    parity = alt.Chart(pd.DataFrame({"parity": [0]})).mark_rule(color="#9CA3AF").encode(y="parity:Q")
    render_chart(styled((line + parity).properties(height=330)), alt=f"{horizon} portfolio relative wealth return versus benchmark; zero means equal growth")


def capture_frame(capture):
    rows = []
    for regime, name in [("up", "Benchmark up days"), ("down", "Benchmark down days")]:
        status = capture.get(regime + "_capture_status", "Unavailable")
        rows.append({"Regime": name, "Capture (%)": capture.get(regime + "_geometric_capture_pct", float("nan")) if status == "CALCULATED" else float("nan"),
                     "Observations": capture.get(regime + "_observations"), "Saved status": status})
    return pd.DataFrame(rows)


def capture_chart(capture):
    data = capture_frame(capture)
    eligible = data.loc[np.isfinite(data["Capture (%)"])]
    if eligible.empty:
        st.info("Daily capture is unavailable for both saved benchmark regimes.")
    else:
        chart = alt.Chart(eligible).mark_bar(color="#0F2A47", cornerRadiusEnd=3).encode(
            y=alt.Y("Regime:N", sort=["Benchmark up days", "Benchmark down days"], title=None, axis=alt.Axis(labelLimit=220)),
            x=alt.X("Capture (%):Q", title="Geometric daily capture (%)", axis=alt.Axis(format=".0f", tickCount=6)),
            tooltip=["Regime", alt.Tooltip("Capture (%):Q", format=".2f"), "Observations"],
        )
        parity = alt.Chart(pd.DataFrame({"parity": [100]})).mark_rule(color="#D97706", strokeWidth=2).encode(x="parity:Q")
        render_chart(styled((chart + parity).properties(height=220)), alt="Daily up and down geometric capture; orange reference is 100 percent benchmark parity")
    for regime, value, observations, status in data.itertuples(index=False, name=None):
        st.caption(f"{regime}: {int(observations):,} observations" if pd.notna(observations) else f"{regime}: observation count unavailable")
        if pd.isna(value):
            st.caption("Capture unavailable: " + status_text(status))
    st.caption(f"Benchmark-flat days excluded: {capture.get('flat_observations', 'unavailable')}. Orange line = 100% benchmark parity. Neutral bars do not assign a quality score. Regimes use benchmark daily returns, not monthly returns; capture compares conditional geometric mean daily returns.")


def supporting_data(paths, raw_paths, risk, periods, ratios, rolling=None, capture=None):
    st.subheader("Supporting chart data")
    frames = {"Plotted growth and drawdown observations": paths, "Original saved drawdown observations": raw_paths,
              "Drawdown summary and durations": risk.reset_index(), "Calendar returns and coverage": periods,
              "Risk-adjusted metrics and eligibility": ratios}
    if rolling is not None:
        frames["Rolling windows and availability · all horizons"] = rolling
    if capture is not None:
        frames["Daily market capture and sample counts"] = pd.DataFrame([capture])
    choice = st.selectbox("Performance supporting data", list(frames), key="performance_supporting_data")
    frame = frames[choice]
    if choice.startswith("Plotted"):
        st.caption("Every plotted row, including the preceding-month opening close at growth 100 and drawdown zero. Null diagnostic fields on that added display anchor are not imputed estimates.")
    if choice.startswith("Rolling"):
        st.caption("All saved horizons and statuses, including unavailable endpoints. Estimation windows can begin before the selected reporting start.")
    display_table(frame, alt="Performance supporting data: " + choice)
    st.caption(f"{len(frame):,} {'row' if len(frame) == 1 else 'rows'} available · Scroll to inspect the full table; no rows silently removed.")
