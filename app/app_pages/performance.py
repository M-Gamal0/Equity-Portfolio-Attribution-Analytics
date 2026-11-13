"""Selected-month Performance, using saved results only."""
import pandas as pd
import streamlit as st

from research_ui import SERIES, context, growth_frame, line_chart, pct, detailed, methodology, report_unavailable
from performance_details import (return_comparisons, ratio_comparisons, monthly_chart, annual_chart,
                                 drawdown_comparisons, rolling_chart, capture_chart, supporting_data)

ctx = context()
ctx.heading("Performance", "Compare growth, consistency, losses and risk-adjusted performance across complete months.")
st.caption("Hypothetical 20-security active strategy versus a custom Top-100 eligible-issuer equal-weight benchmark; both rebalance quarterly. Total returns · Hypothetical frictionless same-close implementation.")
try:
    summary = ctx.load("performance", "performance_window").set_index("portfolio_id")
    relative = ctx.load("performance", "performance_relative_window").iloc[0]
    ratios = ctx.load("performance", "performance_risk_adjusted")
    raw_paths = ctx.load("risk", "risk_drawdown_daily")
    paths = growth_frame(raw_paths, ctx.definition)
    risk = ctx.load("risk", "risk_absolute_summary").set_index("portfolio_id")
    periods = ctx.load("performance", "performance_period")
except (KeyError, ValueError, OSError) as exc:
    report_unavailable("Selected-period results are temporarily unavailable. Please try again later.", exc)
    st.stop()

with st.container(horizontal=True):
    st.metric("Portfolio return", pct(summary.loc[SERIES[0]].period_return), border=True)
    st.metric("Benchmark return", pct(summary.loc[SERIES[1]].period_return), border=True)
    active = relative.arithmetic_active_period_return
    st.metric("Return difference", "Unavailable" if pd.isna(active) else f"{active * 100:+.2f} pp", border=True,
              help="Compounded portfolio return minus compounded benchmark return; percentage points.")
return_comparisons(summary)

with st.container(border=True):
    st.subheader("Portfolio and benchmark growth")
    st.caption("Total-return growth from the preceding month's closing valuation, including the first selected trading day's return.")
    line_chart(paths, "growth", "Growth of 100", height=360)

st.subheader("Monthly total returns")
series = st.segmented_control("Monthly return series", SERIES, default=SERIES[0], key="heatmap_series") or SERIES[0]
with st.container(border=True):
    monthly_chart(periods, series)
    st.caption("The color scale is shared by portfolio and benchmark for these selected months. Unselected months are absent; colors indicate return sign and size, not portfolio identity.")

with st.container(border=True):
    st.subheader("Portfolio and benchmark calendar-year returns")
    annual_chart(periods)
    st.caption("Partial years show their included months beneath the year. Each bar is the actual compounded return for those selected months, without annualization.")

with st.container(border=True):
    st.subheader("Portfolio and benchmark drawdown")
    line_chart(paths, "drawdown", "Drawdown", percent=True, height=330)
    st.caption("Drawdowns restart from the selected preceding-month opening close.")
drawdown_comparisons(risk, card_width=380)
st.page_link("app_pages/risk.py", label="Explore Risk", icon=":material/shield:")

st.subheader("Portfolio and benchmark risk-adjusted performance")
ratio_comparisons(ratios)
available = set(ctx.catalog.output_name)
rolling = None
capture = None
if "performance_rolling_relative" in available:
    with st.container(border=True):
        st.subheader("Rolling portfolio return relative to benchmark")
        rolling = ctx.history("performance", "performance_rolling_relative")
        horizons = sorted(rolling.horizon_id.dropna().unique(), key=lambda v: int(v.rstrip("M")))
        if horizons:
            horizon = st.selectbox("Rolling horizon", horizons, index=horizons.index("12M") if "12M" in horizons else 0, key="performance_rolling_horizon")
            rolling_chart(rolling, horizon)
        else:
            st.info("No saved rolling horizons are available for this selection.")
        st.caption("Zero means equal portfolio and benchmark growth. Only endpoints inside the selected period are shown; saved lookbacks may start earlier. Relative wealth differs from the headline percentage-point return difference. Overlapping rolling windows are not independent observations.")
if "performance_market_capture" in available:
    with st.container(border=True):
        st.subheader("Daily up and down market capture")
        capture = ctx.load("performance", "performance_market_capture").iloc[0]
        capture_chart(capture)

supporting_data(paths, raw_paths, risk, periods, ratios, rolling, capture)
detailed(ctx, "performance")
methodology(ctx, "Return difference subtracts compounded portfolio and benchmark returns; rolling relative wealth divides their growth multiples. Calendar-year bars include only selected months. Daily capture conditions on benchmark-positive or benchmark-negative trading sessions. Full-month selection does not waive saved CAGR, rolling-history or model eligibility requirements. This is hypothetical research performance, not live realized trading.")
