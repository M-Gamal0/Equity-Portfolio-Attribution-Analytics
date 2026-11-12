"""Curated cross-domain view of the applied complete-month interval."""
import streamlit as st
from comparison import render_comparison

from research_ui import SERIES, context, pct, growth_frame, line_chart, methodology
from overview_details import risk_headlines, sector_chart, supporting_data

ctx = context()
ctx.heading("Overview", "The result, the risk and the decisions behind the portfolio.")
st.caption("Hypothetical 20-security active strategy versus a custom Top-100 eligible-issuer equal-weight benchmark; both rebalance quarterly. Total returns · Hypothetical frictionless same-close implementation.")
performance = ctx.load("performance", "performance_window").set_index("portfolio_id")
risk = ctx.load("risk", "risk_absolute_summary").set_index("portfolio_id")
active = ctx.load("performance", "performance_relative_window").iloc[0]
p = performance.loc[SERIES[0]]
with st.container(horizontal=True):
    for label, value, help_text in [
        ("Portfolio return", pct(p.period_return), "Compounded total return for the selected period."),
        ("Benchmark return", pct(performance.loc[SERIES[1]].period_return), "Same selected months and opening close."),
        ("Return difference", f"{active.arithmetic_active_period_return * 100:+.2f} pp", "Portfolio minus benchmark compounded returns; percentage points."),
    ]:
        st.metric(label, value, border=True, help=help_text)
risk_headlines(risk, ctx.load("risk", "risk_active_summary"), ctx.load("performance", "performance_risk_adjusted"))
raw_paths = ctx.load("risk", "risk_drawdown_daily")
paths = growth_frame(raw_paths, ctx.definition)
with st.container(border=True):
    st.subheader("Portfolio and benchmark growth")
    line_chart(paths, "growth", "Growth of 100", height=360)
    st.caption("100 at the preceding-month opening close; the first selected session's return is included.")
    st.page_link("app_pages/performance.py", label="Explore Performance", icon=":material/monitoring:")
with st.container(border=True):
    st.subheader("Sector attribution to return difference (pp)")
    sectors = ctx.load("attribution", "attribution_sector_report")
    sectors = sectors.loc[sectors.linking_method.eq("carino")]
    linked = ctx.load("attribution", "attribution_period_report")
    linked = linked.loc[linked.linking_method.eq("carino")]
    sector_chart(sectors)
    target = linked.iloc[0].get("active_return_pp", active.arithmetic_active_period_return * 100)
    st.caption(f"Carino linking · All sectors shown · Saved target: portfolio minus benchmark compounded returns, {target:+.2f} pp. Teal adds to the difference; orange subtracts from it. Colors indicate effect sign, not portfolio identity.")
    st.page_link("app_pages/attribution.py", label="Explore Attribution", icon=":material/account_tree:")
with st.container(border=True):
    st.subheader("Portfolio and benchmark drawdown")
    line_chart(paths, "drawdown", "Drawdown", percent=True, height=340)
    st.caption("Portfolio and benchmark share the same selected-period starting valuation.")
    st.page_link("app_pages/risk.py", label="Explore Risk", icon=":material/shield:")
st.divider()
render_comparison(ctx, performance, risk, active)
supporting_data(ctx, sectors, linked, paths, raw_paths)
methodology(ctx, "The active strategy holds 20 securities selected from the benchmark universe using the saved sector-momentum/Sortino rules. The custom benchmark selects the Top-100 eligible historical S&P 500 issuers and equal-weights them at quarterly rebalances. Both use hypothetical frictionless same-close implementation; terminal securities follow the saved hypothetical last-close/proportional-reinvestment convention. This is research performance, not live realized trading.")
