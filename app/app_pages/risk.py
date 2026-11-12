"""Realized interval risk; ending-month estimators remain separately labeled."""
import streamlit as st

from performance_details import paired_card, drawdown_comparisons
from research_ui import SERIES, context, pct, growth_frame, line_chart, detailed, methodology, security_labels, status_text
from risk_details import (rolling_chart, risk_evidence, risk_ranking, risk_bars, reconciliation_caption,
                          tail_cards, episode_chart, supporting_data)

ctx = context()
ctx.heading("Risk", "Understand portfolio and benchmark losses, active risk and historical risk contributions.")
st.caption("Hypothetical 20-security active strategy versus a custom Top-100 eligible-issuer equal-weight benchmark; both rebalance quarterly. Total returns assume frictionless same-close implementation; these are not live realized results.")
summary = ctx.load("risk", "risk_absolute_summary").set_index("portfolio_id")
active = ctx.load("risk", "risk_active_summary").iloc[0]
with st.container(horizontal=True, vertical_alignment="distribute"):
    paired_card("Annualized volatility", [summary.loc[o].annualized_volatility for o in SERIES], width=280)
    paired_card("Annualized downside deviation", [summary.loc[o].annualized_downside_deviation for o in SERIES], width=280)
with st.container(horizontal=True):
    for title, value, status, percent in [
        ("Annualized tracking error", active.tracking_error_annualized, active.tracking_error_status, True),
        ("Benchmark beta", active.benchmark_beta, active.beta_status, False),
        ("Benchmark correlation", active.benchmark_correlation, active.correlation_status, False),
    ]:
        with st.container(border=True, width=220):
            st.metric(title, (pct(value) if percent else f"{value:.3f}") if status == "PASS" else "Unavailable")
            st.caption("Portfolio relative to benchmark · selected period")
            if status != "PASS":
                st.caption(status_text(status))
st.caption("Realized risk uses the selected interval. Downside deviation uses a zero daily-return target. Tracking error is annualized volatility of daily portfolio minus benchmark returns. Risk differences are not automatically favorable or unfavorable.")

drawdown_comparisons(summary, card_width=380)
drawdown = growth_frame(ctx.load("risk", "risk_drawdown_daily"), ctx.definition)
with st.container(border=True):
    st.subheader("Portfolio and benchmark drawdown")
    line_chart(drawdown, "drawdown", "Drawdown", percent=True, height=350)
episodes = ctx.load("risk", "risk_drawdown_episodes")
st.subheader("Deepest drawdowns and recovery")
episode_chart(episodes, ctx.definition)

st.subheader("Sources of realized risk")
basis = st.segmented_control("Contribution basis", ["Total volatility", "Tracking error"], default="Total volatility") or "Total volatility"
owner = st.segmented_control("Risk portfolio", SERIES, default=SERIES[0]) if basis == "Total volatility" else SERIES[0]
owner = owner or SERIES[0]
measure = "volatility" if basis == "Total volatility" else "tracking error"
risk_type = "ABSOLUTE" if basis == "Total volatility" else "ACTIVE_ARITHMETIC"
target = summary.loc[owner].annualized_volatility if basis == "Total volatility" else active.tracking_error_annualized
sectors = ctx.load("risk", "risk_sector_contributions")
sectors = sectors.loc[sectors.risk_type.eq(risk_type) & sectors.portfolio_id.eq(owner)]
sector_evidence = risk_evidence(sectors, "contribution_to_annualized_risk", "risk_share")
if basis == "Total volatility":
    securities = ctx.load("risk", "risk_security_contributions")
    securities = securities.loc[securities.portfolio_id.eq(owner)]
    securities = risk_evidence(securities, "contribution_to_annualized_volatility", "risk_share_of_volatility")
else:
    securities = risk_evidence(ctx.load("risk", "risk_active_contributions"), "contribution_to_annualized_tracking_error", "tracking_error_risk_share")
securities = security_labels(ctx, securities)
st.caption(f"{owner}{' relative to benchmark' if basis == 'Tracking error' else ''} · selected-period Euler contributions to annualized {measure}. Navy positive values increase historical risk; gray negative values reduce it. Risk shares divide each contribution by the saved total; they are distinct from contribution percentage points.")
st.markdown("**Sector contributions · all sectors**")
if reconciliation_caption(sector_evidence, target, measure):
    risk_bars(sector_evidence, "sector", measure)
st.markdown("**Largest security contributions by absolute magnitude**")
balance = securities.iloc[:0]
if reconciliation_caption(securities, target, measure):
    try:
        ranked, balance = risk_ranking(securities)
        risk_bars(ranked, "security_label", measure, security=True)
        for row in balance.itertuples():
            st.caption(f"{row.group}: {row.security_count:,} securities · {row.risk_contribution_pp:+.6f} pp")
    except ValueError as error:
        st.warning(str(error))
st.caption("Ranking retains signed contributions and historical security identities. Latest tickers within the interval label the chart; repeated tickers remain separate securities. All rows, including benchmark-only active contributors, remain in supporting data.")

tails = ctx.load("risk", "risk_historical_tail")
tail_cards(tails)
st.subheader("How realized risk changed")
st.caption("Saved rolling estimates with endpoints inside the selection. Lookbacks can begin before the selected starting month. These are distinct from estimators fitted on the selected interval.")
rolling = ctx.history("risk", "risk_rolling")
tracking = ctx.history("risk", "active_risk_rolling")
lookbacks = sorted(rolling.lookback_sessions.dropna().unique())
lookback = st.selectbox("Rolling lookback (trading sessions)", lookbacks, index=lookbacks.index(252) if 252 in lookbacks else 0)
rolling_chart(rolling, lookback, "annualized_volatility", "status", "Rolling annualized volatility", paired=True)
rolling_chart(tracking, lookback, "tracking_error_annualized", "tracking_error_status", "Rolling annualized tracking error")

supporting_data({
    "Selected-period risk comparison": summary.reset_index(), "Selected-period active risk": ctx.load("risk", "risk_active_summary"),
    "Drawdown observations and opening anchors": drawdown, "All drawdown episodes": episodes,
    "Selected sector contribution basis": sector_evidence, "Selected security contribution basis · all securities": securities,
    "Ranked, remaining and total security contributions": balance, "Historical tail estimates and statuses · all bases": tails,
    "Rolling volatility · all lookbacks and statuses": rolling, "Rolling active risk · all lookbacks and statuses": tracking,
    "Saved security reconciliation": ctx.load("risk", "risk_security_reconciliation"),
    "Saved active-risk reconciliation": ctx.load("risk", "risk_active_reconciliation"),
    "Saved sector reconciliation": ctx.load("risk", "risk_sector_reconciliation"),
})
detailed(ctx, "risk")
methodology(ctx, "Realized selected-period risk and closing-weight covariance risk are distinct. Ending-month estimators retain their 252-session lookback and fixed ending weights. All 24 saved Risk result choices remain available above. Low effective tail samples and insufficient rolling history remain unavailable. Negative Euler contributions are permitted; open drawdown durations are incomplete.")
