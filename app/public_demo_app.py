"""Fixed five-year public demo; reuses existing presentation and saved-result readers."""
import json
import os
from pathlib import Path

import duckdb
import streamlit as st

from portfolio_analytics.demo.view import verify_demo_view
from research_ui import (ROOT, SERIES, COLORS, ResearchContext, metadata, detailed, methodology,
                         growth_frame, line_chart, pct, security_labels, display_table)
from comparison import render_comparison
from overview_details import risk_headlines, sector_chart
from performance_details import return_comparisons, ratio_comparisons, monthly_chart, annual_chart
from attribution_details import (waterfall_chart, component_matrix, security_evidence,
                                 security_ranking, signed_chart)
from risk_details import episode_chart, tail_cards
from exposure_details import (headline_comparison, monthly_sectors, sector_history,
                              closing_positions, holdings_ranking, holdings_chart, tilt_chart)
from factor_details import TARGETS, coefficient_chart, linked_evidence, linked_chart, exposure_chart
from turnover_details import paired_headlines, measured_events, event_scope, event_chart, cost_chart, cost_scenarios

st.set_page_config(page_title="Equity | Public-data demo", page_icon=":material/query_stats:", layout="wide")
st.set_option("client.disableDataExport", True)


@st.cache_data(max_entries=4, show_spinner=False)
def load_demo_context(workspace, pointer_version):
    """Read completed public results only; never fetch sources or run analytics."""
    pointer = Path(workspace) / "streamlit/latest.json"
    view = json.loads(pointer.read_text(encoding="utf-8"))
    settings, definition, checks = verify_demo_view(view)
    catalog, _ = metadata(view["database"], view["build_id"])
    return ResearchContext(view["database"], view["build_id"], view["interval_id"], settings,
                            catalog, definition, checks, view["snapshot_id"])


def paths(ctx):
    return growth_frame(ctx.load("risk", "risk_drawdown_daily"), ctx.definition)


def render_overview(ctx):
    performance = ctx.load("performance", "performance_window").set_index("portfolio_id")
    risk = ctx.load("risk", "risk_absolute_summary").set_index("portfolio_id")
    relative = ctx.load("performance", "performance_relative_window").iloc[0]
    with st.container(horizontal=True):
        st.metric("Active portfolio return", pct(performance.loc[SERIES[0]].period_return), border=True)
        st.metric("Benchmark return", pct(performance.loc[SERIES[1]].period_return), border=True)
        st.metric("Return difference", f"{relative.arithmetic_active_period_return * 100:+.2f} pp", border=True)
    risk_headlines(risk, ctx.load("risk", "risk_active_summary"), ctx.load("performance", "performance_risk_adjusted"))
    st.subheader("Portfolio and benchmark growth")
    line_chart(paths(ctx), "growth", "Growth of 100")
    st.caption("100 at the September 30, 2021 opening close; the first October session's return is included.")
    st.subheader("Sector attribution to return difference (pp)")
    sectors = ctx.load("attribution", "attribution_sector_report")
    sector_chart(sectors.loc[sectors.linking_method.eq("carino")])
    st.caption("Carino linking. Teal adds to the portfolio-minus-benchmark return difference; orange subtracts from it.")
    st.subheader("Portfolio and benchmark drawdown")
    line_chart(paths(ctx), "drawdown", "Drawdown", percent=True)
    st.subheader("Detailed comparison")
    render_comparison(ctx, performance, risk, relative)


def render_performance(ctx):
    summary = ctx.load("performance", "performance_window").set_index("portfolio_id")
    return_comparisons(summary)
    st.subheader("Portfolio and benchmark growth")
    line_chart(paths(ctx), "growth", "Growth of 100")
    periods = ctx.load("performance", "performance_period")
    owner = st.segmented_control("Monthly return series", SERIES, default=SERIES[0]) or SERIES[0]
    st.subheader("Monthly total returns")
    monthly_chart(periods, owner)
    st.subheader("Portfolio and benchmark calendar-year returns")
    annual_chart(periods)
    st.caption("Partial years retain their included months; returns are not annualized.")
    ratio_comparisons(ctx.load("performance", "performance_risk_adjusted"))


def render_attribution(ctx):
    method = st.segmented_control("Linking method", ["carino", "menchero"], default="carino") or "carino"
    report = ctx.load("attribution", "attribution_period_report")
    st.subheader("Allocation, selection and interaction (pp)")
    waterfall_chart(report.loc[report.linking_method.eq(method)].iloc[0], method)
    sectors = ctx.load("attribution", "attribution_sector_report")
    sectors = sectors.loc[sectors.linking_method.eq(method)]
    st.subheader("What drove each sector's effect?")
    component_matrix(sectors)
    source = ctx.load("attribution", "security_contributions_period")
    evidence = security_evidence(security_labels(ctx, source.loc[source.linking_method.eq(method)]))
    ranked, balance = security_ranking(evidence)
    st.subheader("Security contributors and detractors (pp)")
    signed_chart(ranked, "security_label", "active_contribution_pp", "Linked active security contribution (pp)", security=True)
    st.caption("Positive effects add to the return difference; negative effects subtract. Full security rows remain in Detailed results.")
    with st.expander("Ranking balance"):
        display_table(balance, alt="Complete ranked, remaining and total security contribution balances")


def render_risk(ctx):
    summary = ctx.load("risk", "risk_absolute_summary").set_index("portfolio_id")
    risk_headlines(summary, ctx.load("risk", "risk_active_summary"), ctx.load("performance", "performance_risk_adjusted"))
    st.subheader("Portfolio and benchmark drawdown")
    line_chart(paths(ctx), "drawdown", "Drawdown", percent=True)
    st.subheader("Deepest drawdowns and recovery")
    episode_chart(ctx.load("risk", "risk_drawdown_episodes"), ctx.definition)
    st.subheader("Historical one-day tail losses")
    tail_cards(ctx.load("risk", "risk_historical_tail"))
    st.caption("Saved short-tail-sample restrictions remain explicit; unavailable estimates are not filled.")


def render_exposures(ctx):
    summary = ctx.load("exposures", "portfolio_window_summary")
    headline_comparison(summary.loc[summary.weight_basis.eq("eop")])
    owner = st.segmented_control("Holdings portfolio", SERIES, default=SERIES[0]) or SERIES[0]
    _, monthly = monthly_sectors(ctx.history("exposures", "sector_exposures"))
    st.subheader("Sector allocation over time")
    sector_history(monthly, owner)
    st.subheader("Closing sector overweights and underweights")
    tilts = ctx.load("exposures", "sector_tilt_window_summary")
    tilts = tilts.loc[tilts.weight_basis.eq("eop")].copy()
    tilts["active_sector_weight_pp"] = tilts.latest_active_sector_weight * 100
    tilt_chart(tilts)
    st.subheader("Largest closing positions")
    closing = closing_positions(ctx.history("exposures", "security_exposures"), ctx.definition["last_return_date"])
    ranked, balance = holdings_ranking(closing, owner)
    holdings_chart(ranked, owner)
    with st.expander("Complete closing positions and ranking balance"):
        display_table(closing, alt="Full closing positions for both public demo portfolios")
        display_table(balance, alt="Top-ten, remaining and all-position weight balance")


def render_factors(ctx):
    fits = ctx.load("factors", "direct_portfolio_factor_fit")
    target = st.selectbox("Direct-regression target", fits.target_id.unique().tolist(), format_func=lambda x: TARGETS.get(x, x))
    fit = fits.loc[fits.target_id.eq(target)]
    available = len(fit) == 1 and fit.iloc[0].fit_status == "FIT_COMPLETED"
    coefficients = ctx.load("factors", "direct_portfolio_factor_coefficients")
    inference = ctx.load("factors", "direct_portfolio_factor_inference")
    st.subheader("Direct factor coefficients and uncertainty")
    coefficient_chart(coefficients.loc[coefficients.target_id.eq(target)], inference.loc[inference.target_id.eq(target)],
                       ctx.settings["factor_columns"], available)
    linked = ctx.load("factors", "portfolio_factor_linked")
    owner = st.segmented_control("Holdings-based return series", SERIES, default=SERIES[0]) or SERIES[0]
    st.subheader("Holdings-based linked contributions to total return (pp)")
    linked_chart(linked_evidence(linked.loc[linked.portfolio_id.eq(owner)].iloc[0], ctx.settings["factor_columns"]))
    st.caption("Strictly lagged security models; RF, factors, intercept, residual and unmodeled excess remain distinct. These explain total return, not the active-return difference.")
    st.subheader("Modeled holdings coverage over time")
    exposure_chart(ctx.history("factors", "portfolio_factor_exposures"), "modeled_gross_weight_coverage", "Modeled gross-weight coverage", coverage=True)


def render_turnover(ctx):
    paired_headlines(ctx.load("turnover", "portfolio_turnover_summary"))
    events = ctx.load("turnover", "rebalance_turnover_events")
    scope = event_scope(events)
    for row in scope.itertuples():
        st.caption(f"{row.portfolio_id}: {row.measured_rebalances} measured recurring rebalances · {row.initial_establishment_events} initial · {row.pending_events} pending.")
    st.subheader("Measured one-way turnover at each rebalance")
    event_chart(measured_events(events), "one_way_turnover", "portfolio_id", SERIES, COLORS, "One-way NAV turnover")
    st.subheader("Hypothetical trading-cost sensitivity")
    cost_chart(cost_scenarios(ctx.load("turnover", "portfolio_cost_sensitivity")))
    st.caption("Costs use both traded sides and do not alter the saved returns. Final September 30 targets remain pending.")


def main():
    workspace = Path(os.environ.get("PORTFOLIO_DEMO_WORKDIR", str(ROOT / "demo_workspace"))).resolve()
    pointer = workspace / "streamlit/latest.json"
    if not pointer.is_file():
        st.title("Public-data portfolio demo")
        st.info("Prepare the fixed five-year public results once, then reopen this app.")
        st.code("portfolio-analytics-demo build --workdir demo_workspace\nportfolio-analytics-demo prepare-app --workdir demo_workspace", language="bash")
        return
    try:
        ctx = load_demo_context(str(workspace), pointer.stat().st_mtime_ns)
    except (OSError, ValueError, KeyError, RuntimeError, duckdb.Error) as exc:
        st.error(f"The public demo view could not be verified: {exc}")
        return
    st.session_state.research_context = ctx
    sections = {"Overview": render_overview, "Performance": render_performance, "Attribution": render_attribution,
                "Risk": render_risk, "Holdings & exposures": render_exposures, "Factors": render_factors,
                "Turnover & rebalancing": render_turnover}
    with st.sidebar:
        st.caption("PUBLIC DATA / FIXED FIVE-YEAR DEMO")
        selected = st.radio("Analysis", [*sections, "About"], key="public_demo_section")
        st.markdown("**October 2021–September 2026**")
        st.caption("Opening close: September 30, 2021. Fixed 60-month reporting period; no date selection or calculations on rerun.")
    ctx.heading(selected, "A public-data demonstration of the existing six-domain analytics package.")
    st.caption("Hypothetical 20-security active portfolio versus a 50-security benchmark. Both baskets are fixed from current Nasdaq-100 / Yahoo-cap rankings and rebalance to equal weights quarterly. Total returns use frictionless same-close implementation.")
    if selected == "About":
        st.write("This fixed five-year example showcases calculations, attribution and validation. It uses public Yahoo market data and French factors, not the original private research portfolios. Current constituents, caps and sectors are applied retrospectively; GOOG and GOOGL remain separate securities. Historical outperformance is not the objective.")
        st.write("Quarterly targets are 2% per benchmark security and 5% per active security. Weights drift using price returns between rebalances; dividends enter the total-return index under the existing package convention. The September 30, 2026 target selection is pending beyond the cutoff. Sources are acquired during offline preparation, never by this app.")
    else:
        sections[selected](ctx)
        domain = {"Holdings & exposures": "exposures", "Turnover & rebalancing": "turnover"}.get(selected, selected.lower())
        if selected != "Overview":
            detailed(ctx, domain)
    methodology(ctx, "Fixed five-year public demo. Opening valuation: September 30, 2021; first return: October 1, 2021. Current Yahoo sectors remain fixed across history. All saved availability, strict lagging and reconciliation checks are retained. No private research database is read.")


main()
