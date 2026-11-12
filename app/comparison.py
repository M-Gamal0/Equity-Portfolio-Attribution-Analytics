"""Overview comparison of saved metrics; no financial estimation or reruns."""
import pandas as pd
import streamlit as st

SERIES = ["Active portfolio", "Benchmark"]


def _format(value, kind):
    if value is None or pd.isna(value):
        return "Unavailable"
    if kind == "percent":
        return f"{value:.2%}"
    if kind == "date":
        return pd.Timestamp(value).strftime("%d %b %Y")
    if kind == "sessions":
        return f"{int(value):,}"
    return f"{value:,.3f}" if kind == "ratio" else f"{value:,.2f}"


def _row(label, a, b, kind):
    difference = "—"
    if kind != "date" and pd.notna(a) and pd.notna(b):
        delta = a - b
        difference = (f"{delta * 100:+.2f} pp" if kind == "percent" else
                      f"{int(delta):+,}" if kind == "sessions" else
                      f"{delta:+.3f}" if kind == "ratio" else f"{delta:+,.2f}")
    return {"Metric": label, SERIES[0]: _format(a, kind), SERIES[1]: _format(b, kind), "Difference": difference}


def comparison_frames(performance, risk, ratios):
    """Return grouped tables aligned by portfolio identity, never source row order."""
    perf = performance.set_index("portfolio_id") if "portfolio_id" in performance else performance
    risks = risk.set_index("portfolio_id") if "portfolio_id" in risk else risk

    def values(frame, field, *, eligibility=None):
        found = []
        for owner in SERIES:
            row = frame.loc[owner]
            eligible = eligibility is None or row.get(eligibility) == "CALCULATED"
            found.append(row.get(field, float("nan")) if eligible else float("nan"))
        return found

    def ratio(name):
        found = []
        for prefix in ("portfolio", "benchmark"):
            rows = ratios.loc[ratios.metric_name.eq(f"{prefix}_{name}")]
            row = rows.iloc[0] if not rows.empty else None
            found.append(row.metric_value if row is not None and row.calculation_status == "CALCULATED" else float("nan"))
        return found

    returns = [
        _row("Total return", *values(perf, "period_return"), "percent"),
        _row("CAGR", *values(perf, "annualized_return_cagr", eligibility="cagr_status"), "percent"),
        _row("Annualized arithmetic mean return", *values(perf, "annualized_arithmetic_mean"), "percent"),
        _row("Annualized mean excess return", *ratio("annualized_arithmetic_excess_return"), "percent"),
        _row("Ending value of 100", *[v * 100 for v in values(perf, "wealth_end_normalized")], "number"),
    ]
    risk_fields = [
        ("Annualized volatility", "annualized_volatility", "percent"),
        ("Annualized downside deviation", "annualized_downside_deviation", "percent"),
        ("Maximum drawdown", "maximum_drawdown", "percent"),
        ("Maximum drawdown trough date", "maximum_drawdown_trough_date", "date"),
        ("Drawdown at period end", "ending_drawdown", "percent"),
        ("Longest underwater spell (sessions)", "longest_observed_underwater_duration_sessions", "sessions"),
        ("Underwater spell at end (sessions)", "ending_underwater_duration_sessions", "sessions"),
    ]
    return {
        "Return and growth": pd.DataFrame(returns),
        "Risk and drawdowns": pd.DataFrame([_row(label, *values(risks, field), kind) for label, field, kind in risk_fields]),
        "Risk-adjusted performance": pd.DataFrame([
            _row("Sharpe ratio", *ratio("sharpe_ratio"), "ratio"),
            _row("Sortino ratio", *ratio("sortino_ratio_zero_target"), "ratio"),
        ]),
    }


def render_comparison(ctx, performance, risk, relative):
    ratios = ctx.load("performance", "performance_risk_adjusted")
    active_risk = ctx.load("risk", "risk_active_summary").iloc[0]
    st.subheader("Detailed comparison")
    st.caption("The same complete-month period for both portfolios. Difference is active portfolio minus benchmark; percentage measures use percentage points (pp). A larger difference is not automatically better for risk measures.")
    groups = comparison_frames(performance, risk, ratios)
    for title, frame in groups.items():
        st.markdown(f"**{title}**")
        st.dataframe(frame, hide_index=True, height=(len(frame) + 1) * 36 + 4, row_height=36,
                     column_config={"Metric": st.column_config.TextColumn(width=300),
                                    **{col: st.column_config.TextColumn(width=155, alignment="right") for col in [*SERIES, "Difference"]}},
                     alt=f"Portfolio and benchmark comparison: {title.lower()}")

    unavailable_cagr = performance.loc[performance.cagr_status.ne("CALCULATED")]
    if not unavailable_cagr.empty:
        st.caption("Complete-month windows of at least 12 months are eligible for CAGR. See the saved CAGR status in the supporting data for any unavailable value.")
    st.caption("Mean excess return and Sharpe use the saved risk-free series. Sortino uses a zero daily-return target. Underwater durations count observed trading sessions inside the selection.")

    info_rows = ratios.loc[ratios.metric_name.eq("information_ratio")]
    information = info_rows.iloc[0] if not info_rows.empty else None
    info_value = information.metric_value if information is not None and information.calculation_status == "CALCULATED" else float("nan")
    rows = [
        ("Compounded return difference", relative.get("arithmetic_active_period_return"), "pp"),
        ("Relative wealth return", relative.get("geometric_relative_period_return"), "percent"),
        ("CAGR spread", relative.get("annualized_cagr_spread") if relative.get("cagr_status") == "CALCULATED" else None, "pp"),
        ("Annualized relative wealth return", relative.get("annualized_geometric_relative_return") if relative.get("cagr_status") == "CALCULATED" else None, "percent"),
        ("Annualized tracking error", active_risk.get("tracking_error_annualized") if active_risk.get("tracking_error_status") == "PASS" else None, "percent"),
        ("Information ratio", info_value, "ratio"),
        ("Beta versus benchmark", active_risk.get("benchmark_beta") if active_risk.get("beta_status") == "PASS" else None, "ratio"),
        ("Correlation with benchmark", active_risk.get("benchmark_correlation") if active_risk.get("correlation_status") == "PASS" else None, "ratio"),
    ]
    relative_frame = pd.DataFrame([{"Metric": label, "Value": "Unavailable" if value is None or pd.isna(value) else
                                   f"{value * 100:+.2f} pp" if kind == "pp" else _format(value, kind)} for label, value, kind in rows])
    st.markdown("**Relative to the benchmark**")
    st.dataframe(relative_frame, hide_index=True, height=(len(relative_frame) + 1) * 36 + 4, row_height=36,
                 column_config={"Metric": st.column_config.TextColumn(width=400), "Value": st.column_config.TextColumn(width=180, alignment="right")},
                 alt="Selected-period benchmark-relative return and risk measures")
    st.caption("Relative wealth compares the growth multiples; it differs from subtracting compounded returns. Tracking error, information ratio, beta and correlation describe the portfolio relative to this benchmark and therefore are shown once.")
