"""Presentation-only attribution evidence from saved, method-specific results."""
import json

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from chart_renderer import render_chart
from overview_details import attribution_evidence
from research_ui import display_table, styled


def pp_copy(source, mapping):
    """Use saved pp fields where present; otherwise convert decimal units once."""
    frame = source.copy()
    for original, target in mapping.items():
        if original in frame:
            if target not in frame:
                frame[target] = frame[original] * 100
            frame = frame.drop(columns=original)
    return frame


def security_evidence(source):
    frame = pp_copy(source, {"linked_portfolio_leg": "portfolio_leg_pp", "linked_benchmark_leg": "benchmark_leg_pp",
                             "linked_active_contribution": "active_contribution_pp"})

    def holding_context(row):
        p, b = row.get("portfolio_held_days"), row.get("benchmark_held_days")
        if pd.isna(p) or pd.isna(b):
            return "Holding history unavailable"
        if p == 0 and b > 0:
            return "Benchmark only during selection"
        if p > 0 and b == 0:
            return "Portfolio only during selection"
        return "Observed in both histories" if p > 0 and b > 0 else "Not held during selection"

    frame["holding_context"] = frame.apply(holding_context, axis=1)
    return frame


def security_ranking(frame):
    """Rank a subset, retaining a separately identified remainder and full total."""
    if frame.security_id.duplicated().any() or not np.isfinite(frame.active_contribution_pp).all():
        raise ValueError("Saved security identities or contributions are incomplete; the ranking cannot be reconciled.")
    ordered = frame.sort_values("active_contribution_pp", kind="stable")
    ranked = pd.concat([ordered.loc[ordered.active_contribution_pp.lt(0)].head(5),
                        ordered.loc[ordered.active_contribution_pp.gt(0)].tail(5)]).copy()
    other = frame.loc[~frame.security_id.isin(ranked.security_id)]
    balance = pd.DataFrame({"group": ["Ranked securities", "Other securities", "All securities"],
                            "security_count": [len(ranked), len(other), len(frame)],
                            "active_contribution_pp": [ranked.active_contribution_pp.sum(), other.active_contribution_pp.sum(), frame.active_contribution_pp.sum()]})
    return ranked, balance


def bridge_evidence(sectors):
    fields = ["sector", "brinson_total_pp", "security_active_contribution_pp", "benchmark_reference_bridge_pp",
              "security_count", "coverage_status", "linking_method", "period_start", "period_end"]
    frame = pp_copy(sectors, {"linked_total_effect": "brinson_total_pp", "linked_security_active_contribution": "security_active_contribution_pp",
                              "linked_benchmark_reference_bridge": "benchmark_reference_bridge_pp"})
    return frame[[field for field in fields if field in frame]]


def waterfall_chart(row, method):
    evidence = attribution_evidence(pd.DataFrame([row]), period=True).iloc[0]
    components = dict(zip(["Allocation", "Selection", "Interaction"], [evidence.allocation_pp, evidence.selection_pp, evidence.interaction_pp]))
    target = evidence.active_return_pp
    if not np.isfinite([*components.values(), target]).all() or abs(sum(components.values()) - target) > 1e-6:
        st.error("Saved components do not reconcile to the return-difference target. Inspect the validation evidence.")
        return
    rows, running = [], 0.0
    for component, value in components.items():
        rows.append({"component": component, "start": running, "end": running + value, "effect_pp": value,
                     "kind": "Positive" if value >= 0 else "Negative"})
        running += value
    rows.append({"component": "Return difference", "start": 0, "end": target, "effect_pp": target, "kind": "Total"})
    chart = alt.Chart(pd.DataFrame(rows)).mark_bar(cornerRadius=3).encode(
        x=alt.X("component:N", sort=[*components, "Return difference"], title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("end:Q", title="Attribution to return difference (pp)", axis=alt.Axis(format=".1f")), y2="start:Q",
        color=alt.Color("kind:N", scale=alt.Scale(domain=["Positive", "Negative", "Total"], range=["#0D9488", "#D97706", "#0F2A47"]), legend=None),
        tooltip=[alt.Tooltip("component:N", title="Component"), alt.Tooltip("effect_pp:Q", title="Effect (pp)", format="+.2f")],
    ).properties(height=340)
    render_chart(styled(chart), alt=f"{method.title()} allocation, selection and interaction reconciling to return difference in pp")
    error = evidence.get("reconciliation_error_pp", float("nan"))
    residual = f"{error:+.3e} pp" if pd.notna(error) else "unavailable"
    st.caption(f"{method.title()} linking · Saved linked total {evidence.total_attribution_pp:+.6f} pp · Target {target:+.6f} pp · Saved residual {residual} · Coverage: {row.get('coverage_status', 'unavailable')}.")
    st.caption("Teal adds to the return difference; orange subtracts from it; navy is the total. Colors represent effect sign, rather than portfolio identity. The target subtracts compounded returns; it is distinct from relative wealth.")


def signed_chart(frame, label, value, title, *, security=False):
    plotted = frame.copy()
    category, axis = label, alt.Axis(labelLimit=240)
    tips = [alt.Tooltip(f"{label}:N", title="Ticker (latest within selection)" if security else "Sector"),
            alt.Tooltip(f"{value}:Q", title="Contribution (pp)", format="+.2f")]
    if security:
        category = "category_key"
        plotted[category] = [f"item_{i}" for i in range(len(plotted))]
        labels = dict(zip(plotted[category], plotted.security_label))
        axis = alt.Axis(labelLimit=200, labelExpr=f"({json.dumps(labels)})[datum.label]")
        tips += [alt.Tooltip("holding_context:N", title="Holding context"),
                 alt.Tooltip("portfolio_leg_pp:Q", title="Portfolio leg (pp)", format="+.2f"),
                 alt.Tooltip("benchmark_leg_pp:Q", title="Benchmark leg (pp)", format="+.2f")]
    if plotted.empty:
        st.info("No nonzero saved security contributions are available for this selection.")
        return
    chart = alt.Chart(plotted).mark_bar(cornerRadiusEnd=3).encode(
        y=alt.Y(f"{category}:N", sort="-x", title=None, axis=axis),
        x=alt.X(f"{value}:Q", title=title, axis=alt.Axis(format=".1f", tickCount=6)),
        color=alt.condition(alt.datum[value] >= 0, alt.value("#0D9488"), alt.value("#D97706")), tooltip=tips,
    ).properties(height=max(330, len(plotted) * 30))
    render_chart(styled(chart), alt=title + ("; ranked securities, ticker labels only" if security else "; all sectors"))


def component_matrix(sectors):
    frame = attribution_evidence(sectors)
    components = {"allocation_pp": "Allocation", "selection_pp": "Selection", "interaction_pp": "Interaction", "brinson_total_pp": "Total"}
    long = frame.melt(id_vars=["sector"], value_vars=list(components), var_name="component", value_name="effect_pp")
    long["component"] = long.component.map(components)
    order = frame.sort_values("brinson_total_pp", ascending=False).sector.tolist()
    limit = max(float(long.effect_pp.abs().max()), 0.01)
    axes = dict(x=alt.X("component:N", sort=list(components.values()), title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("sector:N", sort=order, title=None, axis=alt.Axis(labelLimit=240)))
    cells = alt.Chart(long).mark_rect(stroke="white", strokeWidth=3, cornerRadius=3).encode(
        **axes, color=alt.Color("effect_pp:Q", title="Effect (pp)", scale=alt.Scale(domain=[-limit, 0, limit],
        range=["#D97706", "#F7F9FC", "#0D9488"], interpolate="rgb", clamp=True), legend=alt.Legend(format=".1f")),
        tooltip=["sector", "component", alt.Tooltip("effect_pp:Q", title="Effect (pp)", format="+.2f")],
    ).properties(height=max(330, len(frame) * 32))
    labels = alt.Chart(long).mark_text(color="black", fontSize=11).encode(**axes, text=alt.Text("effect_pp:Q", format="+.2f"))
    render_chart(styled(cells + labels), alt="All-sector allocation, selection, interaction and total effects in pp; shared signed color scale")
    st.caption("All cells are pp on one shared scale. Allocation + selection + interaction = total for each sector. Total is a summary column, not a fourth additive component.")


def bridge_chart(sectors):
    frame = bridge_evidence(sectors)
    long = frame.melt(id_vars=["sector", "benchmark_reference_bridge_pp"], value_vars=["brinson_total_pp", "security_active_contribution_pp"],
                      var_name="view", value_name="effect_pp")
    long["view"] = long["view"].map({"brinson_total_pp": "Brinson sector effect", "security_active_contribution_pp": "Security contributions"})
    order = frame.sort_values("brinson_total_pp", ascending=False).sector.tolist()
    chart = alt.Chart(long).mark_bar().encode(
        y=alt.Y("sector:N", title=None, sort=order, axis=alt.Axis(labelLimit=240)), yOffset="view:N",
        x=alt.X("effect_pp:Q", title="Sector effect / security contributions (pp)", axis=alt.Axis(format=".1f", tickCount=6)),
        color=alt.Color("view:N", title=None, scale=alt.Scale(domain=["Brinson sector effect", "Security contributions"], range=["#0F2A47", "#0D9488"])),
        tooltip=["sector", "view", alt.Tooltip("effect_pp:Q", title="Effect (pp)", format="+.2f"),
                 alt.Tooltip("benchmark_reference_bridge_pp:Q", title="Benchmark-reference bridge (pp)", format="+.2f")],
    ).properties(height=max(380, len(frame) * 36))
    render_chart(styled(chart), alt="Brinson sector effects compared with sector-aggregated security contributions; benchmark-reference bridge in pp")
    st.caption("Security contributions = Brinson sector effect + benchmark-reference bridge. The bridge reconciles these two decompositions by sector; it is not an additional portfolio return or a transaction cost. Both full decompositions reconcile to the same overall return difference.")


def method_comparison(report):
    frame = attribution_evidence(report, period=True).set_index("linking_method")
    rows = []
    for title, field in [("Allocation", "allocation_pp"), ("Selection", "selection_pp"),
                         ("Interaction", "interaction_pp"), ("Linked total", "total_attribution_pp")]:
        carino, menchero = frame.loc["carino", field], frame.loc["menchero", field]
        rows.append({"Component": title, "Carino (pp)": carino, "Menchero (pp)": menchero, "Difference (pp)": menchero - carino})
    return pd.DataFrame(rows)


def supporting_data(method, report, sectors, securities, balance, reconciliation):
    component = attribution_evidence(report, period=True)
    conversion = {"linked_security_total": "linked_security_total_pp", "compounded_active_return": "active_return_pp", "linked_brinson_total": "linked_brinson_total_pp",
                  "security_reconciliation_error": "security_reconciliation_error_pp", "brinson_agreement_error": "brinson_agreement_error_pp",
                  "multiplier_reconciliation_error": "multiplier_reconciliation_error_pp"}
    checks = pp_copy(reconciliation.loc[reconciliation.linking_method.eq(method)], conversion)
    st.subheader("Linking-method comparison")
    st.caption("Both methods for the same period. Component allocations can differ; the overall return-difference target remains the same. Difference = Menchero minus Carino.")
    st.dataframe(method_comparison(report), hide_index=True, height=190,
                 column_config={"Component": st.column_config.TextColumn(width=140),
                                **{name: st.column_config.NumberColumn(format="%+.2f pp", width=110) for name in ["Carino (pp)", "Menchero (pp)", "Difference (pp)"]}},
                 alt="Carino and Menchero saved component allocations and total in percentage points")
    frames = {"Linked components and reconciliation (pp)": component.loc[component.linking_method.eq(method)],
              "All sectors and components (pp)": attribution_evidence(sectors),
              "Sector benchmark-reference bridge (pp)": bridge_evidence(sectors),
              "All security contributions and holding context (pp)": securities,
              "Ranked securities and remaining balance (pp)": balance,
              "Saved security reconciliation (pp)": checks}
    st.subheader("Supporting chart data")
    st.caption(f"Selected method: {method.title()}. Every supporting choice below follows it; no source rows are truncated.")
    choice = st.selectbox("Attribution supporting data", list(frames), key="attribution_supporting_data")
    frame = frames[choice]
    display_table(frame, alt=f"{method.title()} attribution supporting data: {choice}")
    st.caption(f"{len(frame):,} {'row' if len(frame) == 1 else 'rows'} available · Effects and residuals are pp; holding days and counts retain their own units.")
