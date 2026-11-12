"""Small independent financial examples for the promoted calculation kernels."""

import ast
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from portfolio_analytics import (
    performance,
    attribution,
    risk,
    exposures,
    factors,
    turnover,
)
from portfolio_analytics.data import validate_portfolio_returns


def test_all_extracted_functions_retain_notebook_ast():
    """Audit every promoted function against its recorded notebook definition."""
    root = Path(__file__).resolve().parents[1]
    records = json.loads((root / "docs/kernel_provenance.json").read_text())
    assert len(records) == 123
    trees = {}
    for record in records:
        key = (record["domain"], record["module"])
        if key not in trees:
            module = importlib.import_module("portfolio_analytics." + ".".join(key))
            trees[key] = {
                n.name: n
                for n in ast.parse(
                    Path(module.__file__).read_text(encoding="utf-8")
                ).body
                if isinstance(n, ast.FunctionDef)
            }
        digest = hashlib.sha256(
            ast.dump(trees[key][record["function"]]).encode()
        ).hexdigest()
        assert digest == record["function_ast_sha256"], record["function"]


def test_compounding_and_relative_wealth_are_distinct():
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    returns = pd.DataFrame(
        dict(
            portfolio_id=["P"] * 2 + ["B"] * 2,
            date=list(dates) * 2,
            return_total=[0.10, -0.10, 0.05, 0.01],
        )
    )
    before = returns.copy(deep=True)
    wealth = performance.compute_wealth_index(returns)
    assert wealth.loc[wealth.portfolio_id.eq("P"), "wealth_eop"].iloc[
        -1
    ] == pytest.approx(0.99)
    pair = performance.align_performance_pair(
        performance.validate_performance_returns(returns), "P", "B"
    )
    relative = performance.compute_relative_performance(pair)
    numeric = relative.iloc[-1]
    # Identify the documented output explicitly, not a chart-rounded number.
    assert numeric["relative_wealth_index"] == pytest.approx(0.99 / 1.0605)
    pd.testing.assert_frame_equal(returns, before)


@pytest.mark.parametrize(
    "linker",
    [attribution.link_attribution_carino, attribution.link_attribution_menchero],
)
def test_linked_brinson_reconciles_compounded_difference(linker):
    rows = [
        ("2020-01-02", "A", 0.10, 0.05, 0.02, 0.01, 0.0),
        ("2020-01-02", "B", 0.10, 0.05, 0.0, 0.0, 0.02),
        ("2020-01-03", "A", -0.02, 0.01, -0.01, -0.01, 0.0),
        ("2020-01-03", "B", -0.02, 0.01, 0.0, 0.0, -0.01),
    ]
    frame = pd.DataFrame(
        rows,
        columns=[
            "date",
            "sector",
            "portfolio_return_total",
            "benchmark_return_total",
            "allocation_effect",
            "selection_effect",
            "interaction_effect",
        ],
    )
    frame["date"] = pd.to_datetime(frame.date)
    frame["portfolio_id"], frame["benchmark_id"] = "P", "B"
    frame["total_effect"] = frame[
        ["allocation_effect", "selection_effect", "interaction_effect"]
    ].sum(axis=1)
    sectors, summary = linker(frame, portfolio_id="P", benchmark_id="B")
    target = 1.1 * 0.98 - 1.05 * 1.01
    effect_columns = [
        "linked_allocation_effect",
        "linked_selection_effect",
        "linked_interaction_effect",
    ]
    assert len(effect_columns) == 3
    assert sectors[effect_columns].sum().sum() == pytest.approx(target, abs=1e-12)
    assert len(summary) == 1


def test_sample_risk_and_active_risk_identity():
    p, b = np.array([0.01, -0.02, 0.03, 0.015]), np.array([0.005, -0.01, 0.02, 0.01])
    absolute = risk.calculate_absolute_risk(p, 252)
    active = risk.calculate_active_risk(p, b, 252)
    assert absolute["annualized_volatility"] == pytest.approx(
        np.std(p, ddof=1) * np.sqrt(252)
    )
    assert active["tracking_error_annualized"] == pytest.approx(
        np.std(p - b, ddof=1) * np.sqrt(252)
    )
    assert active["active_variance_daily"] == pytest.approx(
        active["active_variance_from_components"]
    )


def test_active_share_and_overlap():
    held = pd.DataFrame(
        dict(
            portfolio_id=["P", "P", "B", "B"],
            date=pd.to_datetime(["2024-01-02"] * 4),
            security_id=["A", "B", "A", "B"],
            sector=["X", "Y", "X", "Y"],
            weight_bop=[0.8, 0.2, 0.5, 0.5],
            weight_eop=[0.8, 0.2, 0.5, 0.5],
        )
    )
    security = exposures.prepare_security_exposures(held, weight_bases=("eop",))
    active = exposures.calculate_active_security_exposures(
        security, portfolio_id="P", benchmark_portfolio_id="B"
    )
    share = exposures.calculate_active_share(active)
    assert share["security_active_share"].iloc[0] == pytest.approx(0.3)
    overlap = exposures.calculate_holdings_overlap(active)
    assert overlap["common_weight_overlap"].iloc[0] == pytest.approx(0.7)


def test_ols_known_coefficients_and_strict_beta_lag():
    rng = np.random.default_rng(418)
    dates = pd.bdate_range("2020-01-01", periods=120)
    x = rng.normal(0, 0.01, (120, 2))
    frame = pd.DataFrame(x, columns=["factor_a", "factor_b"])
    frame["date"], frame["excess_return"] = dates, 0.0003 + x @ np.array([1.2, -0.4])
    _, coeff, _ = factors.fit_ols_factor_regression(
        frame, factor_columns=("factor_a", "factor_b"), model_id="toy"
    )
    np.testing.assert_allclose(
        coeff.set_index("term").loc[
            ["intercept", "factor_a", "factor_b"], "coefficient"
        ],
        [0.0003, 1.2, -0.4],
        atol=1e-12,
    )
    targets = pd.DataFrame(dict(security_id=["A"] * 3, date=dates[:3]))
    estimates = pd.DataFrame(
        dict(security_id=["A"] * 2, estimation_date=dates[:2], beta=[1.0, 2.0])
    )
    result = factors.align_prior_security_factor_coefficients(
        targets, estimates, coefficient_columns=("beta",), model_id="toy"
    )
    assert not result.iloc[0]["model_available"]
    np.testing.assert_allclose(result["beta"].iloc[1:], [1.0, 2.0])


def test_scheduled_turnover_uses_previous_closing_weights():
    dates = pd.to_datetime(["2020-01-02", "2020-03-31", "2020-04-01"])
    rows = []
    for date in dates:
        for security, weight in zip(
            ["A", "B"], ([0.5, 0.5] if date == dates[-1] else [0.6, 0.4])
        ):
            rows.append(
                dict(
                    portfolio_id="P",
                    date=date,
                    security_id=security,
                    weight_bop=weight,
                    weight_eop=weight,
                )
            )
    held = pd.DataFrame(rows)
    targets = pd.DataFrame(
        dict(
            portfolio_id=["P"] * 4,
            security_id=["A", "B"] * 2,
            selection_date=pd.to_datetime(["2019-12-31"] * 2 + ["2020-03-31"] * 2),
            effective_rebalance_date=[dates[0]] * 2 + [dates[-1]] * 2,
            weight_target=[0.6, 0.4, 0.5, 0.5],
        )
    )
    events = turnover.build_rebalance_event_schedule(held, targets)
    _, results = turnover.calculate_scheduled_rebalance_turnover_portable(
        held, targets, events
    )
    measured = results.loc[
        results["selection_date"].eq(pd.Timestamp("2020-03-31"))
    ].iloc[0]
    assert measured["one_way_turnover"] == pytest.approx(0.1)


def test_duplicate_returns_are_rejected():
    frame = pd.DataFrame(
        dict(
            portfolio_id=["P", "P"],
            date=pd.to_datetime(["2020-01-02"] * 2),
            return_total=[0.01, 0.01],
        )
    )
    with pytest.raises(ValueError):
        validate_portfolio_returns(frame)
