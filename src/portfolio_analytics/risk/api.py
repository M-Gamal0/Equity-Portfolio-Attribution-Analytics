"""Independent realized and fixed-weight risk application."""

import numpy as np
import pandas as pd

from portfolio_analytics.common.selection import select_pair
from portfolio_analytics.config import AnalysisSettings
from . import (
    inputs,
    absolute,
    active,
    rolling,
    drawdowns,
    tails,
    contributions,
    covariance,
    fixed_weights,
    windows,
)


def run(
    *,
    portfolio_returns: pd.DataFrame,
    portfolio_holdings: pd.DataFrame,
    security_returns: pd.DataFrame,
    settings: AnalysisSettings,
) -> dict[str, pd.DataFrame]:
    """Return risk by calendar window, with separate covariance estimates.

    Missing window history is reported as unavailable; malformed inputs raise.
    """
    returns = select_pair(portfolio_returns, settings)
    holdings = select_pair(portfolio_holdings, settings)
    pair = inputs.align_portfolio_risk_returns(
        returns, settings.portfolio_id, settings.benchmark_id
    )
    cutoff = pd.Timestamp(settings.as_of_date)
    if cutoff not in pd.DatetimeIndex(pair["date"]):
        raise ValueError("Risk cutoff must be an observed return date.")
    ann, ddof = settings.annualization_sessions, settings.variance_ddof
    out = {"risk_daily_returns": pair}
    out["risk_rolling"], out["active_risk_rolling"] = rolling.calculate_rolling_risk(
        pair,
        settings.portfolio_id,
        settings.benchmark_id,
        settings.rolling_sessions,
        ann,
        ddof,
    )
    panels = {"SI": pair}
    status = []
    for years in settings.reporting_years:
        name = f"TRAILING_{years}Y"
        anchor = cutoff - pd.DateOffset(years=years)
        panel = pair.loc[pair["date"].gt(anchor)]
        eligible = (
            len(panel) > max(1, ddof) and 1 <= (panel["date"].min() - anchor).days <= 7
        )
        status.append(
            dict(
                window_id=name,
                anniversary_anchor=anchor,
                as_of_date=cutoff,
                observations=len(panel),
                coverage_status="AVAILABLE"
                if eligible
                else "UNAVAILABLE_INSUFFICIENT_HISTORY",
            )
        )
        if eligible:
            panels[name] = windows.select_calendar_risk_windows(pair, cutoff, (years,))[
                name
            ]["returns"]
    out["risk_reporting_windows"] = pd.DataFrame(status)
    abs_records, active_records, tail_frames = [], [], []
    (
        realized,
        realized_audits,
        active_details,
        active_audits,
        sector_details,
        sector_audits,
    ) = [], [], [], [], [], []
    for window_id, panel in panels.items():
        if len(panel) <= max(1, ddof):
            raise ValueError("At least two return observations are required.")
        meta = dict(
            window_id=window_id, sample_start=panel["date"].min(), as_of_date=cutoff
        )
        active_records.append(
            {
                **meta,
                "portfolio_id": settings.portfolio_id,
                "benchmark_portfolio_id": settings.benchmark_id,
                **active.calculate_active_risk(
                    panel["portfolio_return_total"],
                    panel["benchmark_return_total"],
                    ann,
                    ddof,
                ),
            }
        )
        for portfolio_id, column in (
            (settings.portfolio_id, "portfolio_return_total"),
            (settings.benchmark_id, "benchmark_return_total"),
        ):
            path = drawdowns.calculate_drawdown_path(panel[column], panel["date"])
            abs_records.append(
                {
                    **meta,
                    "portfolio_id": portfolio_id,
                    **absolute.calculate_absolute_risk(panel[column], ann, ddof),
                    **drawdowns.summarize_drawdown_path(path),
                }
            )
            tail_frames.append(
                tails.calculate_historical_tail_risk(
                    panel[column],
                    panel["date"],
                    cutoff,
                    settings.tail_confidences,
                    settings.minimum_tail_mass,
                ).assign(portfolio_id=portfolio_id, window_id=window_id)
            )
            detail, audit = contributions.calculate_realized_risk_contributions(
                holdings, returns, portfolio_id, cutoff, len(panel), ann, ddof
            )
            realized.append(detail.assign(window_id=window_id))
            realized_audits.append({**audit, "window_id": window_id})
        tail_frames.append(
            tails.calculate_historical_tail_risk(
                panel["active_return_arithmetic"],
                panel["date"],
                cutoff,
                settings.tail_confidences,
                settings.minimum_tail_mass,
                return_kind="ARITHMETIC_ACTIVE",
            ).assign(
                portfolio_id=settings.portfolio_id,
                benchmark_portfolio_id=settings.benchmark_id,
                window_id=window_id,
            )
        )
        detail, audit = contributions.calculate_active_risk_contributions(
            holdings,
            pair,
            settings.portfolio_id,
            settings.benchmark_id,
            cutoff,
            len(panel),
            ann,
            ddof,
        )
        active_details.append(detail.assign(window_id=window_id))
        active_audits.append({**audit, "window_id": window_id})
        detail, audit = contributions.calculate_sector_risk_contributions(
            holdings,
            pair,
            settings.portfolio_id,
            settings.benchmark_id,
            cutoff,
            len(panel),
            ann,
            ddof,
        )
        sector_details.append(detail.assign(window_id=window_id))
        sector_audits.append(audit.assign(window_id=window_id))
    out.update(
        risk_absolute_summary=pd.DataFrame(abs_records),
        risk_active_summary=pd.DataFrame(active_records),
        risk_historical_tail=pd.concat(tail_frames, ignore_index=True),
        risk_security_contributions=pd.concat(realized, ignore_index=True),
        risk_security_reconciliation=pd.DataFrame(realized_audits),
        risk_active_contributions=pd.concat(active_details, ignore_index=True),
        risk_active_reconciliation=pd.DataFrame(active_audits),
        risk_sector_contributions=pd.concat(sector_details, ignore_index=True),
        risk_sector_reconciliation=pd.concat(sector_audits, ignore_index=True),
    )
    paths, episodes = [], []
    for portfolio_id, column in (
        (settings.portfolio_id, "portfolio_return_total"),
        (settings.benchmark_id, "benchmark_return_total"),
    ):
        path = drawdowns.calculate_drawdown_path(pair[column], pair["date"])
        paths.append(path.assign(portfolio_id=portfolio_id))
        episodes.append(
            drawdowns.extract_drawdown_episodes(path).assign(portfolio_id=portfolio_id)
        )
    out["risk_drawdown_daily"] = pd.concat(paths, ignore_index=True)
    out["risk_drawdown_episodes"] = pd.concat(episodes, ignore_index=True)
    relative_returns = (1 + pair["portfolio_return_total"]) / (
        1 + pair["benchmark_return_total"]
    ) - 1
    out["risk_relative_drawdown_daily"] = drawdowns.calculate_drawdown_path(
        relative_returns, pair["date"]
    )
    closing = fixed_weights.select_asof_closing_weights(
        holdings, settings.portfolio_ids, cutoff
    )
    out["risk_fixed_weight_holdings"] = closing
    calendar = pd.DatetimeIndex(pair["date"])
    ids = closing["security_id"].unique()
    dates = calendar[-settings.covariance_sessions :]
    security_panel = security_returns.loc[
        security_returns["security_id"].isin(ids) & security_returns["date"].isin(dates)
    ]
    full_panel = (
        len(dates) == settings.covariance_sessions
        and len(security_panel) == len(dates) * len(ids)
        and not security_panel.duplicated(["security_id", "date"]).any()
    )
    if full_panel:
        cov, corr, diagnostics = covariance.estimate_security_covariance(
            security_returns,
            ids,
            calendar,
            cutoff,
            settings.covariance_sessions,
            ann,
            ddof,
        )
        out["risk_covariance_diagnostics"] = pd.DataFrame([diagnostics])
        # Store matrices in a stable long format with explicit security keys.
        rows, cols = np.indices(cov.shape)
        out["risk_security_covariance"] = pd.DataFrame(
            dict(
                security_id=cov.index.to_numpy()[rows.ravel()],
                other_security_id=cov.columns.to_numpy()[cols.ravel()],
                covariance_daily=cov.to_numpy().ravel(),
                correlation=corr.to_numpy().ravel(),
                as_of_date=cutoff,
            )
        )
        weights = closing.pivot(
            index="security_id", columns="portfolio_id", values="weight_eop"
        ).fillna(0)
        details, diagnostics = [], []
        for label, w in [(pid, weights[pid]) for pid in settings.portfolio_ids] + [
            (
                "ACTIVE_DIFFERENCE",
                weights[settings.portfolio_id] - weights[settings.benchmark_id],
            )
        ]:
            detail, diagnostic = fixed_weights.calculate_fixed_weight_risk_components(
                cov, w.reindex(cov.index, fill_value=0), ann
            )
            details.append(detail.assign(weight_vector=label))
            diagnostics.append({**diagnostic, "weight_vector": label})
        out["risk_fixed_weight_components"] = pd.concat(details, ignore_index=True)
        out["risk_fixed_weight_diagnostics"] = pd.DataFrame(diagnostics)
    else:
        out["risk_covariance_diagnostics"] = pd.DataFrame(
            [
                dict(
                    as_of_date=cutoff,
                    coverage_status="UNAVAILABLE_INCOMPLETE_COVARIANCE_HISTORY",
                    lookback_sessions=settings.covariance_sessions,
                )
            ]
        )
    return out
