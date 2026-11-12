"""Explicit immutable run settings; research identities are isolated from math."""

from dataclasses import asdict, dataclass
import json

import pandas as pd


@dataclass(frozen=True)
class AnalysisSettings:
    portfolio_id: str
    benchmark_id: str
    as_of_date: str
    inception_date: str | None = None
    reporting_years: tuple[int, ...] = (3, 5, 10)
    performance_windows: tuple[str, ...] = ("SI", "YTD", "1Y", "3Y", "5Y", "10Y")
    annualization_sessions: int = 252
    variance_ddof: int = 1
    rolling_sessions: tuple[int, ...] = (63, 126, 252)
    covariance_sessions: int = 252
    tail_confidences: tuple[float, ...] = (0.95, 0.99)
    minimum_tail_mass: float = 20.0
    fully_invested: bool = True
    fixed_sectors: bool = True
    factor_columns: tuple[str, ...] = ("mkt_rf", "smb", "hml", "rmw", "cma", "mom")
    factor_model_id: str = "ff5_mom"
    factor_lookback_years: int = 3
    include_security_factor_history: bool = False

    def __post_init__(self):
        if not self.portfolio_id or not self.benchmark_id or self.portfolio_id == self.benchmark_id:
            raise ValueError("Select two distinct, nonempty portfolio identifiers.")
        for name in ("as_of_date", "inception_date"):
            value = getattr(self, name)
            if value is not None:
                date = pd.Timestamp(value)
                if pd.isna(date) or date.tz is not None or date != date.normalize():
                    raise ValueError(f"{name} must be a timezone-free calendar date.")
        if self.inception_date and pd.Timestamp(self.inception_date) >= pd.Timestamp(self.as_of_date):
            raise ValueError("Inception must precede the reporting cutoff.")
        for name in ("reporting_years", "rolling_sessions"):
            values = getattr(self, name)
            if not values or len(set(values)) != len(values) or any(type(v) is not int or v <= 0 for v in values):
                raise ValueError(f"{name} requires unique positive integers.")
        if self.annualization_sessions <= 0 or self.variance_ddof < 0 or self.covariance_sessions <= self.variance_ddof:
            raise ValueError("Invalid annualization, covariance lookback, or variance degrees of freedom.")
        if not self.factor_columns or len(set(self.factor_columns)) != len(self.factor_columns):
            raise ValueError("Factor names must be nonempty and unique.")
        if not self.factor_model_id or self.factor_lookback_years <= 0:
            raise ValueError("Invalid factor model or lookback.")

    @property
    def portfolio_ids(self):
        return (self.portfolio_id, self.benchmark_id)

    def to_json(self):
        return json.dumps(asdict(self), sort_keys=True)


RESEARCH_SNAPSHOT_ID = "bc5353e846ce3e21"


def research_settings(**overrides):
    """Research profile refreshed through September 30, 2026, including factors."""
    values = dict(
        portfolio_id="active20_d0ee4baef89649118003275d6cb71e95",
        benchmark_id="top100ew_f0e95a00662642a8b8c37c7ed6dc13c9",
        as_of_date="2026-09-30", inception_date="2016-09-30",
    )
    values.update(overrides)
    return AnalysisSettings(**values)
