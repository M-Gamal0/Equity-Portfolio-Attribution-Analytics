"""Select explicit entities and cutoffs without changing reporting-window rules."""

import pandas as pd

from portfolio_analytics.config import AnalysisSettings


def select_pair(frame: pd.DataFrame, settings: AnalysisSettings, *, date_column="date"):
    if "portfolio_id" not in frame:
        raise ValueError("Missing portfolio_id.")
    available = set(frame["portfolio_id"].dropna())
    if not set(settings.portfolio_ids).issubset(available):
        raise ValueError("The selected portfolio/benchmark pair is absent from the input.")
    result = frame.loc[frame["portfolio_id"].isin(settings.portfolio_ids)].copy()
    if date_column in result:
        result[date_column] = pd.to_datetime(result[date_column], errors="raise")
        result = result.loc[result[date_column].le(pd.Timestamp(settings.as_of_date))].copy()
    if result.empty or set(result["portfolio_id"].unique()) != set(settings.portfolio_ids):
        raise ValueError("The selected pair has no observations through the requested cutoff.")
    return result


def require_inception(settings):
    if settings.inception_date is None:
        raise ValueError("Supply inception_date explicitly for calendar-window analysis.")
    return pd.Timestamp(settings.inception_date)


def bind(outputs, names, values):
    """Name a documented tuple of DataFrame outputs, checking the interface."""
    if len(names) != len(values):
        raise ValueError("Unexpected domain output count.")
    for name, value in zip(names, values, strict=True):
        if not isinstance(value, pd.DataFrame):
            raise TypeError(f"{name} is not a DataFrame.")
        outputs[name] = value
