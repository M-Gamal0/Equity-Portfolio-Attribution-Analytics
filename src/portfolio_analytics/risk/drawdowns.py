"""Risk drawdowns: validated research calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd



# Research reference: 09_risk_analytics.ipynb, code cell index 6.
def calculate_drawdown_path(
    returns: pd.Series | np.ndarray,
    dates: pd.Series | pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Calculate normalized wealth and historical drawdowns.

    Parameters
    ----------
    returns:
        Daily simple decimal total returns.

    dates:
        Matching return-interval end dates in ascending
        chronological order.

    Returns
    -------
    pd.DataFrame
        One observation per return date containing:

            date
            wealth_index
            running_peak_wealth
            drawdown
            underwater_duration_sessions

    Methodology
    -----------
    Initial wealth equals 1 immediately before the first
    observed return interval.

    Wealth[t] = product(1 + return[s]), for s <= t.

    Running peak includes the initial wealth of 1.

    Drawdown[t] = Wealth[t] / RunningPeak[t] - 1.

    Underwater duration counts consecutive observed
    sessions below the previous running wealth peak.

    Missing observations are not filled or dropped.

    The function does not modify its inputs.
    """

    values = np.asarray(returns)

    if values.ndim != 1:
        raise ValueError(
            "Returns must be one-dimensional."
        )

    if (
        not pd.api.types.is_numeric_dtype(values.dtype)
        or pd.api.types.is_bool_dtype(values.dtype)
        or pd.api.types.is_complex_dtype(values.dtype)
    ):
        raise TypeError(
            "Returns must contain real numeric values."
        )

    values = values.astype(
        np.float64,
        copy=False,
    )

    observed_dates = pd.DatetimeIndex(
        pd.to_datetime(dates, errors="raise")
    )

    if len(values) == 0:
        raise ValueError(
            "At least one return observation is required."
        )

    if len(values) != len(observed_dates):
        raise ValueError(
            "Returns and dates must have equal lengths."
        )

    if (
        observed_dates.hasnans
        or not observed_dates.is_unique
        or not observed_dates.is_monotonic_increasing
    ):
        raise ValueError(
            "Dates must be non-null, unique, and increasing."
        )

    if not np.isfinite(values).all():
        raise ValueError(
            "Returns contain missing or non-finite values."
        )

    if np.any(values < -1.0):
        raise ValueError(
            "Simple returns cannot fall below -100%."
        )

    # --------------------------------------------------------
    # Calculate geometrically compounded wealth
    # --------------------------------------------------------

    gross_returns = 1.0 + values

    with np.errstate(
        over="raise",
        divide="raise",
        invalid="raise",
    ):
        wealth_index = np.cumprod(gross_returns)

    if not np.isfinite(wealth_index).all():
        raise ArithmeticError(
            "Wealth path contains non-finite values."
        )

    # --------------------------------------------------------
    # Calculate historical running wealth peaks
    # --------------------------------------------------------

    running_peak = np.maximum.accumulate(
        np.concatenate(
            [
                np.array([1.0]),
                wealth_index,
            ]
        )
    )[1:]

    drawdown = (
        wealth_index / running_peak - 1.0
    )

    # --------------------------------------------------------
    # Calculate consecutive underwater sessions
    # --------------------------------------------------------

    underwater = wealth_index < running_peak

    positions = np.arange(len(values))

    last_peak_position = np.maximum.accumulate(
        np.where(
            ~underwater,
            positions,
            -1,
        )
    )

    underwater_duration = np.where(
        underwater,
        positions - last_peak_position,
        0,
    )

    # --------------------------------------------------------
    # Validate drawdown boundaries
    # --------------------------------------------------------

    if (
        np.any(drawdown > 1e-12)
        or np.any(drawdown < -1.0 - 1e-12)
    ):
        raise ArithmeticError(
            "Calculated drawdowns violate valid boundaries."
        )

    if np.any(underwater_duration < 0):
        raise ArithmeticError(
            "Underwater duration cannot be negative."
        )

    return pd.DataFrame(
        {
            "date": observed_dates,
            "wealth_index": wealth_index,
            "running_peak_wealth": running_peak,
            "drawdown": drawdown,
            "underwater_duration_sessions": (
                underwater_duration
            ),
        }
    )


# Research reference: 09_risk_analytics.ipynb, code cell index 6.
def summarize_drawdown_path(
    drawdown_path: pd.DataFrame,
    drawdown_column: str = "drawdown",
    duration_column: str = "underwater_duration_sessions",
) -> dict:
    """
    Summarize a validated historical drawdown path.

    Returns the maximum observed drawdown, its trough date,
    longest observed consecutive underwater duration,
    ending drawdown, and ending underwater duration.

    A full episode/recovery analysis is not performed here.
    """

    required = {
        "date",
        drawdown_column,
        duration_column,
    }

    missing = required.difference(drawdown_path.columns)

    if missing:
        raise ValueError(
            f"Missing drawdown summary fields: {sorted(missing)}"
        )

    if drawdown_path.empty:
        raise ValueError(
            "Cannot summarize an empty drawdown path."
        )

    maximum_drawdown = float(
        drawdown_path[drawdown_column].min()
    )

    trough_position = (
        drawdown_path[drawdown_column].idxmin()
    )

    trough_date = (
        drawdown_path.loc[trough_position, "date"]
        if maximum_drawdown < 0.0
        else pd.NaT
    )

    return {
        "observations": len(drawdown_path),
        "maximum_drawdown": maximum_drawdown,
        "maximum_drawdown_trough_date": trough_date,
        "longest_observed_underwater_duration_sessions": (
            int(drawdown_path[duration_column].max())
        ),
        "ending_drawdown": float(
            drawdown_path[drawdown_column].iloc[-1]
        ),
        "ending_underwater_duration_sessions": int(
            drawdown_path[duration_column].iloc[-1]
        ),
    }


# Research reference: 09_risk_analytics.ipynb, code cell index 7.
def extract_drawdown_episodes(
    drawdown_path: pd.DataFrame,
    wealth_column: str = "wealth_index",
    peak_column: str = "running_peak_wealth",
    drawdown_column: str = "drawdown",
    duration_column: str = "underwater_duration_sessions",
) -> pd.DataFrame:
    """
    Extract distinct drawdown episodes from a wealth path.

    Parameters
    ----------
    drawdown_path:
        Daily observations containing date, wealth,
        running peak, drawdown, and underwater duration.

    wealth_column, peak_column, drawdown_column,
    duration_column:
        Column mappings supporting both absolute and
        benchmark-relative drawdown paths.

    Returns
    -------
    pd.DataFrame
        One row per distinct drawdown episode.

        Records contain the previous peak reference,
        episode start, trough, last underwater date,
        recovery date, depth, session durations,
        and recovery/censoring status.

    Definitions
    -----------
    An episode starts when wealth first falls below
    the running peak.

    The reference peak is the latest observation
    immediately preceding the underwater episode.
    If the series begins underwater, the reference
    is the initial normalized wealth of 1.

    The trough is the first occurrence of the
    minimum drawdown within the episode.

    An episode recovers when wealth first reaches
    or exceeds its previous running peak.

    Underwater sessions count only observations
    strictly below the running peak. The recovery
    observation is excluded from this duration.

    Unrecovered episodes have missing recovery dates
    and recovery durations, never artificial zeros.

    No forward returns are used to invent recoveries.
    """

    episode_columns = [
        "episode_number",
        "peak_date",
        "peak_reference",
        "episode_start_date",
        "trough_date",
        "last_underwater_date",
        "recovery_date",
        "censor_date",
        "episode_status",
        "peak_wealth",
        "trough_wealth",
        "maximum_drawdown",
        "underwater_sessions",
        "peak_to_trough_sessions",
        "trough_to_recovery_sessions",
        "peak_to_recovery_sessions",
    ]

    required_columns = {
        "date",
        wealth_column,
        peak_column,
        drawdown_column,
        duration_column,
    }

    missing_columns = required_columns.difference(
        drawdown_path.columns
    )

    if missing_columns:
        raise ValueError(
            "Drawdown path missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if drawdown_path.empty:
        raise ValueError(
            "Cannot extract episodes from an empty path."
        )

    # --------------------------------------------------------
    # Validate date and numerical structure
    # --------------------------------------------------------

    dates = pd.DatetimeIndex(
        pd.to_datetime(
            drawdown_path["date"],
            errors="raise",
        )
    )

    if (
        dates.hasnans
        or not dates.is_unique
        or not dates.is_monotonic_increasing
    ):
        raise ValueError(
            "Drawdown dates must be non-null, unique, "
            "and chronologically increasing."
        )

    numeric_columns = [
        wealth_column,
        peak_column,
        drawdown_column,
        duration_column,
    ]

    for column in numeric_columns:

        values = drawdown_path[column]

        if (
            not pd.api.types.is_numeric_dtype(values)
            or pd.api.types.is_bool_dtype(values)
            or pd.api.types.is_complex_dtype(values)
        ):
            raise TypeError(
                f"{column} must contain real numeric values."
            )

    numeric_values = drawdown_path[
        numeric_columns
    ].to_numpy(dtype=np.float64)

    if not np.isfinite(numeric_values).all():
        raise ValueError(
            "Drawdown path contains non-finite values."
        )

    wealth, running_peak, drawdown, recorded_duration = (
        numeric_values.T
    )

    if np.any(wealth < 0.0):
        raise ValueError(
            "Normalized wealth cannot be negative."
        )

    if np.any(running_peak <= 0.0):
        raise ValueError(
            "Running peak wealth must remain positive."
        )

    # --------------------------------------------------------
    # Independently validate running peak and drawdown
    # --------------------------------------------------------

    expected_running_peak = np.maximum.accumulate(
        np.concatenate(
            [
                np.array([1.0]),
                wealth,
            ]
        )
    )[1:]

    if not np.allclose(
        running_peak,
        expected_running_peak,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError(
            "Running peak is inconsistent with the wealth path."
        )

    expected_drawdown = (
        wealth / running_peak - 1.0
    )

    if not np.allclose(
        drawdown,
        expected_drawdown,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError(
            "Drawdown is inconsistent with normalized wealth."
        )

    if (
        np.any(drawdown > 1e-10)
        or np.any(drawdown < -1.0 - 1e-10)
    ):
        raise ValueError(
            "Drawdown values violate financial boundaries."
        )

    # --------------------------------------------------------
    # Validate underwater durations independently
    # --------------------------------------------------------

    underwater = wealth < running_peak

    consecutive_underwater = 0

    for position, is_underwater in enumerate(underwater):

        consecutive_underwater = (
            consecutive_underwater + 1
            if is_underwater
            else 0
        )

        if recorded_duration[position] != (
            consecutive_underwater
        ):
            raise ValueError(
                "Recorded underwater duration is inconsistent "
                f"at observation {position}."
            )

    # --------------------------------------------------------
    # Locate underwater episode starts and ends
    # --------------------------------------------------------

    previous_underwater = np.concatenate(
        [
            np.array([False]),
            underwater[:-1],
        ]
    )

    next_underwater = np.concatenate(
        [
            underwater[1:],
            np.array([False]),
        ]
    )

    episode_starts = np.flatnonzero(
        underwater & ~previous_underwater
    )

    episode_ends = np.flatnonzero(
        underwater & ~next_underwater
    )

    if len(episode_starts) != len(episode_ends):
        raise ArithmeticError(
            "Drawdown episode start/end mismatch."
        )

    # --------------------------------------------------------
    # Extract individual episode characteristics
    # --------------------------------------------------------

    episode_records = []

    for episode_number, (
        start_position,
        end_position,
    ) in enumerate(
        zip(episode_starts, episode_ends),
        start=1,
    ):

        peak_position = start_position - 1

        trough_position = (
            start_position
            + int(
                np.argmin(
                    drawdown[
                        start_position:end_position + 1
                    ]
                )
            )
        )

        recovered = (
            end_position + 1 < len(dates)
        )

        recovery_position = (
            end_position + 1
            if recovered
            else None
        )

        peak_date = (
            dates[peak_position]
            if peak_position >= 0
            else pd.NaT
        )

        peak_reference = (
            "OBSERVED_HIGH"
            if peak_position >= 0
            else "INITIAL_BASELINE"
        )

        peak_wealth = (
            float(running_peak[peak_position])
            if peak_position >= 0
            else 1.0
        )

        recovery_date = (
            dates[recovery_position]
            if recovered
            else pd.NaT
        )

        censor_date = (
            pd.NaT
            if recovered
            else dates[end_position]
        )

        trough_to_recovery = (
            int(recovery_position - trough_position)
            if recovered
            else pd.NA
        )

        peak_to_recovery = (
            int(recovery_position - peak_position)
            if recovered
            else pd.NA
        )

        episode_records.append(
            {
                "episode_number": episode_number,
                "peak_date": peak_date,
                "peak_reference": peak_reference,
                "episode_start_date": dates[start_position],
                "trough_date": dates[trough_position],
                "last_underwater_date": dates[end_position],
                "recovery_date": recovery_date,
                "censor_date": censor_date,
                "episode_status": (
                    "RECOVERED"
                    if recovered
                    else "OPEN_AT_WINDOW_END"
                ),
                "peak_wealth": peak_wealth,
                "trough_wealth": float(
                    wealth[trough_position]
                ),
                "maximum_drawdown": float(
                    drawdown[trough_position]
                ),
                "underwater_sessions": int(
                    end_position - start_position + 1
                ),
                "peak_to_trough_sessions": int(
                    trough_position - peak_position
                ),
                "trough_to_recovery_sessions": (
                    trough_to_recovery
                ),
                "peak_to_recovery_sessions": (
                    peak_to_recovery
                ),
            }
        )

    result = pd.DataFrame.from_records(
        episode_records,
        columns=episode_columns,
    )

    for column in (
        "peak_date",
        "episode_start_date",
        "trough_date",
        "last_underwater_date",
        "recovery_date",
        "censor_date",
    ):
        result[column] = pd.to_datetime(
            result[column]
        ).astype("datetime64[ns]")

    # Preserve nullable integer recovery durations.

    if not result.empty:

        for column in (
            "trough_to_recovery_sessions",
            "peak_to_recovery_sessions",
        ):
            result[column] = result[column].astype(
                "Int64"
            )

    return result
