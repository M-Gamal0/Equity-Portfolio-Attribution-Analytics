"""Factors rolling: validated research calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd



# Research reference: 11_fama_french_factor_analytics.ipynb, code cell index 19.
def estimate_rolling_direct_portfolio_ols(
    aligned_df: pd.DataFrame,
    *,
    factor_columns: tuple[str, ...],
    response_mapping: dict[str, str],
    model_id: str,
    lookback_years: int = 3,
    min_observations: int = 700,
    maximum_condition_number: float = 1e10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    required = {
        "date",
        "factor_available",
        *factor_columns,
        *response_mapping.values(),
    }

    if not required.issubset(aligned_df.columns):
        raise ValueError(
            f"Missing rolling inputs: "
            f"{sorted(required - set(aligned_df.columns))}"
        )

    if (
        lookback_years < 1
        or min_observations <= len(factor_columns) + 1
        or not factor_columns
        or len(factor_columns) != len(set(factor_columns))
        or not response_mapping
    ):
        raise ValueError("Invalid rolling regression configuration.")

    working = aligned_df[list(required)].copy()

    working["date"] = pd.to_datetime(
        working["date"], errors="raise"
    ).astype("datetime64[ns]")

    working = working.sort_values("date").reset_index(drop=True)

    if (
        working["date"].isna().any()
        or working["date"].duplicated().any()
        or not working["date"].eq(
            working["date"].dt.normalize()
        ).all()
    ):
        raise ValueError("Invalid rolling estimation calendar.")

    if (
        working["factor_available"].isna().any()
        or not pd.api.types.is_bool_dtype(
            working["factor_available"]
        )
    ):
        raise ValueError("Factor availability must be complete Boolean.")

    available_mask = working[
        "factor_available"
    ].to_numpy(dtype=bool)

    available_positions = np.flatnonzero(available_mask)

    if len(available_positions) == 0:
        raise ValueError("No factor-eligible regression dates.")

    # Reject internal factor gaps. Trailing unavailable dates
    # are retained in the status output.
    first_position = available_positions[0]
    last_position = available_positions[-1]

    if not available_mask[
        first_position:last_position + 1
    ].all():
        raise ValueError(
            "An internal factor gap interrupts rolling history."
        )

    available = (
        working.loc[available_mask]
        .reset_index(drop=True)
    )

    dates = pd.DatetimeIndex(available["date"])
    date_values = dates.to_numpy(dtype="datetime64[ns]")

    n_dates = len(dates)
    p = len(factor_columns) + 1
    target_ids = tuple(response_mapping)

    X = np.column_stack(
        [
            np.ones(n_dates),
            available[
                list(factor_columns)
            ].to_numpy(dtype=np.float64),
        ]
    )

    Y = available[
        list(response_mapping.values())
    ].to_numpy(dtype=np.float64)

    if not (
        np.isfinite(X).all()
        and np.isfinite(Y).all()
    ):
        raise ValueError(
            "Factor-available regression inputs contain "
            "non-finite observations."
        )

    if not np.allclose(
        Y[:, target_ids.index("ACTIVE_RETURN")],
        (
            Y[:, target_ids.index("ACTIVE_PORTFOLIO")]
            - Y[:, target_ids.index("BENCHMARK")]
        ),
        rtol=0,
        atol=1e-12,
    ):
        raise ValueError(
            "Rolling active response does not equal "
            "portfolio minus benchmark excess return."
        )

    # --------------------------------------------------------
    # A. Full calendar lookback eligibility
    # --------------------------------------------------------

    requested_starts = (
        dates - pd.DateOffset(years=lookback_years)
    )

    left = np.searchsorted(
        date_values,
        requested_starts.to_numpy(dtype="datetime64[ns]"),
        side="left",
    )

    right = np.arange(1, n_dates + 1)

    observations = right - left

    # Conservative full-history rule:
    # The source must already exist at the requested
    # three-year calendar anniversary.
    full_history = requested_starts >= dates[0]

    eligible = (
        full_history
        & (observations >= min_observations)
    )

    eligible_positions = np.flatnonzero(eligible)

    if len(eligible_positions) == 0:
        raise ValueError(
            "No complete rolling estimation windows."
        )

    starts = left[eligible]
    ends = right[eligible]
    counts = observations[eligible]

    # --------------------------------------------------------
    # B. Build cumulative sufficient statistics
    # --------------------------------------------------------

    prefix_xx = np.concatenate(
        [
            np.zeros((1, p, p)),
            np.cumsum(
                X[:, :, None] * X[:, None, :],
                axis=0,
            ),
        ],
        axis=0,
    )

    prefix_xy = np.concatenate(
        [
            np.zeros((1, p, len(target_ids))),
            np.cumsum(
                X[:, :, None] * Y[:, None, :],
                axis=0,
            ),
        ],
        axis=0,
    )

    prefix_y2 = np.vstack(
        [
            np.zeros((1, len(target_ids))),
            np.cumsum(Y ** 2, axis=0),
        ]
    )

    prefix_y = np.vstack(
        [
            np.zeros((1, len(target_ids))),
            np.cumsum(Y, axis=0),
        ]
    )

    xx = prefix_xx[ends] - prefix_xx[starts]
    xy = prefix_xy[ends] - prefix_xy[starts]
    yy = prefix_y2[ends] - prefix_y2[starts]
    sy = prefix_y[ends] - prefix_y[starts]

    # --------------------------------------------------------
    # C. Solve batched rolling OLS
    # --------------------------------------------------------

    condition_numbers = np.sqrt(
        np.linalg.cond(xx)
    )

    if (
        not np.isfinite(condition_numbers).all()
        or (condition_numbers > maximum_condition_number).any()
    ):
        raise ValueError(
            "Ill-conditioned rolling factor regression window."
        )

    betas = np.linalg.solve(xx, xy)

    if betas.shape != (
        len(eligible_positions),
        p,
        len(target_ids),
    ):
        raise ValueError(
            "Unexpected batched rolling coefficient dimensions."
        )

    # --------------------------------------------------------
    # D. Calculate exact OLS fit statistics
    # --------------------------------------------------------

    rss_raw = (
        yy - np.sum(betas * xy, axis=1)
    )

    tss = (
        yy - sy ** 2 / counts[:, None]
    )

    if (
        (rss_raw < -1e-10).any()
        or (tss <= 0).any()
    ):
        raise ValueError(
            "Invalid rolling regression residual "
            "or total sum of squares."
        )

    rss = np.maximum(rss_raw, 0.0)

    r_squared = 1.0 - rss / tss

    adjusted_r_squared = (
        1.0
        - (1.0 - r_squared)
        * (counts[:, None] - 1)
        / (counts[:, None] - p)
    )

    residual_se = np.sqrt(
        rss / (counts[:, None] - p)
    )

    if not all(
        np.isfinite(values).all()
        for values in (
            betas,
            r_squared,
            adjusted_r_squared,
            residual_se,
        )
    ):
        raise ValueError("Non-finite rolling OLS statistics.")

    # --------------------------------------------------------
    # E. Assemble completed fit and coefficient outputs
    # --------------------------------------------------------

    eligible_dates = dates[eligible]
    terms = ("intercept", *factor_columns)

    fit_frames = []
    coefficient_frames = []

    for target_index, target_id in enumerate(target_ids):

        fit_frames.append(
            pd.DataFrame(
                {
                    "target_id": target_id,
                    "model_id": model_id,
                    "estimation_date": eligible_dates,
                    "window_requested_start_date": (
                        requested_starts[eligible]
                    ),
                    "window_first_observation_date": (
                        pd.to_datetime(date_values[starts])
                    ),
                    "window_last_observation_date": (
                        eligible_dates
                    ),
                    "lookback_calendar_years": lookback_years,
                    "n_observations": counts,
                    "r_squared": r_squared[:, target_index],
                    "adjusted_r_squared": (
                        adjusted_r_squared[:, target_index]
                    ),
                    "residual_standard_error": (
                        residual_se[:, target_index]
                    ),
                    "condition_number": condition_numbers,
                    "fit_status": "FIT_COMPLETED",
                }
            )
        )

        coefficient_wide = pd.DataFrame(
            betas[:, :, target_index],
            columns=list(terms),
        )

        coefficient_wide.insert(
            0, "estimation_date", eligible_dates
        )

        coefficient_long = coefficient_wide.melt(
            id_vars="estimation_date",
            var_name="term",
            value_name="coefficient",
        )

        coefficient_long["target_id"] = target_id
        coefficient_long["model_id"] = model_id

        coefficient_frames.append(coefficient_long)

    rolling_fit = pd.concat(
        fit_frames, ignore_index=True
    )

    rolling_coefficients = pd.concat(
        coefficient_frames, ignore_index=True
    )

    # --------------------------------------------------------
    # F. Preserve every unavailable source date
    # --------------------------------------------------------

    available_status = pd.DataFrame(
        {
            "estimation_date": dates,
            "n_observations": observations,
            "fit_status": np.where(
                eligible,
                "FIT_COMPLETED",
                "UNAVAILABLE_INSUFFICIENT_3Y_HISTORY",
            ),
        }
    )

    status_base = (
        working[
            ["date", "factor_available"]
        ]
        .rename(columns={"date": "estimation_date"})
        .merge(
            available_status,
            on="estimation_date",
            how="left",
            validate="one_to_one",
        )
    )

    status_base["fit_status"] = (
        status_base["fit_status"].fillna(
            "UNAVAILABLE_NO_CURRENT_FACTOR_RETURN"
        )
    )

    rolling_status = pd.concat(
        [
            status_base.assign(
                target_id=target_id,
                model_id=model_id,
            )
            for target_id in target_ids
        ],
        ignore_index=True,
    )

    return (
        rolling_fit,
        rolling_coefficients,
        rolling_status,
    )
