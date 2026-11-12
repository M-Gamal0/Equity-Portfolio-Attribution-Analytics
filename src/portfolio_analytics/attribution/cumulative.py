"""Attribution cumulative: validated research calculations."""

from __future__ import annotations

from .carino import carino_scaling_factor
import numpy as np
import pandas as pd



# Research reference: 08_portfolio_attribution.ipynb, code cell index 19.
def calculate_cumulative_brinson_attribution(
    daily_attribution: pd.DataFrame,
    *,
    tolerance: float = 1e-10,
) -> pd.DataFrame:
    """Calculate expanding-window Carino and Menchero attribution."""

    effects = [
        "allocation_effect",
        "selection_effect",
        "interaction_effect",
    ]

    results = []

    for (portfolio_id, benchmark_id), daily in (
        daily_attribution.groupby(
            ["portfolio_id", "benchmark_id"],
            sort=False,
        )
    ):

        daily = daily.sort_values("date").reset_index(drop=True)

        rp = daily["portfolio_return_total"].to_numpy(float)
        rb = daily["benchmark_return_total"].to_numpy(float)
        active = rp - rb

        effect_values = daily[effects].to_numpy(float)

        n = np.arange(1, len(daily) + 1, dtype=float)

        # ----------------------------------------------------
        # 1. Compounded performance through every date
        # ----------------------------------------------------

        log_p = np.cumsum(np.log1p(rp))
        log_b = np.cumsum(np.log1p(rb))

        compounded_p = np.expm1(log_p)
        compounded_b = np.expm1(log_b)

        compounded_active = compounded_p - compounded_b

        # ----------------------------------------------------
        # 2. Carino expanding-window coefficients
        # ----------------------------------------------------

        daily_k = carino_scaling_factor(rp, rb)

        cumulative_K = carino_scaling_factor(
            compounded_p,
            compounded_b,
        )

        carino_effects = (
            np.cumsum(
                effect_values * daily_k[:, None],
                axis=0,
            )
            / cumulative_K[:, None]
        )

        # ----------------------------------------------------
        # 3. Menchero expanding-window coefficients
        # ----------------------------------------------------

        log_difference = log_p - log_b

        ratio = np.ones_like(log_difference)

        unequal = log_difference != 0.0

        ratio[unequal] = (
            np.expm1(log_difference[unequal])
            / (
                n[unequal]
                * np.expm1(
                    log_difference[unequal] / n[unequal]
                )
            )
        )

        M = (
            np.exp(log_b * (n - 1.0) / n)
            * ratio
        )

        cumulative_active = np.cumsum(active)
        cumulative_active_squared = np.cumsum(active**2)

        correction = np.divide(
            compounded_active - M * cumulative_active,
            cumulative_active_squared,
            out=np.zeros_like(compounded_active),
            where=cumulative_active_squared > 0,
        )

        menchero_effects = (
            M[:, None] * np.cumsum(effect_values, axis=0)
            + correction[:, None]
            * np.cumsum(
                active[:, None] * effect_values,
                axis=0,
            )
        )

        # ----------------------------------------------------
        # 4. Standardize both methods
        # ----------------------------------------------------

        for method, linked_effects in [
            ("carino", carino_effects),
            ("menchero", menchero_effects),
        ]:

            result = daily[
                ["portfolio_id", "benchmark_id", "date"]
            ].copy()

            result["linking_method"] = method
            result["observations"] = n.astype(int)

            result["portfolio_compounded_return"] = compounded_p
            result["benchmark_compounded_return"] = compounded_b
            result["compounded_active_return"] = compounded_active

            for index, effect in enumerate(effects):
                result[f"linked_{effect}"] = (
                    linked_effects[:, index]
                )

            result["linked_total_effect"] = (
                linked_effects.sum(axis=1)
            )

            result["reconciliation_error"] = (
                result["linked_total_effect"]
                - result["compounded_active_return"]
            )

            if (
                not np.isfinite(linked_effects).all()
                or result["reconciliation_error"].abs().max()
                > tolerance
            ):
                raise ValueError(
                    f"Cumulative {method} attribution "
                    "failed financial reconciliation."
                )

            results.append(result)

    return (
        pd.concat(results, ignore_index=True)
        .sort_values(
            [
                "portfolio_id",
                "benchmark_id",
                "date",
                "linking_method",
            ]
        )
        .reset_index(drop=True)
    )
