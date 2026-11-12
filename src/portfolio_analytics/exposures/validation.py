"""Exposures validation: validated research calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd



# Research reference: 10_holdings_exposure_analytics.ipynb, code cell index 12.
def validate_integrated_exposure_outputs(
    security_exposures: pd.DataFrame,
    sector_exposures: pd.DataFrame,
    active_security_exposures: pd.DataFrame,
    active_sector_exposures: pd.DataFrame,
    active_share: pd.DataFrame,
    security_concentration: pd.DataFrame,
    sector_concentration: pd.DataFrame,
    holdings_overlap: pd.DataFrame,
    *,
    weight_tolerance: float = 1e-10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Audit independently calculated exposure output contracts.

    Assumptions:
        Fully invested, long-only portfolios.

    Units:
        Decimal portfolio weights.

    Date convention:
        Daily return-interval end date.

    Returns:
        portfolio_state_audit:
            One row per portfolio_id, date, weight_basis.

        active_pair_audit:
            One row per portfolio_id,
            benchmark_portfolio_id, date, weight_basis.

    The function does not modify input DataFrames.
    """

    if (
        not np.isfinite(weight_tolerance)
        or not 0 < weight_tolerance < 1
    ):
        raise ValueError(
            "weight_tolerance must lie strictly between "
            "zero and one."
        )

    portfolio_keys = [
        "portfolio_id",
        "date",
        "weight_basis",
    ]

    pair_keys = [
        "portfolio_id",
        "benchmark_portfolio_id",
        "date",
        "weight_basis",
    ]

    # --------------------------------------------------------
    # A. Declare input contracts
    # --------------------------------------------------------

    contracts = {
        "security_exposures": (
            security_exposures,
            portfolio_keys + ["security_id"],
            ["security_weight"],
        ),
        "sector_exposures": (
            sector_exposures,
            portfolio_keys + ["sector"],
            ["sector_weight"],
        ),
        "active_security_exposures": (
            active_security_exposures,
            pair_keys + ["security_id"],
            [
                "portfolio_weight",
                "benchmark_weight",
                "active_weight",
            ],
        ),
        "active_sector_exposures": (
            active_sector_exposures,
            pair_keys + ["sector"],
            ["active_sector_weight"],
        ),
        "active_share": (
            active_share,
            pair_keys,
            [
                "security_active_share",
                "sector_active_share",
                "common_weight_overlap",
                "security_minus_sector_active_share",
            ],
        ),
        "security_concentration": (
            security_concentration,
            portfolio_keys,
            [
                "capital_hhi",
                "security_hhi",
                "effective_holdings",
                "invested_weight_sum",
                "invested_security_count",
            ],
        ),
        "sector_concentration": (
            sector_concentration,
            portfolio_keys,
            [
                "capital_sector_hhi",
                "sector_hhi",
                "effective_sectors",
                "invested_weight_sum",
                "invested_sector_count",
            ],
        ),
        "holdings_overlap": (
            holdings_overlap,
            pair_keys,
            [
                "portfolio_security_count",
                "benchmark_security_count",
                "common_weight_overlap",
                "security_active_share_from_weights",
                "jaccard_constituent_overlap",
                "shared_security_count",
                "union_security_count",
            ],
        ),
    }

    # --------------------------------------------------------
    # B. Validate source/output contracts
    # --------------------------------------------------------

    for name, (frame, keys, values) in contracts.items():

        if (
            not isinstance(frame, pd.DataFrame)
            or frame.empty
        ):
            raise ValueError(
                f"{name}: expected a nonempty DataFrame."
            )

        missing = (
            set(keys) | set(values)
        ) - set(frame.columns)

        if missing:
            raise ValueError(
                f"{name}: missing fields {sorted(missing)}."
            )

        if frame[keys].isna().any().any():
            raise ValueError(
                f"{name}: null output keys."
            )

        if frame.duplicated(keys).any():
            raise ValueError(
                f"{name}: duplicate output keys."
            )

        dates = pd.to_datetime(
            frame["date"],
            errors="raise",
        )

        if dates.isna().any():
            raise ValueError(
                f"{name}: invalid dates."
            )

        if not frame["weight_basis"].isin(
            ["bop", "eop"]
        ).all():
            raise ValueError(
                f"{name}: unsupported weight basis."
            )

        for column in values:

            if not pd.api.types.is_numeric_dtype(
                frame[column]
            ):
                raise TypeError(
                    f"{name}.{column}: must be numeric."
                )

            numeric_values = frame[column].to_numpy(
                dtype=np.float64,
                na_value=np.nan,
            )

            if not np.isfinite(numeric_values).all():
                raise ValueError(
                    f"{name}.{column}: nonfinite values."
                )

    # --------------------------------------------------------
    # C. Define strict one-to-one coverage reconciliation
    # --------------------------------------------------------

    def exact_merge(
        left: pd.DataFrame,
        right: pd.DataFrame,
        keys: list[str],
        name: str,
    ) -> pd.DataFrame:
        """Reject missing or surplus output observations."""

        result = left.merge(
            right,
            on=keys,
            how="outer",
            validate="one_to_one",
            indicator=True,
        )

        if not result["_merge"].eq("both").all():
            raise ValueError(
                f"{name}: output coverage mismatch."
            )

        return result.drop(columns="_merge")

    # --------------------------------------------------------
    # D. Independently reconstruct security-level quantities
    # --------------------------------------------------------

    security_source = (
        security_exposures
        .assign(
            _weight_square=lambda df: (
                df["security_weight"] ** 2
            ),
            _invested=lambda df: (
                df["security_weight"].gt(0)
            ),
        )
        .groupby(
            portfolio_keys,
            as_index=False,
        )
        .agg(
            security_source_total=(
                "security_weight", "sum"
            ),
            security_source_capital_hhi=(
                "_weight_square", "sum"
            ),
            security_source_count=(
                "_invested", "sum"
            ),
        )
    )

    # --------------------------------------------------------
    # E. Independently reconstruct sector-level quantities
    # --------------------------------------------------------

    sector_source = (
        sector_exposures
        .assign(
            _weight_square=lambda df: (
                df["sector_weight"] ** 2
            ),
            _invested=lambda df: (
                df["sector_weight"].gt(0)
            ),
        )
        .groupby(
            portfolio_keys,
            as_index=False,
        )
        .agg(
            sector_source_total=(
                "sector_weight", "sum"
            ),
            sector_source_capital_hhi=(
                "_weight_square", "sum"
            ),
            sector_source_count=(
                "_invested", "sum"
            ),
        )
    )

    state = exact_merge(
        security_source,
        sector_source,
        portfolio_keys,
        "Security/sector sources",
    )

    # --------------------------------------------------------
    # F. Attach independently calculated concentration outputs
    # --------------------------------------------------------

    security_report = (
        security_concentration[
            portfolio_keys + [
                "invested_weight_sum",
                "capital_hhi",
                "security_hhi",
                "effective_holdings",
                "invested_security_count",
            ]
        ]
        .rename(
            columns={
                "invested_weight_sum": (
                    "reported_security_total"
                ),
                "capital_hhi": (
                    "reported_security_capital_hhi"
                ),
                "security_hhi": (
                    "reported_security_hhi"
                ),
                "invested_security_count": (
                    "reported_security_count"
                ),
            }
        )
    )

    sector_report = (
        sector_concentration[
            portfolio_keys + [
                "invested_weight_sum",
                "capital_sector_hhi",
                "sector_hhi",
                "effective_sectors",
                "invested_sector_count",
            ]
        ]
        .rename(
            columns={
                "invested_weight_sum": (
                    "reported_sector_total"
                ),
                "capital_sector_hhi": (
                    "reported_sector_capital_hhi"
                ),
                "sector_hhi": (
                    "reported_sector_hhi"
                ),
                "invested_sector_count": (
                    "reported_sector_count"
                ),
            }
        )
    )

    state = exact_merge(
        state,
        security_report,
        portfolio_keys,
        "Security concentration",
    )

    state = exact_merge(
        state,
        sector_report,
        portfolio_keys,
        "Sector concentration",
    )

    # --------------------------------------------------------
    # G. Define portfolio-state reconciliation identities
    # --------------------------------------------------------

    state_checks = {
        "source_weight_error": (
            state["security_source_total"]
            - state["sector_source_total"]
        ).abs(),

        "security_concentration_weight_error": (
            state["reported_security_total"]
            - state["security_source_total"]
        ).abs(),

        "sector_concentration_weight_error": (
            state["reported_sector_total"]
            - state["sector_source_total"]
        ).abs(),

        "security_capital_hhi_error": (
            state["reported_security_capital_hhi"]
            - state["security_source_capital_hhi"]
        ).abs(),

        "sector_capital_hhi_error": (
            state["reported_sector_capital_hhi"]
            - state["sector_source_capital_hhi"]
        ).abs(),

        "security_effective_count_error": (
            state["reported_security_hhi"]
            * state["effective_holdings"]
            - 1.0
        ).abs(),

        "sector_effective_count_error": (
            state["reported_sector_hhi"]
            * state["effective_sectors"]
            - 1.0
        ).abs(),

        "security_normalized_hhi_error": (
            state["reported_security_hhi"]
            * state["reported_security_total"] ** 2
            - state["reported_security_capital_hhi"]
        ).abs(),

        "sector_normalized_hhi_error": (
            state["reported_sector_hhi"]
            * state["reported_sector_total"] ** 2
            - state["reported_sector_capital_hhi"]
        ).abs(),

        "security_count_error": (
            state["reported_security_count"]
            - state["security_source_count"]
        ).abs(),

        "sector_count_error": (
            state["reported_sector_count"]
            - state["sector_source_count"]
        ).abs(),

        "security_investment_error": (
            state["security_source_total"] - 1.0
        ).abs(),

        "sector_investment_error": (
            state["sector_source_total"] - 1.0
        ).abs(),

        "sector_hhi_hierarchy_excess": (
            state["reported_security_hhi"]
            - state["reported_sector_hhi"]
        ).clip(lower=0),

        "effective_count_hierarchy_excess": (
            state["effective_sectors"]
            - state["effective_holdings"]
        ).clip(lower=0),
    }

    state_check_columns = list(state_checks)

    for name, residual in state_checks.items():
        state[name] = residual

    state["max_abs_residual"] = (
        state[state_check_columns].max(axis=1)
    )

    if (
        not np.isfinite(
            state[state_check_columns].to_numpy(
                dtype=float
            )
        ).all()
        or (
            state["max_abs_residual"]
            > weight_tolerance
        ).any()
    ):

        failed = state.loc[
            state["max_abs_residual"].gt(
                weight_tolerance
            ),
            portfolio_keys + ["max_abs_residual"],
        ].head(5)

        raise ValueError(
            "Integrated portfolio-state reconciliation "
            "failed:\n"
            + failed.to_string(index=False)
        )

    state["audit_status"] = "PASSED"

    # --------------------------------------------------------
    # H. Independently reconstruct active-security metrics
    # --------------------------------------------------------

    pair_security = (
        active_security_exposures
        .assign(
            _abs_active=lambda df: (
                df["active_weight"].abs()
            ),
            _common_weight=lambda df: np.minimum(
                df["portfolio_weight"],
                df["benchmark_weight"],
            ),
        )
        .groupby(
            pair_keys,
            as_index=False,
        )
        .agg(
            portfolio_weight_from_active=(
                "portfolio_weight", "sum"
            ),
            benchmark_weight_from_active=(
                "benchmark_weight", "sum"
            ),
            security_net_active_weight=(
                "active_weight", "sum"
            ),
            security_half_l1=(
                "_abs_active", "sum"
            ),
            common_weight_from_active=(
                "_common_weight", "sum"
            ),
        )
    )

    pair_security["security_half_l1"] *= 0.5

    # --------------------------------------------------------
    # I. Independently reconstruct active-sector metrics
    # --------------------------------------------------------

    pair_sector = (
        active_sector_exposures
        .assign(
            _abs_active_sector=lambda df: (
                df["active_sector_weight"].abs()
            )
        )
        .groupby(
            pair_keys,
            as_index=False,
        )
        .agg(
            sector_net_active_weight=(
                "active_sector_weight", "sum"
            ),
            sector_half_l1=(
                "_abs_active_sector", "sum"
            ),
        )
    )

    pair_sector["sector_half_l1"] *= 0.5

    # --------------------------------------------------------
    # J. Attach independently calculated pair-level outputs
    # --------------------------------------------------------

    pair = exact_merge(
        pair_security,
        pair_sector,
        pair_keys,
        "Active security/sector",
    )

    pair = exact_merge(
        pair,
        active_share[
            pair_keys + [
                "security_active_share",
                "sector_active_share",
                "common_weight_overlap",
                "security_minus_sector_active_share",
            ]
        ],
        pair_keys,
        "Active Share",
    )

    pair = exact_merge(
        pair,
        holdings_overlap[
            pair_keys + [
                "portfolio_security_count",
                "benchmark_security_count",
                "common_weight_overlap",
                "security_active_share_from_weights",
                "jaccard_constituent_overlap",
                "shared_security_count",
                "union_security_count",
            ]
        ].rename(
            columns={
                "common_weight_overlap": (
                    "common_weight_overlap_from_holdings"
                )
            }
        ),
        pair_keys,
        "Holdings overlap",
    )

    # --------------------------------------------------------
    # K. Link active and benchmark portfolio-state references
    # --------------------------------------------------------

    state_columns = [
        "security_source_total",
        "sector_source_total",
        "reported_security_count",
        "reported_sector_count",
        "reported_security_hhi",
        "reported_sector_hhi",
    ]

    for role, id_column in (
        ("portfolio", "portfolio_id"),
        ("benchmark", "benchmark_portfolio_id"),
    ):

        lookup = (
            state[
                portfolio_keys + state_columns
            ]
            .rename(
                columns={
                    "portfolio_id": id_column,
                    **{
                        name: f"{name}_{role}"
                        for name in state_columns
                    },
                }
            )
        )

        pair = pair.merge(
            lookup,
            on=[
                id_column,
                "date",
                "weight_basis",
            ],
            how="left",
            validate="many_to_one",
            indicator="_join_state",
        )

        if not pair["_join_state"].eq("both").all():
            raise ValueError(
                f"Missing {role} state for an "
                "active-pair observation."
            )

        pair = pair.drop(
            columns="_join_state"
        )

    # --------------------------------------------------------
    # L. Define active-pair reconciliation identities
    # --------------------------------------------------------

    pair_checks = {
        "portfolio_weight_alignment_error": (
            pair["portfolio_weight_from_active"]
            - pair["security_source_total_portfolio"]
        ).abs(),

        "benchmark_weight_alignment_error": (
            pair["benchmark_weight_from_active"]
            - pair["security_source_total_benchmark"]
        ).abs(),

        "security_active_weight_neutrality_error": (
            pair["security_net_active_weight"]
        ).abs(),

        "sector_active_weight_neutrality_error": (
            pair["sector_net_active_weight"]
        ).abs(),

        "security_active_share_error": (
            pair["security_active_share"]
            - pair["security_half_l1"]
        ).abs(),

        "sector_active_share_error": (
            pair["sector_active_share"]
            - pair["sector_half_l1"]
        ).abs(),

        "common_weight_source_error": (
            pair["common_weight_overlap"]
            - pair["common_weight_from_active"]
        ).abs(),

        "common_weight_overlap_error": (
            pair["common_weight_overlap"]
            - pair["common_weight_overlap_from_holdings"]
        ).abs(),

        "overlap_active_share_error": (
            pair["security_active_share"]
            - pair["security_active_share_from_weights"]
        ).abs(),

        "active_share_overlap_identity_error": (
            pair["security_active_share"]
            + pair["common_weight_overlap"]
            - 1.0
        ).abs(),

        "share_difference_error": (
            pair["security_minus_sector_active_share"]
            - (
                pair["security_active_share"]
                - pair["sector_active_share"]
            )
        ).abs(),

        "portfolio_count_overlap_error": (
            pair["portfolio_security_count"]
            - pair["reported_security_count_portfolio"]
        ).abs(),

        "benchmark_count_overlap_error": (
            pair["benchmark_security_count"]
            - pair["reported_security_count_benchmark"]
        ).abs(),

        "jaccard_identity_error": (
            pair["jaccard_constituent_overlap"]
            - (
                pair["shared_security_count"]
                / pair["union_security_count"]
            )
        ).abs(),

        "active_share_hierarchy_excess": (
            pair["sector_active_share"]
            - pair["security_active_share"]
        ).clip(lower=0),
    }

    pair_check_columns = list(pair_checks)

    for name, residual in pair_checks.items():
        pair[name] = residual

    pair["max_abs_residual"] = (
        pair[pair_check_columns].max(axis=1)
    )

    if (
        not np.isfinite(
            pair[pair_check_columns].to_numpy(
                dtype=float
            )
        ).all()
        or (
            pair["max_abs_residual"]
            > weight_tolerance
        ).any()
    ):

        failed = pair.loc[
            pair["max_abs_residual"].gt(
                weight_tolerance
            ),
            pair_keys + ["max_abs_residual"],
        ].head(5)

        raise ValueError(
            "Integrated active-pair reconciliation "
            "failed:\n"
            + failed.to_string(index=False)
        )

    pair["audit_status"] = "PASSED"

    return state, pair


# Research reference: 10_holdings_exposure_analytics.ipynb, code cell index 13.
def validate_exposure_output_contract(
    dataframe: pd.DataFrame,
    *,
    output_name: str,
    primary_keys: list[str],
    group_keys: list[str],
    expected_groups: pd.MultiIndex,
    numeric_columns: list[str],
) -> dict:
    """
    Validate one standardized exposure output contract.

    Checks:
        Nonempty DataFrame.
        Required fields.
        Nonmissing primary keys.
        Unique primary keys.
        Valid dates and weight bases.
        Finite required numerical measures.
        Exact expected portfolio/date/basis coverage.

    The expected_groups argument is supplied explicitly,
    keeping the validator independent of research run IDs.
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            f"{output_name}: expected a pandas DataFrame."
        )

    if dataframe.empty:
        raise ValueError(
            f"{output_name}: output is empty."
        )

    required_columns = (
        set(primary_keys)
        | set(group_keys)
        | set(numeric_columns)
    )

    missing_columns = (
        required_columns - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{output_name}: missing columns "
            f"{sorted(missing_columns)}."
        )

    if dataframe[primary_keys].isna().any().any():
        raise ValueError(
            f"{output_name}: null primary keys."
        )

    if dataframe.duplicated(
        subset=primary_keys
    ).any():
        raise ValueError(
            f"{output_name}: duplicate primary keys."
        )

    dates = pd.to_datetime(
        dataframe["date"],
        errors="raise",
    )

    if dates.isna().any():
        raise ValueError(
            f"{output_name}: invalid dates."
        )

    if not dataframe["weight_basis"].isin(
        ["bop", "eop"]
    ).all():
        raise ValueError(
            f"{output_name}: unsupported weight basis."
        )

    for column in numeric_columns:

        if not pd.api.types.is_numeric_dtype(
            dataframe[column]
        ):
            raise TypeError(
                f"{output_name}.{column}: expected numeric."
            )

        values = dataframe[column].to_numpy(
            dtype=np.float64,
            na_value=np.nan,
        )

        if not np.isfinite(values).all():
            raise ValueError(
                f"{output_name}.{column}: nonfinite values."
            )

    observed_groups = pd.MultiIndex.from_frame(
        dataframe[group_keys].drop_duplicates()
    )

    missing_groups = expected_groups.difference(
        observed_groups
    )

    unexpected_groups = observed_groups.difference(
        expected_groups
    )

    if (
        len(missing_groups) > 0
        or len(unexpected_groups) > 0
    ):
        raise ValueError(
            f"{output_name}: observation coverage mismatch. "
            f"Missing groups: {len(missing_groups)}; "
            f"unexpected groups: {len(unexpected_groups)}."
        )

    return {
        "output_name": output_name,
        "rows": len(dataframe),
        "observed_groups": len(observed_groups),
        "expected_groups": len(expected_groups),
        "numeric_fields_checked": len(numeric_columns),
        "contract_status": "PASSED",
    }


# Research reference: 10_holdings_exposure_analytics.ipynb, code cell index 13.
def validate_ranking_contract(
    dataframe: pd.DataFrame,
    *,
    group_keys: list[str],
    maximum_rank: int,
) -> None:
    """
    Validate sequential ranking positions within each group.

    Every observed ranking group must have integer ranks
    starting at one and increasing without gaps.

    Groups with no ranked positions are permitted to be
    absent, subject to their parent output contract.
    """

    if "rank" not in dataframe.columns:
        raise ValueError(
            "Ranking output is missing the rank column."
        )

    ranks = dataframe["rank"].to_numpy(
        dtype=np.float64,
        na_value=np.nan,
    )

    if (
        not np.isfinite(ranks).all()
        or (ranks < 1).any()
        or (ranks > maximum_rank).any()
        or not np.equal(ranks, np.floor(ranks)).all()
    ):
        raise ValueError(
            "Ranking output contains invalid ranking positions."
        )

    ranking_groups = (
        dataframe
        .groupby(group_keys)["rank"]
        .agg(["size", "min", "max"])
    )

    valid_sequences = (
        ranking_groups["min"].eq(1)
        & ranking_groups["max"].eq(
            ranking_groups["size"]
        )
    )

    if not valid_sequences.all():
        raise ValueError(
            "Ranking positions are not sequential."
        )


# Research reference: 10_holdings_exposure_analytics.ipynb, code cell index 13.
def validate_top_k_contract(
    dataframe: pd.DataFrame,
    *,
    group_keys: list[str],
    configured_k: tuple[int, ...],
) -> None:
    """Verify each observation contains every configured K."""

    observed_k = set(dataframe["k"].unique())

    if observed_k != set(configured_k):
        raise ValueError(
            "Top-K output differs from configured K values."
        )

    counts = (
        dataframe.groupby(group_keys)["k"].nunique()
    )

    if not counts.eq(len(configured_k)).all():
        raise ValueError(
            "Incomplete Top-K coverage within an observation."
        )


# Research reference: 10_holdings_exposure_analytics.ipynb, code cell index 13.
def validate_exposure_change_lineage(
    dataframe: pd.DataFrame,
    *,
    approved_dates: pd.DatetimeIndex,
) -> None:
    """
    Verify that each previous_date is the immediately
    preceding observation in the approved trading calendar.
    """

    required = {
        "date",
        "previous_date",
    }

    if not required.issubset(dataframe.columns):
        raise ValueError(
            "Exposure changes lack required date lineage."
        )

    ordered_dates = (
        pd.DatetimeIndex(approved_dates)
        .sort_values()
        .unique()
    )

    date_reference = pd.DataFrame(
        {
            "date": ordered_dates[1:],
            "expected_previous_date": ordered_dates[:-1],
        }
    )

    observed_pairs = (
        dataframe[
            [
                "date",
                "previous_date",
            ]
        ]
        .drop_duplicates()
        .copy()
    )

    observed_pairs["date"] = pd.to_datetime(
        observed_pairs["date"],
        errors="raise",
    )

    observed_pairs["previous_date"] = pd.to_datetime(
        observed_pairs["previous_date"],
        errors="raise",
    )

    checked = observed_pairs.merge(
        date_reference,
        on="date",
        how="left",
        validate="many_to_one",
    )

    if (
        checked["previous_date"].isna().any()
        or checked["expected_previous_date"].isna().any()
        or not checked["previous_date"].eq(
            checked["expected_previous_date"]
        ).all()
    ):
        raise ValueError(
            "Exposure-change previous-date mapping is invalid."
        )
