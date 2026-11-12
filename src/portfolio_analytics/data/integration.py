"""Data integration: validated research calculations."""

from __future__ import annotations

from .contracts import validate_factor_returns
from .contracts import validate_portfolio_holdings
from .contracts import validate_portfolio_metadata
from .contracts import validate_portfolio_returns
from .contracts import validate_rebalance_targets
from .contracts import validate_security_returns
import numpy as np
import pandas as pd



# Research reference: 06_standardized_analytics_data_layer.ipynb, code cell index 18.
def validate_analytics_input_bundle(
    returns_df: pd.DataFrame,
    holdings_df: pd.DataFrame,
    security_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    factors_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    factor_columns: tuple[str, ...],
    *,
    fully_invested: bool = False,
    require_aligned_calendars: bool = False,
    tolerance: float = 1e-10,
) -> dict:
    """
    Validate consistency across six standardized DataFrames.

    All inputs are source-independent pandas DataFrames.

    Optional-field policies:
        - Missing portfolio return_price:
          Skip portfolio-level price reconciliation.
          Total-return reconciliation remains mandatory.

        - Missing effective_rebalance_date:
          Skip target implementation reconciliation.
          Do not infer implementation or pending status.

        - Present effective_rebalance_date:
          Reconcile dated targets to realized BOP holdings.
          Undated targets remain outside that comparison.

    A skipped check is explicitly reported as NOT_ASSESSED.

    Financial scope:
        Daily, long-only equity portfolios.

    Cash-return accounting is not inferred. If portfolio
    returns contain material cash contributions, they must
    be represented through an explicitly extended contract.
    """

    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError(
            "Tolerance must be finite and positive."
        )

    # --------------------------------------------------------
    # Validate individual input contracts
    # --------------------------------------------------------

    returns = validate_portfolio_returns(returns_df)
    holdings = validate_portfolio_holdings(holdings_df)
    securities = validate_security_returns(security_df)
    targets = validate_rebalance_targets(targets_df)

    factors = validate_factor_returns(
        factors_df,
        factor_columns=factor_columns,
        require_rf=True,
    )

    metadata = validate_portfolio_metadata(metadata_df)

    has_portfolio_price = (
        "return_price" in returns.columns
    )

    has_effective_dates = (
        "effective_rebalance_date" in targets.columns
    )

    # --------------------------------------------------------
    # Verify portfolio identity consistency
    # --------------------------------------------------------

    identity_sets = (
        set(returns["portfolio_id"]),
        set(holdings["portfolio_id"]),
        set(targets["portfolio_id"]),
        set(metadata["portfolio_id"]),
    )

    if any(
        identifiers != identity_sets[0]
        for identifiers in identity_sets[1:]
    ):
        raise ValueError(
            "Portfolio identities differ across input datasets."
        )

    # --------------------------------------------------------
    # Validate declared reporting cutoffs when supplied
    # --------------------------------------------------------

    if "reporting_cutoff" in metadata.columns:

        last_return_dates = (
            returns
            .groupby("portfolio_id", as_index=False)
            .agg(last_return_date=("date", "max"))
        )

        cutoff_check = last_return_dates.merge(
            metadata[
                ["portfolio_id", "reporting_cutoff"]
            ],
            on="portfolio_id",
            validate="one_to_one",
        )

        available_cutoffs = cutoff_check[
            "reporting_cutoff"
        ].notna()

        invalid_cutoffs = (
            available_cutoffs
            & (
                cutoff_check["last_return_date"]
                > cutoff_check["reporting_cutoff"]
            )
        )

        if invalid_cutoffs.any():
            raise ValueError(
                "Returns extend beyond the declared reporting cutoff."
            )

    # --------------------------------------------------------
    # Validate optional portfolio calendar alignment
    # --------------------------------------------------------

    return_calendars = [
        set(group["date"])
        for _, group in returns.groupby("portfolio_id")
    ]

    if require_aligned_calendars:

        if any(
            calendar != return_calendars[0]
            for calendar in return_calendars[1:]
        ):
            raise ValueError(
                "Portfolio return calendars are not aligned."
            )

    # --------------------------------------------------------
    # Reconcile holdings to independent security returns
    # --------------------------------------------------------

    security_check = holdings[
        [
            "portfolio_id",
            "date",
            "security_id",
            "return_price",
            "return_total",
        ]
    ].merge(
        securities[
            [
                "security_id",
                "date",
                "return_price",
                "return_total",
            ]
        ],
        on=["security_id", "date"],
        how="left",
        validate="many_to_one",
        suffixes=("_holding", "_history"),
        indicator=True,
    )

    missing_security_returns = int(
        security_check["_merge"].ne("both").sum()
    )

    if missing_security_returns:
        raise ValueError(
            "Missing independent security returns for "
            f"{missing_security_returns:,} holdings observations."
        )

    max_security_price_error = float(
        (
            security_check["return_price_holding"]
            - security_check["return_price_history"]
        ).abs().max()
    )

    max_security_total_error = float(
        (
            security_check["return_total_holding"]
            - security_check["return_total_history"]
        ).abs().max()
    )

    if (
        max_security_price_error > tolerance
        or max_security_total_error > tolerance
    ):
        raise ValueError(
            "Historical security returns disagree with holdings."
        )

    # --------------------------------------------------------
    # Calculate security contributions when not supplied
    # --------------------------------------------------------

    contribution_input = holdings.copy()

    if "contribution_total" in contribution_input.columns:

        contribution_input["_total_component"] = (
            contribution_input["contribution_total"]
        )

    else:

        contribution_input["_total_component"] = (
            contribution_input["weight_bop"]
            * contribution_input["return_total"]
        )

    if "contribution_price" in contribution_input.columns:

        contribution_input["_price_component"] = (
            contribution_input["contribution_price"]
        )

    else:

        contribution_input["_price_component"] = (
            contribution_input["weight_bop"]
            * contribution_input["return_price"]
        )

    # --------------------------------------------------------
    # Aggregate realized daily holdings
    # --------------------------------------------------------

    daily_holdings = (
        contribution_input
        .groupby(
            ["portfolio_id", "date"],
            as_index=False,
        )
        .agg(
            weight_bop_sum=("weight_bop", "sum"),
            weight_eop_sum=("weight_eop", "sum"),
            calculated_total=("_total_component", "sum"),
            calculated_price=("_price_component", "sum"),
        )
    )

    daily_reconciliation = daily_holdings.merge(
        returns,
        on=["portfolio_id", "date"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    unmatched_portfolio_dates = int(
        daily_reconciliation["_merge"].ne("both").sum()
    )

    if unmatched_portfolio_dates:
        raise ValueError(
            "Holdings and portfolio return calendars differ."
        )

    # --------------------------------------------------------
    # Total-return reconciliation is always mandatory
    # --------------------------------------------------------

    max_portfolio_total_error = float(
        (
            daily_reconciliation["calculated_total"]
            - daily_reconciliation["return_total"]
        ).abs().max()
    )

    if max_portfolio_total_error > tolerance:
        raise ValueError(
            "Portfolio total returns do not reconcile "
            "to security contributions."
        )

    total_reconciliation_status = "RECONCILED"

    # --------------------------------------------------------
    # Price reconciliation is conditional on source support
    # --------------------------------------------------------

    if has_portfolio_price:

        max_portfolio_price_error = float(
            (
                daily_reconciliation["calculated_price"]
                - daily_reconciliation["return_price"]
            ).abs().max()
        )

        if max_portfolio_price_error > tolerance:
            raise ValueError(
                "Portfolio price returns do not reconcile "
                "to security contributions."
            )

        price_reconciliation_status = "RECONCILED"

    else:

        max_portfolio_price_error = None

        price_reconciliation_status = (
            "NOT_ASSESSED_PRICE_RETURN_ABSENT"
        )

    # --------------------------------------------------------
    # Apply optional fully invested weight policy
    # --------------------------------------------------------

    max_bop_weight_error = float(
        (
            daily_reconciliation["weight_bop_sum"] - 1.0
        ).abs().max()
    )

    max_eop_weight_error = float(
        (
            daily_reconciliation["weight_eop_sum"] - 1.0
        ).abs().max()
    )

    if fully_invested:

        if (
            max_bop_weight_error > tolerance
            or max_eop_weight_error > tolerance
        ):
            raise ValueError(
                "Fully invested portfolio-weight policy failed."
            )

    # --------------------------------------------------------
    # Reconcile implemented targets only when dates exist
    # --------------------------------------------------------

    implemented_target_rows = None
    targets_without_effective_date = None
    target_membership_mismatches = None
    max_target_weight_error = None

    if has_effective_dates:

        # Effective date is a selection-level attribute.
        date_variants = (
            targets
            .groupby(
                ["portfolio_id", "selection_date"]
            )["effective_rebalance_date"]
            .nunique(dropna=False)
        )

        if date_variants.gt(1).any():
            raise ValueError(
                "Inconsistent effective dates within a selection."
            )

        dated_targets = targets.loc[
            targets["effective_rebalance_date"].notna()
        ].copy()

        implemented_target_rows = len(dated_targets)

        targets_without_effective_date = int(
            targets["effective_rebalance_date"].isna().sum()
        )

        if dated_targets.empty:

            target_reconciliation_status = (
                "NOT_ASSESSED_NO_DATED_TARGETS"
            )

        else:

            implemented_targets = (
                dated_targets
                .rename(
                    columns={
                        "effective_rebalance_date": "date"
                    }
                )[
                    [
                        "portfolio_id",
                        "date",
                        "security_id",
                        "weight_target",
                    ]
                ]
            )

            implementation_dates = (
                implemented_targets[
                    ["portfolio_id", "date"]
                ]
                .drop_duplicates()
            )

            realized_holdings = holdings[
                [
                    "portfolio_id",
                    "date",
                    "security_id",
                    "weight_bop",
                ]
            ].merge(
                implementation_dates,
                on=["portfolio_id", "date"],
                how="inner",
                validate="many_to_one",
            )

            target_comparison = implemented_targets.merge(
                realized_holdings,
                on=[
                    "portfolio_id",
                    "date",
                    "security_id",
                ],
                how="outer",
                validate="one_to_one",
                indicator=True,
            )

            target_membership_mismatches = int(
                target_comparison["_merge"].ne("both").sum()
            )

            if target_membership_mismatches:
                raise ValueError(
                    "Dated targets do not match realized holdings."
                )

            max_target_weight_error = float(
                (
                    target_comparison["weight_bop"]
                    - target_comparison["weight_target"]
                ).abs().max()
            )

            if max_target_weight_error > tolerance:
                raise ValueError(
                    "Target weights do not reconcile "
                    "to realized BOP holdings."
                )

            target_reconciliation_status = "RECONCILED"

    else:

        target_reconciliation_status = (
            "NOT_ASSESSED_EFFECTIVE_DATE_ABSENT"
        )

    # --------------------------------------------------------
    # Report factor coverage without truncating portfolios
    # --------------------------------------------------------

    portfolio_dates = set(returns["date"])
    factor_dates = set(factors["date"])

    factor_first_date = factors["date"].min()
    factor_last_date = factors["date"].max()

    matched_factor_dates = portfolio_dates & factor_dates

    missing_factor_leading = {
        date
        for date in portfolio_dates
        if date < factor_first_date
    }

    missing_factor_internal = {
        date
        for date in portfolio_dates
        if (
            factor_first_date <= date <= factor_last_date
            and date not in factor_dates
        )
    }

    missing_factor_trailing = {
        date
        for date in portfolio_dates
        if date > factor_last_date
    }

    # --------------------------------------------------------
    # Return explicit validation and assessment statuses
    # --------------------------------------------------------

    return {
        "portfolios": len(metadata),
        "portfolio_return_rows": len(returns),
        "portfolio_holding_rows": len(holdings),
        "security_return_rows": len(securities),
        "rebalance_target_rows": len(targets),
        "factor_return_rows": len(factors),

        "matched_holdings_security_rows": len(security_check),
        "missing_security_returns": missing_security_returns,

        "max_security_price_error": max_security_price_error,
        "max_security_total_error": max_security_total_error,

        "max_portfolio_price_error": max_portfolio_price_error,
        "max_portfolio_total_error": max_portfolio_total_error,

        "max_bop_weight_error": max_bop_weight_error,
        "max_eop_weight_error": max_eop_weight_error,

        "implemented_target_rows": implemented_target_rows,
        "targets_without_effective_date": (
            targets_without_effective_date
        ),
        "target_membership_mismatches": (
            target_membership_mismatches
        ),
        "max_target_weight_error": max_target_weight_error,

        "portfolio_trading_dates": len(portfolio_dates),
        "matched_factor_dates": len(matched_factor_dates),
        "missing_factor_leading_dates": (
            len(missing_factor_leading)
        ),
        "missing_factor_internal_dates": (
            len(missing_factor_internal)
        ),
        "missing_factor_trailing_dates": (
            len(missing_factor_trailing)
        ),

        "factor_coverage_ratio": (
            len(matched_factor_dates) / len(portfolio_dates)
        ),

        "total_reconciliation_status": (
            total_reconciliation_status
        ),
        "price_reconciliation_status": (
            price_reconciliation_status
        ),
        "target_reconciliation_status": (
            target_reconciliation_status
        ),
    }
