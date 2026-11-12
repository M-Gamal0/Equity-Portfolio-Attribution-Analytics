"""Minimal own-DataFrame example: synthetic returns, no downloads or files."""

import pandas as pd

from portfolio_analytics.data import validate_portfolio_returns
from portfolio_analytics.performance import compute_wealth_index


def main():
    dates = pd.bdate_range("2024-01-02", periods=4)
    returns = pd.DataFrame(
        {
            "portfolio_id": ["my_portfolio"] * 4 + ["my_benchmark"] * 4,
            "date": list(dates) * 2,
            "return_total": [0.01, -0.005, 0.007, 0.002, 0.008, -0.003, 0.006, 0.001],
        }
    )
    validated = validate_portfolio_returns(returns)
    wealth = compute_wealth_index(validated, initial_wealth=100.0)
    # Each row earns its full previous-close return. Values are illustrative.
    print(wealth[["portfolio_id", "date", "wealth_eop"]].to_string(index=False))
    return wealth


if __name__ == "__main__":
    main()
