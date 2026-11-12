"""Numerically scaled OLS with coefficients returned in original factor units.

Centering/scaling changes the coordinates, not the fitted regression. The
original design's rank, condition limit, reconciliation and orthogonality
thresholds remain in force. The original notebook kernel is kept unchanged.
"""

import numpy as np

from .ols import fit_ols_factor_regression, OLS_RECONCILIATION_TOLERANCE


def fit_period_ols(observations_df, *, factor_columns, model_id, max_condition_number=1e10):
    original = observations_df.copy()
    f = original[list(factor_columns)].to_numpy(dtype=float)
    y = original["excess_return"].to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(y)), f])
    condition = float(np.linalg.cond(x))
    if not np.isfinite(condition) or condition > max_condition_number:
        raise ValueError("Original regression design exceeds the condition-number limit.")
    if np.linalg.matrix_rank(x) != x.shape[1]:
        raise ValueError("Original regression design is rank-deficient.")
    center, scale = f.mean(axis=0), f.std(axis=0)
    if not np.isfinite(f).all() or not np.isfinite(y).all() or (scale <= 0).any():
        raise ValueError("Factor regression requires finite, varying predictors and response.")
    transformed = original.copy()
    transformed[list(factor_columns)] = (f - center) / scale
    transformed["excess_return"] = y - y.mean()
    fit, coefficients, residuals = fit_ols_factor_regression(transformed,
        factor_columns=factor_columns, model_id=model_id, max_condition_number=max_condition_number)
    b = coefficients.set_index("term").loc[["intercept", *factor_columns], "coefficient"].to_numpy(copy=True)
    b[1:] /= scale
    b[0] += y.mean() - center @ b[1:]
    predicted = x @ b
    error = y - predicted
    scores = float(np.max(np.abs(x.T @ error)))
    tolerance = OLS_RECONCILIATION_TOLERANCE
    if scores > tolerance * max(1.0, float(np.linalg.norm(y, ord=1))):
        raise ValueError("OLS residual orthogonality validation failed in original factor units.")
    reconstruction = float(np.max(np.abs(y - (predicted + error))))
    if reconstruction > tolerance:
        raise ValueError("Original-unit OLS reconstruction validation failed.")
    # Independent unscaled solve must agree; no tolerance is relaxed.
    np.testing.assert_allclose(b, np.linalg.lstsq(x, y, rcond=None)[0], atol=1e-10, rtol=1e-9)
    coefficients["coefficient"] = coefficients.term.map(dict(zip(["intercept", *factor_columns], b)))
    rss, tss = float(error @ error), float(np.sum((y - y.mean()) ** 2))
    variance = rss / (len(y) - x.shape[1])
    fit.loc[:, "condition_number"] = condition
    fit.loc[:, "sum_squared_residuals"] = rss
    fit.loc[:, "total_sum_squares"] = tss
    fit.loc[:, "residual_variance"] = variance
    fit.loc[:, "residual_standard_error"] = np.sqrt(variance)
    fit.loc[:, "residual_mean"] = float(error.mean())
    fit.loc[:, "reconstruction_error"] = reconstruction
    fit.loc[:, "max_normal_equation_error"] = scores
    if tss > 0:
        r2 = 1 - rss / tss
        fit.loc[:, "r_squared"] = r2
        fit.loc[:, "adjusted_r_squared"] = 1 - (1 - r2) * (len(y) - 1) / (len(y) - x.shape[1])
    residuals["actual_excess_return"] = y
    residuals["fitted_excess_return"] = predicted
    residuals["residual_excess_return"] = error
    return fit, coefficients, residuals
