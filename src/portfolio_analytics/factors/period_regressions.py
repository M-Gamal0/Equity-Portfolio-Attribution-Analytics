"""Period estimation loop promoted from Notebook 11, cell index 15.

The calculation and eligibility branches retain the validated research behavior.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .stable import fit_period_ols as fit_ols_factor_regression


def estimate_direct_periods(aligned_returns, period_definitions, *, factor_columns,
                            model_id, minimum_observations=60, max_condition_number=1e10):
    direct_aligned_returns = aligned_returns
    direct_regression_periods_df = period_definitions
    direct_source_start = aligned_returns['date'].min()
    direct_source_end = aligned_returns['date'].max()
    primary_factor_columns = factor_columns
    DIRECT_FACTOR_MODEL_ID = model_id
    DIRECT_FACTOR_MINIMUM_OBSERVATIONS = minimum_observations
    OLS_MAX_CONDITION_NUMBER = max_condition_number
    DIRECT_FACTOR_TERMS = ('intercept', *factor_columns)
    direct_response_mapping = {
        'ACTIVE_PORTFOLIO': 'active_portfolio_excess_return',
        'BENCHMARK': 'benchmark_excess_return',
        'ACTIVE_RETURN': 'active_spread_regression_response',
    }
    direct_fit_records, direct_coefficient_frames, direct_residual_frames = [], [], []
    for period in direct_regression_periods_df.itertuples(index=False):
        period_sample = direct_aligned_returns.loc[direct_aligned_returns['date'].between(period.period_start_date, period.period_end_date, inclusive='both')].sort_values('date').reset_index(drop=True)
        if period_sample.empty:
            raise ValueError(f'No returns in reporting period {period.period_id}.')
        observed_days = len(period_sample)
        available_days = int(period_sample['factor_available'].sum())
        missing_days = observed_days - available_days
        insufficient_source_history = period.period_type == 'TRAILING' and (period.period_start_date < direct_source_start or period.period_end_date > direct_source_end)
        for target_id, response_column in direct_response_mapping.items():
            record = {'target_id': target_id, 'model_id': DIRECT_FACTOR_MODEL_ID, 'period_id': period.period_id, 'period_type': period.period_type, 'period_start_date': period.period_start_date, 'period_end_date': period.period_end_date, 'observed_start_date': period_sample['date'].min(), 'observed_end_date': period_sample['date'].max(), 'observed_return_days': observed_days, 'factor_available_days': available_days, 'factor_missing_days': missing_days, 'regression_observations': 0, 'r_squared': np.nan, 'adjusted_r_squared': np.nan, 'residual_standard_error': np.nan, 'condition_number': np.nan, 'fit_status': None}
            if insufficient_source_history:
                record['fit_status'] = 'UNAVAILABLE_INSUFFICIENT_SOURCE_HISTORY'
                direct_fit_records.append(record)
                continue
            if missing_days > 0:
                record['fit_status'] = 'UNAVAILABLE_MISSING_FACTOR_DATES'
                direct_fit_records.append(record)
                continue
            if observed_days < DIRECT_FACTOR_MINIMUM_OBSERVATIONS:
                record['fit_status'] = 'UNAVAILABLE_INSUFFICIENT_OBSERVATIONS'
                direct_fit_records.append(record)
                continue
            regression_sample = period_sample[['date', response_column, *primary_factor_columns]].rename(columns={response_column: 'excess_return'}).copy()
            if not np.isfinite(regression_sample[['excess_return', *primary_factor_columns]].to_numpy(dtype=float)).all():
                raise ValueError(f'Non-finite regression inputs: {target_id}, {period.period_id}.')
            direct_fit, direct_coefficients, _ = fit_ols_factor_regression(regression_sample, factor_columns=tuple(primary_factor_columns), model_id=DIRECT_FACTOR_MODEL_ID, max_condition_number=OLS_MAX_CONDITION_NUMBER)
            if len(direct_fit) != 1:
                raise ValueError('Direct OLS returned an unexpected fit count.')
            coefficients = direct_coefficients.set_index('term').loc[list(DIRECT_FACTOR_TERMS), 'coefficient'].to_numpy(dtype=float)
            X = np.column_stack([np.ones(len(regression_sample)), regression_sample[list(primary_factor_columns)].to_numpy(dtype=float)])
            y = regression_sample['excess_return'].to_numpy(dtype=float)
            independent_coefficients = np.linalg.lstsq(X, y, rcond=None)[0]
            np.testing.assert_allclose(coefficients, independent_coefficients, rtol=1e-09, atol=1e-10)
            fitted = X @ coefficients
            residuals = y - fitted
            n = len(y)
            p = X.shape[1]
            rss = float(residuals @ residuals)
            tss = float(np.sum((y - y.mean()) ** 2))
            if tss <= 0:
                raise ValueError('Direct regression dependent variable has zero total variation.')
            independent_r2 = 1.0 - rss / tss
            adjusted_r2 = 1.0 - (1.0 - independent_r2) * (n - 1) / (n - p)
            residual_se = np.sqrt(rss / (n - p))
            condition_number = float(np.linalg.cond(X))
            np.testing.assert_allclose(float(direct_fit.iloc[0]['r_squared']), independent_r2, rtol=1e-09, atol=1e-10)
            np.testing.assert_allclose(float(direct_fit.iloc[0]['residual_standard_error']), residual_se, rtol=1e-09, atol=1e-10)
            if condition_number > OLS_MAX_CONDITION_NUMBER:
                raise ValueError('Direct regression condition-number threshold exceeded.')
            record.update({'regression_observations': n, 'r_squared': independent_r2, 'adjusted_r_squared': adjusted_r2, 'residual_standard_error': residual_se, 'condition_number': condition_number, 'fit_status': 'FIT_COMPLETED'})
            direct_fit_records.append(record)
            coefficients_output = direct_coefficients[['term', 'coefficient']].copy()
            coefficients_output['target_id'] = target_id
            coefficients_output['period_id'] = period.period_id
            coefficients_output['model_id'] = DIRECT_FACTOR_MODEL_ID
            direct_coefficient_frames.append(coefficients_output)
            direct_residual_frames.append(pd.DataFrame({'target_id': target_id, 'model_id': DIRECT_FACTOR_MODEL_ID, 'period_id': period.period_id, 'date': regression_sample['date'].to_numpy(), 'observed_response': y, 'fitted_response': fitted, 'residual_response': residuals}))

    coefficients = (pd.concat(direct_coefficient_frames, ignore_index=True)
                    if direct_coefficient_frames else pd.DataFrame(columns=[
                        'target_id', 'period_id', 'model_id', 'term', 'coefficient']))
    residuals = (pd.concat(direct_residual_frames, ignore_index=True)
                 if direct_residual_frames else pd.DataFrame(columns=[
                     'target_id', 'period_id', 'model_id', 'date',
                     'observed_response', 'fitted_response', 'residual_response']))
    return pd.DataFrame(direct_fit_records), coefficients, residuals
