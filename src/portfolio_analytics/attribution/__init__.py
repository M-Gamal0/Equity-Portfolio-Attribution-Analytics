"""Independently callable attribution calculations."""

from .inputs import prepare_attribution_inputs as prepare_attribution_inputs
from .security import calculate_security_active_contributions as calculate_security_active_contributions
from .sectors import aggregate_sector_returns as aggregate_sector_returns
from .sectors import build_brinson_sector_comparison as build_brinson_sector_comparison
from .brinson import calculate_daily_brinson_fachler as calculate_daily_brinson_fachler
from .carino import carino_scaling_factor as carino_scaling_factor
from .carino import link_attribution_carino as link_attribution_carino
from .menchero import menchero_linking_coefficients as menchero_linking_coefficients
from .menchero import link_attribution_menchero as link_attribution_menchero
from .windows import build_attribution_reporting_windows as build_attribution_reporting_windows
from .periods import build_period_attribution as build_period_attribution
from .periods import compare_attribution_linking_methods as compare_attribution_linking_methods
from .linking import link_security_contributions_by_period as link_security_contributions_by_period
from .summaries import build_attribution_reporting_tables as build_attribution_reporting_tables
from .cumulative import calculate_cumulative_brinson_attribution as calculate_cumulative_brinson_attribution

__all__ = ['prepare_attribution_inputs', 'calculate_security_active_contributions', 'aggregate_sector_returns', 'build_brinson_sector_comparison', 'calculate_daily_brinson_fachler', 'carino_scaling_factor', 'link_attribution_carino', 'menchero_linking_coefficients', 'link_attribution_menchero', 'build_attribution_reporting_windows', 'build_period_attribution', 'compare_attribution_linking_methods', 'link_security_contributions_by_period', 'build_attribution_reporting_tables', 'calculate_cumulative_brinson_attribution']

from .api import run as run

__all__.append("run")
