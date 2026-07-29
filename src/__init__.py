"""
SSA-N-BEATS hybrid forecasting framework.

Reusable modules implementing the two hybrid scenarios described in
the manuscript "Multichannel SSA decomposition in a hybrid
SSA-N-BEATS framework for complex time series forecasting"
(Tables 1, 3-7, Figs 1-3):

- ssa_module        : core SSA engine (embed, decompose, reconstruct)
- w_correlation     : weighted correlation (w-correlation) matrix
- grouping          : automatic deterministic/trend/seasonal grouping
- rolling_forecast  : walk-forward rolling forecast (denoising & multichannel)
- evaluation        : MAPE / MAE / RMSE / R2 evaluation utilities

Note: the N-BEATS baseline scenario is intentionally out of scope for
now; this package currently covers only the two hybrid scenarios.

Optuna hyperparameter search and exploratory/diagnostic plotting code
(scree plots, W-correlation heatmaps, periodograms, ADF/Ljung-Box
tests, etc.) are intentionally kept out of this package, as they are
not required to reproduce the method itself.
"""

from .ssa_module import SSA, embed, decompose, diagonal_averaging, reconstruct
from .w_correlation import compute_w_correlation
from .grouping import auto_group_deterministic, split_trend_seasonal
from .rolling_forecast import (
    rolling_forecast_denoising,
    rolling_forecast_multichannel,
)
from .evaluation import (
    evaluate_series,
    historical_forecast_metrics,
    historical_forecast_metrics_multichannel,
    print_evaluation_report,
)

__all__ = [
    "SSA",
    "embed",
    "decompose",
    "diagonal_averaging",
    "reconstruct",
    "compute_w_correlation",
    "auto_group_deterministic",
    "split_trend_seasonal",
    "rolling_forecast_denoising",
    "rolling_forecast_multichannel",
    "evaluate_series",
    "historical_forecast_metrics",
    "historical_forecast_metrics_multichannel",
    "print_evaluation_report",
]