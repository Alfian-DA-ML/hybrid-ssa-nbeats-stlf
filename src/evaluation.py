"""
Evaluation utilities for the SSA-N-BEATS hybrid forecasting framework.

Provides a single reusable interface to compute MAPE, MAE, RMSE, and R^2
across the training, validation, and out-of-sample (test) sets for the
two hybrid modeling scenarios described in the manuscript: hybrid
SSA-N-BEATS (denoising) and hybrid SSA-N-BEATS (multichannel).

Note: the N-BEATS baseline scenario is intentionally out of scope for
now; this module currently covers only the two hybrid scenarios.
"""

from darts import concatenate
from darts.metrics import mape, mae, rmse, r2_score


def evaluate_series(actual, predicted):
    """
    Compute MAPE, MAE, RMSE, and R^2 between an actual and a predicted
    TimeSeries.

    Parameters
    ----------
    actual : darts.TimeSeries
    predicted : darts.TimeSeries

    Returns
    -------
    dict
        {"mape": float, "mae": float, "rmse": float, "r2": float}
    """
    return {
        "mape": mape(actual, predicted),
        "mae": mae(actual, predicted),
        "rmse": rmse(actual, predicted),
        "r2": r2_score(actual, predicted),
    }


def _concat_if_list(preds):
    """
    darts' historical_forecasts(last_points_only=False) returns a list
    of TimeSeries chunks; concatenate them into a single series if so.
    """
    if isinstance(preds, list):
        return concatenate(preds)
    return preds


def historical_forecast_metrics(
    model,
    series_scaled,
    start,
    scaler,
    actual_raw,
    forecast_horizon=48,
    stride=48,
):
    """
    Compute train/validation-window evaluation metrics for a single
    N-BEATS model using Darts' historical_forecasts backtesting mode.

    Model-agnostic: works for any scenario where a single model
    produces the full prediction directly from a (scaled) series --
    e.g. the N-BEATS baseline (fed the raw series) or the SSA-N-BEATS
    denoising scenario (fed the SSA-denoised series). For the
    multichannel scenario (two specialist models), use
    historical_forecast_metrics_multichannel() instead.

    Parameters
    ----------
    model : darts.models.NBEATSModel
        Trained model.
    series_scaled : darts.TimeSeries
        The scaled series historical_forecasts should walk over
        (e.g. clean_in_scaled for denoising, ts_*_in_scaled for
        baseline).
    start : int or pandas.Timestamp
        Index or timestamp historical_forecasts should start from.
    scaler : darts.dataprocessing.transformers.Scaler
        Fitted scaler used to inverse-transform predictions back to
        the original (MW) scale.
    actual_raw : darts.TimeSeries
        The actual, unscaled series to compare predictions against
        (sliced to intersect with the predictions).
    forecast_horizon : int, default 48
    stride : int, default 48

    Returns
    -------
    dict
        Metrics dict from evaluate_series().
    """
    preds_scaled = model.historical_forecasts(
        series=series_scaled,
        start=start,
        forecast_horizon=forecast_horizon,
        stride=stride,
        retrain=False,
        last_points_only=False,
        verbose=False,
    )
    preds_scaled = _concat_if_list(preds_scaled)
    preds_mw = scaler.inverse_transform(preds_scaled)

    actual_mw = actual_raw.slice_intersect(preds_mw)
    return evaluate_series(actual_mw, preds_mw)


def historical_forecast_metrics_multichannel(
    model_trend,
    model_seasonal,
    trend_series_scaled,
    seasonal_series_scaled,
    start,
    scaler,
    actual_raw,
    forecast_horizon=48,
    stride=48,
):
    """
    Compute train/validation-window evaluation metrics for the
    multichannel scenario, where the trend and seasonal specialist
    models are backtested independently and their predictions summed
    before inverse-transforming.

    Parameters
    ----------
    model_trend, model_seasonal : darts.models.NBEATSModel
        Trained trend- and seasonal-specialist models.
    trend_series_scaled, seasonal_series_scaled : darts.TimeSeries
        Scaled trend / seasonal component series for historical_forecasts.
    start : int or pandas.Timestamp
    scaler : darts.dataprocessing.transformers.Scaler
        Fitted scaler used to inverse-transform the combined
        (trend + seasonal) prediction back to the original (MW) scale.
    actual_raw : darts.TimeSeries
        The actual, unscaled series to compare predictions against.
    forecast_horizon : int, default 48
    stride : int, default 48

    Returns
    -------
    dict
        Metrics dict from evaluate_series().
    """
    preds_t = model_trend.historical_forecasts(
        series=trend_series_scaled,
        start=start,
        forecast_horizon=forecast_horizon,
        stride=stride,
        retrain=False,
        last_points_only=False,
        verbose=False,
    )
    preds_s = model_seasonal.historical_forecasts(
        series=seasonal_series_scaled,
        start=start,
        forecast_horizon=forecast_horizon,
        stride=stride,
        retrain=False,
        last_points_only=False,
        verbose=False,
    )
    preds_t = _concat_if_list(preds_t)
    preds_s = _concat_if_list(preds_s)

    preds_scaled = preds_t + preds_s
    preds_mw = scaler.inverse_transform(preds_scaled)

    actual_mw = actual_raw.slice_intersect(preds_mw)
    return evaluate_series(actual_mw, preds_mw)


def print_evaluation_report(scenario_name, period_labels, metrics_by_set):
    """
    Print a formatted evaluation report matching the console output
    style used across the project notebooks.

    Parameters
    ----------
    scenario_name : str
        e.g. "HYBRID SSA-N-BEATS DENOISING (October 2024)"
    period_labels : dict
        e.g. {"training": "January 2022 - August 2024",
              "validation": "September 2024",
              "test": "October 2024"}
    metrics_by_set : dict
        e.g. {"training": {...}, "validation": {...}, "test": {...}}
        Each value must be a metrics dict from evaluate_series()
        (keys: mape, mae, rmse, r2). Computing every value here
        (rather than relying on hand-named variables per scenario)
        avoids the kind of leftover/undefined-variable mismatch that
        can creep in when each scenario's report is written by hand.
    """
    set_order = ["training", "validation", "test"]
    display_labels = {
        "training": "[1] TRAINING",
        "validation": "[2] VALIDATION",
        "test": "[3] OUT-OF-SAMPLE / TEST",
    }

    print("\n" + "=" * 50)
    print(f"EVALUATION REPORT: {scenario_name}")
    print("=" * 50)

    present_sets = [s for s in set_order if s in metrics_by_set]
    for idx, set_name in enumerate(present_sets):
        m = metrics_by_set[set_name]
        period = period_labels.get(set_name, "")
        print(f"{display_labels[set_name]} ({period})")
        print(f"   - MAPE : {m['mape']:.2f}%")
        print(f"   - MAE  : {m['mae']:.2f} MW")
        print(f"   - RMSE : {m['rmse']:.2f} MW")
        print(f"   - R2   : {m['r2']:.2f}")
        if idx < len(present_sets) - 1:
            print("-" * 50)
    print("=" * 50)
