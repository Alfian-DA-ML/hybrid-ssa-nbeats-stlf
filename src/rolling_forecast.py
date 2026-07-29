"""
Rolling (walk-forward) forecast implementations for the two hybrid
SSA-N-BEATS scenarios described in the manuscript: denoising and
multichannel.

Both follow the same walk-forward loop: predict over a fixed horizon,
append the newly observed actual values to history, and repeat until
the full out-of-sample period is covered (Tables 4 and 7 in the
manuscript). At every iteration, SSA decomposition and auto-grouping
are re-run on the updated history.

Note: the N-BEATS baseline scenario is intentionally out of scope for
now; this module currently covers only the two hybrid scenarios.
"""

import numpy as np
from darts import TimeSeries, concatenate
from tqdm import tqdm

from .ssa_module import SSA
from .w_correlation import compute_w_correlation
from .grouping import auto_group_deterministic, split_trend_seasonal


def rolling_forecast_denoising(
    model,
    ts_in_scaled,
    ts_out_scaled,
    window_length=336,
    threshold=0.9,
    step_size=48,
):
    """
    Rolling forecast for the hybrid SSA-N-BEATS denoising scenario
    (Table 4). At each iteration, SSA re-decomposes the updated
    history, deterministic components are re-selected via
    auto_group_deterministic, and the denoised signal is fed to the
    trained model.

    Parameters
    ----------
    model : darts.models.NBEATSModel
        Trained denoising-scenario N-BEATS model.
    ts_in_scaled : darts.TimeSeries
        Scaled in-sample series.
    ts_out_scaled : darts.TimeSeries
        Scaled out-of-sample series.
    window_length : int, default 336
        SSA window length (L).
    threshold : float, default 0.9
        Weighted-correlation threshold for auto-grouping.
    step_size : int, default 48
        Forecast horizon per rolling iteration.

    Returns
    -------
    darts.TimeSeries
        Concatenated scaled predictions. Inverse-transform with the
        fitted scaler to obtain MW values.
    """
    history = ts_in_scaled
    preds_list = []
    total_steps = len(ts_out_scaled)

    for i in tqdm(range(0, total_steps, step_size), desc="Rolling forecast (denoising)"):
        current_vals = history.values().flatten()
        comps, _ = SSA(current_vals, window_length=window_length)

        w_corr_matrix = compute_w_correlation(comps, window_length)
        idx_clean = auto_group_deterministic(w_corr_matrix, threshold=threshold)

        clean_vals = np.sum(comps[idx_clean], axis=0)
        h_clean = TimeSeries.from_times_and_values(history.time_index, clean_vals)

        pred = model.predict(n=step_size, series=h_clean)
        preds_list.append(pred)

        if i + step_size <= total_steps:
            actual_chunk = ts_out_scaled[i : i + step_size]
            history = history.append(actual_chunk)

    return concatenate(preds_list)


def rolling_forecast_multichannel(
    model_trend,
    model_seasonal,
    ts_in_scaled,
    ts_out_scaled,
    window_length=336,
    threshold=0.9,
    step_size=48,
):
    """
    Rolling forecast for the hybrid SSA-N-BEATS multichannel scenario
    (Table 7). At each iteration, SSA re-decomposes the updated
    history, deterministic components are split into trend and
    seasonal groups, and each group is fed to its specialist model.
    The two forecasts are summed to form the combined prediction.

    Parameters
    ----------
    model_trend, model_seasonal : darts.models.NBEATSModel
        Trained trend- and seasonal-specialist N-BEATS models.
    ts_in_scaled : darts.TimeSeries
        Scaled in-sample series.
    ts_out_scaled : darts.TimeSeries
        Scaled out-of-sample series.
    window_length : int, default 336
    threshold : float, default 0.9
    step_size : int, default 48

    Returns
    -------
    darts.TimeSeries
        Concatenated scaled predictions (trend + seasonal). Inverse-
        transform with the fitted scaler to obtain MW values.
    """
    history = ts_in_scaled
    preds_list = []
    total_steps = len(ts_out_scaled)

    for i in tqdm(range(0, total_steps, step_size), desc="Rolling forecast (multichannel)"):
        current_vals = history.values().flatten()
        comps, _ = SSA(current_vals, window_length=window_length)

        w_corr_matrix = compute_w_correlation(comps, window_length)
        idx_clean = auto_group_deterministic(w_corr_matrix, threshold=threshold)
        idx_trend, idx_seasonal = split_trend_seasonal(idx_clean)

        h_t_vals = np.sum(comps[idx_trend], axis=0)
        h_s_vals = np.sum(comps[idx_seasonal], axis=0)

        h_t = TimeSeries.from_times_and_values(history.time_index, h_t_vals)
        h_s = TimeSeries.from_times_and_values(history.time_index, h_s_vals)

        p_t = model_trend.predict(n=step_size, series=h_t)
        p_s = model_seasonal.predict(n=step_size, series=h_s)
        preds_list.append(p_t + p_s)

        if i + step_size <= total_steps:
            actual_chunk = ts_out_scaled[i : i + step_size]
            history = history.append(actual_chunk)

    return concatenate(preds_list)