"""
Hyperparameter search (Optuna TPE) for the hybrid SSA-N-BEATS
multichannel scenario (trend + seasonal specialist models, tuned
jointly on combined validation MAPE).

Not part of the core reproducibility package -- run this only if you
want to re-tune from scratch. The best parameters found here are
already saved in configs/ssa_nbeats_multichannel_trend.json and
configs/ssa_nbeats_multichannel_seasonal.json (Tables 5-6 in the
manuscript).
"""

import sys
import warnings
import logging
import gc

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import optuna
from optuna.samplers import TPESampler
from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape
from darts.models import NBEATSModel
from darts.utils.callbacks import TFMProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

sys.path.insert(0, "..")
from src import SSA, compute_w_correlation, auto_group_deterministic, split_trend_seasonal

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

torch.set_num_threads(8)
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

# ---- Data (October target) ----
df = pd.read_parquet("../data/desember_cleaned_2024.parquet")
df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"])
df = df.sort_values("DATE_TIME").set_index("DATE_TIME")
ts_full = TimeSeries.from_series(df["BEBAN"]).astype(np.float32)

split_date = pd.Timestamp("2024-10-01 00:00:00")
ts_in, ts_out = ts_full.split_before(split_date)
ts_out = ts_out.head(1488)

scaler = Scaler()
ts_in_scaled = scaler.fit_transform(ts_in)

# ---- SSA decomposition + trend/seasonal split (computed once, in-sample only) ----
L_window = 336
threshold = 0.9
val_len = 1056

components, s_values = SSA(ts_in_scaled.values().flatten(), window_length=L_window)
w_corr_matrix = compute_w_correlation(components, L_window)
idx_clean = auto_group_deterministic(w_corr_matrix, threshold=threshold)
idx_trend, idx_seasonal = split_trend_seasonal(idx_clean)

trend_in_scaled = TimeSeries.from_times_and_values(ts_in.time_index, np.sum(components[idx_trend], axis=0))
seasonal_in_scaled = TimeSeries.from_times_and_values(ts_in.time_index, np.sum(components[idx_seasonal], axis=0))

train_trend = trend_in_scaled[:-val_len]
val_trend = trend_in_scaled[-val_len:]
train_seasonal = seasonal_in_scaled[:-val_len]
val_seasonal = seasonal_in_scaled[-val_len:]


def objective(trial):
    lr_t = trial.suggest_float("lr_t", 5e-5, 5e-4, log=True)
    width_t = trial.suggest_categorical("width_t", [64, 128, 256])

    lr_s = trial.suggest_float("lr_s", 1e-4, 5e-3, log=True)
    dr_s = trial.suggest_float("dr_s", 0.0, 0.3)
    width_s = trial.suggest_categorical("width_s", [128, 256, 512])
    n_stacks_s = trial.suggest_int("num_stacks_s", 2, 8)
    n_layers_s = trial.suggest_int("num_layers_s", 2, 4)
    n_blocks_s = trial.suggest_int("num_blocks_s", 1, 3)
    icl_s = trial.suggest_categorical("icl_s", [336, 672])
    b_size = trial.suggest_categorical("batch_size", [64, 128, 256])

    es_t = EarlyStopping(monitor="val_loss", patience=10, min_delta=0.0001, mode="min")
    es_s = EarlyStopping(monitor="val_loss", patience=15, min_delta=0.0001, mode="min")
    bar_t = TFMProgressBar(enable_train_bar_only=True)
    bar_s = TFMProgressBar(enable_train_bar_only=True)

    model_t = NBEATSModel(
        input_chunk_length=672, output_chunk_length=48,
        generic_architecture=True, num_stacks=2, num_blocks=1, num_layers=2,
        layer_widths=width_t, loss_fn=nn.MSELoss(), optimizer_kwargs={"lr": lr_t}, n_epochs=100,
        batch_size=256, random_state=42,
        pl_trainer_kwargs={"accelerator": "gpu", "enable_progress_bar": True, "callbacks": [es_t, bar_t]},
    )

    model_s = NBEATSModel(
        input_chunk_length=icl_s, output_chunk_length=48,
        generic_architecture=True, num_stacks=n_stacks_s, num_blocks=n_blocks_s, num_layers=n_layers_s,
        layer_widths=width_s, loss_fn=nn.MSELoss(), dropout=dr_s, optimizer_kwargs={"lr": lr_s}, n_epochs=100,
        batch_size=b_size, random_state=42,
        pl_trainer_kwargs={"accelerator": "gpu", "enable_progress_bar": True, "callbacks": [es_s, bar_s]},
    )

    try:
        dl_kwargs = {"num_workers": 8, "pin_memory": True, "persistent_workers": False}

        print(f"\n[Trial {trial.number}] Training trend...")
        model_t.fit(series=train_trend, val_series=val_trend, verbose=False, dataloader_kwargs=dl_kwargs)

        print(f"[Trial {trial.number}] Training seasonal (ICL={icl_s})...")
        model_s.fit(series=train_seasonal, val_series=val_seasonal, verbose=False, dataloader_kwargs=dl_kwargs)

        p_t = model_t.historical_forecasts(series=trend_in_scaled, start=val_trend.start_time(),
                                            forecast_horizon=48, stride=48, retrain=False,
                                            last_points_only=False, verbose=False)
        p_s = model_s.historical_forecasts(series=seasonal_in_scaled, start=val_seasonal.start_time(),
                                            forecast_horizon=48, stride=48, retrain=False,
                                            last_points_only=False, verbose=False)

        if isinstance(p_t, list): p_t = concatenate(p_t)
        if isinstance(p_s, list): p_s = concatenate(p_s)

        pred_combined = p_t + p_s
        pred_mw = scaler.inverse_transform(pred_combined)
        actual_mw_val = ts_in.slice_intersect(pred_mw)
        res_mape = mape(actual_mw_val, pred_mw)
        print(f"Trial {trial.number} finished. MAPE: {res_mape:.4f}%")

    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        res_mape = 100.0
    finally:
        if "model_t" in locals(): del model_t
        if "model_s" in locals(): del model_s
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return res_mape


if __name__ == "__main__":
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=42),
        study_name="multichannel_scenario_october",
        storage="sqlite:///optuna_ssa_nbeats_multichannel_october.db",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=70, show_progress_bar=True)

    print("\n" + "=" * 60)
    print(f"BEST MAPE: {study.best_value:.4f}%")
    print("Best parameters (split into configs/ssa_nbeats_multichannel_trend.json")
    print("and configs/ssa_nbeats_multichannel_seasonal.json by _t / _s suffix):")
    print(study.best_params)
    print("=" * 60)
