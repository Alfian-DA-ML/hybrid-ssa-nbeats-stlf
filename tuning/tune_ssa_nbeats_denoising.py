"""
Hyperparameter search (Optuna TPE) for the hybrid SSA-N-BEATS denoising
scenario.

Not part of the core reproducibility package -- run this only if you
want to re-tune from scratch. The best parameters found here are
already saved in configs/ssa_nbeats_denoising.json (Table 8 in the
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
from src import SSA, compute_w_correlation, auto_group_deterministic

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

torch.set_num_threads(8)
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")
optuna.logging.set_verbosity(optuna.logging.WARNING)

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

# ---- SSA decomposition + auto-grouping (computed once, in-sample only) ----
L_window = 336
threshold = 0.9
val_len = 1056

components, s_values = SSA(ts_in_scaled.values().flatten(), window_length=L_window)
w_corr_matrix = compute_w_correlation(components, L_window)
idx_clean = auto_group_deterministic(w_corr_matrix, threshold=threshold)

clean_in_scaled = TimeSeries.from_times_and_values(
    ts_in.time_index, np.sum(components[idx_clean], axis=0)
)
train_clean = clean_in_scaled[:-val_len]
val_clean = clean_in_scaled[-val_len:]


def objective(trial):
    optuna_bar = TFMProgressBar(enable_train_bar_only=True)

    in_chunk = trial.suggest_categorical("input_chunk_length", [336, 672])
    n_stacks = trial.suggest_int("num_stacks", 2, 8)
    n_blocks = trial.suggest_int("num_blocks", 1, 5)
    n_layers = trial.suggest_int("num_layers", 2, 6)

    if n_stacks * n_blocks * n_layers > 64:
        raise optuna.exceptions.TrialPruned()

    l_widths = trial.suggest_categorical("layer_widths", [128, 256, 512])
    drop_out = trial.suggest_float("dropout", 0.0, 0.3)
    lr_opt = trial.suggest_float("lr", 5e-5, 5e-3, log=True)
    b_size = trial.suggest_categorical("batch_size", [64, 128, 256])

    stopper = EarlyStopping(monitor="val_loss", patience=15, min_delta=0.0001, mode="min")

    model_trial = NBEATSModel(
        input_chunk_length=in_chunk,
        output_chunk_length=48,
        generic_architecture=True,
        num_stacks=n_stacks,
        num_blocks=n_blocks,
        num_layers=n_layers,
        layer_widths=l_widths,
        dropout=drop_out,
        n_epochs=100,
        batch_size=b_size,
        random_state=42,
        loss_fn=nn.MSELoss(),
        optimizer_kwargs={"lr": lr_opt},
        pl_trainer_kwargs={
            "accelerator": "gpu",
            "callbacks": [stopper, optuna_bar],
            "enable_progress_bar": True,
        },
    )

    try:
        dl_kwargs = {"num_workers": 8, "pin_memory": True, "persistent_workers": True}
        model_trial.fit(series=train_clean, val_series=val_clean, verbose=False, dataloader_kwargs=dl_kwargs)

        pred_val_scaled = model_trial.historical_forecasts(
            series=clean_in_scaled,
            start=val_clean.start_time(),
            forecast_horizon=48,
            stride=48,
            retrain=False,
            last_points_only=False,
            verbose=False,
        )
        if isinstance(pred_val_scaled, list):
            pred_val_scaled = concatenate(pred_val_scaled)

        pred_val_mw = scaler.inverse_transform(pred_val_scaled)
        actual_slice = ts_in.slice_intersect(pred_val_mw)
        res_mape = mape(actual_slice, pred_val_mw)
        print(f"Trial {trial.number:02d} | MAPE Val: {res_mape:.4f}% | "
              f"S:{n_stacks}, B:{n_blocks}, L:{n_layers}, W:{l_widths}, BS:{b_size}")

    except optuna.exceptions.TrialPruned:
        raise optuna.exceptions.TrialPruned()
    except Exception as e:
        print(f"Trial {trial.number:02d} failed: {e}")
        res_mape = 100.0
    finally:
        if "model_trial" in locals():
            del model_trial
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return res_mape


if __name__ == "__main__":
    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name="hybrid_denoising_study_october",
        storage="sqlite:///optuna_ssa_nbeats_denoising_october.db",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=70, show_progress_bar=True)

    print("\n" + "=" * 60)
    print("SEARCH COMPLETE")
    print(f"Best validation MAPE: {study.best_value:.4f}%")
    print("Best parameters (copy into configs/ssa_nbeats_denoising.json):")
    for key, value in study.best_params.items():
        print(f" - {key}: {value}")
    print("=" * 60)
