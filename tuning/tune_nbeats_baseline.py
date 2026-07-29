"""
Hyperparameter search (Optuna TPE) for the N-BEATS baseline scenario.

Not part of the core reproducibility package -- run this only if you
want to re-tune from scratch. The best parameters found here are
already saved in configs/nbeats_baseline.json (Table 8 in the
manuscript).
"""

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
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

warnings.filterwarnings("ignore")
torch.set_num_threads(8)
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---- Data (October target) ----
df = pd.read_parquet("../data/desember_cleaned_2024.parquet")
df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"])
series = TimeSeries.from_dataframe(df, time_col="DATE_TIME", value_cols=["BEBAN"]).astype(np.float32)

split_date = pd.Timestamp("2024-10-01 00:00:00")
ts_in, ts_out = series.split_before(split_date)
ts_out = ts_out.head(1488)

scaler = Scaler()
ts_in_scaled = scaler.fit_transform(ts_in)

val_len = 1056
train_opt = ts_in_scaled[:-val_len]
val_opt = ts_in_scaled[-val_len:]


def objective(trial):
    in_chunk = trial.suggest_categorical("input_chunk_length", [336, 672])
    n_stacks = trial.suggest_int("num_stacks", 2, 10)
    n_blocks = trial.suggest_int("num_blocks", 1, 8)
    n_layers = trial.suggest_int("num_layers", 2, 8)

    if n_stacks * n_blocks * n_layers > 64:
        raise optuna.exceptions.TrialPruned()

    l_widths = trial.suggest_categorical("layer_widths", [128, 256, 512])
    drop_out = trial.suggest_float("dropout", 0.0, 0.3)
    lr_opt = trial.suggest_float("lr", 5e-5, 5e-3, log=True)
    b_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])

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
            "devices": 1,
            "precision": "32-true",
            "callbacks": [stopper],
            "enable_progress_bar": True,
        },
    )

    dl_kwargs = {"num_workers": 8, "pin_memory": True, "persistent_workers": True}
    model_trial.fit(series=train_opt, val_series=val_opt, verbose=False, dataloader_kwargs=dl_kwargs)

    pred_val_scaled = model_trial.historical_forecasts(
        series=ts_in_scaled,
        start=val_opt.start_time(),
        forecast_horizon=48,
        stride=48,
        retrain=False,
        last_points_only=False,
        verbose=False,
    )
    if isinstance(pred_val_scaled, list):
        pred_val_scaled = concatenate(pred_val_scaled)

    pred_val_mw = scaler.inverse_transform(pred_val_scaled)
    actual_val_mw = scaler.inverse_transform(val_opt)
    actual_slice = actual_val_mw.slice_intersect(pred_val_mw)
    error_mape = mape(actual_slice, pred_val_mw)

    del model_trial
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return error_mape


if __name__ == "__main__":
    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name="nbeats_baseline_tuning_october",
        storage="sqlite:///optuna_nbeats_baseline_october.db",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=70, show_progress_bar=True)

    print("\n" + "=" * 50)
    print("SEARCH COMPLETE")
    print(f"Best validation MAPE: {study.best_value:.2f}%")
    print("Best parameters (copy into configs/nbeats_baseline.json):")
    for key, value in study.best_params.items():
        print(f" - {key}: {value}")
    print("=" * 50)
