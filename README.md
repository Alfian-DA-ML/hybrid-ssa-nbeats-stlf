# Hybrid SSA–N-BEATS: Multichannel Singular Spectrum Analysis Decomposition for Complex Time Series Forecasting

Reproducibility package for the manuscript:

> **Multichannel SSA decomposition in a hybrid SSA–N-BEATS framework for complex time series forecasting**
> Alfian Adi Pratama, Dr. Putriaji Hendikawati, S.Si., M.Pd., M.Sc.
> Department of Mathematics, Universitas Negeri Semarang, Indonesia
> _MethodsX_ (Elsevier) — **manuscript in review process. DOI and citation will be added here once available.**
> Manuscript link: `[UNDER REVIEW]`

---

## 1. Overview

Forecasting high-resolution time series (e.g. electric load data) is difficult because the signal mixes several temporal characteristics at once — trend, multiple seasonalities, and high-frequency noise — which can degrade the accuracy of deep learning forecasters like [N-BEATS](http://arxiv.org/abs/1905.10437) when trained directly on the raw signal.

This repository implements and benchmarks **three forecasting scenarios**:

| Scenario                                 | Description                                                                                                                                                                                                                                                       |
| ---------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1. N-BEATS (baseline)**                | A single N-BEATS model trained directly on the raw series, with no decomposition.                                                                                                                                                                                 |
| **2. Hybrid SSA–N-BEATS (Denoising)**    | Singular Spectrum Analysis (SSA) is used to separate the deterministic signal from stochastic noise. The denoised signal is fed into a single N-BEATS model.                                                                                                      |
| **3. Hybrid SSA–N-BEATS (Multichannel)** | The deterministic signal from SSA is further split into a **trend** component and a **seasonal** component (via a weighted-correlation auto-grouping mechanism), each modeled independently by a specialist N-BEATS model. Predictions are combined by summation. |

All three scenarios are optimized with Bayesian hyperparameter search (TPE, via [Optuna](https://optuna.org/)) and evaluated on out-of-sample data using MAPE, MAE, RMSE, and R².

The end-to-end pipeline is summarized below:

![Graphical abstract of the SSA-N-BEATS hybrid forecasting framework](./Graphical_abstract.png)

---

## 2. Reproducibility

**This repository is designed to be reproducible.** The hyperparameters shipped in `configs/*.json` are the exact configurations found by the Bayesian search reported in the manuscript for the **October 2024** validation period (a stable period, per the manuscript's evaluation protocol) — corresponding to Table 8. Running the notebooks in `notebooks/` end-to-end, against the same (confidential) electric load dataset, reproduces the October 2024 numbers reported in the paper.

**This is not limited to October 2024, or to electric load data.** The framework — SSA decomposition, auto-grouping, N-BEATS training, rolling forecast, evaluation — is generic. To apply it elsewhere:

- **A different period on the same dataset** (e.g. the December 2024 disruption period from Table 11, or any other window): change the `split_date` (and, if needed, `val_len` / `head(...)` out-of-sample length) at the top of the relevant notebook, then either re-run the matching script in `tuning/` to search fresh hyperparameters for that period, or supply your own and update the corresponding `configs/*.json`. No other code changes are needed — the notebooks always load hyperparameters from `configs/`, never hardcode them.
- **A different time series / domain entirely** (e.g. demand forecasting, commodity prices — see the manuscript's Limitations section): point the data-loading cell at your own file, adjust the SSA window length `L_window` to match your data's dominant periodicity (see Sub-step 2.1 of the manuscript for the reasoning), adjust `output_chunk_length` / rolling-forecast `step_size` to your desired forecast horizon, and re-run hyperparameter search via `tuning/` for your data before training.

In short: **the code in `src/` and `notebooks/` is dataset-agnostic; only the numbers in `configs/*.json` and the split/window parameters are specific to the October 2024 electric load case reported in the paper.**

> ⚠️ **Environment matters for exact numerical reproduction.** Even with a fixed random seed, different versions of `darts`, `torch`, `numpy`, and `pytorch-lightning` can shift weight initialization and training dynamics enough to change early-stopping timing (and therefore runtime and, at the margin, the metrics). Install the exact versions pinned in `requirements.txt` for the closest reproduction of the reported results.

---

## 3. Repository structure

```
.
├── configs/                 Hyperparameter configurations (one JSON per scenario)
├── data/                    Dataset + data documentation (see data/README.md)
├── notebooks/               Core pipeline notebooks (one per scenario)
├── src/                     Reusable Python modules (SSA engine, grouping, forecasting, evaluation)
├── tuning/                  Optional: Optuna hyperparameter search scripts
├── results/                 Local output directory (not versioned)
├── Graphical_abstract.png   Pipeline diagram (see Section 1 above)
├── requirements.txt         Pinned dependencies for exact reproduction
├── requirements_runpod.txt  Dependencies used on the cloud GPU instance for hyperparameter search
├── LICENSE
└── .gitignore
```

### `configs/`

JSON files holding the exact, final hyperparameters used to train each scenario (matches Tables 5, 6, and 8 in the manuscript). Notebooks read these at runtime — nothing is hardcoded in the training cells.

| File                                    | Scenario                               |
| --------------------------------------- | -------------------------------------- |
| `nbeats_baseline.json`                  | Scenario 1 — N-BEATS baseline          |
| `ssa_nbeats_denoising.json`             | Scenario 2 — SSA–N-BEATS denoising     |
| `ssa_nbeats_multichannel_trend.json`    | Scenario 3 — trend specialist model    |
| `ssa_nbeats_multichannel_seasonal.json` | Scenario 3 — seasonal specialist model |

### `data/`

- `desember_cleaned_2024.parquet` — the cleaned electric load series. **Confidential**, provided under agreement with PLN UP2B Central Java & D.I. Yogyakarta; not redistributable (see the manuscript's Ethics Statement). It is included here for the authors' own reproduction workflow — external users should supply their own series in the same schema.
- `README.md` — describes the expected data schema (columns, frequency) so you can substitute your own dataset.

### `notebooks/`

One notebook per scenario, each self-contained: load & split data → (SSA decomposition, where applicable) → train → rolling forecast → evaluate (MAPE, MAE, RMSE, R²). No hyperparameter search or plotting/diagnostic code is included here — see `tuning/` and Section 5 below.

| Notebook                                     | Scenario                                                  |
| -------------------------------------------- | --------------------------------------------------------- |
| `01_scenario1_nbeats_baseline.ipynb`         | N-BEATS baseline — calls Darts directly (no SSA involved) |
| `02_scenario2_ssa_nbeats_denoising.ipynb`    | Hybrid SSA–N-BEATS, denoising                             |
| `03_scenario3_ssa_nbeats_multichannel.ipynb` | Hybrid SSA–N-BEATS, multichannel (trend + seasonal)       |

`logs_ta/` and `logs_skripsi/` (created automatically when a notebook is run) hold Lightning `CSVLogger` training logs (`metrics.csv`, `hparams.yaml`) — useful for inspecting the training/validation loss curve and confirming when early stopping triggered.

### `src/`

Reusable, dataset-agnostic modules implementing the method itself (Tables 1, 3–7 and Figs. 1–3 of the manuscript). Every notebook imports from here rather than redefining logic inline.

| Module                | Contents                                                                                                                                                     |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `ssa_module.py`       | Core SSA engine: `embed`, `decompose` (Broomhead–King eigendecomposition trick), `diagonal_averaging`, `reconstruct`, and the top-level `SSA()` function     |
| `w_correlation.py`    | `compute_w_correlation()` — weighted correlation matrix between reconstructed components                                                                     |
| `grouping.py`         | `auto_group_deterministic()` (threshold-based RC selection) and `split_trend_seasonal()` (RC1 = trend, rest = seasonal)                                      |
| `rolling_forecast.py` | `rolling_forecast_denoising()` and `rolling_forecast_multichannel()` — walk-forward forecasting with SSA re-decomposition at every step                      |
| `evaluation.py`       | `evaluate_series()`, `historical_forecast_metrics()`, `historical_forecast_metrics_multichannel()`, `print_evaluation_report()` — MAPE/MAE/RMSE/R² utilities |
| `__init__.py`         | Public API — `from src import ...`                                                                                                                           |

The N-BEATS baseline scenario does not use `src/rolling_forecast.py`, since it needs no SSA re-decomposition per step; it calls Darts' `NBEATSModel.predict()` directly inside `01_scenario1_nbeats_baseline.ipynb`.

### `tuning/`

Bayesian (TPE) hyperparameter search scripts, one per scenario, using [Optuna](https://optuna.org/). **These are optional** and not required to reproduce the manuscript's results — they were used once to produce the values already saved in `configs/*.json`. Re-run them only if you want to re-tune from scratch (each search takes hours on a GPU; see Table 14 in the manuscript). Each script writes its study to a local SQLite file so a search can be resumed if interrupted.

### `results/`

Empty by default (`.gitkeep` only) — local scratch space for forecast outputs, plots, or exported metrics you generate while running the notebooks. Not versioned.

---

## 4. Installation

**Requirements:** Python 3.10, a CUDA-capable GPU is strongly recommended (SSA re-decomposition + N-BEATS training at every rolling-forecast step is computationally heavy on CPU).

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows

# 2. Install dependencies (pinned to the environment used for the reported results)
pip install -r requirements.txt
```

> `torch` installs a CUDA-enabled build automatically from PyPI on Linux — no `--extra-index-url` or `+cuXXX` suffix is needed. `torchaudio` / `torchvision` are intentionally excluded; they are not used anywhere in this codebase.

If you are using [`uv`](https://github.com/astral-sh/uv) instead of `venv`/`pip`:

```bash
uv venv --python 3.10
.venv\Scripts\activate           # Windows (or: source .venv/bin/activate on Linux/macOS)
uv pip install -r requirements.txt
```

`requirements_runpod.txt` lists the environment used specifically for hyperparameter search on a cloud GPU instance (NVIDIA RTX 4000 Ada Generation via RunPod, per the manuscript's Background section) — use it only if reproducing the `tuning/` search step on similar cloud infrastructure.

---

## 5. Usage

1. Place your dataset in `data/` following the schema in `data/README.md` (or use the provided confidential file, if you have access to it).
2. Run the notebooks in `notebooks/` in order (`01` → `02` → `03`), or independently — each is self-contained.
3. Each notebook prints a final evaluation report (MAPE, MAE, RMSE, R²) for the training, validation, and out-of-sample sets.
4. To reproduce a different period or dataset, see [Section 2 — Reproducibility](#2-reproducibility) above.
5. To re-run hyperparameter search instead of using the values in `configs/`, see `tuning/README.md`.

---

## Citation

This manuscript is currently in the review/submission process. A full citation (DOI, volume, pages) will be added here once available. In the meantime, please contact the corresponding author (Putriaji Hendikawati, `putriaji.mat@mail.unnes.ac.id`) for citation information.

## License

See [`LICENSE`](https://github.com/Alfian-DA-ML/hybrid-ssa-nbeats-stlf/blob/main/LICENSE).
