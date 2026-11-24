# leakage-aware-funnel

Reproducibility code for the paper **"Leakage-Aware Representation Learning for Multi-Stage
Conversion Prediction"** (under review, anonymous submission).

This repository contains the experiment notebooks, the controlled temporal-leakage injection
harness, the detection oracle, the proposed leakage-regularized Transformer funnel model, and the
baseline models used to produce all reported results. It is released to support anonymous peer review
and reproducibility.

> **Anonymity.** This repository is anonymized for double-blind review. It intentionally contains no
> author names, affiliations, acknowledgements, or absolute file paths. Please keep it anonymous when
> referencing it during review.

## Repository structure

```
leakage-aware-funnel/
├── data/                 # place the dataset here (not distributed; see "Data")
├── notebooks/            # one notebook per research question; each is a single runnable cell
│   ├── 00_data_prep.ipynb      # dataset profiling and summary
│   ├── 01_detection.ipynb      # RQ1 (detection) and RQ2 (oracle blind spot)
│   ├── 02_robustness.ipynb     # RQ3 (robustness under leakage) and RQ4 (no-leak parity)
│   ├── 03_ablation.ipynb       # RQ5 (component ablation)
│   └── 04_guards.ipynb         # RQ6 (deployment guards)
└── results/
    ├── figures/          # figures written here (grayscale, dpi 600, .png and .pdf)
    └── tables/           # result tables written here (.csv)
```

## Research questions

| Notebook | RQ | Question |
| --- | --- | --- |
| `00_data_prep` | — | Dataset schema, label rates, timestamp span, conversion-delay distribution |
| `01_detection` | RQ1 | Does the standalone-AUROC oracle detect injected leakage vs. leak strength? |
| `01_detection` | RQ2 | Where does the oracle itself fail (aggregation-diluted leakage)? |
| `02_robustness` | RQ3 | With leakage present, which model degrades least? |
| `02_robustness` | RQ4 | Without leakage, does the proposed model match baselines (no accuracy cost)? |
| `03_ablation` | RQ5 | Contribution of the monotonicity penalty, leakage regularizer, cross-stage attention |
| `04_guards` | RQ6 | Firing / false-alarm rates of feature pinning and degradation gating |

## Data

The experiments use a public multi-stage conversion dataset with real impression and conversion
timestamps (click → conversion funnel). The dataset is **not redistributed** here. Obtain it from its
original public source and place a preprocessed Parquet file at:

```
data/criteo_clean.parquet
```

The expected schema (per row): impression `timestamp`, `conversion`, `conversion_timestamp`
(`-1` if no conversion), `click`, `campaign`, dense features (`cost`, `cpo`,
`time_since_last_click`, `click_pos`, `click_nb`), categorical features (`cat1`–`cat9`), and derived
columns `day` (= `timestamp // 86400`) and a temporal `split` label (`train` / `valid` / `test`).

A preprocessing recipe is described in the paper's appendix; any dataset matching this schema can be
dropped in without code changes.

## Environment

```
pip install -r requirements.txt
```

Install PyTorch separately for your platform/CUDA (see https://pytorch.org/get-started/locally/):

```
# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
# CPU only
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Tested with Python 3.9–3.11, PyTorch ≥ 2.1.

## How to run

Each notebook is **self-contained in a single cell**: open it and run the one cell. Run the notebooks
with `notebooks/` as the working directory so the relative `../data` and `../results` paths resolve.
Recommended order: `00 → 01 → 02 → 03 → 04`. Notebooks `02` and `03` train models and benefit from a
GPU; `00`, `01`, and `04` run on CPU.

Each notebook writes its figures to `results/figures/` (as `.png` and `.pdf`, grayscale, dpi 600) and
its tables to `results/tables/` (as `.csv`).

To control dataset size, epochs, and the number of seeds, edit the `CFG` dictionary at the top of each
notebook's cell (`CFG["train"]["max_rows"]`, `CFG["train"]["epochs"]`, `CFG["eval"]["n_seeds"]`). The
defaults use the full dataset, 3 epochs, and 5 seeds.

## Speed

Notebooks `00`, `01`, and `04` are vectorized NumPy/pandas and finish in seconds. Notebooks `02` and
`03` train models (`02`: 4 models × ≥5 seeds; `03`: 4 variants × ≥5 seeds) and are the expensive part.
Two things dominate wall-clock time:

1. **Device.** `02`/`03` print the compute device at the top of their output via `report_device()`.
   If it says `device: cpu` on a machine with a GPU, the installed PyTorch is a **CPU-only build** —
   reinstall it with the CUDA index URL (see "Environment"). This is the most common cause of a slow
   run; training on CPU is far slower than on GPU.
2. **Workload.** The defaults run the full dataset, 3 epochs, 5 seeds. For a quick pass, set the
   `FAST` switch at the top of the cell to `True`:

   ```python
   FAST = True   # subsample rows, fewer epochs/seeds — for iterating or sanity checks
   ```

   `FAST = False` (default) is the full, paper-grade run. Leave it `False` for the numbers you report.

The feature tokenizer is vectorized (a single embedding gather for all categorical fields and one
broadcast for dense features), batches are large, and host→device transfers are non-blocking with
`cudnn.benchmark` enabled, so most of the remaining cost on a GPU is the Transformer encoder itself.

## Reproducibility notes

- All randomness is seeded (`GLOBAL_SEED`); model experiments repeat across ≥ 5 seeds.
- Temporal (not random) train/validation/test splits are used throughout.
- The leakage-injection harness records exact ground truth (which features are leaky and how much),
  so detection metrics are computed against a known answer.
- Two leakage families are provided: a *row-level* leak that the detection oracle catches, and an
  *aggregation-diluted* leak that evades it — the latter is a deliberate hard negative examined in RQ2.

## License

Released for anonymous review. A license will be specified upon de-anonymization.
