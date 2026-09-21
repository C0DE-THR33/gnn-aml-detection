# Project Conventions

## Explainable AML Detection using Heterophily-Aware GNNs — CSF 3127

These conventions keep the project consistent and easy to navigate across the data → model → explanation → evaluation pipeline defined in the SRS. Treat this as a living document — update it if a convention changes, don't silently drift from it.

---

## 1. Repository Structure

```
gnn-aml-detection/
├── data/
│   ├── raw/              # untouched HI-Small files, exactly as downloaded
│   ├── interim/          # partially processed (e.g. deduped, normalized)
│   └── processed/        # final graph-ready tensors / PyG Data objects
├── src/
│   ├── data/              # loading, graph construction, feature engineering
│   ├── models/             # GNN architectures (baseline + heterophily-aware)
│   ├── training/            # training loop, loss functions, samplers
│   ├── explain/              # GNNExplainer wrapper, fidelity scoring
│   └── eval/                  # metrics, per-typology breakdown
├── configs/               # YAML configs, one per experiment (see §4)
├── notebooks/              # exploratory work only — see §8
├── scripts/                 # thin CLI entry points that call into src/
├── outputs/
│   ├── checkpoints/
│   ├── metrics/
│   ├── explanations/       # saved subgraph visualizations
│   └── reports/
├── tests/                    # unit tests for graph construction, metrics, etc.
├── CONVENTIONS.md
├── SRS_GNN_AML_Detection.md
└── README.md
```

**Rule:** raw data is never modified in place. Anything derived goes in `interim/` or `processed/`, and both should be regenerable from `raw/` by a script, not hand-edited.

---

## 2. Naming Conventions

| Item | Convention | Example |
|---|---|---|
| Python modules/files | `snake_case` | `graph_builder.py` |
| Classes | `PascalCase` | `HeterophilyGNN`, `TypologyScorer` |
| Functions/variables | `snake_case` | `build_transaction_graph()` |
| Config files | `snake_case`, named after what varies | `heterophily_gnn_hismall.yaml` |
| Experiment run IDs | `YYYYMMDD_model_dataset_tag` | `20260901_heterosage_hismall_baseline` |
| Checkpoints | `{run_id}_epoch{N}.pt` | `20260901_heterosage_hismall_baseline_epoch40.pt` |
| Typology labels (internal) | fixed lowercase set: `fan_out`, `fan_in`, `cycle`, `gather_scatter`, `scatter_gather`, `stack`, `bipartite`, `random` | — |
| Dataset tag (internal) | fixed to `hismall` (HI-Small) throughout configs/run IDs — never `lismall`/`himedium`/etc. | `20260901_heterosage_hismall_baseline` |
| Notebooks | `NN_short-description.ipynb`, numbered by pipeline order | `01_data_exploration.ipynb`, `02_graph_construction.ipynb` |

**Rule:** never hardcode a typology name as a raw string in more than one place — use a shared constant/enum (`src/data/typologies.py`) covering all 8 typologies so a typo doesn't silently create a 9th "typology." Transactions in the `Patterns.txt` "not classified" bucket get their own explicit `unclassified` tag — never bucketed into one of the 8, and excluded from per-typology fidelity scoring.

---

## 3. Code Style

- **Formatting:** PEP 8, enforced with `black` (line length 100) and `isort` for imports.
- **Type hints:** required on all function signatures in `src/` (not required in `notebooks/`).
- **Docstrings:** Google-style, required for every public function/class in `src/`. One-line summary + Args/Returns.
- **No bare `except:`** — catch specific exceptions, especially around dataset loading and explainer calls (GNNExplainer can fail on degenerate subgraphs; that failure should be caught and logged, not crash the run).
- **Random seeds:** every script/notebook that trains a model or samples data must accept and log a `seed` parameter. Default seed is `42` unless a config overrides it.

---

## 4. Configuration Management

- All hyperparameters (learning rate, hidden dims, loss weighting, GNNExplainer epochs/lr) live in a YAML config under `configs/`, never hardcoded in `src/`.
- One config file = one reproducible experiment. Don't reuse a config file for a materially different run — copy it and rename.
- Every config must include: `dataset`, `model`, `seed`, `train` (epochs, lr, batch size), `loss` (type + class weights), and — where relevant — `explain` (GNNExplainer settings).
- Every saved checkpoint and metrics file must record which config produced it (store the config path or a hash in the output metadata).

---

## 5. Experiment Tracking & Logging

- Every training run logs to `outputs/metrics/{run_id}.csv` (or a tracking tool like Weights & Biases / MLflow, if adopted — pick one and use it consistently, don't mix).
- Minimum logged fields per epoch: loss, precision, recall, F1, AUPRC (validation set).
- Per-typology metrics are logged **once at final evaluation**, not per epoch — no need to slow down training loops for a breakdown that's only meaningful at the end.
- Explanation fidelity scores are logged per typology to `outputs/explanations/fidelity_{run_id}.csv`, with one row per sampled transaction (typology, fidelity score, link to saved subgraph image).

---

## 6. Data Conventions

- Train/val/test splits are generated once by a single script (`src/data/make_splits.py`) and saved as index files — never re-split randomly inside training code, or results become non-reproducible run-to-run.
- Splits must avoid leaking the same account across train and test in a way that trivializes prediction (document the split strategy used, e.g., temporal split vs. random edge split).
- Any class-imbalance handling (oversampling, weighting) is applied only to the training split — validation and test sets stay at natural class distribution.

---

## 7. Git Conventions

- **Branches:** `main` stays runnable at all times. Work happens on `feature/<short-description>` branches (e.g., `feature/heterophily-layer`, `feature/typology-fidelity-scoring`).
- **Commits:** imperative mood, scoped prefix where useful — `data:`, `model:`, `explain:`, `eval:`, `docs:`. Example: `model: add heterophily-aware aggregation layer`.
- **Large files:** trained checkpoints and raw data are not committed — add them to `.gitignore` and note download/regeneration instructions in the README instead.

---

## 8. Notebook Conventions

- Notebooks are for exploration and visualization only — no logic that other code depends on should live only in a notebook.
- Any transformation or model logic that proves useful gets promoted into `src/` and imported back into the notebook, not copy-pasted.
- Clear all outputs before committing a notebook, except for the final report/results notebooks, which should be committed with outputs intact so results are visible without re-running.

---

## 9. Report & Writeup Conventions

- Every claim of the form "the model performs well/poorly on typology X" must be backed by a number in `outputs/metrics/` or `outputs/explanations/`, not stated from impression.
- Negative or weak results (e.g., a typology GNNExplainer fails to explain well) are reported, not omitted — the SRS already frames this as an acceptable, honest outcome.
- Figures referenced in the report are saved as standalone files in `outputs/reports/figures/` and referenced by filename, so the report can be regenerated without re-running notebooks.
