# Explainable AML Detection using Heterophily-Aware GNNs

CSF 3127 (AI in Finance) course project. See `SRS_GNN_AML_Detection.md` for the
full spec and `CONVENTIONS.md` for coding, naming and git conventions.

## Status

The full pipeline (data loading → graph construction → training → per-typology
evaluation → GNNExplainer → per-typology fidelity) runs end to end on the real
**HI-Small** data: 5,078,345 transactions, 515,080 accounts, 5,177 illicit
transactions (0.102%). Training and explanation ran on a Colab T4.

- **Last completed sweep** (2026-10-05): 300 epochs, edge features = log-amount
  + payment-format one-hot, checkpoint taken at the best val-AUPRC epoch.
  Results below.
- **Previous sweep** (2026-09-05): 100 epochs, log-amount only. Kept for
  comparison.

## Results so far

Test split: 1,015,669 edges, 1,797 illicit (0.177%), so the AUPRC of a random
ranker is **0.00177**. Three seeds per architecture, same graph, same splits,
configs identical except `model.type` (enforced by `tests/test_configs.py`).

| model | sweep | test AUPRC (mean, range) | lift over chance | recall | F1 |
|---|---|---|---|---|---|
| `heterophily_gnn` | 2026-10-05 | **0.0917** (0.0868–0.0976) | 51.8x | 0.956 | 0.0227 |
| `baseline_sage` | 2026-10-05 | **0.0800** (0.0752–0.0871) | 45.2x | 0.979 | 0.0199 |
| `heterophily_gnn` | 2026-09-05 | 0.0371 (0.0357–0.0383) | 21.0x | 0.832 | 0.0152 |
| `baseline_sage` | 2026-09-05 | 0.0263 (0.0240–0.0278) | 14.8x | 0.794 | 0.0136 |

- **Both changes helped, and both models are still undertrained.** At epoch 100
  (no LR schedule, so like for like with the previous sweep) payment format alone
  raised val AUPRC +29% for the heterophily model and +44% for the baseline.
  Epochs 100→300 added another 55–75%. All three heterophily runs had their best
  val AUPRC at epoch 300 and every run was still rising over epochs 250–300, so
  these numbers are a floor.
- **The heterophily model still leads, but the gap narrowed** from +41.5% to
  **+14.6%** relative AUPRC. The gap (0.0117) is now about equal to the largest
  within-model seed spread (0.97x, was 2.9x), and the ranges touch: the worst
  heterophily seed (0.0868) is below the best baseline seed (0.0871). Payment
  format helped the baseline more, so part of the earlier gap was the baseline
  lacking that signal. n=3 per arm cannot separate the two models.
- **Per typology the result is split.** The heterophily model wins 4 of 8
  (fan_out 1.72x, gather_scatter 1.58x, scatter_gather 1.30x, random 1.24x), ties
  fan_in (0.99x), and loses bipartite (0.66x), cycle (0.73x) and stack (0.89x).
  In the previous sweep it won all eight.
- **The detection/fidelity inverse relation did not replicate.** The previous
  sweep had Spearman(detection lift, fidelity) = −0.893 (p = 0.007, n = 7); this
  sweep gives +0.214 (p = 0.645). Treat the earlier finding as not established.
- **Fidelity rose for the fan and gather/scatter typologies** (fan_out 0.37→0.47,
  gather_scatter 0.40→0.49, scatter_gather 0.16→0.22, fan_in 0.46→0.51) and
  stayed near zero for cycle (0.04), stack (0.07) and bipartite (0.14). Fidelity
  is measured on one checkpoint (heterophily, seed 42) under three explainer
  seeds (n = 150 per typology, seed spread 0.000–0.057). The two sweeps explain
  different true positives, so these are not paired comparisons.
- **Do not quote accuracy.** A model that always predicts "licit" scores 99.82%;
  the real models score ~83–85% because `pos_weight` (~1,325) pushes them to flag
  ~15% of traffic, so precision is ~1%. Lead with AUPRC against the base rate.

Both sweeps are versioned: metrics CSVs (`outputs/metrics/{date}_*`), fidelity
CSVs (`outputs/explanations/fidelity_{date}_*.csv`) and report figures
(`outputs/reports/figures/{date}_*`). `explain.py` writes the fidelity CSV and
figure without a date; the date prefix is added when a sweep is versioned.
Checkpoints, logs and per-transaction explanation images stay gitignored
(CONVENTIONS §7).

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=.   # scripts use absolute imports (from src.data...)
```

Developed on Python 3.12 locally and 3.13 on Colab. On a CPU-only machine,
install torch from the CPU wheel index to avoid the large CUDA download.

**Hardware.** Graph construction and the test suite run fine on a laptop. Full
training does not: it is full-batch over 5.08M edges, and on a 7.7 GB RAM machine
a run had not reached its first log point (epoch 10) after ~7 minutes and was
paging heavily, so it was stopped. Use a GPU; the Colab notebook below is set
up for a T4.

## Data

Download HI-Small from Kaggle ([IBM Transactions for Anti-Money Laundering
(AML)](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml))
and place `HI-Small_Trans.csv`, `HI-Small_Patterns.txt` and
`HI-Small_accounts.csv` in `data/raw/` (gitignored).

Checked against the real files: Trans.csv has the assumed 11 columns;
Patterns.txt has all 8 typology labels and they map cleanly; accounts.csv has
`Bank Name, Bank ID, Account Number, Entity ID, Entity Name` and now loads with
unique column names (it previously returned two columns called `bank_id`).

## Running the pipeline

```bash
# 1. Graph construction (~300 MB output in data/processed/)
python scripts/build_graph.py --config configs/heterophily_gnn_hismall.yaml

# 2. Train + evaluate. --seed overrides the config; it is part of the run id.
python scripts/train.py --config configs/heterophily_gnn_hismall.yaml --seed 42
python scripts/train.py --config configs/baseline_sage_hismall.yaml   --seed 42

# 3. Explain + score fidelity (also takes --seed: GNNExplainer is stochastic)
python scripts/explain.py --config configs/heterophily_gnn_hismall.yaml \
    --checkpoint outputs/checkpoints/<run_id>.pt --seed 42
```

| config | purpose |
|---|---|
| `heterophily_gnn_hismall.yaml` | the model under study, full graph |
| `baseline_sage_hismall.yaml` | homophily control (SRS FR-5), full graph |

Outputs land in `outputs/{checkpoints,metrics,explanations,reports}/`, named
`{date}_{run_id_prefix}_seed{N}`. Each `*_overall.csv` holds one run's test
metrics (with model, seed and config path) so a sweep pools with a glob.
Training keeps the weights from the log point with the best val AUPRC, so the
checkpoint and test metrics come from that epoch (`best_epoch` in
`*_overall.csv`), not necessarily the last.

### On Colab

```bash
python scripts/make_colab_bundle.py   # -> colab/gnn_aml_code.zip
```

Put `gnn_aml_code.zip`, `hismall_graph.pt` and `hismall_splits.pkl` (from
`data/processed/`) plus `colab/run_on_colab.ipynb` in `MyDrive/gnn-aml/`, choose a
**T4 GPU** runtime, and run the cells top to bottom. The notebook trains both
architectures over three seeds, pools the results, then explains the heterophily
checkpoint under three explainer seeds. Each finished run is synced to Drive so a
disconnect loses at most the run in progress.

The graph must be rebuilt whenever the edge feature layout changes. `train.py`
and `explain.py` check this and stop with a clear error on a stale graph.

## Tests

```bash
python tests/make_synthetic_fixture.py   # only if tests/fixtures/ is missing
pytest tests/ -v
```

47 tests against a small synthetic fixture matching the HI-Small schema. They
verify the code runs and that specific past defects stay fixed (fidelity scorer
bugs, the accounts loader, the Colab bundle, config drift). **They say nothing
about model performance on real data**: the fixture's laundering patterns are
random.

## Design notes

### Edge features

`edge_attr` is `[log-amount | payment-format one-hot]`, 9 columns. Payment format
is the strongest signal in the data: the illicit rate is 0.75% for ACH against
0.00% for Wire and Reinvestment (a ~42x spread), and the encoded columns
reproduce those per-format counts exactly on the real graph. Currency was checked
and left out (0.09%–0.42%, ~4x, and 15 categories). The one-hot uses the fixed
list `load_raw.PAYMENT_FORMATS`, so the layout is identical across the real data,
the fixture and any subsample.

### Things worth knowing before you change code

- **The heterophily layer must be built on `MessagePassing`,** not a hand-rolled
  scatter: GNNExplainer's edge-mask hook only sees `propagate()`/`message()`. See
  `src/models/heterophily_gnn.py`.
- **Explanations run on each edge's L-hop subgraph, not the full graph.** The full
  graph exhausts a 15 GB GPU; a test asserts subgraph logits equal full-graph
  logits. Neighbourhoods over 400,000 edges (hub accounts) are skipped, about 10
  instances per seed.
- **Fidelity scorers are heuristic proxies** for "looks like this typology", not
  graph isomorphism. `random` and `unclassified` are deliberately unscored.
- **Per-typology metrics** score a typology's illicit edges against all licit
  edges. Scoring it against its own edges alone pins precision and AUPRC at 1.0.
- **Training is not bit-reproducible on GPU** (CUDA scatter reductions are
  non-deterministic). One config produced AUPRC anywhere from 0.026 to 0.040
  across runs, so report a range over at least three seeds.

### Known limitations

- **Temporal leakage.** Message passing and the in/out-degree node features use
  the full graph, including test edges, so the lifts above are optimistic by an
  unknown amount (SRS §2.5, §6.3; a temporal formulation is out of scope per §1.2).
- **The split is not prevalence-matched.** Illicit density rises across the
  window: train 0.075%, val 0.107%, test 0.177%. Train and val metrics are not
  directly comparable.
- **Node features are in/out-degree only.** `accounts.csv` (entity type, bank) is
  loaded correctly but unused.
- **Account identity is the account number alone.** 8 account numbers exist under
  two different banks, so 8 pairs of distinct accounts are merged into 8 nodes
  (515,080 nodes vs 515,088 distinct bank+account pairs; 0.0016%).

### Next steps

1. Train longer or with a higher learning rate (currently 0.001, no schedule):
   every heterophily run peaked at its last epoch.
2. More seeds per model (e.g. 5): with a +14.6% gap and touching ranges, three
   seeds cannot say whether the heterophily model's lead is real.
3. Account-level node features from `accounts.csv`.
4. Remove the degree leakage.
