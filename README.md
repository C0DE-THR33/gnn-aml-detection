# Explainable AML Detection using Heterophily-Aware GNNs

CSF 3127 (AI in Finance) course project. See `SRS_GNN_AML_Detection.md` for the
full spec and `CONVENTIONS.md` for coding, naming and git conventions.

## Status

The full pipeline (data loading → graph construction → training → per-typology
evaluation → GNNExplainer → per-typology fidelity) runs end to end on the real
**HI-Small** data: 5,078,345 transactions, 515,080 accounts, 5,177 illicit
transactions (0.102%). Training and explanation ran on a Colab T4.

- **Last completed sweep** (2026-10-23 to 25): node features now include each
  account's mean/max incoming and outgoing edge features (x: 2 -> 38 columns),
  1000 epochs at peak lr 0.003 with 30-epoch warmup + cosine decay, and the
  decision threshold chosen on val instead of 0.5. Six runs per architecture.
  Results below.
- **Earlier sweeps** (2026-10-05: 300 epochs, payment format added; 2026-09-05:
  100 epochs, log-amount only). Kept for comparison.

## Results so far

Test split: 1,015,669 edges, 1,797 illicit (0.177%), so the AUPRC of a random
ranker is **0.00177**. Same graph, same splits, configs identical except
`model.type` (enforced by `tests/test_configs.py`).

| model | sweep | runs | test AUPRC (mean, range) | lift over chance | precision | recall | F1 |
|---|---|---|---|---|---|---|---|
| `heterophily_gnn` | 2026-10-23 | 6 | **0.257** (0.167–0.313) | 145x | 0.313 | 0.352 | **0.324** |
| `baseline_sage` | 2026-10-23 | 6 | **0.195** (0.170–0.225) | 110x | 0.219 | 0.407 | **0.282** |
| `heterophily_gnn` | 2026-10-05 | 3 | 0.0917 (0.0868–0.0976) | 51.8x | 0.012 | 0.956 | 0.0227 |
| `baseline_sage` | 2026-10-05 | 3 | 0.0800 (0.0752–0.0871) | 45.2x | 0.010 | 0.979 | 0.0199 |
| `heterophily_gnn` | 2026-09-05 | 3 | 0.0371 (0.0357–0.0383) | 21.0x | | 0.832 | 0.0152 |
| `baseline_sage` | 2026-09-05 | 3 | 0.0263 (0.0240–0.0278) | 14.8x | | 0.794 | 0.0136 |

2026-10-23 precision/recall/F1 are at the threshold that maximises F1 on val
(0.989–0.998); earlier rows are at 0.5. At 0.5 the 2026-10-23 runs score F1
0.047 (heterophily) and 0.034 (baseline).

- **The underfitting is fixed.** AUPRC rose 2.4–2.8x over the 2026-10-05 sweep.
  The model only ever saw degree in message passing before; the edge-feature
  aggregates and 3.3x more optimizer steps changed together, so the gain can't
  be split between them. Train AUPRC now exceeds val (it never did before), and
  val AUPRC is flat over the last ~150 epochs as the lr decays to 0.
- **The heterophily model leads, but only just separably.** +0.062 AUPRC (+32%
  relative), Welch p = 0.034; a Mann-Whitney U test gives p = 0.093, and the
  ranges overlap (worst heterophily run 0.167, best baseline 0.225). F1 +0.043,
  p = 0.089. The heterophily model trades recall (0.35 vs 0.41) for precision
  (0.31 vs 0.22).
- **The heterophily model is much less stable.** Its seed-to-seed std is 2.5x the
  baseline's (0.053 vs 0.021). Seed 42 trained twice with identical code scored
  0.262 and 0.168 (CUDA non-determinism), and one run peaked at epoch 230 of
  1000. A lower peak lr or gradient clipping may tighten it.
- **Per typology the heterophily model wins 7 of 8** on mean AUPRC: fan_out
  3.30x, gather_scatter 2.35x, scatter_gather 2.15x, random 1.60x, cycle 1.49x,
  stack 1.24x, bipartite 1.13x; fan_in 0.95x. The large wins are the high-count
  fan/gather/scatter typologies.
- **Fidelity has three tiers.** One checkpoint (heterophily, 2026-10-23 seed 42,
  at its tuned threshold) under three explainer seeds:

  | typology | mean fidelity | 95% CI | n | seed spread |
  |---|---|---|---|---|
  | bipartite | 0.348 | ±0.116 | 66 | 0.136 |
  | fan_out | 0.346 | ±0.041 | 150 | 0.023 |
  | gather_scatter | 0.233 | ±0.027 | 150 | 0.053 |
  | fan_in | 0.228 | ±0.024 | 93 | 0.027 |
  | scatter_gather | 0.077 | ±0.012 | 150 | 0.007 |
  | stack | 0.067 | ±0.022 | 141 | 0.028 |
  | cycle | 0.013 | ±0.025 | 78 | 0.038 |

  The tuned threshold leaves fewer than 50 true positives for bipartite, cycle
  and fan_in, and bipartite's seed spread is as wide as the gap to the next tier.
  Fidelity is lower than in the 2026-10-05 sweep for most typologies (e.g.
  gather_scatter 0.49 -> 0.23), but the two explain different true positives
  from different models, so this is not a paired comparison.
- **Detection and fidelity are unrelated.** Spearman(per-typology AUPRC lift,
  fidelity) = 0.04 (n = 7), after −0.893 and +0.214 in the two earlier sweeps.
- **Do not quote accuracy.** A model that always predicts "licit" scores 99.82%.
  Lead with AUPRC against the base rate.

All sweeps are versioned: metrics CSVs (`outputs/metrics/{date}_*`), fidelity
CSVs (`outputs/explanations/fidelity_{date}_*.csv`) and report figures
(`outputs/reports/figures/{date}_*`); `train.py` and `explain.py` date them
with the day the run started. Checkpoints, logs and per-transaction
explanation images stay gitignored (CONVENTIONS §7).

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

52 tests against a small synthetic fixture matching the HI-Small schema. They
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
- **Node features are in/out-degree plus edge-feature aggregates.**
  `accounts.csv` (entity type, bank) is loaded correctly but unused. The
  aggregates use all edges, including test edges (same leakage as above).
- **Account identity is the account number alone.** 8 account numbers exist under
  two different banks, so 8 pairs of distinct accounts are merged into 8 nodes
  (515,080 nodes vs 515,088 distinct bank+account pairs; 0.0016%).

### Next steps

1. Stabilise heterophily training (peak lr 0.002 or gradient clipping): its
   seed spread is what keeps the architecture comparison marginal.
2. Ablate the two 2026-10-23 changes (edge aggregates vs longer schedule) to
   attribute the 2.4–2.8x gain.
3. Account-level node features from `accounts.csv`.
4. Remove the degree and aggregate leakage.
