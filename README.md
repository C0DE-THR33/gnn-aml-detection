# Explainable AML Detection using Heterophily-Aware GNNs

CSF 3127 (AI in Finance) course project — see `SRS_GNN_AML_Detection.md` for
the full spec and `CONVENTIONS.md` for coding/naming conventions.

## Status

The full pipeline (data loading → graph construction → training → per-typology
evaluation → GNNExplainer → per-typology fidelity scoring) is implemented and
tested end-to-end against a **synthetic fixture** that matches the real
HI-Small schema. It has **not yet been run against the real Kaggle data** —
that's the first thing to do (see below).

## First thing to do

1. Download HI-Small from Kaggle: [IBM Transactions for Anti-Money
   Laundering (AML)](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml).
   You need `HI-Small_Trans.csv` and `HI-Small_Patterns.txt` at minimum;
   place an `HI-Small_accounts.csv` too if your download includes one.
2. Put them in `data/raw/` (already gitignored / empty except `.gitkeep`).
3. **Check `src/data/load_raw.py`'s assumptions against the real files
   before trusting anything downstream:**
   - `load_transactions()` assumes the standard 11-column Trans.csv layout
     — it'll raise a clear error if the column count doesn't match.
   - `load_accounts()` is written defensively (it doesn't know the real
     accounts.csv schema for certain) — it'll print the columns it found
     and raise if it can't identify an account-ID column. Update
     `ACCOUNT_ID_CANDIDATES`/`BANK_ID_CANDIDATES` in that file if needed.
   - `parse_patterns()` assumes the confirmed `BEGIN/END LAUNDERING
     ATTEMPT - <TYPOLOGY>` block format — sanity-check a few parsed rows
     against the raw file by eye the first time.
4. Run the pipeline (see below).

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Scripts import from `src/` using absolute imports (`from src.data...`), so
run everything from the repo root with the repo root on `PYTHONPATH`:

```bash
export PYTHONPATH=.
```

## Running the pipeline

```bash
# 1. Data ingestion + graph construction (SRS §4 stage 1)
python scripts/build_graph.py --config configs/heterophily_gnn_hismall.yaml

# 2. Model training + classification evaluation (stages 2-3)
python scripts/train.py --config configs/heterophily_gnn_hismall.yaml
# -> prints overall + per-typology test metrics, saves a checkpoint to
#    outputs/checkpoints/ and per-typology metrics to outputs/metrics/

# 3. Explanation generation + fidelity evaluation (stages 4-5)
python scripts/explain.py --config configs/heterophily_gnn_hismall.yaml \
    --checkpoint outputs/checkpoints/<run_id_from_step_2>.pt
# -> saves per-edge fidelity scores to outputs/explanations/
```

## Testing without the real data

`tests/make_synthetic_fixture.py` generates a small fixture matching the
real schema (Trans.csv/accounts.csv/Patterns.txt, all 8 typologies) so the
whole pipeline can be exercised before the Kaggle download:

```bash
python tests/make_synthetic_fixture.py
pytest tests/ -v
```

All 10 tests currently pass against the fixture. **This validates the code
runs correctly — it says nothing about model performance on real data**,
since the fixture's "laundering patterns" are randomly generated, not
learned signal.

## A few things worth knowing before you dig into the code

- **The heterophily-aware layer had to be built on `MessagePassing`, not a
  hand-rolled scatter op** — GNNExplainer's edge-mask hook only works on
  layers that go through `propagate()`/`message()`. See the docstring in
  `src/models/heterophily_gnn.py` if you extend the architecture; a
  from-scratch aggregation layer will silently break explainability.
- **Node features are minimal for now** (in/out-degree only) — real
  account-level features depend on confirming `accounts.csv`'s actual
  schema against step 3 above.
- **Message passing runs over the full graph; only loss/metrics are split**
  by time. This is documented as a known limitation in the SRS (§2.5,
  §6.3) rather than solved — a fully temporal formulation is out of scope
  per SRS §1.2.
- **Typology.RANDOM has no fidelity scorer on purpose** — see
  `src/explain/fidelity.py`'s module docstring for why "not scored" is the
  honest answer, not a bug.
