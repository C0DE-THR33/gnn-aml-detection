"""
Stage 1 (SRS §4): Data Ingestion & Graph Construction.

Usage:
    python scripts/build_graph.py --config configs/heterophily_gnn_hismall.yaml
"""

import argparse
import pickle
from pathlib import Path

import torch
import yaml

from src.data.graph_builder import build_transaction_graph
from src.data.load_raw import load_transactions, parse_patterns
from src.data.make_splits import make_temporal_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    ds = config["dataset"]
    print(f"Loading transactions from {ds['trans_path']} ...")
    trans = load_transactions(ds["trans_path"])
    print(f"  {len(trans):,} transactions loaded.")

    print(f"Parsing patterns from {ds['patterns_path']} ...")
    patterns = parse_patterns(ds["patterns_path"])
    print(f"  {len(patterns):,} pattern-labeled transaction rows across "
          f"{patterns['pattern_group_id'].nunique() if not patterns.empty else 0} pattern groups.")

    print("Building graph ...")
    result = build_transaction_graph(trans, patterns)
    print(f"  {result.data.num_nodes:,} nodes, {result.data.edge_index.shape[1]:,} edges, "
          f"{int(result.data.y.sum()):,} illicit edges.")

    print("Building temporal train/val/test split ...")
    splits = make_temporal_split(
        trans, train_frac=config["split"]["train_frac"], val_frac=config["split"]["val_frac"],
        seed=config["seed"],
    )
    print(f"  train={len(splits['train']):,} val={len(splits['val']):,} test={len(splits['test']):,}")

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = config["dataset"]["name"]
    torch.save(result.data, out_dir / f"{stem}_graph.pt")
    with open(out_dir / f"{stem}_account_maps.pkl", "wb") as f:
        pickle.dump({"account_to_idx": result.account_to_idx, "idx_to_account": result.idx_to_account}, f)
    with open(out_dir / f"{stem}_splits.pkl", "wb") as f:
        pickle.dump(splits, f)

    print(f"Saved graph + splits to {out_dir}/")


if __name__ == "__main__":
    main()
