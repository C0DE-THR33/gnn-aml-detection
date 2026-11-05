"""
Stage 2-3 (SRS §4): Model Training + Classification Evaluation.

Usage:
    python scripts/train.py --config configs/heterophily_gnn_hismall.yaml
"""

import argparse
import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import yaml

from src.data.graph_builder import add_edge_aggregate_features, check_graph_layout
from src.device import describe_device, resolve_device
from src.eval.metrics import best_f1_threshold, classification_metrics, per_typology_metrics
from src.models.baseline_sage import BaselineSAGE
from src.models.heterophily_gnn import HeterophilyGNN
from src.training.train import TrainConfig, train_model


def build_model(
    model_type: str, in_dim: int, hidden_dim: int, num_layers: int, edge_attr_dim: int
):
    """Construct a model, sizing its edge head from the graph it will see.

    edge_attr_dim is required, not defaulted: it must come from
    data.edge_attr.shape[1] so the model and the graph cannot disagree.
    """
    kwargs = dict(in_dim=in_dim, hidden_dim=hidden_dim, num_layers=num_layers,
                  edge_attr_dim=edge_attr_dim)
    if model_type == "heterophily_gnn":
        return HeterophilyGNN(**kwargs)
    if model_type == "baseline_sage":
        return BaselineSAGE(**kwargs)
    raise ValueError(f"Unknown model.type '{model_type}' — expected 'heterophily_gnn' or 'baseline_sage'.")


def load_graph(config: dict):
    """Load the saved graph and splits, add node features, move to device."""
    stem = config["dataset"]["name"]
    data = torch.load(f"data/processed/{stem}_graph.pt", weights_only=False)
    with open(f"data/processed/{stem}_splits.pkl", "rb") as f:
        splits = pickle.load(f)

    check_graph_layout(data)
    add_edge_aggregate_features(data)
    device = resolve_device(config)
    print(f"Device: {describe_device(device)}")
    return data.to(device), splits, device


def score(model, data, splits) -> tuple[dict, pd.DataFrame]:
    """Test metrics at a threshold chosen on val, plus the 0.5 ones for comparison.

    Returns:
        (overall, per_typology). overall holds precision/recall/f1 at the
        val-tuned threshold, the same three at 0.5 suffixed `_at_0.5`, and
        `threshold` itself. AUPRC does not depend on the threshold.
    """
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index, data.edge_attr)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

    y = data.y.cpu().numpy()
    val_idx, test_idx = splits["val"], splits["test"]
    threshold = best_f1_threshold(y[val_idx], probs[val_idx])
    pred = (probs >= threshold).astype(int)

    overall = classification_metrics(y[test_idx], pred[test_idx], probs[test_idx])
    at_half = classification_metrics(y[test_idx], (probs[test_idx] >= 0.5).astype(int), probs[test_idx])
    overall.update({f"{k}_at_0.5": at_half[k] for k in ("precision", "recall", "f1")})
    overall["threshold"] = threshold

    per_typ = per_typology_metrics(
        y[test_idx], pred[test_idx], probs[test_idx],
        [data.edge_typology[i] for i in test_idx],
    )
    return overall, per_typ


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override config seed. CUDA's scatter reductions are "
        "non-deterministic, so repeat runs under several seeds and report a "
        "range rather than a point estimate (CONVENTIONS.md §3).",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    data, splits, device = load_graph(config)

    model = build_model(
        config["model"]["type"], in_dim=data.x.shape[1], edge_attr_dim=data.edge_attr.shape[1],
        hidden_dim=config["model"]["hidden_dim"], num_layers=config["model"]["num_layers"],
    ).to(device)

    seed = args.seed if args.seed is not None else config["seed"]
    print(f"Seed: {seed}  |  model: {config['model']['type']}")
    train_config = TrainConfig(
        epochs=config["train"]["epochs"], lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"], loss_type=config["loss"]["type"],
        seed=seed, log_every=config["train"]["log_every"],
        warmup_epochs=config["train"].get("warmup_epochs", 0),
    )
    result = train_model(model, data, splits, train_config)

    # Seed in the run_id so a multi-seed sweep does not overwrite itself
    # (CONVENTIONS.md §2 run-ID format, §4 one config = one experiment).
    run_id = f"{datetime.now():%Y%m%d}_{config['run_id_prefix']}_seed{seed}"

    overall, per_typ = score(model, data, splits)
    # Header text and the dict on the next line are parsed by the notebook.
    print("\n--- Overall test metrics ---")
    print(overall)
    print(f"(threshold {overall['threshold']:.4f}, chosen to maximise F1 on val)")
    print("\n--- Per-typology test metrics ---")
    print(per_typ)

    Path("outputs/checkpoints").mkdir(parents=True, exist_ok=True)
    Path("outputs/metrics").mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "config_path": args.config,
            "config": config,
            # explain.py selects true positives at this threshold.
            "threshold": overall["threshold"],
        },
        f"outputs/checkpoints/{run_id}.pt",
    )
    per_typ.to_csv(f"outputs/metrics/{run_id}_per_typology.csv")
    pd.DataFrame(result.history).to_csv(f"outputs/metrics/{run_id}.csv", index=False)

    # Overall test metrics were previously printed only, which made comparing
    # runs a log-parsing job. One row per run, tagged with what produced it, so
    # a multi-model multi-seed sweep pools with a glob (CONVENTIONS.md §4, §5).
    overall_row = {
        "run_id": run_id,
        "model": config["model"]["type"],
        "seed": seed,
        "config_path": args.config,
        # Test metrics and the checkpoint come from this epoch, not the last.
        "best_epoch": result.best_epoch,
        "best_val_auprc": result.best_val_auprc,
        **overall,
    }
    pd.DataFrame([overall_row]).to_csv(f"outputs/metrics/{run_id}_overall.csv", index=False)

    print(f"\nSaved checkpoint outputs/checkpoints/{run_id}.pt")
    print(f"Saved per-typology metrics outputs/metrics/{run_id}_per_typology.csv")
    print(f"Saved overall test metrics outputs/metrics/{run_id}_overall.csv")


if __name__ == "__main__":
    main()
