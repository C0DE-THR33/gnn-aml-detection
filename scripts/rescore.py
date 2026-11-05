"""
Re-score saved checkpoints with a decision threshold chosen on val.

Runs that predate the tuned threshold were test-scored at 0.5. This recomputes
their precision/recall/F1 without retraining, and writes the threshold into
each checkpoint so explain.py selects true positives at it.

Usage:
    python scripts/rescore.py outputs/checkpoints/*.pt

Overwrites outputs/metrics/<run_id>_overall.csv and _per_typology.csv for each
checkpoint, keeping the training-time columns (best_epoch, best_val_auprc) of
an existing overall row.
"""

import argparse
from pathlib import Path

import pandas as pd
import torch

from scripts.train import build_model, load_graph, score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+")
    args = parser.parse_args()

    data = None
    for path in sorted(args.checkpoints):
        ckpt = torch.load(path, weights_only=False, map_location="cpu")
        config = ckpt["config"]
        if data is None:  # every checkpoint here is HI-Small; load the graph once
            data, splits, device = load_graph(config)

        model = build_model(
            config["model"]["type"], in_dim=data.x.shape[1], edge_attr_dim=data.edge_attr.shape[1],
            hidden_dim=config["model"]["hidden_dim"], num_layers=config["model"]["num_layers"],
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
        overall, per_typ = score(model, data, splits)

        run_id = Path(path).stem
        overall_path = Path(f"outputs/metrics/{run_id}_overall.csv")
        row = pd.read_csv(overall_path).iloc[0].to_dict() if overall_path.exists() else {
            "run_id": run_id, "model": config["model"]["type"], "config_path": ckpt.get("config_path"),
        }
        row.update(overall)
        pd.DataFrame([row]).to_csv(overall_path, index=False)
        per_typ.to_csv(f"outputs/metrics/{run_id}_per_typology.csv")

        ckpt["threshold"] = overall["threshold"]
        torch.save(ckpt, path)

        print(
            f"{run_id}: threshold {overall['threshold']:.4f} | "
            f"precision {overall['precision']:.4f} recall {overall['recall']:.4f} f1 {overall['f1']:.4f} | "
            f"at 0.5: f1 {overall['f1_at_0.5']:.4f} | auprc {overall['auprc']:.4f}"
        )


if __name__ == "__main__":
    main()
