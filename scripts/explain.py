"""
Stage 4-5 (SRS §4): Explanation Generation + Explanation Evaluation.

Usage:
    python scripts/explain.py --config configs/heterophily_gnn_hismall.yaml \\
        --checkpoint outputs/checkpoints/<run_id>.pt
"""

import argparse
import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import yaml

from src.data.graph_builder import check_graph_layout
from src.device import describe_device, resolve_device
from src.explain.fidelity import score_typology_fidelity
from src.explain.run_explainer import build_explainer, explain_edge, select_explanation_sample
from src.explain.visualize import draw_explanation_subgraph, plot_fidelity_by_typology
from scripts.train import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override config seed. GNNExplainer's mask optimisation is "
        "stochastic, so repeat runs under different seeds to get an "
        "uncertainty estimate on fidelity (CONVENTIONS.md §3).",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    stem = config["dataset"]["name"]
    data = torch.load(f"data/processed/{stem}_graph.pt", weights_only=False)
    with open(f"data/processed/{stem}_splits.pkl", "rb") as f:
        splits = pickle.load(f)

    check_graph_layout(data)
    device = resolve_device(config)
    print(f"Device: {describe_device(device)}")
    data = data.to(device)

    model = build_model(
        config["model"]["type"], in_dim=data.x.shape[1], edge_attr_dim=data.edge_attr.shape[1],
        hidden_dim=config["model"]["hidden_dim"], num_layers=config["model"]["num_layers"],
    ).to(device)
    # Checkpoints carry their producing config alongside the weights
    # (CONVENTIONS.md §4), so unwrap rather than assuming a bare state_dict.
    ckpt = torch.load(args.checkpoint, weights_only=False)
    model.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
    model.eval()

    with torch.no_grad():
        logits = model(data.x, data.edge_index, data.edge_attr)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        pred = (probs >= 0.5).astype(int)

    test_idx = splits["test"]
    sample = select_explanation_sample(
        data, pred[test_idx], test_idx, per_typology_cap=config["explain"]["per_typology_cap"]
    )
    print(f"Explaining {len(sample)} true-positive illicit transactions ...")

    explainer = build_explainer(model, explain_epochs=config["explain"]["explain_epochs"])

    num_hops = config["model"]["num_layers"]
    edge_index_cpu = data.edge_index.cpu()

    # The explainer's mask optimisation is stochastic. Seed it, and carry the
    # seed into every output name so repeat runs sit side by side instead of
    # overwriting each other (CONVENTIONS.md §3, §4).
    seed = args.seed if args.seed is not None else config["seed"]
    torch.manual_seed(seed)
    print(f"Explainer seed: {seed}")
    # Dated like train.py's run IDs (CONVENTIONS.md §2), so a later sweep's
    # fidelity CSV and report figure don't overwrite an earlier one's.
    run_id = f"{datetime.now():%Y%m%d}_{config['run_id_prefix']}_seed{seed}"
    save_figures = config["explain"].get("save_subgraph_figures", True)
    fig_dir = Path("outputs/explanations/figures")

    rows = []
    skipped = []
    for i, edge_idx in enumerate(sample):
        # GNNExplainer can legitimately fail on a degenerate or oversized
        # neighbourhood. Log and carry on rather than losing the whole run
        # (CONVENTIONS.md §3: no bare except, catch what you expect).
        try:
            # Re-seed per instance so a given edge's explanation does not
            # depend on how many edges were explained before it.
            torch.manual_seed(seed + i)
            exp = explain_edge(explainer, data, edge_idx, num_hops=num_hops)
        except (RuntimeError, ValueError, IndexError) as err:
            skipped.append({"edge_idx": edge_idx, "reason": str(err)})
            print(f"  [skip] {err}")
            continue

        top_k = config["explain"]["top_k_edges"]
        fidelity = score_typology_fidelity(
            edge_index_cpu, exp.edge_mask, exp.typology, top_k=top_k
        )

        # CONVENTIONS.md §5: every fidelity row links to its subgraph image.
        # Same top_k as the score above, so figure and number agree.
        image_path = ""
        if save_figures:
            image_path = str(
                draw_explanation_subgraph(
                    edge_index_cpu,
                    exp.edge_mask,
                    edge_idx,
                    fig_dir / f"{run_id}_edge{edge_idx}_{exp.typology or 'untyped'}.png",
                    typology=exp.typology,
                    fidelity=fidelity,
                    top_k=top_k,
                    seed=config["seed"],
                )
            )

        rows.append(
            {
                "edge_idx": edge_idx,
                "typology": exp.typology,
                "fidelity": fidelity,
                "subgraph_edges": exp.subgraph_edges,
                "subgraph_image": image_path,
            }
        )
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(sample)} processed ({len(rows)} explained, {len(skipped)} skipped)")

    if skipped:
        print(f"\n{len(skipped)} of {len(sample)} instances skipped — see the [skip] lines above.")
    if not rows:
        raise SystemExit("No instances could be explained; nothing to score.")

    df = pd.DataFrame(rows)
    print("\n--- Fidelity by typology (mean, count) ---")
    print(df.groupby("typology")["fidelity"].agg(["mean", "count"]))
    print(
        f"\nComputation subgraphs: median {int(df['subgraph_edges'].median()):,} edges, "
        f"max {int(df['subgraph_edges'].max()):,} (full graph is {data.edge_index.shape[1]:,})."
    )

    out_path = f"outputs/explanations/fidelity_{run_id}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")

    if save_figures:
        # CONVENTIONS.md §9: report figures are standalone files under
        # outputs/reports/figures/, referenced by name, so the writeup can be
        # regenerated without re-running anything.
        summary = plot_fidelity_by_typology(
            df, Path("outputs/reports/figures") / f"{run_id}_fidelity_by_typology.png"
        )
        print(f"Saved {len(df)} subgraph figures to {fig_dir}/")
        print(f"Saved {summary}")


if __name__ == "__main__":
    main()
