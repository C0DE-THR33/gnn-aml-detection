"""
Explanation subgraph visualisation (SRS FR-12, CONVENTIONS.md §5).

Renders the SAME top-k subgraph that fidelity.py scores — deliberately, via
the shared build_explanation_subgraph() — so the picture in the report and
the number next to it can never drift apart. A figure showing a different
edge set than the one scored would be worse than no figure.

Matplotlib runs on the Agg backend: these are written from headless script
runs (Colab cells, CI), never an interactive session.
"""

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import torch  # noqa: E402

from src.explain.fidelity import build_explanation_subgraph  # noqa: E402


def draw_explanation_subgraph(
    edge_index: torch.Tensor,
    edge_mask: torch.Tensor,
    target_edge_idx: int,
    out_path: str | Path,
    typology: Optional[str] = None,
    fidelity: Optional[float] = None,
    top_k: int = 6,
    seed: int = 42,
) -> Path:
    """Draw one explanation subgraph and save it to disk.

    The target transaction is drawn in red; the remaining edges are the
    top-k GNNExplainer-weighted edges, with width and opacity proportional
    to their mask value, so a reader can see whether the explanation is
    concentrated or diffuse.

    Args:
        edge_index: [2, num_edges] full-graph edge index (CPU).
        edge_mask: [num_edges] importance scores, full-graph indexed.
        target_edge_idx: Global index of the transaction being explained.
        out_path: Where to write the image. Parent dirs are created.
        typology: Typology label for the title, if known.
        fidelity: Fidelity score for the title, if computed.
        top_k: Number of top-importance edges to draw. Must match the value
            passed to score_typology_fidelity() for the figure and the score
            to describe the same subgraph.
        seed: Layout seed — spring_layout is stochastic, and a fixed seed
            keeps figures reproducible across runs (CONVENTIONS.md §3).

    Returns:
        The path written.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    g = build_explanation_subgraph(edge_index, edge_mask, top_k=top_k)
    src, dst = int(edge_index[0, target_edge_idx]), int(edge_index[1, target_edge_idx])
    g.add_edge(src, dst)  # guarantee the explained edge is visible even if not top-k

    importances = {}
    k = min(top_k, edge_mask.numel())
    top_vals, top_idx = edge_mask.topk(k)
    peak = float(top_vals.max()) or 1.0
    for val, idx in zip(top_vals.tolist(), top_idx.tolist()):
        importances[(int(edge_index[0, idx]), int(edge_index[1, idx]))] = val / peak

    fig, ax = plt.subplots(figsize=(6, 5))
    pos = nx.spring_layout(g, seed=seed)

    nx.draw_networkx_nodes(g, pos, ax=ax, node_size=320, node_color="#dbe4f0", edgecolors="#4a5b73")
    nx.draw_networkx_labels(g, pos, ax=ax, font_size=6)

    for u, v in g.edges():
        is_target = (u, v) == (src, dst)
        weight = importances.get((u, v), 0.0)
        nx.draw_networkx_edges(
            g,
            pos,
            ax=ax,
            edgelist=[(u, v)],
            width=1.0 + 3.0 * weight,
            alpha=1.0 if is_target else 0.35 + 0.55 * weight,
            edge_color="#c0392b" if is_target else "#33506e",
            arrowsize=13,
        )

    bits = [f"edge {target_edge_idx}"]
    if typology:
        bits.append(typology)
    if fidelity is not None:
        bits.append(f"fidelity {fidelity:.3f}")
    ax.set_title(" · ".join(bits), fontsize=10)
    ax.text(
        0.5,
        -0.04,
        "red = explained transaction · width/opacity = GNNExplainer importance",
        transform=ax.transAxes,
        ha="center",
        fontsize=7,
        color="#5b6b80",
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_fidelity_by_typology(
    df,
    out_path: str | Path,
    title: str = "Explanation fidelity by typology",
) -> Path:
    """Bar chart of mean fidelity per typology, with per-bar sample counts.

    Counts are annotated on every bar on purpose: several typologies come
    back with single-digit samples, and a mean over n=8 should not be read
    the same way as a mean over n=38 (CONVENTIONS.md §9).

    Args:
        df: DataFrame with 'typology' and 'fidelity' columns, as written to
            outputs/explanations/fidelity_{run_id}.csv.
        out_path: Where to write the image.
        title: Figure title.

    Returns:
        The path written.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    agg = df.groupby("typology")["fidelity"].agg(["mean", "count"]).sort_values("mean")

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(agg.index, agg["mean"], color="#33506e")
    for bar, n in zip(bars, agg["count"]):
        ax.text(
            bar.get_width() + 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"n={n}",
            va="center",
            fontsize=8,
            color="#5b6b80",
        )
    ax.set_xlim(0, max(1.0, float(agg["mean"].max()) * 1.15))
    ax.set_xlabel("mean fidelity")
    ax.set_title(title, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
