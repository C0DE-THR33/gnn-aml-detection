"""
GNNExplainer wrapper for edge-level predictions (SRS FR-8/FR-9).

IMPORTANT if you write your own model layers: GNNExplainer's edge-mask hook
only works on layers that subclass torch_geometric.nn.MessagePassing and go
through propagate()/message(). A hand-rolled scatter/index_select layer is
invisible to it and raises "Could not compute gradients for edges." See
src/models/heterophily_gnn.py's _MeanAggregator for the pattern that works.
"""

from dataclasses import dataclass
from typing import Optional

import torch
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.utils import k_hop_subgraph


@dataclass
class EdgeExplanation:
    edge_idx: int
    typology: str | None
    edge_mask: torch.Tensor  # [num_edges], importance per edge (global indexing)
    node_mask: torch.Tensor  # [num_nodes, num_node_features] (global indexing)
    subgraph_edges: int  # how many edges the explainer actually optimized over


@dataclass
class _ComputationSubgraph:
    """The L-hop neighbourhood a single edge prediction actually depends on."""

    x: torch.Tensor
    edge_index: torch.Tensor  # relabelled to local node ids
    edge_attr: torch.Tensor
    node_subset: torch.Tensor  # local -> global node id
    edge_subset: torch.Tensor  # local -> global edge id
    target_local_idx: int


def _bidirectional_edge_index(data: Data) -> torch.Tensor:
    """edge_index concatenated with its flip, cached on the Data object.

    DirectionalHeterophilyLayer aggregates over both edge_index and
    edge_index.flip(0), so a node's receptive field expands along edges in
    BOTH directions. Hop expansion has to match that, or the extracted
    subgraph silently omits out-neighbours and the explanation is wrong.

    Args:
        data: Full PyG Data object.

    Returns:
        [2, 2 * num_edges] tensor, cached on `data` after the first call.
    """
    cached = getattr(data, "_bidir_edge_index", None)
    if cached is None:
        cached = torch.cat([data.edge_index, data.edge_index.flip(0)], dim=1)
        data._bidir_edge_index = cached
    return cached


def extract_computation_subgraph(
    data: Data,
    edge_idx: int,
    num_hops: int,
) -> "_ComputationSubgraph":
    """Extract the L-hop neighbourhood around one edge's two endpoints.

    For an L-layer message-passing model the logits for edge (u, v) depend
    only on the L-hop neighbourhood of u and v, so explaining against the
    full graph is both wasteful and — at HI-Small's 5.08M edges — impossible
    to fit on a 15 GB GPU. The induced subgraph over the L-hop node set is a
    superset of the true computation graph, so the target edge's prediction
    is numerically identical to the full-graph one.

    Args:
        data: Full PyG Data object.
        edge_idx: Global index into data.edge_index of the edge to explain.
        num_hops: Number of message-passing layers in the model.

    Returns:
        A _ComputationSubgraph with local tensors plus the local->global
        index maps needed to project the explanation back onto the full graph.
    """
    device = data.edge_index.device
    seed = data.edge_index[:, edge_idx].unique()

    node_subset, _, _, _ = k_hop_subgraph(
        seed, num_hops, _bidirectional_edge_index(data), num_nodes=data.num_nodes
    )

    keep = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    keep[node_subset] = True
    edge_subset = (keep[data.edge_index[0]] & keep[data.edge_index[1]]).nonzero().view(-1)

    relabel = torch.full((data.num_nodes,), -1, dtype=torch.long, device=device)
    relabel[node_subset] = torch.arange(node_subset.numel(), device=device)

    target_local_idx = int((edge_subset == edge_idx).nonzero().item())
    return _ComputationSubgraph(
        x=data.x[node_subset],
        edge_index=relabel[data.edge_index[:, edge_subset]],
        edge_attr=data.edge_attr[edge_subset],
        node_subset=node_subset,
        edge_subset=edge_subset,
        target_local_idx=target_local_idx,
    )


def build_explainer(model: torch.nn.Module, explain_epochs: int = 200) -> Explainer:
    """Constructs a PyG Explainer configured for edge-level classification.

    Args:
        model: A trained BaselineSAGE or HeterophilyGNN (or anything with a
            matching forward(x, edge_index, edge_attr) -> [E, 2] signature).
        explain_epochs: GNNExplainer's internal optimization steps per
            instance. 100-300 is typical; more = slower but often cleaner
            masks. This is the main knob if explanations look noisy.
    """
    return Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=explain_epochs),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(
            mode="multiclass_classification",
            task_level="edge",
            return_type="raw",
        ),
    )


def explain_edge(
    explainer: Explainer,
    data: Data,
    edge_idx: int,
    num_hops: int = 2,
    max_subgraph_edges: Optional[int] = 400_000,
) -> EdgeExplanation:
    """Run GNNExplainer on a single transaction (edge).

    The explainer is handed the edge's L-hop computation subgraph, not the
    full graph. Two reasons, both load-bearing at HI-Small's scale:

    1. Memory. GNNExplainer learns one mask entry per edge and backprops
       through the model once per epoch. Over 5.08M edges that is ~8 GB of
       activations per instance, which OOMs a 15 GB T4.
    2. Time. 200 explainer epochs x 135 sampled edges is 27,000 full-graph
       forward/backward passes; on the L-hop subgraph each pass covers a few
       thousand edges instead.

    The target edge's logits are unchanged by this — see
    extract_computation_subgraph().

    Args:
        explainer: Output of build_explainer().
        data: The full PyG Data object.
        edge_idx: Index into data.edge_index for the transaction to explain.
            Should be a true-positive illicit prediction (SRS FR-8).
        num_hops: Message-passing depth of the model being explained. Must
            match the config's model.num_layers, or the subgraph is too small.
        max_subgraph_edges: Skip instances whose neighbourhood exceeds this,
            rather than OOMing on a hub account. None disables the cap.

    Returns:
        EdgeExplanation whose masks are projected back onto FULL-graph
        indexing (zero outside the subgraph), so downstream fidelity scoring
        keeps addressing data.edge_index unchanged.

    Raises:
        RuntimeError: If the computation subgraph exceeds max_subgraph_edges.
    """
    sub = extract_computation_subgraph(data, edge_idx, num_hops)
    n_sub = int(sub.edge_subset.numel())
    if max_subgraph_edges is not None and n_sub > max_subgraph_edges:
        raise RuntimeError(
            f"edge {edge_idx}: {num_hops}-hop neighbourhood has {n_sub:,} edges "
            f"(cap {max_subgraph_edges:,}) — hub account, skipping."
        )

    explanation = explainer(
        x=sub.x,
        edge_index=sub.edge_index,
        edge_attr=sub.edge_attr,
        index=sub.target_local_idx,
    )

    # Project the local masks back onto full-graph indexing so callers and
    # fidelity scoring keep working against data.edge_index untouched.
    edge_mask = torch.zeros(data.edge_index.shape[1])
    edge_mask[sub.edge_subset.cpu()] = explanation.edge_mask.detach().cpu()
    node_mask = torch.zeros(data.num_nodes, data.x.shape[1])
    node_mask[sub.node_subset.cpu()] = explanation.node_mask.detach().cpu()

    typology = data.edge_typology[edge_idx] if hasattr(data, "edge_typology") else None
    return EdgeExplanation(
        edge_idx=edge_idx,
        typology=typology,
        edge_mask=edge_mask,
        node_mask=node_mask,
        subgraph_edges=n_sub,
    )


def select_explanation_sample(
    data: Data,
    y_pred: torch.Tensor,
    candidate_idx,
    per_typology_cap: int = 15,
) -> list[int]:
    """Pick true-positive illicit edges to explain, capped per typology.

    Implements SRS FR-8: "select a sample of correctly-classified true
    positive illicit transactions." Capping per typology keeps the
    explanation batch balanced instead of dominated by whichever typology
    happens to have the most transactions (e.g. gather-scatter).

    Args:
        data: Full PyG Data (needs data.y and data.edge_typology).
        y_pred: Predicted labels aligned with data.edge_index, e.g. from
            thresholding the trained model's output on `candidate_idx`.
        candidate_idx: Edge indices to consider (typically the test split).
        per_typology_cap: Max number of edges to sample per typology.

    Returns:
        List of edge indices to run explain_edge() on.
    """
    import numpy as np

    candidate_idx = np.asarray(candidate_idx)
    y_true = data.y[candidate_idx].cpu().numpy()
    y_pred = np.asarray(y_pred)
    true_positive_mask = (y_true == 1) & (y_pred == 1)
    tp_idx = candidate_idx[true_positive_mask]

    edge_typology = [data.edge_typology[i] for i in tp_idx]
    selected = []
    seen_per_typ = {}
    for idx, typ in zip(tp_idx, edge_typology):
        count = seen_per_typ.get(typ, 0)
        if count < per_typology_cap:
            selected.append(int(idx))
            seen_per_typ[typ] = count + 1
    return selected
