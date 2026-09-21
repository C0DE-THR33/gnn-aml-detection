"""
Per-typology explanation fidelity scoring (SRS FR-10/FR-11).

Each scorer below is a HEURISTIC proxy for "does the explanation subgraph
structurally resemble this typology's known signature" — not a formal
graph-isomorphism match. That's a deliberate, documented simplification
(see SRS §6.2 Non-Goals): the goal is a reproducible, defensible first-pass
signal ("did GNNExplainer recover something fan-out-shaped"), not a perfect
pattern-matcher. Treat scores as comparable to each other across runs of
the SAME scorer, not as an absolute ground truth.

Typology.RANDOM has no structural signature by construction — a random
walk pattern is deliberately unstructured (SRS Appendix 6.1) — so it is
intentionally NOT scored here. Reporting "no meaningful fidelity score for
Random" is the honest result, not a gap to paper over (this mirrors the
risk flagged in SRS §6.3).
"""

from typing import Optional

import networkx as nx
import torch

from src.data.typologies import Typology


def build_explanation_subgraph(
    edge_index: torch.Tensor,
    edge_mask: torch.Tensor,
    top_k: int = 6,
) -> nx.DiGraph:
    """Builds a directed networkx graph from the top-k most important edges.

    Selection is over distinct (src, dst) ACCOUNT PAIRS, not raw transaction
    rows. HI-Small averages 5.0 transactions per account pair, and a DiGraph
    merges parallel edges — so taking the top-k rows naively produced graphs
    with as few as 2 edges instead of k. Since every scorer below divides by
    g.number_of_edges(), that shrinking denominator was quantising the scores
    and making them jump between runs. Ranking pairs by their best constituent
    edge keeps the subgraph at k edges whenever k distinct pairs exist.

    Args:
        edge_index: [2, num_edges] full graph edge index.
        edge_mask: [num_edges] importance scores from GNNExplainer.
        top_k: How many top-importance account pairs to include. SRS doesn't
            fix this number — 6 is a reasonable default matching the smallest
            typical pattern-group size seen in Patterns.txt, but tune it
            per-typology if larger patterns (e.g. stack) get truncated.

    Returns:
        A networkx.DiGraph with min(top_k, number of distinct pairs) edges.
    """
    nonzero = edge_mask.nonzero().view(-1)
    if nonzero.numel() == 0:
        return nx.DiGraph()

    # Collapse to one score per account pair (max over its transactions),
    # then take the top-k pairs.
    best: dict[tuple[int, int], float] = {}
    for idx in nonzero.tolist():
        pair = (int(edge_index[0, idx]), int(edge_index[1, idx]))
        score = float(edge_mask[idx])
        if score > best.get(pair, float("-inf")):
            best[pair] = score

    ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    g = nx.DiGraph()
    for (src, dst), _ in ranked:
        g.add_edge(src, dst)
    return g


def _fan_out_score(g: nx.DiGraph) -> float:
    """High if one node has most of the subgraph's out-edges, few in-edges."""
    if g.number_of_edges() == 0:
        return 0.0
    out_degs = dict(g.out_degree())
    best_node = max(out_degs, key=out_degs.get)
    return out_degs[best_node] / g.number_of_edges()


def _fan_in_score(g: nx.DiGraph) -> float:
    """High if one node has most of the subgraph's in-edges, few out-edges."""
    if g.number_of_edges() == 0:
        return 0.0
    in_degs = dict(g.in_degree())
    best_node = max(in_degs, key=in_degs.get)
    return in_degs[best_node] / g.number_of_edges()


def _cycle_score(g: nx.DiGraph) -> float:
    """1.0 if the subgraph contains a directed cycle of length >= 2, else 0.0.

    Self-loops are stripped first. 11.64% of HI-Small transactions are
    account-to-itself reinvestments, and networkx counts a self-loop as a
    cycle — so without this the scorer returned 1.0 for any explanation that
    happened to include one, whether or not a real cycle was present.
    """
    probe = g.copy()
    probe.remove_edges_from(nx.selfloop_edges(probe))
    try:
        nx.find_cycle(probe, orientation="original")
        return 1.0
    except nx.NetworkXNoCycle:
        return 0.0


def _gather_scatter_score(g: nx.DiGraph) -> float:
    """High if one node has both high in-degree AND high out-degree (a hub
    that both gathers and scatters funds).

    Clamped to [0, 1]. The raw ratio can exceed 1 on small subgraphs (a hub
    with 2 in and 2 out across 3 edges gives 2 / 1.5 = 1.33), which made this
    scorer incomparable with the others — all of which are bounded.
    """
    if g.number_of_edges() == 0:
        return 0.0
    in_degs = dict(g.in_degree())
    out_degs = dict(g.out_degree())
    best = max(min(in_degs.get(n, 0), out_degs.get(n, 0)) for n in g.nodes())
    return min(best / max(g.number_of_edges() / 2, 1), 1.0)


def _scatter_gather_score(g: nx.DiGraph) -> float:
    """High if there's a source and a sink connected via multiple distinct
    intermediate nodes (source -> intermediate -> sink, several times)."""
    if g.number_of_edges() == 0:
        return 0.0
    best = 0
    for src in g.nodes():
        for dst in g.nodes():
            if src == dst:
                continue
            intermediates = set(g.successors(src)) & set(g.predecessors(dst))
            best = max(best, len(intermediates))
    return best / max(g.number_of_nodes(), 1)


def _stack_score(g: nx.DiGraph) -> float:
    """High if the subgraph is close to a single long chain (few branches)."""
    if g.number_of_nodes() < 2:
        return 0.0
    try:
        longest = nx.dag_longest_path_length(g)
    except (nx.NetworkXError, nx.NetworkXUnfeasible):
        return 0.0
    return longest / max(g.number_of_edges(), 1)


def _bipartite_score(g: nx.DiGraph) -> float:
    """High if the subgraph's edges run cleanly between two disjoint groups
    with little/no internal-group flow (approximated via nx bipartite check
    on the undirected version)."""
    if g.number_of_edges() == 0:
        return 0.0
    undirected = g.to_undirected()
    return 1.0 if nx.is_bipartite(undirected) else 0.0


_SCORERS = {
    Typology.FAN_OUT: _fan_out_score,
    Typology.FAN_IN: _fan_in_score,
    Typology.CYCLE: _cycle_score,
    Typology.GATHER_SCATTER: _gather_scatter_score,
    Typology.SCATTER_GATHER: _scatter_gather_score,
    Typology.STACK: _stack_score,
    Typology.BIPARTITE: _bipartite_score,
    # Typology.RANDOM intentionally omitted — see module docstring.
}


def score_typology_fidelity(
    edge_index: torch.Tensor,
    edge_mask: torch.Tensor,
    typology: str,
    top_k: int = 6,
) -> Optional[float]:
    """Score how well an explanation subgraph matches its typology's signature.

    Args:
        edge_index: [2, num_edges] full graph edge index.
        edge_mask: [num_edges] importance scores for the explained edge.
        typology: Typology value string (e.g. "fan_out") for the edge being
            explained.
        top_k: Passed to build_explanation_subgraph().

    Returns:
        A heuristic fidelity score in [0, 1], or None if this typology has
        no defined scorer (Typology.RANDOM, Typology.UNCLASSIFIED) — report
        this as "not scored," not as a zero.
    """
    try:
        typ_enum = Typology(typology)
    except ValueError:
        return None
    scorer = _SCORERS.get(typ_enum)
    if scorer is None:
        return None
    g = build_explanation_subgraph(edge_index, edge_mask, top_k=top_k)
    return scorer(g)
