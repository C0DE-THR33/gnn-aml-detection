"""
Regression tests for the fidelity scorer defects found when two Colab runs
produced non-replicating fidelity tables (means moving by up to 0.35 between
runs of the same pipeline).

Each test below pins one of those defects closed. See CONVENTIONS.md §9 —
these numbers end up as claims in the report, so the thing being measured has
to be structure, not a data artifact.
"""

import networkx as nx
import torch

from src.explain.fidelity import (
    _cycle_score,
    _gather_scatter_score,
    build_explanation_subgraph,
)


def test_topk_selects_distinct_account_pairs():
    """HI-Small averages 5.0 transactions per account pair, and a DiGraph
    merges them. Ranking raw rows therefore yielded subgraphs far smaller
    than top_k, shrinking every scorer's denominator."""
    # Six transactions, but only two distinct pairs: (0,1) x3 and (2,3) x3.
    edge_index = torch.tensor([[0, 0, 0, 2, 2, 2, 4, 5], [1, 1, 1, 3, 3, 3, 6, 7]])
    edge_mask = torch.tensor([0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55])

    g = build_explanation_subgraph(edge_index, edge_mask, top_k=4)
    # Naive row-ranking would take the six highest rows and collapse to 2
    # edges. Pair-ranking must reach the requested 4.
    assert g.number_of_edges() == 4
    assert set(g.edges()) == {(0, 1), (2, 3), (4, 6), (5, 7)}


def test_topk_caps_at_available_pairs():
    edge_index = torch.tensor([[0, 0, 0], [1, 1, 1]])
    edge_mask = torch.tensor([0.9, 0.5, 0.1])
    g = build_explanation_subgraph(edge_index, edge_mask, top_k=6)
    assert g.number_of_edges() == 1


def test_topk_ignores_edges_outside_the_explanation():
    """explain_edge() zeroes everything outside the computation subgraph;
    those zeros must not be selectable."""
    edge_index = torch.tensor([[0, 2, 4], [1, 3, 5]])
    edge_mask = torch.tensor([0.7, 0.0, 0.0])
    g = build_explanation_subgraph(edge_index, edge_mask, top_k=3)
    assert set(g.edges()) == {(0, 1)}


def test_self_loop_is_not_a_cycle():
    """11.64% of HI-Small transactions are account-to-itself reinvestments,
    and nx.find_cycle counts a self-loop as a cycle."""
    g = nx.DiGraph()
    g.add_edge(7, 7)  # reinvestment
    g.add_edge(1, 2)
    g.add_edge(3, 4)
    assert _cycle_score(g) == 0.0


def test_real_cycle_still_scores():
    g = nx.DiGraph()
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    g.add_edge(3, 1)
    assert _cycle_score(g) == 1.0

    two_cycle = nx.DiGraph()
    two_cycle.add_edge(1, 2)
    two_cycle.add_edge(2, 1)
    assert _cycle_score(two_cycle) == 1.0


def test_self_loop_does_not_mask_a_real_cycle():
    g = nx.DiGraph()
    g.add_edge(7, 7)
    g.add_edge(1, 2)
    g.add_edge(2, 1)
    assert _cycle_score(g) == 1.0


def test_gather_scatter_is_bounded_by_one():
    """The exact shape that produced fidelity 1.333 in the second Colab run:
    a self-loop counts toward both in- and out-degree, so a 3-edge subgraph
    scored 2 / 1.5."""
    g = nx.DiGraph()
    g.add_edge(0, 9)  # a -> h
    g.add_edge(9, 1)  # h -> b
    g.add_edge(9, 9)  # h -> h, counts as both an in- and an out-edge
    assert min(dict(g.in_degree())[9], dict(g.out_degree())[9]) == 2
    assert g.number_of_edges() == 3
    assert _gather_scatter_score(g) == 1.0  # was 1.333


def test_gather_scatter_still_discriminates():
    hub = nx.DiGraph()
    for src in (0, 1):
        hub.add_edge(src, 9)
    for dst in (2, 3):
        hub.add_edge(9, dst)
    chain = nx.DiGraph()
    chain.add_edge(0, 1)
    chain.add_edge(1, 2)
    chain.add_edge(2, 3)
    assert _gather_scatter_score(hub) > _gather_scatter_score(chain)


def test_empty_subgraph_scores_zero_not_crash():
    empty = nx.DiGraph()
    assert _cycle_score(empty) == 0.0
    assert _gather_scatter_score(empty) == 0.0
    assert build_explanation_subgraph(
        torch.tensor([[0], [1]]), torch.tensor([0.0]), top_k=6
    ).number_of_edges() == 0
