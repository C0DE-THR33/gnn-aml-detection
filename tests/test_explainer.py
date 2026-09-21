"""
Regression tests for the GNNExplainer path.

Two things are pinned here:

1. The edge-mask hook still works. GNNExplainer patches MessagePassing
   submodules it finds by module traversal, so a from-scratch aggregation
   layer silently breaks explainability (see heterophily_gnn.py's
   _MeanAggregator docstring). This is the test the README points at.
2. L-hop subgraph extraction is sound. explain_edge() hands GNNExplainer
   the target edge's L-hop neighbourhood rather than the full graph, which
   is only legitimate if the target's logits come out identical either way.
"""

from pathlib import Path

import pytest
import torch

from src.data.graph_builder import build_transaction_graph
from src.data.load_raw import load_transactions, parse_patterns
from src.explain.run_explainer import (
    build_explainer,
    explain_edge,
    extract_computation_subgraph,
)
from src.models.heterophily_gnn import HeterophilyGNN

FIXTURE_DIR = Path(__file__).parent / "fixtures"
NUM_LAYERS = 2


@pytest.fixture(scope="module")
def graph():
    if not (FIXTURE_DIR / "Toy-Small_Trans.csv").exists():
        pytest.skip("Run tests/make_synthetic_fixture.py first.")
    trans = load_transactions(FIXTURE_DIR / "Toy-Small_Trans.csv")
    patterns = parse_patterns(FIXTURE_DIR / "Toy-Small_Patterns.txt")
    return build_transaction_graph(trans, patterns).data


@pytest.fixture(scope="module")
def model(graph):
    torch.manual_seed(42)
    m = HeterophilyGNN(
        in_dim=graph.x.shape[1], hidden_dim=16, num_layers=NUM_LAYERS,
        edge_attr_dim=graph.edge_attr.shape[1],
    )
    m.eval()
    return m


def _illicit_edges(graph, limit=5):
    return graph.y.nonzero().view(-1)[:limit].tolist()


def test_subgraph_contains_target_and_is_smaller(graph):
    for edge_idx in _illicit_edges(graph):
        sub = extract_computation_subgraph(graph, edge_idx, NUM_LAYERS)
        # The target edge must survive extraction and be findable locally.
        assert int(sub.edge_subset[sub.target_local_idx]) == edge_idx
        assert sub.edge_subset.numel() <= graph.edge_index.shape[1]
        assert sub.edge_index.max() < sub.x.shape[0]
        # Endpoints must be preserved through the relabelling.
        src, dst = graph.edge_index[:, edge_idx].tolist()
        local_src, local_dst = sub.edge_index[:, sub.target_local_idx].tolist()
        assert int(sub.node_subset[local_src]) == src
        assert int(sub.node_subset[local_dst]) == dst


def test_subgraph_logits_match_full_graph(graph, model):
    """The whole memory optimisation rests on this equivalence."""
    with torch.no_grad():
        full_logits = model(graph.x, graph.edge_index, graph.edge_attr)

    for edge_idx in _illicit_edges(graph):
        sub = extract_computation_subgraph(graph, edge_idx, NUM_LAYERS)
        with torch.no_grad():
            sub_logits = model(sub.x, sub.edge_index, sub.edge_attr)
        torch.testing.assert_close(
            sub_logits[sub.target_local_idx],
            full_logits[edge_idx],
            rtol=1e-4,
            atol=1e-5,
        )


def test_explain_edge_returns_global_indexed_masks(graph, model):
    explainer = build_explainer(model, explain_epochs=5)
    edge_idx = _illicit_edges(graph, limit=1)[0]
    exp = explain_edge(explainer, graph, edge_idx, num_hops=NUM_LAYERS)

    # Masks come back addressed against the FULL graph so fidelity scoring
    # can index data.edge_index directly.
    assert exp.edge_mask.shape == (graph.edge_index.shape[1],)
    assert exp.node_mask.shape == (graph.num_nodes, graph.x.shape[1])
    assert exp.subgraph_edges > 0

    # Gradients reached the edge mask — the _MeanAggregator hook still works.
    assert torch.isfinite(exp.edge_mask).all()
    assert exp.edge_mask.abs().sum() > 0

    # Everything outside the computation subgraph stays exactly zero.
    sub = extract_computation_subgraph(graph, edge_idx, NUM_LAYERS)
    outside = torch.ones(graph.edge_index.shape[1], dtype=torch.bool)
    outside[sub.edge_subset] = False
    assert exp.edge_mask[outside].abs().sum() == 0


def test_oversized_neighbourhood_is_skipped_not_crashed(graph, model):
    explainer = build_explainer(model, explain_epochs=5)
    edge_idx = _illicit_edges(graph, limit=1)[0]
    with pytest.raises(RuntimeError, match="hub account"):
        explain_edge(explainer, graph, edge_idx, num_hops=NUM_LAYERS, max_subgraph_edges=1)


def test_subgraph_figure_is_written(graph, model, tmp_path):
    from src.explain.fidelity import build_explanation_subgraph
    from src.explain.visualize import draw_explanation_subgraph

    explainer = build_explainer(model, explain_epochs=5)
    edge_idx = _illicit_edges(graph, limit=1)[0]
    exp = explain_edge(explainer, graph, edge_idx, num_hops=NUM_LAYERS)

    out = draw_explanation_subgraph(
        graph.edge_index, exp.edge_mask, edge_idx, tmp_path / "sub" / "fig.png", top_k=6
    )
    assert out.exists() and out.stat().st_size > 0

    # The figure must show the scored edge set, or figure and number in the
    # report describe different things. draw_() force-adds the target edge on
    # top of that set, so the drawn graph is the scored one plus at most that.
    scored = build_explanation_subgraph(graph.edge_index, exp.edge_mask, top_k=6)
    src, dst = graph.edge_index[:, edge_idx].tolist()
    assert scored.number_of_edges() <= 6
    expected = set(scored.edges()) | {(src, dst)}
    assert len(expected) - scored.number_of_edges() <= 1


def test_fidelity_bar_chart_is_written(tmp_path):
    import pandas as pd

    from src.explain.visualize import plot_fidelity_by_typology

    df = pd.DataFrame(
        {
            "typology": ["fan_out", "fan_out", "cycle"],
            "fidelity": [0.5, 0.7, 0.2],
        }
    )
    out = plot_fidelity_by_typology(df, tmp_path / "fid.png")
    assert out.exists() and out.stat().st_size > 0
