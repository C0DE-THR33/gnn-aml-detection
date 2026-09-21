"""
Heterophily-aware GNN (SRS FR-4).

Rationale: standard GNNs (GCN/GraphSAGE/GAT) smooth a node's representation
toward its neighbors' average, which helps when connected nodes tend to
share a label (homophily) but actively hurts when they don't. AML
transaction graphs are structurally heterophilic by design — illicit
accounts deliberately transact with legitimate-looking accounts to blend
in — so an illicit node's neighborhood average looks mostly "licit."

This layer follows the H2GCN idea of keeping self- and neighbor-signal
SEPARATE (concatenated, not summed/averaged together) so the self
representation isn't diluted. It goes one step further for this directed
transaction graph: in-neighbors and out-neighbors are aggregated
*separately*, since "who sends money to me" and "who I send money to" can
carry different signal for a laundering account.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


class _MeanAggregator(MessagePassing):
    """Plain mean-aggregation over whatever edge_index it's given.

    Implemented via MessagePassing.propagate() (rather than raw
    index_select + scatter) specifically so GNNExplainer can find and mask
    it — GNNExplainer's edge-mask hook works by patching MessagePassing
    submodules it discovers via module traversal; a hand-rolled scatter
    call is invisible to it and produces a "could not compute gradients
    for edges" error (learned the hard way while building this — see
    tests/test_explainer.py for the regression test).
    """

    def __init__(self):
        super().__init__(aggr="mean")

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.propagate(edge_index, x=x)

    def message(self, x_j: torch.Tensor) -> torch.Tensor:
        return x_j


class DirectionalHeterophilyLayer(nn.Module):
    """One layer: self / in-neighbor-mean / out-neighbor-mean, concatenated."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.self_lin = nn.Linear(in_dim, out_dim)
        self.in_lin = nn.Linear(in_dim, out_dim)
        self.out_lin = nn.Linear(in_dim, out_dim)
        self.in_agg = _MeanAggregator()
        self.out_agg = _MeanAggregator()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # in-neighbors of node i: accounts j with an edge j -> i.
        # propagate() with the edge_index as-is sends x[src] -> dst, i.e.
        # aggregates senders into each receiver — exactly "in" neighbors.
        in_agg = self.in_agg(x, edge_index)

        # out-neighbors of node i: accounts j with an edge i -> j. Flipping
        # src/dst sends x[dst] -> src, aggregating receivers into each
        # sender — exactly "out" neighbors. Same edge set, same length
        # edge_mask, just swapped columns, so GNNExplainer's per-edge mask
        # still lines up correctly with the original edges.
        out_agg = self.out_agg(x, edge_index.flip(0))

        h_self = self.self_lin(x)
        h_in = self.in_lin(in_agg)
        h_out = self.out_lin(out_agg)
        return F.relu(torch.cat([h_self, h_in, h_out], dim=1))  # [N, 3*out_dim]


class HeterophilyGNN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 32,
        edge_attr_dim: int = 1,
        num_layers: int = 2,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.projections = nn.ModuleList()

        cur_dim = in_dim
        for _ in range(num_layers):
            self.layers.append(DirectionalHeterophilyLayer(cur_dim, hidden_dim))
            concat_dim = 3 * hidden_dim
            # Project back down before the next layer so width doesn't
            # explode across layers (3x per layer, uncontrolled).
            self.projections.append(nn.Linear(concat_dim, hidden_dim))
            cur_dim = hidden_dim

        final_node_dim = hidden_dim
        self.edge_head = nn.Sequential(
            nn.Linear(2 * final_node_dim + edge_attr_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for layer, proj in zip(self.layers, self.projections):
            h = layer(h, edge_index)
            h = proj(h)
        return h

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """Returns raw logits of shape [num_edges, 2] (licit, illicit)."""
        h = self.encode(x, edge_index)
        src, dst = edge_index
        edge_repr = torch.cat([h[src], h[dst], edge_attr], dim=1)
        return self.edge_head(edge_repr)
