"""
Homophily-assuming baseline (SRS FR-5) for comparison against the
heterophily-aware model. Standard GraphSAGE encoder + MLP edge head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class BaselineSAGE(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, edge_attr_dim: int = 1, num_layers: int = 2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))

        self.edge_head = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_attr_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if i < len(self.convs) - 1:
                h = F.relu(h)
        return h

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """Returns raw logits of shape [num_edges, 2] (licit, illicit)."""
        h = self.encode(x, edge_index)
        src, dst = edge_index
        edge_repr = torch.cat([h[src], h[dst], edge_attr], dim=1)
        return self.edge_head(edge_repr)
