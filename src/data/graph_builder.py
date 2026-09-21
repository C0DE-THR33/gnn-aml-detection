"""
Builds a PyTorch Geometric graph from the loaded transaction table.

Design (per SRS FR-1/FR-2, edge-level classification granularity):
    - Nodes  = accounts (deduplicated from_account/to_account values)
    - Edges  = transactions (directed, from_account -> to_account)
    - Edge label = is_laundering
    - Edge typology = typology string (joined from Patterns.txt where
      available; "unclassified"/None for ordinary licit transactions)

Node features are intentionally minimal at this stage (degree-based +
bank one-hot) — richer account-level features can be folded in once
accounts.csv's real schema is confirmed (see load_raw.py docstring).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from src.data.typologies import Typology


@dataclass
class GraphBuildResult:
    data: Data
    account_to_idx: dict
    idx_to_account: dict


def build_transaction_graph(
    trans: pd.DataFrame,
    patterns: pd.DataFrame | None = None,
) -> GraphBuildResult:
    """Construct a directed account-transaction graph.

    Args:
        trans: Output of load_raw.load_transactions().
        patterns: Optional output of load_raw.parse_patterns(), used to
            attach a typology label to each illicit edge on a best-effort
            basis (joined on from_account, to_account, timestamp, amount —
            Patterns.txt doesn't carry a shared txn_id with Trans.csv).

    Returns:
        GraphBuildResult with a PyG Data object (x, edge_index, edge_attr,
        y, edge_typology) and the account<->index mappings.
    """
    accounts = pd.unique(pd.concat([trans["from_account"], trans["to_account"]]))
    account_to_idx = {acct: i for i, acct in enumerate(accounts)}
    idx_to_account = {i: acct for acct, i in account_to_idx.items()}

    src = trans["from_account"].map(account_to_idx).to_numpy()
    dst = trans["to_account"].map(account_to_idx).to_numpy()
    edge_index = torch.from_numpy(np.stack([src, dst])).long()

    edge_attr = torch.tensor(
        trans[["amount_paid"]].to_numpy(dtype="float32"),
    )
    # Log-scale amount to tame the heavy-tailed distribution typical of
    # transaction amounts.
    edge_attr = torch.log1p(edge_attr.clamp(min=0))

    y = torch.tensor(trans["is_laundering"].to_numpy(dtype="int64"))

    edge_typology = _attach_typology(trans, patterns)

    # Minimal starter node features: in/out degree. Replace/extend once
    # accounts.csv's confirmed schema gives richer account-level features.
    num_nodes = len(accounts)
    out_deg = torch.zeros(num_nodes)
    in_deg = torch.zeros(num_nodes)
    out_deg.scatter_add_(0, edge_index[0], torch.ones(edge_index.shape[1]))
    in_deg.scatter_add_(0, edge_index[1], torch.ones(edge_index.shape[1]))
    x = torch.stack([in_deg, out_deg], dim=1)
    x = torch.log1p(x)  # degree also tends to be heavy-tailed

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.edge_typology = edge_typology  # list[str], parallel to edges

    return GraphBuildResult(data=data, account_to_idx=account_to_idx, idx_to_account=idx_to_account)


def _attach_typology(trans: pd.DataFrame, patterns: pd.DataFrame | None) -> list:
    """Best-effort join of Patterns.txt typology labels onto trans edges.

    Joins on (from_account, to_account, timestamp, amount_paid) since that's
    the only shared key between Trans.csv and Patterns.txt. Falls back to
    Typology.UNCLASSIFIED.value for illicit rows with no match, and to None
    for licit rows.
    """
    if patterns is None or patterns.empty:
        return [
            (Typology.UNCLASSIFIED.value if lbl == 1 else None)
            for lbl in trans["is_laundering"]
        ]

    # Vectorized merge rather than iterrows() — HI-Small has ~5M transaction
    # rows, and a Python-level loop over that is a real bottleneck, not a
    # theoretical one.
    join_keys = ["from_account", "to_account", "timestamp", "amount_paid"]
    lookup = patterns.drop_duplicates(subset=join_keys)[join_keys + ["typology"]]

    merged = trans[join_keys + ["is_laundering"]].merge(
        lookup, on=join_keys, how="left"
    )
    typology_col = merged["typology"]
    is_laundering = merged["is_laundering"]

    result = [
        (t if pd.notna(t) else Typology.UNCLASSIFIED.value) if lbl == 1 else None
        for t, lbl in zip(typology_col, is_laundering)
    ]
    return result
