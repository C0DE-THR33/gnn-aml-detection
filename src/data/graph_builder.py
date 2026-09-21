"""
Builds a PyTorch Geometric graph from the loaded transaction table.

Design (per SRS FR-1/FR-2, edge-level classification granularity):
    - Nodes  = accounts (deduplicated from_account/to_account values)
    - Edges  = transactions (directed, from_account -> to_account)
    - Edge label = is_laundering
    - Edge typology = typology string (joined from Patterns.txt where
      available; "unclassified"/None for ordinary licit transactions)

Node features are intentionally minimal at this stage (in/out-degree only) —
richer account-level features can be folded in once accounts.csv's real
schema is confirmed (see load_raw.py docstring). This docstring previously
claimed a bank one-hot that was never implemented; corrected rather than
implemented, since accounts.csv is the more direct source for that and is
still unused (see README "A few things worth knowing").

Edge features: log-amount plus a payment-format one-hot (see
load_raw.PAYMENT_FORMATS for why format and not currency).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from src.data.load_raw import PAYMENT_FORMATS
from src.data.typologies import Typology

# +1 for the "other" bucket — any payment_format value outside the canonical
# list lands here rather than being dropped or crashing graph construction.
_FORMAT_TO_IDX = {fmt: i for i, fmt in enumerate(PAYMENT_FORMATS)}
_OTHER_FORMAT_IDX = len(PAYMENT_FORMATS)
_NUM_FORMAT_BUCKETS = len(PAYMENT_FORMATS) + 1

# [log-amount | payment-format one-hot]. Anything that loads a saved graph
# should call check_graph_layout() so a graph built before a feature change
# fails loudly instead of training silently on the old, narrower layout.
EDGE_ATTR_WIDTH = 1 + _NUM_FORMAT_BUCKETS


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

    log_amount = torch.tensor(
        trans[["amount_paid"]].to_numpy(dtype="float32"),
    )
    # Log-scale amount to tame the heavy-tailed distribution typical of
    # transaction amounts.
    log_amount = torch.log1p(log_amount.clamp(min=0))
    edge_attr = torch.cat([log_amount, _payment_format_onehot(trans)], dim=1)

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


def check_graph_layout(data: Data) -> None:
    """Raise if a loaded graph was built with a different edge feature layout.

    build_model() sizes the edge head from data.edge_attr.shape[1], so a stale
    graph would not crash — it would train for hours without the payment-format
    feature and produce results that look normal. Fail here instead.

    Args:
        data: A graph loaded from data/processed/.

    Raises:
        ValueError: If edge_attr width differs from what the current code builds.
    """
    width = data.edge_attr.shape[1]
    if width != EDGE_ATTR_WIDTH:
        raise ValueError(
            f"Saved graph has edge_attr width {width}, but the current code builds "
            f"{EDGE_ATTR_WIDTH} (log-amount + payment-format one-hot). The graph is stale: "
            "rebuild it with scripts/build_graph.py, or re-upload the rebuilt "
            "hismall_graph.pt if running on Colab."
        )


def _payment_format_onehot(trans: pd.DataFrame) -> torch.Tensor:
    """One-hot encode Payment Format against the fixed canonical list.

    Width is always len(PAYMENT_FORMATS) + 1 regardless of which formats a
    given file contains, so the real data, the synthetic fixture and any
    subsample all produce the same edge_attr layout. Values outside the
    canonical list share the final "other" column instead of being dropped.

    Args:
        trans: Output of load_raw.load_transactions().

    Returns:
        [num_edges, len(PAYMENT_FORMATS) + 1] float32 tensor.
    """
    idx = trans["payment_format"].map(_FORMAT_TO_IDX).fillna(_OTHER_FORMAT_IDX).astype("int64")
    return F.one_hot(torch.from_numpy(idx.to_numpy(copy=True)), num_classes=_NUM_FORMAT_BUCKETS).float()


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
