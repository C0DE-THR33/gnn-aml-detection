"""
Tests for the edge feature layout: [log-amount | payment-format one-hot].

Payment format is the strongest edge signal in HI-Small (0.75% illicit for ACH
vs 0.00% for Wire/Reinvestment). The layout has to be identical across the real
data, the synthetic fixture and any subsample, or a model trained on one cannot
be evaluated on another.
"""

import pandas as pd
import pytest
import torch

from src.data.graph_builder import build_transaction_graph
from src.data.load_raw import PAYMENT_FORMATS

EXPECTED_WIDTH = 1 + len(PAYMENT_FORMATS) + 1  # log-amount + formats + "other"


def _trans(formats):
    n = len(formats)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2022-09-01", periods=n, freq="h"),
            "from_account": [f"a{i}" for i in range(n)],
            "to_account": [f"b{i}" for i in range(n)],
            "amount_paid": [100.0] * n,
            "payment_format": formats,
            "is_laundering": [0] * n,
        }
    )


def test_edge_attr_width_is_fixed():
    data = build_transaction_graph(_trans(["ACH", "Wire"])).data
    assert data.edge_attr.shape == (2, EXPECTED_WIDTH)


def test_width_is_independent_of_which_formats_appear():
    """A subsample containing only one format must not shrink the layout."""
    one = build_transaction_graph(_trans(["ACH"] * 3)).data
    all_ = build_transaction_graph(_trans(list(PAYMENT_FORMATS))).data
    assert one.edge_attr.shape[1] == all_.edge_attr.shape[1] == EXPECTED_WIDTH


def test_each_format_lights_its_own_column():
    data = build_transaction_graph(_trans(list(PAYMENT_FORMATS))).data
    onehot = data.edge_attr[:, 1:]
    assert torch.equal(onehot.sum(dim=1), torch.ones(len(PAYMENT_FORMATS)))
    # Row i is format i, so the one-hot block is the identity over known formats,
    # with the trailing "other" column empty.
    assert torch.equal(onehot[:, :-1], torch.eye(len(PAYMENT_FORMATS)))
    assert onehot[:, -1].sum() == 0


def test_unknown_format_goes_to_other_not_dropped():
    data = build_transaction_graph(_trans(["ACH", "Carrier Pigeon"])).data
    onehot = data.edge_attr[:, 1:]
    assert onehot[1, -1] == 1.0
    assert onehot[1, :-1].sum() == 0
    assert onehot[0, -1] == 0.0


def test_log_amount_column_is_unchanged():
    data = build_transaction_graph(_trans(["ACH"])).data
    assert data.edge_attr[0, 0] == pytest.approx(torch.log1p(torch.tensor(100.0)).item())


def test_models_size_their_edge_head_from_the_graph():
    from scripts.train import build_model

    for kind in ("heterophily_gnn", "baseline_sage"):
        model = build_model(kind, in_dim=2, hidden_dim=8, num_layers=2, edge_attr_dim=EXPECTED_WIDTH)
        data = build_transaction_graph(_trans(["ACH", "Wire", "Cash"])).data
        logits = model(
            torch.zeros(data.num_nodes, 2), data.edge_index, data.edge_attr
        )
        assert logits.shape == (3, 2)


def test_layout_constant_matches_what_is_built():
    from src.data.graph_builder import EDGE_ATTR_WIDTH

    assert EDGE_ATTR_WIDTH == EXPECTED_WIDTH
    assert build_transaction_graph(_trans(["ACH"])).data.edge_attr.shape[1] == EDGE_ATTR_WIDTH


def test_stale_graph_is_rejected_with_a_clear_message():
    """A graph saved before the payment-format feature has edge_attr width 1."""
    from torch_geometric.data import Data

    from src.data.graph_builder import check_graph_layout

    stale = Data(edge_attr=torch.zeros(5, 1))
    with pytest.raises(ValueError, match="stale"):
        check_graph_layout(stale)

    check_graph_layout(build_transaction_graph(_trans(["ACH", "Wire"])).data)  # current: ok
