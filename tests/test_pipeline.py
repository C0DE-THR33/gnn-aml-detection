"""
End-to-end pipeline tests using the synthetic fixture (not the real
HI-Small data — see README "First thing to do" before trusting any result
beyond "does the code run").
"""

from pathlib import Path

import pytest
import torch

from src.data.graph_builder import build_transaction_graph
from src.data.load_raw import load_accounts, load_transactions, parse_patterns
from src.data.make_splits import make_temporal_split
from src.explain.fidelity import score_typology_fidelity
from src.explain.run_explainer import build_explainer, explain_edge, select_explanation_sample
from src.models.baseline_sage import BaselineSAGE
from src.models.heterophily_gnn import HeterophilyGNN
from src.training.train import TrainConfig, train_model

FIXTURE_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def fixtures_exist():
    if not (FIXTURE_DIR / "Toy-Small_Trans.csv").exists():
        pytest.skip("Run tests/make_synthetic_fixture.py first.")


@pytest.fixture(scope="module")
def loaded_data(fixtures_exist):
    trans = load_transactions(FIXTURE_DIR / "Toy-Small_Trans.csv")
    patterns = parse_patterns(FIXTURE_DIR / "Toy-Small_Patterns.txt")
    accounts = load_accounts(FIXTURE_DIR / "Toy-Small_accounts.csv")
    return trans, patterns, accounts


def test_load_raw_preserves_leading_zeros(loaded_data):
    trans, _, accounts = loaded_data
    assert trans["from_bank"].str.len().min() == 5  # e.g. "00004", not "4"
    assert accounts["bank_id"].str.len().min() == 5


def test_graph_construction(loaded_data):
    trans, patterns, _ = loaded_data
    result = build_transaction_graph(trans, patterns)
    data = result.data
    assert data.edge_index.shape[1] == len(trans)
    assert int(data.y.sum()) == int(patterns["is_laundering"].sum()) or int(data.y.sum()) == len(patterns)
    # every illicit edge should have a non-None typology (fan_out.. or unclassified)
    illicit_typologies = [t for t, y in zip(data.edge_typology, data.y.tolist()) if y == 1]
    assert all(t is not None for t in illicit_typologies)


def test_temporal_split_is_disjoint_and_covers_all_edges(loaded_data):
    trans, _, _ = loaded_data
    splits = make_temporal_split(trans)
    all_idx = set(splits["train"]) | set(splits["val"]) | set(splits["test"])
    assert len(all_idx) == len(trans)
    assert set(splits["train"]) & set(splits["val"]) == set()
    assert set(splits["train"]) & set(splits["test"]) == set()


@pytest.mark.parametrize("model_cls", [BaselineSAGE, HeterophilyGNN])
def test_model_trains_without_error(loaded_data, model_cls):
    trans, patterns, _ = loaded_data
    data = build_transaction_graph(trans, patterns).data
    splits = make_temporal_split(trans)
    model = model_cls(in_dim=data.x.shape[1], edge_attr_dim=data.edge_attr.shape[1])
    result = train_model(model, data, splits, TrainConfig(epochs=5, log_every=5))
    assert len(result.history) >= 1
    assert torch.isfinite(torch.tensor(result.history[-1]["train_loss"]))


def test_explainer_runs_on_heterophily_gnn(loaded_data):
    """Regression test for the MessagePassing requirement — a hand-rolled
    scatter layer would fail here with 'Could not compute gradients for
    edges' (see heterophily_gnn.py's _MeanAggregator docstring)."""
    trans, patterns, _ = loaded_data
    data = build_transaction_graph(trans, patterns).data
    splits = make_temporal_split(trans)
    model = HeterophilyGNN(in_dim=data.x.shape[1], edge_attr_dim=data.edge_attr.shape[1])
    train_model(model, data, splits, TrainConfig(epochs=5, log_every=5))

    illicit_idx = [i for i in splits["test"] if data.y[i] == 1]
    if not illicit_idx:
        pytest.skip("No illicit edges landed in the test split for this fixture run.")

    explainer = build_explainer(model, explain_epochs=20)
    exp = explain_edge(explainer, data, illicit_idx[0])
    assert exp.edge_mask.shape[0] == data.edge_index.shape[1]
    assert exp.node_mask.shape[0] == data.num_nodes


def test_fidelity_scorer_handles_all_typologies(loaded_data):
    trans, patterns, _ = loaded_data
    data = build_transaction_graph(trans, patterns).data
    from src.data.typologies import EVALUATED_TYPOLOGIES

    for typ in EVALUATED_TYPOLOGIES:
        score = score_typology_fidelity(data.edge_index, torch.rand(data.edge_index.shape[1]), typ.value)
        if typ.value == "random":
            assert score is None  # random has no defined signature — see fidelity.py
        else:
            assert score is None or 0.0 <= score <= 1.0


def test_load_accounts_on_real_schema_has_unique_columns(tmp_path):
    """The real HI-Small_accounts.csv has 'Bank Name' AND 'Bank ID'. A substring
    match on 'bank' used to rename the NAME onto bank_id, producing two columns
    with the same name."""
    path = tmp_path / "accounts.csv"
    path.write_text(
        "Bank Name,Bank ID,Account Number,Entity ID,Entity Name\n"
        "Portugal Bank #4507,331579,80B779D80,80062E240,Sole Proprietorship #50438\n"
        "Canada Bank #27,210,809D86900,800C998A0,Corporation #33520\n",
        encoding="utf-8",
    )
    df = load_accounts(path)
    assert list(df.columns).count("bank_id") == 1, list(df.columns)
    assert df["bank_id"].tolist() == ["331579", "210"]  # the ID, not the name
    assert df["bank_name"].tolist() == ["Portugal Bank #4507", "Canada Bank #27"]
    assert df["account_id"].tolist() == ["80B779D80", "809D86900"]
