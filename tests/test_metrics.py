"""
Tests for the classification metrics, with emphasis on the imbalance traps.

HI-Small's test split is 0.177% positive. Metrics that look reasonable on a
balanced problem behave very differently here, and the report depends on
quoting the ones that mean something (CONVENTIONS.md §9).
"""

import numpy as np

from src.eval.metrics import classification_metrics


def _imbalanced(n=10_000, n_pos=18):
    """Roughly HI-Small's 0.177% positive rate."""
    y = np.zeros(n, dtype=int)
    y[:n_pos] = 1
    return y


def test_accuracy_is_reported_with_its_trivial_baseline():
    y = _imbalanced()
    pred = np.zeros_like(y)  # the useless constant predictor
    m = classification_metrics(y, pred, np.zeros(len(y), dtype=float))
    assert "accuracy" in m and "trivial_accuracy" in m
    # A constant "always licit" predictor IS the trivial baseline, so they match.
    assert np.isclose(m["accuracy"], m["trivial_accuracy"])
    assert m["accuracy"] > 0.99  # ...while catching nothing at all
    assert m["recall"] == 0.0


def test_high_recall_model_scores_worse_accuracy_than_doing_nothing():
    """The pattern seen in every real run: pos_weight pushes the model to flag
    ~20% of traffic, which is useful for ranking and terrible for accuracy."""
    y = _imbalanced()
    rng = np.random.default_rng(0)
    pred = np.zeros_like(y)
    pred[:16] = 1  # catches 16 of 18 positives
    flagged = rng.choice(np.arange(18, len(y)), size=2000, replace=False)
    pred[flagged] = 1  # ...at the cost of 2000 false positives

    m = classification_metrics(y, pred, pred.astype(float))
    assert m["recall"] > 0.8
    assert m["accuracy"] < m["trivial_accuracy"], (
        "on this prevalence a useful-recall model should score WORSE accuracy "
        "than predicting the majority class, which is the whole reason "
        "accuracy is not a headline metric here"
    )


def test_trivial_accuracy_matches_negative_fraction():
    y = _imbalanced(n=1000, n_pos=10)
    m = classification_metrics(y, np.zeros_like(y), np.zeros(len(y), dtype=float))
    assert np.isclose(m["trivial_accuracy"], 0.99)


def test_all_negative_labels_returns_nan_ranking_metrics_but_real_accuracy():
    y = np.zeros(100, dtype=int)
    pred = np.zeros(100, dtype=int)
    m = classification_metrics(y, pred, np.zeros(100, dtype=float))
    assert np.isnan(m["precision"]) and np.isnan(m["auprc"])
    assert m["accuracy"] == 1.0
    assert m["trivial_accuracy"] == 1.0


def test_auprc_beats_base_rate_when_ranking_is_informative():
    y = _imbalanced()
    score = np.zeros(len(y), dtype=float)
    score[:18] = 0.9  # positives ranked top
    score[18:] = np.linspace(0.0, 0.5, len(y) - 18)
    m = classification_metrics(y, (score >= 0.5).astype(int), score)
    base_rate = y.mean()
    assert m["auprc"] > 10 * base_rate


def test_train_eval_subsample_preserves_prevalence_and_is_deterministic():
    """select_train_eval_idx must sample uniformly, not class-balanced: the
    train/val AUPRC gap is only interpretable if both splits are scored at the
    same prevalence."""
    import torch

    from src.training.train import MAX_TRAIN_EVAL_EDGES, select_train_eval_idx

    n = MAX_TRAIN_EVAL_EDGES + 500_000
    y = torch.zeros(n, dtype=torch.long)
    y[torch.arange(0, n, 1000)] = 1  # 0.1% positives
    train_idx = torch.arange(n)

    a = select_train_eval_idx(train_idx, y, seed=42)
    b = select_train_eval_idx(train_idx, y, seed=42)
    c = select_train_eval_idx(train_idx, y, seed=7)

    assert a.numel() == MAX_TRAIN_EVAL_EDGES
    assert torch.equal(a, b), "same seed must give the same evaluation subset"
    assert not torch.equal(a, c), "different seeds should differ"

    full_rate = float((y == 1).float().mean())
    sample_rate = float((y[a] == 1).float().mean())
    assert abs(sample_rate - full_rate) < 0.2 * full_rate, (
        f"uniform sampling should preserve prevalence: {sample_rate:.5f} "
        f"vs {full_rate:.5f}"
    )


def test_train_eval_returns_everything_when_split_is_small():
    import torch

    from src.training.train import select_train_eval_idx

    y = torch.tensor([0, 1, 0, 1])
    train_idx = torch.arange(4)
    assert torch.equal(select_train_eval_idx(train_idx, y, seed=42), train_idx)
