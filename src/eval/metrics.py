"""
Classification metrics, overall and broken down per typology (SRS FR-7).
"""

from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

from src.data.typologies import EVALUATED_TYPOLOGIES


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict:
    """Standard metrics for the illicit (positive) class.

    Args:
        y_true: [N] ground-truth binary labels.
        y_pred: [N] predicted binary labels (thresholded).
        y_score: [N] predicted probability of the illicit class.

    Returns:
        Dict with precision, recall, f1, auprc, accuracy, trivial_accuracy
        and n. Returns NaNs if y_true has no positive examples (can't compute
        AUPRC/precision meaningfully).

    Note on accuracy: reported only because it gets asked for, and always
    beside `trivial_accuracy` — the score of a constant "always licit"
    predictor. At HI-Small's 0.177% test prevalence that constant predictor
    scores 99.82%, so accuracy rewards the useless answer and any accuracy
    quoted without its trivial baseline beside it is misleading. The SRS
    success criteria (§174) are stated in recall/F1 on the illicit class for
    exactly this reason; AUPRC against the base rate is the number to lead on.
    """
    n_neg = int((y_true == 0).sum())
    trivial_accuracy = n_neg / len(y_true) if len(y_true) else float("nan")
    if y_true.sum() == 0:
        return {
            "precision": float("nan"), "recall": float("nan"), "f1": float("nan"),
            "auprc": float("nan"), "accuracy": float((y_pred == y_true).mean()),
            "trivial_accuracy": trivial_accuracy, "n": len(y_true),
        }
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auprc": average_precision_score(y_true, y_score),
        "accuracy": float((y_pred == y_true).mean()),
        "trivial_accuracy": trivial_accuracy,
        "n": len(y_true),
    }


def per_typology_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    edge_typology: list,
) -> pd.DataFrame:
    """Metrics computed separately for each of the 8 evaluated typologies.

    Licit edges (typology=None) and UNCLASSIFIED illicit edges are excluded
    from this breakdown — this table answers "how well do we detect each
    known typology," not overall performance (see classification_metrics
    for that).

    Args:
        y_true, y_pred, y_score: as in classification_metrics, over ALL edges.
        edge_typology: list of typology strings (or None), parallel to
            y_true/y_pred/y_score.

    Returns:
        DataFrame indexed by typology, one row per typology in
        EVALUATED_TYPOLOGIES, with precision/recall/f1/auprc/n columns. The
        "n" column is the number of illicit edges *of that typology* in this
        split (not the size of the scoring set, which also includes the licit
        edges used as negatives). Typologies with zero examples in this split
        get NaN metrics rather than being silently dropped, so gaps are
        visible in the report.
    """
    edge_typology = np.array(edge_typology, dtype=object)
    rows = []
    for typ in EVALUATED_TYPOLOGIES:
        n_typ = int((edge_typology == typ.value).sum())
        if n_typ == 0:
            rows.append({"typology": typ.value, "precision": np.nan, "recall": np.nan, "f1": np.nan, "auprc": np.nan, "n": 0})
            continue
        # Score this typology's illicit edges against all licit edges as
        # negatives. Masking to the typology's edges alone would make every
        # label positive, which pins precision at 1.0 and AUPRC at 1.0 by
        # construction and leaves recall as the only real number.
        mask = build_typology_eval_mask(edge_typology, typ.value)
        yt, yp, ys = y_true[mask], y_pred[mask], y_score[mask]
        m = classification_metrics(yt, yp, ys)
        m["n"] = n_typ
        m["typology"] = typ.value
        rows.append(m)
    return pd.DataFrame(rows).set_index("typology")[["precision", "recall", "f1", "auprc", "n"]]


def build_typology_eval_mask(edge_typology: list, typology_value: str) -> np.ndarray:
    """Boolean mask selecting this typology's illicit edges + all licit edges.

    Per-typology recall/precision needs a comparison set: "of edges that are
    either licit or this specific typology, how well did we separate them?"
    Mixing in every other typology's illicit edges as extra negatives would
    unfairly penalize/reward the typology being scored.
    """
    edge_typology = np.array(edge_typology, dtype=object)
    return (edge_typology == typology_value) | (edge_typology == None)  # noqa: E711
