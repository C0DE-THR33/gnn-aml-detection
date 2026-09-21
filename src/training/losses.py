"""
Loss functions for the illicit/licit edge classifier (SRS FR-6).

Illicit transactions are a small minority (roughly 0.1% in HI-Small), so
plain cross-entropy will collapse to predicting "licit" for everything.
Both options below are provided; pick one via the config's loss.type.
"""

import torch
import torch.nn.functional as F


def weighted_cross_entropy(logits: torch.Tensor, y: torch.Tensor, pos_weight: float) -> torch.Tensor:
    """Cross-entropy with the illicit (positive) class up-weighted.

    Args:
        logits: [N, 2] raw class logits.
        y: [N] int labels (0=licit, 1=illicit).
        pos_weight: Weight multiplier for the illicit class. A reasonable
            starting point is (num_licit / num_illicit) computed on the
            training split.
    """
    class_weights = torch.tensor([1.0, pos_weight], dtype=logits.dtype, device=logits.device)
    return F.cross_entropy(logits, y, weight=class_weights)


def focal_loss(logits: torch.Tensor, y: torch.Tensor, gamma: float = 2.0, alpha: float = 0.75) -> torch.Tensor:
    """Focal loss — down-weights easy (already well-classified) examples so
    training focuses on the rare, harder illicit cases.

    Args:
        logits: [N, 2] raw class logits.
        y: [N] int labels (0=licit, 1=illicit).
        gamma: Focusing parameter; higher = more down-weighting of easy examples.
        alpha: Weight on the illicit (positive) class.
    """
    log_probs = F.log_softmax(logits, dim=1)
    probs = log_probs.exp()
    pt = probs.gather(1, y.unsqueeze(1)).squeeze(1)
    log_pt = log_probs.gather(1, y.unsqueeze(1)).squeeze(1)

    alpha_t = torch.where(y == 1, alpha, 1 - alpha)
    loss = -alpha_t * (1 - pt) ** gamma * log_pt
    return loss.mean()


def compute_pos_weight(y_train: torch.Tensor) -> float:
    """num_licit / num_illicit on the training split, clamped to avoid /0."""
    num_illicit = int((y_train == 1).sum())
    num_licit = int((y_train == 0).sum())
    if num_illicit == 0:
        return 1.0
    return num_licit / num_illicit
