"""
Training loop for the edge classifier (baseline or heterophily-aware).

Message passing runs over the FULL graph (all edges); only the loss and
metrics are restricted to the current split's edge indices. This is
standard practice for this kind of static-graph edge classification setup
and keeps the graph connected for propagation — see graph_builder.py /
make_splits.py docstrings for the temporal-leakage caveat this implies.
"""

from dataclasses import dataclass, field

import numpy as np
import torch
from torch_geometric.data import Data

from src.eval.metrics import classification_metrics
from src.training.losses import compute_pos_weight, focal_loss, weighted_cross_entropy


# Cap on how many training edges the per-epoch train metrics are computed over.
# The full train split is ~3.05M edges; average_precision_score over all of them
# at every log point is pure overhead for a number that barely moves. 1M keeps
# roughly 750 positives, which is ample for a stable AUPRC estimate.
MAX_TRAIN_EVAL_EDGES = 1_000_000


def select_train_eval_idx(
    train_idx: torch.Tensor,
    y: torch.Tensor,
    seed: int,
) -> torch.Tensor:
    """Pick which training edges to compute train-split metrics on.

    Called ONCE before the training loop; the returned indices are reused at
    every log point so the train curve is comparable epoch to epoch.

    Args:
        train_idx: [n_train] indices of the training split (~3.05M edges on
            full HI-Small, of which ~2,298 are illicit).
        y: Full-graph label tensor, indexable by train_idx.
        seed: Config seed, for any sampling this does.

    Returns:
        1D LongTensor of indices into the full graph, on the same device as
        train_idx.
    """
    if train_idx.numel() <= MAX_TRAIN_EVAL_EDGES:
        return train_idx

    # Uniform subsample, NOT a class-balanced one. Sampling uniformly keeps the
    # train split's natural ~0.075% prevalence, which is what makes train AUPRC
    # comparable to val AUPRC — and the train/val gap the reason to compute it
    # at all. Oversampling positives here would inflate train AUPRC and
    # manufacture a gap that is an artifact of the sampling.
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(train_idx.numel(), generator=generator)[:MAX_TRAIN_EVAL_EDGES]
    sampled = train_idx[perm.to(train_idx.device)]

    # AUPRC is undefined without positives; at ~2,298 positives in ~3.05M edges
    # a 1M sample expects ~750, so this is a guard against pathology, not a
    # routine path.
    if int((y[sampled] == 1).sum()) == 0:
        return train_idx
    return sampled


@dataclass
class TrainConfig:
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-5
    loss_type: str = "weighted_ce"  # "weighted_ce" or "focal"
    seed: int = 42
    log_every: int = 10


@dataclass
class TrainResult:
    model: torch.nn.Module
    history: list = field(default_factory=list)


def _compute_loss(logits, y, loss_type, pos_weight):
    if loss_type == "focal":
        return focal_loss(logits, y)
    return weighted_cross_entropy(logits, y, pos_weight)


def train_model(
    model: torch.nn.Module,
    data: Data,
    splits: dict,
    config: TrainConfig = TrainConfig(),
) -> TrainResult:
    """Train `model` on `data` using the given train/val split indices.

    Args:
        model: A model exposing forward(x, edge_index, edge_attr) -> logits
            over ALL edges (BaselineSAGE or HeterophilyGNN).
        data: PyG Data with x, edge_index, edge_attr, y.
        splits: dict with 'train'/'val'/'test' -> np.ndarray of edge indices,
            as produced by make_splits.make_temporal_split.
        config: TrainConfig.

    Returns:
        TrainResult with the trained model and per-epoch history.
    """
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # Index tensors must live wherever the caller put the graph — scripts/
    # move `data` and `model` onto the device before calling in.
    device = data.y.device
    train_idx = torch.as_tensor(splits["train"], dtype=torch.long, device=device)
    val_idx = torch.as_tensor(splits["val"], dtype=torch.long, device=device)
    pos_weight = compute_pos_weight(data.y[train_idx])

    # Fixed once so the train curve is comparable across epochs (CONVENTIONS.md
    # §5: minimum logged fields per epoch).
    train_eval_idx = select_train_eval_idx(train_idx, data.y, config.seed)

    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index, data.edge_attr)
        loss = _compute_loss(logits[train_idx], data.y[train_idx], config.loss_type, pos_weight)
        loss.backward()
        optimizer.step()

        if epoch % config.log_every == 0 or epoch == config.epochs:
            model.eval()
            with torch.no_grad():
                logits = model(data.x, data.edge_index, data.edge_attr)
                val_probs = torch.softmax(logits[val_idx], dim=1)[:, 1].cpu().numpy()
                val_pred = (val_probs >= 0.5).astype(int)
                val_metrics = classification_metrics(
                    data.y[val_idx].cpu().numpy(), val_pred, val_probs
                )
                tr_probs = torch.softmax(logits[train_eval_idx], dim=1)[:, 1].cpu().numpy()
                tr_pred = (tr_probs >= 0.5).astype(int)
                train_metrics = classification_metrics(
                    data.y[train_eval_idx].cpu().numpy(), tr_pred, tr_probs
                )
            record = {
                "epoch": epoch,
                "train_loss": float(loss.item()),
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
            history.append(record)
            print(
                f"epoch {epoch:4d} | train_loss {loss.item():.4f} | "
                f"train_auprc {train_metrics['auprc']:.4f} | "
                f"val_precision {val_metrics['precision']:.3f} | "
                f"val_recall {val_metrics['recall']:.3f} | "
                f"val_f1 {val_metrics['f1']:.3f} | "
                f"val_auprc {val_metrics['auprc']:.4f} | "
                f"gap {train_metrics['auprc'] - val_metrics['auprc']:+.4f}"
            )

    return TrainResult(model=model, history=history)
