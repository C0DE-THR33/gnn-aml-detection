"""
Generates a temporal train/val/test split over edges (transactions).

Split strategy (per CONVENTIONS.md §6 — document the strategy used):
    Transactions are sorted by timestamp and split 60/20/20 into
    train/val/test. This avoids the leakage of "predicting the past from
    the future" that a random edge split would allow, and mirrors how the
    model would actually be used (train on history, evaluate on what comes
    next).

Known limitation (flagged in SRS §2.5 / §6.3, not solved here): node
features (in/out-degree) are computed over the FULL graph regardless of
split, so some information from future edges leaks into node features used
at train time. Solving this properly needs a temporal graph formulation,
which SRS §1.2 explicitly defers to future work. Documented here so it
isn't silently forgotten.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def make_temporal_split(
    trans: pd.DataFrame,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    seed: int = 42,
) -> dict:
    """Split transaction row indices into train/val/test by time order.

    Args:
        trans: Output of load_raw.load_transactions() (must have 'timestamp').
        train_frac: Fraction of (time-ordered) transactions for training.
        val_frac: Fraction for validation (remainder goes to test).
        seed: Unused for the split itself (it's deterministic/temporal), but
            accepted for interface consistency with anything downstream that
            expects a seed.

    Returns:
        Dict with keys 'train', 'val', 'test', each a 1D numpy array of row
        indices into `trans` (in its original, not time-sorted, order).
    """
    del seed  # deterministic split, kept for interface consistency only
    order = trans["timestamp"].to_numpy().argsort(kind="stable")
    n = len(order)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_idx = order[:n_train]
    val_idx = order[n_train : n_train + n_val]
    test_idx = order[n_train + n_val :]

    return {"train": train_idx, "val": val_idx, "test": test_idx}


def save_splits(splits: dict, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, idx in splits.items():
        np.save(out_dir / f"{name}_idx.npy", idx)
