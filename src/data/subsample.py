"""Transaction-level subsampling for runs that can't hold the full graph.

SRS NFR-2 allows the full HI-Small run to "be subsampled if this is not
achievable" on the available hardware, and the SRS risk table names
subsampling as the mitigation for training cost. This module implements that
fallback explicitly rather than leaving it as an ad-hoc edit to a script.

Strategy: `temporal_prefix` keeps the EARLIEST n transactions by timestamp.
Chosen over a random edge sample because a random sample would shred the
laundering pattern subgraphs (a 16-edge fan-out would survive as ~2 unrelated
edges), and per-typology explanation fidelity — the point of this project —
is only meaningful when those subgraphs stay intact. A contiguous prefix
keeps every pattern that falls entirely inside the window, keeps the natural
class balance, and leaves make_splits.make_temporal_split's train/val/test
ordering semantics unchanged.

Known cost, to be reported rather than hidden: patterns straddling the cutoff
are truncated, and the run covers only the first part of HI-Small's 10-day
window, so per-typology counts shrink roughly in proportion.
"""

from typing import Optional

import pandas as pd


def temporal_prefix(trans: pd.DataFrame, n_transactions: int) -> pd.DataFrame:
    """Keep the earliest `n_transactions` rows by timestamp.

    Args:
        trans: Output of load_raw.load_transactions() (must have 'timestamp').
        n_transactions: Number of transactions to keep. Values >= len(trans)
            return the frame unchanged.

    Returns:
        A DataFrame containing the earliest `n_transactions` rows, reindexed
        from 0 so that downstream row indices (txn_id, split indices, edge
        indices) stay contiguous and mutually consistent.
    """
    if n_transactions >= len(trans):
        return trans

    order = trans["timestamp"].to_numpy().argsort(kind="stable")[:n_transactions]
    kept = trans.iloc[sorted(order)].reset_index(drop=True)
    kept["txn_id"] = kept.index.astype(str)
    return kept


def apply_subsample(trans: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Apply the config's `subsample` block, if present.

    Args:
        trans: Output of load_raw.load_transactions().
        config: Parsed experiment config. Reads the optional top-level
            `subsample` block: {strategy: "temporal_prefix", n_transactions: int}.

    Returns:
        The subsampled DataFrame, or `trans` unchanged when no `subsample`
        block is configured.

    Raises:
        ValueError: If `subsample.strategy` is not a recognized strategy.
    """
    spec = config.get("subsample")
    if not spec:
        return trans

    strategy = spec.get("strategy", "temporal_prefix")
    if strategy != "temporal_prefix":
        raise ValueError(
            f"Unknown subsample.strategy '{strategy}' — only 'temporal_prefix' "
            "is implemented (see module docstring for why random edge sampling "
            "is deliberately not offered)."
        )
    return temporal_prefix(trans, int(spec["n_transactions"]))


def processed_stem(config: dict) -> str:
    """Basename for this config's artifacts under data/processed/.

    The dataset tag stays `hismall` per CONVENTIONS.md §2; a subsampled run
    gets a suffix appended so it doesn't overwrite the full-graph artifacts
    (one config = one reproducible experiment, CONVENTIONS.md §4).

    Args:
        config: Parsed experiment config.

    Returns:
        e.g. "hismall" for a full run, "hismall_sub2000000" for a subsampled one.
    """
    name = config["dataset"]["name"]
    spec: Optional[dict] = config.get("subsample")
    if not spec:
        return name
    return f"{name}_sub{int(spec['n_transactions'])}"
