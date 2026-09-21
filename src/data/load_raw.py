"""
Loaders for the three raw HI-Small files: Trans.csv, accounts.csv, Patterns.txt.

Schema notes (confirmed against the public IBM AML / AMLworld documentation):

Trans.csv columns, in order:
    Timestamp, From Bank, Account, To Bank, Account (dup name -> "Account.1"
    after pandas dedup), Amount Received, Receiving Currency, Amount Paid,
    Payment Currency, Payment Format, Is Laundering

Patterns.txt structure (confirmed format):
    BEGIN LAUNDERING ATTEMPT - <TYPOLOGY>[: extra info]
    <transaction row, same 11 columns as Trans.csv, comma-separated, no header>
    ...
    END LAUNDERING ATTEMPT - <TYPOLOGY>

accounts.csv: schema is NOT independently confirmed for this specific Kaggle
release (some derived/companion versions of this dataset include one, some
don't). load_accounts() is written defensively: it inspects whatever columns
are present, tries to identify an account-ID and bank-ID column by common
naming patterns, and raises a clear error naming the actual columns found if
it can't. Check the printed column names against reality the first time you
run this on the real file, and adjust ACCOUNT_ID_CANDIDATES /
BANK_ID_CANDIDATES below if needed rather than guessing blind.
"""

import re
from pathlib import Path
from typing import Optional

import pandas as pd

from src.data.typologies import normalize_typology

TRANS_COLUMNS = [
    "timestamp",
    "from_bank",
    "from_account",
    "to_bank",
    "to_account",
    "amount_received",
    "receiving_currency",
    "amount_paid",
    "payment_currency",
    "payment_format",
    "is_laundering",
]

# Payment Format values confirmed present in the real HI-Small Trans.csv (and
# Patterns.txt). Fixed and canonical per CONVENTIONS.md §2 — never re-derive
# this list from whatever a given file happens to contain, or the one-hot
# width (and column order) drifts between the real data and the synthetic
# fixture, and between runs if a rare format is absent from a subsample.
# graph_builder.py buckets anything outside this list into a shared "other"
# column rather than dropping or crashing on it.
#
# Rates are the deciding signal for including this as an edge feature at all:
# illicit rate is 0.75% for ACH vs 0.00% for Wire/Reinvestment across the full
# HI-Small Trans.csv — a ~42x spread the model previously had no access to
# (edge_attr was log-amount only). Receiving/Payment Currency was checked too
# (0.09%-0.42%, ~4x spread) and left out: real but much weaker, and its 15
# categories would have diluted a strong signal with a weak, noisier one for
# the width they'd cost. Revisit if amount + payment format alone plateaus.
PAYMENT_FORMATS: tuple[str, ...] = (
    "ACH",
    "Bitcoin",
    "Cash",
    "Cheque",
    "Credit Card",
    "Reinvestment",
    "Wire",
)

ACCOUNT_ID_CANDIDATES = ["account", "account_number", "account_id", "acct"]
BANK_ID_CANDIDATES = ["bank", "bank_id", "from_bank", "institution"]

_PATTERN_HEADER_RE = re.compile(
    r"^\s*BEGIN LAUNDERING ATTEMPT\s*-\s*([A-Z\-]+)", re.IGNORECASE
)
_PATTERN_FOOTER_RE = re.compile(r"^\s*END LAUNDERING ATTEMPT", re.IGNORECASE)


def load_transactions(path: str | Path) -> pd.DataFrame:
    """Load Trans.csv into a DataFrame with normalized column names.

    Args:
        path: Path to e.g. HI-Small_Trans.csv.

    Returns:
        DataFrame with columns TRANS_COLUMNS, timestamp parsed as datetime,
        is_laundering cast to int.
    """
    # from_bank/to_bank/account columns are ID-like strings (e.g. "00004")
    # where leading zeros are meaningful for exact joins against accounts.csv
    # — force them to stay strings rather than let pandas infer int and
    # silently drop the leading zeros.
    df = pd.read_csv(
        path, header=0, dtype={0: str, 1: str, 2: str, 3: str, 4: str}
    )
    if len(df.columns) != len(TRANS_COLUMNS):
        raise ValueError(
            f"Expected {len(TRANS_COLUMNS)} columns in {path}, found "
            f"{len(df.columns)}: {list(df.columns)}. The loader assumes the "
            "standard 11-column Trans.csv layout — inspect the file and "
            "update TRANS_COLUMNS if the real schema differs."
        )
    df.columns = TRANS_COLUMNS
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["is_laundering"] = df["is_laundering"].astype(int)
    # Stable transaction ID (row order) — used to join against Patterns.txt
    # extracted transactions and to key explanation/fidelity outputs.
    df["txn_id"] = df.index.astype(str)
    return df


def load_accounts(path: str | Path) -> pd.DataFrame:
    """Load accounts.csv defensively — see module docstring.

    Args:
        path: Path to e.g. HI-Small_accounts.csv.

    Returns:
        DataFrame with at least a normalized "account_id" column, plus
        whatever other columns were present (renamed to snake_case).
    """
    # Read everything as string — account/bank IDs are ID-like and may have
    # meaningful leading zeros; numeric feature columns (if any) can be cast
    # downstream once the real schema is confirmed.
    df = pd.read_csv(path, dtype=str)
    original_columns = list(df.columns)
    df.columns = [re.sub(r"[^0-9a-zA-Z]+", "_", c).strip("_").lower() for c in df.columns]

    account_col = _find_column(df.columns, ACCOUNT_ID_CANDIDATES)
    if account_col is None:
        raise ValueError(
            f"Could not identify an account-ID column in {path}. "
            f"Columns found: {original_columns}. Add the real column name "
            "to ACCOUNT_ID_CANDIDATES in load_raw.py."
        )
    df = df.rename(columns={account_col: "account_id"})

    bank_col = _find_column(df.columns, BANK_ID_CANDIDATES)
    if bank_col is not None and bank_col != "bank_id":
        df = df.rename(columns={bank_col: "bank_id"})

    return df


def parse_patterns(path: str | Path) -> pd.DataFrame:
    """Parse Patterns.txt into a per-transaction typology label table.

    Each row in the returned DataFrame corresponds to one transaction line
    inside a BEGIN/END LAUNDERING ATTEMPT block, tagged with its Typology
    and a pattern_group_id so transactions from the same laundering attempt
    can be grouped (useful for both graph construction sanity checks and
    for building per-typology "reference subgraphs" for fidelity scoring).

    Args:
        path: Path to e.g. HI-Small_Patterns.txt.

    Returns:
        DataFrame with columns: pattern_group_id, typology (Typology enum
        value, as string), and TRANS_COLUMNS (unindexed — join back to
        load_transactions() output on the transaction's raw fields if you
        need txn_id; Patterns.txt does not carry txn_id directly).
    """
    rows = []
    group_id = -1
    current_typology = None

    with open(path, "r") as f:
        for line in f:
            header_match = _PATTERN_HEADER_RE.match(line)
            if header_match:
                group_id += 1
                raw_label = header_match.group(1)
                current_typology = normalize_typology(raw_label)
                continue
            if _PATTERN_FOOTER_RE.match(line):
                current_typology = None
                continue
            if current_typology is not None and line.strip():
                fields = line.strip().split(",")
                if len(fields) != len(TRANS_COLUMNS):
                    continue  # skip malformed lines rather than crash
                row = dict(zip(TRANS_COLUMNS, fields))
                row["pattern_group_id"] = group_id
                row["typology"] = current_typology.value
                rows.append(row)

    result = pd.DataFrame(rows)
    if not result.empty:
        result["timestamp"] = pd.to_datetime(result["timestamp"])
        result["is_laundering"] = result["is_laundering"].astype(int)
        result["amount_received"] = result["amount_received"].astype(float)
        result["amount_paid"] = result["amount_paid"].astype(float)
    return result


def _find_column(columns, candidates) -> Optional[str]:
    """Find the column matching one of `candidates`, preferring exact names.

    Exact matches win over substring matches, in candidate priority order. The
    previous single-pass version returned the first substring hit, so on the
    real accounts.csv ("Bank Name", "Bank ID", ...) the candidate "bank"
    matched "bank_name" before the exact "bank_id" was reached; the bank NAME
    was then renamed to "bank_id", leaving two columns with that name.

    Args:
        columns: Iterable of normalized (snake_case) column names.
        candidates: Acceptable names, most preferred first.

    Returns:
        The matching column name, or None.
    """
    cols = list(columns)
    for cand in candidates:
        if cand in cols:
            return cand
    for cand in candidates:
        for col in cols:
            if cand in col:
                return col
    return None
