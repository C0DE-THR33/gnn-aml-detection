"""
Generates a tiny synthetic Trans.csv / accounts.csv / Patterns.txt fixture
matching the real HI-Small schema, so the pipeline can be smoke-tested before
the real Kaggle files are downloaded. NOT a substitute for validating against
the real files — see README "First thing to do" section.
"""

import random
from pathlib import Path

random.seed(0)

FIXTURE_DIR = Path(__file__).parent / "fixtures"
FIXTURE_DIR.mkdir(exist_ok=True)

BANKS = [f"{i:05d}" for i in range(1, 6)]
CURRENCIES = ["US Dollar", "Euro"]
FORMATS = ["ACH", "Wire", "Cheque", "Credit Card"]


def rand_account():
    return "".join(random.choices("0123456789ABCDEF", k=9))


def rand_row(ts, frm_bank, frm_acct, to_bank, to_acct, is_laundering=0):
    amt = round(random.uniform(10, 20000), 2)
    cur = random.choice(CURRENCIES)
    fmt = random.choice(FORMATS)
    return f"{ts},{frm_bank},{frm_acct},{to_bank},{to_acct},{amt},{cur},{amt},{cur},{fmt},{is_laundering}"


def main():
    accounts = [rand_account() for _ in range(60)]
    trans_lines = ["Timestamp,From Bank,Account,To Bank,Account,Amount Received,"
                   "Receiving Currency,Amount Paid,Payment Currency,Payment Format,Is Laundering"]
    pattern_lines = []

    # Normal (licit) transactions
    for i in range(300):
        ts = f"2022/09/{1 + i % 10:02d} {i % 24:02d}:{i % 60:02d}"
        frm, to = random.sample(accounts, 2)
        trans_lines.append(rand_row(ts, random.choice(BANKS), frm, random.choice(BANKS), to, 0))

    # Injected typology patterns, mirroring the real BEGIN/END block format
    typology_headers = [
        "FAN-OUT", "FAN-IN", "CYCLE", "GATHER-SCATTER",
        "SCATTER-GATHER", "STACK", "BIPARTITE", "RANDOM",
    ]
    for typ in typology_headers:
        for attempt in range(6):  # spread attempts across the full date range
            pattern_lines.append(f"BEGIN LAUNDERING ATTEMPT - {typ}")
            hub = random.choice(accounts)
            n_edges = random.randint(3, 6)
            base_day = 1 + (attempt * 7) % 9  # spread across days 1-10
            for j in range(n_edges):
                ts = f"2022/09/{min(base_day + j, 10):02d} {j:02d}:{(j*7)%60:02d}"
                other = random.choice(accounts)
                if typ in ("FAN-OUT", "SCATTER-GATHER", "BIPARTITE", "STACK"):
                    row = rand_row(ts, random.choice(BANKS), hub, random.choice(BANKS), other, 1)
                else:
                    row = rand_row(ts, random.choice(BANKS), other, random.choice(BANKS), hub, 1)
                trans_lines.append(row)
                pattern_lines.append(row)
            pattern_lines.append(f"END LAUNDERING ATTEMPT - {typ}")

    (FIXTURE_DIR / "Toy-Small_Trans.csv").write_text("\n".join(trans_lines) + "\n")
    (FIXTURE_DIR / "Toy-Small_Patterns.txt").write_text("\n".join(pattern_lines) + "\n")

    account_lines = ["Account,Bank"]
    for acct in accounts:
        account_lines.append(f"{acct},{random.choice(BANKS)}")
    (FIXTURE_DIR / "Toy-Small_accounts.csv").write_text("\n".join(account_lines) + "\n")

    print(f"Wrote fixtures to {FIXTURE_DIR}")


if __name__ == "__main__":
    main()
