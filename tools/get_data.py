"""Fetch and verify ETTh1.csv, the one dataset this project needs.

The assignment allows shipping the dataset *or* documenting a download. We
document the download, because `.gitignore` excludes `*.csv` and because ETDataset
is the canonical upstream: TQNet's own README points there, and re-hosting a copy
would make it impossible to tell later whether we had evaluated on the same bytes
the paper did.

Verification is the point of this script, not the download. A silently different
CSV is the one failure mode that produces plausible-looking numbers that cannot be
compared to 0.3712, so the file is checked four ways -- byte digest, row count,
column names, and the date range -- and any mismatch is fatal.

Usage, from the repository root:

    python3 tools/get_data.py            # download if missing, then verify
    python3 tools/get_data.py --verify   # verify only, never touch the network
"""

import argparse
import hashlib
import os
import sys
import urllib.request

# ETDataset, CC BY-ND 4.0. Pinned to a commit rather than to `main`, so the URL
# cannot start serving different content later. This is the commit that last touched
# ETT-small/ETTh1.csv ("update ETT-small data", 2020-12-09); the file has not changed
# since, so this is also what `main` currently serves.
URL = ("https://raw.githubusercontent.com/zhouhaoyi/ETDataset/"
       "11ab373cf9c9f5be7698e219a5a170e1b1c8a930/ETT-small/ETTh1.csv")

# Digest of the file this project was developed and validated against.
EXPECTED_SHA256 = "f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066"

EXPECTED_ROWS = 17420
EXPECTED_COLUMNS = ["date", "HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
EXPECTED_FIRST_DATE = "2016-07-01 00:00:00"
EXPECTED_LAST_DATE = "2018-06-26 19:00:00"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# TQNet's scripts pass `--root_path ./dataset/` and are run from inside TQNet/.
DEST = os.path.join(REPO_ROOT, "TQNet", "dataset", "ETTh1.csv")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(dest, url=URL):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print("downloading {}\n         -> {}".format(url, dest))
    # Write to a temporary name first: an interrupted download must not leave a
    # truncated file that a later --verify would report as corrupt rather than absent.
    tmp = dest + ".part"
    with urllib.request.urlopen(url, timeout=120) as response, open(tmp, "wb") as handle:
        handle.write(response.read())
    os.replace(tmp, dest)
    print("downloaded {:.1f} MiB".format(os.path.getsize(dest) / 1024 ** 2))


def verify(dest, strict_digest=True):
    """Check the file four ways. Returns the digest; raises on any mismatch."""
    import pandas as pd

    if not os.path.exists(dest):
        raise SystemExit(
            "ETTh1.csv not found at {}\nRun: python3 tools/get_data.py".format(dest)
        )

    digest = sha256(dest)
    frame = pd.read_csv(dest)

    problems = []

    if strict_digest and digest != EXPECTED_SHA256:
        problems.append(
            "sha256 is {}\n    expected {}".format(digest, EXPECTED_SHA256)
        )
    if len(frame) != EXPECTED_ROWS:
        problems.append("row count is {}, expected {}".format(len(frame), EXPECTED_ROWS))
    if list(frame.columns) != EXPECTED_COLUMNS:
        problems.append(
            "columns are {}, expected {}".format(list(frame.columns), EXPECTED_COLUMNS)
        )
    else:
        first, last = str(frame["date"].iloc[0]), str(frame["date"].iloc[-1])
        if first != EXPECTED_FIRST_DATE:
            problems.append("first date is {}, expected {}".format(first, EXPECTED_FIRST_DATE))
        if last != EXPECTED_LAST_DATE:
            problems.append("last date is {}, expected {}".format(last, EXPECTED_LAST_DATE))

    print("path      : {}".format(dest))
    print("sha256    : {}".format(digest))
    print("rows      : {}".format(len(frame)))
    print("columns   : {}".format(", ".join(map(str, frame.columns))))
    print("date range: {} .. {}".format(frame["date"].iloc[0], frame["date"].iloc[-1]))
    print("missing   : {} cells".format(int(frame.isna().sum().sum())))

    if problems:
        raise SystemExit(
            "ETTh1.csv does not match the expected file:\n  - "
            + "\n  - ".join(problems)
            + "\n\nEvery number in this project is computed on 14,400 of these rows. A "
              "different file cannot be compared to the paper, so this is fatal rather "
              "than a warning."
        )

    print("\nOK: this is the ETTh1.csv the project was validated against.")
    return digest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--verify", action="store_true",
                        help="verify an existing file and never use the network")
    parser.add_argument("--force", action="store_true",
                        help="re-download even if the file is already present")
    parser.add_argument("--dest", default=DEST, help="destination path")
    parser.add_argument("--print-digest", action="store_true",
                        help="report the digest without checking it, for pinning a new file")
    args = parser.parse_args(argv)

    if not args.verify and (args.force or not os.path.exists(args.dest)):
        download(args.dest)
    elif os.path.exists(args.dest):
        print("already present, verifying without downloading")

    verify(args.dest, strict_digest=not args.print_digest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
