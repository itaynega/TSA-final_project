"""The ETTh1 split, stated once so that every arm of the project shares it.

Requirement C2 of the brief is pass/fail: the improved method must be evaluated on
the same split as the reconstruction. The only way to keep that promise is to have
exactly one definition of the split, and a fingerprint that a run can assert
against. Both live here.

The numbers are not ours. They are read out of TQNet's own
`data_provider/data_loader.py:49-50`, which inherits them from Informer (Zhou et
al., 2021) and which every model in this comparison -- iTransformer, TimeXer,
CycleNet -- also uses:

    border1s = [0, 12*30*24 - L, 12*30*24 + 4*30*24 - L]
    border2s = [12*30*24,  12*30*24 + 4*30*24,  12*30*24 + 8*30*24]

So the split is chronological and expressed in *calendar months of 30 days*:
12 months train, 4 months validation, 4 months test. Three consequences are worth
stating explicitly, because each one is a way to get a number that looks right and
is not comparable to the paper:

  1. **20 months, not 24.** 12 + 4 + 4 = 20 months = 14,400 hourly rows, but
     ETTh1.csv has 17,420. The final 3,020 rows -- roughly four months -- are never
     touched by any split. The paper's Table 1 lists ETTh1 as 14,400 timesteps
     without mentioning that rows were dropped. Reproducing the paper means
     dropping them too.

  2. **Validation and test start `seq_len` early.** Subtracting L from the two
     later `border1`s is not leakage in the direction that matters: it means the
     *first target* of each split lands exactly on the month boundary, and the
     history feeding that target is the real history rather than zeros. Data before
     the boundary is only ever used as model input, never as a target, and never to
     fit anything.

  3. **The 6:2:2 ratio in the paper's Appendix A.2 describes the 14,400 rows it
     kept**, not the file. Against the file it is nearer 50:17:17.

Kept to the standard library and numpy, and to syntax Python 3.8 accepts, because
this has to import inside TQNet's pinned environment as well as ours.
"""

import hashlib
import json
from typing import Dict, Tuple

__all__ = [
    "SPLIT_NAMES",
    "MONTHS",
    "TOTAL_USED_ROWS",
    "borders",
    "split_lengths",
    "n_windows",
    "describe",
    "split_hash",
]

SPLIT_NAMES = ("train", "val", "test")

# 30-day months, which is what the loader's arithmetic means by "month".
HOURS_PER_MONTH = 30 * 24
MONTHS = {"train": 12, "val": 4, "test": 4}
TOTAL_USED_ROWS = HOURS_PER_MONTH * (MONTHS["train"] + MONTHS["val"] + MONTHS["test"])  # 14400


def borders(seq_len: int) -> Dict[str, Tuple[int, int]]:
    """Half-open row ranges `[start, stop)` into the raw CSV, per split.

    Transcribed from `data_provider/data_loader.py:49-50`. `seq_len` enters because
    the validation and test ranges are extended backwards by one look-back window;
    see point 2 in the module docstring.
    """
    train_end = HOURS_PER_MONTH * MONTHS["train"]
    val_end = train_end + HOURS_PER_MONTH * MONTHS["val"]
    test_end = val_end + HOURS_PER_MONTH * MONTHS["test"]

    return {
        "train": (0, train_end),
        "val": (train_end - seq_len, val_end),
        "test": (val_end - seq_len, test_end),
    }


def split_lengths(seq_len: int) -> Dict[str, int]:
    """Number of raw rows in each split range."""
    return {name: stop - start for name, (start, stop) in borders(seq_len).items()}


def n_windows(seq_len: int, pred_len: int) -> Dict[str, int]:
    """Number of (input, target) windows each split yields at stride 1.

    Mirrors `Dataset_ETT_hour.__len__`: `len(rows) - seq_len - pred_len + 1`. At
    L = H = 96 this gives train 8,449 and val/test 2,785 each -- the 2,785 that the
    paper's 0.3712 is averaged over.
    """
    return {
        name: length - seq_len - pred_len + 1
        for name, length in split_lengths(seq_len).items()
    }


def describe(seq_len: int, pred_len: int) -> Dict[str, object]:
    """A canonical, JSON-serialisable statement of the split.

    This is both what gets printed by the audit and what gets hashed, so that the
    fingerprint cannot drift away from the thing it is supposed to fingerprint.
    """
    return {
        "dataset": "ETTh1",
        "scheme": "informer-monthly-chronological",
        "months": dict(MONTHS),
        "hours_per_month": HOURS_PER_MONTH,
        "rows_used": TOTAL_USED_ROWS,
        "seq_len": seq_len,
        "pred_len": pred_len,
        "stride": 1,
        "borders": {name: list(rng) for name, rng in borders(seq_len).items()},
        "lengths": split_lengths(seq_len),
        "windows": n_windows(seq_len, pred_len),
        "scaler": "sklearn.StandardScaler fit on rows [0, 8640) only",
    }


def split_hash(seq_len: int, pred_len: int, data_sha256: str) -> str:
    """A fingerprint of "which windows of which file", as 16 hex characters.

    Includes the data digest deliberately. Two runs that agree on the boundary
    arithmetic but disagree on the underlying CSV are *not* on the same split, and a
    hash over the arithmetic alone would call them identical -- which is exactly the
    mistake this is meant to catch.

    `common.results.assert_split_hash` compares two of these and refuses to accept a
    missing one, so a run that forgot to record its split fails loudly rather than
    quietly joining a comparison table it does not belong in.
    """
    if not data_sha256:
        raise ValueError(
            "data_sha256 is required: a split fingerprint that ignores the data file "
            "would call two different CSVs the same split"
        )

    payload = dict(describe(seq_len, pred_len))
    payload["data_sha256"] = data_sha256
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
