"""ETTh1 loading and windowing, reproducing TQNet's loader in plain numpy.

Why this exists at all, given that `TQNet/data_provider/data_loader.py` already
loads the data: the baseline and the leakage audit both need the *same windows* the
network sees, and neither of them wants a `torch.utils.data.Dataset`, a DataLoader,
or a GPU. A seasonal-naive baseline scored on almost-the-same windows, or on the raw
rather than the z-scored scale, is not comparable to 0.3712 -- and that mistake
stays invisible, because the resulting number is still a plausible MSE.

So this module is deliberately a *reimplementation of one specific file*, not an
improvement on it. Every step below is the upstream step, including the parts that
look redundant:

  * the scaler is fitted on rows `[0, 8640)` and then applied to **all 17,420 rows**
    before any slicing (`data_loader.py:61-63`). Slicing first would give identical
    numbers, but transforming first is what upstream does, and this file's job is to
    be indistinguishable from it;

  * the cycle index is the **absolute** CSV row number mod W, not an offset within
    the split (`data_loader.py:84`). Getting this wrong shifts every temporal query
    by a constant phase, which silently degrades the model rather than crashing;

  * the cycle index attached to a window is the one at `s_end` -- the first
    *forecast* step, not the first input step (`data_loader.py:97`). The paper's
    equation 9 reads as though it were the window start. The two agree only when W
    divides L, which is true here (96 = 4 x 24) and false on Electricity.

`tests/test_data.py` pins the equivalence element-by-element against the real
upstream `Dataset_ETT_hour`, so this claim is checked rather than asserted.

Kept to numpy, pandas and scikit-learn, and to syntax Python 3.8 accepts.
"""

import hashlib
import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from common import split as split_mod

__all__ = [
    "DEFAULT_CSV",
    "CHANNELS",
    "Windows",
    "data_sha256",
    "load_raw",
    "fit_scaler",
    "scaled_matrix",
    "make_windows",
    "seasonal_naive",
]

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CSV = os.path.join(_REPO_ROOT, "TQNet", "dataset", "ETTh1.csv")

# The seven numeric channels, in file order. Six power-load measurements plus oil
# temperature. `OT` is last, which is why `--features MS` predicts it.
CHANNELS = ("HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT")


class Windows(object):
    """One split's worth of supervised examples, plus what is needed to trace them.

    Attributes:
        x: inputs, shape (n_windows, seq_len, n_channels), z-scored.
        y: targets, shape (n_windows, pred_len, n_channels), z-scored.
        cycle_index: (n_windows,) int, the phase `row mod W` at the first forecast step.
        row_index: (n_windows,) int, the absolute CSV row of each window's first
            input step. Carried so that any window can be traced back to a timestamp,
            which is what makes a leakage claim checkable rather than rhetorical.
        split: "train", "val" or "test".
    """

    __slots__ = ("x", "y", "cycle_index", "row_index", "split")

    def __init__(self, x, y, cycle_index, row_index, split):
        self.x = x
        self.y = y
        self.cycle_index = cycle_index
        self.row_index = row_index
        self.split = split

    def __len__(self):
        return int(self.x.shape[0])

    def __repr__(self):
        return "Windows(split={!r}, n={}, x={}, y={})".format(
            self.split, len(self), self.x.shape, self.y.shape
        )


def data_sha256(csv_path: str = DEFAULT_CSV) -> str:
    """Digest of the CSV, which is half of what a split fingerprint means."""
    digest = hashlib.sha256()
    with open(csv_path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_raw(csv_path: str = DEFAULT_CSV) -> pd.DataFrame:
    """The CSV as-is: a `date` column plus the seven channels, 17,420 rows.

    No resampling, no interpolation, no gap filling. ETTh1 is already on a strict
    hourly grid with no missing values, and `tools/audit_split.py` verifies that
    rather than assuming it. Cleaning a clean series would be a silent deviation
    from the paper.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            "{} not found. Run: python3 tools/get_data.py".format(csv_path)
        )

    frame = pd.read_csv(csv_path, parse_dates=["date"])
    missing = [c for c in CHANNELS if c not in frame.columns]
    if missing:
        raise ValueError("ETTh1.csv is missing expected channels: {}".format(missing))
    return frame


def fit_scaler(frame: pd.DataFrame, seq_len: int = 96) -> StandardScaler:
    """A `StandardScaler` fitted on the **training rows only**.

    This is the single most important line in the pipeline for the brief's
    no-future-information requirement. The training range is `[0, 8640)` and is
    independent of `seq_len`; the argument is accepted only so callers do not have to
    know that, and so this signature does not change if the split scheme ever does.

    `sklearn`'s scaler uses the population standard deviation (ddof=0), matching
    upstream, which passes `df_data.values` to the same class.
    """
    train_start, train_stop = split_mod.borders(seq_len)["train"]
    scaler = StandardScaler()
    scaler.fit(frame[list(CHANNELS)].values[train_start:train_stop])
    return scaler


def scaled_matrix(
    frame: pd.DataFrame, scaler: Optional[StandardScaler] = None, seq_len: int = 96
) -> Tuple[np.ndarray, StandardScaler]:
    """All 17,420 rows, z-scored with the train-fitted scaler.

    Transforming the whole series and slicing afterwards is upstream's order of
    operations. It does not leak: the scaler's mean and variance come only from the
    training rows, and applying a fixed affine map to later rows tells the model
    nothing about them.
    """
    if scaler is None:
        scaler = fit_scaler(frame, seq_len=seq_len)
    return scaler.transform(frame[list(CHANNELS)].values), scaler


def make_windows(
    which: str,
    seq_len: int = 96,
    pred_len: int = 96,
    cycle: int = 24,
    csv_path: str = DEFAULT_CSV,
    scaler: Optional[StandardScaler] = None,
    frame: Optional[pd.DataFrame] = None,
) -> Windows:
    """Build one split's windows at stride 1, exactly as the upstream loader does.

    Stride 1 with a single fixed model is *rolling-origin evaluation*: the forecast
    origin advances one hour at a time across the test months and the model is never
    refitted. That is the protocol the paper uses and the one the brief names as
    acceptable. It is deliberately not walk-forward validation, which refits per
    fold and would produce numbers not comparable to the paper's.
    """
    if which not in split_mod.SPLIT_NAMES:
        raise ValueError(
            "unknown split {!r}; expected one of {}".format(which, split_mod.SPLIT_NAMES)
        )

    if frame is None:
        frame = load_raw(csv_path)
    data, _ = scaled_matrix(frame, scaler=scaler, seq_len=seq_len)

    start, stop = split_mod.borders(seq_len)[which]
    if stop > len(data):
        raise ValueError(
            "split {!r} needs rows up to {} but the CSV has {}".format(which, stop, len(data))
        )

    # Absolute row index mod W, then restricted to this split -- upstream's order.
    cycle_all = np.arange(len(data)) % cycle
    rows = data[start:stop]
    cycle_rows = cycle_all[start:stop]

    count = len(rows) - seq_len - pred_len + 1
    if count <= 0:
        raise ValueError(
            "split {!r} has {} rows, too few for seq_len={} + pred_len={}".format(
                which, len(rows), seq_len, pred_len
            )
        )

    # A strided view would avoid the copy, but these arrays are ~180 MB at most and a
    # view here would alias into `rows`, making an accidental in-place write corrupt
    # every other window. Not worth the risk for the memory saved.
    offsets = np.arange(count)
    x_idx = offsets[:, None] + np.arange(seq_len)[None, :]
    y_idx = offsets[:, None] + seq_len + np.arange(pred_len)[None, :]

    return Windows(
        x=rows[x_idx],
        y=rows[y_idx],
        cycle_index=cycle_rows[offsets + seq_len],
        row_index=start + offsets,
        split=which,
    )


def seasonal_naive(window: Windows, period: int = 24) -> np.ndarray:
    """Seasonal-naive forecast: repeat the last observed period, tile it forward.

    The required simple baseline (brief requirement B6). For horizon step `h`, the
    prediction is the input value `period` steps before the corresponding future
    time, which for h >= period means re-using an already-tiled value:

        yhat[h] = x[-period + (h mod period)]

    Two things make this comparable to TQNet rather than merely adjacent to it, and
    both are easy to get wrong:

      * it is computed on the **z-scored** series, using the train-fitted scaler,
        because that is the scale the paper's 0.3712 lives on;
      * it is computed on the **identical windows**, because it takes a `Windows`
        object rather than re-deriving its own.

    Period 24 matches TQNet's own `--cycle 24` for ETTh1, so the baseline and the
    model are making the same periodicity assumption and the comparison isolates
    what the network adds on top of it.
    """
    if period <= 0:
        raise ValueError("period must be positive, got {}".format(period))
    if period > window.x.shape[1]:
        raise ValueError(
            "period {} exceeds the look-back length {}, so the last full season is "
            "not in the input".format(period, window.x.shape[1])
        )

    pred_len = window.y.shape[1]
    last_season = window.x[:, -period:, :]
    reps = int(np.ceil(pred_len / float(period)))
    return np.tile(last_season, (1, reps, 1))[:, :pred_len, :]


def summary(seq_len: int = 96, pred_len: int = 96, csv_path: str = DEFAULT_CSV) -> Dict[str, object]:
    """Everything the audit and the report need to state about the data, in one dict."""
    frame = load_raw(csv_path)
    scaler = fit_scaler(frame, seq_len=seq_len)
    digest = data_sha256(csv_path)

    return {
        "csv_path": csv_path,
        "data_sha256": digest,
        "rows_in_file": int(len(frame)),
        "channels": list(CHANNELS),
        "first_timestamp": str(frame["date"].iloc[0]),
        "last_timestamp": str(frame["date"].iloc[-1]),
        "split": split_mod.describe(seq_len, pred_len),
        "split_hash": split_mod.split_hash(seq_len, pred_len, digest),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }
