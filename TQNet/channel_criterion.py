"""Channel-count-conditional TQ/attention criterion (Arm D — J-09).

Decides, from the training split alone, whether the Temporal Query and the
channel-attention layer (`TQNet.models.TQNet.Model.temporalQuery` /
`.channelAggregator`) are worth running at all. See
`report/prereg-improvement.md` section 3 "Arm D" and its `## Amendments`
block for the pre-registered derivation, prediction, and parameter-count
reconciliation this module implements. That file is frozen; this module
does not edit it and does not change its predictions or threshold after
the fact -- the threshold below is fixed once, here, before being applied.

**B2 (never select on leaked information).** The criterion reads training
rows only: `common.split.borders(seq_len)['train']`, which is `[0, 8640)`
for ETTh1's 12/4/4-month split and does not move with `seq_len` (only the
val/test borders do -- see `common/split.py`'s docstring, point 2). It never
reads validation or test rows, and never reads more of the training range
than that.

Statistic: the C x C Pearson channel-correlation matrix on those rows,
reduced to one scalar as the mean absolute value of its off-diagonal
entries (`STATISTIC_NAME`). Deterministic: closed-form `numpy.corrcoef` on
float64 data, no randomness, no fitting. Pearson correlation is invariant
to each channel's own affine (mean/scale) transform, so this gives the same
value whether it is computed on the raw CSV columns or on the z-scored data
`common/data.py` produces -- both were checked to agree to float rounding
(5.6e-17) during development of this module.
"""

import os
import sys

import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from common import split as split_mod  # noqa: E402  (sys.path insert must precede this)

__all__ = [
    "STATISTIC_NAME",
    "CHANNEL_CORR_DROP_THRESHOLD",
    "THRESHOLD_JUSTIFICATION",
    "compute_offdiag_statistic",
    "evaluate_criterion",
]

STATISTIC_NAME = "mean_abs_offdiag_pearson_correlation"

# Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*
# (2nd ed.), Lawrence Erlbaum Associates: a Pearson |r| of roughly 0.10 / 0.30
# / 0.50 is the conventional small / medium / large correlation magnitude.
# This threshold is external to ETTh1 and to this study -- it comes from a
# pre-existing statistical convention, not from tuning against Arm D's
# registered prediction or against any Stage-2 run. Reading: a training-only
# average |off-diagonal r| at or above the "medium" convention means the
# channels carry materially redundant information with each other, so the
# cross-channel mixing `channelAggregator` (attention over channels) and
# `temporalQuery` (indexed per channel) perform is largely recombining
# near-duplicate signal rather than exploiting complementary signal across
# channels -- which is a defensible basis, from the training split alone, to
# drop both rather than pay their parameter and quadratic-attention cost.
CHANNEL_CORR_DROP_THRESHOLD = 0.30
THRESHOLD_JUSTIFICATION = (
    "Cohen (1988) 'medium effect' convention for a Pearson correlation "
    "magnitude (small ~0.10, medium ~0.30, large ~0.50). Fixed in advance "
    "of, and external to, this study; not tuned to make any arm's "
    "prediction come out a particular way."
)


def _resolve_csv_path(root_path, data_path):
    return os.path.join(root_path, data_path)


def compute_offdiag_statistic(root_path, data_path, seq_len):
    """Mean |off-diagonal Pearson r| on the training rows only.

    Returns a dict: the statistic, the exact row range it was computed on
    (half-open, as `common.split.borders` defines it), the channel names
    (every non-`date` column, in file order -- matches how
    `TQNet/data_provider/data_loader.py` selects columns for `--features M`),
    and the full correlation matrix (for the log; not needed for the
    decision itself).
    """
    csv_path = _resolve_csv_path(root_path, data_path)
    if not os.path.exists(csv_path):
        raise FileNotFoundError("channel criterion: {} not found".format(csv_path))

    frame = pd.read_csv(csv_path)
    channel_cols = [c for c in frame.columns if c != "date"]
    if len(channel_cols) < 2:
        raise ValueError(
            "channel criterion needs at least 2 channels to have an "
            "off-diagonal entry, found {}".format(len(channel_cols))
        )

    train_start, train_stop = split_mod.borders(seq_len)["train"]
    train_rows = frame[channel_cols].values[train_start:train_stop].astype(np.float64)

    corr = np.corrcoef(train_rows, rowvar=False)
    if np.isnan(corr).any():
        raise ValueError(
            "channel criterion: NaN in the training-split correlation matrix "
            "(a channel is constant on rows [{}, {}))".format(train_start, train_stop)
        )

    c = corr.shape[0]
    offdiag_mask = ~np.eye(c, dtype=bool)
    offdiag = corr[offdiag_mask]
    statistic = float(np.mean(np.abs(offdiag)))

    return {
        "statistic_name": STATISTIC_NAME,
        "statistic_value": statistic,
        "row_range": [int(train_start), int(train_stop)],
        "n_channels": int(c),
        "channels": channel_cols,
        "csv_path": csv_path,
        "corr_matrix": corr.tolist(),
    }


def evaluate_criterion(root_path, data_path, seq_len):
    """Full criterion record: statistic + threshold + decision, ready to log.

    `decision` is `"drop"` (statistic >= threshold: turn the Temporal Query
    and channel attention both off, i.e. `use_tq=0, channel_aggre=0`) or
    `"keep"` (statistic < threshold: leave both on -- the published model).
    """
    record = compute_offdiag_statistic(root_path, data_path, seq_len)
    threshold = CHANNEL_CORR_DROP_THRESHOLD
    decision = "drop" if record["statistic_value"] >= threshold else "keep"

    record.update(
        {
            "threshold_name": "CHANNEL_CORR_DROP_THRESHOLD",
            "threshold_value": threshold,
            "threshold_justification": THRESHOLD_JUSTIFICATION,
            "decision": decision,
            "use_tq": 0 if decision == "drop" else 1,
            "channel_aggre": 0 if decision == "drop" else 1,
        }
    )
    return record


if __name__ == "__main__":
    # Manual sanity check: python3 channel_criterion.py [root_path] [data_path] [seq_len]
    import json

    _root = sys.argv[1] if len(sys.argv) > 1 else "./dataset/"
    _data = sys.argv[2] if len(sys.argv) > 2 else "ETTh1.csv"
    _seq = int(sys.argv[3]) if len(sys.argv) > 3 else 96
    _rec = evaluate_criterion(_root, _data, _seq)
    _rec.pop("corr_matrix", None)
    print(json.dumps(_rec, indent=2, sort_keys=True))
