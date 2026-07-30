"""The paper's own numbers, transcribed once, with their provenance attached.

Every figure in this file was read off a specific page of `files/project/TQnet.pdf`
or a specific line of TQNet's committed `result.txt`. Nothing here is computed and
nothing here is ours. It is kept as code rather than as a markdown table so that the
report's comparison tables are assembled by a program that cannot make a
transcription error twice in two places.

The distinction that matters, and the reason the reproduction has a three-way check
instead of a two-way one:

  * **`PAPER_TABLE_5`** is what the paper prints -- three decimals, e.g. 0.371.
  * **`AUTHORS_RESULT_TXT`** is what the authors' own run produced, at full float
    precision, committed to `result.txt` in the repository. For ETTh1/96 that is
    0.3712165653705597.

Because the second exists, a mismatch is diagnosable rather than ambiguous. Agreeing
to many decimals means our environment is effectively theirs; agreeing to two or
three means ordinary hardware and library drift; disagreeing in the first decimal
means the fault is ours and should be looked for here, not in the paper.
"""

from typing import Dict, Optional, Tuple

__all__ = [
    "TARGET_CELL",
    "AUTHORS_RESULT_TXT",
    "PAPER_TABLE_5",
    "SEED_STUDY",
    "DATASET_FACTS",
    "HYPERPARAMETERS",
    "target_reference",
    "best_baseline",
]

# ---------------------------------------------------------------------------
# The one cell we reproduce.
# ---------------------------------------------------------------------------

TARGET_CELL = {
    "dataset": "ETTh1",
    "features": "M",          # multivariate in, multivariate out: 7 -> 7
    "seq_len": 96,
    "pred_len": 96,
    "cycle": 24,
    "seed": 2024,
    "setting": "ETTh1_96_96_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed2024",
    "mse": 0.3712165653705597,
    "mae": 0.3928201496601105,
    "source": "TQNet result.txt, committed by the authors at 15e19cb",
}

# ---------------------------------------------------------------------------
# Authors' own execution output, ETTh1, all four horizons, full precision.
# Source: `TQNet/result_authors_reference.txt` (our untouched copy of upstream
# `result.txt`). Keyed by pred_len.
# ---------------------------------------------------------------------------

AUTHORS_RESULT_TXT = {
    96: {"mse": 0.3712165653705597, "mae": 0.3928201496601105},
    192: {"mse": 0.4283985197544098, "mae": 0.4260946214199066},
    336: {"mse": 0.475707083940506, "mae": 0.4460628032684326},
    720: {"mse": 0.48742958903312683, "mae": 0.46976661682128906},
}

# ---------------------------------------------------------------------------
# Paper Table 5, p. 15 -- the ETTh1 block. Rounded to three decimals as printed.
#
# The caption states that the baseline columns were *copied* from TimeXer,
# iTransformer and CycleNet rather than re-run by TQNet's authors. So each baseline
# inherits whatever setup its own paper used. They are context for our result, not
# a like-for-like comparison, and the report has to say so.
# ---------------------------------------------------------------------------

PAPER_TABLE_5 = {
    96: {
        "TQNet": (0.371, 0.393),
        "TimeXer": (0.382, 0.403),
        "CycleNet": (0.375, 0.395),
        "iTransformer": (0.386, 0.405),
        "MSGNet": (0.390, 0.411),
        "TimesNet": (0.384, 0.402),
        "PatchTST": (0.414, 0.419),
        "Crossformer": (0.423, 0.448),
        "DLinear": (0.386, 0.400),
        "SCINet": (0.654, 0.599),
    },
    192: {
        "TQNet": (0.428, 0.426),
        "TimeXer": (0.429, 0.435),
        "CycleNet": (0.436, 0.428),
        "iTransformer": (0.441, 0.436),
        "MSGNet": (0.443, 0.442),
        "TimesNet": (0.436, 0.429),
        "PatchTST": (0.460, 0.445),
        "Crossformer": (0.471, 0.474),
        "DLinear": (0.437, 0.432),
        "SCINet": (0.719, 0.631),
    },
    336: {
        "TQNet": (0.476, 0.446),
        "TimeXer": (0.468, 0.448),
        "CycleNet": (0.496, 0.455),
        "iTransformer": (0.487, 0.458),
        "MSGNet": (0.482, 0.469),
        "TimesNet": (0.491, 0.469),
        "PatchTST": (0.501, 0.466),
        "Crossformer": (0.570, 0.546),
        "DLinear": (0.481, 0.459),
        "SCINet": (0.778, 0.659),
    },
    720: {
        "TQNet": (0.487, 0.470),
        "TimeXer": (0.469, 0.461),
        "CycleNet": (0.520, 0.484),
        "iTransformer": (0.503, 0.491),
        "MSGNet": (0.496, 0.488),
        "TimesNet": (0.521, 0.500),
        "PatchTST": (0.500, 0.488),
        "Crossformer": (0.653, 0.621),
        "DLinear": (0.519, 0.516),
        "SCINet": (0.836, 0.699),
    },
}

# ---------------------------------------------------------------------------
# Paper Table 9, p. 18 -- TQNet on ETTh1 under three random seeds.
#
# This is the single most important table for judging any result, ours or theirs.
# At horizon 96 the run-to-run standard deviation is 0.001 MSE while TQNet's margin
# over the best baseline is 0.004. Any claimed improvement smaller than a few
# thousandths is inside the noise, and the report must not present it as real.
# ---------------------------------------------------------------------------

SEED_STUDY = {
    96: {"seeds": {2024: (0.371, 0.393), 2025: (0.371, 0.394), 2026: (0.372, 0.393)},
         "mean": (0.371, 0.393), "std": (0.001, 0.000)},
    192: {"seeds": {2024: (0.428, 0.426), 2025: (0.430, 0.424), 2026: (0.430, 0.423)},
          "mean": (0.429, 0.424), "std": (0.001, 0.002)},
    336: {"seeds": {2024: (0.476, 0.446), 2025: (0.481, 0.453), 2026: (0.476, 0.446)},
          "mean": (0.478, 0.448), "std": (0.003, 0.004)},
    720: {"seeds": {2024: (0.487, 0.470), 2025: (0.510, 0.487), 2026: (0.491, 0.472)},
          "mean": (0.496, 0.476), "std": (0.012, 0.009)},
}

# ---------------------------------------------------------------------------
# Paper Table 1, p. 5, ETTh1 row -- and the one place it disagrees with the file.
# ---------------------------------------------------------------------------

DATASET_FACTS = {
    "channels": 7,
    "timesteps_claimed": 14400,   # Table 1
    "timesteps_in_csv": 17420,    # measured; the loader stops at 14,400
    "interval": "1 hour",
    "W": 24,
    "domain": "Electricity",
    "note": ("Table 1 reports 14,400 timesteps. ETTh1.csv has 17,420 rows and the "
             "loader hard-stops at 14,400, discarding the final 3,020 rows (~4 "
             "months) without comment anywhere in the paper."),
}

# ---------------------------------------------------------------------------
# The pinned configuration for the target cell, and where each value comes from.
# "hard-coded" means the value is a literal in the model source with no flag, so it
# cannot be reported as a tunable hyperparameter.
# ---------------------------------------------------------------------------

HYPERPARAMETERS = [
    # (name, value, source)
    ("look-back L (seq_len)", 96, "scripts/TQNet/etth1.sh"),
    ("horizon H (pred_len)", 96, "scripts/TQNet/etth1.sh"),
    ("channels C (enc_in)", 7, "scripts/TQNet/etth1.sh"),
    ("cycle W", 24, "scripts/TQNet/etth1.sh"),
    ("d_model", 512, "run.py default, never overridden"),
    ("attention heads", 4, "hard-coded, models/TQNet.py:24"),
    ("attention dropout", 0.5, "hard-coded, models/TQNet.py:24"),
    ("output dropout", 0.5, "scripts/TQNet/etth1.sh (run.py default is 0)"),
    ("instance norm (use_revin)", 1, "run.py default"),
    ("loss", "MSE (L2)", "exp/exp_main.py:_select_criterion"),
    ("optimiser", "Adam", "exp/exp_main.py:_select_optimizer"),
    ("learning rate", 0.001, "scripts/TQNet/etth1.sh"),
    ("lr schedule", "type3", "utils/tools.py:adjust_learning_rate"),
    ("batch size", 256, "scripts/TQNet/etth1.sh"),
    ("epochs", 30, "scripts/TQNet/etth1.sh"),
    ("early-stopping patience", 5, "scripts/TQNet/etth1.sh, on validation MSE"),
    ("seed", 2024, "scripts/TQNet/etth1.sh, single seed"),
    ("trainable parameters", 661640, "measured by instantiating the model"),
]


def target_reference(pred_len: int = 96) -> Dict[str, float]:
    """The authors' full-precision MSE/MAE for an ETTh1 horizon.

    Raises rather than returning None for an unknown horizon: silently comparing
    against a missing reference is how an empty column ends up in a report table.
    """
    if pred_len not in AUTHORS_RESULT_TXT:
        raise KeyError(
            "no authors' reference for pred_len={}; have {}".format(
                pred_len, sorted(AUTHORS_RESULT_TXT)
            )
        )
    return dict(AUTHORS_RESULT_TXT[pred_len])


def best_baseline(pred_len: int = 96, metric: str = "mse") -> Tuple[str, float]:
    """The strongest non-TQNet model in Table 5 at this horizon.

    Used by the report to state the margin TQNet is claiming, so that our
    reproduction error can be compared against something meaningful rather than
    judged on whether it "looks close".
    """
    row = PAPER_TABLE_5[pred_len]
    position = 0 if metric.lower() == "mse" else 1
    candidates = [(name, values[position]) for name, values in row.items() if name != "TQNet"]
    return min(candidates, key=lambda item: item[1])


def seed_std(pred_len: int = 96, metric: str = "mse") -> Optional[float]:
    """The paper's own run-to-run standard deviation, the yardstick for "different"."""
    entry = SEED_STUDY.get(pred_len)
    if entry is None:
        return None
    return entry["std"][0 if metric.lower() == "mse" else 1]
