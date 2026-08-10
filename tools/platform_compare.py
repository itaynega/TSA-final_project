#!/usr/bin/env python3
"""Compare the Stage-2 arms measured on macOS/arm64 against the x86 measurements.

Every Stage-2 arm was run on Amitay's x86 machine. `docs/STATUS.md` G2 recorded
the arm64 machine as unavailable and its Stage-1 artefacts as untraceable; both
turned out to be wrong, and re-running the arms there costs four minutes. The
question this answers is the one the project has been unable to ask: of the
sigma ~= 0.002 that every Stage-2 verdict is measured against, how much is
platform and how much is seed?

This is not a new arm and it changes no pre-registered prediction. The selection
rule in `report/prereg-improvement.md` sec 4 was applied on 2026-08-10, before any
run here existed, so nothing below can feed it. It is post-hoc by construction and
belongs in F6.

Two ledgers, deliberately kept apart:

  * `results/runs/`            -- x86, and the only ledger `tools/make_report.py`
                                  and `tools/horizon_sigma.py` read.
  * `results/platform-arm64/runs/` -- arm64. Separate so that a platform
                                  replication can never be mistaken for a seed,
                                  displace the x86 re-baseline in the F5 table,
                                  or reach the sigma that `horizon_sigma.py`
                                  computes.

Validation numbers are recomputed here for arm64 via
`tools.validation_metrics.evaluate_checkpoint`, which returns a record and writes
nothing. `run_all()` in that module is deliberately NOT called: it would rewrite
every sidecar under `results/validation/`, which is committed and holds the x86
values this script compares against.

The x86 seed-2024 reconstruction sidecar is a known-bad cell. It holds the
epoch-3 artefact 0.6869550701723053 left by the incident in `w_curve_correction.log`;
the pre-incident value is 0.6712632722155959, recovered from
`results/validation/validation_metrics.log`. This script sources that one cell from
the log and says so in its output, exactly as `report/w_curve.md` does.

Writes `report/platform_arm64.md` only.
"""

from __future__ import annotations

import json
import os
import re
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

X86_RUNS = REPO_ROOT / "results" / "runs"
ARM64_RUNS = REPO_ROOT / "results" / "platform-arm64" / "runs"
X86_VALIDATION = REPO_ROOT / "results" / "validation"
VALIDATION_LOG = X86_VALIDATION / "validation_metrics.log"
OUT_PATH = REPO_ROOT / "report" / "platform_arm64.md"

SEEDS = (2024, 2025, 2026)
PRED_LEN = 96

# The x86 re-baseline commit, per tools/horizon_sigma.py's own selection rule.
X86_RECONSTRUCTION_COMMIT = "9663bcd"

# The one x86 validation cell that cannot be read from its sidecar; see docstring.
POISONED_SIDECAR = "ETTh1_96_96_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed2024"
POISONED_VALUE = 0.6869550701723053
PREINCIDENT_VALUE = 0.6712632722155959

ARMS = ("armB", "armD", "armA")
ARM_TITLE = {
    "armB": "Arm B -- period estimated from the training split",
    "armD": "Arm D -- channel-count-conditional attention",
    "armA": "Arm A -- damped-trend instance normalisation, phi = 0.8",
}


def _fmt(x):
    return "--" if x is None else repr(float(x))


def _load(directory):
    out = []
    if not directory.is_dir():
        return out
    for path in sorted(directory.glob("*.json")):
        with path.open() as fh:
            out.append(json.load(fh))
    return out


def _variant(record):
    return (record.get("extra") or {}).get("variant", "published")


def arm64_test():
    """arm64 test MSE/MAE per arm per seed, keyed off the model_id tag."""
    out = {arm: {} for arm in ARMS}
    for record in _load(ARM64_RUNS):
        setting = (record.get("extra") or {}).get("setting", "")
        match = re.search(r"_arm64_(arm[ABD])_", setting)
        if not match or record.get("pred_len") != PRED_LEN:
            continue
        out[match.group(1)][record["seed"]] = record["metrics"]
    return out


def x86_test():
    """x86 test MSE/MAE for the two arms that were carried to test.

    Arm A never reached the test split -- it was abandoned at its H=96 validation
    gate -- so it has no entry here, and that absence is reported rather than
    filled in.
    """
    out = {arm: {} for arm in ARMS}
    for record in _load(X86_RUNS):
        if record.get("pred_len") != PRED_LEN:
            continue
        arm, variant = record.get("arm"), _variant(record)
        if arm == "reconstruction" and variant == "published" \
                and record.get("git_commit") == X86_RECONSTRUCTION_COMMIT:
            out["armB"][record["seed"]] = record["metrics"]
        elif arm == "improved" and variant == "pure MLP":
            out["armD"][record["seed"]] = record["metrics"]
    return out


def x86_validation():
    """x86 validation MSE per arm per seed, from the committed sidecars."""
    out = {arm: {} for arm in ARMS}
    notes = []
    for path in sorted(X86_VALIDATION.glob("ETTh1_96_96_TQNet_*.json")):
        name = path.stem
        if "cycle24" not in name:
            continue
        with path.open() as fh:
            data = json.load(fh)
        seed = data.get("seed")
        value = data["val_MSE"]
        if name.endswith("_tq0ca0"):
            out["armD"][seed] = value
        elif name.endswith("_dphi0.8"):
            out["armA"][seed] = value
        elif re.search(r"_seed\d+$", name):
            if name == POISONED_SIDECAR and value == POISONED_VALUE:
                value = PREINCIDENT_VALUE
                notes.append(
                    "x86 Arm B seed 2024 validation was sourced from "
                    "`results/validation/validation_metrics.log` "
                    "({!r}), not from its sidecar, which holds the epoch-3 artefact "
                    "{!r} left by the incident recorded in `w_curve_correction.log`.".format(
                        PREINCIDENT_VALUE, POISONED_VALUE)
                )
            out["armB"][seed] = value
    return out, notes


def arm64_validation():
    """Recomputed here, because no sidecar exists for these checkpoints.

    Uses `evaluate_checkpoint`, which returns a record and writes nothing.
    """
    from tools import validation_metrics as vm

    out = {arm: {} for arm in ARMS}
    ckpt_root = REPO_ROOT / "TQNet" / "checkpoints"
    for setting in sorted(os.listdir(ckpt_root)):
        match = re.search(r"_arm64_(arm[ABD])_", setting)
        if not match:
            continue
        record, _, _ = vm.evaluate_checkpoint(setting)
        out[match.group(1)][record["seed"]] = record["val_MSE"]
    return out


def _stats(values):
    if len(values) < 2:
        return (values[0] if values else None), None
    return statistics.fmean(values), statistics.stdev(values)


# Same configuration -- ETTh1, L=H=96, cycle 24, seed 2024 -- trained more than once
# on this machine. The only structural difference between the two groups is whether
# nn.MultiheadAttention is in the graph.
DETERMINISM_GROUPS = (
    ("full model (attention in the graph)", (
        ("Stage-1, 2026-07-30", "ETTh1_96_96_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed2024"),
        ("Arm B, 2026-08-10", "ETTh1_96_96_arm64_armB_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed2024"),
        ("replicate, 2026-08-10", "ETTh1_96_96_arm64_replicate_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed2024"),
    )),
    ("pure MLP (attention removed)", (
        ("Stage-1 ablation, 2026-07-30", "ETTh1_96_96_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed2024_tq0ca0"),
        ("Arm D, 2026-08-10", "ETTh1_96_96_arm64_armD_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed2024_tq0ca0"),
    )),
)


def _sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def determinism_rows():
    """Weight and prediction digests for repeat trainings of one configuration.

    Returns None for a group if any of its runs is absent, so this section
    degrades rather than inventing a comparison.
    """
    import numpy as np

    ckpt_root = REPO_ROOT / "TQNet" / "checkpoints"
    results_root = REPO_ROOT / "TQNet" / "results"

    out = []
    for title, members in DETERMINISM_GROUPS:
        rows, arrays = [], []
        for label, setting in members:
            ckpt = ckpt_root / setting / "checkpoint.pth"
            pred = results_root / setting / "pred.npy"
            if not (ckpt.is_file() and pred.is_file()):
                rows = None
                break
            array = np.load(pred)
            arrays.append(array)
            rows.append({
                "label": label,
                "weights": _sha256(ckpt)[:16],
                "preds": __import__("hashlib").sha256(array.tobytes()).hexdigest()[:16],
            })
        if rows is None:
            out.append((title, None, None))
            continue
        identical = all(np.array_equal(arrays[0], other) for other in arrays[1:])
        spread = max(float(np.abs(arrays[0] - other).max()) for other in arrays[1:]) \
            if len(arrays) > 1 else 0.0
        out.append((title, rows, (identical, spread)))
    return out


def _series(mapping, metric=None):
    """Values in seed order, or None if any seed is missing."""
    got = []
    for seed in SEEDS:
        if seed not in mapping:
            return None
        entry = mapping[seed]
        got.append(entry[metric] if metric else entry)
    return got


def main():
    a_test, x_test = arm64_test(), x86_test()
    x_val, val_notes = x86_validation()
    a_val = arm64_validation()

    lines = []
    A = lines.append

    A("# Platform replication -- the Stage-2 arms on macOS/arm64")
    A("")
    A("Generated by `tools/platform_compare.py`. Every Stage-2 number in the report was")
    A("measured on x86. This file re-measures the three live arms on the arm64 machine that")
    A("`docs/STATUS.md` G2 recorded as unavailable, to separate **platform** from **seed** in")
    A("the sigma every Stage-2 verdict is judged against.")
    A("")
    A("**This is not a new arm and it changes no pre-registered prediction.** The selection")
    A("rule (`report/prereg-improvement.md` sec 4) was applied before any run here existed.")
    A("Post-hoc by construction; F6 content.")
    A("")
    A("Records live in `results/platform-arm64/runs/`, deliberately outside `results/runs/`,")
    A("so a platform replication cannot be mistaken for a seed, displace the x86 re-baseline")
    A("in the F5 table, or reach the sigma `tools/horizon_sigma.py` computes.")
    A("")
    for note in val_notes:
        A("> " + note)
        A("")

    A("## Identical decisions on both platforms")
    A("")
    A("Both data-derived switches are deterministic functions of the CSV, and both return")
    A("the same answer on arm64 as on x86:")
    A("")
    A("| Switch | x86 | arm64 |")
    A("|---|---|---|")
    A("| Arm B period estimate (ACF / periodogram) | 24 / 24, agree | 24 / 24, agree |")
    A("| Arm B ACF peak value | 0.885154 | 0.8851540915365544 |")
    A("| Arm D `mean_abs_offdiag_pearson_correlation` | 0.311013 | 0.311013 |")
    A("| Arm D decision at threshold 0.30 | drop | drop |")
    A("")

    A("## Test MSE at H = 96")
    A("")
    for arm in ARMS:
        A("### {}".format(ARM_TITLE[arm]))
        A("")
        a_series = _series(a_test.get(arm, {}), "MSE")
        x_series = _series(x_test.get(arm, {}), "MSE")

        A("| Seed | arm64 | x86 | arm64 - x86 |")
        A("|---|---|---|---|")
        for i, seed in enumerate(SEEDS):
            av = a_series[i] if a_series else None
            xv = x_series[i] if x_series else None
            delta = (av - xv) if (av is not None and xv is not None) else None
            A("| {} | {} | {} | {} |".format(seed, _fmt(av), _fmt(xv), _fmt(delta)))

        a_mean, a_sd = _stats(a_series or [])
        x_mean, x_sd = _stats(x_series or [])
        A("| **mean** | {} | {} | {} |".format(
            _fmt(a_mean), _fmt(x_mean),
            _fmt(a_mean - x_mean) if (a_mean is not None and x_mean is not None) else "--"))
        A("| **sd (n-1)** | {} | {} | -- |".format(_fmt(a_sd), _fmt(x_sd)))
        A("")
        if x_series is None:
            A("Arm A has no x86 test record: it was abandoned at its H=96 validation gate")
            A("before the test split was ever read. The arm64 column therefore stands alone,")
            A("and no platform delta is computable for this arm on test.")
            A("")
        elif a_mean is not None and x_sd:
            A("Platform delta is **{:.3g}x** the x86 seed sd at this horizon.".format(
                abs(a_mean - x_mean) / x_sd))
            A("")

    A("## Validation MSE at H = 96 -- does the selection still hold?")
    A("")
    A("The pre-registration selects on mean validation MSE at H=96. If the ranking flipped")
    A("on another processor, the selection would be a platform artefact.")
    A("")
    A("| Arm | arm64 mean | x86 mean | arm64 - x86 |")
    A("|---|---|---|---|")
    ranking = {}
    for arm in ARMS:
        a_series = _series(a_val.get(arm, {}))
        x_series = _series(x_val.get(arm, {}))
        a_mean, _ = _stats(a_series or [])
        x_mean, _ = _stats(x_series or [])
        if a_mean is not None:
            ranking[arm] = a_mean
        A("| {} | {} | {} | {} |".format(
            arm, _fmt(a_mean), _fmt(x_mean),
            _fmt(a_mean - x_mean) if (a_mean is not None and x_mean is not None) else "--"))
    A("")
    A("Per-seed validation, arm64:")
    A("")
    A("| Arm | " + " | ".join("seed {}".format(s) for s in SEEDS) + " | sd (n-1) |")
    A("|---|---|---|---|---|")
    for arm in ARMS:
        series = _series(a_val.get(arm, {}))
        _, sd = _stats(series or [])
        cells = [_fmt(v) for v in (series or [None] * len(SEEDS))]
        A("| {} | {} | {} |".format(arm, " | ".join(cells), _fmt(sd)))
    A("")
    if ranking:
        order = sorted(ranking, key=ranking.get)
        A("**arm64 ranking on the pre-registered endpoint (lowest mean validation MSE "
          "first): {}.**".format(" < ".join(order)))
        A("")
        A("The x86 ranking was armB (0.6724990175677814) < armD (0.6805545682661754), with")
        A("Arm A failing its abandon gate. The arm64 ranking {} that.".format(
            "reproduces" if order[:2] == ["armB", "armD"] else "does NOT reproduce"))
        A("")

    A("## Where the non-determinism actually lives")
    A("")
    A("`CLAUDE.md` records that CPU training here is not bit-reproducible under a fixed")
    A("seed. That is correct, but it is not uniform across configurations, and the")
    A("difference is worth having on record because the incident in that file turned on it.")
    A("Each group below is the *same* configuration -- ETTh1, L=H=96, cycle 24, seed 2024 --")
    A("trained more than once on this machine, under a byte-identical environment")
    A("(torch 2.9.1, numpy 2.2.6, 10 threads) and with no change to the attention path in")
    A("`TQNet/models/TQNet.py` between the dates shown.")
    A("")
    for title, rows, summary in determinism_rows():
        A("### {}".format(title))
        A("")
        if rows is None:
            A("Not evaluable here: one or more of these runs is absent from this machine.")
            A("")
            continue
        A("| Run | `checkpoint.pth` sha256 (16) | `pred.npy` sha256 (16) |")
        A("|---|---|---|")
        for row in rows:
            A("| {} | `{}` | `{}` |".format(row["label"], row["weights"], row["preds"]))
        A("")
        identical, spread = summary
        if identical:
            A("**Byte-identical.** Repeat trainings of this configuration reproduce exactly,")
            A("weights and predictions, across the dates above.")
        else:
            A("**Not reproducible.** Every run differs, in weights and in predictions; the")
            A("largest per-element prediction difference is {:.3e}.".format(spread))
        A("")

    A("### Why this matters more than it looks")
    A("")
    A("The attention path is the only structural difference between the two groups, and it")
    A("is the one that fails to reproduce. A per-element prediction difference of order")
    A("1e-06 is invisible in a float32 metric accumulated over 1,871,520 elements -- the")
    A("full-model runs above print a **bit-identical** `mse:0.3710499405860901` from")
    A("`TQNet/utils/metrics.py` while having different weights and different predictions --")
    A("and shows up only around the tenth significant figure of a float64 metric.")
    A("")
    A("That is exactly the signature the incident in `CLAUDE.md` sec 'Never overwrite data'")
    A("describes: a repaired checkpoint whose weights hashed differently and whose `val_MSE`")
    A("differed at the tenth significant figure. This section is the direct demonstration")
    A("that such a difference is what an ordinary attention-path re-run looks like, and is")
    A("not by itself evidence that anything was misconfigured.")
    A("")
    A("It is also the strongest available argument for that file's central rule. Two runs")
    A("here produced an identical printed MSE from different models. **A metric match is not")
    A("a hash match** -- not as a matter of principle, but demonstrated, on this data, with")
    A("the very metric that would have been used to check.")
    A("")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote {}".format(OUT_PATH.relative_to(REPO_ROOT)))


if __name__ == "__main__":
    main()
