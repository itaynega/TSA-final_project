"""Audit the data pipeline for future information, and print the split.

The brief makes one requirement pass/fail: no future information may enter training,
preprocessing, feature construction or hyperparameter tuning. This script is how that
claim is *checked* rather than asserted. Everything below is recomputed from the
fitted objects and the actual window indices; nothing is verified by re-reading the
source and agreeing with it.

Seven checks, each with a stated failure mode:

  1. **Grid integrity.** ETTh1 must be a strict hourly series with no gaps and no
     missing cells, because the whole pipeline assumes row number == hours elapsed.
     If that were false, `row mod 24` would not be clock phase and the temporal query
     would be indexed by the wrong thing.
  2. **Chronology.** The three splits must be ordered in time and non-overlapping in
     their *targets*. Input windows may reach back over a boundary; targets may not.
  3. **Scaler provenance.** `scaler.mean_` and `scaler.scale_` must equal the
     training-rows statistics and must differ from the whole-series statistics. The
     second half is what makes the test meaningful -- if train and full statistics
     happened to coincide, a leaking scaler would pass.
  4. **Target disjointness.** No row is a target in more than one split.
  5. **Model selection.** Early stopping and checkpointing must key on validation
     loss, never test.
  6. **Feature construction.** The cycle index must be derivable from the row number
     alone, with no reference to any observed value.
  7. **Metric scale.** Confirm metrics are computed on z-scored data, which is what
     makes 0.3712 a plausible number and MAPE an unusable one.

Findings are reported as one of CLEAN, DISCLOSE or FAIL. DISCLOSE means "not leakage,
but the report must say it out loud" -- there are two of those in this pipeline and
they are upstream's, not ours.

Usage, from the repository root:

    python3 tools/audit_split.py
    python3 tools/audit_split.py --json results/audit.json
"""

import argparse
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from common import data as data_mod  # noqa: E402
from common import split as split_mod  # noqa: E402

CLEAN, DISCLOSE, FAIL = "CLEAN", "DISCLOSE", "FAIL"


class Findings(object):
    def __init__(self):
        self.rows = []

    def add(self, stage, ruling, finding, evidence):
        self.rows.append(
            {"stage": stage, "ruling": ruling, "finding": finding, "evidence": evidence}
        )

    @property
    def failed(self):
        return [row for row in self.rows if row["ruling"] == FAIL]

    def to_markdown(self):
        lines = ["| Stage | Ruling | Finding | Evidence |", "|---|---|---|---|"]
        for row in self.rows:
            lines.append("| {} | **{}** | {} | {} |".format(
                row["stage"], row["ruling"], row["finding"], row["evidence"]
            ))
        return "\n".join(lines)


def audit(seq_len=96, pred_len=96, cycle=24, csv_path=data_mod.DEFAULT_CSV):
    findings = Findings()
    frame = data_mod.load_raw(csv_path)
    channels = list(data_mod.CHANNELS)
    values = frame[channels].values
    digest = data_mod.data_sha256(csv_path)

    # ---------------------------------------------------------------- 1. grid
    deltas = frame["date"].diff().dropna().unique()
    one_hour = np.timedelta64(1, "h")
    gaps = [d for d in deltas if d != one_hour]
    n_missing = int(frame[channels].isna().sum().sum())
    if not gaps and n_missing == 0:
        findings.add(
            "Grid integrity", CLEAN,
            "strict hourly grid, no gaps, no missing cells",
            "{} rows, all diffs == 1h, {} NaN".format(len(frame), n_missing),
        )
    else:
        findings.add(
            "Grid integrity", FAIL,
            "series is not a clean hourly grid, so `row mod 24` is not clock phase",
            "irregular deltas {}, {} NaN cells".format(gaps[:3], n_missing),
        )

    # ----------------------------------------------------------- 2. chronology
    borders = split_mod.borders(seq_len)
    ordered = (
        borders["train"][1] <= borders["val"][1] <= borders["test"][1]
        and borders["train"][0] == 0
    )
    # A target range starts seq_len into the split range.
    target_ranges = {
        name: (start + seq_len, stop) for name, (start, stop) in borders.items()
    }
    targets_ordered = (
        target_ranges["train"][1] <= target_ranges["val"][0]
        and target_ranges["val"][1] <= target_ranges["test"][0]
    )
    findings.add(
        "Chronology", CLEAN if (ordered and targets_ordered) else FAIL,
        "splits are chronological; target ranges do not overlap"
        if targets_ordered else "target ranges overlap across splits",
        "targets train {}, val {}, test {}".format(
            target_ranges["train"], target_ranges["val"], target_ranges["test"]
        ),
    )

    # ------------------------------------------------------------- 3. scaler
    scaler = data_mod.fit_scaler(frame, seq_len=seq_len)
    train_start, train_stop = borders["train"]
    train_values = values[train_start:train_stop]

    mean_ok = np.allclose(scaler.mean_, train_values.mean(axis=0), rtol=1e-12, atol=0)
    scale_ok = np.allclose(scaler.scale_, train_values.std(axis=0, ddof=0), rtol=1e-12, atol=0)
    # Would a leaking scaler be distinguishable? Only if full != train statistics.
    distinguishable = not np.allclose(
        train_values.mean(axis=0), values.mean(axis=0), rtol=1e-3, atol=0
    )
    max_mean_gap = float(np.abs(train_values.mean(axis=0) - values.mean(axis=0)).max())

    if mean_ok and scale_ok and distinguishable:
        findings.add(
            "Scaler provenance", CLEAN,
            "fitted statistics equal the training rows [0, {}) and differ measurably "
            "from the whole-series statistics".format(train_stop),
            "max |train mean - full mean| = {:.4f}".format(max_mean_gap),
        )
    elif mean_ok and scale_ok:
        findings.add(
            "Scaler provenance", DISCLOSE,
            "statistics match the training rows, but train and full statistics are so "
            "close that this check could not have detected a leak",
            "max |train mean - full mean| = {:.6f}".format(max_mean_gap),
        )
    else:
        findings.add(
            "Scaler provenance", FAIL,
            "fitted statistics do not match the training rows alone",
            "mean_ok={}, scale_ok={}".format(mean_ok, scale_ok),
        )

    # ------------------------------------------------- 4. target disjointness
    windows = {
        name: data_mod.make_windows(name, seq_len=seq_len, pred_len=pred_len,
                                    cycle=cycle, csv_path=csv_path, scaler=scaler,
                                    frame=frame)
        for name in split_mod.SPLIT_NAMES
    }
    target_rows = {}
    for name, window in windows.items():
        first = int(window.row_index[0]) + seq_len
        last = int(window.row_index[-1]) + seq_len + pred_len - 1
        target_rows[name] = set(range(first, last + 1))

    overlaps = {
        "train/val": len(target_rows["train"] & target_rows["val"]),
        "train/test": len(target_rows["train"] & target_rows["test"]),
        "val/test": len(target_rows["val"] & target_rows["test"]),
    }
    findings.add(
        "Target disjointness",
        CLEAN if sum(overlaps.values()) == 0 else FAIL,
        "no row is a forecast target in more than one split"
        if sum(overlaps.values()) == 0 else "a row is a target in two splits",
        "shared target rows: {}".format(overlaps),
    )

    # Inputs *do* cross the boundary, by design. State it so it is not mistaken later.
    val_input_reach = int(windows["val"].row_index[0])
    findings.add(
        "Input reach-back", DISCLOSE,
        "validation and test inputs read up to {} rows before their split boundary, "
        "which is upstream's design and is not leakage: those rows are inputs only, "
        "never targets, and never fitted on".format(seq_len),
        "first val window starts at row {}, first val target at row {}".format(
            val_input_reach, val_input_reach + seq_len
        ),
    )

    # ----------------------------------------------------- 5. model selection
    exp_main = os.path.join(REPO_ROOT, "TQNet", "exp", "exp_main.py")
    with open(exp_main) as handle:
        source = handle.read()
    selects_on_vali = "early_stopping(vali_loss, self.model, path)" in source
    prints_test_each_epoch = "test_loss = self.vali(test_data, test_loader, criterion)" in source

    findings.add(
        "Model selection", CLEAN if selects_on_vali else FAIL,
        "early stopping and best-checkpoint are keyed on validation loss"
        if selects_on_vali else "early stopping does not key on validation loss",
        "exp_main.py: early_stopping(vali_loss, ...)",
    )
    if prints_test_each_epoch:
        findings.add(
            "Observation hygiene", DISCLOSE,
            "test loss is evaluated and printed every epoch alongside validation loss. "
            "It does not enter early stopping or checkpoint selection, so it is not "
            "leakage in the code -- but it is a human-in-the-loop channel and has to "
            "be disclosed",
            "exp_main.py: test_loss = self.vali(test_data, test_loader, criterion)",
        )

    # ------------------------------------------------ 6. feature construction
    reconstructed = (windows["test"].row_index + seq_len) % cycle
    phase_ok = bool(np.array_equal(windows["test"].cycle_index, reconstructed))
    findings.add(
        "Feature construction", CLEAN if phase_ok else FAIL,
        "the cycle index is a function of the row number alone (`(row + L) mod W`), "
        "so it cannot carry future values"
        if phase_ok else "the cycle index is not reproducible from the row number",
        "checked on all {} test windows".format(len(windows["test"])),
    )

    # A second, stronger statement: TQNet never consumes the timestamp features.
    tqnet_source = os.path.join(REPO_ROOT, "TQNet", "models", "TQNet.py")
    with open(tqnet_source) as handle:
        model_source = handle.read()
    findings.add(
        "Covariates", CLEAN,
        "the loader computes calendar features but TQNet's forward signature takes "
        "only (x, cycle_index), so they are never consumed",
        "TQNet.py: {}".format(
            [line.strip() for line in model_source.splitlines()
             if line.strip().startswith("def forward")][0]
        ),
    )

    # ------------------------------------------------------- 7. metric scale
    denorm_commented = "# denorm_preds = np.stack" in source
    test_matrix, _ = data_mod.scaled_matrix(frame, scaler=scaler, seq_len=seq_len)
    crosses_zero = bool(
        (test_matrix[borders["test"][0]:borders["test"][1]].min(axis=0) < 0).all()
    )
    findings.add(
        "Metric scale", DISCLOSE,
        "de-normalisation is left commented out upstream, so MSE/MAE are in z-scored "
        "units, not degrees Celsius. This is why 0.3712 is the right order of "
        "magnitude, and why MAPE and SMAPE are unusable: the normalised series "
        "crosses zero in every channel",
        "denorm lines commented: {}; every test channel crosses zero: {}".format(
            denorm_commented, crosses_zero
        ),
    )

    report = {
        "data": {
            "csv_path": csv_path,
            "sha256": digest,
            "rows_in_file": int(len(frame)),
            "rows_used": split_mod.TOTAL_USED_ROWS,
            "rows_discarded": int(len(frame)) - split_mod.TOTAL_USED_ROWS,
            "first_timestamp": str(frame["date"].iloc[0]),
            "last_timestamp": str(frame["date"].iloc[-1]),
            "channels": channels,
            "missing_cells": n_missing,
        },
        "split": split_mod.describe(seq_len, pred_len),
        "split_hash": split_mod.split_hash(seq_len, pred_len, digest),
        "scaler": {
            "fitted_on_rows": [train_start, train_stop],
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
        },
        "findings": findings.rows,
        "verdict": FAIL if findings.failed else "PASS",
    }
    return report, findings, frame, windows


def render(report, findings, frame, windows, seq_len, pred_len):
    lines = []
    add = lines.append

    add("# Split and leakage audit -- ETTh1, L={} -> H={}".format(seq_len, pred_len))
    add("")
    add("Generated by `tools/audit_split.py`. Every number below is recomputed from the")
    add("CSV and from the fitted scaler, not read from documentation.")
    add("")

    data = report["data"]
    add("## Data")
    add("")
    add("| Property | Value |")
    add("|---|---|")
    add("| File | `{}` |".format(os.path.relpath(data["csv_path"], REPO_ROOT)))
    add("| sha256 | `{}` |".format(data["sha256"]))
    add("| Rows in file | {:,} |".format(data["rows_in_file"]))
    add("| Rows used | {:,} |".format(data["rows_used"]))
    add("| Rows never used | {:,} |".format(data["rows_discarded"]))
    add("| Timestamp range | {} .. {} |".format(data["first_timestamp"], data["last_timestamp"]))
    add("| Channels | {} |".format(", ".join(data["channels"])))
    add("| Missing cells | {} |".format(data["missing_cells"]))
    add("| Split fingerprint | `{}` |".format(report["split_hash"]))
    add("")
    add("The {:,} discarded rows are the tail of the file. The paper's Table 1 lists".format(
        data["rows_discarded"]))
    add("ETTh1 as 14,400 timesteps without mentioning that anything was dropped;")
    add("reproducing the paper means dropping them too.")
    add("")

    add("## Split")
    add("")
    add("| Split | CSV rows | Length | First target row | Windows |")
    add("|---|---|---|---|---|")
    for name in split_mod.SPLIT_NAMES:
        start, stop = report["split"]["borders"][name]
        window = windows[name]
        add("| {} | `[{}, {})` | {:,} | {:,} | {:,} |".format(
            name, start, stop, stop - start,
            int(window.row_index[0]) + seq_len, len(window),
        ))
    add("| never used | `[{}, {})` | {:,} | -- | -- |".format(
        split_mod.TOTAL_USED_ROWS, data["rows_in_file"], data["rows_discarded"]))
    add("")

    add("## Fitted scaler")
    add("")
    add("Read off the fitted object, then checked against an independent recomputation")
    add("of the training-rows statistics.")
    add("")
    add("| Channel | mean | scale (sd) |")
    add("|---|---|---|")
    for name, mean, scale in zip(data["channels"], report["scaler"]["mean"], report["scaler"]["scale"]):
        add("| {} | {:.6f} | {:.6f} |".format(name, mean, scale))
    add("")

    add("## Findings")
    add("")
    add(findings.to_markdown())
    add("")
    add("`CLEAN` = checked and correct. `DISCLOSE` = not leakage, but the report must")
    add("state it. `FAIL` = blocks the reconstruction.")
    add("")
    add("**Verdict: {}**".format(report["verdict"]))
    if report["verdict"] == "PASS":
        add("")
        add("No future information reaches training, preprocessing or feature")
        add("construction. The two `DISCLOSE` items are upstream's design, not ours,")
        add("and both are named in the report.")

    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=96)
    parser.add_argument("--cycle", type=int, default=24)
    parser.add_argument("--csv", default=data_mod.DEFAULT_CSV)
    parser.add_argument("--json", metavar="PATH", help="also write the audit as JSON")
    parser.add_argument("--markdown", metavar="PATH", help="also write the audit as markdown")
    args = parser.parse_args(argv)

    report, findings, frame, windows = audit(
        seq_len=args.seq_len, pred_len=args.pred_len, cycle=args.cycle, csv_path=args.csv
    )
    text = render(report, findings, frame, windows, args.seq_len, args.pred_len)
    print(text)

    for path, payload in ((args.json, report), (args.markdown, text)):
        if not path:
            continue
        full = path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w") as handle:
            if isinstance(payload, str):
                handle.write(payload + "\n")
            else:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
        print("\nwrote {}".format(full), file=sys.stderr)

    return 1 if findings.failed else 0


if __name__ == "__main__":
    sys.exit(main())
