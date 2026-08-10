"""Turn a finished TQNet run into a traceable run record, and cross-check it.

A training run leaves `pred.npy`, `true.npy` and `metrics.json` under
`TQNet/results/<setting>/`. That is raw output, not evidence: nothing in it states
which split produced it, which commit was checked out, or how it compares to the
paper. This script converts it into a record under `results/runs/`, which is what
`tools/make_report.py` reads.

The cross-check is the interesting part. Two independent metric implementations see
the same arrays:

  * TQNet's `utils/metrics.py`, which computes `pred - true` and whose value is
    already recorded in `metrics.json`;
  * our `common/metrics.py`, which computes `y - f(x)` in the course's notation.

Every metric here squares or takes a modulus, so the sign difference cancels and the
two must agree. They agree to about 1e-8 rather than exactly, and the reason is worth
knowing: the saved arrays are float32, so upstream's `np.mean` accumulates in float32
over 2,785 x 96 x 7 = 1.87M elements, while `common/metrics.py` casts to float64
first. Ours is the more accurate of the two. A float32 sum of that length carries a
relative error around 1e-7, which is what the tolerance below allows -- and which is
still three orders of magnitude below the third decimal the paper prints.

Anything larger than that is not rounding. It means one of the two is reducing over
the wrong axis or reading a mis-shaped array, so it is fatal rather than a warning:
this is exactly the class of bug that produces a plausible-looking MSE.

The parsed `setting` string also has to agree with the run's own `metrics.json`
about horizon and seed, which catches a stale directory being attributed to a new run.

Usage, from the repository root:

    python3 tools/collect_results.py                    # ingest every run found
    python3 tools/collect_results.py --arm improved     # label them as the improvement
    python3 tools/collect_results.py --dry-run
"""

import argparse
import json
import os
import re
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from common import data as data_mod  # noqa: E402
from common import metrics as metrics_mod  # noqa: E402
from common import results as results_mod  # noqa: E402
from common import split as split_mod  # noqa: E402

TQNET_RESULTS = os.path.join(REPO_ROOT, "TQNet", "results")
OUR_RESULTS = os.path.join(REPO_ROOT, "results")

# ETTh1_96_96_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed2024[_tq0ca1][_dphi0.8]
#
# The `_dphi` group matches the tag TQNet/run.py appends for Arm A. Without it a
# damped-trend run does not parse at all and its numbers cannot leave
# TQNet/results/; with it but without the `variant_label` branch below, it parses
# as "published" and `tools/make_report.py`'s variant filter would offer it as the
# reconstruction.
SETTING_RE = re.compile(
    r"^(?P<model_id>.+?)_(?P<model>[A-Za-z]+)_(?P<data>[A-Za-z0-9]+)"
    r"_ft(?P<features>[A-Z]+)_sl(?P<seq_len>\d+)_pl(?P<pred_len>\d+)"
    r"_cycle(?P<cycle>\d+)_seed(?P<seed>\d+)"
    r"(?:_tq(?P<use_tq>\d)ca(?P<channel_aggre>\d))?"
    r"(?:_dphi(?P<damped_phi>[0-9]+(?:\.[0-9]+)?))?$"
)

# Agreement tolerance between the two metric implementations, as a *relative* error.
# Sized for float32 accumulation over ~1.9M elements, which is what upstream does;
# see the module docstring. Deliberately not tighter, because a tighter bound would
# fail on correct arithmetic, and not looser, because 1e-6 is still far below the
# third decimal the paper reports.
AGREEMENT_RTOL = 1e-6


def parse_setting(setting):
    match = SETTING_RE.match(setting)
    if not match:
        return None
    fields = match.groupdict()
    phi = fields["damped_phi"]
    parsed = {
        "model_id": fields["model_id"],
        "model": fields["model"],
        "data": fields["data"],
        "features": fields["features"],
        "seq_len": int(fields["seq_len"]),
        "pred_len": int(fields["pred_len"]),
        "cycle": int(fields["cycle"]),
        "seed": int(fields["seed"]),
        "use_tq": int(fields["use_tq"]) if fields["use_tq"] is not None else 1,
        "channel_aggre": int(fields["channel_aggre"]) if fields["channel_aggre"] is not None else 1,
        "use_damped_trend": phi is not None,
        "damped_phi": float(phi) if phi is not None else None,
    }
    return parsed


def variant_label(parsed):
    """A short human name for the ablation variant this run represents.

    Arm A is checked first and reported with its phi. It changes the normalisation
    rather than the TQ/attention wiring, so it leaves `use_tq` and `channel_aggre`
    at 1 and would otherwise be indistinguishable from the published model here.
    """
    if parsed.get("use_damped_trend"):
        return "damped trend (phi={:g})".format(parsed["damped_phi"])
    if parsed["use_tq"] and parsed["channel_aggre"]:
        return "published"
    if not parsed["use_tq"] and parsed["channel_aggre"]:
        return "no-TQ (self-attention)"
    if not parsed["use_tq"] and not parsed["channel_aggre"]:
        return "pure MLP"
    return "TQ without channel attention"


def discover(root=TQNET_RESULTS):
    """Directories that contain a complete set of run outputs."""
    if not os.path.isdir(root):
        return []
    found = []
    for name in sorted(os.listdir(root)):
        directory = os.path.join(root, name)
        if not os.path.isdir(directory):
            continue
        if all(os.path.exists(os.path.join(directory, f))
               for f in ("pred.npy", "true.npy")):
            found.append(directory)
    return found


def ingest(directory, arm, results_dir=OUR_RESULTS, record=True, csv_path=data_mod.DEFAULT_CSV):
    setting = os.path.basename(directory.rstrip(os.sep))
    parsed = parse_setting(setting)
    if parsed is None:
        raise ValueError(
            "cannot parse run directory name {!r}. Expected the pattern TQNet's run.py "
            "builds, e.g. ETTh1_96_96_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed2024".format(setting)
        )

    preds = np.load(os.path.join(directory, "pred.npy"))
    trues = np.load(os.path.join(directory, "true.npy"))
    if preds.shape != trues.shape:
        raise ValueError(
            "pred.npy is {} but true.npy is {} in {}".format(preds.shape, trues.shape, setting)
        )

    summary_path = os.path.join(directory, "metrics.json")
    upstream = {}
    if os.path.exists(summary_path):
        with open(summary_path) as handle:
            upstream = json.load(handle)

        # A stale directory reused for a new run is a real hazard; the name and the
        # JSON must agree about what was run.
        for key in ("pred_len", "seed", "cycle"):
            if key in upstream and int(upstream[key]) != parsed[key]:
                raise ValueError(
                    "{}: directory name says {}={} but metrics.json says {}. The "
                    "directory is stale -- delete it and re-run.".format(
                        setting, key, parsed[key], upstream[key]
                    )
                )

    ours = metrics_mod.all_metrics(trues, preds)

    # Cross-check against upstream's own numbers on the same arrays.
    agreement = {}
    for name, key in (("MSE", "upstream_mse"), ("MAE", "upstream_mae"), ("RMSE", "upstream_rmse")):
        if key in upstream:
            theirs = float(upstream[key])
            delta = abs(ours[name] - theirs)
            rel = delta / abs(theirs) if theirs else delta
            agreement[name] = {
                "ours": ours[name],
                "upstream": theirs,
                "abs_diff": delta,
                "rel_diff": rel,
                "agree": rel <= AGREEMENT_RTOL,
            }
    disagreements = [name for name, entry in agreement.items() if not entry["agree"]]
    if disagreements:
        detail = "; ".join(
            "{}: ours {:.12f} vs upstream {:.12f} (relative {:.3e})".format(
                name, agreement[name]["ours"], agreement[name]["upstream"],
                agreement[name]["rel_diff"])
            for name in disagreements
        )
        raise ValueError(
            "{}: our metrics disagree with TQNet's own on the same arrays by more than "
            "float32 accumulation can explain -- {}.\nBoth implementations square or "
            "take a modulus of the error, so the sign convention cancels and they must "
            "agree. A gap this large means one of them is reducing over the wrong axis, "
            "and the number cannot be reported.".format(setting, detail)
        )

    digest = data_mod.data_sha256(csv_path)
    fingerprint = split_mod.split_hash(parsed["seq_len"], parsed["pred_len"], digest)

    expected_windows = split_mod.n_windows(parsed["seq_len"], parsed["pred_len"])["test"]
    if preds.shape[0] != expected_windows:
        raise ValueError(
            "{}: got {} windows but the split defines {} test windows at L={}, H={}. "
            "The evaluation did not cover the split this project pins.".format(
                setting, preds.shape[0], expected_windows,
                parsed["seq_len"], parsed["pred_len"])
        )

    written = None
    if record:
        written = results_mod.record_run(
            results_dir=results_dir,
            arm=arm,
            model=parsed["model"],
            seed=parsed["seed"],
            seq_len=parsed["seq_len"],
            pred_len=parsed["pred_len"],
            split_hash=fingerprint,
            y_true=trues,
            y_pred=preds,
            notes="Ingested from TQNet/results/{}. Variant: {}.".format(
                setting, variant_label(parsed)),
            extra={
                "setting": setting,
                "variant": variant_label(parsed),
                "use_tq": parsed["use_tq"],
                "channel_aggre": parsed["channel_aggre"],
                "use_damped_trend": parsed["use_damped_trend"],
                "damped_phi": parsed["damped_phi"],
                "cycle": parsed["cycle"],
                "features": parsed["features"],
                "data_sha256": digest,
                "n_params": upstream.get("n_params"),
                "accelerator": upstream.get("accelerator"),
                "upstream_metric_agreement": agreement,
            },
        )

    return {
        "setting": setting,
        "variant": variant_label(parsed),
        "parsed": parsed,
        "metrics": ours,
        "agreement": agreement,
        "n_windows": int(preds.shape[0]),
        "split_hash": fingerprint,
        "record": written,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=TQNET_RESULTS,
                        help="directory holding TQNet run output folders")
    parser.add_argument("--arm", default="reconstruction", choices=list(results_mod.ARMS),
                        help="which column of the report's table these runs belong to")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would be ingested without writing records")
    args = parser.parse_args(argv)

    directories = discover(args.root)
    if not directories:
        print("No completed runs found under {}".format(args.root))
        print("")
        print("A run writes pred.npy and true.npy there only if it finished the test")
        print("phase with --save_outputs 1. Start one with:")
        print("    bash repro/run_reconstruction.sh")
        return 1

    print("Found {} run(s) under {}".format(len(directories), args.root))
    print("")

    failures = 0
    for directory in directories:
        try:
            outcome = ingest(directory, arm=args.arm, record=not args.dry_run)
        except ValueError as exc:
            failures += 1
            print("FAILED {}\n  {}\n".format(os.path.basename(directory), exc))
            continue

        print("{}  [{}]".format(outcome["setting"], outcome["variant"]))
        print("  windows   : {:,}   split {}".format(outcome["n_windows"], outcome["split_hash"]))
        print("  " + "   ".join(
            "{} {:.6f}".format(name, outcome["metrics"][name])
            for name in metrics_mod.METRIC_NAMES
        ))
        if outcome["agreement"]:
            worst = max(entry["rel_diff"] for entry in outcome["agreement"].values())
            print("  agrees with TQNet's own float32 metrics to {:.1e} relative".format(worst))
        if outcome["record"]:
            print("  recorded  : {}".format(outcome["record"]["run_id"]))
        print("")

    if args.dry_run:
        print("dry run: nothing written")
    else:
        print("Run records are under {}/runs/".format(os.path.relpath(OUR_RESULTS, REPO_ROOT)))
        print("Next: python3 tools/make_report.py")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
