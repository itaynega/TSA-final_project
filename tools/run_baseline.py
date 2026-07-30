"""Score the seasonal-naive baseline on the reconstruction's exact test windows.

Requirement B6 of the brief asks for at least one simple baseline. A seasonal-naive
forecast at period 24 is the right one here, and not only because it is cheap: TQNet's
own `--cycle 24` encodes the same daily-periodicity assumption, so the gap between the
two is a direct measure of what the network contributes *beyond* knowing that ETTh1
repeats daily. A baseline that did not share that assumption would conflate the two.

Three things make this comparable to 0.3712 rather than merely adjacent to it. All
three are easy to get wrong, and each produces a plausible-looking wrong number:

  1. **Same scale.** The forecast is computed on z-scored data using the
     train-fitted scaler. A raw-scale baseline would be off by roughly the variance
     of each channel and comparable to nothing in the paper's table.
  2. **Same windows.** It reads `common.data.make_windows("test", ...)`, the same
     2,785 windows the network is evaluated on, rather than re-deriving its own.
  3. **Same metrics.** It calls `common.metrics.all_metrics`, the single
     implementation the whole project shares.

Note that `--period 1` degenerates to the plain persistence naive -- repeat the last
observed value across the whole horizon -- so both required styles of naive baseline
come from one code path.

Usage, from the repository root:

    python3 tools/run_baseline.py
    python3 tools/run_baseline.py --period 1 --no-record
"""

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from common import data as data_mod  # noqa: E402
from common import metrics as metrics_mod  # noqa: E402
from common import results as results_mod  # noqa: E402
from common import split as split_mod  # noqa: E402
from tools import paper_reference  # noqa: E402


def run(seq_len=96, pred_len=96, period=24, cycle=24, which="test",
        csv_path=data_mod.DEFAULT_CSV, record=True, results_dir=None):
    frame = data_mod.load_raw(csv_path)
    scaler = data_mod.fit_scaler(frame, seq_len=seq_len)
    windows = data_mod.make_windows(
        which, seq_len=seq_len, pred_len=pred_len, cycle=cycle,
        csv_path=csv_path, scaler=scaler, frame=frame,
    )

    predictions = data_mod.seasonal_naive(windows, period=period)
    scores = metrics_mod.all_metrics(windows.y, predictions)

    digest = data_mod.data_sha256(csv_path)
    split_fingerprint = split_mod.split_hash(seq_len, pred_len, digest)
    model_name = "naive" if period == 1 else "seasonal_naive_{}".format(period)

    record_written = None
    if record:
        record_written = results_mod.record_run(
            results_dir=results_dir or os.path.join(REPO_ROOT, "results"),
            arm="baseline",
            model=model_name,
            seed=None,  # deterministic: there is nothing to seed
            seq_len=seq_len,
            pred_len=pred_len,
            split_hash=split_fingerprint,
            y_true=windows.y,
            y_pred=predictions,
            notes=(
                "Seasonal-naive, period {}, on z-scored ETTh1 using the train-fitted "
                "scaler, over the identical {} {} windows as the reconstruction."
                .format(period, len(windows), which)
            ),
            extra={"period": period, "split": which, "data_sha256": digest},
        )

    return {
        "model": model_name,
        "split": which,
        "period": period,
        "n_windows": len(windows),
        "metrics": scores,
        "split_hash": split_fingerprint,
        "record": record_written,
    }


def render(outcome, pred_len):
    lines = []
    add = lines.append

    add("Baseline: {} on the {} split".format(outcome["model"], outcome["split"]))
    add("-" * 68)
    add("windows evaluated : {:,}".format(outcome["n_windows"]))
    add("split fingerprint : {}".format(outcome["split_hash"]))
    add("scale             : z-scored (train-fitted scaler)")
    add("")

    for name in metrics_mod.METRIC_NAMES:
        add("{:<6}: {:.6f}".format(name, outcome["metrics"][name]))
    add("")

    if outcome["split"] == "test":
        try:
            reference = paper_reference.target_reference(pred_len)
        except KeyError:
            reference = None
        if reference:
            ratio = outcome["metrics"]["MSE"] / reference["mse"]
            add("For context, TQNet's published MSE at this cell is {:.4f}.".format(reference["mse"]))
            add("The baseline is {:.2f}x that. A seasonal-naive forecast is not".format(ratio))
            add("supposed to be competitive; it is there to establish that the task is")
            add("non-trivial and that the model's number is doing real work.")

    if outcome["record"]:
        add("")
        add("recorded as {}".format(outcome["record"]["run_id"]))

    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=96)
    parser.add_argument("--period", type=int, default=24,
                        help="seasonal period; 1 degenerates to the persistence naive")
    parser.add_argument("--cycle", type=int, default=24)
    parser.add_argument("--split", default="test", choices=list(split_mod.SPLIT_NAMES))
    parser.add_argument("--csv", default=data_mod.DEFAULT_CSV)
    parser.add_argument("--no-record", action="store_true",
                        help="print the metrics without writing a run record")
    args = parser.parse_args(argv)

    outcome = run(
        seq_len=args.seq_len, pred_len=args.pred_len, period=args.period,
        cycle=args.cycle, which=args.split, csv_path=args.csv,
        record=not args.no_record,
    )
    print(render(outcome, args.pred_len))
    return 0


if __name__ == "__main__":
    sys.exit(main())
