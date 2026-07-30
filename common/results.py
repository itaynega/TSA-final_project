"""Recording run results so that every number in the report is traceable.

The rule: a number that cannot be traced to a file does not get printed. This
module is how that rule is kept, and it is deliberately boring.

One run writes one JSON file under `results/runs/`. Not a shared CSV, for two
reasons. First, `.gitignore` excludes `*.csv` wholesale, so a CSV ledger would
never reach the other person's clone and neither of us would notice until the
report was being assembled. Second, one file per run means Itay's reconstruction
runs and my improvement runs never write to the same file, so the one-writer rule
holds without anyone having to think about it, and there is no merge conflict to
resolve at the deadline.

Reading them back with `load_runs()` is what makes the report's three-way table
(requirement F5: paper / reconstruction / improved) assemble mechanically rather
than by retyping numbers out of terminal scrollback.

Kept to the standard library and numpy, and to syntax Python 3.8 accepts, because
this has to import inside TQNet's pinned environment.
"""

import json
import os
import subprocess
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from common import metrics

__all__ = ["ARMS", "record_run", "load_runs", "assert_split_hash"]

# The columns requirement F5 fixes, plus the baseline requirement B6 asks for.
# "paper" is included because the paper's own numbers are transcribed rather than
# run, and they need the same provenance discipline as everything else.
ARMS = ("paper", "baseline", "reconstruction", "improved")

PathLike = Union[str, "os.PathLike"]


def assert_split_hash(expected: str, actual: str) -> None:
    """Fail loudly unless a run is using the reconstruction's exact split.

    Requirement C2 is pass/fail: the improved method must be evaluated on the
    same dataset split as the reconstruction. This turns that from a promise
    into a check, and it is intended to be called at the top of every run
    script rather than trusted to code review.

    An empty hash on both sides is treated as a failure, not a match: two runs
    that both forgot to record their split are not thereby using the same one.
    """
    if not expected or not actual:
        raise AssertionError(
            "split hash is missing (expected={!r}, actual={!r}); C2 cannot be "
            "checked without it".format(expected, actual)
        )
    if expected != actual:
        raise AssertionError(
            "split hash mismatch: expected {!r}, got {!r}. This run is not on "
            "the reconstruction's split, so its numbers are not comparable "
            "(requirement C2).".format(expected, actual)
        )


def _git_commit() -> Optional[str]:
    """Best effort. Provenance if git is available, None if it is not."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            timeout=5,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        if out.returncode == 0:
            return out.stdout.decode().strip()
    except Exception:
        pass
    return None


def record_run(
    results_dir: PathLike,
    arm: str,
    model: str,
    seed: Optional[int],
    seq_len: int,
    pred_len: int,
    split_hash: str,
    y_true: Union[np.ndarray, Sequence[float]],
    y_pred: Union[np.ndarray, Sequence[float]],
    notes: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute the metrics for one run and write them, with their provenance.

    Returns the record that was written, so a caller can log it without
    re-deriving anything.

    `arm` must be one of ARMS. A typo there would silently create a fifth column
    in a table that has four, which is why it is validated rather than accepted.
    """
    if arm not in ARMS:
        raise ValueError(
            "unknown arm {!r}; must be one of {}".format(arm, ", ".join(ARMS))
        )
    if not split_hash:
        raise ValueError(
            "split_hash is required — a run that cannot name its split cannot "
            "be compared against the reconstruction (requirement C2)"
        )

    true = np.asarray(y_true, dtype=np.float64)

    # Windows are the leading axis for long-horizon predictions shaped
    # (n_windows, pred_len, n_features). A flat 1-D array is one window.
    n_windows = int(true.shape[0]) if true.ndim > 1 else 1

    recorded_ns = time.time_ns()
    run_id = "{}-{}-s{}-h{}-{}".format(
        arm, model, "na" if seed is None else seed, pred_len, recorded_ns
    )

    record = {
        "run_id": run_id,
        "arm": arm,
        "model": model,
        "seed": seed,
        "seq_len": seq_len,
        "pred_len": pred_len,
        "split_hash": split_hash,
        "n_windows": n_windows,
        "n_points": int(true.size),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "recorded_ns": recorded_ns,
        "git_commit": _git_commit(),
        "metrics": metrics.all_metrics(y_true, y_pred),
        "notes": notes,
    }
    if extra:
        record["extra"] = extra

    runs_dir = os.path.join(str(results_dir), "runs")
    os.makedirs(runs_dir, exist_ok=True)
    path = os.path.join(runs_dir, run_id + ".json")
    with open(path, "w") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return record


def load_runs(results_dir: PathLike) -> List[Dict[str, Any]]:
    """Every recorded run, oldest first. Missing directory means no runs yet."""
    runs_dir = os.path.join(str(results_dir), "runs")
    if not os.path.isdir(runs_dir):
        return []

    loaded = []
    for name in sorted(os.listdir(runs_dir)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(runs_dir, name)) as handle:
            loaded.append(json.load(handle))

    return sorted(loaded, key=lambda record: record.get("recorded_ns", 0))
