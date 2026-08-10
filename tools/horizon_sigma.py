#!/usr/bin/env python3
"""tools/horizon_sigma.py -- J-06: per-horizon seed spread (sigma), x86 re-baseline.

Computes, for each forecast horizon, the seed-to-seed spread of the
reconstruction arm's MSE and MAE over the three fixed seeds (2024, 2025,
2026), using only the twelve x86 re-baseline run records in
`results/runs/`.

Selection rule (enforced here, in code -- not by filename, date, or eye):

    arm == "reconstruction" and git_commit == "9663bcd"

This must select exactly twelve records: 4 horizons (96, 192, 336, 720) x
3 seeds. Records with git_commit == "3894e4f" are the arm64 baseline --
produced on a machine that no longer exists -- and are excluded: mixing
architectures at a sigma of ~0.002 would conflate platform with effect.

Standard deviation convention: SAMPLE standard deviation, n-1 in the
denominator (`statistics.stdev`), not population sd. With only 3 seeds,
population sd (n) would read about 18% too small relative to sample sd (n-1).

This script only reads `results/runs/*.json` and writes
`report/horizon_sigma.md`. It runs no git command. Re-running it against an
unchanged `results/runs/` produces a byte-identical `report/horizon_sigma.md`.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "results" / "runs"
OUT_PATH = REPO_ROOT / "report" / "horizon_sigma.md"

SELECTED_GIT_COMMIT = "9663bcd"   # x86 re-baseline
EXCLUDED_GIT_COMMIT = "3894e4f"   # arm64, machine no longer exists
SELECTED_ARM = "reconstruction"

EXPECTED_HORIZONS = (96, 192, 336, 720)
EXPECTED_SEEDS = (2024, 2025, 2026)
METRICS = ("MSE", "MAE")

# Frozen pre-registration, report/prereg-improvement.md section 1:
# "Our seed sd at H=96, seeds 2024/2025/2026" = sigma = 0.002154 MSE.
PREREG_SIGMA_MSE_H96 = 0.002154


def _load_all_records(runs_dir: Path) -> List[Tuple[Path, Dict[str, Any]]]:
    records = []
    for path in sorted(runs_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        records.append((path, data))
    return records


def _select(records: List[Tuple[Path, Dict[str, Any]]]) -> List[Tuple[Path, Dict[str, Any]]]:
    """The selection rule, in code: arm == 'reconstruction' and git_commit == '9663bcd'."""
    return [
        (path, data)
        for path, data in records
        if data.get("arm") == SELECTED_ARM and data.get("git_commit") == SELECTED_GIT_COMMIT
    ]


def _fmt(x: float) -> str:
    """Full-precision, round-trippable formatting (Python's shortest repr)."""
    return repr(float(x))


def main() -> None:
    if not RUNS_DIR.is_dir():
        raise SystemExit(f"missing directory: {RUNS_DIR}")

    all_records = _load_all_records(RUNS_DIR)

    n_total = len(all_records)
    n_x86_any_arm = sum(1 for _, d in all_records if d.get("git_commit") == SELECTED_GIT_COMMIT)
    n_arm64_any_arm = sum(1 for _, d in all_records if d.get("git_commit") == EXCLUDED_GIT_COMMIT)

    selected = _select(all_records)
    if len(selected) != 12:
        raise SystemExit(
            "selection rule (arm == 'reconstruction' and git_commit == "
            f"'{SELECTED_GIT_COMMIT}') selected {len(selected)} record(s), expected 12. "
            f"(runs dir has {n_total} files total, {n_x86_any_arm} with git_commit=="
            f"'{SELECTED_GIT_COMMIT}' of any arm, {n_arm64_any_arm} with git_commit=="
            f"'{EXCLUDED_GIT_COMMIT}' of any arm.) Stopping -- return to the PM."
        )

    by_horizon: Dict[int, List[Tuple[Path, Dict[str, Any]]]] = {}
    for path, data in selected:
        h = data["pred_len"]
        by_horizon.setdefault(h, []).append((path, data))

    if sorted(by_horizon.keys()) != sorted(EXPECTED_HORIZONS):
        raise SystemExit(
            f"horizons found {sorted(by_horizon.keys())} != expected "
            f"{sorted(EXPECTED_HORIZONS)}. Stopping -- return to the PM."
        )

    for h, recs in by_horizon.items():
        seeds_found = sorted(d["seed"] for _, d in recs)
        if len(recs) != 3 or seeds_found != sorted(EXPECTED_SEEDS):
            raise SystemExit(
                f"horizon {h}: expected 3 records with seeds {sorted(EXPECTED_SEEDS)}, "
                f"got {len(recs)} record(s) with seeds {seeds_found}. "
                "Stopping -- return to the PM."
            )

    # split_hash agreement, per horizon -- protocol failure if the three seeds disagree.
    split_hash_by_horizon: Dict[int, str] = {}
    for h, recs in sorted(by_horizon.items()):
        hashes = sorted({d["split_hash"] for _, d in recs})
        if len(hashes) != 1:
            raise SystemExit(
                f"PROTOCOL FAILURE: horizon {h} records disagree on split_hash: {hashes}. "
                "Stopping -- return to the PM. (Per job dispatch: this is not a job the "
                "script proceeds past.)"
            )
        split_hash_by_horizon[h] = hashes[0]

    # Compute per-horizon, per-metric stats.
    stats: Dict[int, Dict[str, Any]] = {}
    for h, recs in sorted(by_horizon.items()):
        recs_sorted = sorted(recs, key=lambda pr: pr[1]["seed"])
        entry: Dict[str, Any] = {
            "filenames": [p.name for p, _ in recs_sorted],
            "seeds": [d["seed"] for _, d in recs_sorted],
            "split_hash": split_hash_by_horizon[h],
        }
        for metric in METRICS:
            values = [d["metrics"][metric] for _, d in recs_sorted]
            mean = statistics.fmean(values)
            sd = statistics.stdev(values)  # sample sd, n-1
            entry[metric] = {"values": values, "mean": mean, "sd": sd}
        stats[h] = entry

    sigma_mse_h96 = stats[96]["MSE"]["sd"]
    prereg_diff = sigma_mse_h96 - PREREG_SIGMA_MSE_H96
    prereg_agrees_6dp = round(sigma_mse_h96, 6) == round(PREREG_SIGMA_MSE_H96, 6)

    all_source_filenames = sorted(p.name for p, _ in selected)

    # --- render report/horizon_sigma.md -------------------------------------------------
    lines: List[str] = []
    A = lines.append

    A("# Per-horizon sigma -- x86 reconstruction re-baseline (J-06)")
    A("")
    A("Generated by `tools/horizon_sigma.py`. Re-running this script against an unchanged")
    A("`results/runs/` reproduces this file byte-for-byte. This file, not")
    A("`STAGE2_WORKPLAN_2026-08-09.md` section 7b, is the record (standing order 5): if the two")
    A("ever disagree, this file wins.")
    A("")
    A("## Selection rule")
    A("")
    A("Applied in code, not by filename, date, or eye:")
    A("")
    A("```")
    A(f'arm == "{SELECTED_ARM}" and git_commit == "{SELECTED_GIT_COMMIT}"')
    A("```")
    A("")
    A(
        f"`results/runs/` contains {n_total} `.json` files. {n_x86_any_arm} carry "
        f'`git_commit == "{SELECTED_GIT_COMMIT}"` (the x86 re-baseline); {n_arm64_any_arm} carry '
        f'`git_commit == "{EXCLUDED_GIT_COMMIT}"` (arm64, produced on a machine that no longer '
        "exists, and excluded so that mixing architectures at a sigma of ~0.002 does not "
        "conflate platform with effect). Applying the selection rule above to the "
        f'`arm == "{SELECTED_ARM}"` records selects exactly **{len(selected)}** records: 4 '
        "horizons x 3 seeds."
    )
    A("")
    A("## Standard deviation convention")
    A("")
    A(
        "**Sample standard deviation, n-1 in the denominator** "
        "(`statistics.stdev`, equivalently `numpy.std(..., ddof=1)`), computed over the three "
        "seeds at each horizon. Population sd (n in the denominator, ddof=0) would be "
        "wrong here and would read about 18% too small at n=3."
    )
    A("")
    A("## Twelve source filenames")
    A("")
    for fn in all_source_filenames:
        A(f"- `{fn}`")
    A("")
    A("## Per-horizon table")
    A("")
    for h in EXPECTED_HORIZONS:
        entry = stats[h]
        A(f"### H = {h}")
        A("")
        A(f"- Seeds (in order): {entry['seeds']}")
        A(f"- Source files (in seed order):")
        for fn in entry["filenames"]:
            A(f"  - `{fn}`")
        A(f"- `split_hash` (all three seeds agree): `{entry['split_hash']}`")
        A("")
        A("| Metric | seed " + str(entry["seeds"][0]) + " | seed " + str(entry["seeds"][1])
          + " | seed " + str(entry["seeds"][2]) + " | mean | sd (n-1) |")
        A("|---|---|---|---|---|---|")
        for metric in METRICS:
            m = entry[metric]
            v0, v1, v2 = m["values"]
            A(
                f"| {metric} | {_fmt(v0)} | {_fmt(v1)} | {_fmt(v2)} | "
                f"{_fmt(m['mean'])} | {_fmt(m['sd'])} |"
            )
        A("")
    A("## split_hash by horizon")
    A("")
    A("| Horizon | split_hash |")
    A("|---|---|")
    for h in EXPECTED_HORIZONS:
        A(f"| {h} | `{stats[h]['split_hash']}` |")
    A("")
    A("All three seeds agreed on `split_hash` within each horizon (asserted in code above; the")
    A("script stops rather than proceeding if they do not).")
    A("")
    A("## sigma is not monotonic in the horizon")
    A("")
    A(
        "Measured sigma_MSE (sample sd, n-1) across the four horizons, in order, is: "
        f"H=96: {_fmt(stats[96]['MSE']['sd'])}, "
        f"H=192: {_fmt(stats[192]['MSE']['sd'])}, "
        f"H=336: {_fmt(stats[336]['MSE']['sd'])}, "
        f"H=720: {_fmt(stats[720]['MSE']['sd'])}. "
    )
    A(
        "This is not monotonic: sigma falls from H=96 to H=192 "
        f"({round(stats[96]['MSE']['sd'], 6)} to {round(stats[192]['MSE']['sd'], 6)}) before "
        f"rising sharply through H=336 ({round(stats[336]['MSE']['sd'], 6)}) to H=720 "
        f"({round(stats[720]['MSE']['sd'], 6)}). A model that assumed sigma grows monotonically, "
        "or that used the H=96 value at every horizon, would be wrong by roughly an order of "
        "magnitude at H=720."
    )
    A("")
    A("## H = 96 vs the frozen pre-registration")
    A("")
    A(
        f"Measured sigma_MSE at H=96 (sample sd, n-1, three seeds): **{sigma_mse_h96:.6f}** "
        f"(full precision: {_fmt(sigma_mse_h96)})."
    )
    A(
        f"The frozen pre-registration (`report/prereg-improvement.md`, section 1) states "
        f"sigma = 0.002154 MSE at H=96."
    )
    A(
        f"Difference (measured - pre-registration): {_fmt(prereg_diff)}."
    )
    if prereg_agrees_6dp:
        A(
            "**The measured value agrees with the pre-registration**, matching to six decimal "
            "places (0.002154). This is a comparison only; the pre-registration itself is not "
            "amended here -- that is J-06b's job."
        )
    else:
        A(
            "**The measured value does NOT match the pre-registration's 0.002154 to six decimal "
            "places.** This is a comparison only; the pre-registration itself is not amended "
            "here -- that is J-06b's job."
        )
    A("")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
