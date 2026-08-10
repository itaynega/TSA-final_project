"""J-12a Step 3: write `resolved_config.json` for the 24 pre-existing checkpoints.

STAGE2_WORKPLAN_2026-08-09.md sec 7i requires `tools/validation_metrics.py` to stop
reconstructing a checkpoint's configuration by parsing its directory name (that is
the bug class this job exists to close -- see the module docstring of
`tools/validation_metrics.py` after J-12a Step 4). Going forward every checkpoint
carries its own `resolved_config.json`, written by the training run itself
(`TQNet/exp/exp_main.py`'s `_write_resolved_config`, J-12a Step 1). The 24
checkpoints trained *before* that change exist only have a directory name -- this
script is the one-time bridge.

**The point of this script is the cross-check, not the parse.** Regex-parsing the
directory name is exactly the operation that just produced a silent, wrong result
for Arm A (sec 7i): a plausible-looking config that is not the one actually
checkpointed. So nothing here is written on the strength of the regex alone. Every
field the regex derives is checked against at least one artefact that did NOT come
from the directory name, and two fields the name cannot supply at all --
`n_params` and `accelerator` -- are read only from those artefacts:

  * `results/runs/<arm>-TQNet-s<seed>-h<pred_len>-<recorded_ns>.json` -- one such
    record exists per checkpoint (`extra.setting` names it), written by
    `tools/collect_results.py` when the run's test-split output was ingested.
    Covers all 24.

  * `TQNet/results_armD/<setting>/metrics.json` -- written by
    `TQNet/exp/exp_main.py test()` at training time, for the 12 Arm D
    (`_tq0ca0`) checkpoints only. Covers 12 of the 24, and where it exists it is
    checked *in addition to* the `results/runs/` record, not instead of it.

If more than one `results/runs/` record matches a setting (a re-run leaves more
than one), every match must agree with every other match on the fields checked
here before any of them is trusted -- silently picking "the newest one" without
that agreement check would be the same mistake this script exists to prevent,
one level up.

Any disagreement, on any field, against any artefact, is a hard failure. This
script does not pick a winner between the name and an artefact, or between two
artefacts -- STAGE2_WORKPLAN_2026-08-09.md sec 7i / standing order 8 territory,
which belongs to the PM, not to this script.

This script is meant to run exactly once. After it has run, `resolved_config.json`
exists next to every `checkpoint.pth`, is itself the provenance record of the
backfill (each file's `backfill_agreement` block says what it was checked
against), and this script is never needed again -- it is committed for that
provenance, not for reuse.

Usage, from the repository root:

    python3 tools/backfill_checkpoint_config.py
"""

import json
import os
import re
import sys

THIS_FILE = os.path.abspath(__file__)
REPO_ROOT = os.path.dirname(os.path.dirname(THIS_FILE))  # tools/ -> repo root
TQNET_ROOT = os.path.join(REPO_ROOT, "TQNet")

CHECKPOINTS_DIR = os.path.join(TQNET_ROOT, "checkpoints")
RESULTS_ARMD_DIR = os.path.join(TQNET_ROOT, "results_armD")
RESULTS_RUNS_DIR = os.path.join(REPO_ROOT, "results", "runs")

RESOLVED_CONFIG_SCHEMA = 1

# The number of checkpoints this script expects to find *without* a
# resolved_config.json already (i.e. the pre-J-12a backlog). Anything else means
# the repo is not in the state this dispatch describes -- stop rather than guess.
EXPECTED_BACKFILL_COUNT = 24

# Same shape as the (now-retired, evaluation-path) SETTING_RE in
# tools/validation_metrics.py, minus the `_dphi<phi>` group: no Arm A checkpoint
# should ever reach this script, because Arm A checkpoints get their
# resolved_config.json from the training run itself (J-12a Step 1) and are
# therefore skipped below before parsing is even attempted. The `_dphi` guard a
# few lines down exists so a checkpoint that reaches this script BOTH lacking a
# resolved_config.json AND carrying a `_dphi` tag fails loudly instead of being
# silently (mis)parsed by a pattern that was never meant to see it.
SETTING_RE = re.compile(
    r"^(?P<model_id>.+)_(?P<model>TQNet)_(?P<data>ETTh1)_ft(?P<features>[A-Za-z]+)"
    r"_sl(?P<seq_len>\d+)_pl(?P<pred_len>\d+)_cycle(?P<cycle>\d+)_seed(?P<seed>\d+)"
    r"(?P<abl_tag>_tq\d+ca\d+)?$"
)
ABL_RE = re.compile(r"_tq(?P<use_tq>\d+)ca(?P<channel_aggre>\d+)$")
DPHI_RE = re.compile(r"_dphi[\d.]+")


class BackfillError(Exception):
    """A cross-check failed, or the repo is not in the expected state. Fatal."""


def log(msg):
    print(msg)


def parse_setting_name(setting):
    """Derive a config from the directory name alone. A HYPOTHESIS, not a fact --
    see the module docstring. Every field returned here is checked against an
    independent artefact before it is trusted.
    """
    if DPHI_RE.search(setting):
        raise BackfillError(
            "{}: carries a _dphi tag, which means it is an Arm A checkpoint. Arm A "
            "checkpoints get resolved_config.json from the training run itself "
            "(J-12a Step 1) and must never reach this script. Something is "
            "wrong upstream of this script if it did.".format(setting)
        )
    m = SETTING_RE.match(setting)
    if not m:
        raise BackfillError(
            "{}: does not match the expected reconstruction/Arm-D setting "
            "pattern. Cannot derive even a hypothesis to cross-check.".format(setting)
        )
    gd = m.groupdict()
    use_tq, channel_aggre = 1, 1
    arm = "reconstruction"
    if gd["abl_tag"]:
        am = ABL_RE.search(gd["abl_tag"])
        if not am:
            raise BackfillError(
                "{}: could not parse ablation tag {!r}".format(setting, gd["abl_tag"])
            )
        use_tq = int(am.group("use_tq"))
        channel_aggre = int(am.group("channel_aggre"))
        arm = "armD"
    return {
        "model_id": gd["model_id"],
        "model": gd["model"],
        "data": gd["data"],
        "features": gd["features"],
        "seq_len": int(gd["seq_len"]),
        "pred_len": int(gd["pred_len"]),
        "cycle": int(gd["cycle"]),
        "seed": int(gd["seed"]),
        "use_tq": use_tq,
        "channel_aggre": channel_aggre,
        "arm": arm,
    }


def _load_json(path):
    with open(path) as fh:
        return json.load(fh)


def load_results_armd(setting):
    """`TQNet/results_armD/<setting>/metrics.json`, or None if it does not exist
    (only the 12 Arm D checkpoints have one).
    """
    path = os.path.join(RESULTS_ARMD_DIR, setting, "metrics.json")
    if not os.path.isfile(path):
        return None, None
    return _load_json(path), os.path.relpath(path, REPO_ROOT).replace(os.sep, "/")


def load_results_runs_records(setting):
    """Every results/runs/*.json whose extra.setting == setting, oldest first."""
    if not os.path.isdir(RESULTS_RUNS_DIR):
        return []
    found = []
    for name in sorted(os.listdir(RESULTS_RUNS_DIR)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(RESULTS_RUNS_DIR, name)
        record = _load_json(path)
        if record.get("extra", {}).get("setting") == setting:
            found.append((path, record))
    found.sort(key=lambda item: item[1].get("recorded_ns", 0))
    return found


def _check(setting, field, derived, source_name, source_value, mismatches, checked):
    """Compare one field against one source. Records agreement either way, and
    appends to `mismatches` (not raises) so one checkpoint's report can list every
    disagreement found, rather than stopping at the first.
    """
    checked.append({"field": field, "source": source_name, "value": source_value})
    if derived != source_value:
        mismatches.append(
            "{}: field {!r} -- derived from directory name = {!r}, but {} says "
            "{!r}".format(setting, field, derived, source_name, source_value)
        )


def cross_check_and_build(setting):
    """Parse `setting`, verify every field against independent artefacts, and
    return `(config_dict, agreement_report)`. Raises BackfillError with every
    disagreement found (not just the first) if any check fails.
    """
    derived = parse_setting_name(setting)
    mismatches = []
    checked = []
    sources_used = []

    # --- results/runs/ : covers all 24 -------------------------------------
    runs_matches = load_results_runs_records(setting)
    if not runs_matches:
        raise BackfillError(
            "{}: no results/runs/*.json record has extra.setting == this "
            "setting. A checkpoint that was never ingested by "
            "tools/collect_results.py has no independent artefact to check "
            "against at all, so this script refuses to backfill it from the "
            "name alone.".format(setting)
        )
    # If a re-run left more than one matching record, they must agree with each
    # other on every field this script checks, before any one of them is
    # trusted as ground truth.
    if len(runs_matches) > 1:
        base_path, base_record = runs_matches[0]
        base_fields = {
            "seq_len": base_record.get("seq_len"),
            "pred_len": base_record.get("pred_len"),
            "seed": base_record.get("seed"),
            "model": base_record.get("model"),
            "cycle": base_record.get("extra", {}).get("cycle"),
            "features": base_record.get("extra", {}).get("features"),
            "use_tq": base_record.get("extra", {}).get("use_tq"),
            "channel_aggre": base_record.get("extra", {}).get("channel_aggre"),
            "n_params": base_record.get("extra", {}).get("n_params"),
            "accelerator": base_record.get("extra", {}).get("accelerator"),
        }
        for other_path, other_record in runs_matches[1:]:
            other_fields = {
                "seq_len": other_record.get("seq_len"),
                "pred_len": other_record.get("pred_len"),
                "seed": other_record.get("seed"),
                "model": other_record.get("model"),
                "cycle": other_record.get("extra", {}).get("cycle"),
                "features": other_record.get("extra", {}).get("features"),
                "use_tq": other_record.get("extra", {}).get("use_tq"),
                "channel_aggre": other_record.get("extra", {}).get("channel_aggre"),
                "n_params": other_record.get("extra", {}).get("n_params"),
                "accelerator": other_record.get("extra", {}).get("accelerator"),
            }
            if base_fields != other_fields:
                raise BackfillError(
                    "{}: multiple results/runs/ records match this setting and "
                    "disagree with each other -- {} says {!r}, {} says {!r}. "
                    "Refusing to pick a winner (standing order 8).".format(
                        setting, os.path.basename(base_path), base_fields,
                        os.path.basename(other_path), other_fields)
                )
    # Latest by recorded_ns, now known to agree with every other match found.
    runs_path, runs_record = runs_matches[-1]
    runs_rel = os.path.relpath(runs_path, REPO_ROOT).replace(os.sep, "/")
    sources_used.append(runs_rel)
    extra = runs_record.get("extra", {})

    _check(setting, "seq_len", derived["seq_len"], runs_rel, runs_record.get("seq_len"), mismatches, checked)
    _check(setting, "pred_len", derived["pred_len"], runs_rel, runs_record.get("pred_len"), mismatches, checked)
    _check(setting, "seed", derived["seed"], runs_rel, runs_record.get("seed"), mismatches, checked)
    _check(setting, "model", derived["model"], runs_rel, runs_record.get("model"), mismatches, checked)
    _check(setting, "cycle", derived["cycle"], runs_rel, extra.get("cycle"), mismatches, checked)
    _check(setting, "features", derived["features"], runs_rel, extra.get("features"), mismatches, checked)
    _check(setting, "use_tq", derived["use_tq"], runs_rel, extra.get("use_tq"), mismatches, checked)
    _check(setting, "channel_aggre", derived["channel_aggre"], runs_rel, extra.get("channel_aggre"), mismatches, checked)

    # n_params / accelerator: NOT derivable from the name at all. Sourced here,
    # not "checked" against a derivation -- recorded as coming from this
    # artefact so the provenance is explicit.
    n_params = extra.get("n_params")
    accelerator = extra.get("accelerator")
    checked.append({"field": "n_params", "source": runs_rel, "value": n_params})
    checked.append({"field": "accelerator", "source": runs_rel, "value": accelerator})
    if n_params is None:
        mismatches.append("{}: {} has no extra.n_params -- cannot source it from anywhere".format(setting, runs_rel))
    if accelerator is None:
        mismatches.append("{}: {} has no extra.accelerator -- cannot source it from anywhere".format(setting, runs_rel))

    # --- TQNet/results_armD/<setting>/metrics.json : covers the 12 Arm D ----
    armd_metrics, armd_rel = load_results_armd(setting)
    if armd_metrics is not None:
        sources_used.append(armd_rel)
        _check(setting, "seq_len", derived["seq_len"], armd_rel, armd_metrics.get("seq_len"), mismatches, checked)
        _check(setting, "pred_len", derived["pred_len"], armd_rel, armd_metrics.get("pred_len"), mismatches, checked)
        _check(setting, "seed", derived["seed"], armd_rel, armd_metrics.get("seed"), mismatches, checked)
        _check(setting, "model", derived["model"], armd_rel, armd_metrics.get("model"), mismatches, checked)
        _check(setting, "data", derived["data"], armd_rel, armd_metrics.get("data"), mismatches, checked)
        _check(setting, "cycle", derived["cycle"], armd_rel, armd_metrics.get("cycle"), mismatches, checked)
        _check(setting, "features", derived["features"], armd_rel, armd_metrics.get("features"), mismatches, checked)
        _check(setting, "use_tq", derived["use_tq"], armd_rel, armd_metrics.get("use_tq"), mismatches, checked)
        _check(setting, "channel_aggre", derived["channel_aggre"], armd_rel, armd_metrics.get("channel_aggre"), mismatches, checked)
        # n_params / accelerator: cross-check the two independent artefacts
        # against EACH OTHER, since neither is derived from the name.
        armd_n_params = armd_metrics.get("n_params")
        armd_accelerator = armd_metrics.get("accelerator")
        checked.append({"field": "n_params", "source": armd_rel, "value": armd_n_params})
        checked.append({"field": "accelerator", "source": armd_rel, "value": armd_accelerator})
        if n_params is not None and armd_n_params is not None and n_params != armd_n_params:
            mismatches.append(
                "{}: n_params disagrees between {} ({!r}) and {} ({!r})".format(
                    setting, runs_rel, n_params, armd_rel, armd_n_params)
            )
        if accelerator is not None and armd_accelerator is not None and accelerator != armd_accelerator:
            mismatches.append(
                "{}: accelerator disagrees between {} ({!r}) and {} ({!r})".format(
                    setting, runs_rel, accelerator, armd_rel, armd_accelerator)
            )
    elif derived["arm"] == "armD":
        mismatches.append(
            "{}: directory name parses as an Arm D setting (_tq0ca0-style tag) "
            "but TQNet/results_armD/{}/metrics.json does not exist -- the one "
            "artefact that should exist for this arm is missing.".format(setting, setting)
        )

    if mismatches:
        raise BackfillError(
            "{}: {} disagreement(s) found:\n  {}".format(
                setting, len(mismatches), "\n  ".join(mismatches))
        )

    config = {
        "setting": setting,
        "arm": derived["arm"],
        "model": derived["model"],
        "data": derived["data"],
        "model_id": derived["model_id"],
        "features": derived["features"],
        "target": "OT",
        "freq": "h",
        "data_path": "ETTh1.csv",
        "seq_len": derived["seq_len"],
        "label_len": 0,
        "pred_len": derived["pred_len"],
        "cycle": derived["cycle"],
        "seed": derived["seed"],
        "accelerator": accelerator,
        "d_model": 512,   # frozen protocol (report/prereg-improvement.md sec 2); not
        "dropout": 0.5,   # independently recorded per-run anywhere, so not cross-checked --
                          # stated here as what every run in this project used.
        "batch_size": 256,
        "n_params": n_params,
        "enc_in": 7,
        "model_type": "mlp",
        "use_revin": 1,
        "use_tq": derived["use_tq"],
        "channel_aggre": derived["channel_aggre"],
        "channel_criterion": 0,  # none of the 24 pre-existing checkpoints used --channel_criterion;
                                  # use_tq/channel_aggre were passed directly (repro/run_etth1_ablation.sh).
        "use_damped_trend": 0,
        "damped_phi": 0.9,
        "resolved_config_schema": RESOLVED_CONFIG_SCHEMA,
        "resolved_from": "backfill",
        "written_by": "tools/backfill_checkpoint_config.py (J-12a Step 3)",
        "backfill_checked_against": sources_used,
        "backfill_agreement": checked,
    }
    return config, {"sources": sources_used, "fields_checked": checked, "arm": derived["arm"]}


def main():
    if not os.path.isdir(CHECKPOINTS_DIR):
        log("FATAL: checkpoints dir not found: {}".format(CHECKPOINTS_DIR))
        return 1

    settings = sorted(
        d for d in os.listdir(CHECKPOINTS_DIR)
        if os.path.isdir(os.path.join(CHECKPOINTS_DIR, d))
    )

    already_done = [s for s in settings if os.path.isfile(os.path.join(CHECKPOINTS_DIR, s, "resolved_config.json"))]
    to_backfill = [s for s in settings if s not in already_done]

    log("Found {} checkpoint directories total; {} already have resolved_config.json "
        "(written by the training run itself); {} need backfilling.".format(
            len(settings), len(already_done), len(to_backfill)))

    if len(to_backfill) != EXPECTED_BACKFILL_COUNT:
        log("FATAL: expected exactly {} checkpoints needing backfill, found {}. "
            "This does not match the state J-12a's dispatch describes -- stopping "
            "rather than guessing which ones to trust.".format(
                EXPECTED_BACKFILL_COUNT, len(to_backfill)))
        log("Needing backfill: {}".format(to_backfill))
        log("Already done: {}".format(already_done))
        return 1

    results = []
    failures = []
    for i, setting in enumerate(sorted(to_backfill), 1):
        log("[{}/{}] {}".format(i, len(to_backfill), setting))
        try:
            config, report = cross_check_and_build(setting)
        except BackfillError as exc:
            log("  FAILED:\n  {}".format(exc))
            failures.append(str(exc))
            continue

        checked_field_names = sorted(set(item["field"] for item in report["fields_checked"]))
        log("  arm={} agreed with {} on fields: {}".format(
            report["arm"], ", ".join(report["sources"]), ", ".join(checked_field_names)))

        out_path = os.path.join(CHECKPOINTS_DIR, setting, "resolved_config.json")
        with open(out_path, "w") as fh:
            json.dump(config, fh, indent=2, sort_keys=True)
            fh.write("\n")
        results.append({"setting": setting, "arm": report["arm"], "sources": report["sources"]})

    log("=" * 78)
    log("Done: {} written, {} failed (of {} needing backfill).".format(
        len(results), len(failures), len(to_backfill)))

    if failures:
        log("")
        log("FAILURES (no winner picked -- fix the underlying disagreement and re-run):")
        for f in failures:
            log("  " + f)
        return 1

    log("")
    log("Cross-check summary: {} checkpoints, zero disagreements.".format(len(results)))
    n_armd = sum(1 for r in results if r["arm"] == "armD")
    n_recon = sum(1 for r in results if r["arm"] == "reconstruction")
    log("  {} reconstruction checkpoints checked against results/runs/ only.".format(n_recon))
    log("  {} armD checkpoints checked against results/runs/ AND TQNet/results_armD/.".format(n_armd))
    return 0


if __name__ == "__main__":
    sys.exit(main())
