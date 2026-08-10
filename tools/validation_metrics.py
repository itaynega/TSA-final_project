"""J-10b: validation-split MSE/MAE for every TQNet checkpoint, without retraining.

Why this exists: the pre-registration (report/prereg-improvement.md sec 4, items 1-2)
selects the winning Stage-2 arm on *validation* MSE at H=96. No run record under
results/runs/ contains a validation metric -- every one of the 26 existing records
is a test-split metric (n_windows=2785 at H=96 is the *test* count). This script is
the missing instrument: it rebuilds each of the 24 checkpointed models from the
config encoded in its own setting string, loads the saved weights, evaluates them
on the validation split (not test), and writes the result as one sidecar JSON per
checkpoint under results/validation/.

Design decisions, stated here because each one is a way to get a number that looks
right and is not what was asked for:

1. Metrics come from common/metrics.py (flat MSE/MAE over every element of the
   (n_windows, pred_len, n_channels) array), not from TQNet's nn.MSELoss and not
   hand-rolled -- so validation and test numbers are produced by the same code
   (dispatch requirement).

2. No de-normalisation. TQNet/exp/exp_main.py's test() computes
   `metric(preds, trues)` on the model's raw (z-scored) output; the de-norm block
   a few lines above it is commented out and has been since before this job. vali()
   never touches the scale at all -- it only ever sees what the DataLoader hands it,
   which is the same z-scored tensors as train and test (common/data.py's own
   Windows docstring is explicit: "the scaler is fitted on rows [0, 8640) ... and
   then applied to all 17,420 rows before any slicing"). So both splits live on the
   same scale, and this script does not call `.inverse_transform` either.

3. The validation DataLoader used here is built with shuffle=False, drop_last=False
   -- deliberately *not* the loader TQNet/data_provider/data_factory.py hands to
   vali() during training (which sets shuffle=True, drop_last=True for the 'val'
   flag, because during training only a scalar running loss is needed and a
   reproducible, complete evaluation is not the goal). A validation metric that
   feeds an arm-selection decision needs to be complete and deterministic, so this
   script evaluates every window in the validation split, in row order, exactly as
   TQNet/exp/exp_main.py's test() does for the test split. This is a considered
   deviation from vali()'s own loader settings, not an oversight; it is called out
   again in the run log and in this job's return to the PM.

Consequence of (3), worth stating before it looks like a bug: because ETTh1's
val and test blocks are both exactly 4 calendar months long (common/split.py's own
MONTHS = {"train": 12, "val": 4, "test": 4}), the two spans have identical *row
count* regardless of pred_len, and therefore identical *window count* too --
2,785 windows each at H=96, evaluated in full here. The window counts are equal by
construction; the rows underneath them are completely disjoint
([8544,11520) vs [11424,14400)). Equal counts are therefore not evidence of a
test-split fallback. The sanity anchor (val MSE at H=96 seed 2024 reconstruction
must differ from the known test MSE 0.37104994668966473) is the check that actually
catches that failure mode, and this script asserts it.

Nothing here retrains, and nothing here writes under TQNet/results/ or
TQNet/test_results/ -- the per-checkpoint evaluation loop is a trimmed copy of
exp_main.py's test() (same model call, same output slicing) with the
folder-creation / npy-saving / visual() side effects removed, because those are
what exp.test() would otherwise do to TQNet/results/<setting>/ and
TQNet/test_results/<setting>/.

Config source (J-12a, STAGE2_WORKPLAN_2026-08-09.md sec 7i -- read this before
touching build_args() or evaluate_checkpoint() below):

Earlier versions of this script reconstructed each checkpoint's configuration by
regex over its directory name. That broke silently for Arm A: the regex didn't
match `_dphi<phi>`-tagged directories, and -- the dangerous part -- had it been
patched to match, `build_args()` still never set `use_damped_trend` / `damped_phi`,
so the evaluator would have built a *plain, un-detrended* model, loaded the Arm A
checkpoint into it with **zero missing and zero unexpected keys** (the trend
buffers are `persistent=False`, so they are never in `state_dict()`), and emitted a
plausible-looking validation MSE for the wrong model. Arm B (`--cycle auto`) would
have broken the regex again the same way. The bug class, not the regex, was the
problem.

**Configuration now comes only from `resolved_config.json`, written beside
`checkpoint.pth` by the training run itself (`TQNet/exp/exp_main.py`) or, for the
24 checkpoints trained before this job, backfilled once and cross-checked against
independent artefacts by `tools/backfill_checkpoint_config.py`.** A checkpoint
directory with no `resolved_config.json` is a hard failure here, never a fallback
to name parsing and never a default. The directory name is still used as an output
*label* (and as the lookup key for `checkpoint.pth` and for the output filename)
-- it is never again used as a *source of configuration*. Because
`TQNet/checkpoints/` is gitignored, `resolved_config.json` itself is not
committable; the full resolved config is therefore embedded verbatim into every
sidecar this script writes under `results/validation/`, which is committed, so
T15' traceability survives even though the source file next to the checkpoint does
not.

Before evaluating, this script runs two independent checks, both enforced by a
raise, never a warning (sec 7i requirement 1):

* `_validate_resolved_config_self_consistent` -- does `resolved_config.json`
  agree with *itself*? Its own recorded `setting` string (fixed at
  training/backfill time, untouched by later renaming the checkpoint
  directory) and its own `damped_phi`/`use_damped_trend` fields must imply
  the same thing. Without this check there is nothing for a corrupted
  `damped_phi` to disagree with: `build_args()` and the model both read that
  same field, so "does the model match the args that built it" holds by
  construction no matter what the field says. This is the check that catches
  a hand-edited or corrupted file.
* `_assert_arm_is_live` -- does the *instantiated model* actually reflect
  `args` (`use_damped_trend`, and when on, `damped_trend.phi`)? This is the
  check that catches a code bug in how the flag reaches the model, which the
  first check cannot see.

`missing_keys` / `unexpected_keys` from `load_state_dict` are captured (via
`strict=False`) and recorded even when empty, and a non-empty list is a hard
failure: empty key lists are a necessary fact about this checkpoint, never
evidence by themselves that the right model was built (sec 7i: the
damped-trend buffers are non-persistent, so a wrong model loads with empty key
lists too).
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime

THIS_FILE = os.path.abspath(__file__)
REPO_ROOT = os.path.dirname(os.path.dirname(THIS_FILE))  # tools/ -> repo root
TQNET_ROOT = os.path.join(REPO_ROOT, "TQNet")

for p in (REPO_ROOT, TQNET_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    import torch
except ImportError:
    print(
        "FATAL: `import torch` failed. This script must run in the environment "
        "that has TQNet's dependencies installed (per the job dispatch: MINGW64 "
        "on the Windows host). Stopping rather than silently skipping.",
        file=sys.stderr,
    )
    raise

import numpy as np
from torch.utils.data import DataLoader

from common import split as split_mod
from common import metrics as metrics_mod
from common import data as data_mod

from exp.exp_main import Exp_Main, _load_state_dict  # TQNet/exp/exp_main.py

CHECKPOINTS_DIR = os.path.join(TQNET_ROOT, "checkpoints")
OUT_DIR = os.path.join(REPO_ROOT, "results", "validation")
LOG_PATH = os.path.join(OUT_DIR, "validation_metrics.log")

# Transcribed read-only from TQNet/run.py (`_ETTH1_SPLIT_HASH_BY_PRED_LEN`) so this
# script can cross-check its own computed split_hash without importing run.py
# (run.py is argparse-driven and D5-owned by J-11/J-12a right now; it is not
# imported).
_ETTH1_SPLIT_HASH_BY_PRED_LEN = {
    96: "b66ee6b47e2b2eb8",
    192: "5b9f41f467356285",
    336: "a5bcaa4090739908",
    720: "17f9f51a6d81e0a2",
}

# Known sanity anchor from the dispatch / results/runs/reconstruction-TQNet-s2024-h96-*.json
ANCHOR_SETTING = "ETTh1_96_96_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed2024"
ANCHOR_TEST_MSE = 0.37104994668966473

# Used ONLY by _validate_resolved_config_self_consistent below, as a second,
# independent witness for a *file's own* damped_phi/use_damped_trend fields --
# never to build args or a model. See that function's docstring: without this,
# corrupting resolved_config.json's damped_phi has nothing to disagree with,
# because build_args() and the model both read that same corrupted value and
# therefore agree with each other by construction. This regex reads
# `cfg["setting"]` (the setting string *recorded inside the file* at
# training/backfill time), never the live checkpoint directory name -- so
# renaming the directory (criterion 5) does not touch what this checks.
_DPHI_TAG_RE = re.compile(r"_dphi(?P<phi>[0-9]+(?:\.[0-9]+)?)$")

_log_fh = None


def log(msg):
    print(msg)
    if _log_fh is not None:
        _log_fh.write(msg + "\n")
        _log_fh.flush()


def load_resolved_config(setting):
    """The one and only source of a checkpoint's configuration (J-12a sec 7i).

    Raises FileNotFoundError -- a hard failure, not a fallback -- if
    `resolved_config.json` is missing. There is deliberately no code path here
    that falls back to parsing `setting`; see the module docstring.
    """
    path = os.path.join(CHECKPOINTS_DIR, setting, "resolved_config.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "{}: no resolved_config.json found at {}. This is a hard failure -- "
            "STAGE2_WORKPLAN_2026-08-09.md sec 7i requires every checkpoint's "
            "configuration to come from this file, never from parsing the "
            "directory name. If this is one of the 24 checkpoints that predate "
            "J-12a, run `python tools/backfill_checkpoint_config.py` first.".format(
                setting, path)
        )
    with open(path) as fh:
        return json.load(fh)


def _phi_implied_by_recorded_setting(recorded_setting):
    """`(use_damped_trend, phi)` implied by `cfg["setting"]`'s own `_dphi<x>`
    suffix (or its absence), purely as an independent witness -- see
    `_validate_resolved_config_self_consistent`. Returns `(None, None)` if
    `recorded_setting` is falsy (nothing to check against).
    """
    if not recorded_setting:
        return None, None
    m = _DPHI_TAG_RE.search(recorded_setting)
    if not m:
        return False, None
    return True, float(m.group("phi"))


def _validate_resolved_config_self_consistent(cfg):
    """A second, independent cross-check on top of `_assert_arm_is_live`
    (STAGE2_WORKPLAN_2026-08-09.md sec 7i requirement 1), run first, before
    anything is built.

    Without this, corrupting `resolved_config.json`'s `damped_phi` field has
    nothing to disagree with: `build_args()` reads that same field, the model
    is built from those same args, and "does the model match the args that
    built it" therefore holds *by construction* no matter what the field's
    value is -- it cannot detect a corrupted or hand-edited file, only a code
    bug in how args reach the model. This function supplies the missing
    independent witness: `cfg["setting"]`, the setting string recorded
    *inside the file* at training/backfill time. That string is copied
    verbatim by both writers and is never touched by later renaming the
    checkpoint *directory* (J-12a acceptance criterion 5 requires the
    directory be freely renameable without affecting evaluation) -- so this
    check still passes after a rename, and still catches the file's own
    `damped_phi`/`use_damped_trend` fields disagreeing with what its own
    `setting` field implies.

    Deliberately independent of any `--force-use-damped-trend` override: that
    flag changes what is *built*, not what the file itself claims about
    itself, so this check runs the same way whether or not the CLI is forcing
    something different for a differential comparison.
    """
    recorded_setting = cfg.get("setting")
    implied_on, implied_phi = _phi_implied_by_recorded_setting(recorded_setting)
    if implied_on is None:
        return  # no recorded setting to check against -- nothing to compare

    cfg_on = bool(int(cfg.get("use_damped_trend", 0)))
    if cfg_on is not implied_on:
        raise AssertionError(
            "resolved_config.json is internally inconsistent -- its own "
            "'setting' field ({!r}) implies use_damped_trend={!r}, but its "
            "'use_damped_trend' field says {!r}. The file has been corrupted "
            "or hand-edited (J-12a acceptance criterion 6).".format(
                recorded_setting, implied_on, cfg_on)
        )
    if implied_on:
        cfg_phi = float(cfg.get("damped_phi", 0.9))
        if cfg_phi != implied_phi:
            raise AssertionError(
                "resolved_config.json is internally inconsistent -- its own "
                "'setting' field ({!r}) implies damped_phi={!r}, but its "
                "'damped_phi' field says {!r}. The file has been corrupted or "
                "hand-edited (J-12a acceptance criterion 6).".format(
                    recorded_setting, implied_phi, cfg_phi)
            )


def build_args(cfg, force_use_damped_trend=None):
    """An argparse.Namespace built entirely from a loaded resolved_config.json.

    `force_use_damped_trend`, when not None, overrides `use_damped_trend` after
    reading `cfg` -- used only by the `--setting ... --force-use-damped-trend`
    single-checkpoint CLI path below, for J-12a acceptance criterion 7's
    differential check (arm-on vs arm-off on an otherwise identical build). It
    is never set during a normal full run.
    """
    ns = argparse.Namespace()

    ns.is_training = 0
    ns.model_id = cfg.get("model_id") or cfg["setting"]
    ns.model = cfg["model"]
    ns.data = cfg["data"]
    ns.root_path = os.path.join(TQNET_ROOT, "dataset")
    ns.data_path = cfg.get("data_path", "ETTh1.csv")
    ns.features = cfg["features"]
    ns.target = cfg.get("target", "OT")
    ns.freq = cfg.get("freq", "h")
    ns.checkpoints = CHECKPOINTS_DIR

    ns.seq_len = int(cfg["seq_len"])
    ns.label_len = int(cfg.get("label_len", 0))  # fixed, per TQNet/run.py default
    ns.pred_len = int(cfg["pred_len"])

    ns.cycle = int(cfg["cycle"])
    ns.model_type = cfg.get("model_type", "mlp")
    ns.enc_in = int(cfg.get("enc_in", 7))
    ns.use_revin = int(cfg.get("use_revin", 1))
    ns.use_tq = int(cfg["use_tq"])
    ns.channel_aggre = int(cfg["channel_aggre"])
    ns.channel_criterion = int(cfg.get("channel_criterion", 0))  # decision already baked into the checkpoint

    ns.use_damped_trend = int(cfg.get("use_damped_trend", 0))
    ns.damped_phi = float(cfg.get("damped_phi", 0.9))
    if force_use_damped_trend is not None:
        ns.use_damped_trend = int(force_use_damped_trend)

    # PatchTST-family args (unused by TQNet's Model, kept for Exp_Main/args completeness)
    ns.fc_dropout = 0.05
    ns.head_dropout = 0.0
    ns.patch_len = 16
    ns.stride = 8
    ns.padding_patch = "end"
    ns.revin = 0
    ns.affine = 0
    ns.subtract_last = 0
    ns.decomposition = 0
    ns.kernel_size = 25
    ns.individual = 0

    # SegRNN args (unused by TQNet)
    ns.rnn_type = "gru"
    ns.dec_way = "pmf"
    ns.seg_len = 48
    ns.channel_id = 1

    # Former-family args (unused by TQNet)
    ns.embed_type = 0
    ns.dec_in = 7
    ns.c_out = 7
    ns.d_model = cfg.get("d_model", 512)
    ns.n_heads = 8
    ns.e_layers = 2
    ns.d_layers = 1
    ns.d_ff = 2048
    ns.moving_avg = 25
    ns.factor = 1
    ns.distil = True
    ns.dropout = cfg.get("dropout", 0.5)  # inert at eval (model.eval() disables dropout)
    ns.embed = "timeF"
    ns.activation = "gelu"
    ns.output_attention = False
    ns.do_predict = False

    ns.num_workers = 0
    ns.itr = 1
    ns.train_epochs = 30
    ns.batch_size = cfg.get("batch_size", 256)
    ns.patience = 5
    ns.learning_rate = 0.001
    ns.des = "test"
    ns.loss = "mse"
    ns.lradj = "type3"
    ns.pct_start = 0.3
    ns.use_amp = False

    ns.gpu = 0
    ns.use_multi_gpu = False
    ns.devices = "0,1"
    ns.test_flop = False

    # Deliberately the *live* machine's accelerator, not resolved_config.json's
    # (which records what accelerator the checkpoint was *trained* on -- kept in
    # the embedded resolved_config for provenance, but evaluation runs on
    # whatever this machine actually has).
    ns.accelerator = "auto"
    if torch.cuda.is_available():
        ns.accelerator = "cuda"
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        ns.accelerator = "mps"
    else:
        ns.accelerator = "cpu"
    ns.use_gpu = ns.accelerator in ("cuda", "mps")

    ns.result_path = os.path.join(TQNET_ROOT, "result_ours.txt")  # unused (no exp.test() call here)
    ns.save_outputs = 0  # unused (no exp.test() call here); explicit 0 so nothing writes if it ever is

    ns.random_seed = cfg.get("seed")

    return ns


def _assert_arm_is_live(setting, exp, args):
    """Sec 7i requirement 1: assert the constructed model's damped-trend state
    matches what it was configured to be, BEFORE evaluating. Enforced by a
    raise. This is the check that would have caught the silent failure this
    job exists to close -- a model built with the flag off, evaluated anyway,
    with `load_state_dict` reporting empty missing/unexpected key lists either
    way (the buffers are `persistent=False`; see the module docstring).
    """
    expected_on = bool(args.use_damped_trend)
    actual_on = bool(exp.model.use_damped_trend)
    if actual_on is not expected_on:
        raise AssertionError(
            "{}: model.use_damped_trend is {!r} but the build asked for "
            "use_damped_trend={!r}. Refusing to evaluate -- the arm did not "
            "reach the model as configured (STAGE2_WORKPLAN_2026-08-09.md "
            "sec 7i, requirement 1).".format(setting, actual_on, expected_on)
        )
    if expected_on:
        expected_phi = float(args.damped_phi)
        actual_phi = float(exp.model.damped_trend.phi)
        if actual_phi != expected_phi:
            raise AssertionError(
                "{}: model.damped_trend.phi is {!r} but the build asked for "
                "damped_phi={!r}. Refusing to evaluate.".format(
                    setting, actual_phi, expected_phi)
            )
    else:
        if hasattr(exp.model, "damped_trend"):
            raise AssertionError(
                "{}: use_damped_trend is False but the model still has a "
                "damped_trend submodule. Refusing to evaluate.".format(setting)
            )


def evaluate_checkpoint(setting, force_use_damped_trend=None):
    cfg = load_resolved_config(setting)
    _validate_resolved_config_self_consistent(cfg)  # before anything is built; see docstring
    args = build_args(cfg, force_use_damped_trend=force_use_damped_trend)

    ckpt_dir = os.path.join(CHECKPOINTS_DIR, setting)
    ckpt_path = os.path.join(ckpt_dir, "checkpoint.pth")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError("no checkpoint.pth under {}".format(ckpt_dir))

    exp = Exp_Main(args)
    state_dict = _load_state_dict(ckpt_path, exp.device)
    # strict=False so missing/unexpected keys are captured explicitly (sec 7i
    # requirement 9) rather than only appearing inside a raised exception's
    # message. Empty lists are asserted below, not just reported: the
    # damped-trend buffers are non-persistent (TQNet/layers/DampedTrend.py),
    # so a WRONG model (arm off when it should be on) also loads with empty
    # missing/unexpected keys -- see the module docstring. Emptiness alone is
    # therefore never treated as evidence the arm is live; `_assert_arm_is_live`
    # below is what actually proves that.
    load_result = exp.model.load_state_dict(state_dict, strict=False)
    missing_keys = list(load_result.missing_keys)
    unexpected_keys = list(load_result.unexpected_keys)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "{}: state_dict mismatch -- missing_keys={!r} unexpected_keys={!r}. "
            "An evaluation with a mismatched state dict is not a result "
            "(standing order 6 / R11).".format(setting, missing_keys, unexpected_keys)
        )
    exp.model.eval()

    _assert_arm_is_live(setting, exp, args)

    n_params = int(sum(p.numel() for p in exp.model.parameters() if p.requires_grad))

    vali_data, _discarded_val_loader = exp._get_data(flag="val")
    test_data, _discarded_test_loader = exp._get_data(flag="test")
    n_val_raw = len(vali_data)
    n_test_raw = len(test_data)

    # Deliberately shuffle=False, drop_last=False -- see module docstring point 3.
    val_loader = DataLoader(vali_data, batch_size=args.batch_size, shuffle=False,
                             drop_last=False, num_workers=0)

    preds, trues = [], []
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle in val_loader:
            batch_x = batch_x.float().to(exp.device)
            batch_y = batch_y.float().to(exp.device)
            batch_cycle = batch_cycle.int().to(exp.device)

            # Mirrors exp_main.py test()'s model-call dispatch: args.model == 'TQNet'
            # matches the {'CycleNet', 'TQ'} substring branch.
            outputs = exp.model(batch_x, batch_cycle)

            f_dim = -1 if args.features == "MS" else 0  # features == 'M' here -> 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:]

            preds.append(outputs.detach().cpu().numpy())
            trues.append(batch_y.detach().cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

    n_windows_evaluated = int(preds.shape[0])

    m = metrics_mod.all_metrics(trues, preds)  # common/metrics.py, z-scored, flat reduction

    csv_path = os.path.join(args.root_path, args.data_path)
    digest = data_mod.data_sha256(csv_path)
    computed_hash = split_mod.split_hash(args.seq_len, args.pred_len, digest)
    expected_hash = _ETTH1_SPLIT_HASH_BY_PRED_LEN.get(args.pred_len)
    hash_ok = (expected_hash is not None) and (computed_hash == expected_hash)

    record = {
        "setting": setting,
        "arm": cfg.get("arm"),  # informational only; not used to build args (see load_resolved_config)
        "model": args.model,
        "data": args.data,
        "seed": cfg.get("seed"),
        "seq_len": args.seq_len,
        "pred_len": args.pred_len,
        "cycle": args.cycle,
        "use_tq": args.use_tq,
        "channel_aggre": args.channel_aggre,
        "use_damped_trend": bool(args.use_damped_trend),
        "damped_phi": float(args.damped_phi),
        "forced_use_damped_trend": force_use_damped_trend,
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "val_MSE": m["MSE"],
        "val_MAE": m["MAE"],
        "val_RMSE": m["RMSE"],
        "val_MdAE": m["MdAE"],
        "n_windows_val": n_windows_evaluated,
        "n_windows_val_dataset_len": n_val_raw,
        "n_windows_test_dataset_len": n_test_raw,
        "n_points": int(trues.size),
        "n_params": n_params,
        "checkpoint_path": os.path.relpath(ckpt_path, REPO_ROOT).replace(os.sep, "/"),
        "data_sha256": digest,
        "split_hash": computed_hash,
        "split_hash_expected": expected_hash,
        "split_hash_ok": hash_ok,
        "accelerator": args.accelerator,
        "scale": "z-scored (no inverse_transform applied; matches exp_main.py test(), "
                 "whose de-norm block is commented out)",
        "val_loader": "shuffle=False, drop_last=False, batch_size={} "
                      "(deliberately not vali()'s train-style shuffle=True/drop_last=True "
                      "loader; see script docstring point 3)".format(args.batch_size),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "script_path": "tools/validation_metrics.py",
        # J-12a Step 4: the full resolved config, embedded verbatim. checkpoints/
        # is gitignored, so this sidecar (which IS committed, under
        # results/validation/) is the only place this configuration is
        # traceable from after the fact (T15').
        "resolved_config": cfg,
    }

    if not hash_ok:
        log("  WARNING: split_hash mismatch or unknown pred_len={} "
            "(computed={}, expected={})".format(args.pred_len, computed_hash, expected_hash))

    return record, n_val_raw, n_test_raw


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--setting", default=None,
        help="Evaluate a single checkpoint directory under TQNet/checkpoints/ and "
             "print its record to stdout. Does not write to results/validation/ "
             "and does not touch the full-run log. For ad hoc checks (J-12a "
             "acceptance criteria 6/7), not for normal use.",
    )
    parser.add_argument(
        "--force-use-damped-trend", type=int, choices=[0, 1], default=None,
        help="Override resolved_config.json's use_damped_trend for this one "
             "evaluation. Only valid together with --setting. For J-12a "
             "acceptance criterion 7's differential check: run once without "
             "this flag and once with --force-use-damped-trend 0 on the same "
             "Arm A checkpoint and compare val_MSE.",
    )
    args = parser.parse_args(argv)

    if args.force_use_damped_trend is not None and args.setting is None:
        parser.error("--force-use-damped-trend requires --setting")

    if args.setting is not None:
        record, n_val_raw, n_test_raw = evaluate_checkpoint(
            args.setting, force_use_damped_trend=args.force_use_damped_trend
        )
        print(json.dumps(record, indent=2, sort_keys=True))
        return 0

    return run_all()


def run_all():
    os.makedirs(OUT_DIR, exist_ok=True)
    global _log_fh
    _log_fh = open(LOG_PATH, "a")

    log("=" * 78)
    log("validation_metrics.py run starting {}".format(datetime.now().isoformat(timespec="seconds")))
    log("REPO_ROOT = {}".format(REPO_ROOT))
    log("TQNET_ROOT = {}".format(TQNET_ROOT))
    log("torch version = {}".format(torch.__version__))
    log("torch.cuda.is_available() = {}".format(torch.cuda.is_available()))

    if not os.path.isdir(CHECKPOINTS_DIR):
        log("FATAL: checkpoints dir not found: {}".format(CHECKPOINTS_DIR))
        return 1

    settings = sorted(
        d for d in os.listdir(CHECKPOINTS_DIR)
        if os.path.isdir(os.path.join(CHECKPOINTS_DIR, d))
    )
    log("Found {} checkpoint directories.".format(len(settings)))

    records = []
    failures = []
    t0 = time.time()

    for i, setting in enumerate(settings, 1):
        log("[{}/{}] {}".format(i, len(settings), setting))
        try:
            record, n_val_raw, n_test_raw = evaluate_checkpoint(setting)
        except Exception:
            log("  FAILED:\n" + traceback.format_exc())
            failures.append(setting)
            continue

        log("  arm={} seed={} pred_len={} val_MSE={!r} val_MAE={!r}".format(
            record["arm"], record["seed"], record["pred_len"], record["val_MSE"], record["val_MAE"]))
        log("  use_damped_trend={} damped_phi={!r} missing_keys={!r} unexpected_keys={!r}".format(
            record["use_damped_trend"], record["damped_phi"],
            record["missing_keys"], record["unexpected_keys"]))
        log("  n_windows_val(evaluated)={} n_windows_val(dataset_len)={} "
            "n_windows_test(dataset_len)={}".format(
                record["n_windows_val"], n_val_raw, n_test_raw))
        if n_val_raw != n_test_raw:
            log("  val/test dataset-length windows DIFFER ({} vs {})".format(n_val_raw, n_test_raw))
        else:
            log("  val/test dataset-length windows are EQUAL ({} each) -- expected for this split "
                "(val and test are both exactly 4 calendar months, see script docstring); "
                "rows underneath are disjoint ([8544,11520) vs [11424,14400)).".format(n_val_raw))

        if setting == ANCHOR_SETTING:
            diff = record["val_MSE"] - ANCHOR_TEST_MSE
            log("  SANITY ANCHOR ({}): val_MSE={!r} test_MSE={!r} diff={!r}".format(
                setting, record["val_MSE"], ANCHOR_TEST_MSE, diff))
            if record["val_MSE"] == ANCHOR_TEST_MSE:
                log("  FATAL: validation MSE is bit-identical to the known test MSE -- "
                    "this indicates the test split was evaluated instead of validation. Stopping.")
                _log_fh.close()
                sys.exit(2)
            ratio = record["val_MSE"] / ANCHOR_TEST_MSE
            if not (0.1 <= ratio <= 10):
                log("  WARNING: val_MSE is more than 10x away from test_MSE (ratio={!r}); "
                    "implausible neighbourhood per dispatch trap 3.".format(ratio))

        out_path = os.path.join(OUT_DIR, setting + ".json")
        with open(out_path, "w") as fh:
            json.dump(record, fh, indent=2, sort_keys=True)
            fh.write("\n")
        records.append(record)

    elapsed = time.time() - t0
    log("Done: {} succeeded, {} failed, {:.1f}s elapsed.".format(len(records), len(failures), elapsed))
    if failures:
        log("Failed settings: {}".format(failures))

    summary_path = os.path.join(OUT_DIR, "_summary.json")
    with open(summary_path, "w") as fh:
        json.dump(
            {
                "generated": datetime.now().isoformat(timespec="seconds"),
                "n_records": len(records),
                "n_failures": len(failures),
                "failures": failures,
                "records": records,
            },
            fh,
            indent=2,
            sort_keys=True,
        )
        fh.write("\n")
    log("Wrote summary: {}".format(summary_path))
    log("=" * 78)
    _log_fh.close()

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
