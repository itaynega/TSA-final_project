# Rules for any agent working in this repository

This project's full protocol lives in `../STAGE2_WORKPLAN_2026-08-09.md` (§6, standing orders) and
`report/prereg-improvement.md` (frozen pre-registration). Read those before doing substantive work.
This file exists so the single most important rule survives even a session that skips the workplan.

## Never overwrite data without a before/after hash check

Before running any command that could train into, delete, or otherwise modify an existing
checkpoint, result file, or dataset artifact:

1. Hash (sha256) every file the command could plausibly touch — not just the ones it is *supposed*
   to touch. `find <dir> -name checkpoint.pth -exec sha256sum {} \;` or similar.
2. Run the command.
3. Hash again. Diff the two lists. The diff must show only the intended changes.
4. Anything else that changed is a stop-and-report event — not something to reconcile afterward from
   memory, from stdout, or from a loss trajectory that "looks the same."

**A trajectory match is not a hash match.** This rule exists because on 2026-08-10 a verification
command reused an existing checkpoint's setting string and (via a `head`-truncated pipe that
truncated the display, not the training process) silently overwrote one of 26 explicitly protected
checkpoints. The repair that followed was verified by eyeballing per-epoch losses against the wrong
reference run and judged "exact" — but the actual weights it produced have a different sha256 from
both the original and the reference, and a different `val_MSE` at the 10th significant figure. No
backup existed in any snapshot, checkpoints are not git-tracked, and training on this machine is not
bit-reproducible under a fixed seed on CPU (`torch.backends.cudnn.deterministic` is set but is a
CUDA-only guarantee; nothing pins CPU thread count or calls
`torch.use_deterministic_algorithms()`). The original anchor value was never recovered from any path
available to an agent. Full incident record: `STAGE2_WORKPLAN_2026-08-09.md` §7j and §7k.

## Two other things that follow from the incident

- **Before treating a "repaired" or "restored" file as equivalent to the original, prove it by hash,
  not by re-deriving a metric and eyeballing whether it's close.** A metric landing close is
  consistent with either a true restore or an independent retrain that converged nearby — those are
  very different outcomes and only the hash distinguishes them.
- **If a protected file is ever found to have changed unexpectedly, stop before running anything
  else that writes.** Check for a real backup (OneDrive version history, if the repo is inside
  OneDrive, is outside the sandbox mount and must be checked by the human directly) before accepting
  the loss and adopting a new value.
