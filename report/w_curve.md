# Arm B — the W-sensitivity curve

Run: `run_w_curve.sh`, 2026-08-10 15:32:55 -> 16:04:06 (wall-clock 1,911s). Log: `w_curve.log`
(sha256 `2909b511665bfa1e3865af807bfb8ef62a84cc2ed7a1bdc2ed9e1fabd0bd45d4`). Source data:
`results/validation/*.json`, rolled up in `results/validation/_summary.json` (sha256
`a8be034cec651dbc78f810f8bcc3cdeaa35fd16c863bfdc6f27ac825c44b03be`). Every training run asserted
split hash `b66ee6b47e2b2eb8` at L=H=96 (standing order 12) — confirmed in `w_curve.log` for all 22
runs.

**Anchor note — corrected 2026-08-10, job J-15d.** The pre-incident reconstruction anchor at
W=24/seed2024 (`0.6712632722155959`) was briefly treated as lost — see
`STAGE2_WORKPLAN_2026-08-09.md` §7j/§7k — and the W=24 row below was written from the
reproducibly-observed value then sitting in the checkpoint directory, `0.6869550701723053`, noted at
the time as "the working anchor." **That value's origin is now identified: it is an epoch-3 snapshot
of an interrupted retrain, not a competing measurement of the reconstruction.** See "Provenance of
the W = 24 row" below for the evidence. The table has been corrected accordingly; the as-written
value is retained, struck through, in its own row for the record.

## The eight-point table

Reconstruction validation MSE, three seeds, per W. W=24 is **not retrained** — its row is the
existing checkpoint at `TQNet/checkpoints/ETTh1_96_96_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed*`,
re-evaluated, per the dispatch's instruction not to re-run an identical computation to fill a cell.

| W | seed 2024 | seed 2025 | seed 2026 | mean | sd (n-1) |
|---|---|---|---|---|---|
| 6 | 0.6712658063450453 | 0.6726172796379547 | 0.6703286251080852 | 0.6714039036970284 | 0.0011505598846829 |
| 8 | 0.6715714846197122 | 0.6731067525019135 | 0.6706429722567896 | 0.6717737364594718 | 0.0012442799641079 |
| 12 | 0.6716537568893131 | 0.6737847414964826 | 0.6733536080981822 | 0.6729307021613260 | 0.0011266813757978 |
| 23 | 0.6712007869417990 | 0.6736105864591796 | 0.6712481906014808 | 0.6720198546674865 | 0.0013778180219337 |
| **24 (corrected — see "Provenance of the W = 24 row")** | 0.6712632722155959 | 0.6735143635280876 | 0.6727194169596606 | 0.6724990175677814 | 0.0011416150591374683 |
| ~~24, as written 2026-08-10 15:32–16:04 (superseded — epoch-3 artefact, see below)~~ | ~~0.6869550701723053~~ | ~~0.6735143635280876~~ | ~~0.6727194169596606~~ | ~~0.6777296168866845~~ | ~~0.0079993578652003~~ |
| 25 | 0.6722750953831385 | 0.6745795077563317 | 0.6716480837470329 | 0.6728342289621677 | 0.0015436270436773 |
| 48 | 0.6713818477132550 | 0.6734766012401661 | 0.6727290271493210 | 0.6725291587009140 | 0.0010615830782977 |
| 168 | 0.6713853217988164 | 0.6732518433132122 | 0.6733619263729219 | 0.6726663638283168 | 0.0011107794925271 |

n_params = 661,640 for every reconstruction checkpoint in this table (confirmed by
`validation_metrics.py` for each).

## Provenance of the W = 24 row

**The cell now printed is the pre-incident measurement.** `0.6712632722155959` is traced to
`results/validation/validation_metrics.log` (sha256
`4af37f4fe4be4fcc01fb17e30a15db0b06c3b390fc764d22d205e0b4bb9b573d`), which records five prior runs of
`validation_metrics.py` (before §7j's incident) each logging, for
`ETTh1_96_96_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed2024`: `val_MSE=0.6712632722155959` and a `SANITY
ANCHOR` line quoting the same figure. The same log's five later entries, after the incident, show the
value flipping to `0.6869550701723053` and staying there — the same file documents both states of the
same directory over time. **Note for whoever reads this next:** the correction dispatch pointed at
`results/validation/ETTh1_96_96_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed2024.json` as the source for
`0.6712632722155959`; that JSON is auto-regenerated on every `validation_metrics.py` run and today
holds the *current* checkpoint's value, `0.6869550701723053`, not the pre-incident one — it cannot
source the corrected figure. `validation_metrics.log` is the file that actually carries both values
with their history, and is what the sourcing above uses instead.

**What is now sitting in that checkpoint directory is an epoch-3 artefact, not an alternative
measurement.** §7j records that a `head`-truncated verification command, piped from a training run
that reused the protected setting string, kept training in the background after the pipe closed and
overwrote `checkpoint.pth` with an early-stopping snapshot from a partial run. Fingerprint evidence:
a separate, independently-truncated run at the same setting (`ETTh1_96_96_armB_smoke_...cycle24
_seed2024`, `results/validation/ETTh1_96_96_armB_smoke_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed2024
.json`) reports `val_MSE=0.6869550703100449` — agreeing with the protected directory's
`0.6869550701723053` to **1.4e-10**. Two runs truncated the same way landing on the same value to ten
significant figures is deterministic training stopped at the same epoch, not coincidence.

**Two independent retrains, run to completion, corroborate the pre-incident value instead:**

| Run | val_MSE | Deviation from `0.6712632722155959` |
|---|---|---|
| `ETTh1_96_96_armB_auto_...cycle24_seed2024` (`--cycle auto`, same architecture/seed, distinct `model_id`) | 0.6712632724477633 | +2.3e-10 |
| `ETTh1_96_96_reconstruction_v2_...` (dedicated clean retrain) | 0.6712632718563285 | −3.6e-10 |

Three runs that trained to completion — the original pre-incident measurement and these two
retrains — agree to under 4e-10. The checkpoint now in the protected directory disagrees with all
three by roughly 0.0157, about eight orders of magnitude larger than that spread.

**`ETTh1_96_96_reconstruction_v2` is the replacement artefact** — the re-evaluable checkpoint to cite
going forward — while the reported *number* in the table above stays the pre-incident measurement,
because its provenance is unified with the committed test-split records in `results/runs/` from the
same 09.08 training (F5's reconstruction column). The original and v2 differ by 3.6e-10 against
σ_seed = 0.0011416150591374683 — about six orders of magnitude below σ, so no verdict in this study
turns on which of the two is used.

**The epoch-3 checkpoint is retired as a data source, not deleted.** It remains on disk at
`TQNet/checkpoints/ETTh1_96_96_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed2024/checkpoint.pth`
(sha256 `c5d0f7bbc057d48608c15e60a4872712b363a5fa4c12238ba77fb16860773ca2`, unchanged) — it is the
evidence for §7j and for F6, and this project's norm is that a superseded value is kept and marked,
never silently deleted.

**Effect on the spread measurement:**

| W=24 seed-2024 cell | spread across the eight per-W means | vs 2σ (`0.0022832301182749366`) |
|---|---|---|
| as written (`0.6869550701723053`) | 0.0063257 | 2.77 × 2σ |
| corrected (`0.6712632722155959`) | 0.0015268 | 0.67 × 2σ |

The two spreads fall on opposite sides of the 2σ line.

## Spread against 2σ

Reference: σ_seed = 0.0011416150591374683 (reconstruction's own three-seed validation sd at
H=96, from the three seed sidecars). 2σ = 0.0022832301182749366.

**Corrected table (current W=24 row, `0.6712632722155959`):** max mean (W=12, 0.6729307021613260) −
min mean (W=6, 0.6714039036970284) = **0.0015267984642975962**, which is **0.6687 × 2σ**.

**As written 2026-08-10 15:32–16:04 (superseded row, `0.6869550701723053`), kept for the record:**
max mean (W=24, 0.6777296168866845) − min mean (W=6, 0.6714039036970284) = **0.006325713189656135**,
which is **2.77 × 2σ**.

Both numbers are stated as measurements. Whether the corrected spread's position relative to 2σ
confirms or falsifies §3's flatness prediction is J-16's ruling, not this job's.

## What this run adds to the anchor question

**Superseded by "Provenance of the W = 24 row" above (job J-15d, 2026-08-10), kept for the record
rather than deleted.** This section is the same-day narrative of how the evidence below was first
noticed, written before the PM ruling that the adopted-anchor checkpoint is an epoch-3 artefact and
the W=24 cell should read the pre-incident value. It is left as originally written.

Not a re-litigation of §7k's decision — a new, independent data point that arrived as a side effect
of this run and should be read before anyone treats the adopted anchor as final.

Every W in the sweep at seed 2024 — including the `--cycle auto` run, which uses the *same*
architecture, cycle=24, and seed as the reconstruction, just a different `model_id`
(`ETTh1_96_96_armB_auto`) — lands in a tight band:

| Setting | seed-2024 val_MSE |
|---|---|
| W=6 | 0.6712658063450453 |
| W=8 | 0.6715714846197122 |
| W=12 | 0.6716537568893131 |
| W=23 | 0.6712007869417990 |
| W=25 | 0.6722750953831385 |
| W=48 | 0.6713818477132550 |
| W=168 | 0.6713853217988164 |
| **`--cycle auto` (cycle=24, distinct model_id)** | **0.6712632724477633** |
| **W=24, the current (adopted-anchor) reconstruction checkpoint** | **0.6869550701723053** |

The `--cycle auto` run's value (`0.6712632724477633`) differs from the **lost original anchor**
(`0.6712632722155959`) by `2.3e-10` — the same order of magnitude as the noise floor already
established for this codebase (the smoke-run/repair-run pair in §7k differ by `2.4e-10`, at a
*different* value). It differs from the **adopted anchor** (`0.6869550701723053`) by `-0.0157` —
roughly 68 million times larger than that noise floor.

Put plainly: everything else this run touched at cycle≈24, seed 2024 — including a completely
independent training run with identical hyperparameters — lands within noise of the number that was
lost, not the number that was adopted. The adopted anchor is the outlier in its own neighborhood:
its row has a three-seed sd of `0.008`, seven times every other row's sd (`0.0011`–`0.0015`), and it
alone accounts for the entire gap between the "spread including W=24" and "spread excluding W=24"
figures above.

This does not retroactively fail Precondition 1 — that check is about the file on disk today, and
today's file still deterministically produces `0.6869550701723053`. It is a factual observation for
whoever decides what happens next with that checkpoint: this run produced strong independent
evidence that the pre-incident anchor value was the well-behaved one, and the current checkpoint at
`TQNet/checkpoints/ETTh1_96_96_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed2024/` is not representative of
what this architecture/seed/cycle combination normally produces.

## `--cycle auto` demonstration (§3 prediction 2)

`ETTh1_96_96_armB_auto_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed2024`, `resolved_config.json`:
`cycle: 24`, `cycle_source: "estimated"` — confirmed in `w_curve.log`
(`[cycle_auto] acf=24 (peak=0.8851540915365546) periodogram=24 (power=19641471.641798608) agree=True`).

val_MSE = `0.6712632724477633`.

- Against the **adopted anchor** (`0.6869550701723053`): difference = `-0.01569179772454199`, far
  outside the predicted ±0.0005.
- Against the **lost original anchor** (`0.6712632722155959`): difference = `0.0000000002321674`,
  inside ±0.0005 by roughly six orders of magnitude — this is the comparison §3 predicted, and it is
  the one that holds.

## Checkpoints created

21 W-curve checkpoints (`TQNet/checkpoints/*_cycle{6,8,12,23,25,48,168}_seed{2024,2025,2026}/`) plus
`TQNet/checkpoints/ETTh1_96_96_armB_auto_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed2024/`. All 27
pre-existing checkpoints (including the 26 originally protected) are untouched by this run —
`diff /tmp/ckpt_before.txt /tmp/ckpt_after.txt` in `w_curve.log` shows additions only, no removed or
modified lines among the pre-existing 27.

No test-split number appears in this file.
