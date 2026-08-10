# Fact sheet — every number the report may print, frozen

**Job J-17. Written 2026-08-10. Sole writer of this file.**

This file exists so that F1–F7 quote and never re-derive. Every row below carries **value ·
what it is · source path · sha256 of that file · split label**. A number that cannot carry all
four does not appear here, and under standing order T15′ does not appear in the report either —
see [`## Withheld under T15′`](#withheld-under-t15) for the ones that are barred.

**Nothing in this file was computed by re-running an evaluator.** Every value was read from a
file already on disk, or arithmetically combined from values so read, with the combination shown.
No training ran, no `git` command that writes ran, and `tools/validation_metrics.py` was not
invoked.

---

## 0. How to read this sheet

### 0.1 Split labels

| Tag | Meaning |
|---|---|
| `[validation]` | Validation rows `[8544, 11520)`. §4.1's selection endpoint. |
| `[test]` | Test rows `[11424, 14400)`. §4.3's reporting split. **Never a selection input.** |
| `[training]` | Training rows `[0, 8640)`. Estimator and criterion statistics only. |
| `[config]` | Not a split quantity — parameter counts, hashes, flags, wall-clock, read from a resolved config or run log. |

`val_MSE ≈ 1.8 × test_MSE` at H = 96 on this split. The two are **never** comparable and no
sentence in this file places one beside the other without both labels.

### 0.2 Precision rule (criterion 10)

**Every value is quoted at the precision of its source. No rounding, no reformatting.** Where a
document rounds and a result file does not, the result file's precision is what appears here and
the document's rounding is flagged in §8. This matters: §7's `5.65%` vs `5.66%` disagreement is a
rounding artefact of the single quantity `5.655038994014872` %, and a sheet that rounded would
have reproduced the same ambiguity instead of dissolving it.

### 0.3 Mean and sd conventions

- **Three-seed means** are `statistics.mean` (exact-rational accumulation, correctly rounded once).
  `sum(x)/3` and `math.fsum(x)/3` differ from it in the sixteenth digit. Per `report/selection.md`
  §0, `statistics.mean` is this project's convention and is what every mean below uses.
- **Standard deviations** are sample sd, **n−1** in the denominator (`statistics.stdev`,
  `numpy.std(ddof=1)`), per `report/horizon_sigma.md`'s stated convention.

### 0.4 Derived rows

A row marked **[derived]** is arithmetic over rows in this same sheet, with the inputs named. Its
`sha256` column carries the hashes of the files the inputs came from. No derived row introduces a
quantity that is not already traceable.

---

## 1. The selection — §4.2 applied

**Endpoint (§4.1, frozen): mean validation MSE at H = 96 over seeds 2024/2025/2026.** Two arms
reached it: B and D. Arms A and C did not produce a three-seed mean at H = 96 and cannot be ranked.

### 1.1 Arm B — per-seed values behind the mean

| Value | What it is | Source path | sha256 | Split |
|---|---|---|---|---|
| `0.6712632722155959` | Arm B / reconstruction val_MSE, H = 96, seed 2024 | `results/validation/validation_metrics.log` — **not** the seed-2024 sidecar; see §2 | `4af37f4fe4be4fcc01fb17e30a15db0b06c3b390fc764d22d205e0b4bb9b573d` | `[validation]` |
| `0.6735143635280876` | Arm B / reconstruction val_MSE, H = 96, seed 2025 | `results/validation/ETTh1_96_96_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed2025.json` | `9907d633e153b6d667b8f9f4007091e86b1160313256d64b21e2e738be650fbb` | `[validation]` |
| `0.6727194169596606` | Arm B / reconstruction val_MSE, H = 96, seed 2026 | `results/validation/ETTh1_96_96_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed2026.json` | `193f4d1976563ce99ca1d467596df85908e1a64c6309da24b2f41daa42c60094` | `[validation]` |
| **`0.6724990175677814`** | **Arm B three-seed mean val_MSE at H = 96 — the §4.1 endpoint value** **[derived]** | `statistics.mean` of the three rows above | `4af37f4f…` + `9907d633…` + `193f4d19…` | `[validation]` |
| `0.0011416150591374683` | Arm B / reconstruction three-seed sd (n−1) at H = 96 — **σ_validation**, the σ ruling (c) applies to validation comparisons | `statistics.stdev` of the three rows above | `4af37f4f…` + `9907d633…` + `193f4d19…` | `[validation]` |
| `0.0022832301182749365` | 2 × σ_validation, used for the W-curve flatness band **[derived]** | `2 × 0.0011416150591374683` | as above | `[validation]` |

> **`0.0022832301182749365`, not `…366`.** `report/w_curve.md` prints `0.0022832301182749366`.
> Every accumulation route (`sd*2`, `sd+sd`, `numpy`) returns `…365`. Standing order 5 — the run
> wins — so `…365` is the frozen value. See §8.

### 1.2 Arm D — per-seed values behind the mean

| Value | What it is | Source path | sha256 | Split |
|---|---|---|---|---|
| `0.6795092456048932` | Arm D val_MSE, H = 96, seed 2024 | `results/validation/ETTh1_96_96_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed2024_tq0ca0.json` | `5f1e3f69eb8628e31e6fc3a684c85839e7e4bd8be6aed942b52d9234cd7bd13d` | `[validation]` |
| `0.6812630824349228` | Arm D val_MSE, H = 96, seed 2025 | `results/validation/ETTh1_96_96_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed2025_tq0ca0.json` | `896415fcf2113915aa8a5a4662df2188ec0c2e300447c7ae07c51ca9b1e0e862` | `[validation]` |
| `0.6808913767587101` | Arm D val_MSE, H = 96, seed 2026 | `results/validation/ETTh1_96_96_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed2026_tq0ca0.json` | `8933efc4958712bc66041ce36a539bb35e6697c61172a806d1acaf8beb76b5b7` | `[validation]` |
| **`0.6805545682661754`** | **Arm D three-seed mean val_MSE at H = 96** **[derived]** | `statistics.mean` of the three rows above | `5f1e3f69…` + `896415fc…` + `8933efc4…` | `[validation]` |
| `0.0009241568465767698` | Arm D three-seed sd (n−1) at H = 96 | `statistics.stdev` of the three rows above | as above | `[validation]` |

All six sidecars in §1.1–§1.2 assert `split_hash = b66ee6b47e2b2eb8` with `split_hash_ok: true`
(standing order 12), verified by decoding each file (D9′).

### 1.3 The ranking, the margin, the selected arm

| Value | What it is | Source path | sha256 | Split |
|---|---|---|---|---|
| `0.6724990175677814` | **Arm B — rank 1 (lowest).** The selected arm. | as §1.1 | `4af37f4f…` + `9907d633…` + `193f4d19…` | `[validation]` |
| `0.6805545682661754` | Arm D — rank 2 | as §1.2 | `5f1e3f69…` + `896415fc…` + `8933efc4…` | `[validation]` |
| **`0.008055550698394032`** | **Margin, Arm B under Arm D** **[derived]** | `0.6805545682661754 − 0.6724990175677814` | as above | `[validation]` |
| `16.1` | Margin as a multiple of §4.2's 0.0005 tie band — the tiebreak clause is therefore inert **[derived]** | `0.008055550698394032 / 0.0005` | as above | `[validation]` |
| `+7.056275785710372` | Arm D's degradation in units of σ_validation **[derived]** | `0.008055550698394032 / 0.0011416150591374683` | as above | `[validation]` |
| `+3.7394663...` | The same degradation against §4.4's parenthesised σ_test, recorded only to show ruling (c) is outcome-neutral | `0.008055550698394032 / 0.0021541981747125473` | as above | mixed — **do not quote in the report**; see note |

> **The last row mixes splits by construction** (a `[validation]` numerator over a `[test]`
> denominator) and exists only because `report/selection.md` ruling (c) had to show that the σ
> choice changes no verdict. It is not a reportable quantity. F-authors quote `+7.056275785710372`
> `[validation]`.

### 1.4 ⚠ Arm B's dual status — both halves are required

**Arm B is the selected arm under §4.2 *and* is classified "no measurable effect" under §4.4.
These are not in tension and a fact sheet that carried only one half would read as an error.**

- **Selected (§4.2).** §4.2 is a total order on one measured quantity. Arm B's mean,
  `0.6724990175677814` `[validation]`, is the lowest among arms that reached the endpoint. The
  rule conditions on nothing else.
- **No measurable effect (§4.4).** Arm B's estimator returned W = 24 `[training]`, which is the
  value the reconstruction already used. The `--cycle auto` path therefore produces the
  byte-identical setting string and an architecturally identical model
  (`n_params = 661640` `[config]`). **Arm B's W = 24 row is not a retrain — it re-evaluates the
  reconstruction's own three checkpoints.** Its difference from the reconstruction is exactly `0`
  on the shared checkpoints, and `2.321674e-10` `[validation]` via the independent `--cycle auto`
  run — roughly seven orders of magnitude below 1σ_validation.

**In words: this study's four-arm screen selected, as its improved column, an arm whose
contribution is a procedure for deriving a hyperparameter that was already correct — producing a
model numerically identical to the baseline it was screened against.** §4.2 ranks; §4.4
classifies; the frozen text applies them independently. F6 carries this.

---

## 2. The landmine — three artefacts, none of them quotable

**All three rows below are recorded as provenance. No number anywhere in this sheet is sourced
from any of them.**

| Value | What it is | Source path | sha256 | Split |
|---|---|---|---|---|
| `0.6869550701723053` | **SUPERSEDED.** `val_MSE` currently in the seed-2024 sidecar. An **epoch-3 artefact** of a checkpoint overwritten during the §7j incident — a faithful measurement of a broken model, not a measurement of the reconstruction. **Not to be quoted.** | `results/validation/ETTh1_96_96_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed2024.json` | `ed715742bd1602e447e99fe1af2c9138f6c87a96e59911ab18bd9834124ea6e3` | `[validation]` |
| `0.6869550701723053` | **SUPERSEDED AND DERIVED.** The same poisoned value, re-aggregated. This 201,494-byte roll-up of all 50 sidecars **looks authoritative and is not**: it carries the poisoned figure and does **not** contain the true one, because it is generated from the same sidecars. **Not to be quoted, for any number, ever.** | `results/validation/_summary.json` | `d6f77703d135cea199a028b496ed210f2e1bfa1a341f03589b1f68b27dd3dbbc` | `[validation]` |
| — (binary; no value) | **RETAINED BROKEN CHECKPOINT.** The epoch-3 weights themselves, kept as §7j / F6 evidence and never deleted. **Gitignored — it exists in the submission only as this hash.** Hashed read-only; not loaded, not evaluated. | `TQNet/checkpoints/ETTh1_96_96_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed2024/checkpoint.pth` | `c5d0f7bbc057d48608c15e60a4872712b363a5fa4c12238ba77fb16860773ca2` | `[config]` |

### 2.1 The true value and its four corroborating sources

**In words, because a reader who misses this computes the wrong selection: the seed-2024 sidecar
and `_summary.json` both hold a superseded value. The reconstruction's validation MSE at H = 96,
seed 2024, is `0.6712632722155959`.**

| Value | What it is | Source path | sha256 | Split |
|---|---|---|---|---|
| **`0.6712632722155959`** | **The true value.** Run log carrying both states of the same directory over time: five pre-incident entries log this figure with a `SANITY ANCHOR` line; five later entries flip to the poisoned figure and stay there. | `results/validation/validation_metrics.log` | `4af37f4fe4be4fcc01fb17e30a15db0b06c3b390fc764d22d205e0b4bb9b573d` | `[validation]` |
| `0.6712632722155959` | Corroboration 1 — the corrected W = 24 row, with the superseded value retained struck through | `report/w_curve.md` | `0e320a97388ac59997b5389a349236bf42f29c4811417a1733b64c5ee1e56e89` | `[validation]` |
| `0.6712632718563285` | Corroboration 2 — dedicated clean retrain, **−3.6e-10** from the anchor | `results/validation/ETTh1_96_96_reconstruction_v2_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed2024.json` | `577938c9d080f64327e18504ac3e46f45a4495cfb3b404c07f407b13e1ae430a` | `[validation]` |
| `0.6712632724477633` | Corroboration 3 — independent `--cycle auto` retrain, **+2.3e-10** from the anchor | `results/validation/ETTh1_96_96_armB_auto_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed2024.json` | `23ec5927826b1e7c56cb183d59caa718b5a743bc4299253e646af92dec02c2ce` | `[validation]` |
| `0.6869550703100449` | Corroboration 4, **of the artefact reading** — an independently `head`-truncated run of the same setting, agreeing with the poisoned sidecar to **1.4e-10** | `results/validation/ETTh1_96_96_armB_smoke_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed2024.json` | `7e8bd2d55eb35c520339e72e8437faa8331be43b97ace30f133fd9736fa8e980` | `[validation]` |

Three runs trained to completion agree to under **4e-10**. The truncated pair agree with each
other and disagree with all three by **~0.0157**, about eight orders of magnitude larger.

### 2.2 The tripwire

| Value | What it is | Source path | sha256 | Split |
|---|---|---|---|---|
| **`0.6724990175677814`** | **CORRECT** three-seed mean, H = 96 | §1.1 | `4af37f4f…` + `9907d633…` + `193f4d19…` | `[validation]` |
| `0.6777296168866845` | **POISONED — the tripwire.** Reading the seed-2024 sidecar or `_summary.json` instead produces this. **Computing it means a poisoned source was read.** | `ed715742…` / `d6f77703…` | as §2 | `[validation]` |
| `0.007999357865200299` | Poisoned three-seed sd — seven times every other W-row's sd, which is how the artefact announced itself | as above | as §2 | `[validation]` |

---

## 3. Arm D — channel-count-conditional attention

### 3.1 Test-split MSE, all four horizons — the split prediction verdicts are adjudicated on

**PM ruling: §3's Arm D per-horizon MSE prediction is adjudicated on `[test]`.** Every number in
this subsection is `[test]`.

Per-seed reconstruction values are `report/horizon_sigma.md`'s twelve x86 records
(sha256 `b28c8f2376091cabb2cf5934bc6869f73cd8bd6d97730479cfcb8af30b706538`); per-seed Arm D values
are the twelve `results/runs/improved-*.json` records. Three-seed means:

| Value | What it is | Source path | sha256 | Split |
|---|---|---|---|---|
| `0.3726664257448507` | Reconstruction mean MSE, H = 96 | `report/horizon_sigma.md` | `b28c8f2376091cabb2cf5934bc6869f73cd8bd6d97730479cfcb8af30b706538` | `[test]` |
| `0.3719532075084556` | Arm D mean MSE, H = 96 | `results/runs/improved-TQNet-s{2024,2025,2026}-h96-*.json` | `a4d58b13c7128c4ea125cce5599cfaecf583edea7f10c37c42ab285bc54beb0a`, `d4175c438439f667e402ee6cfae003abeb1f32f7ea9f7ec2ddb4cadf52b38c72`, `916c65a688cde09e49c2ddb2567a00e8731077d39ea031410c32ff9e7d43301b` | `[test]` |
| `−0.0007132182363950856` | Δ (D − reconstruction), H = 96 **[derived]** | difference of the two rows above | as above | `[test]` |
| `−0.331` | Δ/σ, H = 96 → **confirmed** (inside ±1σ) **[derived]** | `−0.0007132182363950856 / 0.0021541981747125473` | as above | `[test]` |
| `0.4305209998692137` | Reconstruction mean MSE, H = 192 | `report/horizon_sigma.md` | `b28c8f23…` | `[test]` |
| `0.4278058235574245` | Arm D mean MSE, H = 192 | `results/runs/improved-TQNet-s{2024,2025,2026}-h192-*.json` | `a2d44d32e9411e1bbde9c8571bc42b74d9ccbaf95464c49af5458d26d601dd1d`, `1600d177f47c568df2c284e91f68e9df99b9c56fa9ed116707ea9ccb3fe4f055`, `f1949a204882108a59412b3ae8e96c04da94a88bc9ce603993457b83161b2892` | `[test]` |
| **`−0.0027151763117891914`** | **Δ (D − reconstruction), H = 192** **[derived]** | difference of the two rows above | as above | `[test]` |
| **`−3.241`** | **Δ/σ, H = 192 → falsified, in the favourable direction** **[derived]** | `−0.0027151763117891914 / 0.0008376947093460351` | as above | `[test]` |
| `0.47695683245890524` | Reconstruction mean MSE, H = 336 | `report/horizon_sigma.md` | `b28c8f23…` | `[test]` |
| `0.47992154301875284` | Arm D mean MSE, H = 336 | `results/runs/improved-TQNet-s{2024,2025,2026}-h336-*.json` | `bcd1377b5f7ecbadfdcd9b3161243cae862b68216136b7c4a94a4a3a298ebaf8`, `8109ee0c335a300d84517766e0307008490964b4edb55c6911d39d7f84cac79`, `12647d65b980bd261fd5deb3fd3a9cf2702a55ee36d0ff8f8539cd0c163a71e6` | `[test]` |
| `+0.0029647105598475942` | Δ, H = 336 **[derived]** | difference of the two rows above | as above | `[test]` |
| `+0.619` | Δ/σ, H = 336 → **confirmed** **[derived]** | `+0.0029647105598475942 / 0.004789584272220202` | as above | `[test]` |
| `0.5039997349691424` | Reconstruction mean MSE, H = 720 | `report/horizon_sigma.md` | `b28c8f23…` | `[test]` |
| `0.5044445613048515` | Arm D mean MSE, H = 720 | `results/runs/improved-TQNet-s{2024,2025,2026}-h720-*.json` | `4d098c16a2cf3544f9dc04e16db55e9e7573b3c873e53f9193d41cd99fe5ae2a`, `a94583a1c904b099c3cc737376ab0ad08de576ffbe1120d6c9a26a85772feecf`, `d65f38c9874a853fe77d9b7d1c165261fc737bd10922399662998069a60af3d5` | `[test]` |
| `+0.0004448263357090809` | Δ, H = 720 **[derived]** | difference of the two rows above | as above | `[test]` |
| `+0.020` | Δ/σ, H = 720 → **confirmed** **[derived]** | `+0.0004448263357090809 / 0.022769078789010144` | as above | `[test]` |

### 3.2 Validation-split MSE, all four horizons — labelled, and not a selection input

**These are `[validation]` and are recorded as §4.3 disclosure. They did not enter the selection
beyond H = 96, which is §4.1's endpoint.**

| Value | What it is | Source path | sha256 | Split |
|---|---|---|---|---|
| `0.6724990175677814` | Reconstruction mean val_MSE, H = 96 | §1.1 | `4af37f4f…` + `9907d633…` + `193f4d19…` | `[validation]` |
| `0.6805545682661754` | Arm D mean val_MSE, H = 96 | §1.2 | `5f1e3f69…` + `896415fc…` + `8933efc4…` | `[validation]` |
| `+0.008055550698394032` | Δ, H = 96 **[derived]** | difference of the two rows above | as above | `[validation]` |
| `+7.056275785710372` | Δ/σ_validation, H = 96 **[derived]** | `÷ 0.0011416150591374683` | as above | `[validation]` |
| `0.9842914948494415` | Reconstruction mean val_MSE, H = 192 | `results/validation/ETTh1_96_192_TQNet_ETTh1_ftM_sl96_pl192_cycle24_seed{2024,2025,2026}.json` | `d10c05124bc48e6a07d9e0e2236033056cc2e630ad0a67c5c3d051f781bf136b`, `aa01d2db88fed144d39315f254680aad50a02a5d0be2c01e3e07354b7f8c0b35`, `b4bdcd878f69ab03d208cfb6dbd92b3f0de3025bef151b0d1cfc495d3c8e865e` | `[validation]` |
| `0.9890891460813452` | Arm D mean val_MSE, H = 192 | `…_pl192_cycle24_seed{2024,2025,2026}_tq0ca0.json` | `ecf76286b02c7bcbeb0cf460fec3706ff667b2759b2c2a66f8bd37c041294d41`, `2e80667eacf73bdedb39f87ed76a5ec0255194e1f2a85f86b13640146b362173`, `a7e8d31bf8987f5eeb70a421191903afd53d211d1e1671469b0ed29c74ec37d4` | `[validation]` |
| **`+0.004797651231903788`** | **Δ, H = 192** **[derived]** | difference of the two rows above | as above | `[validation]` |
| **`+2.0875462099686`** | **Δ/σ_validation, H = 192** **[derived]** | `÷ 0.0022982251645466342` | as above | `[validation]` |
| `0.0022982251645466342` | σ_validation at H = 192 (reconstruction three-seed sd, n−1) | the three H = 192 reconstruction sidecars above | `d10c0512…` + `aa01d2db…` + `b4bdcd87…` | `[validation]` |
| `1.285395820044505` | Reconstruction mean val_MSE, H = 336 | `…_pl336_cycle24_seed{2024,2025,2026}.json` | `15bbf3eea02e1f4f0d3988c65e905389515e070bd3960df50be7dc631c61449b`, `952910216c0480ff536bdb70f72631c7a00c86a9ca0554a91f70129e6e720d7c`, `78eed01ac9957fe997dde4e914cc76394e00140e4ab64e3f050c5b24d1de1322` | `[validation]` |
| `1.2857378973138993` | Arm D mean val_MSE, H = 336 | `…_pl336_cycle24_seed{2024,2025,2026}_tq0ca0.json` | `d04673552b6af231268440ff49c6a8bbd6b12c17d30970ca3ff5893c9dcdec6f`, `78dcfbf379fb908b797ec7581d443787e251596657d47376e79334126f65cfbe`, `a060530f590035998acf31d2eabf53313a348dd946e316ca1a823cf8a9e3d578` | `[validation]` |
| `+0.00034207726939428085` | Δ, H = 336 **[derived]** | difference of the two rows above | as above | `[validation]` |
| `+0.08238554378617262` | Δ/σ_validation, H = 336 **[derived]** | `÷ 0.004152151623616451` | as above | `[validation]` |
| `1.5638983537808584` | Reconstruction mean val_MSE, H = 720 | `…_pl720_cycle24_seed{2024,2025,2026}.json` | `5e2ad58fab43119bdf7ab00ac6acadbeaaae5ab515445bf627091c1beb4be737`, `fa9eb58122843fbf61ed9d47fb7ee2aeef8f62d02849d4b3934ef28ae2624fa8`, `186a04c40fcf4ef0648de0a44fbab9b48fa679e30d4f97e6b670e34067f18532` | `[validation]` |
| `1.5655464148321416` | Arm D mean val_MSE, H = 720 | `…_pl720_cycle24_seed{2024,2025,2026}_tq0ca0.json` | `35aba4001632308877f0f6bd64f1e1f2df9b5a52c1739589b9cf6e1b750df6a4`, `0332c4a59b6f26413e0eca378bfb5678df8f527c0e78f0b2c153b630ee7f69af`, `85b231964919e14c8a40d992e1110f2adaa601f9dbebcac828e8b73c2e9c3033` | `[validation]` |
| `+0.0016480610512832339` | Δ, H = 720 **[derived]** | difference of the two rows above | as above | `[validation]` |
| `+0.706808182690825` | Δ/σ_validation, H = 720 **[derived]** | `÷ 0.002331694923237944` | as above | `[validation]` |

### 3.3 ⚠ The H = 192 sign reversal — stated explicitly, both numbers labelled

**At H = 192 the two splits point in opposite directions, and this must not be compressed.**

- On the **test** split, Arm D is **better** than the reconstruction by **`−0.0027151763117891914`
  `[test]`**, which is **3.241 σ_test**. This falsifies the registered ±1σ prediction — in the
  **favourable** direction. A pre-registered null broken by an *improvement*.
- On the **validation** split, Arm D is **worse** than the reconstruction by
  **`+0.004797651231903788` `[validation]`**, which is **2.088 σ_validation**.

Both statements are true, each of its own split, and neither may be reported without its label.
The verdict of record is the `[test]` one (PM ruling); the `[validation]` figure is §4.3
disclosure. **A sentence placing `−0.0027…` and `+0.0048…` side by side without both labels is
the error this sheet exists to prevent.**

### 3.4 ⚠ Arm D's abandon condition — retrospective disclosure, not an abandonment

**PM ruling, carried not re-litigated.** §3 registers Arm D's abandon condition as *"MSE degrades
by more than 1σ at any horizon."*

- Read on `[test]` — the split D3–D6 are adjudicated on — it **did not fire**. Maximum degradation
  is `+0.619` σ_test at H = 336.
- Read on `[validation]`, it **would have fired at H = 96** (`+7.056275785710372` σ_validation)
  **and at H = 192** (`+2.0875462099686` σ_validation).

**This is recorded as a retrospective disclosure. Arm D was not abandoned and this sheet does not
re-adjudicate an in-flight gate after the fact.** F6 reports it in exactly these terms.

### 3.5 Parameter counts — full precision, and the origin of the 5.65/5.66 ambiguity

`37,416` parameters are removed at **every** horizon (the count is horizon-independent); the
*percentage* falls with horizon because the denominator grows.

| Value | What it is | Source path | sha256 | Split |
|---|---|---|---|---|
| `661640` | Reconstruction `n_params`, H = 96 | `results/validation/ETTh1_96_96_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed2024.json` (this field is unaffected by the landmine, which poisons `val_MSE` only) | `ed715742bd1602e447e99fe1af2c9138f6c87a96e59911ab18bd9834124ea6e3` | `[config]` |
| `624224` | Arm D `n_params`, H = 96 | `…_seed2024_tq0ca0.json` | `5f1e3f69eb8628e31e6fc3a684c85839e7e4bd8be6aed942b52d9234cd7bd13d` | `[config]` |
| `37416` | Parameters removed, H = 96 **[derived]** | `661640 − 624224` | as above | `[config]` |
| **`5.655038994014872`** %| **Reduction at H = 96, full precision.** Rounds to `5.65%` (half-even) or `5.66%` (half-up) — **this single number is the whole of §7's 5.65-vs-5.66 disagreement.** Quote it unrounded, or quote `−5.7%` as the pre-registration does. | `100 × 37416 / 661640` | as above | `[config]` |
| `710888` / `673472` | `n_params`, H = 192 | `…_pl192_cycle24_seed2024.json` / `…_tq0ca0.json` | `d10c05124bc48e6a07d9e0e2236033056cc2e630ad0a67c5c3d051f781bf136b` / `ecf76286b02c7bcbeb0cf460fec3706ff667b2759b2c2a66f8bd37c041294d41` | `[config]` |
| `5.26327635295574` % | Reduction at H = 192 **[derived]** | `100 × 37416 / 710888` | as above | `[config]` |
| `784760` / `747344` | `n_params`, H = 336 | `…_pl336_cycle24_seed2024.json` / `…_tq0ca0.json` | `15bbf3eea02e1f4f0d3988c65e905389515e070bd3960df50be7dc631c61449b` / `d04673552b6af231268440ff49c6a8bbd6b12c17d30970ca3ff5893c9dcdec6f` | `[config]` |
| `4.767827106376472` % | Reduction at H = 336 **[derived]** | `100 × 37416 / 784760` | as above | `[config]` |
| `981752` / `944336` | `n_params`, H = 720 | `…_pl720_cycle24_seed2024.json` / `…_tq0ca0.json` | `5e2ad58fab43119bdf7ab00ac6acadbeaaae5ab515445bf627091c1beb4be737` / `35aba4001632308877f0f6bd64f1e1f2df9b5a52c1739589b9cf6e1b750df6a4` | `[config]` |
| **`3.811145788345733`** % | **Reduction at H = 720** — the low end of the "5.66% → 3.81%" range **[derived]** | `100 × 37416 / 981752` | as above | `[config]` |

**Decomposition of 37,416** (from `report/prereg-improvement.md`'s single `## Amendments` block,
sha256 `d5ee1bd6130e21d8694f290e106d8f2c6a3b3b03983332de64e6d1bc6f70611f`, `[config]`):
`37248` (channel-attention block alone) `+ 168` (Temporal Query alone) `= 37416` (both removed).
The figure `37248` appearing anywhere as Arm D's total is the corrected error, not the count.

### 3.6 The criterion statistic

| Value | What it is | Source path | sha256 | Split |
|---|---|---|---|---|
| **`0.3110131176996824`** | `mean_abs_offdiag_pearson_correlation` on ETTh1, C = 7, rows `[0, 8640)` | `channel_criterion_check.log` | `e2fed6c9197075df253d065ba8383c797260f2ec083a037850508859696f748a` | `[training]` |
| `0.3` | `CHANNEL_CORR_DROP_THRESHOLD`, fixed in advance (Cohen 1988 "medium effect") | `channel_criterion_check.log` | `e2fed6c9197075df253d065ba8383c797260f2ec083a037850508859696f748a` | `[config]` |
| `0.0110131176996824` | Margin above threshold **[derived]** — **narrow, and F6 should say so** | `0.3110131176996824 − 0.3` | as above | `[training]` |
| `drop` (`use_tq=0`, `channel_aggre=0`) | The criterion's decision — the registered prediction was that it fires, and it fired | `channel_criterion_check.log`; `armD_runs.log` | `e2fed6c9…`; `32a0ff132caba3705e0c19096e10917a06111814660b32c4cc0af8dda0241bc7` | `[training]` |

### 3.7 The twelve wall-clock timings

All twelve are `[config]`. **They do not support a speed claim** — no reconstruction wall-clock was
ever recorded, at any horizon, in any file, so D7 is `unevaluable` for want of a baseline. They are
additionally incoherent as a compute measure: H = 96 averages `242.33…` s while H = 336 averages
`97.66…` s, which reflects early-stopping epoch counts and machine load, not architecture cost.

Source for all twelve: `armD_wallclock.log`, sha256
`7ef94b9e64dcff638d8baffa9770c42b9e0cb5fd3ede0175fed439ffff8ec0d9`, split `[config]`.

| H | seed 2024 | seed 2025 | seed 2026 |
|---|---|---|---|
| 96 | `258s` | `234s` | `235s` |
| 192 | `250s` | `151s` | `180s` |
| 336 | `94s` | `107s` | `92s` |
| 720 | `99s` | `99s` | `166s` |

---

## 4. Arm B — the cycle estimator and the W-curve

### 4.1 W = 24 by both methods, independently

| Value | What it is | Source path | sha256 | Split |
|---|---|---|---|---|
| **`24`** | ACF estimate — largest **local** maximum over lags `[2, 400]`, channel-mean, rows `[0, 8640)` | `report/cycle_estimate.md` | `34ed05b46a4775f37a3a9b67df1574602a7f74c31ebe636ae0bef89d1b0b235c` | `[training]` |
| `0.8851540915365546` | `ac[24]`, the ACF peak value | `report/cycle_estimate.md`; independently in `w_curve.log` | `34ed05b4…`; `2909b511665bfa1e3865af807bfb8ef62a84cc2ed7a1bdc2ed9e1fabd0bd45d4` | `[training]` |
| **`24`** | Periodogram estimate — argmax power over periods `[2, 400]`, same series, same rows | `report/cycle_estimate.md` | `34ed05b46a4775f37a3a9b67df1574602a7f74c31ebe636ae0bef89d1b0b235c` | `[training]` |
| `19641471.641798608` | Periodogram power at period 24 | `report/cycle_estimate.md`; `w_curve.log` | `34ed05b4…`; `2909b511…` | `[training]` |
| `True` | `agree` — **strict integer equality**; harmonics and subharmonics never count as agreement | `report/cycle_estimate.md`; `w_curve.log` | `34ed05b4…`; `2909b511…` | `[training]` |
| `estimated` | `cycle_source` in the resolved config of the end-to-end `--cycle auto` run | `results/validation/ETTh1_96_96_armB_auto_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed2024.json` | `23ec5927826b1e7c56cb183d59caa718b5a743bc4299253e646af92dec02c2ce` | `[config]` |

**The abandon condition ("ACF and periodogram disagree") did not fire.**

### 4.2 The eight-point W-curve

Three-seed means, `[validation]`, per W. Each mean is `statistics.mean` over the three sidecars
`results/validation/ETTh1_96_96_TQNet_ETTh1_ftM_sl96_pl96_cycle<W>_seed<S>.json`. All eight means
were **recomputed from the sidecars for this sheet** and reproduce `report/w_curve.md` exactly.

| W | Mean val_MSE | seed-2024 sha256 | seed-2025 sha256 | seed-2026 sha256 | Split |
|---|---|---|---|---|---|
| 6 | **`0.6714039036970284`** ← min | `2c60d00c88f4944a427c96427f2ad4c2874b981642624283c5f850f3fe748bf9` | `23a9f46536b9455c9e2a3a0818b1b2211b074dde238ae1d75d4fb478f1537dff` | `2e8ab6951c0bdce7867718644710228fab2ec410953b61a33ca9a254924f1979` | `[validation]` |
| 8 | `0.6717737364594718` | `402c99d0adef28914287ebea41c554e2dcae3b83f9c8aabe353e6f656d27150f` | `26c6f9084c63df42f480027f0fc961921fa91e677684e05c3ecccff1eaa8d05b` | `702f03b48d2768421c1b350abd1ef8381b25e6243bac23219ef90d8f59a91678` | `[validation]` |
| 12 | **`0.672930702161326`** ← max | `ad1eabf0fa04fcd5ec11d31a4ea0c7b920caea5abc948e38e352424f05694a0f` | `9e86323697c191360abe68b437681636d2f29086c70f5751801af60da0c67f1e` | `779d312c4d885d534a540ea3c0728bfe6cc1d0c053176a25ff199d6fb0ce0ec6` | `[validation]` |
| 23 | `0.6720198546674865` | `7548cf197169bff72c810177033311309a138f1f7e3276c55b41a4cd15415291` | `c0154d36baa9c5cf37390588e4563082f7be89d77824e2034867bce3accb6385` | `0673b759dd64840dea104228731f728d8f2d121eb02efbd2f39928598f19aa85` | `[validation]` |
| **24** | **`0.6724990175677814`** | seed-2024 value from `validation_metrics.log` `4af37f4f…` — **not** the sidecar; see §2 | `9907d633e153b6d667b8f9f4007091e86b1160313256d64b21e2e738be650fbb` | `193f4d1976563ce99ca1d467596df85908e1a64c6309da24b2f41daa42c60094` | `[validation]` |
| 25 | `0.6728342289621677` | `420c90e89731f2a8c659c02b5a8ed36962e422c68abbdf431f7fa03eb03677b1` | `0e62e2e07948adc559878c5217f25d5c105f690c7e17db6df58125a60f19a0bc` | `805c18c8840cd97f8f5762bf753b580a8c245c2bc6e808ff6247753eb7b9966f` | `[validation]` |
| 48 | `0.672529158700914` | `32a8870a3ec57c5649b0cb7bbab1ebb126bfc9050e32e341701c47750a161717` | `3b84e9023f37cb6d23bcb9a34c9e2352e549414a89964c339081c7b138c921fb` | `d1de62d4278ea706b4eafc3ad9d2e3e3f4f7a84894cf80fad5cc071990a216b9` | `[validation]` |
| 168 | `0.6726663638283168` | `ad4b47a8856ea746810f1a14ab0ebc8694c99bd5bcc344d1bc6acedac714b55c` | `32dc50bb03f8076acc20dc74f9990ee7a9000fec15148dc213753764c706dca7` | `5c40d68ca64c2751484c515f5add9446a1b5cd2a0fbb85f96e45521a2a63ae4d` | `[validation]` |

### 4.3 The spread and the flatness verdict

| Value | What it is | Source path | sha256 | Split |
|---|---|---|---|---|
| **`0.0015267984642975962`** | **Spread across the eight per-W means** — max (W = 12) − min (W = 6) **[derived]** | `0.672930702161326 − 0.6714039036970284`, over the 24 sidecars in §4.2 | all 24 hashes in §4.2 | `[validation]` |
| `0.0022832301182749365` | 2 × σ_validation, the registered ±2σ band | §1.1 | `4af37f4f…` + `9907d633…` + `193f4d19…` | `[validation]` |
| `0.6687010880231152` | Spread as a fraction of 2σ → **every pair of the eight is inside ±2σ: the curve is flat, prediction confirmed** **[derived]** | `0.0015267984642975962 / 0.0022832301182749365` | as above | `[validation]` |
| `0.006325713189656135` | **SUPERSEDED** spread, computed with the poisoned W = 24 cell — `2.77 × 2σ`, i.e. the opposite verdict. **Not to be quoted.** Retained to show the landmine flips this result. | `report/w_curve.md`, superseded row | `0e320a97388ac59997b5389a349236bf42f29c4811417a1733b64c5ee1e56e89` | `[validation]` |

**The corrected and poisoned spreads fall on opposite sides of the 2σ line.** This is the single
clearest demonstration of why `_summary.json` and the seed-2024 sidecar are barred.

### 4.4 The 2.3e-10 agreement — Arm B's §3 prediction

| Value | What it is | Source path | sha256 | Split |
|---|---|---|---|---|
| `0.6712632724477633` | `--cycle auto` end-to-end run, H = 96, seed 2024 | `results/validation/ETTh1_96_96_armB_auto_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed2024.json` | `23ec5927826b1e7c56cb183d59caa718b5a743bc4299253e646af92dec02c2ce` | `[validation]` |
| `0.6712632722155959` | Reconstruction anchor, same setting | `results/validation/validation_metrics.log` | `4af37f4fe4be4fcc01fb17e30a15db0b06c3b390fc764d22d205e0b4bb9b573d` | `[validation]` |
| **`2.321674e-10`** | **Difference — inside the registered ±0.0005 band by ~6.5 orders of magnitude. Prediction confirmed.** **[derived]** | difference of the two rows above | as above | `[validation]` |
| `0` (exact) | Difference on the shared W = 24 checkpoints, which Arm B re-evaluates rather than retrains | §1.4 | as §1.1 | `[validation]` |

---

## 5. Per-horizon σ — two per horizon, all four horizons

**Two σ per horizon: one for MSE, one for MAE.** All from `report/horizon_sigma.md`, sha256
`b28c8f2376091cabb2cf5934bc6869f73cd8bd6d97730479cfcb8af30b706538`, computed over the twelve x86
reconstruction records (`git_commit == "9663bcd"`), sample sd n−1, split `[test]` throughout.

| H | Metric | Mean | **σ (n−1)** | Split |
|---|---|---|---|---|
| 96 | MSE | `0.3726664257448507` | **`0.0021541981747125473`** | `[test]` |
| 96 | MAE | `0.3931172899123145` | **`0.0003752412844557897`** | `[test]` |
| 192 | MSE | `0.4305209998692137` | **`0.0008376947093460351`** | `[test]` |
| 192 | MAE | `0.4246375970880016` | **`0.0015920913669081958`** | `[test]` |
| 336 | MSE | `0.47695683245890524` | **`0.004789584272220202`** | `[test]` |
| 336 | MAE | `0.44605284372176857` | **`0.001368805006375168`** | `[test]` |
| 720 | MSE | `0.5039997349691424` | **`0.022769078789010144`** | `[test]` |
| 720 | MAE | `0.4798055414967954` | **`0.01486937531963135`** | `[test]` |

**σ is not monotonic in the horizon.** σ_MSE falls from H = 96 to H = 192 (`0.0021541981747125473`
→ `0.0008376947093460351`) before rising through H = 336 to H = 720. A model assuming monotonic
growth, or reusing the H = 96 value everywhere, is wrong by roughly an order of magnitude at
H = 720. This is why Arm D's H = 192 falsification clears 1σ: it is measured against the
**smallest** of the four σ.

Per-seed MSE and MAE values behind every mean and σ above are in `report/horizon_sigma.md`'s
per-horizon tables, same sha256, same split.

### 5.1 split_hash by horizon (standing order 12)

| H | `split_hash` | Source path | sha256 | Split |
|---|---|---|---|---|
| 96 | `b66ee6b47e2b2eb8` | `report/horizon_sigma.md` | `b28c8f2376091cabb2cf5934bc6869f73cd8bd6d97730479cfcb8af30b706538` | `[config]` |
| 192 | `5b9f41f467356285` | `report/horizon_sigma.md` | `b28c8f23…` | `[config]` |
| 336 | `a5bcaa4090739908` | `report/horizon_sigma.md` | `b28c8f23…` | `[config]` |
| 720 | `17f9f51a6d81e0a2` | `report/horizon_sigma.md` | `b28c8f23…` | `[config]` |

Every arm asserts `b66ee6b47e2b2eb8` at L = H = 96, verified by decoding each sidecar (D9′).

---

## 6. Reproduction — the target cell, both records, all four metrics

**Target cell: ETTh1, multivariate, L = 96 → H = 96, seed 2024, 2,785 test windows, z-scored.**
Two records of the same cell exist. Both are `[test]`.

- **arm64** — `git_commit 3894e4f`, recorded 2026-07-30, produced on a machine that no longer
  exists. **Excluded from `report/horizon_sigma.md`'s σ** so that platform is not conflated with
  effect at a σ of ~0.002.
- **x86** — `git_commit 9663bcd`, recorded 2026-08-09. This is the re-baseline record and the one
  F5's reconstruction column uses.

| Metric | arm64 value | x86 value | **Δ (x86 − arm64)** | Split |
|---|---|---|---|---|
| MSE | `0.37104994617301906` | `0.37104994668966473` | **`5.166456706895417e-10`** | `[test]` |
| MAE | `0.39272399040064576` | `0.3927239906699211` | **`2.692753242605761e-10`** | `[test]` |
| RMSE | `0.6091386920669373` | `0.6091386924910162` | **`4.240788831211262e-10`** | `[test]` |
| MdAE | `0.25126033974811435` | `0.2512602503411472` | **`−8.940696716308594e-08`** | `[test]` |

| Record | Source path | sha256 | Split |
|---|---|---|---|
| arm64 | `results/runs/reconstruction-TQNet-s2024-h96-1785426343196465000.json` | `791f79d82c3be794ca2ee1753a744466efa4b889d0432fa7e20d87c819b0ddc0` | `[test]` |
| x86 | `results/runs/reconstruction-TQNet-s2024-h96-1786272620501519500.json` | `ee3aadfbcd36c71fb4d6484b8d4f7fd24a115cc076dcd0a657bfba9491910ed2` | `[test]` |

Both records assert `split_hash = b66ee6b47e2b2eb8`, `n_windows = 2785`, `n_points = 1871520`,
`n_params = 661640`, `data_sha256 = f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066`
`[config]`. **The two architectures agree to ~5e-10 on MSE** — three orders of magnitude below the
seed σ and eight below the reproduction gap, which is why excluding arm64 from σ costs nothing.

> **MdAE's Δ is ~170× the other three** (`−8.9e-08` vs `~5e-10`). MdAE is an order statistic — a
> selected element, not a reduction — so it moves in discrete jumps and is not expected to track
> floating-point reduction order. Recorded so a reader does not read it as an anomaly.

### 6.1 The reproduction gap against the authors' own run

| Value | What it is | Source path | sha256 | Split |
|---|---|---|---|---|
| `0.3710499405860901` | Authors' own run, MSE — as carried in both our records' `upstream_metric_agreement` block | `results/runs/reconstruction-TQNet-s2024-h96-1786272620501519500.json` | `ee3aadfbcd36c71fb4d6484b8d4f7fd24a115cc076dcd0a657bfba9491910ed2` | `[test]` |
| `0.39272406697273254` | Authors' own run, MAE (as compared against the x86 record) | same | `ee3aadfb…` | `[test]` |
| `0.6091386675834656` | Authors' own run, RMSE | same | `ee3aadfb…` | `[test]` |
| `6.103574645699439e-09` | MSE `abs_diff`, x86 vs upstream; `agree: true` | same | `ee3aadfb…` | `[test]` |
| `1.644946940581251e-08` | MSE `rel_diff`, x86 vs upstream | same | `ee3aadfb…` | `[test]` |
| `7.630281145809548e-08` | MAE `abs_diff`, x86 vs upstream; `agree: true` | same | `ee3aadfb…` | `[test]` |
| `2.4907550577601967e-08` | RMSE `abs_diff`, x86 vs upstream; `agree: true` | same | `ee3aadfb…` | `[test]` |

> **Do not quote `report/results.md`'s reproduction-gap table as source.** It rounds to six
> decimals (`0.371217`, `0.371050`, `−0.000167`, `−0.045%`, `0.17x`) and compares against the
> **paper's printed** figure `0.371217`, which is a different quantity from the
> `upstream_metric_agreement` block's `0.3710499405860901`. The rounded table is fine as narrative;
> the rows above are what F1 quotes. `report/results.md` sha256
> `4f38e2defcf03963345c9d104b9ad1aefc5b1ffbb51be878d94e71fdc966af47`.

### 6.2 Baseline and paper context

| Value | What it is | Source path | sha256 | Split |
|---|---|---|---|---|
| `0.512225108181955` | Seasonal-naive (period 24) MSE, same 2,785 windows | `results/runs/baseline-seasonal_naive_24-sna-h96-1785426202205168000.json` | *(unhashed — see §9)* | `[test]` |
| `0.43330271118779884` | Seasonal-naive MAE | same | *(unhashed — see §9)* | `[test]` |
| `0.7156990346381327` | Seasonal-naive RMSE | same | *(unhashed — see §9)* | `[test]` |
| `0.26066610775972177` | Seasonal-naive MdAE | same | *(unhashed — see §9)* | `[test]` |

**Paper Table 5 figures (`0.371` / `0.393` and the ten-model comparison set) are transcriptions
from `PTQNet.pdf` p. 15, not result files.** They are citable as *the paper's printed values* with
a page reference under D17, but they are not measurements of this study and no Δ in this sheet is
computed against them.

---

## 7. Arm A — damped-trend instance normalisation

**Arm A did not reach the §4.1 endpoint** (seed 2024 only; φ = 0.9 and φ = 0.95 never run; no φ
ever frozen). It is not ranked. What exists:

| Value | What it is | Source path | sha256 | Split |
|---|---|---|---|---|
| `0.6952801971213933` | Arm A val_MSE, H = 96, seed 2024, **φ = 0.8** | `results/validation/ETTh1_96_96_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed2024_dphi0.8.json` | `6d79c1e20bd7f3bf7fdf673c80ffef153280e785ffea40ac9d7367d1c24be75e` | `[validation]` |
| `0.9491608951697321` | Arm A val_MSE, H = 96, seed 2024, **φ = 1.0** | `results/validation/ETTh1_96_96_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed2024_dphi1.json` | `86b90be10d9287f75767fcd656f385448d13d5aa62aa1476df7e9183893b930f` | `[validation]` |
| `0.6712632722155959` | Reconstruction anchor, seed 2024, for both deltas | `results/validation/validation_metrics.log` | `4af37f4fe4be4fcc01fb17e30a15db0b06c3b390fc764d22d205e0b4bb9b573d` | `[validation]` |
| `+0.024016924905797432` | Δ at φ = 0.8 **[derived]** | `0.6952801971213933 − 0.6712632722155959` | `6d79c1e2…` + `4af37f4f…` | `[validation]` |
| **`+21.037673525385248`** | **Δ/σ_validation at φ = 0.8 — the registered abandon condition fired** **[derived]** | `÷ 0.0011416150591374683` | as above | `[validation]` |
| `+0.2778976229541362` | Δ at φ = 1.0 **[derived]** | `0.9491608951697321 − 0.6712632722155959` | `86b90be1…` + `4af37f4f…` | `[validation]` |
| **`+243.42498001392693`** | **Δ/σ_validation at φ = 1.0** **[derived]** | `÷ 0.0011416150591374683` | as above | `[validation]` |
| `661640` | Arm A `n_params` at both φ — the mechanism changes normalisation, not parameter count | both `dphi` sidecars above | `6d79c1e2…`, `86b90be1…` | `[config]` |
| `12.0` | Δ at φ = 0.8 as a multiple of §3's registered ±0.002 null band **[derived]** | `0.024016924905797432 / 0.002` | as above | `[validation]` |
| `139` | Δ at φ = 1.0 as a multiple of the same band **[derived]** | `0.2778976229541362 / 0.002` | as above | `[validation]` |

**The abandonment is robust to the anchor question** (§7l item 4). Recomputed against the epoch-3
value, φ = 0.8 reads `+7.29` σ_validation; against `reconstruction_v2`'s
`0.6712632718563285` `[validation]` it reads `+21.04` — identical to the anchor at the precision
that matters. All three anchors clear the 1σ abandon threshold by one to two orders of magnitude.
**The prose figures explaining *why* the arm fails are withheld — see below.**

---

## Withheld under T15′

**These numbers exist in prose and in no result file. Under standing order T15′ — *"a number that
cannot be traced to a named result file, by path and sha256, does not get printed"* — they are
barred from the report until an artifact exists. This job did not compute them; that is a separate
job.**

### W1 — λ̂ = −0.34

| Field | Content |
|---|---|
| **Value as it appears in prose** | `λ̂ = −0.34` (mean-reverting) |
| **Where the prose lives** | `STAGE2_WORKPLAN_2026-08-09.md` §-1 arm table, Arm A row (sha256 `cfbd14a1cbe445b0845d78c7495dfe644a5e439955c6f2fe27f8cbd4bdcbbe77`); `PM_HANDOFF_2026-08-10_1730.md` Arm A row (sha256 `2d032c6d12b788b09515552f74b2dc28b521bddf084a93a9251fd61789d5305f`). Both narrative. |
| **What it claims** | The AR(1) coefficient of the in-window linear-fit slope on the training split is negative — the slope *anti-predicts*, so projecting it forward is worse than projecting nothing. |
| **Why it is not traceable** | No log, sidecar or run record contains it. `damped_trend_measure.py` computes numerical-identity checks, not this quantity. Searched across `results/`, all logs, and all `report/*.md`. |
| **What artifact would discharge it** | A result file recording the slope-series AR(1) fit over training rows `[0, 8640)` — estimator, row range, per-channel and aggregate λ̂, and the split-hash assertion — written to `results/` with a sha256, `[training]`. |
| **What is blocked** | Arm A's **mechanism-level explanation** in F4 (method) and F6 (discussion). The *qualitative* claim — the in-window slope carries little within-window variance and is mean-reverting — may still be stated, supported by the measured degradation in §7. The **number** may not. |

### W2 — median 4.0% within-window variance explained

| Field | Content |
|---|---|
| **Value as it appears in prose** | `median 4.0%` of within-window variance explained by the in-window linear fit |
| **Where the prose lives** | Same two documents and same two rows as W1; same sha256s. Both narrative. |
| **What it claims** | Across training windows, the linear trend accounts for a median 4.0% of variance — i.e. almost none — which is the mechanism-level reason the damped-trend projection degrades the model. |
| **Why it is not traceable** | As W1. No result file carries an R² distribution over windows. |
| **What artifact would discharge it** | A result file recording per-window R² of the in-window linear fit over training rows `[0, 8640)` — with the median, the distribution or quantiles, window definition, and the split-hash assertion — written to `results/` with a sha256, `[training]`. |
| **What is blocked** | As W1: Arm A's mechanism explanation in F4 and F6. |

**Both figures would discharge on the same run**, over the same rows, with the same window
definition — one job, two artifacts. Neither is required for any *verdict* in this study: Arm A's
abandonment rests on the measured `+21.04σ` / `+243.43σ` degradations in §7, which are fully
traceable. What is blocked is only the *explanation of the mechanism*, not the finding.

---

## 8. Documents that disagree with the files hashed here

Standing order 5 — **the run wins.** In each case the value frozen above is the file's, and the
document needs correcting.

| # | Document | What it says | What the files say | Frozen value |
|---|---|---|---|---|
| 1 | `report/w_curve.md` (`0e320a97…`) | `_summary.json` sha256 = `a8be034cec651dbc78f810f8bcc3cdeaa35fd16c863bfdc6f27ac825c44b03be` | **`d6f77703d135cea199a028b496ed210f2e1bfa1a341f03589b1f68b27dd3dbbc`** — the file was regenerated after w_curve.md was written | `d6f77703…`. **Immaterial to every number here**, since no number is sourced from `_summary.json`; recorded so the two files do not appear to disagree unexplained. |
| 2 | `report/w_curve.md` (`0e320a97…`) | `2σ = 0.0022832301182749366` | **`0.0022832301182749365`** by every accumulation route (`sd*2`, `sd+sd`, `numpy`) | `…365`. ~1e-19 relative; changes no verdict. Already raised by J-16 §5.3 item 4. |
| 3 | `report/w_curve.md` (`0e320a97…`) | sd column truncated to 16 characters, e.g. W = 6 sd `0.0011505598846829` | **`0.0011505598846829254`** from the three sidecars | The sidecar-derived value. **The eight *means* are unaffected** and reproduce exactly. |
| 4 | `report/results.md` (`4f38e2de…`) | Reproduction table rounded to six decimals | Full precision in `upstream_metric_agreement`, §6.1 | §6.1's values. |
| 5 | `PM_STATUS_2026-08-10_1840.md` (`ffe880f9…`) | `HEAD = 3d807b0 = origin/main, nothing committed since` | `HEAD = 63a902d`; `3d807b0` is an ancestor two commits back (`aee6229`, `63a902d`) | Status doc naturally superseded by Amitay's GitHub Desktop commit. No number here depends on it. |

**No disagreement above touches a reported value.** Items 1–3 are metadata and precision; 4 is
presentation; 5 is repository state.

---

## 9. Coverage note — one row without a sha256

`results/runs/baseline-seasonal_naive_24-sna-h96-1785426202205168000.json` (§6.2, four
seasonal-naive metrics) was **not** in J-16 §5.1's hash inventory and is **not** among the 61 files
in this job's criterion-9 before/after inventory (51 `results/validation/*.json` + the log + 9
`report/*.md`). Its four values are recorded above with path and split, and are marked
*(unhashed)* rather than dropped silently.

**Under T15′ these four values are not yet quotable.** Discharging is trivial — one `sha256sum` of
a file that already exists — but it is outside this job's declared inventory, so it is raised to
the PM rather than performed here. The `27.6%` improvement over the seasonal-naive baseline in
`report/results.md` depends on this row and is blocked with it.

---

## 10. Frozen inventory — the 61 files backing this sheet

Every file hashed before and after writing this sheet, with an empty diff (criterion 9). The
inventory is the 51 `results/validation/*.json` (50 sidecars + `_summary.json`),
`results/validation/validation_metrics.log`, and all 9 `report/*.md`.

| File | sha256 |
|---|---|
| `report/audit.md` | `fbab40b3df90f901b2f5e0aa209a7629821f12d660966a0fc6c36b8e539db58f` |
| `report/cycle_estimate.md` | `34ed05b46a4775f37a3a9b67df1574602a7f74c31ebe636ae0bef89d1b0b235c` |
| `report/horizon_sigma.md` | `b28c8f2376091cabb2cf5934bc6869f73cd8bd6d97730479cfcb8af30b706538` |
| `report/metrics.md` | `3feeb98c3751b3abca4bd2a9bea1c9fd94a4a996be468487a5cc4f31ef1408af` |
| `report/paper_code_divergences.md` | `c57de205554646897ab7aaaa0bc0e25dfa6e146176cbfa2da7e2b6b468675a00` |
| `report/prereg-improvement.md` | `d5ee1bd6130e21d8694f290e106d8f2c6a3b3b03983332de64e6d1bc6f70611f` |
| `report/results.md` | `4f38e2defcf03963345c9d104b9ad1aefc5b1ffbb51be878d94e71fdc966af47` |
| `report/selection.md` | `94335686f6417aff9e9e17b844f42fc0eb86d0db4443b8994dd88ca75abe5e31` |
| `report/w_curve.md` | `0e320a97388ac59997b5389a349236bf42f29c4811417a1733b64c5ee1e56e89` |
| `results/validation/validation_metrics.log` | `4af37f4fe4be4fcc01fb17e30a15db0b06c3b390fc764d22d205e0b4bb9b573d` |
| `results/validation/_summary.json` | `d6f77703d135cea199a028b496ed210f2e1bfa1a341f03589b1f68b27dd3dbbc` **(superseded — barred)** |

The 50 sidecar hashes are given inline at the point of use (§1, §2, §3, §4, §7) rather than
duplicated here, so that every number sits beside the hash of the file it came from.

**Files hashed but outside the 61-file inventory**, cited above and recorded for completeness:
`results/runs/reconstruction-TQNet-s2024-h96-1785426343196465000.json` `791f79d8…`;
`results/runs/reconstruction-TQNet-s2024-h96-1786272620501519500.json` `ee3aadfb…`;
the twelve `results/runs/improved-*.json` (hashes in §3.1);
`armD_wallclock.log` `7ef94b9e…`; `armD_runs.log` `32a0ff13…`; `armA_phi.log` `1915f800…`;
`w_curve.log` `2909b511…`; `channel_criterion_check.log` `e2fed6c9…`;
`TQNet/checkpoints/…/checkpoint.pth` `c5d0f7bb…` (gitignored; hash only).

---

## 11. For J-24

**Rule on whether `results/validation/_summary.json` ships in the final submission.**

The case for excluding it: it is a 201,494-byte derived roll-up that carries the poisoned
`0.6869550701723053` `[validation]` and does not contain the true `0.6712632722155959`
`[validation]`. It looks authoritative, it is the most likely file for a reader to open first, and
sourcing any number from it reproduces the wrong selection. The case for including it: this
project's norm is that superseded artefacts are retained and marked, never deleted, and the same
norm already retains the broken checkpoint.

**Not decided here.** If it ships, it needs a header or a companion note marking it superseded.
The same question applies to `tools/validation_metrics.py`, which regenerates the sidecars from
whatever checkpoint is on disk and is the path that created the landmine — it is now committed
(`aee6229`) and will ship by default.

---

*End of J-17. This file froze the numbers for F1–F7. It wrote no other file, ran no training, ran
no writing `git` command, invoked no evaluator, and sourced no number from `_summary.json`.*

---

# Amendments

Append-only, per this project's norm for frozen documents (`report/prereg-improvement.md`
carries its correction the same way). **No value in §§0–11 is altered or deleted by anything
below; every entry is an addition.** Amendments to a sole-writer file are raised for the
writer's approval, not adopted unilaterally — this block is written by the arm64 track and
needs Amitay's sign-off.

### 2026-08-10 — four provenance corrections

**Explicitly out of scope: no measurement from the arm64 platform replication
(`report/platform_arm64.md`) is introduced here, and no arm's value is restated from it.**
Substituting arm64 measurements into this sheet would make them reported results, which is
the post-hoc move `report/prereg-improvement.md` §4.5 forbids, and would reverse in prose the
exclusion `tools/horizon_sigma.py` enforces in code. All four entries below concern
**provenance and digests only**.

#### A1 — §6.1 labels one of our own numbers as the authors' run

§6.1's first row reads *"Authors' own run, MSE — `0.3710499405860901`"*, sourced from the x86
record's `upstream_metric_agreement` block, and §6.1 computes a reproduction gap of
`6.103574645699439e-09` from it.

**That value is not the authors' run.** Inside `upstream_metric_agreement`, `upstream` means
*upstream's metric code* — TQNet's `utils/metrics.py` — not *upstream's run*. The block exists
to cross-check two scorings of **one array, ours**, which is why its sibling key is `ours`.
`0.3710499405860901` is byte-identical to the `upstream_mse` field of our own
`TQNet/results/ETTh1_96_96_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed2024/metrics.json`.

The authors' published figures for this cell:

| Value | What it is | Source path | sha256 | Split |
|---|---|---|---|---|
| **`0.3712165653705597`** | **Authors' own run, MSE**, ETTh1 L = 96 → H = 96, seed 2024 | `TQNet/result_authors_reference.txt` | `5264c43d52262f1ac267637f209ab54c5ffe8472a1bdcd75eaf41c78fe2f3040` | `[test]` |
| **`0.3928201496601105`** | **Authors' own run, MAE**, same cell | `TQNet/result_authors_reference.txt` | `5264c43d…` | `[test]` |
| **`−0.0001666186808949588`** | **The reproduction gap**, ours − authors' **[derived]** | `0.37104994668966473 − 0.3712165653705597` | `ee3aadfb…` + `5264c43d…` | `[test]` |
| `−0.04488...` % | The same gap, relative **[derived]** | `100 × −0.0001666186808949588 / 0.3712165653705597` | as above | `[test]` |
| **`0.1666`** | **Gap in units of the paper's own H = 96 seed sd (0.001, Table 9 p. 18)** **[derived]** | `0.0001666186808949588 / 0.001` | as above | `[test]` |

**Why this is not a precision quibble.** §6.1 as written reports a reproduction five orders of
magnitude better than the real one — a gap so small it would read as a copied number rather
than an independent run — and in doing so discards the project's strongest result, which is
that an independent data path and an independent metric land **0.167σ** from the authors'
published figure. F-authors quote the rows above. `report/F1-F3.md` §F3.1 already does.

#### A2 — §6's stated reason for excluding the arm64 record is false

§6 describes the arm64 record as *"produced on a machine that no longer exists"*, and
`docs/STATUS.md` G2 built on the same premise. **The machine exists.** It is the macOS/arm64
machine this amendment was written on; its five Stage-1 checkpoints and all five sets of
`pred.npy`/`true.npy` have been on disk untouched since 2026-07-30 18:44–18:51. They were
invisible to every audit conducted from git because `.gitignore` excludes `TQNet/results/`
wholesale, not because they were lost.

Two of the previously-unrecorded runs now have committed records:

| Value | What it is | Source path | sha256 | Split |
|---|---|---|---|---|
| `0.3717761290076582` | ETTh1 ablation, no-TQ (self-attention), seed 2024 | `results/runs/reconstruction-TQNet-s2024-h96-1786379059318751000.json` | `b4a47af0fc50cb7a343255451c4b5d222dc39a9876a2e2563f4e4bed0cc65064` | `[test]` |
| `0.37096275348328595` | ETTh1 ablation, pure MLP, seed 2024 | `results/runs/reconstruction-TQNet-s2024-h96-1786379059385966000.json` | `37adae9a1ef493bff5796028155560b0962f075e0fb87b8aab348408f1886856` | `[test]` |

**The exclusion itself stands, and this amendment does not disturb it.** Mixing architectures
under a σ of ~0.002 is bad practice whether or not the second machine is reachable, so
`tools/horizon_sigma.py`'s rule (`git_commit == "9663bcd"`) and §5's σ are unchanged. Only the
*reason* recorded in §6 needs correcting: the record is excluded by protocol, not by loss.

#### A3 — §2 and §10 state a machine-local hash as though it were universal

§2's third row and §10's closing list both record
`TQNet/checkpoints/ETTh1_96_96_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed2024/checkpoint.pth`
with sha256 `c5d0f7bbc057d48608c15e60a4872712b363a5fa4c12238ba77fb16860773ca2`, the retained
epoch-3 artefact. On the arm64 clone that same path holds
`44759d7a224071280f4319e6fbbe88d878e308989976b00612989ef3a789a20c` — the original Stage-1
weights, which were never overwritten because the incident happened on the x86 clone.

Neither hash is wrong; the path is **gitignored and machine-local**, so it does not have one
hash. A reader verifying §2 on this clone gets a mismatch and would reasonably conclude the
sheet is broken. Both values are recorded here with the clone each belongs to:

| Clone | sha256 of that path | What it is |
|---|---|---|
| x86 (Amitay) | `c5d0f7bbc057d48608c15e60a4872712b363a5fa4c12238ba77fb16860773ca2` | epoch-3 artefact, retained as §7j / F6 evidence |
| arm64 | `44759d7a224071280f4319e6fbbe88d878e308989976b00612989ef3a789a20c` | original Stage-1 weights, never overwritten |

Every other hash in this sheet is of a **committed** file and is therefore clone-independent.
This row is the only exception, and it is the only one that needed qualifying.

#### A4 — §9's missing digest, discharged

§9 records that `results/runs/baseline-seasonal_naive_24-sna-h96-1785426202205168000.json`
carries no sha256, and bars §6.2's four seasonal-naive values under T15′ — which also blocks
the `27.6%`-over-baseline figure in `report/results.md`.

| Value | What it is | Source path | sha256 | Split |
|---|---|---|---|---|
| — | The digest §9 asked for. Read-only `sha256sum` of a file already committed; no measurement, no evaluator, nothing re-run. | `results/runs/baseline-seasonal_naive_24-sna-h96-1785426202205168000.json` | **`85951e46b25653ae67a53c4a1f4f990b397fda24b204c52da61a572afc1f1948`** | `[test]` |

**With this, §6.2's four values satisfy T15′ and become quotable**, and the `27.6%` figure is
unblocked. §6.2's values themselves are unchanged; only the missing column is supplied.

#### Also raised, and deliberately not amended here

§6.2 attributes paper Table 5 to *"`PTQNet.pdf` p. 15"*. PTQNet is a different paper — Xun et
al., *Information Processing & Management* 63(7):104785, 2026 — which is paywalled and unread
(`report/prereg-improvement.md` §3, Arm B). Table 5 is in `files/project/TQnet.pdf` p. 15. A
citation in the frozen body is the sole writer's to fix; it is flagged rather than edited.
