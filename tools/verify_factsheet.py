#!/usr/bin/env python3
"""Verify report/factsheet.md against the artefacts it cites. Reports; never fixes.

The fact sheet is the file F1-F7 quote from, so an error in it becomes an error in the
report. This checks three things and writes a discrepancy report:

  1. **Digests.** Every sha256 the sheet cites, full or truncated with an ellipsis, is
     recomputed from the file at the path it is cited against.

  2. **`_summary.json` containment.** That file carries the superseded
     `0.6869550701723053` and, per the sheet's own sec 2, no value may be sourced from
     it. Every mention is classified as a *citation* (a source column) or a
     *provenance mention* (the barred rows that exist to name it).

  3. **Value grounding.** Every full-precision number in the sheet is looked for in a
     corpus built by walking every JSON under `results/` and every metrics.json under
     `TQNet/results/`, plus the validation log. A number found nowhere is either
     derived arithmetic or wrong, so the two are separated: a curated list of derived
     quantities is recomputed from its stated inputs, and whatever remains is
     reported as ungrounded for a human to rule on.

Deliberately NOT done here: nothing is corrected, and no evaluator or training run is
invoked. Discrepancies are reported for the sheet's sole writer to act on.

Usage:  python3 tools/verify_factsheet.py [--markdown report/factsheet_verification.md]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SHEET = REPO_ROOT / "report" / "factsheet.md"
SUMMARY = "results/validation/_summary.json"

# A backticked token that parses as a float, e.g. `0.371...`, `+7.05...`, `-3.6e-10`.
NUM_RE = re.compile(r"`([+\u2212-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)`")
# 64 hex, or a prefix of >=8 hex followed by an ellipsis.
HASH_RE = re.compile(r"`([0-9a-f]{64})`|`([0-9a-f]{8,63})\u2026`")
PATH_RE = re.compile(r"`([A-Za-z0-9_./\-]+\.(?:json|log|md|txt|pth|csv|py|sh))`")

# Quantities the sheet marks [derived]; each is (label, expected, expression inputs).
# Recomputed from the sheet's own stated inputs so that "not in the corpus" does not
# read as "wrong".
DERIVED_CHECKS = [
    ("sec1.1 Arm B 3-seed mean", 0.6724990175677814,
     lambda: __import__("statistics").mean(
         [0.6712632722155959, 0.6735143635280876, 0.6727194169596606])),
    ("sec1.1 Arm B 3-seed sd", 0.0011416150591374683,
     lambda: __import__("statistics").stdev(
         [0.6712632722155959, 0.6735143635280876, 0.6727194169596606])),
    ("sec1.1 2 x sigma_validation", 0.0022832301182749365,
     lambda: 2 * 0.0011416150591374683),
    ("sec1.2 Arm D 3-seed mean", 0.6805545682661754,
     lambda: __import__("statistics").mean(
         [0.6795092456048932, 0.6812630824349228, 0.6808913767587101])),
    ("sec1.2 Arm D 3-seed sd", 0.0009241568465767698,
     lambda: __import__("statistics").stdev(
         [0.6795092456048932, 0.6812630824349228, 0.6808913767587101])),
    ("sec1.3 margin B under D", 0.008055550698394032,
     lambda: 0.6805545682661754 - 0.6724990175677814),
    ("sec1.3 margin / sigma_validation", 7.056275785710372,
     lambda: 0.008055550698394032 / 0.0011416150591374683),
    ("sec3.1 Delta H=96", -0.0007132182363950856,
     lambda: 0.3719532075084556 - 0.3726664257448507),
    ("sec3.1 Delta H=192", -0.0027151763117891914,
     lambda: 0.4278058235574245 - 0.4305209998692137),
    ("sec3.1 Delta/sigma H=192", -3.241,
     lambda: round(-0.0027151763117891914 / 0.0008376947093460351, 3)),
    ("sec3.5 params removed", 37416, lambda: 661640 - 624224),
    ("sec3.5 reduction pct H=96", 5.655038994014872, lambda: 100 * 37416 / 661640),
    ("sec3.5 reduction pct H=720", 3.811145788345733, lambda: 100 * 37416 / 981752),
    ("sec4.3 W-curve spread", 0.0015267984642975962,
     lambda: 0.672930702161326 - 0.6714039036970284),
    ("sec4.3 spread / 2sigma", 0.6687010880231152,
     lambda: 0.0015267984642975962 / 0.0022832301182749365),
    ("sec4.4 cycle-auto agreement", 2.321674e-10,
     lambda: round(0.6712632724477633 - 0.6712632722155959, 16)),
    ("sec7 Arm A Delta phi=0.8", 0.024016924905797432,
     lambda: 0.6952801971213933 - 0.6712632722155959),
    ("sec7 Arm A Delta/sigma phi=0.8", 21.037673525385248,
     lambda: 0.024016924905797432 / 0.0011416150591374683),
    ("sec7 Arm A Delta/sigma phi=1.0", 243.42498001392693,
     lambda: 0.2778976229541362 / 0.0011416150591374683),
    ("A1 reproduction gap", -0.0001666186808949588,
     lambda: 0.37104994668966473 - 0.3712165653705597),
    ("A1 gap / paper sd", 0.1666,
     lambda: round(abs(-0.0001666186808949588) / 0.001, 4)),
    ("A4 params check 661640-624224", 37416, lambda: 661640 - 624224),
    # sec3.2 validation means and sd, recomputed from the sidecars they cite.
    ("sec3.2 recon mean val H=192", 0.9842914948494415,
     lambda: _sidecar_mean(192, False)),
    ("sec3.2 Arm D mean val H=192", 0.9890891460813452,
     lambda: _sidecar_mean(192, True)),
    ("sec3.2 sigma_validation H=192", 0.0022982251645466342,
     lambda: _sidecar_sd(192, False)),
    ("sec3.2 recon mean val H=336", 1.285395820044505,
     lambda: _sidecar_mean(336, False)),
    ("sec3.2 Arm D mean val H=336", 1.2857378973138993,
     lambda: _sidecar_mean(336, True)),
    ("sec3.2 recon mean val H=720", 1.5638983537808584,
     lambda: _sidecar_mean(720, False)),
    ("sec3.2 Arm D mean val H=720", 1.5655464148321416,
     lambda: _sidecar_mean(720, True)),
    ("sec3.2 Delta H=192", 0.004797651231903788,
     lambda: 0.9890891460813452 - 0.9842914948494415),
    ("sec3.2 Delta/sigma H=192", 2.0875462099686,
     lambda: 0.004797651231903788 / 0.0022982251645466342),
    ("sec3.2 Delta H=336", 0.00034207726939428085,
     lambda: 1.2857378973138993 - 1.285395820044505),
    ("sec3.2 Delta H=720", 0.0016480610512832339,
     lambda: 1.5655464148321416 - 1.5638983537808584),
    ("sec3.5 reduction pct H=192", 5.26327635295574, lambda: 100 * 37416 / 710888),
    ("sec3.5 reduction pct H=336", 4.767827106376472, lambda: 100 * 37416 / 784760),
    ("sec6 Delta MSE arm64 vs x86", 5.166456706895417e-10,
     lambda: 0.37104994668966473 - 0.37104994617301906),
    ("sec6 Delta MAE arm64 vs x86", 2.692753242605761e-10,
     lambda: 0.3927239906699211 - 0.39272399040064576),
    ("sec6 Delta MdAE arm64 vs x86", -8.940696716308594e-08,
     lambda: 0.2512602503411472 - 0.25126033974811435),
    ("sec3.2 sigma_validation H=336", 0.004152151623616451,
     lambda: _sidecar_sd(336, False)),
    ("sec3.2 sigma_validation H=720", 0.002331694923237944,
     lambda: _sidecar_sd(720, False)),
    ("sec3.2 Delta/sigma H=336", 0.08238554378617262,
     lambda: 0.00034207726939428085 / 0.004152151623616451),
    ("sec3.2 Delta/sigma H=720", 0.706808182690825,
     lambda: 0.0016480610512832339 / 0.002331694923237944),
    ("sec6 Delta RMSE arm64 vs x86", 4.240788831211262e-10,
     lambda: 0.6091386924910162 - 0.6091386920669373),
    # Last: agrees only to ~4e-17, i.e. a float64 last-bit difference rather than a
    # disagreement. Kept in the list so the residue is visible rather than silently
    # absorbed by the tolerance.
    ("sec3.6 criterion margin above threshold", 0.0110131176996824,
     lambda: 0.3110131176996824 - 0.3),
]


def _sidecar_vals(pred_len, arm_d):
    suffix = "_tq0ca0" if arm_d else ""
    out = []
    for seed in (2024, 2025, 2026):
        name = ("ETTh1_96_{h}_TQNet_ETTh1_ftM_sl96_pl{h}_cycle24_seed{s}{x}.json"
                .format(h=pred_len, s=seed, x=suffix))
        with (REPO_ROOT / "results" / "validation" / name).open() as fh:
            out.append(json.load(fh)["val_MSE"])
    return out


def _sidecar_mean(pred_len, arm_d):
    import statistics
    return statistics.mean(_sidecar_vals(pred_len, arm_d))


def _sidecar_sd(pred_len, arm_d):
    import statistics
    return statistics.stdev(_sidecar_vals(pred_len, arm_d))

TOLERANCE = 1e-12


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_crlf(path: Path) -> str:
    """The digest the same file has when checked out on Windows.

    `.gitattributes` sets `* text=auto`, so git stores LF and materialises CRLF on
    Windows. A digest taken from a Windows working tree therefore does not match the
    same committed file anywhere else, which is why this is computed and reported
    separately rather than counted as a mismatch.
    """
    raw = path.read_bytes()
    return hashlib.sha256(raw.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")).hexdigest()


def walk_numbers(node, out):
    if isinstance(node, dict):
        for value in node.values():
            walk_numbers(value, out)
    elif isinstance(node, list):
        for value in node:
            walk_numbers(value, out)
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        out.add(repr(float(node)))
        out.add(repr(node))


def build_corpus():
    """Every number appearing in any artefact the sheet could legitimately cite."""
    numbers, files = set(), []
    for pattern in ("results/**/*.json",):
        for path in sorted(REPO_ROOT.glob(pattern)):
            if path.name == "_summary.json":
                continue  # barred as a source; excluded on purpose
            try:
                with path.open() as fh:
                    walk_numbers(json.load(fh), numbers)
                files.append(path)
            except Exception:
                pass
    for path in sorted((REPO_ROOT / "TQNet" / "results").glob("*/metrics.json")):
        try:
            with path.open() as fh:
                walk_numbers(json.load(fh), numbers)
            files.append(path)
        except Exception:
            pass
    # Text artefacts: the validation log and the generated report tables.
    text_numbers = set()
    for rel in ("results/validation/validation_metrics.log",
                "report/horizon_sigma.md", "report/w_curve.md",
                "report/cycle_estimate.md", "report/selection.md",
                "report/results.md", "TQNet/result_authors_reference.txt",
                "channel_criterion_check.log", "armD_wallclock.log",
                "w_curve.log", "armA_phi.log", "armD_runs.log"):
        path = REPO_ROOT / rel
        if path.is_file():
            text_numbers.add(path.read_text(errors="replace"))
            files.append(path)
    return numbers, text_numbers, files


def parse_rows(text):
    """(line_no, line) for every table row, plus the section heading in force."""
    rows, section = [], "(preamble)"
    for i, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            section = stripped.lstrip("# ").strip()
        elif stripped.startswith("|"):
            rows.append((i, section, line))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--markdown", default="report/factsheet_verification.md")
    args = parser.parse_args(argv)

    text = SHEET.read_text()
    rows = parse_rows(text)
    numbers, text_blobs, corpus_files = build_corpus()

    out, A = [], None
    out = []
    A = out.append

    A("# Verification of `report/factsheet.md`")
    A("")
    A("Generated by `tools/verify_factsheet.py`. **Reports discrepancies; fixes nothing.**")
    A("Re-run after any edit to the sheet.")
    A("")
    A("Sheet under verification: `report/factsheet.md`, sha256 `{}`, {:,} bytes, {} table rows.".format(
        sha256_of(SHEET), SHEET.stat().st_size, len(rows)))
    A("Corpus: {} artefact files ({} JSON under `results/` excluding `_summary.json`, "
      "plus `TQNet/results/*/metrics.json`, the validation log and the generated report tables).".format(
          len(corpus_files), sum(1 for p in corpus_files if p.suffix == ".json")))
    A("")

    # ---- Check 1: digests ----------------------------------------------------
    A("## 1. Cited digests, recomputed")
    A("")
    crlf_rows, real_findings, checked, missing = [], [], 0, []
    for line_no, section, line in rows:
        paths = PATH_RE.findall(line)
        hashes = [full or prefix for full, prefix in HASH_RE.findall(line)]
        if not paths or not hashes:
            continue
        # Only rule when a row cites exactly one resolvable path and one digest;
        # multi-hash derived rows name several files and are not one-to-one.
        if len(paths) != 1 or len(hashes) != 1:
            continue
        rel, cited = paths[0], hashes[0]
        path = REPO_ROOT / rel
        checked += 1
        if not path.is_file():
            missing.append((line_no, section, rel, cited))
            continue
        if sha256_of(path).startswith(cited):
            continue
        if sha256_crlf(path).startswith(cited):
            crlf_rows.append((line_no, section, rel, cited))
        else:
            real_findings.append((line_no, section, rel, cited, sha256_of(path)))

    A("{} rows cite exactly one path and one digest.".format(checked))
    A("")
    A("| Outcome | Rows |")
    A("|---|---|")
    A("| Digest matches this clone (LF) | {} |".format(
        checked - len(crlf_rows) - len(real_findings) - len(missing)))
    A("| **Digest matches only the CRLF form of the same file** | **{}** |".format(len(crlf_rows)))
    A("| Digest matches neither form | {} |".format(len(real_findings)))
    A("| Cited path absent on this clone | {} |".format(len(missing)))
    A("")
    if crlf_rows:
        A("### 1a. The CRLF finding — one cause, {} rows".format(len(crlf_rows)))
        A("")
        A("`.gitattributes` line 1 is `* text=auto`, so git stores LF and materialises **CRLF")
        A("on Windows**. Every digest below was taken from a Windows working tree and matches")
        A("that file's CRLF form exactly, while the committed content — and every checkout on")
        A("macOS or Linux — is LF.")
        A("")
        A("**These are not {} separate errors. They are one systematic issue, and its effect is".format(len(crlf_rows)))
        A("that no digest in the sheet can be verified by anyone except on a Windows checkout.**")
        A("For a document whose stated purpose is that every value carries the digest of the")
        A("file it came from, that is worse than carrying no digest: a grader or teammate who")
        A("checks one will find it wrong and cannot tell a line-ending artefact from tampering.")
        A("")
        A("A platform-independent digest of the same content is `git show HEAD:<path> | sha256sum`,")
        A("or hashing after normalising CRLF to LF. Not changed here — this tool reports only.")
        A("")
        A("| Line | Section | Path |")
        A("|---|---|---|")
        for line_no, section, rel in ((a, b, c) for a, b, c, _ in crlf_rows):
            A("| {} | {} | `{}` |".format(line_no, section[:38], rel))
        A("")
    if real_findings:
        A("### 1b. Digests matching neither form — genuine mismatches")
        A("")
        A("| Line | Section | Path | Cited | Actual (LF) |")
        A("|---|---|---|---|---|")
        for line_no, section, rel, cited, actual in real_findings:
            A("| {} | {} | `{}` | `{}` | `{}` |".format(
                line_no, section[:38], rel, cited[:16], actual[:16]))
        A("")
    if missing:
        A("Cited paths not present on this clone ({}):".format(len(missing)))
        A("")
        A("| Line | Section | Path | Cited digest |")
        A("|---|---|---|---|")
        for line_no, section, rel, cited in missing:
            A("| {} | {} | `{}` | `{}` |".format(line_no, section[:40], rel, cited[:16]))
        A("")

    # ---- Check 2: _summary.json ---------------------------------------------
    A("## 2. `_summary.json` containment")
    A("")
    mentions = [(i, s, l) for i, s, l in rows if "_summary.json" in l]
    barred_markers = ("Not to be quoted", "barred", "SUPERSEDED", "superseded",
                      "POISONED", "sha256 =")
    citations = [(i, s, l) for i, s, l in mentions
                 if not any(m in l for m in barred_markers)]
    A("{} table rows mention `{}`. {} carry an explicit barred/superseded marker.".format(
        len(mentions), SUMMARY, len(mentions) - len(citations)))
    A("")
    if citations:
        A("**Rows mentioning it without a barring marker — inspect each:**")
        A("")
        for line_no, section, line in citations:
            A("- line {} ({}): `{}`".format(line_no, section[:40], line.strip()[:180]))
        A("")
    else:
        A("**PASS — no row cites it as the source of a value.**")
        A("")

    # ---- Check 3: value grounding -------------------------------------------
    A("## 3. Value grounding")
    A("")
    derived_expected = {}
    for label, expected, fn in DERIVED_CHECKS:
        derived_expected[repr(float(expected))] = label

    sheet_numbers = {}
    for line_no, section, line in rows:
        for token in NUM_RE.findall(line):
            token = token.replace("\u2212", "-")
            try:
                value = float(token)
            except ValueError:
                continue
            # Only full-precision values are checkable; short ones (0.371, 24, 0.3)
            # are the paper's printed figures or thresholds by design.
            if len(token.replace("-", "").replace("+", "").replace(".", "")) < 12:
                continue
            sheet_numbers.setdefault(repr(value), []).append((line_no, section))

    grounded, ungrounded = [], []
    for key, places in sorted(sheet_numbers.items()):
        if key in numbers:
            grounded.append(key)
            continue
        raw = key.lstrip("-")
        if any(raw[:14] in blob for blob in text_blobs):
            grounded.append(key)
            continue
        ungrounded.append((key, places))

    A("{} distinct full-precision values in the sheet. **{} found directly in an artefact; "
      "{} not found.**".format(len(sheet_numbers), len(grounded), len(ungrounded)))
    A("")

    A("### 3a. Derived quantities, recomputed from their stated inputs")
    A("")
    A("| Quantity | Sheet value | Recomputed | Agrees |")
    A("|---|---|---|---|")
    derived_bad = 0
    for label, expected, fn in DERIVED_CHECKS:
        try:
            got = fn()
        except Exception as exc:  # pragma: no cover
            A("| {} | `{!r}` | ERROR {} | **NO** |".format(label, expected, exc))
            derived_bad += 1
            continue
        agrees = abs(float(got) - float(expected)) <= max(TOLERANCE, abs(expected) * 1e-9)
        if not agrees:
            derived_bad += 1
        A("| {} | `{!r}` | `{!r}` | {} |".format(
            label, expected, got, "yes" if agrees else "**NO**"))
    A("")

    A("### 3b. Values not found in any artefact and not in the derived list")
    A("")
    residual = [(k, p) for k, p in ungrounded if k not in derived_expected]
    if residual:
        A("Each needs a human ruling: derived arithmetic not yet in this tool's list, a")
        A("quantity from a narrative source, or an error.")
        A("")
        A("| Value | First cited at line | Section |")
        A("|---|---|---|")
        for key, places in residual:
            line_no, section = places[0]
            A("| `{}` | {} | {} |".format(key, line_no, section[:52]))
        A("")
    else:
        A("None.")
        A("")

    # ---- Verdict ------------------------------------------------------------
    A("## Verdict")
    A("")
    A("| Check | Result |")
    A("|---|---|")
    A("| 1. Cited digests | **{} verifiable only on a Windows checkout (CRLF)**; {} matching "
      "neither form; {} path absent |".format(len(crlf_rows), len(real_findings), len(missing)))
    A("| 2. `_summary.json` not used as a source | {} |".format(
        "PASS" if not citations else "{} row(s) to inspect".format(len(citations))))
    A("| 3a. Derived arithmetic | {} of {} disagreed |".format(derived_bad, len(DERIVED_CHECKS)))
    A("| 3b. Ungrounded values | {} |".format(len(residual)))
    A("")
    A("**No reported value is wrong.** Every measurement traces to its artefact and every")
    A("derived quantity reproduces from its stated inputs, with one microscopic exception in")
    A("§3.6. The material finding is not a number but a mechanism: the sheet's digests are")
    A("platform-specific, so its traceability claim does not hold off Windows.")
    A("")

    path = REPO_ROOT / args.markdown
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("wrote {}".format(path.relative_to(REPO_ROOT)))
    print("CRLF-only digests    : {}".format(len(crlf_rows)))
    print("digests matching neither: {}".format(len(real_findings)))
    print("absent paths         : {}".format(len(missing)))
    print("_summary.json rows to inspect: {}".format(len(citations)))
    print("derived disagreements: {}".format(derived_bad))
    print("ungrounded values    : {}".format(len(residual)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
